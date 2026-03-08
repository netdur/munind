use anyhow::{Context, Result};
use clap::Parser;
use munind_api::MunindEngine;
use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::{FilterExpression, MemoryId, SearchRequest};
use munind_core::engine::VectorEngine;
use munind_rag::{DeterministicEmbedder, EmbeddingProvider, OpenAICompatibleEmbedder};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

const PRIMARY_SOURCE: &str = "TinyStories";
const SECONDARY_SOURCE: &str = "TinyStoriesAlt";

#[derive(Parser, Debug)]
#[command(author, version, about = "Munind benchmark runner")]
struct Args {
    /// JSONL file containing {"text": "..."} rows
    #[arg(long, default_value = "benchmark/data/tinystories_subset.jsonl")]
    input: PathBuf,

    /// Benchmark database path (will be deleted/recreated)
    #[arg(long, default_value = "benchmark/tmp_db")]
    db_path: PathBuf,

    /// Embedding/vector dimension
    #[arg(long, default_value_t = 512)]
    dimension: usize,

    /// Number of documents to load from input
    #[arg(long, default_value_t = 1000)]
    limit: usize,

    /// Number of queries to run
    #[arg(long, default_value_t = 200)]
    queries: usize,

    /// Top-K for search requests and quality metrics
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// ef_search override for query-time search breadth
    #[arg(long, default_value_t = 80)]
    ef_search: usize,

    /// Optional OpenAI-compatible embedding endpoint (for real embeddings)
    #[arg(long)]
    embedding_endpoint: Option<String>,

    /// Embedding model id sent to embedding endpoint
    #[arg(long, default_value = "deterministic-v1")]
    embedding_model: String,

    /// Optional API key for embedding endpoint authorization
    #[arg(long)]
    embedding_api_key: Option<String>,

    /// Batch size used for embedding calls
    #[arg(long, default_value_t = 64)]
    embedding_batch_size: usize,

    /// Optional JSON file for benchmark summary
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct InputRow {
    text: String,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    inserted_docs: usize,
    queried_docs: usize,
    dimension: usize,
    embedding_model: String,
    quality_top_k: usize,
    insert: Stats,
    search: Stats,
    filtered_search: Stats,
    search_quality: QualityStats,
    filtered_search_quality: QualityStats,
}

#[derive(Debug, Serialize)]
struct Stats {
    total_ops: usize,
    total_seconds: f64,
    ops_per_second: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
}

#[derive(Debug, Serialize)]
struct QualityStats {
    queries_evaluated: usize,
    mean_recall_at_k: f64,
    p50_recall_at_k: f64,
    p95_recall_at_k: f64,
    mean_mrr_at_k: f64,
    mean_ndcg_at_k: f64,
}

#[derive(Debug, Clone)]
struct BenchDoc {
    id: MemoryId,
    source: &'static str,
    vector: Vec<f32>,
}

#[derive(Default)]
struct QualityAccumulator {
    recalls: Vec<f64>,
    mrrs: Vec<f64>,
    ndcgs: Vec<f64>,
}

impl QualityAccumulator {
    fn push(&mut self, predicted: &[MemoryId], exact: &[MemoryId]) {
        self.recalls.push(recall_at_k(predicted, exact));
        self.mrrs.push(mrr_at_k(predicted, exact));
        self.ndcgs.push(ndcg_at_k(predicted, exact));
    }

    fn summarize(&self) -> QualityStats {
        QualityStats {
            queries_evaluated: self.recalls.len(),
            mean_recall_at_k: mean(&self.recalls),
            p50_recall_at_k: percentile(&self.recalls, 50.0),
            p95_recall_at_k: percentile(&self.recalls, 95.0),
            mean_mrr_at_k: mean(&self.mrrs),
            mean_ndcg_at_k: mean(&self.ndcgs),
        }
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn compute_stats(latencies_ms: &[f64], total_seconds: f64) -> Stats {
    let total_ops = latencies_ms.len();
    let mean_ms = mean(latencies_ms);

    let (min_ms, max_ms) = if latencies_ms.is_empty() {
        (0.0, 0.0)
    } else {
        (
            latencies_ms.iter().copied().fold(f64::INFINITY, f64::min),
            latencies_ms
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        )
    };

    Stats {
        total_ops,
        total_seconds,
        ops_per_second: if total_seconds > 0.0 {
            total_ops as f64 / total_seconds
        } else {
            0.0
        },
        p50_ms: percentile(latencies_ms, 50.0),
        p95_ms: percentile(latencies_ms, 95.0),
        p99_ms: percentile(latencies_ms, 99.0),
        min_ms,
        max_ms,
        mean_ms,
    }
}

fn recall_at_k(predicted: &[MemoryId], exact: &[MemoryId]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_set: std::collections::HashSet<MemoryId> = exact.iter().copied().collect();
    let hit_count = predicted.iter().filter(|id| exact_set.contains(id)).count();
    hit_count as f64 / exact.len() as f64
}

fn mrr_at_k(predicted: &[MemoryId], exact: &[MemoryId]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_set: std::collections::HashSet<MemoryId> = exact.iter().copied().collect();
    for (rank, id) in predicted.iter().enumerate() {
        if exact_set.contains(id) {
            return 1.0 / (rank as f64 + 1.0);
        }
    }
    0.0
}

fn ndcg_at_k(predicted: &[MemoryId], exact: &[MemoryId]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let mut ideal_gains = Vec::with_capacity(exact.len());
    let mut relevance = HashMap::with_capacity(exact.len());
    for (i, id) in exact.iter().enumerate() {
        let gain = 1.0 / (i as f64 + 1.0);
        ideal_gains.push(gain);
        relevance.insert(*id, gain);
    }

    let dcg = predicted
        .iter()
        .enumerate()
        .map(|(rank, id)| {
            let gain = relevance.get(id).copied().unwrap_or(0.0);
            gain / ((rank as f64 + 2.0).log2())
        })
        .sum::<f64>();

    let idcg = ideal_gains
        .iter()
        .enumerate()
        .map(|(rank, gain)| gain / ((rank as f64 + 2.0).log2()))
        .sum::<f64>();

    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
}

fn exact_top_k_ids(
    query: &[f32],
    docs: &[BenchDoc],
    top_k: usize,
    source_filter: Option<&str>,
) -> Vec<MemoryId> {
    let mut scored = Vec::with_capacity(docs.len());

    for doc in docs {
        if let Some(source) = source_filter
            && doc.source != source
        {
            continue;
        }

        let distance = cosine_distance(query, &doc.vector);
        scored.push((doc.id, distance));
    }

    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.0.cmp(&b.0.0))
    });
    scored.truncate(top_k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn load_texts(path: &PathBuf, limit: usize) -> Result<Vec<String>> {
    let f = File::open(path)
        .with_context(|| format!("failed to open input file {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut texts = Vec::with_capacity(limit);

    for (line_no, line) in reader.lines().enumerate() {
        if texts.len() >= limit {
            break;
        }
        let line = line.with_context(|| format!("failed reading line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: InputRow = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSONL row at line {}", line_no + 1))?;
        if !row.text.trim().is_empty() {
            texts.push(row.text);
        }
    }

    Ok(texts)
}

fn print_stats(name: &str, stats: &Stats) {
    println!();
    println!("{name}");
    println!("  ops: {}", stats.total_ops);
    println!("  total_s: {:.3}", stats.total_seconds);
    println!("  ops/s: {:.2}", stats.ops_per_second);
    println!(
        "  latency_ms: p50={:.3} p95={:.3} p99={:.3} min={:.3} max={:.3} mean={:.3}",
        stats.p50_ms, stats.p95_ms, stats.p99_ms, stats.min_ms, stats.max_ms, stats.mean_ms
    );
}

fn print_quality(name: &str, stats: &QualityStats) {
    println!();
    println!("{name}");
    println!("  queries: {}", stats.queries_evaluated);
    println!(
        "  recall@k: mean={:.4} p50={:.4} p95={:.4}",
        stats.mean_recall_at_k, stats.p50_recall_at_k, stats.p95_recall_at_k
    );
    println!(
        "  ranking: mrr@k={:.4} ndcg@k={:.4}",
        stats.mean_mrr_at_k, stats.mean_ndcg_at_k
    );
}

fn build_embedder(args: &Args) -> Result<Box<dyn EmbeddingProvider>> {
    if let Some(endpoint) = &args.embedding_endpoint {
        if endpoint.trim().is_empty() {
            anyhow::bail!("--embedding-endpoint cannot be empty");
        }

        let mut embedder =
            OpenAICompatibleEmbedder::new(endpoint.clone(), args.embedding_model.clone());
        if let Some(api_key) = &args.embedding_api_key {
            embedder = embedder.with_api_key(api_key.clone());
        }

        return Ok(Box::new(embedder));
    }

    Ok(Box::new(DeterministicEmbedder::new(
        args.embedding_model.clone(),
    )))
}

fn embed_texts(
    embedder: &dyn EmbeddingProvider,
    texts: &[String],
    dimension: usize,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(texts.len());
    let chunk_size = batch_size.max(1);

    for chunk in texts.chunks(chunk_size) {
        let mut batch = embedder
            .embed_batch(chunk, dimension)
            .map_err(|e| anyhow::anyhow!("embedding batch failed: {e:?}"))?;
        vectors.append(&mut batch);
    }

    if vectors.len() != texts.len() {
        anyhow::bail!(
            "embedding count mismatch: expected {}, got {}",
            texts.len(),
            vectors.len()
        );
    }

    Ok(vectors)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let texts = load_texts(&args.input, args.limit)?;
    if texts.is_empty() {
        anyhow::bail!("no usable text rows loaded from {}", args.input.display());
    }
    let query_count = args.queries.min(texts.len());

    if args.db_path.exists() {
        fs::remove_dir_all(&args.db_path)
            .with_context(|| format!("failed to remove old db path {}", args.db_path.display()))?;
    }

    let embedder = build_embedder(&args)?;
    let doc_vectors = embed_texts(
        embedder.as_ref(),
        &texts,
        args.dimension,
        args.embedding_batch_size,
    )?;
    let query_texts: Vec<String> = texts.iter().take(query_count).cloned().collect();
    let query_vectors = embed_texts(
        embedder.as_ref(),
        &query_texts,
        args.dimension,
        args.embedding_batch_size,
    )?;

    let mut cfg = EngineConfig::default();
    cfg.index.metric = DistanceMetric::Cosine;
    cfg.index.ef_search = args.ef_search;
    cfg.query.ef_search = args.ef_search;
    let engine = MunindEngine::create(&args.db_path, args.dimension, cfg)?;

    let mut bench_docs = Vec::with_capacity(texts.len());
    let mut insert_latencies_ms = Vec::with_capacity(texts.len());
    let insert_t0 = Instant::now();
    for (i, (text, vec)) in texts.iter().zip(doc_vectors.into_iter()).enumerate() {
        let source = if i % 2 == 0 {
            PRIMARY_SOURCE
        } else {
            SECONDARY_SOURCE
        };

        let doc = json!({
            "text": text,
            "source": source,
            "row_idx": i
        });

        let t0 = Instant::now();
        let id = engine.insert_json(vec.clone(), doc)?;
        insert_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        bench_docs.push(BenchDoc {
            id,
            source,
            vector: vec,
        });
    }
    let insert_total_s = insert_t0.elapsed().as_secs_f64();
    let insert_stats = compute_stats(&insert_latencies_ms, insert_total_s);

    let mut search_quality_acc = QualityAccumulator::default();
    let mut search_latencies_ms = Vec::with_capacity(query_count);
    let search_t0 = Instant::now();
    for query_vec in &query_vectors {
        let req = SearchRequest {
            vector: query_vec.clone(),
            top_k: args.top_k,
            text_query: None,
            hybrid_alpha: None,
            lexical_top_k: None,
            filter: None,
            ef_search: Some(args.ef_search),
            radius: None,
        };

        let t0 = Instant::now();
        let hits = engine.search(req)?;
        search_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let ann_ids: Vec<MemoryId> = hits.into_iter().map(|h| h.id).collect();
        let exact_ids = exact_top_k_ids(query_vec, &bench_docs, args.top_k, None);
        search_quality_acc.push(&ann_ids, &exact_ids);
    }
    let search_total_s = search_t0.elapsed().as_secs_f64();
    let search_stats = compute_stats(&search_latencies_ms, search_total_s);
    let search_quality = search_quality_acc.summarize();

    let mut filtered_quality_acc = QualityAccumulator::default();
    let mut filtered_latencies_ms = Vec::with_capacity(query_count);
    let filtered_t0 = Instant::now();
    for query_vec in &query_vectors {
        let req = SearchRequest {
            vector: query_vec.clone(),
            top_k: args.top_k,
            text_query: None,
            hybrid_alpha: None,
            lexical_top_k: None,
            filter: Some(FilterExpression::Eq(
                "source".to_string(),
                json!(PRIMARY_SOURCE),
            )),
            ef_search: Some(args.ef_search),
            radius: None,
        };

        let t0 = Instant::now();
        let hits = engine.search(req)?;
        filtered_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let ann_ids: Vec<MemoryId> = hits.into_iter().map(|h| h.id).collect();
        let exact_ids = exact_top_k_ids(query_vec, &bench_docs, args.top_k, Some(PRIMARY_SOURCE));
        filtered_quality_acc.push(&ann_ids, &exact_ids);
    }
    let filtered_total_s = filtered_t0.elapsed().as_secs_f64();
    let filtered_stats = compute_stats(&filtered_latencies_ms, filtered_total_s);
    let filtered_quality = filtered_quality_acc.summarize();

    println!("Munind Benchmark Summary");
    println!("  input: {}", args.input.display());
    println!("  docs_loaded: {}", texts.len());
    println!("  queries: {}", query_count);
    println!("  dimension: {}", args.dimension);
    println!("  embedding_model: {}", embedder.model_id());
    println!("  top_k: {}", args.top_k);
    println!("  ef_search: {}", args.ef_search);

    print_stats("Insert", &insert_stats);
    print_stats("Search", &search_stats);
    print_stats(
        &format!("Filtered Search (source == {PRIMARY_SOURCE})"),
        &filtered_stats,
    );

    print_quality("Search Quality vs Exact", &search_quality);
    print_quality(
        &format!("Filtered Search Quality vs Exact (source == {PRIMARY_SOURCE})"),
        &filtered_quality,
    );

    if let Some(path) = args.output_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let summary = BenchmarkSummary {
            inserted_docs: texts.len(),
            queried_docs: query_count,
            dimension: args.dimension,
            embedding_model: embedder.model_id().to_string(),
            quality_top_k: args.top_k,
            insert: insert_stats,
            search: search_stats,
            filtered_search: filtered_stats,
            search_quality,
            filtered_search_quality: filtered_quality,
        };
        fs::write(&path, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("failed to write summary to {}", path.display()))?;
        println!();
        println!("Wrote JSON summary to {}", path.display());
    }

    Ok(())
}
