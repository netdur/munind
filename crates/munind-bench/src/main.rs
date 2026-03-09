use anyhow::{Context, Result};
use clap::Parser;
use munind_api::MunindEngine;
use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::{FilterExpression, SearchHit, SearchRequest};
use munind_core::engine::VectorEngine;
use munind_rag::{DeterministicEmbedder, EmbeddingProvider, OpenAICompatibleEmbedder};
use munind_storage::StorageEngine;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

const PRIMARY_SOURCE: &str = "TinyStories";
const SECONDARY_SOURCE: &str = "TinyStoriesAlt";
type EvalId = usize;

#[derive(Parser, Debug)]
#[command(author, version, about = "Munind benchmark runner")]
struct Args {
    /// JSONL file containing {"text": "..."} rows
    #[arg(long, default_value = "benchmark/data/tinystories_subset.jsonl")]
    input: PathBuf,

    /// Benchmark database path
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

    /// Max retries for each embedding request batch
    #[arg(long, default_value_t = 8)]
    embedding_max_retries: usize,

    /// Base backoff in milliseconds for embedding retries
    #[arg(long, default_value_t = 250)]
    embedding_retry_base_ms: u64,

    /// Max backoff in milliseconds for embedding retries
    #[arg(long, default_value_t = 3000)]
    embedding_retry_max_ms: u64,

    /// Delay between embedding request batches in milliseconds
    #[arg(long, default_value_t = 40)]
    embedding_request_delay_ms: u64,

    /// Build DB and insert documents only; skip search/quality phase
    #[arg(long, default_value_t = false, conflicts_with = "use_existing_db")]
    prepare_only: bool,

    /// Open existing DB path and run benchmark without recreating/reinserting
    #[arg(long, default_value_t = false)]
    use_existing_db: bool,

    /// Search latency benchmark only (skip exact-quality computation)
    #[arg(long, default_value_t = false)]
    latency_only: bool,

    /// Require real embeddings from --embedding-endpoint (disable deterministic fallback)
    #[arg(long, default_value_t = false)]
    require_real_embeddings: bool,

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
    mode: String,
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
    row_idx: EvalId,
    source: String,
    vector: Vec<f32>,
}

#[derive(Default)]
struct QualityAccumulator {
    recalls: Vec<f64>,
    mrrs: Vec<f64>,
    ndcgs: Vec<f64>,
}

impl QualityAccumulator {
    fn push(&mut self, predicted: &[EvalId], exact: &[EvalId]) {
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

struct SearchBenchmarkResult {
    search_stats: Stats,
    filtered_stats: Stats,
    search_quality: QualityStats,
    filtered_quality: QualityStats,
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

fn compute_stats(latencies_ms: &[f64]) -> Stats {
    let total_ops = latencies_ms.len();
    let mean_ms = mean(latencies_ms);
    let total_seconds = latencies_ms.iter().sum::<f64>() / 1000.0;

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

fn zero_stats() -> Stats {
    compute_stats(&[])
}

fn recall_at_k(predicted: &[EvalId], exact: &[EvalId]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_set: HashSet<EvalId> = exact.iter().copied().collect();
    let hit_count = predicted.iter().filter(|id| exact_set.contains(id)).count();
    hit_count as f64 / exact.len() as f64
}

fn mrr_at_k(predicted: &[EvalId], exact: &[EvalId]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }

    let exact_set: HashSet<EvalId> = exact.iter().copied().collect();
    for (rank, id) in predicted.iter().enumerate() {
        if exact_set.contains(id) {
            return 1.0 / (rank as f64 + 1.0);
        }
    }
    0.0
}

fn ndcg_at_k(predicted: &[EvalId], exact: &[EvalId]) -> f64 {
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
) -> Vec<EvalId> {
    let mut scored = Vec::with_capacity(docs.len());

    for doc in docs {
        if let Some(source) = source_filter
            && doc.source != source
        {
            continue;
        }

        let distance = cosine_distance(query, &doc.vector);
        scored.push((doc.row_idx, distance));
    }

    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
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
    if args.require_real_embeddings && args.embedding_endpoint.is_none() {
        anyhow::bail!(
            "--require-real-embeddings is set, but --embedding-endpoint was not provided"
        );
    }

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
    max_retries: usize,
    retry_base_ms: u64,
    retry_max_ms: u64,
    request_delay_ms: u64,
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(texts.len());
    let chunk_size = batch_size.max(1);
    let retries = max_retries.max(1);

    for (chunk_idx, chunk) in texts.chunks(chunk_size).enumerate() {
        let mut maybe_batch = None;

        for attempt in 1..=retries {
            match embedder.embed_batch(chunk, dimension) {
                Ok(batch) => {
                    maybe_batch = Some(batch);
                    break;
                }
                Err(err) => {
                    if attempt == retries {
                        return Err(anyhow::anyhow!(
                            "embedding batch failed after {} attempts (chunk {}): {:?}",
                            retries,
                            chunk_idx,
                            err
                        ));
                    }

                    let exp = 1_u64 << (attempt.saturating_sub(1).min(10));
                    let backoff_ms = retry_base_ms.saturating_mul(exp).min(retry_max_ms.max(1));
                    eprintln!(
                        "embedding request retry {}/{} (chunk {}): {:?} (sleep {} ms)",
                        attempt, retries, chunk_idx, err, backoff_ms
                    );
                    thread::sleep(Duration::from_millis(backoff_ms));
                }
            }
        }

        let mut batch = maybe_batch.ok_or_else(|| {
            anyhow::anyhow!(
                "embedding retry loop ended without result (chunk {})",
                chunk_idx
            )
        })?;

        vectors.append(&mut batch);

        if request_delay_ms > 0 {
            thread::sleep(Duration::from_millis(request_delay_ms));
        }

        if (chunk_idx + 1) % 100 == 0 || vectors.len() == texts.len() {
            eprintln!("embedded {}/{} texts", vectors.len(), texts.len());
        }
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

fn build_bench_docs(vectors: &[Vec<f32>]) -> Vec<BenchDoc> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, vec)| BenchDoc {
            row_idx: i,
            source: if i % 2 == 0 {
                PRIMARY_SOURCE.to_string()
            } else {
                SECONDARY_SOURCE.to_string()
            },
            vector: vec.clone(),
        })
        .collect()
}

fn row_idx_from_document(doc: &Value) -> Option<EvalId> {
    let idx_u64 = doc.get("row_idx")?.as_u64()?;
    EvalId::try_from(idx_u64).ok()
}

fn source_from_document(doc: &Value) -> Option<String> {
    doc.get("source")?.as_str().map(ToOwned::to_owned)
}

fn extract_row_ids(hits: &[SearchHit]) -> Result<Vec<EvalId>> {
    let mut row_ids = Vec::with_capacity(hits.len());
    for hit in hits {
        let row_idx = row_idx_from_document(&hit.document).ok_or_else(|| {
            anyhow::anyhow!(
                "search hit {} is missing numeric document.row_idx (required for benchmark quality evaluation)",
                hit.id.0
            )
        })?;
        row_ids.push(row_idx);
    }
    Ok(row_ids)
}

fn load_bench_docs_from_db(
    db_path: &PathBuf,
    max_row_idx: usize,
    dimension: usize,
) -> Result<Vec<BenchDoc>> {
    let storage = StorageEngine::open(db_path)
        .with_context(|| format!("failed to open storage at {}", db_path.display()))?;

    let ids = storage.get_all_ids()?;
    let mut docs = Vec::new();
    let mut seen = HashSet::new();

    for (i, id) in ids.iter().copied().enumerate() {
        let Some(doc) = storage.get_document(id)? else {
            continue;
        };

        let Some(row_idx) = row_idx_from_document(&doc) else {
            continue;
        };
        if row_idx >= max_row_idx {
            continue;
        }

        let source = source_from_document(&doc).ok_or_else(|| {
            anyhow::anyhow!(
                "document for id {} is missing string field 'source' required for filtered quality",
                id.0
            )
        })?;

        let Some(vector) = storage.get_vector(id)? else {
            continue;
        };
        if vector.len() != dimension {
            anyhow::bail!(
                "vector dimension mismatch in DB at id {}: expected {}, got {}",
                id.0,
                dimension,
                vector.len()
            );
        }

        if seen.insert(row_idx) {
            docs.push(BenchDoc {
                row_idx,
                source,
                vector,
            });
        }

        if (i + 1) % 50_000 == 0 {
            eprintln!("loaded {} benchmark vectors from db...", docs.len());
        }
    }

    if docs.is_empty() {
        anyhow::bail!(
            "no benchmark docs were loaded from DB (expected documents with row_idx < {})",
            max_row_idx
        );
    }

    docs.sort_by_key(|d| d.row_idx);
    eprintln!("loaded {} benchmark vectors from existing db", docs.len());
    Ok(docs)
}

fn run_search_benchmark(
    engine: &MunindEngine,
    query_vectors: &[Vec<f32>],
    bench_docs: Option<&[BenchDoc]>,
    top_k: usize,
    ef_search: usize,
) -> Result<SearchBenchmarkResult> {
    let mut search_quality_acc = QualityAccumulator::default();
    let mut search_latencies_ms = Vec::with_capacity(query_vectors.len());
    for query_vec in query_vectors {
        let req = SearchRequest {
            vector: query_vec.clone(),
            top_k,
            text_query: None,
            hybrid_alpha: None,
            lexical_top_k: None,
            filter: None,
            ef_search: Some(ef_search),
            radius: None,
        };

        let t0 = Instant::now();
        let hits = engine.search(req)?;
        search_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        if let Some(docs) = bench_docs {
            let ann_ids = extract_row_ids(&hits)?;
            let exact_ids = exact_top_k_ids(query_vec, docs, top_k, None);
            search_quality_acc.push(&ann_ids, &exact_ids);
        }
    }
    let search_stats = compute_stats(&search_latencies_ms);
    let search_quality = search_quality_acc.summarize();

    let mut filtered_quality_acc = QualityAccumulator::default();
    let mut filtered_latencies_ms = Vec::with_capacity(query_vectors.len());
    for query_vec in query_vectors {
        let req = SearchRequest {
            vector: query_vec.clone(),
            top_k,
            text_query: None,
            hybrid_alpha: None,
            lexical_top_k: None,
            filter: Some(FilterExpression::Eq(
                "source".to_string(),
                json!(PRIMARY_SOURCE),
            )),
            ef_search: Some(ef_search),
            radius: None,
        };

        let t0 = Instant::now();
        let hits = engine.search(req)?;
        filtered_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        if let Some(docs) = bench_docs {
            let ann_ids = extract_row_ids(&hits)?;
            let exact_ids = exact_top_k_ids(query_vec, docs, top_k, Some(PRIMARY_SOURCE));
            filtered_quality_acc.push(&ann_ids, &exact_ids);
        }
    }
    let filtered_stats = compute_stats(&filtered_latencies_ms);
    let filtered_quality = filtered_quality_acc.summarize();

    Ok(SearchBenchmarkResult {
        search_stats,
        filtered_stats,
        search_quality,
        filtered_quality,
    })
}

fn mode_label(args: &Args) -> &'static str {
    if args.prepare_only {
        "prepare_only"
    } else if args.use_existing_db && args.latency_only {
        "search_existing_db_latency_only"
    } else if args.use_existing_db {
        "search_existing_db"
    } else if args.latency_only {
        "full_latency_only"
    } else {
        "full"
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let texts = load_texts(&args.input, args.limit)?;
    if texts.is_empty() {
        anyhow::bail!("no usable text rows loaded from {}", args.input.display());
    }

    let evaluate_quality = !args.prepare_only && !args.latency_only;
    let embedder = build_embedder(&args)?;

    let mut doc_vectors: Vec<Vec<f32>> = Vec::new();
    let mut bench_docs: Option<Vec<BenchDoc>> = None;

    if args.use_existing_db {
        if evaluate_quality {
            bench_docs = Some(load_bench_docs_from_db(
                &args.db_path,
                args.limit,
                args.dimension,
            )?);
        }
    } else {
        doc_vectors = embed_texts(
            embedder.as_ref(),
            &texts,
            args.dimension,
            args.embedding_batch_size,
            args.embedding_max_retries,
            args.embedding_retry_base_ms,
            args.embedding_retry_max_ms,
            args.embedding_request_delay_ms,
        )?;

        if evaluate_quality {
            bench_docs = Some(build_bench_docs(&doc_vectors));
        }
    }

    let mut inserted_docs = 0usize;
    let mut insert_stats = zero_stats();

    let engine = if args.use_existing_db {
        if !args.db_path.exists() {
            anyhow::bail!(
                "--use-existing-db requires an existing --db-path: {}",
                args.db_path.display()
            );
        }

        let engine = MunindEngine::open(&args.db_path)?;
        let actual_dim = engine.embedding_dimension();
        if actual_dim != args.dimension {
            anyhow::bail!(
                "dimension mismatch for existing DB: db={}, requested={} (hint: set --dimension to database dimension)",
                actual_dim,
                args.dimension
            );
        }

        engine
    } else {
        if args.db_path.exists() {
            fs::remove_dir_all(&args.db_path).with_context(|| {
                format!("failed to remove old db path {}", args.db_path.display())
            })?;
        }

        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::Cosine;
        cfg.index.ef_search = args.ef_search;
        cfg.query.ef_search = args.ef_search;

        let engine = MunindEngine::create(&args.db_path, args.dimension, cfg)?;

        let mut insert_latencies_ms = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
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
            engine.insert_json(doc_vectors[i].clone(), doc)?;
            insert_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        inserted_docs = texts.len();
        insert_stats = compute_stats(&insert_latencies_ms);
        engine
    };

    let mut queried_docs = 0usize;
    let mut search_stats = zero_stats();
    let mut filtered_stats = zero_stats();
    let mut search_quality = QualityAccumulator::default().summarize();
    let mut filtered_quality = QualityAccumulator::default().summarize();

    if !args.prepare_only {
        let query_count = args.queries.min(texts.len());
        queried_docs = query_count;

        let query_texts: Vec<String> = texts.iter().take(query_count).cloned().collect();
        let query_vectors = embed_texts(
            embedder.as_ref(),
            &query_texts,
            args.dimension,
            args.embedding_batch_size,
            args.embedding_max_retries,
            args.embedding_retry_base_ms,
            args.embedding_retry_max_ms,
            args.embedding_request_delay_ms,
        )?;

        let search_result = run_search_benchmark(
            &engine,
            &query_vectors,
            bench_docs.as_deref(),
            args.top_k,
            args.ef_search,
        )?;
        search_stats = search_result.search_stats;
        filtered_stats = search_result.filtered_stats;
        search_quality = search_result.search_quality;
        filtered_quality = search_result.filtered_quality;
    }

    println!("Munind Benchmark Summary");
    println!("  mode: {}", mode_label(&args));
    println!("  input: {}", args.input.display());
    println!("  docs_loaded: {}", texts.len());
    println!("  docs_inserted: {}", inserted_docs);
    println!("  queries: {}", queried_docs);
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

    if let Some(ref path) = args.output_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let summary = BenchmarkSummary {
            inserted_docs,
            queried_docs,
            dimension: args.dimension,
            embedding_model: embedder.model_id().to_string(),
            quality_top_k: args.top_k,
            mode: mode_label(&args).to_string(),
            insert: insert_stats,
            search: search_stats,
            filtered_search: filtered_stats,
            search_quality,
            filtered_search_quality: filtered_quality,
        };
        fs::write(path, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("failed to write summary to {}", path.display()))?;
        println!();
        println!("Wrote JSON summary to {}", path.display());
    }

    Ok(())
}
