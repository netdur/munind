use anyhow::{Context, Result};
use clap::Parser;
use munind_api::MunindEngine;
use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::{FilterExpression, SearchRequest};
use munind_core::engine::VectorEngine;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

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

    /// Top-K for search requests
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// ef_search override for query-time search breadth
    #[arg(long, default_value_t = 80)]
    ef_search: usize,

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
    insert: Stats,
    search: Stats,
    filtered_search: Stats,
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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

fn compute_stats(latencies_ms: &[f64], total_seconds: f64) -> Stats {
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total_ops = latencies_ms.len();
    let sum: f64 = latencies_ms.iter().sum();
    let mean_ms = if total_ops > 0 { sum / total_ops as f64 } else { 0.0 };

    Stats {
        total_ops,
        total_seconds,
        ops_per_second: if total_seconds > 0.0 {
            total_ops as f64 / total_seconds
        } else {
            0.0
        },
        p50_ms: percentile(&sorted, 50.0),
        p95_ms: percentile(&sorted, 95.0),
        p99_ms: percentile(&sorted, 99.0),
        min_ms: *sorted.first().unwrap_or(&0.0),
        max_ms: *sorted.last().unwrap_or(&0.0),
        mean_ms,
    }
}

fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if dim == 0 {
        return v;
    }

    for (i, b) in text.bytes().enumerate() {
        let idx = i % dim;
        let centered = (b as f32 - 127.5) / 127.5;
        v[idx] += centered;
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }

    v
}

fn load_texts(path: &PathBuf, limit: usize) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("failed to open input file {}", path.display()))?;
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
    println!("  latency_ms: p50={:.3} p95={:.3} p99={:.3} min={:.3} max={:.3} mean={:.3}",
        stats.p50_ms, stats.p95_ms, stats.p99_ms, stats.min_ms, stats.max_ms, stats.mean_ms);
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

    let mut cfg = EngineConfig::default();
    cfg.index.metric = DistanceMetric::Cosine;
    cfg.index.ef_search = args.ef_search;
    cfg.query.ef_search = args.ef_search;
    let engine = MunindEngine::create(&args.db_path, args.dimension, cfg)?;

    let mut insert_latencies_ms = Vec::with_capacity(texts.len());
    let insert_t0 = Instant::now();
    for (i, text) in texts.iter().enumerate() {
        let vec = embed_text(text, args.dimension);
        let doc = json!({
            "text": text,
            "source": "TinyStories",
            "row_idx": i
        });
        let t0 = Instant::now();
        engine.insert_json(vec, doc)?;
        insert_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let insert_total_s = insert_t0.elapsed().as_secs_f64();
    let insert_stats = compute_stats(&insert_latencies_ms, insert_total_s);

    let mut search_latencies_ms = Vec::with_capacity(query_count);
    let search_t0 = Instant::now();
    for text in texts.iter().take(query_count) {
        let req = SearchRequest {
            vector: embed_text(text, args.dimension),
            top_k: args.top_k,
            filter: None,
            ef_search: Some(args.ef_search),
            radius: None,
        };
        let t0 = Instant::now();
        let _hits = engine.search(req)?;
        search_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let search_total_s = search_t0.elapsed().as_secs_f64();
    let search_stats = compute_stats(&search_latencies_ms, search_total_s);

    let mut filtered_latencies_ms = Vec::with_capacity(query_count);
    let filtered_t0 = Instant::now();
    for text in texts.iter().take(query_count) {
        let req = SearchRequest {
            vector: embed_text(text, args.dimension),
            top_k: args.top_k,
            filter: Some(FilterExpression::Eq(
                "source".to_string(),
                json!("TinyStories"),
            )),
            ef_search: Some(args.ef_search),
            radius: None,
        };
        let t0 = Instant::now();
        let _hits = engine.search(req)?;
        filtered_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let filtered_total_s = filtered_t0.elapsed().as_secs_f64();
    let filtered_stats = compute_stats(&filtered_latencies_ms, filtered_total_s);

    println!("Munind Benchmark Summary");
    println!("  input: {}", args.input.display());
    println!("  docs_loaded: {}", texts.len());
    println!("  queries: {}", query_count);
    println!("  dimension: {}", args.dimension);
    println!("  top_k: {}", args.top_k);
    println!("  ef_search: {}", args.ef_search);

    print_stats("Insert", &insert_stats);
    print_stats("Search", &search_stats);
    print_stats("Filtered Search (source == TinyStories)", &filtered_stats);

    if let Some(path) = args.output_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let summary = BenchmarkSummary {
            inserted_docs: texts.len(),
            queried_docs: query_count,
            dimension: args.dimension,
            insert: insert_stats,
            search: search_stats,
            filtered_search: filtered_stats,
        };
        fs::write(&path, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("failed to write summary to {}", path.display()))?;
        println!();
        println!("Wrote JSON summary to {}", path.display());
    }

    Ok(())
}
