use anyhow::{Context, Result};
use clap::Parser;
use munind_api::MunindEngine;
use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::{OptimizeRequest, SearchRequest};
use munind_core::engine::VectorEngine;
use munind_storage::StorageEngine;
use serde::Serialize;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Benchmark Munind with ANN-Benchmarks style GloVe data"
)]
struct Args {
    /// Raw little-endian float32 matrix for train vectors.
    #[arg(long, default_value = "benchmark/data/glove-100-angular.train.f32")]
    train_f32: PathBuf,

    /// Raw little-endian float32 matrix for test/query vectors.
    #[arg(long, default_value = "benchmark/data/glove-100-angular.test.f32")]
    test_f32: PathBuf,

    /// Raw little-endian int32 matrix for exact nearest-neighbor ids.
    #[arg(long, default_value = "benchmark/data/glove-100-angular.neighbors.i32")]
    neighbors_i32: PathBuf,

    /// Database path for this benchmark.
    #[arg(long, default_value = "benchmark/glove_100_db")]
    db_path: PathBuf,

    /// Vector dimension.
    #[arg(long, default_value_t = 100)]
    dimension: usize,

    /// Query-time top-k.
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Ground-truth width in neighbors matrix.
    #[arg(long, default_value_t = 100)]
    ground_truth_k: usize,

    /// Query-time ef_search.
    #[arg(long, default_value_t = 80)]
    ef_search: usize,

    /// Query count (0 means all available test vectors).
    #[arg(long, default_value_t = 0)]
    queries: usize,

    /// Optional train cap (0 means all train vectors in file).
    #[arg(long, default_value_t = 0)]
    train_limit: usize,

    /// Open an existing DB instead of recreating/reinserting.
    #[arg(long, default_value_t = false)]
    use_existing_db: bool,

    /// Safety mode: run search benchmark only against an existing DB (forbids build/insert path).
    #[arg(long, default_value_t = false)]
    search_existing_db_only: bool,

    /// Build DB only; skip query benchmark.
    #[arg(long, default_value_t = false)]
    prepare_only: bool,

    /// In prepare-only mode, write checkpoint + truncate WAL after inserts (fast, no full compaction rewrite).
    #[arg(long, default_value_t = false)]
    checkpoint_wal_after_prepare: bool,

    /// Keep fsync enabled during inserts (slower, more durable).
    #[arg(long, default_value_t = false)]
    fsync_enabled: bool,

    /// Fail if the benchmark DB contains fewer vectors than full train matrix.
    #[arg(long, default_value_t = false)]
    require_full_dataset: bool,

    /// Optional JSON output file path.
    #[arg(long)]
    output_json: Option<PathBuf>,
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
struct RecallStats {
    queries_evaluated: usize,
    mean_recall_at_k: f64,
    p50_recall_at_k: f64,
    p95_recall_at_k: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    benchmark_type: String,
    mode: String,
    db_path: String,
    dimension: usize,
    train_vectors: usize,
    query_vectors: usize,
    top_k: usize,
    ef_search: usize,
    insert: Stats,
    search: Stats,
    recall: RecallStats,
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
    let total_seconds = latencies_ms.iter().sum::<f64>() / 1000.0;
    let min_ms = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let max_ms = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
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
        mean_ms: mean(latencies_ms),
    }
}

fn infer_f32_rows(path: &PathBuf, dim: usize) -> Result<usize> {
    let bytes = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len() as usize;
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("dimension too large for byte math"))?;
    if bytes % row_bytes != 0 {
        anyhow::bail!(
            "file {} size {} is not divisible by row_bytes {}",
            path.display(),
            bytes,
            row_bytes
        );
    }
    Ok(bytes / row_bytes)
}

fn infer_i32_rows(path: &PathBuf, cols: usize) -> Result<usize> {
    let bytes = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len() as usize;
    let row_bytes = cols
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("column count too large for byte math"))?;
    if bytes % row_bytes != 0 {
        anyhow::bail!(
            "file {} size {} is not divisible by row_bytes {}",
            path.display(),
            bytes,
            row_bytes
        );
    }
    Ok(bytes / row_bytes)
}

fn load_f32_matrix(path: &PathBuf, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let expected = rows
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("matrix byte size overflow"))?;
    if bytes.len() < expected {
        anyhow::bail!(
            "matrix too small for {}: expected at least {} bytes, got {}",
            path.display(),
            expected,
            bytes.len()
        );
    }

    let mut out = Vec::with_capacity(rows);
    let mut offset = 0usize;
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let b = &bytes[offset..offset + 4];
            row.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            offset += 4;
        }
        out.push(row);
    }
    Ok(out)
}

fn load_i32_matrix(path: &PathBuf, rows: usize, cols: usize) -> Result<Vec<Vec<i32>>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let expected = rows
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("matrix byte size overflow"))?;
    if bytes.len() < expected {
        anyhow::bail!(
            "matrix too small for {}: expected at least {} bytes, got {}",
            path.display(),
            expected,
            bytes.len()
        );
    }

    let mut out = Vec::with_capacity(rows);
    let mut offset = 0usize;
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let b = &bytes[offset..offset + 4];
            row.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            offset += 4;
        }
        out.push(row);
    }
    Ok(out)
}

fn recall_at_k(predicted: &[usize], exact: &[usize]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let hits = predicted.iter().filter(|id| exact.contains(id)).count();
    hits as f64 / exact.len() as f64
}

fn row_idx_from_doc(doc: &serde_json::Value) -> Option<usize> {
    let idx = doc.get("row_idx")?.as_u64()?;
    usize::try_from(idx).ok()
}

fn mode_label(args: &Args) -> &'static str {
    if args.prepare_only {
        if args.use_existing_db {
            "prepare_only_existing_db"
        } else {
            "prepare_only"
        }
    } else if args.use_existing_db {
        "search_existing_db"
    } else {
        "build_and_search"
    }
}

fn zero_stats() -> Stats {
    compute_stats(&[])
}

fn zero_recall() -> RecallStats {
    RecallStats {
        queries_evaluated: 0,
        mean_recall_at_k: 0.0,
        p50_recall_at_k: 0.0,
        p95_recall_at_k: 0.0,
    }
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

fn main() -> Result<()> {
    let args = Args::parse();

    if args.search_existing_db_only {
        if !args.use_existing_db {
            anyhow::bail!(
                "--search-existing-db-only requires --use-existing-db (build path is forbidden)"
            );
        }
        if args.prepare_only {
            anyhow::bail!(
                "--search-existing-db-only cannot be combined with --prepare-only (search is required)"
            );
        }
    }

    let train_rows_all = infer_f32_rows(&args.train_f32, args.dimension)?;
    let test_rows = if args.prepare_only {
        0
    } else {
        infer_f32_rows(&args.test_f32, args.dimension)?
    };
    let neighbors_rows = if args.prepare_only {
        0
    } else {
        infer_i32_rows(&args.neighbors_i32, args.ground_truth_k)?
    };
    let train_rows = if args.train_limit == 0 {
        train_rows_all
    } else {
        args.train_limit.min(train_rows_all)
    };

    if !args.prepare_only && neighbors_rows < test_rows {
        anyhow::bail!(
            "neighbors rows ({}) must be >= test rows ({})",
            neighbors_rows,
            test_rows
        );
    }

    let query_count = if args.prepare_only {
        0
    } else {
        let max_queries = test_rows.min(neighbors_rows);
        if args.queries == 0 {
            max_queries
        } else {
            args.queries.min(max_queries)
        }
    };

    println!("GloVe Benchmark (ANN-only)");
    println!(
        "  train rows: {} (available: {})",
        train_rows, train_rows_all
    );
    println!("  test rows: {}", test_rows);
    println!("  queries: {}", query_count);
    println!("  dim: {}", args.dimension);
    println!("  top_k: {}", args.top_k);
    println!("  ef_search: {}", args.ef_search);
    println!("  fsync_enabled: {}", args.fsync_enabled);
    println!("  db_path: {}", args.db_path.display());

    let mut insert_latencies_ms = Vec::new();
    let mut effective_train_rows = train_rows;
    let mode = mode_label(&args).to_string();

    let engine = if args.use_existing_db {
        let engine = if args.search_existing_db_only {
            MunindEngine::open_ann_only(&args.db_path)
        } else {
            MunindEngine::open(&args.db_path)
        }
        .with_context(|| format!("failed to open {}", args.db_path.display()))?;
        if engine.embedding_dimension() != args.dimension {
            anyhow::bail!(
                "dimension mismatch: db={}, requested={}",
                engine.embedding_dimension(),
                args.dimension
            );
        }
        let storage = StorageEngine::open(&args.db_path)
            .with_context(|| format!("failed to open storage {}", args.db_path.display()))?;
        effective_train_rows = storage.get_all_ids()?.len();
        engine
    } else {
        if args.db_path.exists() {
            fs::remove_dir_all(&args.db_path)
                .with_context(|| format!("failed to remove old db {}", args.db_path.display()))?;
        }

        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::Cosine;
        cfg.index.ef_search = args.ef_search;
        cfg.query.ef_search = args.ef_search;
        cfg.storage.fsync_enabled = args.fsync_enabled;

        let engine = MunindEngine::create(&args.db_path, args.dimension, cfg)
            .with_context(|| format!("failed to create {}", args.db_path.display()))?;

        let file = File::open(&args.train_f32)
            .with_context(|| format!("failed to open {}", args.train_f32.display()))?;
        let mut reader = BufReader::new(file);
        let mut row_buf = vec![0u8; args.dimension * 4];

        for i in 0..train_rows {
            reader
                .read_exact(&mut row_buf)
                .with_context(|| format!("failed reading train row {}", i))?;

            let mut vector = Vec::with_capacity(args.dimension);
            for chunk in row_buf.chunks_exact(4) {
                vector.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            let doc = json!({ "row_idx": i });
            let t0 = Instant::now();
            engine.insert_json(vector, doc)?;
            insert_latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            if (i + 1) % 50_000 == 0 || i + 1 == train_rows {
                eprintln!("inserted {}/{} vectors", i + 1, train_rows);
            }
        }

        if args.prepare_only && args.checkpoint_wal_after_prepare {
            eprintln!("running checkpoint_wal_only optimize after prepare inserts...");
            let report = engine.optimize(OptimizeRequest {
                force_full_compaction: false,
                repair_graph: false,
                checkpoint_wal_only: true,
            })?;
            eprintln!(
                "checkpoint_wal_only complete: space_reclaimed_bytes={}",
                report.space_reclaimed_bytes
            );
        }

        engine
    };

    if args.require_full_dataset && effective_train_rows < train_rows_all {
        anyhow::bail!(
            "full-dataset check failed: db has {} vectors but train matrix has {}. Rebuild with TRAIN_LIMIT=0.",
            effective_train_rows,
            train_rows_all
        );
    }

    let mut search_latencies_ms = Vec::new();
    let mut recalls = Vec::new();

    if !args.prepare_only {
        let test_vectors = load_f32_matrix(&args.test_f32, query_count, args.dimension)?;
        let neighbors = load_i32_matrix(&args.neighbors_i32, query_count, args.ground_truth_k)?;

        search_latencies_ms = Vec::with_capacity(query_count);
        recalls = Vec::with_capacity(query_count);

        for i in 0..query_count {
            let req = SearchRequest {
                vector: test_vectors[i].clone(),
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

            let mut predicted = Vec::with_capacity(hits.len());
            for hit in hits {
                let row_idx = row_idx_from_doc(&hit.document).ok_or_else(|| {
                    anyhow::anyhow!("query {}: search result missing numeric row_idx", i)
                })?;
                predicted.push(row_idx);
            }

            let gt_ids: Vec<usize> = neighbors[i]
                .iter()
                .take(args.top_k)
                .filter_map(|&v| usize::try_from(v).ok())
                .filter(|&id| id < effective_train_rows)
                .collect();
            if !gt_ids.is_empty() {
                recalls.push(recall_at_k(&predicted, &gt_ids));
            }

            if (i + 1) % 1000 == 0 || i + 1 == query_count {
                eprintln!("searched {}/{} queries", i + 1, query_count);
            }
        }
    }

    let insert_stats = compute_stats(&insert_latencies_ms);
    let search_stats = if args.prepare_only {
        zero_stats()
    } else {
        compute_stats(&search_latencies_ms)
    };
    let recall_stats = if args.prepare_only {
        zero_recall()
    } else {
        RecallStats {
            queries_evaluated: recalls.len(),
            mean_recall_at_k: mean(&recalls),
            p50_recall_at_k: percentile(&recalls, 50.0),
            p95_recall_at_k: percentile(&recalls, 95.0),
        }
    };

    println!();
    println!("Munind GloVe Benchmark Summary");
    println!("  benchmark_type: ann_only");
    println!("  mode: {}", mode);
    println!("  train_vectors: {}", effective_train_rows);
    println!("  query_vectors: {}", query_count);
    println!("  top_k: {}", args.top_k);
    println!("  ef_search: {}", args.ef_search);
    print_stats("Insert", &insert_stats);
    print_stats("Search", &search_stats);
    if !args.prepare_only {
        println!();
        println!(
            "Recall@{} vs provided neighbors: mean={:.4} p50={:.4} p95={:.4} (queries={})",
            args.top_k,
            recall_stats.mean_recall_at_k,
            recall_stats.p50_recall_at_k,
            recall_stats.p95_recall_at_k,
            recall_stats.queries_evaluated
        );
    }

    if let Some(path) = args.output_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let summary = Summary {
            benchmark_type: "ann_only".to_string(),
            mode,
            db_path: args.db_path.display().to_string(),
            dimension: args.dimension,
            train_vectors: effective_train_rows,
            query_vectors: query_count,
            top_k: args.top_k,
            ef_search: args.ef_search,
            insert: insert_stats,
            search: search_stats,
            recall: recall_stats,
        };
        fs::write(&path, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("failed writing {}", path.display()))?;
        println!("Wrote JSON summary to {}", path.display());
    }

    Ok(())
}
