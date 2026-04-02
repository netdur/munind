/// munind SQLite extension benchmark — GloVe-100
///
/// Measures: insert, build, per-query latency (p50/p95/p99),
///           single vs multi-thread, warm vs cold cache, memory footprint.
///
/// Usage:
///     cargo build --release -p munind-sqlite --bin bench
///     target/release/bench

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use munind_core::api::{Distance, Index, IndexConfig, SearchResult};

const DIM: usize = 100;
const TOP_K: usize = 10;
const EPSILONS: [f32; 3] = [0.1, 0.2, 0.4];

fn load_tsv_vectors(path: &str) -> Vec<Vec<f32>> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {}", path, e));
    let reader = BufReader::new(file);
    let mut vecs = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f32> = line
            .split(|c: char| c == '\t' || c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        vecs.push(vals[..DIM].to_vec());
    }
    vecs
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn format_duration(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{:.1}s", s)
    } else {
        format!("{:.1}ms", s * 1000.0)
    }
}

fn recall_at_k(found: &[Vec<u32>], truth: &[Vec<usize>], k: usize) -> f64 {
    let mut hits = 0usize;
    let total = found.len() * k;
    for (f, t) in found.iter().zip(truth.iter()) {
        for &fid in f.iter().take(k) {
            // munind IDs are 1-based, ground truth is 0-based
            if t.iter().take(k).any(|&tid| tid == (fid as usize - 1)) {
                hits += 1;
            }
        }
    }
    hits as f64 / total as f64
}

fn load_ground_truth(path: &str, k: usize) -> Vec<Vec<usize>> {
    // Load from HDF5 — use a simple approach: read the binary file
    // Since we can't easily depend on hdf5 crate, we'll compute ground truth
    // via brute-force on a subset, or load from the existing index.
    // For now, we'll skip recall and focus on latency/throughput.
    // The Python script already validated recall.
    Vec::new()
}

fn bench_single_thread(
    index: &Index,
    queries: &[Vec<f32>],
    epsilon: f32,
) -> Vec<f64> {
    let mut latencies_us = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = Instant::now();
        let _ = index.search_with(q, TOP_K, epsilon, None).unwrap();
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    latencies_us
}

fn bench_multi_thread(
    index: &Index,
    queries: &[Vec<f32>],
    epsilon: f32,
) -> (Duration, usize) {
    let t0 = Instant::now();
    // search_batch uses rayon internally but hardcodes epsilon=0.1
    // So we do our own parallel loop
    use rayon::prelude::*;
    let results: Vec<_> = queries
        .par_iter()
        .map(|q| index.search_with(q, TOP_K, epsilon, None).unwrap())
        .collect();
    let elapsed = t0.elapsed();
    (elapsed, results.len())
}

fn print_latency_table(latencies_us: &[f64], label: &str) {
    let mut sorted = latencies_us.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let avg = sum / n as f64;

    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);
    let qps = 1_000_000.0 / avg;

    println!(
        "  {:18} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0}",
        label, avg, p50, p95, p99, qps,
    );
}

fn main() {
    let train_path = "benches/data/glove-100-angular.train.tsv";
    let test_path = "benches/data/glove-100-angular.test.tsv";
    let index_dir = "benches/indexes/bench_native";

    if !Path::new(train_path).exists() {
        eprintln!("Train data not found: {}", train_path);
        std::process::exit(1);
    }

    println!();
    println!("  munind benchmark — GloVe-100, cosine");
    println!("  =====================================");
    println!();

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    print!("  Loading test queries...");
    let queries = load_tsv_vectors(test_path);
    println!(" {} queries", queries.len());

    // -----------------------------------------------------------------------
    // Build (or open existing)
    // -----------------------------------------------------------------------
    let index = if Path::new(&format!("{}/obj", index_dir)).exists() {
        print!("  Opening existing index...");
        let t0 = Instant::now();
        let idx = Index::open(index_dir).unwrap();
        println!(" {} vectors in {}", idx.len(), format_duration(t0.elapsed()));
        idx
    } else {
        print!("  Loading train vectors...");
        let train = load_tsv_vectors(train_path);
        println!(" {} vectors", train.len());

        // Insert
        let config = IndexConfig::new(DIM, Distance::Cosine);
        let mut idx = Index::create(config).unwrap();

        print!("  Inserting...");
        let t0 = Instant::now();
        for v in &train {
            idx.insert(v).unwrap();
        }
        let insert_time = t0.elapsed();
        println!(
            " {:.1}s ({:.0} vec/s)",
            insert_time.as_secs_f64(),
            train.len() as f64 / insert_time.as_secs_f64()
        );

        // Build (timed separately)
        print!("  Building graph...");
        let t0 = Instant::now();
        idx.build().unwrap();
        let build_time = t0.elapsed();
        println!(
            " {:.1}s ({:.0} vec/s)",
            build_time.as_secs_f64(),
            train.len() as f64 / build_time.as_secs_f64()
        );

        // Save
        print!("  Saving...");
        let t0 = Instant::now();
        std::fs::create_dir_all(index_dir).unwrap();
        idx.save(index_dir).unwrap();
        println!(" {}", format_duration(t0.elapsed()));

        idx
    };

    // -----------------------------------------------------------------------
    // Memory footprint (file sizes)
    // -----------------------------------------------------------------------
    println!();
    println!("  File sizes:");
    let mut total_bytes = 0u64;
    for entry in std::fs::read_dir(index_dir).unwrap() {
        let entry = entry.unwrap();
        let size = entry.metadata().unwrap().len();
        total_bytes += size;
        println!(
            "    {:12} {:>10.1} MB",
            entry.file_name().to_string_lossy(),
            size as f64 / (1024.0 * 1024.0)
        );
    }
    println!(
        "    {:12} {:>10.1} MB",
        "TOTAL", total_bytes as f64 / (1024.0 * 1024.0)
    );

    // RSS estimate: index is fully in memory after open
    // We can approximate from file sizes since munind loads everything into Vec<>
    println!(
        "  ~RSS estimate: {:.0} MB (all structures in memory)",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    // -----------------------------------------------------------------------
    // Warm cache — single-thread latency
    // -----------------------------------------------------------------------
    println!();
    println!("  Single-thread latency (warm cache):");
    println!(
        "  {:18} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "", "avg_us", "p50_us", "p95_us", "p99_us", "qps"
    );
    println!("  {:18} {:>8} {:>8} {:>8} {:>8} {:>8}", "", "------", "------", "------", "------", "------");

    for &eps in &EPSILONS {
        // Warmup run
        let _ = bench_single_thread(&index, &queries[..100], eps);
        // Real run
        let latencies = bench_single_thread(&index, &queries, eps);
        print_latency_table(&latencies, &format!("e={:.1}", eps));
    }

    // -----------------------------------------------------------------------
    // Multi-thread latency
    // -----------------------------------------------------------------------
    let n_threads = rayon::current_num_threads();
    println!();
    println!("  Multi-thread throughput ({} threads, warm cache):", n_threads);
    println!(
        "  {:18} {:>10} {:>10}",
        "", "total_s", "qps"
    );
    println!("  {:18} {:>10} {:>10}", "", "-------", "-------");

    for &eps in &EPSILONS {
        // Warmup
        let _ = bench_multi_thread(&index, &queries[..100], eps);
        // Real
        let (elapsed, n) = bench_multi_thread(&index, &queries, eps);
        let qps = n as f64 / elapsed.as_secs_f64();
        println!(
            "  {:18} {:>10.3} {:>10.0}",
            format!("e={:.1}", eps),
            elapsed.as_secs_f64(),
            qps,
        );
    }

    // -----------------------------------------------------------------------
    // Cold cache — reopen from disk and search
    // -----------------------------------------------------------------------
    println!();
    println!("  Cold cache (reopen from disk, first 1000 queries):");
    println!(
        "  {:18} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "", "avg_us", "p50_us", "p95_us", "p99_us", "open_ms"
    );
    println!("  {:18} {:>8} {:>8} {:>8} {:>8} {:>10}", "", "------", "------", "------", "------", "-------");

    let cold_queries = &queries[..1000];
    for &eps in &EPSILONS {
        // Drop the current index from memory by reopening
        let t_open = Instant::now();
        let cold_index = Index::open(index_dir).unwrap();
        let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;

        let latencies = bench_single_thread(&cold_index, cold_queries, eps);
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let avg = sum / n as f64;
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        println!(
            "  {:18} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>10.1}",
            format!("e={:.1}", eps), avg, p50, p95, p99, open_ms,
        );
    }
    println!();
}
