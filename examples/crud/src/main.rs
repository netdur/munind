mod create;
mod delete;
mod read;
mod update;

use anyhow::Result;
use munind_api::MunindEngine;
use munind_core::config::{
    DistanceMetric, EngineConfig, IndexConfig, QueryConfig, RuntimeConfig, StorageConfig,
    TelemetryConfig,
};

fn main() -> Result<()> {
    // Change this with:
    //   MUNIND_EXAMPLE_DB=/path/to/db cargo run --manifest-path examples/crud/Cargo.toml
    let db_path =
        std::env::var("MUNIND_EXAMPLE_DB").unwrap_or_else(|_| "./examples/crud/tmp_db".to_string());

    let db_dir = std::path::Path::new(&db_path);
    if db_dir.exists() {
        // Keep the example repeatable: every run starts from a clean DB.
        std::fs::remove_dir_all(db_dir)?;
    }

    // Build a full explicit config so developers can see each option.
    // For most apps, start from `EngineConfig::default()` and only override what you need.
    let config = EngineConfig {
        storage: StorageConfig {
            // Logical path stored in config/manifest metadata. The `create` call still
            // uses `db_dir` as the concrete location for the DB files.
            path: db_path.clone(),
            // `true`: safer on crash (fsync WAL writes), slower writes.
            // `false`: faster writes, less durability for sudden power loss.
            fsync_enabled: false,
            // Future-facing snapshot cadence hint (seconds).
            snapshot_interval_sec: 3600,
        },
        index: IndexConfig {
            // Vector distance metric for ANN graph ranking.
            // - Cosine: common for normalized embeddings.
            // - L2: euclidean distance.
            // - InnerProduct: max dot-product style retrieval.
            metric: DistanceMetric::Cosine,
            // Max neighbors in upper HNSW layers.
            m: 16,
            // Max neighbors in base layer (usually 2*m).
            m0: 32,
            // Level generation multiplier; controls graph layer distribution.
            // This is the same heuristic as default config.
            ml: 1.0 / 16.0_f32.ln(),
            // Candidate breadth during insert/build.
            // Higher -> better recall potential, slower build.
            ef_construction: 200,
            // Default candidate breadth during ANN search.
            // Higher -> better recall, slower queries.
            ef_search: 80,
        },
        query: QueryConfig {
            // Default top-k when callers do not provide one.
            default_top_k: 10,
            // Default query-time ANN breadth if request omits `ef_search`.
            ef_search: 80,
            // Exploration tuning hook (kept at default).
            exploration_factor: 1.1,
        },
        runtime: RuntimeConfig {
            // Worker count for runtime-managed tasks.
            // Use available CPU parallelism by default.
            thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        },
        telemetry: TelemetryConfig {
            // Internal metrics/tracing toggles.
            metrics_enabled: false,
            tracing_enabled: false,
        },
    };

    // For this example we use 3 dimensions to keep vectors short/readable.
    let engine = MunindEngine::create(db_dir, 3, config)?;
    println!("Created example DB at {}", db_path);

    let id = create::run(&engine)?;
    read::run(&engine, id, vec![1.0, 0.0, 0.0], "before_update")?;
    update::run(&engine, id)?;
    read::run(&engine, id, vec![0.0, 1.0, 0.0], "after_update")?;
    delete::run(&engine, id)?;

    println!("CRUD example completed successfully.");
    Ok(())
}
