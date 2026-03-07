# 03 - Architecture

## Workspace Layout
Recommended Rust workspace:
- `crates/munind-core`: domain types, traits, configs, errors.
- `crates/munind-storage`: WAL, segments, snapshots, mmap helpers.
- `crates/munind-index`: graph index, optional tree seed index, search.
- `crates/munind-rag`: retrieval pipeline, reranking, context assembly.
- `crates/munind-cli`: command line tools.
- `crates/munind-local-api` (optional): local adapter layer for desktop/mobile integration (desktop prioritized first).
- `crates/munind-bench`: benchmark harness + recall tooling.

## Core Traits
```rust
pub trait VectorEngine {
    fn create_database(&self, embedding_dimension: usize, config: CreateConfig) -> Result<()>;
    fn insert_json(&self, embedding: Vec<f32>, document: serde_json::Value) -> Result<MemoryId>;
    fn insert_json_batch(&self, rows: Vec<(Vec<f32>, serde_json::Value)>) -> Result<Vec<MemoryId>>;
    fn search(&self, query: SearchRequest) -> Result<Vec<SearchHit>>;
    fn remove(&self, id: MemoryId) -> Result<()>;
    fn flush(&self) -> Result<()>;
    fn optimize(&self, req: OptimizeRequest) -> Result<OptimizeReport>;
}
```

## Main Components
- `EngineCoordinator`: orchestrates writes/searches across storage and index.
- `VectorStore`: dense vector persistence + retrieval by ID.
- `JsonDocumentStore`: JSON document persistence and retrieval by ID.
- `NeighborGraphIndex`: ANN graph data structure.
- `SeedIndex` (optional): VP-tree for seed candidate generation.
- `FilterIndex`: metadata index for fast pre/post filtering.

## Concurrency Model
- Read-heavy lock strategy:
  - Search path uses mostly lock-free or `RwLock` read sections.
  - Write path uses append-only WAL and batched graph mutation lock.
- Background workers:
  - graph maintenance (edge pruning/repair)
  - compaction/snapshot
  - optional re-embedding queue for model upgrades

## ID And Versioning Strategy
- `MemoryId`: monotonic `u64` allocator.
- `RecordVersion`: increment per update.
- Tombstones for delete; physical cleanup on compaction.

## Distance Support
Distance enum:
- `L2`
- `Cosine`
- `InnerProduct`

Implementation plan:
- SIMD accelerated kernels where safe (`std::simd`/target features).
- Fallback scalar kernels for compatibility.

## Config Model
`EngineConfig` sections:
- `storage`: paths, fsync policy, snapshot interval.
- `index`: graph degree bounds, ef parameters, seed strategy, immutable `embedding_dimension`.
- `query`: defaults for top-k, rerank depth, filters.
- `runtime`: thread counts, queue sizes.
- `telemetry`: metrics/tracing toggles.

## State Lifecycle
- `create`: initialize directories + manifest + empty stores with fixed `embedding_dimension`.
- `open`: load manifest, mmap segments, replay WAL, open index.
- `run`: allow local read/write traffic.
- `optimize`: compact and graph repair.
- `close`: flush and checkpoint.
