# 08 - Testing, Benchmarking, And Release

## Test Pyramid
- Unit tests:
  - distance kernels
  - heap/search utilities
  - config parsing
  - WAL codec
- Integration tests:
  - insert/search/delete flows
  - crash and recovery
  - snapshot/restore
- Property tests:
  - ID consistency
  - deterministic behavior in fixed-seed mode
- Fuzz tests:
  - WAL/index parser robustness

## Correctness Baselines
- Compare ANN results against brute-force exact search.
- Track recall@k, NDCG, and hit stability across versions.

## Benchmark Harness
Scenarios:
- synthetic random vectors
- public ANN-style datasets (local fixtures)
- domain-like corpus with metadata filters (for example: personal notes/messages)

Measure:
- p50/p95/p99 search latency
- indexing throughput
- memory footprint
- disk footprint
- build/optimize durations

## CI Gates
Required for merge:
- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test`
- minimal benchmark smoke gate

Required for release branch:
- recall threshold gate on fixed benchmark datasets
- no regression beyond latency/error budget thresholds

## Release Strategy
- `alpha`: local use, no compatibility guarantees.
- `beta`: file format freeze candidate, migration tests active.
- `v1.0`: compatibility contract + stable CLI/API.

## Runbooks
Document runbooks for:
- crash recovery
- corruption/repair
- backup and restore
- index reconfiguration and rebuild
