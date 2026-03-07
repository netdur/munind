# 02 - NGT Reference Mapping (Without NGTQ)

This project uses `vendor/NGT/lib/NGT` as reference for architecture and algorithms.

## Explicit Exclusion
Do not base core design on:
- `vendor/NGT/lib/NGT/NGTQ/*`

## Key NGT Concepts To Preserve
From `NGT/lib/NGT`:
- `Index` as top-level facade for create/open/insert/search/remove/save/load.
- `Property` style config for dimension, distance, and index mode.
- `GraphRepository` as ANN graph backbone.
- `ObjectRepository` as object store with stable IDs.
- `DVPTree` style optional tree for seed generation.
- `Command` style CLI around core engine.

## Rust Mapping
- `Index` -> `Engine` trait + `GraphEngine` implementation.
- `Index::Property` -> `EngineConfig` (`serde` serializable), including immutable `embedding_dimension` set at DB creation.
- `GraphRepository` -> `NeighborGraphStore`.
- `ObjectRepository` -> `VectorStore` + `JsonDocumentStore`.
- `DVPTree` -> `SeedIndex` trait with `VpTreeSeedIndex` optional module.
- `Command` -> `munind-cli` crate using `clap`.

## Algorithmic Mapping
- NGT incremental insertion flow -> greedy ANN insertion with local edge repair.
- NGT search with epsilon/exploration -> `ef_search` + `exploration_factor`.
- NGT removal with edge cleanup -> tombstone + periodic graph repair jobs.
- NGT tree-assisted seed -> optional seed provider interface.

## Modernization Changes
- C++ macro-heavy options -> typed Rust `enum` and feature flags.
- Manual memory handling -> safe ownership + `Arc` + arenas where needed.
- File writes -> atomic manifest updates, checksums, WAL replay.
- Thread pool tuning -> Rayon/custom executor with bounded queues.
- Error handling -> `thiserror` + context-rich error propagation.

## What To Keep Intentionally Similar
- 1-based or stable object ID semantics (with explicit policy).
- Separate concerns between object storage and graph connectivity.
- Ability to build index in batches and update incrementally.
- Configurable distance metrics and object types.

## What To Deliberately Improve
- Deterministic build mode for reproducible benchmarks.
- Better corruption detection (checksums + magic headers + versioning).
- Clear online/offline maintenance commands.
- Rich metrics for recall/latency/build quality.
- Test-first API boundaries.
