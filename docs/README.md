# Munind Documentation

This directory contains implementation and usage documentation for Munind.

## Start Here

- [Architecture](./architecture.md)
  - Crate boundaries, storage/index lifecycle, search pipeline.
- [Search and Ranking](./search.md)
  - Vector search, hybrid retrieval, payload-indexed filters, reranking.
- [CLI Guide](./cli.md)
  - End-to-end local usage with deterministic or external embedding/reranker services.
- [Rust API Guide](./rust-api.md)
  - Programmatic create/insert/search/optimize examples.
- [Benchmarking](./benchmarking.md)
  - Speed and quality evaluation (recall@k, filtered recall@k, MRR, nDCG).
- [Operations](./operations.md)
  - Local data layout, backup/restore, troubleshooting.

## Scope

Munind is local-only by design. Documentation focuses on desktop/mobile local deployments rather than cloud scaling.
