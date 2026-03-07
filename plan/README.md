# Munind Vector DB Plan

This folder contains a full implementation plan for building a Rust vector database for personal memory RAG.

Reference base:
- Use `vendor/NGT/lib/NGT` as the design reference.
- Do not use `vendor/NGT/lib/NGT/NGTQ` as a base.

Design direction:
- Keep NGT's strong core ideas (graph index, optional tree seeding, incremental insertion/removal).
- Rebuild with modern Rust engineering: typed APIs, safe persistence, strong test strategy, observability, and staged rollout.

## Reading Order
1. `EXPECTATIONS.md`
2. `01-product-and-requirements.md`
3. `02-ngt-reference-mapping.md`
4. `03-architecture.md`
5. `04-storage-and-durability.md`
6. `05-indexing-and-search.md`
7. `06-rag-memory-features.md`
8. `07-api-cli-and-observability.md`
9. `08-testing-benchmarking-release.md`
10. `09-implementation-roadmap.md`

## End State
By the end of this plan, the project should provide:
- Persistent local-only memory store for embeddings + JSON documents.
- Desktop/mobile oriented architecture (single-device focus).
- High recall approximate nearest-neighbor search with low-latency reads.
- Hybrid retrieval for RAG (vector + filters + optional lexical/rerank).
- Durable write path with WAL + snapshot + recovery.
- Production-quality CI, benchmarking, and observability.

## Scope Notes
- Quantization research can be added later, but initial core should ship without NGTQ dependencies.
- Priority is correctness and maintainability first, then aggressive optimization.
