# 01 - Product And Requirements

## Product Goal
Build a local-only, general-purpose vector database in Rust that stores embeddings with JSON documents and retrieves relevant results quickly and reliably.

## Platform Priority
- Desktop integration first.
- Mobile integration second.

## Primary Use Cases
- Insert a record as `embedding + JSON document`.
- Search documents by semantic similarity with metadata filters.
- Build context windows for downstream applications (including RAG) from top-k JSON search results.
- Update or delete records cleanly.
- Rebuild or optimize the index without data loss.

## Non-Goals (Phase 1)
- Distributed multi-node clustering.
- Cloud-native scaling.
- Server-first deployment model.
- GPU-native index build.
- Billion-scale indexing.
- Full NGTQ/QBG style quantized systems.

## Workload Assumptions
- Data scale: 100k to 20M documents/chunks over time.
- Embedding dimension is selected at database creation and fixed for that database.
- Default first profile dimension: `512`.
- Typical dimension choices over time: 384 / 512 / 768 / 1024 / 1536.
- Write pattern: steady appends, occasional bulk imports.
- Read pattern: frequent top-k queries with filters.

## Functional Requirements
- Insert, update, delete records.
- Insert accepts `embedding + JSON document`.
- Search returns JSON documents with ids/scores.
- Batch insert API.
- ANN search with configurable latency/recall tradeoff.
- Vector search supports equality filter on JSON fields in v1 (`field = value`).
- Filter by metadata fields (tags, source, date range, type).
- Radius search support for threshold retrieval.
- Snapshot/export/import support.
- Enforce embedding dimension match against database schema.

## Non-Functional Requirements
- Durability: no acknowledged writes lost after crash.
- Predictable latency: p95 under target for configured dataset profile.
- Crash recovery from WAL.
- Strong input validation and corruption detection.
- Backward-compatible file format migration path.

## Initial SLO Targets
- Search p95: <= 20 ms for top-20 on 1M vectors (512 dim, cosine), on agreed target hardware.
- Ingestion throughput: > 20k vectors/min single machine baseline.
- Recovery time: < 2 minutes for clean restart on medium dataset.
- Recall target: >= 0.92 recall@10 against brute-force baseline dataset.

## Data Model
Record shape:
- `doc_id: u64`
- `embedding: Vec<f32>` (length must equal database `embedding_dimension`)
- `document: serde_json::Value` (primary payload returned by search)
- `created_at: i64` (unix epoch ms)
- `updated_at: i64`
- `is_deleted: bool`

## Query Model
- `VectorSearch`: query vector + k + filters + score options.
- `FilterExpression` v1: JSON equality predicates (`field = value`) with AND composition.
- `RangeSearch`: vector + radius + max_results.
- `HybridSearch`: vector search + lexical stage + rerank stage.

## Ranking Model For Application Signals
Composite score example:
- `final = alpha * semantic + beta * recency + gamma * importance`
- recency decay configurable by half-life.
- explicit importance boosts can be enabled by applications.

This model is especially useful for personal-memory style applications, but remains optional at DB core level.

## Security/Privacy Requirements
- Local-only defaults.
- Optional encryption at rest for JSON payload and metadata.
- Redaction path for sensitive JSON fields in logs.
