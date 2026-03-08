# Architecture

## Design Intent

Munind is a local-only Rust vector database.

Key constraints:
- Fixed embedding dimension per database.
- Insert path is database-style: `embedding + JSON document`.
- Search returns JSON documents.
- No cloud coordination assumptions.

Example workloads this architecture supports:
- local personal-memory retrieval
- local RAG context retrieval
- semantic search over structured app documents

## Workspace Layout

- `crates/munind-core`
  - Shared config/domain/error types (`SearchRequest`, `FilterExpression`, `MemoryId`).
- `crates/munind-storage`
  - Durable storage engine (WAL + vector/doc segment files + ID allocator).
- `crates/munind-index`
  - In-memory ANN graph index and exact scoring helpers.
- `crates/munind-api`
  - `MunindEngine`: composes storage, vector index, lexical index, payload index.
- `crates/munind-rag`
  - Ingestion/retrieval pipeline module with embedding provider abstraction and optional reranking.
- `crates/munind-cli`
  - User-facing create/ingest/search/check-health commands.
- `crates/munind-bench`
  - Benchmark harness for speed and quality.

## Data Path

### Write Path

1. Client calls `insert_json(embedding, document)`.
2. Storage appends to WAL first.
3. Storage writes vector bytes and JSON bytes to segment files.
4. ID allocator maps `MemoryId -> segment offsets`.
5. API layer updates in-memory indexes:
   - ANN graph/vector index
   - BM25F lexical index
   - payload exact-match index

### Read Path

1. Client issues `SearchRequest`.
2. API plans filter with payload index (if filter exists).
3. Candidate retrieval:
   - ANN search, or
   - exact filtered vector scoring for indexed candidate sets.
4. Optional lexical candidate scoring (hybrid mode).
5. Score fusion for hybrid mode.
6. Final filter verification for partial plans.
7. JSON payload fetch from storage for returned IDs.
8. Optional second-stage reranking in the pipeline module.

## Indexes

### Vector Index (`munind-index`)

- Graph-based ANN index for low-latency nearest-neighbor retrieval.
- Exact filtered scorer available for payload-constrained candidate sets.

### Lexical Index (`munind-api::lexical`)

- In-memory BM25F over fields:
  - `text`
  - `title` (or `metadata.title`)
  - `tags` (or `metadata.tags`)

### Payload Index (`munind-api::payload_index`)

Exact-match postings on frequent filter fields:
- `doc_id`, `source`, `type`, `created_at`, `tags`, `session_id`
- `metadata.doc_id`, `metadata.source`, `metadata.type`, `metadata.created_at`, `metadata.tags`, `metadata.session_id`

## Durability Model

- WAL is authoritative for recovery.
- On open, storage replays WAL and rebuilds segments/state.
- `optimize(force_full_compaction)` compacts live records and rebuilds index state.

## Current Non-Goals

- Multi-node distribution.
- Cloud-native autoscaling primitives.
- Cross-process transactional guarantees beyond current local engine behavior.
