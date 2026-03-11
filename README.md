# Munind

Munind is a local-only, general-purpose vector database in Rust.

It stores fixed-dimension embeddings with JSON payloads and returns matching JSON documents for vector, hybrid, and filtered retrieval.

Munind is application-agnostic at the core. It is not tied to a single memory or RAG workflow.

## Why Munind

- Local-only by design (desktop/mobile-oriented, no cloud scaling goal)
- Database-style workflow: insert `embedding + JSON`, search returns JSON
- Fixed embedding dimension set at database creation
- Hybrid retrieval support (vector + BM25F)
- Optional second-stage reranking
- Indexed JSON filter execution for common metadata fields

## Why Munind (no, really)

- I built Munind because my AI agents needed a memory database, and tools like Milvus felt like overkill for a local-first setup

## Current Capabilities

- Create/open DB with immutable embedding dimension
- CRUD over vectors with JSON payloads (`insert`, `get`, `update`, `delete`)
- Vector ANN search with optional `ef_search` and radius
- Hybrid search (`text_query` + vector fusion)
- Payload-filtered search (`Eq`, `And`) with indexed execution paths
- Storage optimize/compaction to checkpoint state and truncate WAL
- Ingestion/search pipeline crate with pluggable embedding and reranker providers
- Benchmarking for both speed and quality (`recall@k`, filtered recall, MRR, nDCG)

## Example Use Cases

- Semantic search over app documents/messages
- Metadata-filtered vector retrieval for local products
- Personal memory and note retrieval
- RAG context retrieval for local assistants

## Quick Start (CLI)

### 1) Build

```bash
cargo build --workspace
```

### 2) Create DB (512 dim)

```bash
cargo run -p munind-cli -- --db ./munind_data create --embedding-dim 512
```

### 3) Ingest a text file

```bash
cargo run -p munind-cli -- --db ./munind_data ingest --file ./test_doc.txt --doc-id demo-1
```

### 4) Search

```bash
cargo run -p munind-cli -- --db ./munind_data search "quick brown fox" -k 5
```

### 4b) CRUD via JSON (Developer-Friendly)

Insert:

```bash
cargo run -p munind-cli -- --db ./munind_data \
  insert \
  --embedding-json '[0.1,0.2,0.3]' \
  --document-json '{"doc_id":"x1","text":"hello"}'
```

Get:

```bash
cargo run -p munind-cli -- --db ./munind_data get --id 1 --include-embedding
```

Update:

```bash
cargo run -p munind-cli -- --db ./munind_data \
  update --id 1 \
  --embedding-json '[0.2,0.1,0.0]' \
  --document-json '{"doc_id":"x1","text":"updated"}'
```

Delete:

```bash
cargo run -p munind-cli -- --db ./munind_data delete --id 1
```

### 5) Health check

```bash
cargo run -p munind-cli -- --db ./munind_data check-health
```

### 6) Optimize (Compaction / WAL Truncation)

```bash
cargo run -p munind-cli -- --db ./munind_data optimize
```

## Benchmark Quick Start (Real Embeddings)

Defaults are already in scripts:
- endpoint: `http://localhost:8082/v1/embeddings`
- model: `nomic-embed-text-v1.5`

Set `EMBEDDING_API_KEY` only if your endpoint requires auth.

Run the 3 phases:

```bash
bash benchmark/01_build_dataset.sh
bash benchmark/02_create_database.sh
bash benchmark/03_run_benchmark.sh
```

Phase 2 runs post-prepare compaction by default (checkpoint + WAL truncation).  
Set `COMPACT_AFTER_PREPARE=0` to skip it.

`03_run_benchmark.sh` is latency-only by default. For exact quality metrics too:

```bash
BENCH_WITH_QUALITY=1 bash benchmark/03_run_benchmark.sh
```

Quality mode uses vectors already stored in the DB for exact baseline (it does not re-embed all docs from input).

Or one command:

```bash
bash benchmark/run_benchmark.sh
```

## GloVe ANN Benchmark (Full Dataset)

Build full DB once:

```bash
TRAIN_LIMIT=0 EXPORT_RAW=0 CHECKPOINT_WAL_AFTER_PREPARE=1 COMPACT_AFTER_PREPARE=0 bash benchmark/02_create_glove_database.sh
```

Run ANN search benchmark on existing full DB:

```bash
EF_SEARCH=400 QUERIES=10000 bash benchmark/03_run_glove_benchmark.sh
```

Latest snapshot (2026-03-08, cosine ANN, `top_k=10`, `train_vectors=1183514`):
- Search: `238.38 ops/s`, `p50 4.255 ms`, `p95 5.624 ms`, `p99 6.406 ms`, `mean 4.195 ms`
- Recall@10: `mean 0.9100`, `p50 1.0000`, `p95 1.0000`

## Documentation

- [Documentation Index](./docs/README.md)
- [Architecture](./docs/architecture.md)
- [Search and Ranking](./docs/search.md)
- [CLI Guide](./docs/cli.md)
- [Language Integration](./docs/language-integration.md)
- [Rust API Guide](./docs/rust-api.md)
- [Benchmarking](./docs/benchmarking.md)
- [Operations](./docs/operations.md)

## Repository Layout

- `crates/munind-core`
- `crates/munind-storage`
- `crates/munind-index`
- `crates/munind-api`
- `crates/munind-rag`
- `crates/munind-cli`
- `crates/munind-bench`
- `benchmark/`
- `docs/`

## License

MIT. See [LICENSE](./LICENSE).
