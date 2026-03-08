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

## Current Capabilities

- Create/open DB with immutable embedding dimension
- Insert/delete vectors with JSON payloads
- Vector ANN search with optional `ef_search` and radius
- Hybrid search (`text_query` + vector fusion)
- Payload-filtered search (`Eq`, `And`) with indexed execution paths
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

### 5) Health check

```bash
cargo run -p munind-cli -- --db ./munind_data check-health
```

## Benchmark Quick Start

```bash
bash benchmark/run_benchmark.sh
```

Manual run:

```bash
python3 benchmark/download_tinystories_subset.py \
  --output benchmark/data/tinystories_subset.jsonl \
  --limit 1000

cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --dimension 512 \
  --limit 1000 \
  --queries 200 \
  --top-k 10 \
  --ef-search 80 \
  --output-json benchmark/results/summary.json
```

## Documentation

- [Documentation Index](./docs/README.md)
- [Architecture](./docs/architecture.md)
- [Search and Ranking](./docs/search.md)
- [CLI Guide](./docs/cli.md)
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
