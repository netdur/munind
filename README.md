# Munind

Munind is a local-only memory vector database in Rust for personal RAG workflows.
It stores fixed-dimension embeddings with JSON documents, then returns matching JSON documents on vector search.

## What this is for

- Building personal memory systems (notes, chats, docs, knowledge snippets)
- Running semantic retrieval on your own machine
- Combining vector similarity with JSON filtering (for example: search only where `source == \"journal\"`)

## Who this is for

- Developers building local AI assistants
- Makers who want a lightweight embedded vector DB
- Desktop-first and mobile-oriented apps that do not need cloud scaling

## Why Munind

- Local-only by design: no cloud dependency required
- Database-style workflow: insert embedding + JSON, search returns JSON docs
- Fixed embedding dimension set at database creation
- Fast query path with low-latency targets

## Current capabilities

- Create/open database with immutable embedding dimension
- Insert vectors with JSON payloads
- Vector search with Top-K
- Filtered vector search with JSON predicates (equality in v1)
- CLI ingestion/search flow for text files
- Benchmark runner with JSON summary output

## Repository layout

- `crates/munind-core`: core types/config/domain
- `crates/munind-index`: vector index implementation
- `crates/munind-storage`: segments/WAL/snapshots storage
- `crates/munind-api`: engine API (`MunindEngine`)
- `crates/munind-rag`: RAG-oriented ingest/search pipeline
- `crates/munind-cli`: command-line interface
- `crates/munind-bench`: benchmark binary
- `benchmark/`: benchmark scripts, data, results

## Quick start

### 1) Build

```bash
cargo build --workspace
```

### 2) Create a database (512-dim)

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

## Benchmark quick start

One-command run:

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

## Performance snapshot

From local benchmark (`1000` docs, `200` queries, `512` dim):

- Insert: `158.74 ops/s`, latency `p50 6.125 ms`, `p95 8.400 ms`, `p99 9.624 ms`
- Search: `3845.66 ops/s`, latency `p50 0.245 ms`, `p95 0.333 ms`, `p99 0.406 ms`
- Filtered search (`source == "TinyStories"`): `2138.95 ops/s`, latency `p50 0.453 ms`, `p95 0.526 ms`, `p99 0.598 ms`

Current latency target baseline: `p95 <= 20 ms`.

## License

MIT. See [LICENSE](./LICENSE).
