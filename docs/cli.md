# CLI Guide

`munind-cli` provides local database creation, ingestion, and retrieval.

## Build

```bash
cargo build -p munind-cli
```

## Create Database

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  create --embedding-dim 512
```

Embedding dimension is immutable for that database directory.

## Ingest File

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  ingest --file ./test_doc.txt --doc-id demo-1
```

## CRUD Commands (JSON I/O)

Insert one record:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  insert \
  --embedding-json '[0.1,0.2,0.3]' \
  --document-json '{"doc_id":"x1","text":"hello"}'
```

Get by id:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  get --id 1 --include-embedding
```

Update by id:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  update --id 1 \
  --embedding-json '[0.2,0.1,0.0]' \
  --document-json '{"doc_id":"x1","text":"updated"}'
```

Delete by id:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  delete --id 1
```

For `insert` and `update`, you can also pass files:
- `--embedding-file <path>` instead of `--embedding-json`
- `--document-file <path>` instead of `--document-json`

## Search

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  search "what did I write about habits" -k 5
```

Machine-readable output:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  search "what did I write about habits" -k 5 --json
```

## Health Check

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  check-health
```

## Optimize (Compaction / WAL Truncation)

Compaction rewrites live records, writes a checkpoint, and truncates WAL.

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  optimize
```

Optional flags:
- `--no-compact` (skip full compaction/WAL truncation)
- `--checkpoint-wal-only` (fast path: write checkpoint + truncate WAL without full segment rewrite or index rebuild)
- `--repair-graph` (also rebuild graph/index state)

## Embedding Provider Options

By default, CLI uses deterministic local embeddings.

Use an OpenAI-compatible embedding endpoint:

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  --embedding-endpoint http://localhost:8082/v1/embeddings \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-api-key "$EMBED_API_KEY" \
  search "project notes" -k 5
```

Global flags:
- `--embedding-endpoint`
- `--embedding-model`
- `--embedding-api-key`

## Reranker Options

Reranking is optional and applies during `search`.

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  --reranker-endpoint http://localhost:8082/v1/rerank \
  --reranker-model bge-reranker-v2-m3 \
  --reranker-api-key "$RERANK_API_KEY" \
  --rerank-candidates 100 \
  search "best notes about vector search" -k 5
```

Global flags:
- `--reranker-endpoint`
- `--reranker-model`
- `--reranker-api-key`
- `--rerank-candidates`

## Notes

- CLI ingest currently chunks text and stores one JSON record per chunk.
- Search output returns top documents with score, doc_id, and text snippet.
- CLI CRUD commands (`insert/get/update/delete`) are intended as a stable cross-language integration surface.
