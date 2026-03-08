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

## Search

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  search "what did I write about habits" -k 5
```

## Health Check

```bash
cargo run -p munind-cli -- \
  --db ./munind_data \
  check-health
```

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
