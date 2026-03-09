# Benchmarking

Munind benchmark covers both speed and retrieval quality.

## Metrics

Latency/throughput:
- Insert: ops/s and latency percentiles
- Search: ops/s and latency percentiles
- Filtered search: ops/s and latency percentiles

Quality (against exact brute-force baseline):
- `recall@k` (mean, p50, p95)
- `mrr@k`
- `ndcg@k`
- same metrics for filtered search

## 3-Phase Benchmark Flow (Real Embeddings)

Defaults are already in scripts:
- endpoint: `http://localhost:8082/v1/embeddings`
- model: `nomic-embed-text-v1.5`
- dataset size: `LIMIT=10000`
- db path: `benchmark/tinystories_db`

Set `EMBEDDING_API_KEY` only if your endpoint requires auth.

### Phase 1: Build TinyStories subset

```bash
bash benchmark/01_build_dataset.sh
```

### Phase 2: Create database + insert

```bash
bash benchmark/02_create_database.sh
```

- Runs `munind-bench --prepare-only --require-real-embeddings`.
- Auto-detects embedding dimension from the endpoint unless `DIMENSION` is set.
- Optionally runs post-prepare compaction via `munind-cli optimize` (enabled by default in script).
- Writes `benchmark/results/prepare_summary.json`.

Phase 2 compaction toggles:
- `COMPACT_AFTER_PREPARE=1` (default) truncates WAL after checkpoint.
- `COMPACT_AFTER_PREPARE=0` skips compaction.
- `COMPACT_REPAIR_GRAPH=1` also rebuilds graph/index during optimize.

### Phase 3: Benchmark existing database

```bash
bash benchmark/03_run_benchmark.sh
```

- Runs `munind-bench --use-existing-db --require-real-embeddings --latency-only` by default.
- Reuses DB from phase 2.
- Insert metrics are expected to stay `0` in this phase (no inserts are performed).
- Set `BENCH_WITH_QUALITY=1` to include exact quality metrics.
- In quality mode, exact baseline vectors are read from DB (not re-embedded from input).
- Writes `benchmark/results/summary.json`.

Quality mode example:

```bash
BENCH_WITH_QUALITY=1 bash benchmark/03_run_benchmark.sh
```

## Latest Baseline Snapshot (2026-03-08)

From:

```bash
BENCH_WITH_QUALITY=1 bash benchmark/03_run_benchmark.sh
```

With defaults (`LIMIT=10000`, `QUERIES=200`, `TOP_K=10`, `EF_SEARCH=80`, model `nomic-embed-text-v1.5`):

- Search: `812.29 ops/s`, `p50 1.234 ms`, `p95 1.543 ms`, `p99 1.641 ms`
- Filtered search (`source == TinyStories`): `191.24 ops/s`, `p50 5.207 ms`, `p95 5.561 ms`, `p99 5.622 ms`
- Search quality vs exact: `recall@k mean 0.9955`, `mrr@k 1.0000`, `ndcg@k 0.9989`
- Filtered quality vs exact: `recall@k mean 1.0000`, `mrr@k 1.0000`, `ndcg@k 1.0000`

These values are hardware/provider dependent; keep `benchmark/results/summary.json` as the source of truth for each run.

## One-Command Runner

```bash
bash benchmark/run_benchmark.sh
```

## GloVe-100 ANN-Only Benchmark

Use this path for ANN-Benchmarks style data (`train/test/neighbors`) where you want pure vector ANN metrics.

Create DB:

```bash
bash benchmark/02_create_glove_database.sh
```

Search benchmark on existing DB (03-style phase):

```bash
bash benchmark/03_run_glove_benchmark.sh
```

One command:

```bash
bash benchmark/run_glove_benchmark.sh
```

This workflow runs ANN-only search (no lexical/hybrid/filter metrics).
- `benchmark/02_create_glove_database.sh` defaults to `COMPACT_AFTER_PREPARE=0` for faster full builds.
- `benchmark/02_create_glove_database.sh` defaults to `CHECKPOINT_WAL_AFTER_PREPARE=1` to avoid heavy WAL replay on phase 3 startup.
- `benchmark/02_create_glove_database.sh` writes an ANN snapshot (`index/ann-index.snapshot`) during prepare/checkpoint so later opens do not rebuild HNSW from scratch.
- `benchmark/03_run_glove_benchmark.sh` is enforced search-only (`--search-existing-db-only --use-existing-db`).
- `benchmark/03_run_glove_benchmark.sh` opens in ANN-only mode and skips lexical/payload index rebuild.
- `benchmark/03_run_glove_benchmark.sh` defaults to `REQUIRE_FULL_DATASET=1` and rejects subset DBs.

Full-dataset commands:

```bash
# Build full train DB once
TRAIN_LIMIT=0 EXPORT_RAW=0 bash benchmark/02_create_glove_database.sh

# Run ANN benchmark against full existing DB
QUERIES=10000 bash benchmark/03_run_glove_benchmark.sh
```

Latest GloVe full-dataset snapshot (2026-03-08):

Command:

```bash
EF_SEARCH=400 QUERIES=10000 bash benchmark/03_run_glove_benchmark.sh
```

Observed (`train_vectors=1183514`, `top_k=10`, cosine ANN):
- Search: `238.38 ops/s`, `p50 4.255 ms`, `p95 5.624 ms`, `p99 6.406 ms`, `mean 4.195 ms`
- Recall@10: `mean 0.9100`, `p50 1.0000`, `p95 1.0000` (10,000 queries)

## Direct `munind-bench` Modes

Prepare-only:

```bash
cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --db-path benchmark/tinystories_db \
  --dimension 768 \
  --limit 10000 \
  --embedding-endpoint http://localhost:8082/v1/embeddings \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-batch-size 1 \
  --embedding-max-retries 8 \
  --embedding-retry-base-ms 250 \
  --embedding-retry-max-ms 3000 \
  --embedding-request-delay-ms 40 \
  --prepare-only \
  --require-real-embeddings
```

Search-only latency benchmark on existing DB:

```bash
cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --db-path benchmark/tinystories_db \
  --dimension 768 \
  --limit 10000 \
  --queries 200 \
  --top-k 10 \
  --ef-search 80 \
  --embedding-endpoint http://localhost:8082/v1/embeddings \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-batch-size 1 \
  --embedding-max-retries 8 \
  --embedding-retry-base-ms 250 \
  --embedding-retry-max-ms 3000 \
  --embedding-request-delay-ms 40 \
  --use-existing-db \
  --latency-only \
  --require-real-embeddings \
  --output-json benchmark/results/summary.json
```

Search + quality benchmark on existing DB:

```bash
cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --db-path benchmark/tinystories_db \
  --dimension 768 \
  --limit 10000 \
  --queries 200 \
  --top-k 10 \
  --ef-search 80 \
  --embedding-endpoint http://localhost:8082/v1/embeddings \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-batch-size 1 \
  --embedding-max-retries 8 \
  --embedding-retry-base-ms 250 \
  --embedding-retry-max-ms 3000 \
  --embedding-request-delay-ms 40 \
  --use-existing-db \
  --require-real-embeddings \
  --output-json benchmark/results/summary.json
```

## Interpreting Results

- If latency is good but `recall@k` drops, ANN parameters are too aggressive.
- If filtered recall drops, check filter plan coverage/indexed fields.
- If style hint shows non-normalized vectors, tune distance metric/index settings accordingly.
