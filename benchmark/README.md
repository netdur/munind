# Munind Benchmark

This folder benchmarks Munind insert/search performance and retrieval quality on a TinyStories subset.

## What It Measures
- Insert latency and throughput.
- Search latency and throughput.
- Filtered search latency (exact filter on `source`).
- Search quality versus exact baseline:
  - `recall@k` (mean/p50/p95)
  - `mrr@k`
  - `ndcg@k`
- Filtered search quality versus exact filtered baseline.

## Real Embeddings Workflow (3 Phases)

### Prerequisites
- OpenAI-compatible embedding endpoint (`POST /v1/embeddings`)
- `curl` and `jq`

Defaults are already in scripts:
- endpoint: `http://localhost:8082/v1/embeddings`
- model: `nomic-embed-text-v1.5`

Set `EMBEDDING_API_KEY` only if your endpoint requires auth.

### 1) Build dataset

```bash
bash benchmark/01_build_dataset.sh
```

Overrides:
- `LIMIT` (default `10000`)
- `OUTPUT_FILE` (default `benchmark/data/tinystories_subset.jsonl`)

### 2) Create database (insert phase)

```bash
bash benchmark/02_create_database.sh
```

What happens:
- probes the embedding endpoint to detect dimension + style hint,
- recreates DB and inserts rows,
- optionally runs `munind-cli optimize` to write checkpoint + truncate WAL (enabled by default),
- writes `benchmark/results/prepare_summary.json`.

Overrides:
- `INPUT_FILE`, `DB_PATH`, `RESULT_FILE`
- `LIMIT`, `EF_SEARCH`, `EMBEDDING_BATCH_SIZE` (default `1`, one input per request), `EMBEDDING_MAX_RETRIES`, `EMBEDDING_RETRY_BASE_MS`, `EMBEDDING_RETRY_MAX_MS`, `EMBEDDING_REQUEST_DELAY_MS`
- `COMPACT_AFTER_PREPARE` (default `1`; set `0` to skip post-prepare compaction)
- `COMPACT_REPAIR_GRAPH` (default `0`; set `1` to also rebuild graph/index during optimize)
- `DIMENSION` (optional; auto-detected when unset)

### 3) Benchmark existing database (search phase)

```bash
bash benchmark/03_run_benchmark.sh
```

To include exact quality metrics:

```bash
BENCH_WITH_QUALITY=1 bash benchmark/03_run_benchmark.sh
```

What happens:
- reuses existing DB (`--use-existing-db`),
- runs search + filtered search latency benchmark by default,
- keeps insert metrics at `0` in this phase (no new rows are inserted),
- reads exact-baseline vectors directly from DB when `BENCH_WITH_QUALITY=1` (does not re-embed all docs from input),
- writes `benchmark/results/summary.json`.

Overrides:
- `INPUT_FILE`, `DB_PATH`, `RESULT_FILE`
- `LIMIT`, `QUERIES`, `TOP_K`, `EF_SEARCH`, `EMBEDDING_BATCH_SIZE` (default `1`, one input per request), `EMBEDDING_MAX_RETRIES`, `EMBEDDING_RETRY_BASE_MS`, `EMBEDDING_RETRY_MAX_MS`, `EMBEDDING_REQUEST_DELAY_MS`
- `BENCH_WITH_QUALITY` (`0` default = latency-only, `1` = include exact quality metrics)
- `DIMENSION` (optional; auto-detected when unset)

## Latest Quality Run Snapshot (2026-03-08)

Command:

```bash
BENCH_WITH_QUALITY=1 bash benchmark/03_run_benchmark.sh
```

Result highlights (`LIMIT=10000`, `QUERIES=200`, `TOP_K=10`, `EF_SEARCH=80`):
- Search: `812.29 ops/s`, `p50 1.234 ms`, `p95 1.543 ms`, `p99 1.641 ms`
- Filtered search (`source == TinyStories`): `191.24 ops/s`, `p50 5.207 ms`, `p95 5.561 ms`, `p99 5.622 ms`
- Search quality vs exact: `recall@k mean 0.9955`, `mrr@k 1.0000`, `ndcg@k 0.9989`
- Filtered quality vs exact: `recall@k mean 1.0000`, `mrr@k 1.0000`, `ndcg@k 1.0000`

Raw numbers are written to `benchmark/results/summary.json` for each run.

## One-Command Runner

```bash
bash benchmark/run_benchmark.sh
```

This runs all three phases in order.

## GloVe-100 (ANN-Only) Workflow

For ANN-Benchmarks style GloVe data (`train/test/neighbors`), use:

```bash
bash benchmark/02_create_glove_database.sh
bash benchmark/03_run_glove_benchmark.sh
```

One command:

```bash
bash benchmark/run_glove_benchmark.sh
```

Notes:
- This workflow is ANN-only (vector search only, no lexical/hybrid/filter metrics).
- `03_run_glove_benchmark.sh` is the "search existing DB" phase analogous to `03_run_benchmark.sh`.
- `03_run_glove_benchmark.sh` is search-only by design (`--search-existing-db-only --use-existing-db`); it cannot rebuild/reinsert DB data.
- `02_create_glove_database.sh` supports `EXPORT_RAW=1` (default) to convert `glove-100-angular.hdf5` into raw matrix files first.
- `02_create_glove_database.sh` defaults to `COMPACT_AFTER_PREPARE=0` (full-dataset compaction can be very slow).
- `02_create_glove_database.sh` defaults to `CHECKPOINT_WAL_AFTER_PREPARE=1` for fast WAL checkpoint/truncation after inserts.
- Phase 2 now also writes an ANN snapshot at `index/ann-index.snapshot` so later opens avoid full ANN graph rebuild.
- Phase 3 opens DB in ANN-only mode and skips lexical/payload index rebuild.
- `03_run_glove_benchmark.sh` defaults to `REQUIRE_FULL_DATASET=1` and fails if DB rows < full train rows.

Full-dataset commands:

```bash
# Build full DB once (train_limit=0 means full train matrix)
TRAIN_LIMIT=0 EXPORT_RAW=0 bash benchmark/02_create_glove_database.sh

# Benchmark existing full DB only
QUERIES=10000 bash benchmark/03_run_glove_benchmark.sh
```

Latest full-dataset ANN snapshot (2026-03-08):

Command:

```bash
EF_SEARCH=400 QUERIES=10000 bash benchmark/03_run_glove_benchmark.sh
```

Result highlights (`train_vectors=1183514`, `top_k=10`, cosine metric):
- Search: `238.38 ops/s`, `p50 4.255 ms`, `p95 5.624 ms`, `p99 6.406 ms`, `mean 4.195 ms`
- Quality: `recall@10 mean 0.9100`, `p50 1.0000`, `p95 1.0000`

## Optional: inspect embedding profile directly

```bash
bash benchmark/detect_embedding_profile.sh \
  --endpoint "$EMBEDDING_ENDPOINT" \
  --model "$EMBEDDING_MODEL" \
  --output summary
```

## Notes
- The downloader uses Hugging Face datasets server API and requires internet access.
- Docs are assigned two sources (`TinyStories`, `TinyStoriesAlt`) so filtered-recall metrics are meaningful.
- Quality baseline uses brute-force cosine ranking over embedded vectors keyed by `row_idx`.
