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

## Dataset

Default benchmark uses TinyStories subset JSONL:

```bash
python3 benchmark/download_tinystories_subset.py \
  --output benchmark/data/tinystories_subset.jsonl \
  --limit 1000
```

## Deterministic Embeddings Run

```bash
cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --dimension 512 \
  --limit 1000 \
  --queries 200 \
  --top-k 10 \
  --ef-search 80 \
  --output-json benchmark/results/summary.json
```

## Real Embeddings Run (Optional)

```bash
cargo run --release -p munind-bench -- \
  --input benchmark/data/tinystories_subset.jsonl \
  --dimension 512 \
  --limit 1000 \
  --queries 200 \
  --top-k 10 \
  --ef-search 80 \
  --embedding-endpoint http://localhost:8082/v1/embeddings \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-api-key "$EMBED_API_KEY" \
  --embedding-batch-size 64 \
  --output-json benchmark/results/summary_real_embed.json
```

## One-Command Runner

```bash
bash benchmark/run_benchmark.sh
```

Optional environment variables:
- `LIMIT`, `QUERIES`, `DIMENSION`, `TOP_K`, `EF_SEARCH`
- `EMBEDDING_ENDPOINT`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_BATCH_SIZE`

## Interpreting Results

- If latency is good but `recall@k` drops, ANN parameters are too aggressive.
- If filtered recall drops, check filter plan coverage/indexed fields.
- Compare deterministic vs real embeddings before drawing retrieval-quality conclusions.

## Latest Smoke Snapshot

From `benchmark/results/summary_quality_smoke.json`:
- Search `mean recall@10`: `0.9980`
- Filtered search `mean recall@10`: `1.0000`
- Search `mean nDCG@10`: `0.9922`
