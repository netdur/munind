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

## 1) Download TinyStories Subset
```bash
python3 benchmark/download_tinystories_subset.py \
  --output benchmark/data/tinystories_subset.jsonl \
  --limit 1000
```

## 2) Run Benchmark (Deterministic Embeddings)
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

## 3) Run Benchmark with Real Embeddings (Optional)
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

Environment overrides:
- `LIMIT` (default `1000`)
- `QUERIES` (default `200`)
- `DIMENSION` (default `512`)
- `TOP_K` (default `10`)
- `EF_SEARCH` (default `80`)

## Notes
- The downloader uses Hugging Face datasets server API and requires internet access.
- Docs are assigned two sources (`TinyStories`, `TinyStoriesAlt`) so filtered-recall metrics are meaningful.
- Exact quality baseline uses brute-force cosine ranking over the same embedded vectors.
