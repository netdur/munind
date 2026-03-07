# Munind Benchmark

This folder benchmarks Munind insert/search performance using a TinyStories subset.

## What It Measures
- Insert latency and throughput.
- Search latency and throughput.
- Filtered search latency with `source == TinyStories`.

## 1) Download TinyStories Subset
```bash
python3 benchmark/download_tinystories_subset.py \
  --output benchmark/data/tinystories_subset.jsonl \
  --limit 1000
```

## 2) Run Benchmark
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
- Bench embeddings are deterministic text-hash vectors (so benchmark isolates Munind DB behavior).
