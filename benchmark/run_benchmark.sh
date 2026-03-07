#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="${ROOT_DIR}/benchmark/data/tinystories_subset.jsonl"
RESULT_FILE="${ROOT_DIR}/benchmark/results/summary.json"

LIMIT="${LIMIT:-1000}"
QUERIES="${QUERIES:-200}"
DIMENSION="${DIMENSION:-512}"
TOP_K="${TOP_K:-10}"
EF_SEARCH="${EF_SEARCH:-80}"

mkdir -p "${ROOT_DIR}/benchmark/data" "${ROOT_DIR}/benchmark/results"

python3 "${ROOT_DIR}/benchmark/download_tinystories_subset.py" \
  --output "${DATA_FILE}" \
  --limit "${LIMIT}"

cargo run --release -p munind-bench -- \
  --input "${DATA_FILE}" \
  --dimension "${DIMENSION}" \
  --limit "${LIMIT}" \
  --queries "${QUERIES}" \
  --top-k "${TOP_K}" \
  --ef-search "${EF_SEARCH}" \
  --output-json "${RESULT_FILE}"

echo "Benchmark results written to ${RESULT_FILE}"
