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
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-deterministic-v1}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"

mkdir -p "${ROOT_DIR}/benchmark/data" "${ROOT_DIR}/benchmark/results"

python3 "${ROOT_DIR}/benchmark/download_tinystories_subset.py" \
  --output "${DATA_FILE}" \
  --limit "${LIMIT}"

CMD=(
  cargo run --release -p munind-bench --
  --input "${DATA_FILE}"
  --dimension "${DIMENSION}"
  --limit "${LIMIT}"
  --queries "${QUERIES}"
  --top-k "${TOP_K}"
  --ef-search "${EF_SEARCH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --output-json "${RESULT_FILE}"
)

if [[ -n "${EMBEDDING_ENDPOINT}" ]]; then
  CMD+=(--embedding-endpoint "${EMBEDDING_ENDPOINT}")
fi

if [[ -n "${EMBEDDING_API_KEY}" ]]; then
  CMD+=(--embedding-api-key "${EMBEDDING_API_KEY}")
fi

"${CMD[@]}"

echo "Benchmark results written to ${RESULT_FILE}"
