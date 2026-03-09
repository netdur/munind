#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/benchmark/data/tinystories_subset.jsonl}"
DB_PATH="${DB_PATH:-${ROOT_DIR}/benchmark/tinystories_db}"
RESULT_FILE="${RESULT_FILE:-${ROOT_DIR}/benchmark/results/summary.json}"

LIMIT="${LIMIT:-10000}"
QUERIES="${QUERIES:-200}"
TOP_K="${TOP_K:-10}"
EF_SEARCH="${EF_SEARCH:-80}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1}"
EMBEDDING_MAX_RETRIES="${EMBEDDING_MAX_RETRIES:-8}"
EMBEDDING_RETRY_BASE_MS="${EMBEDDING_RETRY_BASE_MS:-250}"
EMBEDDING_RETRY_MAX_MS="${EMBEDDING_RETRY_MAX_MS:-3000}"
EMBEDDING_REQUEST_DELAY_MS="${EMBEDDING_REQUEST_DELAY_MS:-40}"
BENCH_WITH_QUALITY="${BENCH_WITH_QUALITY:-0}"

EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-http://localhost:8082/v1/embeddings}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text-v1.5}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
EMBEDDING_PROBE_TEXT="${EMBEDDING_PROBE_TEXT:-Munind benchmark query probe.}"
DIMENSION="${DIMENSION:-}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Input dataset not found: ${INPUT_FILE}" >&2
  echo "Run benchmark/01_build_dataset.sh first." >&2
  exit 1
fi
if [[ ! -d "${DB_PATH}" ]]; then
  echo "Database path not found: ${DB_PATH}" >&2
  echo "Run benchmark/02_create_database.sh first." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

PROFILE_CMD=(
  "${ROOT_DIR}/benchmark/detect_embedding_profile.sh"
  --endpoint "${EMBEDDING_ENDPOINT}"
  --model "${EMBEDDING_MODEL}"
  --probe-text "${EMBEDDING_PROBE_TEXT}"
  --output json
)
if [[ -n "${EMBEDDING_API_KEY}" ]]; then
  PROFILE_CMD+=(--api-key "${EMBEDDING_API_KEY}")
fi
PROFILE_JSON="$("${PROFILE_CMD[@]}")"

DIMENSION_SOURCE="provided"
if [[ -z "${DIMENSION}" ]]; then
  DIMENSION="$(jq -r '.dimension' <<<"${PROFILE_JSON}")"
  DIMENSION_SOURCE="auto-detected"
fi

STYLE_HINT="$(jq -r '.style_hint' <<<"${PROFILE_JSON}")"

echo "Benchmark phase"
if [[ "${BENCH_WITH_QUALITY}" == "1" ]]; then
  echo "  benchmark_mode: quality (exact baseline enabled)"
else
  echo "  benchmark_mode: latency-only (set BENCH_WITH_QUALITY=1 for exact quality metrics)"
fi
echo "  input: ${INPUT_FILE}"
echo "  db_path: ${DB_PATH}"
echo "  embedding_endpoint: ${EMBEDDING_ENDPOINT}"
echo "  embedding_model: ${EMBEDDING_MODEL}"
echo "  dimension: ${DIMENSION} (${DIMENSION_SOURCE}; override with DIMENSION=... )"
echo "  style_hint: ${STYLE_HINT}"

mkdir -p "$(dirname "${RESULT_FILE}")"

CMD=(
  cargo run --release -p munind-bench --
  --input "${INPUT_FILE}"
  --db-path "${DB_PATH}"
  --dimension "${DIMENSION}"
  --limit "${LIMIT}"
  --queries "${QUERIES}"
  --top-k "${TOP_K}"
  --ef-search "${EF_SEARCH}"
  --embedding-endpoint "${EMBEDDING_ENDPOINT}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --embedding-max-retries "${EMBEDDING_MAX_RETRIES}"
  --embedding-retry-base-ms "${EMBEDDING_RETRY_BASE_MS}"
  --embedding-retry-max-ms "${EMBEDDING_RETRY_MAX_MS}"
  --embedding-request-delay-ms "${EMBEDDING_REQUEST_DELAY_MS}"
  --use-existing-db
  --require-real-embeddings
  --output-json "${RESULT_FILE}"
)

if [[ "${BENCH_WITH_QUALITY}" != "1" ]]; then
  CMD+=(--latency-only)
fi

if [[ -n "${EMBEDDING_API_KEY}" ]]; then
  CMD+=(--embedding-api-key "${EMBEDDING_API_KEY}")
fi

"${CMD[@]}"

echo "Benchmark summary written to ${RESULT_FILE}"
