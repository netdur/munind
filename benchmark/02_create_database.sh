#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/benchmark/data/tinystories_subset.jsonl}"
DB_PATH="${DB_PATH:-${ROOT_DIR}/benchmark/tinystories_db}"
RESULT_FILE="${RESULT_FILE:-${ROOT_DIR}/benchmark/results/prepare_summary.json}"

LIMIT="${LIMIT:-10000}"
EF_SEARCH="${EF_SEARCH:-80}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1}"
EMBEDDING_MAX_RETRIES="${EMBEDDING_MAX_RETRIES:-8}"
EMBEDDING_RETRY_BASE_MS="${EMBEDDING_RETRY_BASE_MS:-250}"
EMBEDDING_RETRY_MAX_MS="${EMBEDDING_RETRY_MAX_MS:-3000}"
EMBEDDING_REQUEST_DELAY_MS="${EMBEDDING_REQUEST_DELAY_MS:-40}"
COMPACT_AFTER_PREPARE="${COMPACT_AFTER_PREPARE:-1}"
COMPACT_REPAIR_GRAPH="${COMPACT_REPAIR_GRAPH:-0}"

EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-http://localhost:8082/v1/embeddings}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text-v1.5}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
EMBEDDING_PROBE_TEXT="${EMBEDDING_PROBE_TEXT:-Munind benchmark dimension probe.}"
DIMENSION="${DIMENSION:-}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Input dataset not found: ${INPUT_FILE}" >&2
  echo "Run benchmark/01_build_dataset.sh first." >&2
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

echo "Create database phase"
echo "  input: ${INPUT_FILE}"
echo "  db_path: ${DB_PATH}"
echo "  embedding_endpoint: ${EMBEDDING_ENDPOINT}"
echo "  embedding_model: ${EMBEDDING_MODEL}"
echo "  dimension: ${DIMENSION} (${DIMENSION_SOURCE}; override with DIMENSION=... )"
echo "  style_hint: ${STYLE_HINT}"
if [[ "${COMPACT_AFTER_PREPARE}" == "1" ]]; then
  echo "  post_prepare_compaction: enabled (COMPACT_REPAIR_GRAPH=${COMPACT_REPAIR_GRAPH})"
else
  echo "  post_prepare_compaction: disabled (set COMPACT_AFTER_PREPARE=1 to enable)"
fi

mkdir -p "$(dirname "${RESULT_FILE}")"

CMD=(
  cargo run --release -p munind-bench --
  --input "${INPUT_FILE}"
  --db-path "${DB_PATH}"
  --dimension "${DIMENSION}"
  --limit "${LIMIT}"
  --queries 0
  --top-k 10
  --ef-search "${EF_SEARCH}"
  --embedding-endpoint "${EMBEDDING_ENDPOINT}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --embedding-max-retries "${EMBEDDING_MAX_RETRIES}"
  --embedding-retry-base-ms "${EMBEDDING_RETRY_BASE_MS}"
  --embedding-retry-max-ms "${EMBEDDING_RETRY_MAX_MS}"
  --embedding-request-delay-ms "${EMBEDDING_REQUEST_DELAY_MS}"
  --prepare-only
  --require-real-embeddings
  --output-json "${RESULT_FILE}"
)

if [[ -n "${EMBEDDING_API_KEY}" ]]; then
  CMD+=(--embedding-api-key "${EMBEDDING_API_KEY}")
fi

"${CMD[@]}"

wal_size_bytes() {
  local wal_path="${DB_PATH}/wal/000001.wal"
  if [[ ! -f "${wal_path}" ]]; then
    echo 0
    return
  fi
  if stat -f "%z" "${wal_path}" >/dev/null 2>&1; then
    stat -f "%z" "${wal_path}"
  else
    stat -c "%s" "${wal_path}"
  fi
}

if [[ "${COMPACT_AFTER_PREPARE}" == "1" ]]; then
  wal_before="$(wal_size_bytes)"

  echo "Running post-prepare optimize (WAL checkpoint/truncate)..."
  OPT_CMD=(
    cargo run --release -p munind-cli --
    --db "${DB_PATH}"
    optimize
  )
  if [[ "${COMPACT_REPAIR_GRAPH}" == "1" ]]; then
    OPT_CMD+=(--repair-graph)
  fi
  "${OPT_CMD[@]}"

  wal_after="$(wal_size_bytes)"
  echo "  wal_bytes_before: ${wal_before}"
  echo "  wal_bytes_after:  ${wal_after}"
fi

echo "Prepare summary written to ${RESULT_FILE}"
