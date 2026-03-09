#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_F32="${TRAIN_F32:-${ROOT_DIR}/benchmark/data/glove-100-angular.train.f32}"
TEST_F32="${TEST_F32:-${ROOT_DIR}/benchmark/data/glove-100-angular.test.f32}"
NEIGHBORS_I32="${NEIGHBORS_I32:-${ROOT_DIR}/benchmark/data/glove-100-angular.neighbors.i32}"

DB_PATH="${DB_PATH:-${ROOT_DIR}/benchmark/glove_100_db}"
RESULT_FILE="${RESULT_FILE:-${ROOT_DIR}/benchmark/results/glove_100_summary.json}"

DIMENSION="${DIMENSION:-100}"
TOP_K="${TOP_K:-10}"
GROUND_TRUTH_K="${GROUND_TRUTH_K:-100}"
EF_SEARCH="${EF_SEARCH:-80}"
QUERIES="${QUERIES:-10000}"
REQUIRE_FULL_DATASET="${REQUIRE_FULL_DATASET:-1}"

if [[ ! -d "${DB_PATH}" ]]; then
  echo "Database path not found: ${DB_PATH}" >&2
  echo "Run benchmark/02_create_glove_database.sh first." >&2
  exit 1
fi
if [[ ! -f "${TEST_F32}" || ! -f "${NEIGHBORS_I32}" ]]; then
  echo "Missing test/neighbors raw files. Run benchmark/02_create_glove_database.sh with EXPORT_RAW=1." >&2
  exit 1
fi

echo "GloVe ANN-only benchmark phase"
echo "  benchmark_type: ann_only"
echo "  db_path: ${DB_PATH}"
echo "  test_f32: ${TEST_F32}"
echo "  neighbors_i32: ${NEIGHBORS_I32}"
echo "  dimension: ${DIMENSION}"
echo "  queries: ${QUERIES}"
echo "  top_k: ${TOP_K}"
echo "  ground_truth_k: ${GROUND_TRUTH_K}"
echo "  ef_search: ${EF_SEARCH}"

mkdir -p "$(dirname "${RESULT_FILE}")"

CMD=(
  env MUNIND_OPEN_PROGRESS=1
  cargo run --release -p munind-bench --bin glove_hdf5_bench --
  --train-f32 "${TRAIN_F32}"
  --test-f32 "${TEST_F32}"
  --neighbors-i32 "${NEIGHBORS_I32}"
  --db-path "${DB_PATH}"
  --dimension "${DIMENSION}"
  --top-k "${TOP_K}"
  --ground-truth-k "${GROUND_TRUTH_K}"
  --ef-search "${EF_SEARCH}"
  --queries "${QUERIES}"
  --search-existing-db-only
  --use-existing-db
  --output-json "${RESULT_FILE}"
)

if [[ "${REQUIRE_FULL_DATASET}" == "1" ]]; then
  CMD+=(--require-full-dataset)
fi

"${CMD[@]}"

echo "GloVe benchmark summary written to ${RESULT_FILE}"
