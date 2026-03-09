#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_HDF5="${DATASET_HDF5:-${ROOT_DIR}/benchmark/data/glove-100-angular.hdf5}"
TRAIN_F32="${TRAIN_F32:-${ROOT_DIR}/benchmark/data/glove-100-angular.train.f32}"
TEST_F32="${TEST_F32:-${ROOT_DIR}/benchmark/data/glove-100-angular.test.f32}"
NEIGHBORS_I32="${NEIGHBORS_I32:-${ROOT_DIR}/benchmark/data/glove-100-angular.neighbors.i32}"

DB_PATH="${DB_PATH:-${ROOT_DIR}/benchmark/glove_100_db}"
RESULT_FILE="${RESULT_FILE:-${ROOT_DIR}/benchmark/results/glove_100_prepare_summary.json}"

DIMENSION="${DIMENSION:-100}"
TOP_K="${TOP_K:-10}"
GROUND_TRUTH_K="${GROUND_TRUTH_K:-100}"
EF_SEARCH="${EF_SEARCH:-80}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
FSYNC_ENABLED="${FSYNC_ENABLED:-0}"
EXPORT_RAW="${EXPORT_RAW:-1}"
COMPACT_AFTER_PREPARE="${COMPACT_AFTER_PREPARE:-0}"
COMPACT_REPAIR_GRAPH="${COMPACT_REPAIR_GRAPH:-0}"
CHECKPOINT_WAL_AFTER_PREPARE="${CHECKPOINT_WAL_AFTER_PREPARE:-1}"

echo "Create GloVe database phase (ANN-only)"
echo "  dataset_hdf5: ${DATASET_HDF5}"
echo "  train_f32: ${TRAIN_F32}"
echo "  test_f32: ${TEST_F32}"
echo "  neighbors_i32: ${NEIGHBORS_I32}"
echo "  db_path: ${DB_PATH}"
echo "  dimension: ${DIMENSION}"
echo "  train_limit: ${TRAIN_LIMIT} (0 means full dataset)"
echo "  fsync_enabled: ${FSYNC_ENABLED}"
if [[ "${COMPACT_AFTER_PREPARE}" == "1" ]]; then
  echo "  post_prepare_compaction: enabled (COMPACT_REPAIR_GRAPH=${COMPACT_REPAIR_GRAPH})"
else
  echo "  post_prepare_compaction: disabled (set COMPACT_AFTER_PREPARE=1 to enable)"
fi
if [[ "${CHECKPOINT_WAL_AFTER_PREPARE}" == "1" ]]; then
  echo "  checkpoint_wal_after_prepare: enabled"
else
  echo "  checkpoint_wal_after_prepare: disabled"
fi

if [[ "${EXPORT_RAW}" == "1" ]]; then
  if [[ ! -f "${DATASET_HDF5}" ]]; then
    echo "Dataset not found: ${DATASET_HDF5}" >&2
    exit 1
  fi
  python - "${DATASET_HDF5}" "${TRAIN_F32}" "${TEST_F32}" "${NEIGHBORS_I32}" <<'PY'
import h5py
import numpy as np
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_train = Path(sys.argv[2])
out_test = Path(sys.argv[3])
out_neighbors = Path(sys.argv[4])

with h5py.File(src, "r") as f:
    np.asarray(f["train"], dtype="<f4").tofile(out_train)
    np.asarray(f["test"], dtype="<f4").tofile(out_test)
    np.asarray(f["neighbors"], dtype="<i4").tofile(out_neighbors)

print(f"wrote {out_train}")
print(f"wrote {out_test}")
print(f"wrote {out_neighbors}")
PY
fi

if [[ ! -f "${TRAIN_F32}" || ! -f "${TEST_F32}" || ! -f "${NEIGHBORS_I32}" ]]; then
  echo "Missing raw matrix files. Set EXPORT_RAW=1 or provide TRAIN_F32/TEST_F32/NEIGHBORS_I32." >&2
  exit 1
fi

mkdir -p "$(dirname "${RESULT_FILE}")"

CMD=(
  cargo run --release -p munind-bench --bin glove_hdf5_bench --
  --train-f32 "${TRAIN_F32}"
  --test-f32 "${TEST_F32}"
  --neighbors-i32 "${NEIGHBORS_I32}"
  --db-path "${DB_PATH}"
  --dimension "${DIMENSION}"
  --top-k "${TOP_K}"
  --ground-truth-k "${GROUND_TRUTH_K}"
  --ef-search "${EF_SEARCH}"
  --train-limit "${TRAIN_LIMIT}"
  --prepare-only
  --output-json "${RESULT_FILE}"
)
if [[ "${CHECKPOINT_WAL_AFTER_PREPARE}" == "1" ]]; then
  CMD+=(--checkpoint-wal-after-prepare)
fi

if [[ "${FSYNC_ENABLED}" == "1" ]]; then
  CMD+=(--fsync-enabled)
fi

"${CMD[@]}"

if [[ "${COMPACT_AFTER_PREPARE}" == "1" ]]; then
  echo "Running optional full compaction (can be slow on full dataset)..."
  OPT_CMD=(
    cargo run --release -p munind-cli --
    --db "${DB_PATH}"
    optimize
  )
  if [[ "${COMPACT_REPAIR_GRAPH}" == "1" ]]; then
    OPT_CMD+=(--repair-graph)
  fi
  "${OPT_CMD[@]}"
fi

echo "GloVe prepare summary written to ${RESULT_FILE}"
