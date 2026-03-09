#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/benchmark/data/tinystories_subset.jsonl}"
LIMIT="${LIMIT:-10000}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

if [[ $# -gt 0 ]]; then
  python3 "${ROOT_DIR}/benchmark/download_tinystories_subset.py" "$@"
else
  python3 "${ROOT_DIR}/benchmark/download_tinystories_subset.py" \
    --output "${OUTPUT_FILE}" \
    --limit "${LIMIT}"
fi
