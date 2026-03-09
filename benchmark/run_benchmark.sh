#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/benchmark/01_build_dataset.sh"
bash "${ROOT_DIR}/benchmark/02_create_database.sh"
bash "${ROOT_DIR}/benchmark/03_run_benchmark.sh"
