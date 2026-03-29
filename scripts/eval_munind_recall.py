#!/usr/bin/env python3
"""Evaluate munind recall@K against ANN benchmark ground truth."""

import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MUNIND_BIN = ROOT / "target/release/munind"
INDEX_PATH = ROOT / "benches/indexes/glove-100-angular-munind"
QUERY_PATH = ROOT / "benches/data/glove-100-angular.test.tsv"
HDF5_PATH = ROOT / "benches/data/glove-100-angular.hdf5"

TOP_K = 10
#EPSILONS = [0.2]
EPSILONS = [0.1, 0.4]


def load_ground_truth(hdf5_path: Path, k: int) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        neighbors = np.array(f["neighbors"], dtype=np.int64)
    return neighbors[:, :k]


def run_munind_search(
    munind_bin: Path,
    index_path: Path,
    query_path: Path,
    k: int,
    epsilon: float,
) -> str:
    cmd = [
        str(munind_bin),
        "search",
        "-n",
        str(k),
        "-e",
        str(epsilon),
        str(index_path),
        str(query_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout, result.stderr


def parse_munind_output(stdout: str, stderr: str, k: int) -> tuple[np.ndarray, float]:
    lines = stdout.splitlines()

    results = []
    current = []

    # munind output format:
    #   Query No.1
    #   1	4	0.1414214
    #   2	2	0.9055385
    rank_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9eE+.\-]+)\s*$")

    for line in lines:
        m_rank = rank_re.match(line)
        if m_rank:
            rank = int(m_rank.group(1))
            idx = int(m_rank.group(2))
            if rank == 1 and current:
                results.append(current)
                current = []
            current.append(idx)
            if len(current) == k:
                results.append(current)
                current = []

    if current:
        results.append(current)

    if not results:
        raise RuntimeError("Failed to parse any munind search results.")

    parsed = np.array(results, dtype=np.int64)

    # munind IDs are 1-based. Convert to 0-based for ANN benchmark ground truth.
    parsed -= 1

    # Extract average query time from stderr.
    avg_ms = float("nan")
    avg_re = re.compile(r"Average query time:\s*([0-9.]+)\s*ms")
    for line in stderr.splitlines():
        m = avg_re.search(line)
        if m:
            avg_ms = float(m.group(1))

    return parsed, avg_ms


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    if found.shape != truth.shape:
        raise ValueError(f"shape mismatch: found={found.shape}, truth={truth.shape}")

    total = found.shape[0] * found.shape[1]
    hits = 0

    for i in range(found.shape[0]):
        hits += len(set(found[i].tolist()) & set(truth[i].tolist()))

    return hits / total


def main() -> None:
    if not MUNIND_BIN.exists():
        raise SystemExit(
            f"munind binary not found: {MUNIND_BIN}\n"
            f"Run: cargo build --release"
        )
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Index not found: {INDEX_PATH}\n"
            f"Run: target/release/munind create -d 100 -D c {INDEX_PATH} benches/data/glove-100-angular.train.tsv"
        )
    if not QUERY_PATH.exists():
        raise SystemExit(f"Query file not found: {QUERY_PATH}")
    if not HDF5_PATH.exists():
        raise SystemExit(f"HDF5 file not found: {HDF5_PATH}")

    truth = load_ground_truth(HDF5_PATH, TOP_K)

    print(f"Loaded ground truth: {truth.shape[0]} queries, top-{TOP_K}")
    print()

    for epsilon in EPSILONS:
        stdout, stderr = run_munind_search(
            MUNIND_BIN, INDEX_PATH, QUERY_PATH, TOP_K, epsilon
        )
        found, avg_ms = parse_munind_output(stdout, stderr, TOP_K)

        if found.shape[0] != truth.shape[0]:
            raise RuntimeError(
                f"query count mismatch: found={found.shape[0]}, truth={truth.shape[0]}"
            )

        r = recall_at_k(found, truth)
        print(f"-e {epsilon}")
        print(f"  recall@{TOP_K}: {r:.6f}")
        print(f"  avg_query_ms: {avg_ms:.6f}")
        print()


if __name__ == "__main__":
    main()
