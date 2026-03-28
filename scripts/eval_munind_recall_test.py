#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
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
EPSILONS = [0.2]


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
    debug: int,
) -> str:
    cmd = [
        str(munind_bin),
        "-X",
        str(debug),
        "search",
        "-n",
        str(k),
        "-e",
        str(epsilon),
        str(index_path),
        str(query_path),
    ]
    if debug > 0:
        process = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        stdout, _ = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout)
        return stdout

    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def parse_search_output(output: str, k: int) -> tuple[np.ndarray, float]:
    lines = output.splitlines()

    results = []
    current = []
    avg_ms = None

    avg_re = re.compile(r"Average Query Time=.*?,\s*([0-9.]+)\s*\(msec\)")
    rank_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9eE+.\-]+)\s*$")

    for line in lines:
        m_avg = avg_re.search(line)
        if m_avg:
            avg_ms = float(m_avg.group(1))
            continue

        m_rank = rank_re.match(line)
        if not m_rank:
            continue

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
    parsed -= 1

    return parsed, avg_ms if avg_ms is not None else float("nan")


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    if found.shape != truth.shape:
        raise ValueError(f"shape mismatch: found={found.shape}, truth={truth.shape}")

    total = found.shape[0] * found.shape[1]
    hits = 0

    for i in range(found.shape[0]):
        hits += len(set(found[i].tolist()) & set(truth[i].tolist()))

    return hits / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", dest="binary", type=Path, default=MUNIND_BIN)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--query", type=Path, default=QUERY_PATH)
    parser.add_argument("--truth", type=Path, default=HDF5_PATH)
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
    parser.add_argument("-e", "--epsilon", type=float, action="append", dest="epsilons")
    parser.add_argument("-X", "--debug", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.binary.exists():
        raise SystemExit(f"munind binary not found: {args.binary}")
    if not args.index.exists():
        raise SystemExit(f"Index not found: {args.index}")
    if not args.query.exists():
        raise SystemExit(f"Query file not found: {args.query}")
    if not args.truth.exists():
        raise SystemExit(f"HDF5 file not found: {args.truth}")

    truth = load_ground_truth(args.truth, args.top_k)
    print(f"Loaded ground truth: {truth.shape[0]} queries, top-{args.top_k}")
    print()

    epsilons = args.epsilons if args.epsilons else EPSILONS
    for epsilon in epsilons:
        started = time.perf_counter()
        output = run_munind_search(
            args.binary,
            args.index,
            args.query,
            args.top_k,
            epsilon,
            args.debug,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        found, avg_ms = parse_search_output(output, args.top_k)

        if found.shape[0] != truth.shape[0]:
            raise RuntimeError(
                f"query count mismatch: found={found.shape[0]}, truth={truth.shape[0]}"
            )

        recall = recall_at_k(found, truth)
        print(f"-e {epsilon}")
        print(f"  recall@{args.top_k}: {recall:.6f}")
        print(f"  avg_query_ms: {avg_ms:.6f}")
        if args.debug > 0:
            print(f"  end_to_end_wall_ms: {wall_ms:.3f}")
        print()


if __name__ == "__main__":
    main()
