#!/usr/bin/env python3
"""Benchmark munind SQLite extension: insert GloVe, measure recall@K and latency.

Uses sqlite3 CLI (Python's sqlite3 module lacks extension loading on macOS).

Usage:
    cargo build --release -p munind-sqlite
    python3 scripts/bench_sqlite_recall.py
"""

import os
import struct
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAIN_TSV = ROOT / "benches/data/glove-100-angular.train.tsv"
TEST_TSV = ROOT / "benches/data/glove-100-angular.test.tsv"
HDF5_PATH = ROOT / "benches/data/glove-100-angular.hdf5"
MUNIND_EXT = ROOT / "target/release/libmunind"
DB_PATH = ROOT / "benches/indexes/bench_sqlite.db"

DIM = 100
TOP_K = 10
EPSILONS = [0.1, 0.2, 0.4]


def load_ground_truth(k: int) -> np.ndarray:
    with h5py.File(HDF5_PATH, "r") as f:
        neighbors = np.array(f["neighbors"], dtype=np.int64)
    return neighbors[:, :k]


def vec_to_hex(vec: list[float]) -> str:
    return struct.pack(f"<{len(vec)}f", *vec).hex()


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    total = found.shape[0] * found.shape[1]
    hits = 0
    for i in range(found.shape[0]):
        hits += len(set(found[i].tolist()) & set(truth[i].tolist()))
    return hits / total


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_sqlite_index():
    """Insert all GloVe train vectors via sqlite3 CLI."""
    import shutil

    if DB_PATH.exists():
        os.remove(DB_PATH)
    idx_dir = Path(f"{DB_PATH}-munind-glove")
    if idx_dir.exists():
        shutil.rmtree(idx_dir)

    # Read train vectors
    train_vecs = []
    with open(TRAIN_TSV) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            train_vecs.append(vals[:DIM])
    n = len(train_vecs)
    print(f"  {n} vectors, dim={DIM}")

    # Generate SQL
    sql_file = DB_PATH.parent / "bench_insert.sql"
    with open(sql_file, "w") as f:
        f.write(f".load {MUNIND_EXT}\n")
        f.write("CREATE VIRTUAL TABLE glove USING munind(dim=100, metric=cosine);\n")
        f.write("BEGIN;\n")
        for i, vec in enumerate(train_vecs):
            hexblob = vec_to_hex(vec)
            f.write(f"INSERT INTO glove(rowid, vector) VALUES ({i+1}, X'{hexblob}');\n")
            if (i + 1) % 100000 == 0:
                f.write("COMMIT;\nBEGIN;\n")
        f.write("COMMIT;\n")

    t0 = time.time()
    result = subprocess.run(
        ["sqlite3", str(DB_PATH)],
        stdin=open(sql_file),
        capture_output=True, text=True, timeout=1200,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        raise RuntimeError("insert failed")

    print(f"  insert + build: {elapsed:.1f}s ({n/elapsed:.0f} vec/s)")
    sql_file.unlink()
    print()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def bench_sqlite_search():
    """Run test queries at each epsilon, measure recall and latency."""
    # Load test vectors
    test_vecs = []
    with open(TEST_TSV) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            test_vecs.append(vals[:DIM])
    n_queries = len(test_vecs)

    truth = load_ground_truth(TOP_K)

    print(f"  {n_queries} queries, recall@{TOP_K}")
    print()
    print(f"  {'epsilon':>9}  {'recall@10':>10}  {'avg_ms':>10}  {'qps':>8}")
    print(f"  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}")

    for epsilon in EPSILONS:
        sql_file = DB_PATH.parent / "bench_query.sql"
        with open(sql_file, "w") as f:
            f.write(f".load {MUNIND_EXT}\n")
            for vec in test_vecs:
                hexblob = vec_to_hex(vec)
                f.write(
                    f"SELECT rowid FROM glove "
                    f"WHERE vector MATCH X'{hexblob}' AND k = {TOP_K} AND epsilon = {epsilon};\n"
                )

        t0 = time.time()
        result = subprocess.run(
            ["sqlite3", str(DB_PATH)],
            stdin=open(sql_file),
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  ERROR at e={epsilon}: {result.stderr[:300]}")
            continue

        # Parse: each query produces TOP_K lines of rowid
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        found_all = []
        current = []
        for line in lines:
            try:
                rid = int(line)
                current.append(rid - 1)  # 1-based → 0-based
                if len(current) == TOP_K:
                    found_all.append(current)
                    current = []
            except ValueError:
                continue
        if current:
            while len(current) < TOP_K:
                current.append(-1)
            found_all.append(current)

        sql_file.unlink()

        avg_ms = elapsed / n_queries * 1000
        qps = n_queries / elapsed

        found_arr = np.array(found_all[:n_queries], dtype=np.int64)
        if found_arr.shape[0] < n_queries:
            pad = np.full((n_queries - found_arr.shape[0], TOP_K), -1, dtype=np.int64)
            found_arr = np.vstack([found_arr, pad])

        r = recall_at_k(found_arr, truth)
        print(f"  {epsilon:>9.1f}  {r:>10.4f}  {avg_ms:>10.3f}  {qps:>8.0f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ext_path = Path(str(MUNIND_EXT) + ".dylib")
    if not ext_path.exists():
        ext_path = Path(str(MUNIND_EXT) + ".so")
    if not ext_path.exists():
        raise SystemExit(
            f"Extension not found: {MUNIND_EXT}.*\n"
            f"Run: cargo build --release -p munind-sqlite"
        )
    if not TRAIN_TSV.exists():
        raise SystemExit(f"Train data not found: {TRAIN_TSV}")
    if not HDF5_PATH.exists():
        raise SystemExit(f"HDF5 ground truth not found: {HDF5_PATH}")

    print()
    print("  munind SQLite extension — GloVe-100 benchmark")
    print("  1.18M vectors, 10K queries, cosine metric")
    print()

    build_sqlite_index()
    bench_sqlite_search()


if __name__ == "__main__":
    main()
