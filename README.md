# munind

Fast approximate nearest neighbor search in Rust. Use it as a library, a C FFI, a CLI tool, or a SQLite extension.

Graph+tree ANN engine (ANNG + DVPTree) with NEON SIMD on Apple Silicon.

```sql
-- as a SQLite extension
.load ./libmunind
CREATE VIRTUAL TABLE movies USING munind(dim=384, metric=cosine);
INSERT INTO movies(rowid, vector) VALUES (1, ?embedding);
SELECT rowid, distance FROM movies
WHERE vector MATCH ?query AND k = 10 AND epsilon = 0.2;
```

```rust
// as a Rust library
let mut index = Index::create(IndexConfig::new(384, Distance::Cosine)).unwrap();
index.insert(&embedding).unwrap();
index.build().unwrap();
let results = index.search(&query, 10).unwrap();
```

## Install

```bash
cargo build --release -p munind-sqlite
# produces: target/release/libmunind.dylib (macOS) or libmunind.so (Linux)
```

Load in any SQLite client:

```sql
.load ./target/release/libmunind
```

## Usage

### Create a vector table

```sql
CREATE VIRTUAL TABLE embeddings USING munind(dim=384, metric=cosine);
```

Supported metrics: `l2`, `cosine`, `l1`, `ip` (inner product), `angle`.

### Insert vectors

```sql
INSERT INTO embeddings(rowid, vector)
VALUES (1, munind_vector('[0.1, 0.2, 0.3, ...]'));
```

Vectors can be passed as:
- Raw BLOBs (little-endian f32 array) — fastest
- JSON text arrays via `munind_vector()` — convenient

### Search

```sql
-- basic KNN search
SELECT rowid, distance FROM embeddings
WHERE vector MATCH ?query_blob AND k = 10;

-- tune recall/speed tradeoff with epsilon
SELECT rowid, distance FROM embeddings
WHERE vector MATCH ?query_blob AND k = 10 AND epsilon = 0.4;
```

Higher epsilon = better recall, slower search. Default is 0.2.

### Join with metadata

The vector table handles vectors. Metadata lives in regular SQLite tables — use the full power of SQL:

```sql
CREATE TABLE photo_meta (
    id INTEGER PRIMARY KEY,
    date TEXT,
    location TEXT,
    tags TEXT
);

-- similar photos in June from Morocco
SELECT v.rowid, v.distance, m.location
FROM embeddings v
JOIN photo_meta m ON m.id = v.rowid
WHERE v.vector MATCH ?query AND v.k = 20 AND v.epsilon = 0.2
  AND m.date BETWEEN '2025-06-01' AND '2025-06-30'
  AND m.location IN ('fes', 'rabat', 'marrakech')
LIMIT 10;
```

### Helper functions

```sql
SELECT munind_vector('[1.0, 2.0, 3.0]');              -- JSON array -> BLOB
SELECT munind_vector_json(vector_blob);                -- BLOB -> JSON array
SELECT munind_distance(blob1, blob2, 'cosine');        -- compute distance
SELECT munind_version();                               -- extension version
```

### Delete

```sql
DELETE FROM embeddings WHERE rowid = 42;
```

### Persistence

The index saves automatically when the connection closes. Files are stored adjacent to the database:

```
mydb.db
mydb.db-munind-embeddings/
    prf    # properties
    obj    # vectors
    grp    # graph
    tre    # tree
    rowmap # rowid mapping
```

## Benchmarks

GloVe-100-angular, 1.18M vectors, dim=100, cosine metric, Apple M-series.

### The epsilon tradeoff

`epsilon` controls how aggressively the graph search explores. Higher epsilon = more nodes visited = better recall but slower. Pick the right one for your use case:

| epsilon | recall@10 | p50 latency | qps (1 thread) | Use case |
|---------|-----------|-------------|-----------------|----------|
| 0.1 | 63.5% | 137 us | 6,326 | Real-time serving, speed-critical |
| **0.2 (default)** | **84.7%** | **392 us** | **1,743** | **Good balance for most applications** |
| 0.4 | 98.7% | 3.5 ms | 104 | High-recall, accuracy-critical |

```sql
-- default (epsilon=0.2, good balance)
SELECT rowid, distance FROM vecs WHERE vector MATCH ?q AND k = 10;

-- fast, lower recall
SELECT rowid, distance FROM vecs WHERE vector MATCH ?q AND k = 10 AND epsilon = 0.1;

-- thorough, near-exact
SELECT rowid, distance FROM vecs WHERE vector MATCH ?q AND k = 10 AND epsilon = 0.4;
```

### Full latency distribution (single-thread, warm cache)

| epsilon | recall@10 | avg | p50 | p95 | p99 | qps |
|---------|-----------|-----|-----|-----|-----|-----|
| 0.1 | 0.635 | 158 us | 137 us | 325 us | 454 us | 6,326 |
| 0.2 | 0.847 | 574 us | 392 us | 1.6 ms | 2.3 ms | 1,743 |
| 0.4 | 0.987 | 9.6 ms | 3.5 ms | 37 ms | 50 ms | 104 |

### Multi-thread (10 threads)

| epsilon | recall@10 | qps |
|---------|-----------|-----|
| 0.1 | 0.635 | 39,493 |
| 0.2 | 0.847 | 11,907 |
| 0.4 | 0.987 | 733 |

### Build

| Phase | Time | Rate |
|-------|------|------|
| Insert 1.18M vectors | 0.2s | 5M vec/s |
| Build graph | 55s | 21K vec/s |
| Save to disk | 0.3s | |
| Open from disk (cold) | 250 ms | |

### Memory / disk

| Component | Size |
|-----------|------|
| Vectors | 452.6 MB |
| Graph | 155.5 MB |
| Tree | 23.3 MB |
| **Total** | **631 MB** |

## Rust API

munind-core can be used directly as a Rust library:

```rust
use munind_core::api::{Index, IndexConfig, Distance};

let config = IndexConfig::new(128, Distance::Cosine);
let mut index = Index::create(config).unwrap();

index.insert(&vec![0.1; 128]).unwrap();
index.insert(&vec![0.2; 128]).unwrap();
index.build().unwrap();

let results = index.search(&vec![0.15; 128], 10).unwrap();
println!("nearest: id={}, distance={}", results[0].id, results[0].distance);

index.save("my_index").unwrap();
```

## CLI

```bash
cargo build --release -p munind-core

# create index from TSV data
munind-core create -d 100 -D c index_dir data.tsv

# search
munind-core search -n 10 -e 0.1 index_dir queries.tsv

# memory-mapped search (zero-copy vectors)
munind-core search-mmap -n 10 -e 0.1 index_dir queries.tsv
```

## C FFI

```c
#include "munind.h"

MunindIndex* idx = munind_create(128, "cosine");
float vec[] = {0.1, 0.2, ...};
munind_insert(idx, vec, 128);
munind_build(idx);

MunindResult results[10];
uint32_t count;
munind_search(idx, query, 128, 10, 0.1, results, &count);

munind_save(idx, "my_index");
munind_free(idx);
```

## Project structure

```
munind/
  munind-core/    # ANN library (Rust)
  munind-sqlite/  # SQLite loadable extension
```

## Distance metrics

| SQL name | Description |
|----------|-------------|
| `cosine` | Cosine similarity (pre-normalizes on insert) |
| `l2` | Euclidean distance (default) |
| `l1` | Manhattan distance |
| `ip` | Inner product |
| `angle` | Angular distance |

## License

MIT
