# munind

Fast approximate nearest neighbor search library in Rust.

Ported from [NGT](https://github.com/yahoojapan/NGT) (Yahoo Japan) with NEON SIMD, flat contiguous storage, and optional TurboQuant compression.

## Features

- Graph-based ANN search (DVPTree + ANNG)
- NEON SIMD distance functions on Apple Silicon / ARM
- Flat contiguous object storage for cache-friendly access
- TurboQuant quantization (3/4/8-bit, data-oblivious, no training)
- Memory-mapped index loading (memmap2)
- Parallel index building (rayon)
- C FFI (`libmunind.dylib` / `.so`)
- CLI tool

## Quick Start

### Rust

```rust
use munind::api::{Index, IndexConfig, Distance};

let config = IndexConfig::new(128, Distance::Cosine);
let mut index = Index::create(config).unwrap();

index.insert(&vec![0.1; 128]).unwrap();
index.insert(&vec![0.2; 128]).unwrap();
index.build().unwrap();

let results = index.search(&vec![0.15; 128], 10).unwrap();
println!("nearest: id={}, distance={}", results[0].id, results[0].distance);

index.save("my_index").unwrap();
```

### CLI

```bash
# Build
cargo build --release

# Create index from TSV data
munind create -d 100 -D c index_dir data.tsv

# Search
munind search -n 10 -e 0.1 index_dir queries.tsv

# Create with TurboQuant compression (8-bit)
munind create -d 100 -D c -q 8 index_dir data.tsv
```

### C FFI

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

## Performance

glove-100-angular (1.18M vectors, dim=100, cosine):

| | C++ NGT | munind | munind TQ-8 |
|---|---|---|---|
| Build | 1:49 | 0:57 | 0:55 + 0.8s |
| Objects | 453 MB | 453 MB | 164 MB |
| Search (ms) | 0.272 | 0.158 | 0.254 |
| Recall@10 | 0.628 | 0.635 | 0.625 |

## Distance Functions

`-D` flag or `Distance` enum:

| Flag | Distance | Notes |
|---|---|---|
| `2` | L2 (Euclidean) | Default |
| `c` | Cosine | Pre-normalizes on insert |
| `1` | L1 (Manhattan) | |
| `i` | Inner Product | |
| `a` | Angle | |
| `A` | Normalized Angle | |
| `E` | Normalized L2 | |

## TurboQuant

Optional lossy compression using the TurboQuant algorithm (arXiv 2504.19874). Data-oblivious: works on any dataset without training.

```bash
munind create -d 100 -D c -q 8 compressed_index data.tsv   # 8-bit, ~2.8x compression
munind create -d 100 -D c -q 4 compressed_index data.tsv   # 4-bit, higher compression, lower recall
```

Search auto-detects native vs TQ from the saved index.

## Building

```bash
cargo build --release          # library + CLI
cargo test                     # run tests
```

Produces:
- `target/release/munind` (CLI binary)
- `target/release/libmunind.dylib` (macOS) or `libmunind.so` (Linux)
- `target/release/libmunind.rlib` (Rust library)

C header: `include/munind.h`

## License

MIT
