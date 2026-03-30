# munind Library API Polish Plan

## 1. Error Handling
- Replace `type NgtError = String` with proper enum using `thiserror`
- Variants: `DimensionMismatch`, `InvalidId`, `IoError`, `IndexNotFound`, `EmptyIndex`
- All public methods return `Result<T, munind::Error>`

## 2. Clean Public API
Strip internal wrappers. Public surface:

```rust
// Core types
pub struct Index { /* opaque */ }
pub struct SearchResult { pub id: u32, pub distance: f32 }
pub struct IndexConfig { pub dimension: usize, pub distance: Distance, pub edge_size: i32, ... }
pub enum Distance { L2, Cosine, InnerProduct, L1, ... }

// Lifecycle
Index::create(config) -> Result<Index>
Index::open(path) -> Result<Index>
index.save(path) -> Result<()>

// Mutation
index.insert(vector) -> Result<u32>
index.insert_batch(vectors) -> Result<Vec<u32>>
index.build() -> Result<()>
index.remove(id) -> Result<()>

// Search
index.search(query, k) -> Result<Vec<SearchResult>>
index.search_with(query, k, epsilon, edge_size) -> Result<Vec<SearchResult>>
index.search_batch(queries, k) -> Result<Vec<Vec<SearchResult>>>
index.linear_search(query, k) -> Result<Vec<SearchResult>>

// Info
index.len() -> usize
index.dimension() -> usize
index.get(id) -> Result<Vec<f32>>
```

## 3. Incremental Insert
- `index.insert(v)` on an already-built index: append + immediately build edges for new object
- No full rebuild needed

## 4. Batch Search
- `search_batch(&[queries], k)` — parallel search using rayon
- Each query independent, embarrassingly parallel

## 5. FFI (C API)
```c
// munind.h
typedef struct munind_index munind_index;
typedef struct { uint32_t id; float distance; } munind_result;

munind_index* munind_create(uint32_t dim, const char* distance);
munind_index* munind_open(const char* path);
void munind_free(munind_index* idx);

int munind_insert(munind_index* idx, const float* vec, uint32_t dim);
void munind_build(munind_index* idx);
int munind_save(munind_index* idx, const char* path);

int munind_search(munind_index* idx, const float* query, uint32_t dim,
                  uint32_t k, float epsilon,
                  munind_result* results, uint32_t* result_count);

uint32_t munind_len(munind_index* idx);
```

Build as `cdylib` + `staticlib` for linking from C, Python, Swift, etc.

## 6. Documentation
- Rustdoc on all public types and methods
- Top-level crate doc with example
- README.md with usage examples

## 7. Crate Metadata
- Cargo.toml: description, license, repository, keywords, categories
- lib target: cdylib + rlib

## Execution Order
1. Error enum (everything depends on it)
2. Clean public API module (`api.rs`)
3. Batch search
4. FFI module (`ffi.rs`)
5. Documentation
6. Crate metadata
