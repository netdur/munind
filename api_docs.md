# Munind API Docs

This document focuses on behavior, not just method names.

## Core Concepts

- The index always ranks by **smaller distance first**.
- Even for dot product mode, the library converts score to a distance-like value.
- Object IDs are 1-based.

## Distance Types (What They Actually Mean)

`IndexDistanceType` supports:

- `L2`
- `L1`
- `Cosine`
- `Angle`
- `DotProduct`

### Quick comparison

| Type | Preprocessing | Effective distance used for ranking | Notes |
|---|---|---|---|
| `L2` | None | Euclidean distance | Standard metric space distance |
| `L1` | None | Manhattan distance | More robust to outliers than L2 in some datasets |
| `Cosine` | Vectors are normalized at insert/query time | ~`1 - dot(a, b)` on normalized vectors | Zero vectors are rejected |
| `Angle` | Vectors are normalized at insert/query time | `acos(dot(a, b))` on normalized vectors | Angular distance in radians |
| `DotProduct` | No normalization | `max_magnitude - dot(query, item)` | Maximizing dot product becomes minimizing distance |

### Cosine vs DotProduct (important)

#### `Cosine`
- Ignores vector magnitude (because vectors are normalized).
- Measures direction similarity.
- Good when embedding norm should not influence ranking.

#### `DotProduct`
- Preserves magnitude effects.
- Uses a transformed distance:
  - `distance = max_magnitude - dot(query, item)`
  - where `max_magnitude` is the max self-dot (`v·v`) seen in indexed objects.
- Good when norm carries signal and you want higher raw dot product preferred.

## `Index` API

### Lifecycle / Persistence

- `Index::create(path, property)`
- `Index::open(path)`
- `Index::open_directory(path)`
- `index.save(path_opt)`
- `index.save_as_directory(path)`
- `index.save_as_mmap(path)`

### Mutation

- `index.insert(object)`
- `index.insert_batch(objects)`
- `index.delete(id)`
- `index.delete_batch(ids)`
- `index.build()`
- `index.build_with_debug(debug)`
- `index.set_batch_auto_build(enabled)`

### Query / Introspection

- `index.search(query, options)`
- `index.linear_search(query, k)`
- `index.object_count()`
- `index.all_objects()`

## Batch Build Behavior

`insert_batch` and `delete_batch` follow an internal toggle:

- Default: `batch_auto_build = true`
- Change with: `index.set_batch_auto_build(false)`

When disabled:
- batch mutation updates objects/IDs
- graph/tree is not rebuilt automatically
- call `index.build()` before ANN search quality is expected

## Delete Semantics

`delete_batch` compacts remaining objects and can reassign IDs.

If ID stability is required externally, keep your own external ID mapping layer.

## Persistence Methods: Difference Between `save*`

### `save(path_opt)`
- Writes a **single bincode file** snapshot.
- Best when you want one portable file.
- Load with: `Index::open(file_path)`.

### `save_as_directory(path)`
- Writes a **mutable directory layout**:
  - `prf` (properties)
  - `obj` (objects)
  - `grp` (graph)
  - `tre` (tree)
  - `robj` (reserved/compat side file)
- Load with: `Index::open_directory(dir_path)`.
- Good for tooling/debugging and explicit component files.

### `save_as_mmap(path)`
- Writes an **mmap-oriented read path layout**:
  - `prf`
  - `obj.mmap`
  - `grp.mmap`
  - `tre.bin`
- Open with: `MmapIndex::open(dir_path)` (read-only search index).
- Good for fast startup / mmap-based serving.

## `MmapIndex` API

- `MmapIndex::open(path)`
- `mmap.object_count()`
- `mmap.search(query, options)`
- `mmap.linear_search(query, k)`

## CLI Commands (implemented)

- `create`
- `search`
- `append`
- `info`
- `export`
- `import`
- `export-graph`
- `export-objects`
- `rebuild`
- `remove`

## Examples

### Default batch behavior (auto build ON)

```rust
use munind::{Index, IndexDistanceType, IndexProperty, SearchOptions};

let mut prop = IndexProperty::new(2);
prop.set_distance_type(IndexDistanceType::Cosine);

let mut index = Index::create("./idx", prop)?;
index.insert_batch(&vec![
    vec![1.0, 0.0],
    vec![0.0, 1.0],
])?; // auto-build by default

let res = index.search(
    &[1.0, 0.1],
    &SearchOptions { k: 1, epsilon: 0.0, edge_size: Some(10) },
)?;
println!("top id = {}", res[0].id);
# Ok::<(), String>(())
```

### Manual build mode (auto build OFF)

```rust
index.set_batch_auto_build(false);
index.insert_batch(&vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
index.delete_batch(&[1])?;
index.build();
# Ok::<(), String>(())
```
