# NGT → Rust Port Plan

## Goals

- Strict 1:1 port of `vendors/ngt/lib/NGT` (core NGT + NGTQ extension) to pure Rust under `src/`
- Target: desktop library (not server), production quality — no shortcuts
- Phase 1: correctness, all `tests/ngt_engine.rs` tests pass
- Phase 2: performance (SIMD, memmap2 optimizations) — only after Phase 1 is complete and verified
- Do NOT be inspired by deleted leftover Rust files in git history — port only from the C++ source

---

## Guiding Principles

1. **1:1 fidelity first** — algorithms, data structures, and on-disk formats match C++ exactly
2. **No lazy simplifications** — e.g. `CompactVector` must enforce the `u16::MAX` limit, not just wrap `Vec`
3. **Rust idioms only where C++ has no equivalent** — ownership replaces `new`/`delete`, `Result` replaces exceptions, traits replace virtual dispatch
4. **No premature optimization** — no SIMD, no memmap2-backed hot paths, no rayon parallelism until Phase 2
5. **Each step reviewed before the next begins**

---

## Module Layout

```
src/
├── lib.rs                  — crate root, public re-exports
├── common.rs               — ✅ DONE: ObjectID, Distance, ObjectDistance, ResultSet,
│                             CompactVector, Repository, BooleanSet,
│                             BooleanVectorByEpoch, PropertySet, SearchContainer,
│                             serializer module
├── primitive_comparator.rs — distance functions (L1, L2, Cosine, Angle, Hamming, …)
├── object_space.rs         — ObjectSpace + ObjectRepository (object storage + distance dispatch)
├── node.rs                 — Node, InternalNode, LeafNode, Node::ID
├── tree.rs                 — DVPTree
├── graph.rs                — NeighborhoodGraph, GraphRepository
├── index.rs                — Index, GraphIndex, GraphAndTreeIndex, IndexProperty
├── mmap_index.rs           — MmapIndex (read-only memmap2-backed)
└── ngtq/                   — Phase 2: quantization
    ├── mod.rs
    ├── quantizer.rs
    ├── hierarchical_kmeans.rs
    ├── quantized_graph.rs
    └── quantized_blob_graph.rs
```

---

## Phase 1 — Core NGT

### Step 1 · `common.rs` ✅ DONE

Ports `NGT/Common.h`.

Key types and why they matter:

| Rust type | C++ source | Critical detail |
|---|---|---|
| `ObjectDistance` | `#pragma pack(2)` struct | `#[repr(C, packed(2))]`; fields copied to locals before compare (Rust safety) |
| `ResultSet` | `std::vector` + `push_heap` (max-heap) | Largest distance at top for bounded-k ejection |
| `BooleanSet` | Bit-vector, power-of-2 + padding | Fast visited tracking |
| `BooleanVectorByEpoch` | Epoch counter; array cleared only on overflow | Avoids O(n) clear per search |
| `CompactVector<T>` | `uint16_t` size/cap, aborts at 65 535 | Enforces NGT's per-node edge limit |
| `Repository<T>` | Sparse `Vec<Option<T>>`, min-heap freed list | Slot 0 = null; smallest freed ID reused first |
| `PropertySet` | `std::map<string,string>` with load/save | Index config persistence (tab-separated) |
| `SearchContainer` | `explorationCoefficient = epsilon + 1.0` | Internal coefficient, not epsilon directly |
| `serializer` module | `os.write((char*)&v, sizeof(v))` = LE bytes | Exact binary format compatibility |

---

### Step 2 · `primitive_comparator.rs`

Ports `NGT/PrimitiveComparator.h` + `PrimitiveComparatorNoArch.h`.

C++ structure: nested classes inside `PrimitiveComparator` for each metric × data type combination.

```rust
pub enum DistanceType {
    L1, L2, Hamming, Jaccard, Cosine, Angle,
    NormalizedAngle, NormalizedCosine,
    Poincare, Lorentz, DotProduct,
    // Phase 2: float16, bfloat16, qint4, qsint8 variants
}

// One function per metric — scalar implementations only in Phase 1.
// SIMD variants added in Phase 2 under #[cfg(target_arch = "x86_64")].
pub fn l1(a: &[f32], b: &[f32]) -> f32
pub fn l2(a: &[f32], b: &[f32]) -> f32
pub fn cosine(a: &[f32], b: &[f32]) -> f32     // 1 - dot(a,b)/(|a|*|b|)
pub fn angle(a: &[f32], b: &[f32]) -> f32      // acos(dot/(|a|*|b|))
pub fn hamming(a: &[u8], b: &[u8]) -> f32
pub fn jaccard(a: &[u8], b: &[u8]) -> f32
pub fn dot_product(a: &[f32], b: &[f32]) -> f32  // 1 - dot(a,b)
pub fn poincare(a: &[f32], b: &[f32]) -> f32
pub fn lorentz(a: &[f32], b: &[f32]) -> f32

// Helpers
pub fn normalize(v: &mut [f32])
pub fn norm(v: &[f32]) -> f32
```

---

### Step 3 · `object_space.rs`

Ports `NGT/ObjectSpace.h/cpp` + `NGT/ObjectRepository.h` + `NGT/ObjectSpaceRepository.h`.

C++ structure:
- `ObjectSpace` — abstract base; manages dimension, distance type, object storage
- `ObjectRepository` — `Repository<PersistentObject>` with serialize/deserialize
- `ObjectSpaceRepository` — combines ObjectSpace + ObjectRepository, adds `insert`/`get`/`distance`

Object types in C++ (Phase 1 implements float only):
```
ObjectType::Float   → Vec<f32>   ← Phase 1
ObjectType::Uint8   → Vec<u8>    ← later
ObjectType::Float16 → Vec<f16>   ← later
```

Key behaviors:
- `insert(v)`: if `DistanceType::Cosine` or `Angle`, normalize `v` before storing
- `distance(a_id, b_id)`: dispatch through `DistanceType` to `primitive_comparator`
- `distance_from_vec(query, id)`: same but query is a `&[f32]` (already normalized if Cosine/Angle)
- Serialization: binary format matching C++ `ObjectRepository::serialize`

```rust
pub struct ObjectSpace {
    pub dim: usize,
    pub distance_type: DistanceType,
    pub repository: ObjectRepository,
}

pub struct ObjectRepository {
    // Repository<Object> where Object = Vec<f32> for Phase 1
    inner: Repository<Vec<f32>>,
}
```

---

### Step 4 · `node.rs`

Ports `NGT/Node.h/cpp`.

C++ `Node::ID` is a packed 32-bit value: high bit = leaf/internal flag, low 31 bits = index.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeID {
    pub id: u32,     // 31-bit index
    pub leaf: bool,  // true = LeafNode, false = InternalNode
}
```

C++ `Node::Object`:
```rust
pub struct NodeObject {
    pub id: ObjectID,
    pub distance: Distance,   // distance to the node's pivot
}
```

Internal node: children + distance borders between them (for nearest-child lookup).
Leaf node: pivot ObjectID + list of `ObjectDistance` (members of this leaf cluster).

Serialization: mirrors `Node::serialize` / `Node::deserialize` binary format.

Pivot selection (used during tree splits):
- `select_pivot_by_max_distance` — find the pair with maximum distance; use the farther one
- `select_pivot_by_max_variance` — pick the dimension with max variance; pivot is extreme value

---

### Step 5 · `tree.rs`

Ports `NGT/Tree.h/cpp` (DVPTree — Dimensional Vantage Point Tree).

```rust
pub struct DVPTree {
    pub internal_nodes: Repository<InternalNode>,
    pub leaf_nodes: Repository<LeafNode>,
    pub root: Option<NodeID>,
    pub leaf_node_size: usize,           // default 100
    pub internal_children_size: usize,   // default 129
}
```

Three main operations:

**insert(id, object_space)**
1. Descend: at each InternalNode pick the child whose pivot is nearest to the new object.
2. At leaf: if `objects.len() < leaf_node_size`, append.
3. Otherwise split:
   - Pick pivot via `select_pivot_by_max_distance` from leaf members.
   - Partition objects into two new leaves by closer pivot.
   - Create InternalNode to replace old leaf.
   - If InternalNode overflows `internal_children_size`, split InternalNode too (recurse up).

**leaf_for_query(query, object_space) → leaf_id**
Descend by nearest-pivot at each level.

**get_object_ids_from_leaf(leaf_id) → &[ObjectDistance]**
Direct read from leaf node's `objects` field.

---

### Step 6 · `graph.rs`

Ports `NGT/Graph.h/cpp` (NeighborhoodGraph).

```rust
pub enum GraphType { ANNG, ONNG, KNNG, BKNG, IANNG }
pub enum IdenticalObjectEdgeType { SharedEdge, DirectedEdge }

pub struct NeighborhoodGraph {
    pub repository: GraphRepository,        // stores edges per node
    pub property: GraphProperty,
}

pub struct GraphRepository {
    // Vec of CompactVector<ObjectDistance> — node i has edges repository[i]
    nodes: Repository<CompactVector<ObjectDistance>>,
}
```

**ANNG build** (default graph type):
- For new object O with found k-NN neighbors:
  1. Insert directed edges O → each neighbor (up to `edge_size_for_creation`, sorted by dist).
  2. For each neighbor N: insert reverse edge N → O if N's edge count < `incoming_edge` limit.

**ONNG post-processing** (after full build):
- Outgoing pruning: keep ≤ `outgoing_edge` per node.
- Incoming pass: ensure each node has ≥ `incoming_edge` incoming.

**Search** (`NeighborhoodGraph::search`):
```
seeds ← tree.leaf_for_query(q) || random sample
candidates ← min-heap (BinaryHeap with Reverse)
result ← ResultSet (max-heap, bounded to k)
visited ← BooleanVectorByEpoch

for seed in seeds: push seed to candidates

loop:
    c ← candidates.pop_min()
    if c.distance > result.top().distance * exploration_coefficient: break
    for edge in graph[c.id]:
        if visited.visit(edge.id): continue
        d ← object_space.distance_from_vec(query, edge.id)
        candidates.push(edge.id, d)
        if result.len() < k || d < result.top().distance:
            result.push(d)
            if result.len() > k: result.pop()

return result sorted ascending
```

---

### Step 7 · `index.rs`

Ports `NGT/Index.h/cpp`.

```rust
pub enum IndexType         { GraphAndTree, Graph }
pub enum IndexDistanceType { L1, L2, Cosine, Angle, Hamming, Jaccard, DotProduct,
                             NormalizedAngle, NormalizedCosine }

pub struct IndexProperty {
    pub dim: usize,
    pub distance_type: IndexDistanceType,
    pub index_type: IndexType,
    pub thread_pool_size: usize,          // default 32
    pub edge_size_for_creation: usize,    // default 10
    pub edge_size_for_search: usize,      // default 0
    pub seed_size: usize,                 // default 10
    pub batch_size_for_creation: usize,   // default 200
    pub outgoing_edge: usize,             // default 10
    pub incoming_edge: usize,             // default 80
    pub leaf_node_size: usize,            // default 100
    pub internal_children_size: usize,    // default 129
    pub identical_object_edge_type: IdenticalObjectEdgeType,
    pub seed_type: SeedType,
    pub graph_type: GraphType,
}

pub struct Index {
    pub objects: Vec<Vec<f32>>,           // mirror of object_space (public for tests)
    pub graph: NeighborhoodGraph,
    pub tree: Option<DVPTree>,
    pub object_space: Option<ObjectSpace>,
    property: IndexProperty,
    path: Option<String>,
    batch_auto_build: bool,
    pending: Vec<ObjectID>,
}
```

**build()**: the C++ `Index::createIndexWithInsertionOrder` / `createIndex`:
1. Split pending objects into batches of `batch_size_for_creation`.
2. For each batch: search existing graph for k-NN (uses `seed_size` seeds from tree or random).
3. Insert edges (ANNG rules).
4. Insert objects into DVPTree if `GraphAndTree`.
5. ONNG post-pass if `graph_type == ONNG`.

**delete_batch()**: validate IDs → remove objects → compact IDs (renumber 1..n) → full rebuild.

**save / open**: binary format via `serializer` module matching C++ `Index::save` / `Index::open`.

**Directory layout** (`save_as_directory`):
```
{path}/
  grp          — graph (GraphRepository binary)
  tre          — tree (DVPTree binary), absent for Graph-only
  obj          — objects (ObjectRepository binary)
  prf          — properties (PropertySet tab-separated text)
```

---

### Step 8 · `mmap_index.rs`

Ports the read-only memory-mapped index path from `NGT/Index.h`.

Custom binary layout for fast open-without-copy:

```
{path}/
  header.bin   — dim, distance_type, object_count  (fixed-size, bincode)
  objects.bin  — packed f32: flat array, stride = dim
  offsets.bin  — u64 per node: byte offset into graph.bin
  graph.bin    — per node: [n_edges: u32, (id: u32, dist: f32) × n_edges]
```

```rust
pub struct MmapIndex {
    _mmap_objects: memmap2::Mmap,
    _mmap_graph:   memmap2::Mmap,
    _mmap_offsets: memmap2::Mmap,
    // slices into the mmaps
    objects:  &'static [f32],   // transmuted from Mmap bytes
    offsets:  &'static [u64],
    graph_bytes: &'static [u8],
    dim: usize,
    distance_type: DistanceType,
    object_count: usize,
}
```

---

### Step 9 · `lib.rs`

Final public API surface:
```rust
pub use common::{Distance, NgtError, ObjectDistance, ObjectID, SearchOptions, SearchResult};
pub use index::{Index, IndexDistanceType, IndexProperty, IndexType};
pub use mmap_index::MmapIndex;
pub use graph::{Edge, IdenticalObjectEdgeType};
pub use index::IdenticalObjectEdgeType;   // re-exported for tests
```

---

## Phase 2 — NGTQ Extension

Gate behind `ngtq` Cargo feature flag. Port after all Phase 1 tests pass.

| C++ file | Rust module | Key work |
|---|---|---|
| `NGTQ/Quantizer.h` | `ngtq/quantizer.rs` | Product quantization, lookup tables, ScalarQuantizedInt8 |
| `NGTQ/HierarchicalKmeans.h/cpp` | `ngtq/hierarchical_kmeans.rs` | Multi-layer k-means, HKNode tree |
| `NGTQ/ObjectFile.h` | `ngtq/object_file.rs` | NGTQ object storage |
| `NGTQ/QuantizedGraph.h/cpp` | `ngtq/quantized_graph.rs` | NGTQ::Index over quantized objects |
| `NGTQ/QuantizedBlobGraph.h/cpp` | `ngtq/quantized_blob_graph.rs` | QBG::Index with hierarchical clustering |
| `NGTQ/Optimizer.h/cpp` | `ngtq/optimizer.rs` | Post-build NGTQ graph optimization |

---

## Phase 3 — Performance

Only after Phase 1 + Phase 2 pass all tests.

- SIMD distance functions (`std::arch` x86_64 AVX2/AVX512) behind `#[cfg]`
- `rayon` parallel batch build
- `memmap2` hot-path for large index loading
- Profile and optimize hot search path (BooleanVectorByEpoch already in Phase 1)

---

## Implementation Order (Phase 1)

```
Step 1  common.rs             ✅ done
Step 2  primitive_comparator  no deps beyond common
Step 3  object_space          depends on common, primitive_comparator
Step 4  node.rs               depends on common
Step 5  tree.rs               depends on common, node, object_space
Step 6  graph.rs              depends on common, object_space
Step 7  index.rs              depends on all above
Step 8  mmap_index.rs         depends on common, primitive_comparator
Step 9  lib.rs                re-exports, final wiring
```

Each step: read the C++ source → implement → `cargo check` → run relevant tests → fix → done.

---

## Test Mapping

| Test | Steps required |
|---|---|
| `test_basic_insert_search` | 1–7 |
| `test_save_open` | 1–7 |
| `test_cosine_normalizes_*` | 2–7 (normalize on insert) |
| `test_default_property_matches_ngt_defaults` | 7 (IndexProperty defaults) |
| `test_save_open_directory_directory_layout` | 1–7 |
| `test_identical_object_directed_edge_behavior` | 6–7 |
| `test_tree_splits_and_returns_leaf_seeds` | 4–7 |
| `test_graph_only_build_does_not_create_tree` | 6–7 |
| `test_tree_guided_insertion_uses_current_object_leaf` | 4–7 |
| `test_parallel_batch_build_produces_graph` | 7 (batch build) |
| `test_save_open_mmap_directory_layout` | 8 |
| `test_delete_batch_rebuilds_and_compacts_ids` | 7 (delete + compact + rebuild) |
| `test_delete_batch_rejects_out_of_range_ids` | 7 (ID validation) |
| `test_insert_and_rebuild_*` | 7 |
| `test_batch_mutation_api_*` | 7 |
| `test_batch_build_toggle_false_*` | 7 |

---

## What Is NOT Ported (Phase 1)

| C++ component | Reason |
|---|---|
| `SharedMemoryAllocator` | Rust ownership replaces manual slab allocation |
| `MmapManager` / `ArrayFile` | Replaced by direct `memmap2` + serializer in Step 8 |
| C API (`Capi.cpp`) | Not needed for library use |
| `Command.h` / CLI | Not needed (tests drive the API) |
| `Thread.h` / custom thread pool | Phase 3 (rayon) |
| `HashBasedBooleanSet` | `BooleanVectorByEpoch` covers the use case |
| Float16 / bfloat16 / Qint4 object types | Phase 2 (NGTQ) |
| Poincaré / Lorentz distance | Uncommon; add if needed after Phase 1 |
| `GraphReconstructor`, `GraphOptimizer` | Post-build optimization; Phase 3 |
