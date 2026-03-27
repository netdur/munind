# Mmap-Native Index Plan

The project is past the phase where strict NGT internal parity is the goal. NGT was useful as a sanity check for correctness and performance, but the next direction should be a Rust-native storage and search design that keeps the current quality and speed while improving startup cost, memory usage, and code structure.

## Goals

- Replace serde-heavy load paths with mmap-backed read paths.
- Keep the mutable in-memory builder for create-time simplicity.
- Finalize built indexes into flat files that are efficient to map and search.
- Preserve current search behavior and property semantics where they are already good enough.
- Avoid porting NGT's shared-memory allocator design.

## Architecture

### 1. Builder vs Reader split
- `Index` remains the mutable builder and reference implementation.
- A new `MmapIndex` becomes the read-only search path.
- The builder owns `Vec`-based objects, graph, and tree during construction.
- Finalization writes flat files that `MmapIndex` can open with `memmap2`.

### 2. Flat on-disk layout
- Keep the existing `prf` property file as human-readable metadata.
- Store objects in a dedicated flat binary file with:
  - magic/version
  - object count
  - dimension
  - max magnitude for dot-product distance
  - contiguous normalized vectors
- Store graph adjacency in a dedicated flat binary file with:
  - magic/version
  - node count
  - edge count
  - adjacency offsets
  - packed `(id, distance)` edge records
- Keep the current tree as a sidecar file initially.
  - This preserves current GraphAndTree behavior while avoiding an all-at-once tree rewrite.

### 3. Search path
- `MmapIndex::search` should:
  - prepare the query with the same distance semantics as the builder
  - get initial seeds from the tree when available
  - traverse graph edges directly from mapped adjacency bytes
  - read object vectors directly from mapped object bytes
- `MmapIndex::linear_search` should provide a correctness baseline over mapped objects.

## Phase Breakdown

### Phase 1
- Add `memmap2`.
- Define the object and graph flat file formats.
- Implement `Index::save_as_mmap`.
- Implement `MmapIndex::open`.
- Implement mmap-backed object access, graph adjacency access, `linear_search`, and graph search.
- Reuse the existing tree as a serialized sidecar.

Status: implemented.

### Phase 2
- Add CLI support for opening/searching mmap indexes directly.
- Add benchmarking to compare mmap open/search against the current in-memory load path.
- Verify recall parity between `Index` and `MmapIndex` on the same built index.

Status: pending.

### Phase 3
- Replace the serialized tree sidecar with a flat mmap-friendly tree format.
- Remove unnecessary object duplication from open/load paths.
- Consider explicit SIMD/NEON-friendly alignment in the finalized flat files.

Status: pending.

## Constraints

- Do not chase native NGT binary compatibility.
- Do not port NGT's shared-memory allocator.
- Prefer flat arrays and borrowed slices over pointer-rich layouts.
- Keep the mutable builder code path stable while introducing the mmap reader alongside it.
