# TurboQuant Integration Plan

## Overview

Add TurboQuant as an alternative quantization backend to munind, selectable via
`-tq` flag on the CLI.  TurboQuant replaces NGTQ's trained product quantization
with a data-oblivious approach: random rotation + scalar quantization + 1-bit
QJL residual correction.

```
munind create -d 100 -D c index data.tsv          # native (full precision)
munind create -d 100 -D c -tq 4 index data.tsv    # TurboQuant at 4 bits/dim
munind search -e 0.1 index queries.tsv             # auto-detects native vs tq
munind search-mmap -e 0.1 index queries.tsv        # mmap path, same auto-detect
```

## Why TurboQuant over NGTQ

| Aspect | NGTQ | TurboQuant |
|--------|------|------------|
| Codebook training | k-means (minutes) | None (precomputed) |
| Data-dependent | Yes | No (data-oblivious) |
| Implementation | ~18K lines C++ | ~500 lines Rust |
| Indexing overhead | Significant | Near zero |
| Inner product bias | Variable | Proven unbiased |
| Theoretical bounds | None | Near-optimal distortion rate |

## Algorithm Summary

### TurboQuant_mse (b bits per dimension)

```
Global precomputation (once):
  Π ← random orthogonal matrix (d × d), via QR of N(0,1) matrix
  codebook ← Max-Lloyd optimal centroids for Beta((d-1)/2, (d-1)/2) distribution
             mapped to [-1/√d, 1/√d] range, with 2^b levels

Quantize(x):
  y ← Π · (x / ‖x‖)              # rotate unit vector
  idx[j] ← argmin_k |y[j] - c[k]|  # scalar quantize each coordinate
  return (idx, ‖x‖)

Dequantize(idx, norm):
  ỹ[j] ← c[idx[j]]               # lookup centroid per coordinate
  x̃ ← norm · Π^T · ỹ             # inverse rotate
  return x̃
```

### TurboQuant_prod (b bits, unbiased inner product)

```
Global precomputation (once):
  Same as above, plus:
  S ← random matrix (d × d) with i.i.d. N(0,1) entries

Quantize(x):
  (idx, norm) ← TurboQuant_mse.Quantize(x) at (b-1) bits
  x̃_mse ← TurboQuant_mse.Dequantize(idx, norm)
  r ← x - x̃_mse                   # residual
  qjl ← sign(S · r)                # 1-bit QJL (d bits = d/8 bytes)
  γ ← ‖r‖₂
  return (idx, qjl, γ)

Distance(query, quantized_db_vec):
  x̃_mse ← Dequantize_mse(idx)
  x̃_qjl ← (√(π/2)/d) · γ · S^T · qjl
  x̃ ← x̃_mse + x̃_qjl
  return distance(query, x̃)        # exact distance on reconstructed vector
```

### Asymmetric search (fast path)

Query stays at full precision. Only database vectors are quantized.
Distance = `dist(q, dequant(x_compressed))`.  No lookup tables needed —
just scalar dequantization + rotation + standard distance function.

## Memory Layout

### Quantized object storage (`obj.tq`)

```
Header (32 bytes):
  [8]  u64  slot_count
  [8]  u64  dim
  [4]  u32  bits_per_dim (b)
  [4]  u32  mode (0 = mse, 1 = prod)
  [4]  f32  reserved
  [4]  u32  reserved

Norms (slot_count × 4 bytes):
  [f32]  per-object L2 norm

Codes (slot_count × ceil(dim * b / 8) bytes):
  packed bit codes, b bits per dimension, row-major

QJL residuals (only if mode=prod):
  Gamma (slot_count × 4 bytes): [f32] residual norms
  Signs (slot_count × ceil(dim / 8) bytes): packed 1-bit signs

Presence bitmap (slot_count bytes):
  [u8] 0=absent, 1=present
```

### Rotation matrix (`rotation.tq`)

```
[8]  u64  d (dimension)
[d × d × 4 bytes]  f32 rotation matrix Π, row-major
```

### Codebook (`codebook.tq`)

```
[4]  u32  bits_per_dim (b)
[4]  u32  num_levels (2^b)
[num_levels × 4 bytes]  f32 centroid values
```

### Random matrix for QJL (`qjl.tq`, only if mode=prod)

```
[8]  u64  d
[d × d × 4 bytes]  f32 matrix S, row-major
```

## Module Layout

```
src/
├── tq/
│   ├── mod.rs            — pub exports, TqIndex type
│   ├── rotation.rs       — random orthogonal matrix generation (QR decomposition)
│   ├── codebook.rs       — Max-Lloyd scalar quantizer for Beta distribution
│   ├── quantizer.rs      — TurboQuant_mse and TurboQuant_prod encode/decode
│   ├── storage.rs        — quantized object storage (flat packed bits)
│   └── search.rs         — asymmetric distance + graph search with quantized objects
└── ...existing modules...
```

## Implementation Steps

### Step 1: `tq/rotation.rs`
- Generate random orthogonal matrix via QR decomposition of N(0,1) matrix
- Use `rand` crate for random generation
- Matrix-vector multiply `Π · x` and `Π^T · y`
- Serialize/deserialize rotation matrix
- **Test**: Π is orthogonal (Π^T · Π = I), rotation preserves norms

### Step 2: `tq/codebook.rs`
- Precompute Max-Lloyd optimal codebook for Beta((d-1)/2, (d-1)/2) on [-1, 1]
- Map to actual range [-1/√d, 1/√d] based on dimension
- For common bit widths (1, 2, 3, 4): hardcode known optimal centroids
- Scalar quantize: given value, find nearest centroid index
- Scalar dequantize: given index, return centroid value
- **Test**: quantize → dequantize roundtrip error matches theoretical bound

### Step 3: `tq/quantizer.rs`
- `TqQuantizer` struct holding rotation matrix + codebook + optional QJL matrix
- `encode_mse(x) → (codes, norm)` — rotate, scalar quantize each dim
- `decode_mse(codes, norm) → x̃` — dequantize, inverse rotate, scale by norm
- `encode_prod(x) → (codes, qjl_signs, gamma, norm)` — mse + residual + QJL
- `decode_prod(...) → x̃` — full reconstruction
- **Test**: inner product estimation is unbiased (average error → 0 over many samples)

### Step 4: `tq/storage.rs`
- `TqObjectSpace` — flat packed storage for quantized vectors
- Bit-packing: for b bits/dim, pack `d*b` bits per object
- Efficient decode of single object (for asymmetric distance)
- Serialize/deserialize matching the `obj.tq` format above
- Mmap-friendly: header + norms + codes + bitmap are contiguous
- **Test**: pack → unpack roundtrip, mmap load

### Step 5: `tq/search.rs`
- Asymmetric distance: full-precision query vs quantized database vector
- Integrate with existing `NeighborhoodGraph::search` via `ObjectAccessor` trait
- `TqObjectAccessor`: implements `get_object` by dequantizing on the fly
- `TqObjectAccessor`: implements `distance` as `dist(query, dequant(db_vec))`
- Linear search over quantized objects
- **Test**: recall on small dataset matches full-precision within expected bounds

### Step 6: `tq/mod.rs` + CLI integration
- `TqIndex` struct: wraps `NgtIndex` build + `TqObjectSpace` for search
- Build flow:
  1. Insert objects at full precision → build graph + tree (existing code)
  2. Generate rotation matrix, quantize all objects, save quantized storage
  3. Save: `prf`, `grp`, `tre`, `obj.tq`, `rotation.tq`, `codebook.tq`
- Search flow:
  1. Load quantized objects (mmap)
  2. Load graph + tree (existing code)
  3. Search using asymmetric distance through graph
- CLI: `-tq <bits>` flag on `create`, auto-detected on `search`

### Step 7: Benchmarks
- Compare on glove-100-angular:
  - Native (32-bit): current baseline
  - TQ-4 (4 bits/dim): 8× compression
  - TQ-2 (2 bits/dim): 16× compression
- Measure: build time, index size, recall@10, query latency
- Compare with NGTQ numbers from literature if available

## Expected Results

### Memory reduction (glove-100-angular, 1.18M × 100-dim)

| Mode | Bits/dim | Object size | Total objects | Compression |
|------|----------|-------------|---------------|-------------|
| Native | 32 | 400 B | 453 MB | 1× |
| TQ-4 | 4 | 50 B + 4 B norm | 63 MB | 7.2× |
| TQ-3 | 3 | 38 B + 4 B norm | 49 MB | 9.2× |
| TQ-2 | 2 | 25 B + 4 B norm | 34 MB | 13× |

### Recall expectations (from paper)

| Bits/dim | MSE distortion | Expected recall@10 impact |
|----------|---------------|--------------------------|
| 4 | 0.009 | <1% recall loss |
| 3 | 0.03 | ~2% recall loss |
| 2 | 0.117 | ~5-10% recall loss |

## Open Questions

1. **Rotation matrix size**: d×d matrix is d²×4 bytes. For d=100, that's 40KB (fine).
   For d=1000, it's 4MB (still fine). For d=10000, it's 400MB — may need
   structured random rotations (Hadamard + diagonal sign) instead.

2. **QJL matrix size**: Same as rotation. For `prod` mode, S is another d×d matrix.
   Consider using the same structured approach if d is large.

3. **NEON SIMD for dequantization**: The decode path (scalar dequant + matrix multiply)
   is the hot path during search. NEON can accelerate the matrix-vector multiply.

4. **Batch dequantization**: During graph traversal, multiple neighbors are decoded.
   Batching the rotation inverse could amortize matrix setup.
