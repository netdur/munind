# 05 - Indexing And Search

## Core Index Shape
Primary ANN structure:
- Directed neighbor graph with bounded out-degree.

Optional accelerator:
- VP-tree seed index for initial candidates (NGT-inspired Graph+Tree mode).

## Build Modes
- Online insert mode: each new vector is inserted incrementally.
- Batch build mode: builds graph from bulk dataset with better global quality.

## Insertion Flow (NGT-Inspired)
1. Choose seeds (`SeedIndex` or entry point heuristic).
2. Greedy search to gather candidate neighbors.
3. Select top candidates with pruning strategy.
4. Add bidirectional edges (subject to degree limits).
5. Local repair to preserve connectivity.

## Search Flow
1. Generate initial seed set.
2. Best-first graph traversal using candidate/result heaps.
3. Stop by `ef_search`/visit budget.
4. Apply metadata filters.
5. Optional rerank (exact distance or cross-encoder stage).

Search controls:
- `k`
- `ef_search`
- `exploration_factor` (NGT epsilon analog)
- `radius`
- `max_visits`

## Edge Selection And Pruning
- Degree bounds: `min_degree`, `max_degree`.
- Candidate pruning using distance-diversity heuristic.
- Optional relative-neighborhood style pruning for better graph quality.

## Deletion Model
- Mark tombstone in record metadata.
- Remove or invalidate edges lazily.
- Background graph repair:
  - reconnect neighbors of removed node
  - prune orphaned/invalid edges

## Exact Baseline Path
For recall evaluation and tiny datasets, provide brute-force exact search path:
- dense vector scan with SIMD distance kernels.
- used in tests and benchmark truth generation.

## Filter-Aware Retrieval
- Pre-filter candidate IDs when cheap/selective.
- Post-filter final hits always for correctness.
- Maintain light metadata inverted indexes for common fields.
- v1 required operator: JSON equality filter (`field = value`) combined with AND.

## Parameter Defaults (Start Point)
- `max_degree = 32`
- `ef_construction = 200`
- `ef_search = 80`
- `exploration_factor = 1.1`

Tune per dataset via benchmark harness before hardening.

## Future Extensions
- Multi-layer graph.
- Product/scalar quantization plugin (post-v1).
- Learned routing for seed selection.
