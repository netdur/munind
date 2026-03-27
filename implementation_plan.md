# Discrepancies between Munind & NGT Search Implementations

The lower recall (0.6206 vs 0.6282) and higher query latency (0.43ms vs 0.27ms) of the Rust `munind` implementation compared to the original C++ NGT baseline stems directly from three primary discrepancies in how seeds are selected from the graph and tree, as well as minor queuing differences.

## 1. Single Path Greedy Leaf Search vs Beam Search
- **C++ NGT:** During `SearchLeaf` mode, the C++ tree search only follows the single closest branch at each step (greedy traversal), explicitly ignoring other boundaries and avoiding the `pending` queue entirely. It lands on exactly **one** leaf node.
- **Rust Munind:** `nearest_leaves_for_query` does a full branch-and-bound exploration using `pending` queues and searches up to `internal_children_size` leaves (default 5). This fetches and checks many more pivots, contributing significantly to the slower query latency.

## 2. Seed Selection and Random Thinning
- **C++ NGT:** After finding the single nearest leaf, NGT selects seeds using a deterministic fallback to random thinning. If the leaf contains 100 items and we need 10 seeds, NGT applies an in-place seeded random shuffle (`srand(seeds[0].id)`) to pick 10 random nodes scattered across the leaf cluster. This provides maximum spatial diversity for the graph search.
- **Rust Munind:** Rather than randomly selecting, Rust sorts the objects inside the nearest leaves by their distance to the leaf's pivot, and truncates the list. This forces the graph search to start from a group of seeds mathematically clamped to the absolute center of a single cluster. This lack of spatial diversity directly causes the graph traversal to get trapped in local minima, resulting in the lower recall.

## 3. Seed Size Dynamic Scaling
- **C++ NGT:** When `seedSize` configuration is 0 (the default), C++ dynamically falls back to taking exactly `k` (the top-k parameter requested by the user query).
- **Rust Munind:** When `seedSize` is 0, Rust falls back to `edge_size_for_creation` (which defaults to 10). While the benchmark runs with `k=10`, if the user ever queries for `k=100`, Rust will dramatically under-seed the search starting point.

## Proposed Changes

### [MODIFY] `src/tree.rs`
- Add a new `greedy_leaf_for_query` function that rigorously follows the C++ NGT single path greedy logic for `SearchLeaf`.

### [MODIFY] `src/index.rs`
- Update `effective_seed_count` to accept `k` as a parameter and fallback to it when `property.seed_size == 0`.
- Update `get_seeds_from_tree` to fetch the single greedy leaf using `greedy_leaf_for_query`.
- Update `thin_tree_seeds` to replicate NGT's `srand` deterministic random thinning logic using the target's first seed ID to correctly thin data down to `seedSize`.

### [MODIFY] `src/graph.rs`
- Minor update to alignment for `SearchContainer` and `search` loop logic, verifying any remaining differences.

## User Action Required
Please approve this plan so that I can implement these modifications to perfectly align Rust's search implementation with C++ NGT to close the recall and latency gap.
