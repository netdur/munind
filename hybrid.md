# Hybrid Search Plan

## Goal
Add hybrid retrieval so each document supports:
- vector search (ANN)
- metadata storage and filtering (JSON input, BSON storage)
- lexical search (BM25 on selected KV fields)
- rerank over merged candidates

This must support queries like:
- image similarity where `tag == "nyc"`
- vector search where `timestamp >= start && timestamp < end`

## Scope
1. Data model: vector + metadata payload per document.
2. Filtering: exact match and numeric/date range predicates.
3. BM25: inverted index over configured metadata text fields.
4. Hybrid retrieval: ANN + BM25 candidate generation.
5. Rerank: weighted score fusion on combined candidates.
6. Persistence: compatible with current binary and directory layouts.

## Query Semantics
1. `vector_only`: ANN only.
2. `vector + filter`: ANN constrained by metadata predicates.
3. `text_only`: BM25 only.
4. `hybrid`: ANN + BM25, apply filters, merge, rerank.

Filter operators for v1:
- equality: `field == value`
- inclusion: `field in [v1, v2, ...]`
- range: `field >= x`, `field > x`, `field <= x`, `field < x`

Type handling:
- `string`, `bool`, `number`, `timestamp` (stored as i64 epoch ms for range index)

## Architecture
### 1) Document Storage
- Keep existing vector/object ID behavior.
- Add `metadata_bson_by_id: Vec<Option<Vec<u8>>>` aligned to object IDs.
- Add parsed typed shadow indexes:
  - `keyword_index[field][term] -> Vec<doc_id>`
  - `numeric_index[field] -> sorted Vec<(value, doc_id)>`

Reason:
- BSON is compact canonical storage.
- Shadow indexes make filtering fast.

### 2) BM25 Index
- New module: `src/bm25.rs`
- Configured fields only (example: `title`, `caption`, `tags`).
- Structures:
  - `postings[(field, term)] -> Vec<(doc_id, tf)>`
  - `doc_len[field][doc_id]`
  - `avg_doc_len[field]`
  - `doc_freq[(field, term)]`
  - `num_docs`
- Tokenization v1:
  - lowercase
  - split on non-alphanumeric
  - drop empty tokens

### 3) Hybrid Search Pipeline
Inputs:
- optional query vector
- optional text query
- optional filters
- `k`, candidate sizes, fusion weights

Pipeline:
1. Resolve filter bitset/candidate set from metadata indexes.
2. ANN candidate generation:
   - get ANN top-N
   - apply filter (or prefilter seeds if feasible later)
3. BM25 candidate generation:
   - retrieve top-M by BM25 on selected fields
   - apply same filters
4. Merge by doc ID.
5. Normalize scores:
   - vector distance to similarity score
   - BM25 min-max or z-score normalization
6. Rerank by weighted fusion:
   - `final = w_vector * vector_score + w_bm25 * bm25_score`
7. Return top-k with component scores.

## API and CLI Changes
### Core API
- Keep existing `insert(&[f32])` and `search(...)` stable.
- Add:
  - `insert_with_metadata(vector, metadata_json_or_bson)`
  - `search_with_filters(...)`
  - `hybrid_search(...)`

### CLI
Extend `munind` with:
- create/append:
  - support JSONL input records:
    - `{"vector":[...], "meta":{...}}`
- search:
  - `--text "..."` for BM25/hybrid
  - `--filter 'tag == "nyc" and timestamp >= 1700000000000 and timestamp < 1710000000000'`
  - `--vector-weight`, `--bm25-weight`
  - `--ann-candidates`, `--bm25-candidates`

## Persistence and Compatibility
### Directory format
Add sidecars:
- `meta.bson` (or serialized per-doc metadata array)
- `meta.idx` (keyword + numeric filter indexes)
- `bm25.idx`

### Binary format
Add new fields with serde defaults to keep older files readable.

Compatibility requirements:
1. Old index without metadata/BM25 must still open.
2. Missing sidecars means metadata/BM25 features disabled, not fatal.
3. Existing ANN behavior unchanged when hybrid features are unused.

## Implementation Phases
### Phase 1: Metadata foundation
1. Add metadata storage by doc ID.
2. Add JSONL ingest and JSON->BSON conversion.
3. Add keyword and numeric filter indexes.
4. Add filter parser and evaluator.
5. Tests for equality and range filters.

### Phase 2: BM25
1. Implement BM25 module and indexing on selected fields.
2. Add BM25 query API and tests.
3. Add config in index property and persistence.

### Phase 3: Hybrid + rerank
1. Implement merged candidate generation.
2. Add score normalization and weighted rerank.
3. Extend CLI search options.
4. Add relevance and regression tests.

### Phase 4: mmap and performance
1. Add mmap-compatible metadata/BM25 loading strategy.
2. Benchmark latency impact and memory footprint.
3. Add guardrails for large-cardinality fields.

## Acceptance Criteria
1. Query supports ANN constrained by metadata:
   - `tag == "nyc"`
   - `timestamp` range predicates
2. BM25 returns lexical matches on selected KV fields.
3. Hybrid query returns reranked merged results.
4. Existing vector-only API/CLI remains backward compatible.
5. Open/save works for:
   - binary
   - directory
   - mmap directory (for ANN + filters at minimum in v1)

## Risks and Mitigations
1. Filter-first ANN can be slow if filter selectivity is high and graph traversal is unconstrained.
   - Mitigation: candidate oversampling + post-filter initially, then filter-aware traversal.
2. BM25 memory growth on high-cardinality fields.
   - Mitigation: explicit field allowlist and posting compression later.
3. Score fusion instability across datasets.
   - Mitigation: expose weights and normalization mode in config/CLI.
