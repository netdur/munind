# Search and Ranking

## Request Model

Search is driven by `SearchRequest`:
- `vector: Vec<f32>`
- `top_k: usize`
- `text_query: Option<String>`
- `hybrid_alpha: Option<f32>`
- `lexical_top_k: Option<usize>`
- `filter: Option<FilterExpression>`
- `ef_search: Option<usize>`
- `radius: Option<f32>`

## Filter Syntax (v1)

`FilterExpression` supports:
- `Eq(path, value)`
- `And(vec![...])`

Examples:
- `Eq("source", "journal")`
- `And([Eq("doc_id", "x"), Eq("source", "journal")])`

## Execution Modes

### 1) Vector-Only Search

- ANN retrieval from the graph index.
- Sorted by vector score.

### 2) Vector + Filter Search

- Payload index builds a filter plan.
- Fully indexed filters run on indexed candidate IDs.
- Partially indexed filters narrow candidates, then verify with JSON evaluation.

### 3) Hybrid Search (Vector + BM25F)

Enabled when `text_query` is non-empty.

- Vector candidates scored by metric-derived score.
- Lexical candidates scored by BM25F.
- Scores are min-max normalized and fused:
  - `fused = alpha * vector + (1 - alpha) * lexical`
- `alpha` defaults to `0.65` when omitted.

## Reranking (Second Stage)

Reranking is exposed via the pipeline crate (`RagPipeline` type in `munind-rag`).

Flow:
1. Retrieve top-N candidates (`rerank_candidate_count`, default `100`).
2. Send candidate documents to reranker.
3. Return final `top_k` results.

Available rerankers:
- `DeterministicReranker` (local test/dev baseline)
- `OpenAICompatibleReranker` (`POST /v1/rerank` style API)

## Indexed Filter Fields

Top-level:
- `doc_id`, `source`, `type`, `created_at`, `tags`, `session_id`

Metadata fallback:
- `metadata.doc_id`, `metadata.source`, `metadata.type`, `metadata.created_at`, `metadata.tags`, `metadata.session_id`

## Practical Guidance

- For strict constraints, always include a filter in the search request.
- For best relevance in natural-language queries, use hybrid + reranking.
- Tune `ef_search` upward for recall, downward for latency.

## Example Use Cases

- Personal memory retrieval
- RAG context retrieval for local assistants
- Semantic retrieval over app datasets
