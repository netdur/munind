# 06 - RAG Memory Features

## Memory Ingestion Pipeline
1. Normalize raw note/message/document.
2. Chunk content (semantic or token-window chunker).
3. Generate embeddings via configured model/provider.
4. Attach metadata and scoring hints.
5. Persist and index.

## Memory Record Types
- `note`
- `conversation_turn`
- `document_chunk`
- `task`
- `event`

Type affects defaults for:
- chunking policy
- recency decay
- importance priors

## Metadata Schema (Recommended)
- `owner_id`
- `session_id`
- `source`
- `tags[]`
- `created_at`
- `last_accessed_at`
- `importance`
- `language`
- `privacy_level`

## Retrieval Pipeline For RAG
1. Embed query.
2. ANN retrieve top `N1` with filters.
3. Optional lexical pass for keyword-sensitive queries.
4. Merge and deduplicate candidates.
5. Optional rerank to top `N2`.
6. Build final context pack (`N3` items + token budget).

## V1 JSON Filter Operators
- Required: equality filter (`field = value`) applied with vector search.
- Composition: AND across multiple equality predicates.
- Future extension candidates: range, contains, OR groups.

## Recency And Importance Policy
Scoring components:
- semantic similarity
- recency decay (`exp(-age/half_life)`)
- explicit importance
- optional feedback boost from user actions

Keep scoring pluggable so behavior can evolve without storage migration.

## Context Assembly Rules
- Enforce token budget per response.
- Prefer diversity across sources/tags.
- Merge adjacent chunks from same source when beneficial.
- Include citation metadata for traceability.

## Update And Re-Embedding
- On embedding model change:
  - mark records with old `embedding_model_id`
  - re-embed in background
  - dual-read old/new vectors until migration complete

## Privacy Controls
- Selective field encryption.
- Per-record access policy tags.
- Secure erase path during compaction for deleted sensitive JSON documents.
