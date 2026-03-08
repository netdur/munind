# Rust API Guide

## Crates

Most integrations use:
- `munind-api` (`MunindEngine`)
- `munind-core` (`EngineConfig`, `SearchRequest`, `FilterExpression`)

## Create or Open Database

```rust
use munind_api::MunindEngine;
use munind_core::config::EngineConfig;

let config = EngineConfig::default();
let engine = MunindEngine::create("./my_vectors", 512, config.clone())?;
// Later:
let reopened = MunindEngine::open("./my_vectors")?;
```

## Insert Embedding + JSON

```rust
use munind_core::engine::VectorEngine;
use serde_json::json;

let embedding = vec![0.0_f32; 512];
let doc = json!({
  "doc_id": "note-001",
  "source": "journal",
  "type": "note",
  "tags": ["health", "sleep"],
  "text": "I slept earlier and felt better."
});

let id = engine.insert_json(embedding, doc)?;
```

## Vector Search

```rust
use munind_core::domain::SearchRequest;

let hits = engine.search(SearchRequest {
    vector: vec![0.0_f32; 512],
    top_k: 10,
    text_query: None,
    hybrid_alpha: None,
    lexical_top_k: None,
    filter: None,
    ef_search: Some(80),
    radius: None,
})?;
```

## Filtered Search

```rust
use munind_core::domain::{FilterExpression, SearchRequest};
use serde_json::json;

let hits = engine.search(SearchRequest {
    vector: vec![0.0_f32; 512],
    top_k: 10,
    text_query: None,
    hybrid_alpha: None,
    lexical_top_k: None,
    filter: Some(FilterExpression::And(vec![
        FilterExpression::Eq("source".into(), json!("journal")),
        FilterExpression::Eq("type".into(), json!("note")),
    ])),
    ef_search: Some(80),
    radius: None,
})?;
```

## Hybrid Search

```rust
let hits = engine.search(SearchRequest {
    vector: vec![0.0_f32; 512],
    top_k: 10,
    text_query: Some("sleep quality".into()),
    hybrid_alpha: Some(0.65),
    lexical_top_k: Some(200),
    filter: None,
    ef_search: Some(80),
    radius: None,
})?;
```


## Lexical-Only Search

Set `text_query` and pass an empty `vector`:

```rust
let hits = engine.search(SearchRequest {
    vector: Vec::new(),
    top_k: 10,
    text_query: Some("sleep quality".into()),
    hybrid_alpha: Some(0.0),
    lexical_top_k: Some(200),
    filter: None,
    ef_search: None,
    radius: None,
})?;
```
## Delete and Optimize

```rust
use munind_core::domain::OptimizeRequest;

engine.remove(id)?;

let report = engine.optimize(OptimizeRequest {
    force_full_compaction: true,
    repair_graph: true,
})?;
```

## Optional Pipeline Module

For chunking + embedding + optional reranking, use `RagPipeline` in `munind-rag`.
See `crates/munind-rag/src/pipeline.rs` for provider wiring patterns.

## Example Application Categories

- Personal memory apps
- RAG context retrieval
- Semantic retrieval over local content
