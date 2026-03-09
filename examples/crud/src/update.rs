use anyhow::Result;
use munind_api::MunindEngine;
use munind_core::domain::{MemoryId, SearchRequest};
use munind_core::engine::VectorEngine;
use serde_json::json;

/// UPDATE:
/// Replaces embedding + document for an existing ID.
/// The record keeps the same ID, only content changes.
pub fn run(engine: &MunindEngine, id: MemoryId) -> Result<()> {
    let new_embedding = vec![0.0_f32, 1.0, 0.0];
    let new_document = json!({
        "doc_id": "example-1",
        "title": "CRUD Example",
        "text": "updated from examples/crud/update.rs",
        "source": "example",
        "folder": "engineering",
        "version": 2
    });

    engine.update_json(id, new_embedding.clone(), new_document)?;
    println!("UPDATE: updated id={}", id.0);

    // Verify update by running a query near the new embedding.
    let hits = engine.search(SearchRequest {
        vector: new_embedding,
        top_k: 1,
        text_query: None,
        hybrid_alpha: None,
        lexical_top_k: None,
        filter: None,
        ef_search: Some(80),
        radius: None,
    })?;

    if let Some(hit) = hits.first() {
        println!(
            "UPDATE(check): top hit id={} score={:.4}",
            hit.id.0, hit.score
        );
    }

    Ok(())
}
