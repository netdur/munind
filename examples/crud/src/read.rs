use anyhow::Result;
use munind_api::MunindEngine;
use munind_core::domain::{FilterExpression, MemoryId, SearchHit, SearchRequest};
use munind_core::engine::VectorEngine;
use serde_json::json;

/// READ:
/// Demonstrates several read styles:
/// 1) direct lookup by ID via `get_record`
/// 2) pure vector ANN search
/// 3) text-only lexical search
/// 4) hybrid (vector + text) search
/// 5) folder-filtered search
pub fn run(engine: &MunindEngine, id: MemoryId, query: Vec<f32>, label: &str) -> Result<()> {
    // 1) Direct record lookup by ID.
    if let Some((embedding, document)) = engine.get_record(id)? {
        println!(
            "READ(get): id={} embedding_dim={} doc_id={}",
            id.0,
            embedding.len(),
            document
                .get("doc_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );
    } else {
        println!("READ(get): id={} not found", id.0);
    }

    // 2) Pure vector ANN search.
    let vector_hits = engine.search(SearchRequest {
        vector: query,
        top_k: 3,
        text_query: None,
        hybrid_alpha: None,
        lexical_top_k: None,
        filter: None,
        ef_search: Some(80),
        radius: None,
    })?;
    print_hits("vector", &vector_hits);

    // 3) Text-only search (lexical/BM25F): vector is intentionally empty.
    let text_hits = engine.search(SearchRequest {
        vector: Vec::new(),
        top_k: 3,
        text_query: Some("rust vectors".to_string()),
        hybrid_alpha: Some(0.0),
        lexical_top_k: Some(20),
        filter: None,
        ef_search: None,
        radius: None,
    })?;
    print_hits("text_only", &text_hits);

    // 4) Hybrid search: combines vector and text relevance.
    let hybrid_hits = engine.search(SearchRequest {
        vector: vec![0.8, 0.2, 0.0],
        top_k: 3,
        text_query: Some("ann graph".to_string()),
        hybrid_alpha: Some(0.65),
        lexical_top_k: Some(20),
        filter: None,
        ef_search: Some(80),
        radius: None,
    })?;
    print_hits("hybrid", &hybrid_hits);

    // 5) Folder-filtered search:
    // only return docs where `folder == "engineering"`.
    let folder_hits = engine.search(SearchRequest {
        vector: vec![0.8, 0.2, 0.0],
        top_k: 3,
        text_query: None,
        hybrid_alpha: None,
        lexical_top_k: None,
        filter: Some(FilterExpression::Eq(
            "folder".to_string(),
            json!("engineering"),
        )),
        ef_search: Some(80),
        radius: None,
    })?;
    print_hits("folder_filter(engineering)", &folder_hits);

    println!("READ(stage={label}): completed examples");

    Ok(())
}

fn print_hits(name: &str, hits: &[SearchHit]) {
    if hits.is_empty() {
        println!("READ(search:{name}): no hits");
        return;
    }

    println!("READ(search:{name}): {} hit(s)", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        let doc_id = hit
            .document
            .get("doc_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let folder = hit
            .document
            .get("folder")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!(
            "  {}. id={} score={:.4} doc_id={} folder={}",
            i + 1,
            hit.id.0,
            hit.score,
            doc_id,
            folder
        );
    }
}
