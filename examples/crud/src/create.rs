use anyhow::Result;
use munind_api::MunindEngine;
use munind_core::domain::MemoryId;
use munind_core::engine::VectorEngine;
use serde_json::json;

/// CREATE:
/// Inserts a new `(embedding, document)` record and returns its generated ID.
pub fn run(engine: &MunindEngine) -> Result<MemoryId> {
    // Any fixed-size vector works as long as it matches DB embedding dimension.
    // This is the record we will update/delete later in the flow.
    let embedding = vec![1.0_f32, 0.0, 0.0];

    // Document is arbitrary JSON payload.
    let document = json!({
        "doc_id": "example-1",
        "title": "CRUD Example",
        "text": "created from examples/crud/create.rs",
        "source": "example",
        "folder": "engineering"
    });

    let id = engine.insert_json(embedding, document)?;
    println!("CREATE: inserted id={}", id.0);

    // Seed extra records so read examples have multiple candidates.
    let _id2 = engine.insert_json(
        vec![0.7_f32, 0.3, 0.0],
        json!({
            "doc_id": "example-2",
            "title": "Rust Vector Notes",
            "text": "text-search example: rust vectors and ANN graph",
            "source": "example",
            "folder": "engineering"
        }),
    )?;

    let _id3 = engine.insert_json(
        vec![0.0_f32, 0.2, 0.8],
        json!({
            "doc_id": "example-3",
            "title": "Personal Notes",
            "text": "shopping list and weekend plans",
            "source": "example",
            "folder": "personal"
        }),
    )?;

    Ok(id)
}
