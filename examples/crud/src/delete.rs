use anyhow::Result;
use munind_api::MunindEngine;
use munind_core::domain::MemoryId;
use munind_core::engine::VectorEngine;

/// DELETE:
/// Removes a record by ID and verifies it no longer exists.
pub fn run(engine: &MunindEngine, id: MemoryId) -> Result<()> {
    engine.remove(id)?;
    println!("DELETE: removed id={}", id.0);

    let exists_after_delete = engine.get_record(id)?.is_some();
    println!("DELETE(check): exists_after_delete={}", exists_after_delete);
    Ok(())
}
