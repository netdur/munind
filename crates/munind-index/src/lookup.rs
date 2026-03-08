use munind_core::domain::MemoryId;
use std::collections::HashMap;

/// Read-only vector access abstraction used by graph/search components.
pub trait VectorLookup {
    fn get_vector(&self, id: MemoryId) -> Option<&[f32]>;
}

impl VectorLookup for HashMap<MemoryId, Vec<f32>> {
    fn get_vector(&self, id: MemoryId) -> Option<&[f32]> {
        self.get(&id).map(Vec::as_slice)
    }
}
