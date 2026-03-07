use std::collections::HashMap;
use munind_core::domain::MemoryId;

/// In-memory tracker of record locations within segments.
#[derive(Debug, Clone)]
pub struct RecordLocation {
    pub vector_offset: u64,
    pub json_offset: u64,
    pub tombstoned: bool,
}

/// Allocates IDs and tracks segment locations.
pub struct IdAllocator {
    next_id: u64,
    locations: HashMap<MemoryId, RecordLocation>,
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl IdAllocator {
    pub fn new() -> Self {
        Self {
            next_id: 1, // 1-based IDs like NGT
            locations: HashMap::new(),
        }
    }

    /// Allocates a new MemoryId.
    pub fn allocate(&mut self) -> MemoryId {
        let id = MemoryId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Peeks at the next available ID without allocating it.
    pub fn next_id(&self) -> MemoryId {
        MemoryId(self.next_id)
    }

    /// Sets the physical offset for an ID in the segment files.
    pub fn set_location(&mut self, id: MemoryId, vector_offset: u64, json_offset: u64) {
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }
        self.locations.insert(id, RecordLocation {
            vector_offset,
            json_offset,
            tombstoned: false,
        });
    }

    /// Retrieves the location if it exists and is not tombstoned.
    pub fn get_location(&self, id: MemoryId) -> Option<&RecordLocation> {
        match self.locations.get(&id) {
            Some(loc) if !loc.tombstoned => Some(loc),
            _ => None,
        }
    }

    /// Marks a record as deleted (tombstone).
    pub fn tombstone(&mut self, id: MemoryId) -> bool {
        if let Some(loc) = self.locations.get_mut(&id)
            && !loc.tombstoned
        {
            loc.tombstoned = true;
            return true;
        }
        false
    }
    
    pub fn len(&self) -> usize {
        self.locations.iter().filter(|(_, v)| !v.tombstoned).count()
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
