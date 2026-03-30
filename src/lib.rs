// Internal modules.
pub mod common;
pub mod error;
pub mod graph;
pub mod index;
pub mod mmap_index;
pub mod node;
pub mod object_space;
pub mod primitive_comparator;
pub mod tq;
pub mod tree;

// Clean public API.
pub mod api;
pub mod ffi;

// Legacy re-exports (for existing tests and CLI).
pub use common::{Distance, NgtError, ObjectDistance, ObjectID, SearchOptions, SearchResult};
pub use index::{IdenticalObjectEdgeType, Index, IndexDistanceType, IndexProperty, IndexType};
pub use mmap_index::MmapIndex;
