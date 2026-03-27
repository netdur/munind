pub type ObjectID = u32;
pub type Distance = f32;

// If we need float16, we can add it later using the `half` crate.
// For now, float is enough.

#[derive(Debug, Clone)]
pub struct Exception {
    pub message: String,
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Exception {}

// CompactVector in C++ NGT is used as a memory efficient vector.
// We'll mimic it as needed, but standard Vec is a good starting point.
