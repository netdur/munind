use thiserror::Error;

#[derive(Error, Debug)]
pub enum MunindError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization/Deserialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Database corrupted: {0}")]
    Corruption(String),

    #[error("Memory ID not found: {0}")]
    NotFound(u64),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Internal engine error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MunindError>;
