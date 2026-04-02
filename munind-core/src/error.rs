/// munind error types.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid object id: {0}")]
    InvalidId(u32),

    #[error("object id {0} not found")]
    NotFound(u32),

    #[error("index is empty")]
    EmptyIndex,

    #[error("dimension must be > 0")]
    ZeroDimension,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Internal(String),
}

/// Convert legacy String errors.
impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Internal(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Internal(s.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
