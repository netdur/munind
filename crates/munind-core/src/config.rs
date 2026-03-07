use serde::{Deserialize, Serialize};

/// High-level configuration for the entire Munind Engine space
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    pub storage: StorageConfig,
    pub index: IndexConfig,
    pub query: QueryConfig,
    pub runtime: RuntimeConfig,
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub path: String,
    pub fsync_enabled: bool,
    pub snapshot_interval_sec: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: "./munind_data".to_string(),
            fsync_enabled: true,
            snapshot_interval_sec: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub metric: DistanceMetric,
    /// Maximum number of connections per node in layers > 0
    pub m: usize,
    /// Maximum number of connections per node in layer 0 (base layer). Generally 2*m
    pub m0: usize,
    /// Level generation multiplier ($m_L$)
    pub ml: f32,
    /// ef construction: size of the dynamic candidate list during insertion
    pub ef_construction: usize,
    /// ef search: size of the dynamic candidate list during search
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ml: 1.0 / 16.0f32.ln(), // Standard heuristic: 1 / ln(M)
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IndexMode {
    GraphOnly,
    GraphAndTree,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    pub default_top_k: usize,
    pub ef_search: usize,
    pub exploration_factor: f32, // NGT epsilon equivalent
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            default_top_k: 10,
            ef_search: 80,
            exploration_factor: 1.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub thread_count: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelemetryConfig {
    pub metrics_enabled: bool,
    pub tracing_enabled: bool,
}
