use serde::{Deserialize, Serialize};

/// The unique identifier assigned to each inserted memory chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub u64);

/// Request parameters for a search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    // Vector query. May be empty for lexical-only search when `text_query` is non-empty.
    pub vector: Vec<f32>,
    pub top_k: usize,
    /// Optional lexical query used for hybrid search (BM25F + vector fusion).
    pub text_query: Option<String>,
    /// Optional hybrid weight in [0,1]. 1.0 means vector-only, 0.0 means lexical-only.
    pub hybrid_alpha: Option<f32>,
    /// Optional lexical candidate count for BM25F retrieval before fusion.
    pub lexical_top_k: Option<usize>,
    /// Additional search filters, e.g. exact JSON equality filters depending on the engine's capabilities.
    pub filter: Option<FilterExpression>,
    pub ef_search: Option<usize>,
    pub radius: Option<f32>,
}

/// Simple filter expressions, focusing heavily on equality first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    /// Field path, e.g. "metadata.source", matches value exactly.
    Eq(String, serde_json::Value),
    /// Logical AND combining multiple conditions.
    And(Vec<FilterExpression>),
}

/// A search hit result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: MemoryId,
    pub score: f32, // Distance/Similarity or fused relevance
    pub document: serde_json::Value,
}

/// Options to guide the optimization/compaction runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizeRequest {
    pub force_full_compaction: bool,
    pub repair_graph: bool,
}

/// Report after optimization completes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizeReport {
    pub records_compacted: usize,
    pub space_reclaimed_bytes: u64,
    pub graph_edges_repaired: usize,
}
