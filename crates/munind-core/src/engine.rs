use crate::{
    config::EngineConfig,
    domain::{MemoryId, OptimizeReport, OptimizeRequest, SearchHit, SearchRequest},
    error::Result,
};
use serde_json::Value;

pub trait VectorEngine {
    fn create_database(&self, embedding_dimension: usize, config: EngineConfig) -> Result<()>;
    fn insert_json(&self, embedding: Vec<f32>, document: Value) -> Result<MemoryId>;
    fn insert_json_batch(&self, rows: Vec<(Vec<f32>, Value)>) -> Result<Vec<MemoryId>>;
    fn search(&self, query: SearchRequest) -> Result<Vec<SearchHit>>;
    fn remove(&self, id: MemoryId) -> Result<()>;
    fn flush(&self) -> Result<()>;
    fn optimize(&self, req: OptimizeRequest) -> Result<OptimizeReport>;
}
