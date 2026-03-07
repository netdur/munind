use munind_core::config::EngineConfig;
use munind_core::domain::{
    FilterExpression, MemoryId, OptimizeReport, OptimizeRequest, SearchHit, SearchRequest,
};
use munind_core::error::{MunindError, Result};
use munind_core::engine::VectorEngine;
use munind_storage::StorageEngine;
use munind_index::IndexEngine;
use serde_json::Value;
use std::path::Path;
use std::collections::HashMap;

use std::sync::RwLock;

fn json_path_get<'a>(doc: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = doc;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

fn matches_filter(doc: &Value, filter: &FilterExpression) -> bool {
    match filter {
        FilterExpression::Eq(path, expected) => {
            json_path_get(doc, path).map(|actual| actual == expected).unwrap_or(false)
        }
        FilterExpression::And(parts) => parts.iter().all(|f| matches_filter(doc, f)),
    }
}

/// The central Munind database engine wrapping durable storage and graph index
pub struct MunindEngine {
    config: EngineConfig,
    storage: StorageEngine,
    index: RwLock<IndexEngine>,
    // MVP hack to satisfy Rust lifetime constraints for the IndexEngine graph.
    // The graph borrows f32 vectors with a lifetime, but reading bytes from storage returns temporary 
    // owned vectors. We keep them alive here. In a real system the index would manage memory maps.
    memory_cache: RwLock<HashMap<MemoryId, Vec<f32>>>,
}

impl MunindEngine {
    /// Creates a new database with an immutable embedding dimension.
    pub fn create<P: AsRef<Path>>(
        data_dir: P,
        embedding_dimension: usize,
        config: EngineConfig,
    ) -> Result<Self> {
        let path = data_dir.as_ref().to_path_buf();
        let storage = StorageEngine::create(&path, embedding_dimension, config.clone())?;
        let index = IndexEngine::new(&config);
        let memory_cache = HashMap::new();

        Ok(Self {
            config,
            storage,
            index: RwLock::new(index),
            memory_cache: RwLock::new(memory_cache),
        })
    }

    /// Opens an existing database.
    pub fn open<P: AsRef<Path>>(data_dir: P, config: EngineConfig) -> Result<Self> {
        let path = data_dir.as_ref().to_path_buf();

        let manifest_path = path.join("MANIFEST.json");
        if !manifest_path.exists() {
            return Err(MunindError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Database does not exist. Create it first with a fixed embedding dimension.",
            )));
        }
        let storage = StorageEngine::open(&path)?;
        let mut index = IndexEngine::new(&config);
        let mut memory_cache = HashMap::new();

        // 1. Recover storage from WAL/segments
        let ids = storage.get_all_ids();
        
        // 2. Load vectors into memory cache
        for id in &ids {
            if let Ok(Some(vector)) = storage.get_vector(*id) {
                memory_cache.insert(*id, vector);
            }
        }
        
        // 3. Rebuild Graph Index and Seed Tree
        // The index.insert needs a fetcher that returns Option<&[f32]>.
        for id in &ids {
            if let Some(vec) = memory_cache.get(id) {
                let fetcher = |q_id: MemoryId| memory_cache.get(&q_id).map(|v: &Vec<f32>| v.as_slice());
                index.insert(*id, vec, &fetcher);
            }
        }
        
        Ok(Self {
            config,
            storage,
            index: RwLock::new(index),
            memory_cache: RwLock::new(memory_cache),
        })
    }

    pub fn embedding_dimension(&self) -> usize {
        self.storage.manifest.embedding_dimension
    }
}

impl VectorEngine for MunindEngine {
    fn create_database(&self, _embedding_dimension: usize, _config: EngineConfig) -> Result<()> {
        Err(MunindError::Internal(
            "Use MunindEngine::create(path, embedding_dimension, config)".into(),
        ))
    }

    fn insert_json(&self, embedding: Vec<f32>, document: Value) -> Result<MemoryId> {
        let id = self.storage.insert_json(embedding.clone(), document)?;
        {
            let mut cache = self.memory_cache.write().unwrap();
            cache.insert(id, embedding);
        }
        
        // MVP logic for inline graph update
        // The fetcher needs to pull from the current state of the memory cache
        let cache_read = self.memory_cache.read().unwrap();
        let vec_ref = cache_read.get(&id).unwrap();
        
        let mut idx = self.index.write().unwrap();
        let fetcher = |q_id: MemoryId| cache_read.get(&q_id).map(|v: &Vec<f32>| v.as_slice());
        idx.insert(id, vec_ref, &fetcher);
        
        Ok(id)
    }

    fn insert_json_batch(&self, rows: Vec<(Vec<f32>, Value)>) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        for (emb, doc) in rows {
            ids.push(self.insert_json(emb, doc)?);
        }
        Ok(ids)
    }

    fn search(&self, req: SearchRequest) -> Result<Vec<SearchHit>> {
        if req.top_k == 0 {
            return Ok(Vec::new());
        }
        if req.vector.len() != self.embedding_dimension() {
            return Err(MunindError::DimensionMismatch {
                expected: self.embedding_dimension(),
                actual: req.vector.len(),
            });
        }

        let cache = self.memory_cache.read().unwrap();
        let idx = self.index.read().unwrap();
        let fetcher = |id: MemoryId| cache.get(&id).map(|v: &Vec<f32>| v.as_slice());

        let ef = req.ef_search.unwrap_or_else(|| idx.default_ef_search());
        let candidate_k = if req.filter.is_some() || req.radius.is_some() {
            req.top_k
                .saturating_mul(20)
                .max(req.top_k)
                .min(cache.len().max(req.top_k))
        } else {
            req.top_k
        };
        let raw_results = idx.search_with_ef(&req.vector, candidate_k, ef, &fetcher);

        let mut hits = Vec::new();
        for hit in raw_results {
            if let Some(radius) = req.radius
                && hit.distance > radius
            {
                continue;
            }
            if let Ok(Some(doc)) = self.storage.get_document(hit.id) {
                if let Some(filter) = &req.filter
                    && !matches_filter(&doc, filter)
                {
                    continue;
                }
                hits.push(SearchHit {
                    id: hit.id,
                    score: hit.distance, // MVP maps score directly to distance
                    document: doc,
                });
            }
        }

        hits.truncate(req.top_k);
        Ok(hits)
    }

    fn remove(&self, id: MemoryId) -> Result<()> {
        self.storage.remove(id)?;
        {
            let mut cache = self.memory_cache.write().unwrap();
            cache.remove(&id);
        }
        let mut idx = self.index.write().unwrap();
        idx.delete(id);
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(()) // MVP Segment closing handles fsyncs already for WAL
    }

    fn optimize(&self, req: OptimizeRequest) -> Result<OptimizeReport> {
        let storage_report = self.storage.optimize(req.clone())?;
        if !req.repair_graph && !req.force_full_compaction {
            return Ok(storage_report);
        }

        let cache = self.memory_cache.read().unwrap();
        let mut new_index = IndexEngine::new(&self.config);
        for (id, vector) in cache.iter() {
            let fetcher = |q_id: MemoryId| cache.get(&q_id).map(|v: &Vec<f32>| v.as_slice());
            new_index.insert(*id, vector, &fetcher);
        }

        let mut idx = self.index.write().unwrap();
        *idx = new_index;

        Ok(OptimizeReport {
            records_compacted: if req.force_full_compaction {
                cache.len()
            } else {
                storage_report.records_compacted
            },
            space_reclaimed_bytes: storage_report.space_reclaimed_bytes,
            graph_edges_repaired: if req.repair_graph { cache.len() } else { 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MunindEngine;
    use munind_core::config::{DistanceMetric, EngineConfig};
    use munind_core::domain::{FilterExpression, MemoryId, OptimizeRequest, SearchRequest};
    use munind_core::engine::VectorEngine;
    use munind_core::error::MunindError;
    use tempfile::tempdir;

    #[test]
    fn test_create_open_with_fixed_dimension() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let cfg = EngineConfig::default();

        let engine = MunindEngine::create(&db_path, 8, cfg.clone()).unwrap();
        assert_eq!(engine.embedding_dimension(), 8);
        drop(engine);

        let opened = MunindEngine::open(&db_path, cfg).unwrap();
        assert_eq!(opened.embedding_dimension(), 8);
    }

    #[test]
    fn test_open_missing_database_fails() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("missing-db");
        let cfg = EngineConfig::default();

        match MunindEngine::open(&db_path, cfg) {
            Err(MunindError::Io(io_err)) => {
                assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound)
            }
            Err(other) => panic!("unexpected error type: {:?}", other),
            Ok(_) => panic!("expected open to fail for missing database"),
        }
    }

    #[test]
    fn test_insert_enforces_embedding_dimension() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let cfg = EngineConfig::default();
        let engine = MunindEngine::create(&db_path, 8, cfg).unwrap();

        let err = engine
            .insert_json(vec![1.0, 2.0, 3.0], serde_json::json!({"x": 1}))
            .unwrap_err();
        assert!(matches!(
            err,
            MunindError::DimensionMismatch {
                expected: 8,
                actual: 3
            }
        ));
    }

    #[test]
    fn test_remove_removes_from_search_results() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let cfg = EngineConfig::default();
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        let _id1 = engine
            .insert_json(vec![0.0, 0.0, 0.0], serde_json::json!({"name": "a"}))
            .unwrap();
        let id2 = engine
            .insert_json(vec![10.0, 10.0, 10.0], serde_json::json!({"name": "b"}))
            .unwrap();

        engine.remove(id2).unwrap();

        let hits = engine
            .search(SearchRequest {
                vector: vec![10.0, 10.0, 10.0],
                top_k: 3,
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert!(!hits.iter().any(|h| h.id == id2));
    }

    #[test]
    fn test_remove_missing_propagates_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let cfg = EngineConfig::default();
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        let err = engine.remove(MemoryId(404)).unwrap_err();
        assert!(matches!(err, MunindError::NotFound(404)));
    }

    #[test]
    fn test_optimize_rebuild_keeps_search_working() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let cfg = EngineConfig::default();
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        let id1 = engine
            .insert_json(vec![1.0, 0.0, 0.0], serde_json::json!({"name": "n1"}))
            .unwrap();
        let _id2 = engine
            .insert_json(vec![0.0, 1.0, 0.0], serde_json::json!({"name": "n2"}))
            .unwrap();

        let before = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 1,
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert_eq!(before[0].id, id1);

        let report = engine
            .optimize(OptimizeRequest {
                force_full_compaction: false,
                repair_graph: true,
            })
            .unwrap();
        assert!(report.graph_edges_repaired >= 1);

        let after = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 1,
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert_eq!(after[0].id, id1);
    }

    #[test]
    fn test_search_json_eq_and_filter() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::L2;
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        engine
            .insert_json(
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"x": "a", "meta": {"source": "desk"}}),
            )
            .unwrap();
        engine
            .insert_json(
                vec![1.0, 0.1, 0.0],
                serde_json::json!({"x": "b", "meta": {"source": "desk"}}),
            )
            .unwrap();
        engine
            .insert_json(
                vec![1.0, 0.0, 0.1],
                serde_json::json!({"x": "a", "meta": {"source": "mobile"}}),
            )
            .unwrap();

        let hits = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 10,
                filter: Some(FilterExpression::And(vec![
                    FilterExpression::Eq("x".to_string(), serde_json::json!("a")),
                    FilterExpression::Eq(
                        "meta.source".to_string(),
                        serde_json::json!("desk"),
                    ),
                ])),
                ef_search: None,
                radius: None,
            })
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document["x"], serde_json::json!("a"));
        assert_eq!(hits[0].document["meta"]["source"], serde_json::json!("desk"));
    }

    #[test]
    fn test_search_respects_radius_and_query_dimension() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::L2;
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        engine
            .insert_json(vec![0.0, 0.0, 0.0], serde_json::json!({"name": "near"}))
            .unwrap();
        engine
            .insert_json(vec![2.0, 0.0, 0.0], serde_json::json!({"name": "far"}))
            .unwrap();

        let hits = engine
            .search(SearchRequest {
                vector: vec![0.0, 0.0, 0.0],
                top_k: 10,
                filter: None,
                ef_search: Some(4),
                radius: Some(1.0),
            })
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document["name"], serde_json::json!("near"));

        let mismatch = engine.search(SearchRequest {
            vector: vec![0.0, 0.0],
            top_k: 10,
            filter: None,
            ef_search: None,
            radius: None,
        });
        assert!(matches!(
            mismatch,
            Err(MunindError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }
}
