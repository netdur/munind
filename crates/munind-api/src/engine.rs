use crate::lexical::LexicalIndex;
use crate::payload_index::{PayloadFilterPlan, PayloadIndex};
use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::{
    FilterExpression, MemoryId, OptimizeReport, OptimizeRequest, SearchHit, SearchRequest,
};
use munind_core::engine::VectorEngine;
use munind_core::error::{MunindError, Result};
use munind_index::IndexEngine;
use munind_storage::StorageEngine;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

const FILTERED_EXACT_SCAN_THRESHOLD: usize = 50_000;

fn json_path_get<'a>(doc: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = doc;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

fn matches_filter(doc: &Value, filter: &FilterExpression) -> bool {
    match filter {
        FilterExpression::Eq(path, expected) => json_path_get(doc, path)
            .map(|actual| actual == expected)
            .unwrap_or(false),
        FilterExpression::And(parts) => parts.iter().all(|f| matches_filter(doc, f)),
    }
}

fn distance_to_score(metric: &DistanceMetric, distance: f32) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::L2 => -distance,
        DistanceMetric::InnerProduct => -distance,
    }
}

fn normalize_scores(scores: &HashMap<MemoryId, f32>) -> HashMap<MemoryId, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for score in scores.values() {
        min = min.min(*score);
        max = max.max(*score);
    }

    let span = max - min;
    if span.abs() < 1e-12 {
        return scores.keys().map(|id| (*id, 1.0)).collect();
    }

    scores
        .iter()
        .map(|(id, score)| (*id, (*score - min) / span))
        .collect()
}

/// The central Munind database engine wrapping durable storage, graph index, lexical index,
/// and payload indexes for metadata filters.
pub struct MunindEngine {
    config: EngineConfig,
    storage: StorageEngine,
    index: RwLock<IndexEngine>,
    lexical: RwLock<LexicalIndex>,
    payload: RwLock<PayloadIndex>,
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
        let lexical = LexicalIndex::new();
        let payload = PayloadIndex::new();

        Ok(Self {
            config,
            storage,
            index: RwLock::new(index),
            lexical: RwLock::new(lexical),
            payload: RwLock::new(payload),
        })
    }

    fn build_indexes_from_storage(
        storage: &StorageEngine,
        config: &EngineConfig,
    ) -> Result<(IndexEngine, LexicalIndex, PayloadIndex)> {
        let mut index = IndexEngine::new(config);
        let mut lexical = LexicalIndex::new();
        let mut payload = PayloadIndex::new();

        for id in storage.get_all_ids()? {
            if let Some(vector) = storage.get_vector(id)? {
                index.insert(id, vector);
            }
            if let Some(doc) = storage.get_document(id)? {
                lexical.insert(id, &doc);
                payload.insert(id, &doc);
            }
        }

        Ok((index, lexical, payload))
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
        let (index, lexical, payload) = Self::build_indexes_from_storage(&storage, &config)?;

        Ok(Self {
            config,
            storage,
            index: RwLock::new(index),
            lexical: RwLock::new(lexical),
            payload: RwLock::new(payload),
        })
    }

    pub fn embedding_dimension(&self) -> usize {
        self.storage.manifest.embedding_dimension
    }

    fn index_read(&self) -> Result<RwLockReadGuard<'_, IndexEngine>> {
        self.index
            .read()
            .map_err(|_| MunindError::Internal("index read lock poisoned".to_string()))
    }

    fn index_write(&self) -> Result<RwLockWriteGuard<'_, IndexEngine>> {
        self.index
            .write()
            .map_err(|_| MunindError::Internal("index write lock poisoned".to_string()))
    }

    fn lexical_read(&self) -> Result<RwLockReadGuard<'_, LexicalIndex>> {
        self.lexical
            .read()
            .map_err(|_| MunindError::Internal("lexical read lock poisoned".to_string()))
    }

    fn lexical_write(&self) -> Result<RwLockWriteGuard<'_, LexicalIndex>> {
        self.lexical
            .write()
            .map_err(|_| MunindError::Internal("lexical write lock poisoned".to_string()))
    }

    fn payload_read(&self) -> Result<RwLockReadGuard<'_, PayloadIndex>> {
        self.payload
            .read()
            .map_err(|_| MunindError::Internal("payload read lock poisoned".to_string()))
    }

    fn payload_write(&self) -> Result<RwLockWriteGuard<'_, PayloadIndex>> {
        self.payload
            .write()
            .map_err(|_| MunindError::Internal("payload write lock poisoned".to_string()))
    }

    fn rebuild_indexes_from_storage(&self) -> Result<(IndexEngine, LexicalIndex, PayloadIndex)> {
        Self::build_indexes_from_storage(&self.storage, &self.config)
    }
}

impl VectorEngine for MunindEngine {
    fn create_database(&self, _embedding_dimension: usize, _config: EngineConfig) -> Result<()> {
        Err(MunindError::Internal(
            "Use MunindEngine::create(path, embedding_dimension, config)".into(),
        ))
    }

    fn insert_json(&self, embedding: Vec<f32>, document: Value) -> Result<MemoryId> {
        let doc_for_indexes = document.clone();
        let id = self.storage.insert_json(embedding.clone(), document)?;

        let mut idx = self.index_write()?;
        idx.insert(id, embedding);
        drop(idx);

        let mut lexical = self.lexical_write()?;
        lexical.insert(id, &doc_for_indexes);
        drop(lexical);

        let mut payload = self.payload_write()?;
        payload.insert(id, &doc_for_indexes);

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

        let filter_plan: Option<PayloadFilterPlan> = if let Some(filter) = req.filter.as_ref() {
            let payload = self.payload_read()?;
            payload.plan_filter(filter)
        } else {
            None
        };

        if let Some(plan) = &filter_plan
            && plan.candidate_ids.is_empty()
        {
            return Ok(Vec::new());
        }

        let filter_candidate_ids = filter_plan.as_ref().map(|p| &p.candidate_ids);
        let filter_fully_indexed = filter_plan
            .as_ref()
            .map(|p| p.fully_indexed)
            .unwrap_or(false);

        let hybrid_query = req
            .text_query
            .as_ref()
            .map(|q| q.trim())
            .filter(|q| !q.is_empty());
        let hybrid_enabled = hybrid_query.is_some();

        let idx = self.index_read()?;
        let ef = req.ef_search.unwrap_or_else(|| idx.default_ef_search());
        let indexed_count = idx.vector_count();

        let needs_broad_candidates = req.filter.is_some() || req.radius.is_some() || hybrid_enabled;
        let mut candidate_k = if needs_broad_candidates {
            req.top_k
                .saturating_mul(20)
                .max(req.top_k)
                .min(indexed_count.max(req.top_k))
        } else {
            req.top_k
        };

        if let Some(ids) = filter_candidate_ids {
            candidate_k = candidate_k.min(ids.len().max(req.top_k));
        }

        let raw_results = if let Some(ids) = filter_candidate_ids {
            if ids.len() <= FILTERED_EXACT_SCAN_THRESHOLD {
                idx.exact_search_filtered(&req.vector, ids, candidate_k.max(req.top_k))
            } else {
                idx.search_with_ef(&req.vector, candidate_k.max(req.top_k), ef)
                    .into_iter()
                    .filter(|hit| ids.contains(&hit.id))
                    .collect()
            }
        } else {
            idx.search_with_ef(&req.vector, candidate_k, ef)
        };
        drop(idx);

        let mut vector_distances: HashMap<MemoryId, f32> = HashMap::new();
        let mut vector_scores: HashMap<MemoryId, f32> = HashMap::new();
        for hit in raw_results {
            vector_distances.insert(hit.id, hit.distance);
            vector_scores.insert(
                hit.id,
                distance_to_score(&self.config.index.metric, hit.distance),
            );
        }

        let mut lexical_scores: HashMap<MemoryId, f32> = HashMap::new();
        if let Some(query) = hybrid_query {
            let lexical_k = req.lexical_top_k.unwrap_or(candidate_k).max(req.top_k);
            let lexical = self.lexical_read()?;

            let lexical_hits = if let Some(ids) = filter_candidate_ids {
                lexical.search_filtered(query, lexical_k, ids)
            } else {
                lexical.search(query, lexical_k)
            };

            for (id, score) in lexical_hits {
                lexical_scores.insert(id, score);
            }
        }

        let alpha = req.hybrid_alpha.unwrap_or(0.65).clamp(0.0, 1.0);
        let vector_norm = if hybrid_enabled {
            normalize_scores(&vector_scores)
        } else {
            HashMap::new()
        };
        let lexical_norm = if hybrid_enabled {
            normalize_scores(&lexical_scores)
        } else {
            HashMap::new()
        };

        let mut candidate_ids: HashSet<MemoryId> = HashSet::new();
        candidate_ids.extend(vector_scores.keys().copied());
        candidate_ids.extend(lexical_scores.keys().copied());

        let mut hits = Vec::new();
        for id in candidate_ids {
            if let Some(ids) = filter_candidate_ids
                && !ids.contains(&id)
            {
                continue;
            }

            if let Some(radius) = req.radius {
                let Some(distance) = vector_distances.get(&id) else {
                    continue;
                };
                if *distance > radius {
                    continue;
                }
            }

            let Some(doc) = self.storage.get_document(id)? else {
                continue;
            };

            if let Some(filter) = &req.filter
                && !filter_fully_indexed
                && !matches_filter(&doc, filter)
            {
                continue;
            }

            let score = if hybrid_enabled {
                let v = vector_norm.get(&id).copied().unwrap_or(0.0);
                let l = lexical_norm.get(&id).copied().unwrap_or(0.0);
                alpha * v + (1.0 - alpha) * l
            } else {
                vector_scores
                    .get(&id)
                    .copied()
                    .or_else(|| lexical_scores.get(&id).copied())
                    .unwrap_or(0.0)
            };

            hits.push(SearchHit {
                id,
                score,
                document: doc,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.0.cmp(&b.id.0))
        });
        hits.truncate(req.top_k);
        Ok(hits)
    }

    fn remove(&self, id: MemoryId) -> Result<()> {
        self.storage.remove(id)?;

        let mut idx = self.index_write()?;
        idx.delete(id);
        drop(idx);

        let mut lexical = self.lexical_write()?;
        lexical.remove(id);
        drop(lexical);

        let mut payload = self.payload_write()?;
        payload.remove(id);

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.storage.flush()
    }

    fn optimize(&self, req: OptimizeRequest) -> Result<OptimizeReport> {
        let storage_report = self.storage.optimize(req.clone())?;
        if !req.repair_graph && !req.force_full_compaction {
            return Ok(storage_report);
        }

        let (new_index, new_lexical, new_payload) = self.rebuild_indexes_from_storage()?;
        let rebuilt_records = new_index.vector_count();

        let mut idx = self.index_write()?;
        *idx = new_index;
        drop(idx);

        let mut lexical = self.lexical_write()?;
        *lexical = new_lexical;
        drop(lexical);

        let mut payload = self.payload_write()?;
        *payload = new_payload;

        Ok(OptimizeReport {
            records_compacted: if req.force_full_compaction {
                rebuilt_records
            } else {
                storage_report.records_compacted
            },
            space_reclaimed_bytes: storage_report.space_reclaimed_bytes,
            graph_edges_repaired: if req.repair_graph { rebuilt_records } else { 0 },
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
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
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
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
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
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
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
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: Some(FilterExpression::And(vec![
                    FilterExpression::Eq("x".to_string(), serde_json::json!("a")),
                    FilterExpression::Eq("meta.source".to_string(), serde_json::json!("desk")),
                ])),
                ef_search: None,
                radius: None,
            })
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document["x"], serde_json::json!("a"));
        assert_eq!(
            hits[0].document["meta"]["source"],
            serde_json::json!("desk")
        );
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
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
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
            text_query: None,
            hybrid_alpha: None,
            lexical_top_k: None,
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

    #[test]
    fn test_hybrid_search_promotes_lexical_match() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::Cosine;
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        let id_vector = engine
            .insert_json(
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"title": "noise", "text": "nothing useful"}),
            )
            .unwrap();

        let id_lexical = engine
            .insert_json(
                vec![0.0, 1.0, 0.0],
                serde_json::json!({"title": "apple guide", "tags": ["fruit", "apple"], "text": "orchard notes"}),
            )
            .unwrap();

        let vector_only = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 1,
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert_eq!(vector_only[0].id, id_vector);

        let hybrid = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 1,
                text_query: Some("apple".to_string()),
                hybrid_alpha: Some(0.2),
                lexical_top_k: Some(10),
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert_eq!(hybrid[0].id, id_lexical);
    }

    #[test]
    fn test_payload_indexed_doc_id_filter() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::Cosine;
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        engine
            .insert_json(
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"doc_id": "doc-a", "source": "desk", "text": "a"}),
            )
            .unwrap();
        engine
            .insert_json(
                vec![1.0, 0.1, 0.0],
                serde_json::json!({"doc_id": "doc-b", "source": "desk", "text": "b"}),
            )
            .unwrap();

        let hits = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 10,
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: Some(FilterExpression::Eq(
                    "doc_id".to_string(),
                    serde_json::json!("doc-b"),
                )),
                ef_search: None,
                radius: None,
            })
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document["doc_id"], serde_json::json!("doc-b"));
    }

    #[test]
    fn test_payload_partial_plan_with_fallback_filter_eval() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let mut cfg = EngineConfig::default();
        cfg.index.metric = DistanceMetric::Cosine;
        let engine = MunindEngine::create(&db_path, 3, cfg).unwrap();

        engine
            .insert_json(
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"source": "desk", "x": "keep", "text": "a"}),
            )
            .unwrap();
        engine
            .insert_json(
                vec![0.9, 0.1, 0.0],
                serde_json::json!({"source": "desk", "x": "drop", "text": "b"}),
            )
            .unwrap();

        let hits = engine
            .search(SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: 10,
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: Some(FilterExpression::And(vec![
                    FilterExpression::Eq("source".to_string(), serde_json::json!("desk")),
                    FilterExpression::Eq("x".to_string(), serde_json::json!("keep")),
                ])),
                ef_search: None,
                radius: None,
            })
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document["x"], serde_json::json!("keep"));
    }
}
