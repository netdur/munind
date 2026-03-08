use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::MemoryId;
use std::collections::{HashMap, HashSet};

use crate::exact::{ExactSearcher, ScoredHit};
use crate::graph::GraphIndex;
use crate::lookup::VectorLookup;
use crate::search::AnnSearcher;

/// Orchestrates the graph and vector storage together.
pub struct IndexEngine {
    pub graph: GraphIndex,
    vectors: HashMap<MemoryId, Vec<f32>>,
    metric: DistanceMetric,
    ef_construction: usize,
    ef_search: usize,
}

impl IndexEngine {
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            graph: GraphIndex::new(&config.index),
            vectors: HashMap::new(),
            metric: config.index.metric.clone(),
            ef_construction: config.index.ef_construction,
            ef_search: config.index.ef_search,
        }
    }

    pub fn insert(&mut self, id: MemoryId, vector: Vec<f32>) {
        // Replace semantics for duplicate IDs keeps graph/vector storage consistent.
        if self.vectors.insert(id, vector).is_some() {
            self.graph.remove_node(id);
        }

        let (graph, vectors) = (&mut self.graph, &self.vectors);
        let Some(vec_ref) = vectors.get_vector(id) else {
            return;
        };

        graph.insert(
            id,
            vec_ref,
            self.metric.clone(),
            self.ef_construction,
            vectors,
        );
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<ScoredHit> {
        self.search_with_ef(query, top_k, self.ef_search)
    }

    pub fn search_with_ef(&self, query: &[f32], top_k: usize, ef_search: usize) -> Vec<ScoredHit> {
        let best_seed = self.graph.entry_point;
        let searcher = AnnSearcher::new(&self.graph, query, &self.metric, &self.vectors);
        searcher.search(best_seed, top_k, ef_search)
    }

    /// Exact scoring restricted to a candidate ID set; useful for payload-indexed filters.
    pub fn exact_search_filtered(
        &self,
        query: &[f32],
        allowed_ids: &HashSet<MemoryId>,
        top_k: usize,
    ) -> Vec<ScoredHit> {
        if top_k == 0 || allowed_ids.is_empty() {
            return Vec::new();
        }

        let mut searcher = ExactSearcher::new(query, self.metric.clone(), top_k);
        for id in allowed_ids {
            if let Some(vector) = self.vectors.get_vector(*id) {
                searcher.push(*id, vector);
            }
        }

        searcher.take_results()
    }

    pub fn default_ef_search(&self) -> usize {
        self.ef_search
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    pub fn delete(&mut self, id: MemoryId) {
        self.vectors.remove(&id);
        self.graph.remove_node(id);
    }
}
