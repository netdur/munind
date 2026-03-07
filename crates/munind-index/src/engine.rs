use munind_core::domain::MemoryId;
use munind_core::config::{EngineConfig, DistanceMetric};

use crate::graph::GraphIndex;
use crate::search::AnnSearcher;

/// Orchestrates the graph and seed indices together
pub struct IndexEngine {
    pub graph: GraphIndex,
    metric: DistanceMetric,
    ef_construction: usize,
    ef_search: usize,
}

impl IndexEngine {
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            graph: GraphIndex::new(&config.index),
            metric: config.index.metric.clone(),
            ef_construction: config.index.ef_construction,
            ef_search: config.index.ef_search,
        }
    }

    pub fn insert<'a, F>(&mut self, id: MemoryId, vector: &'a [f32], vector_fetcher: &F) 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        self.graph.insert(id, vector, self.metric.clone(), self.ef_construction, vector_fetcher);
    }

    pub fn search<'a, F>(&self, query: &'a [f32], top_k: usize, vector_fetcher: &F) -> Vec<crate::exact::ScoredHit> 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        self.search_with_ef(query, top_k, self.ef_search, vector_fetcher)
    }

    pub fn search_with_ef<'a, F>(
        &self,
        query: &'a [f32],
        top_k: usize,
        ef_search: usize,
        vector_fetcher: &F,
    ) -> Vec<crate::exact::ScoredHit>
    where
        F: Fn(MemoryId) -> Option<&'a [f32]>,
    {
        // MVP seed strategy: graph entry point.
        let best_seed = self.graph.entry_point;

        let searcher = AnnSearcher::new(&self.graph, query, &self.metric, vector_fetcher);
        searcher.search(best_seed, top_k, ef_search)
    }

    pub fn default_ef_search(&self) -> usize {
        self.ef_search
    }

    pub fn delete(&mut self, id: MemoryId) {
        self.graph.remove_node(id);
    }


}
