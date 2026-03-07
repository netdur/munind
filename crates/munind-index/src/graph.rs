use std::collections::HashMap;
use munind_core::domain::MemoryId;
use munind_core::config::{DistanceMetric, IndexConfig};
use crate::search::AnnSearcher;

/// A Hierarchical Navigable Small World (HNSW) graph representation
pub struct GraphIndex {
    /// maps layer -> (map node_id -> neighbors)
    pub layers: HashMap<u8, HashMap<MemoryId, Vec<MemoryId>>>,
    pub max_layer: u8,
    pub entry_point: Option<MemoryId>,
    pub m: usize,
    pub m0: usize,
    pub ml: f32,
}

impl GraphIndex {
    pub fn new(config: &IndexConfig) -> Self {
        Self {
            layers: HashMap::new(),
            max_layer: 0,
            entry_point: None,
            m: config.m,
            m0: config.m0,
            ml: config.ml,
        }
    }

    /// Generates a random level for a new node assignment
    fn get_random_level(&self) -> u8 {
        let r: f32 = rand::random::<f32>();
        // Avoid ln(0) exactly
        let r = if r == 0.0 { 0.000001 } else { r };
        (-r.ln() * self.ml).floor() as u8
    }

    pub fn insert<'v, F>(&mut self, id: MemoryId, vector: &'v [f32], metric: DistanceMetric, ef_construction: usize, vector_fetcher: F) 
    where F: Fn(MemoryId) -> Option<&'v [f32]>
    {
        let insert_level = self.get_random_level();
        
        let entry_point = self.entry_point;

        // If graph is empty
        if entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = insert_level;
            
            for lc in 0..=insert_level {
                self.layers.entry(lc).or_default().insert(id, Vec::new());
            }
            return;
        }

        let mut current_node = entry_point.unwrap();

        // Phase 1: Top-down fast routing (evaluate single closest neighbor down to Layer L+1)
        for lc in (insert_level + 1..=self.max_layer).rev() {
            let searcher = AnnSearcher::new(self, vector, &metric, &vector_fetcher);
            // Search top-layer downwards with ef=1 to find the closest node in that sparse layer
            let candidates = searcher.search_layer(Some(current_node), 1, 1, lc);
            if let Some(closest) = candidates.first() {
                current_node = closest.id;
            }
        }

        // Phase 2: Insert into all layers from min(insert_level, max_layer) down to 0
        let connect_top = std::cmp::min(insert_level, self.max_layer);
        
        for lc in (0..=connect_top).rev() {
            // Allocate node
            self.layers.entry(lc).or_default().entry(id).or_default();

            let searcher = AnnSearcher::new(self, vector, &metric, &vector_fetcher);
            
            // Limit connections: layer 0 gets m0, higher layers get m
            let max_edges = if lc == 0 { self.m0 } else { self.m };
            
            // Search layer for neighbors to connect
            let candidates = searcher.search_layer(Some(current_node), max_edges, ef_construction, lc);
            
            for hit in &candidates {
                if hit.id != id {
                    self.add_edge_internal(id, hit.id, lc, max_edges, &metric, &vector_fetcher);
                    self.add_edge_internal(hit.id, id, lc, max_edges, &metric, &vector_fetcher);
                }
            }

            // The closest node becomes the entry point for the layer below
            if let Some(closest) = candidates.first() {
                current_node = closest.id;
            }
        }

        // Update entry point if this node reached higher than previous max
        if insert_level > self.max_layer {
            self.max_layer = insert_level;
            self.entry_point = Some(id);
            
            // Pre-allocate upper layers
            for lc in connect_top + 1..=insert_level {
                self.layers.entry(lc).or_default().insert(id, Vec::new());
            }
        }
    }

    pub fn remove_node(&mut self, id: MemoryId) {
        for map in self.layers.values_mut() {
            map.remove(&id);
            for neighbors in map.values_mut() {
                neighbors.retain(|&n| n != id);
            }
        }
        
        if self.entry_point == Some(id) {
            // Very naive entry point re-assignment for MVP
            self.entry_point = self.layers.get(&0).and_then(|l0| l0.keys().next().copied());
            // In a real implementation max_layer must be re-calculated based on what remains
        }
    }

    fn add_edge_internal<'a, F>(&mut self, from: MemoryId, to: MemoryId, lc: u8, max_edges: usize, metric: &DistanceMetric, vector_fetcher: &F) 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        if let Some(layer) = self.layers.get_mut(&lc)
            && let Some(neighbors) = layer.get_mut(&from)
            && !neighbors.contains(&to)
        {
            neighbors.push(to);

            // HNSW edge selection heuristic
            if neighbors.len() > max_edges {
                *neighbors = Self::select_edges_hnsw(metric, from, neighbors, max_edges, vector_fetcher);
            }
        }
    }

    /// Evaluates candidate connections and attempts to prevent clustered connections
    /// by ensuring long-distance links are preserved to satisfy small-world properties.
    pub fn select_edges_hnsw<'a, F>(
        metric: &DistanceMetric,
        base_id: MemoryId,
        candidates: &[MemoryId],
        max_edges: usize,
        vector_fetcher: &F,
    ) -> Vec<MemoryId>
    where
        F: Fn(MemoryId) -> Option<&'a [f32]>,
    {
        let base_vec = match vector_fetcher(base_id) {
            Some(v) => v,
            None => return candidates.iter().take(max_edges).copied().collect(),
        };

        // Tuple of (candidate_id, dist_to_base)
        let mut sorted_candidates: Vec<(MemoryId, f32)> = candidates
            .iter()
            .filter_map(|&id| {
                vector_fetcher(id).map(|v| (id, crate::vector::calculate_distance(metric, base_vec, v)))
            })
            .collect();

        sorted_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::with_capacity(max_edges);

        for (candidate_id, dist_to_base) in sorted_candidates {
            if selected.len() >= max_edges {
                break;
            }

            let mut is_closer_to_base_than_selected = true;
            if let Some(candidate_vec) = vector_fetcher(candidate_id) {
                for &selected_id in &selected {
                    if let Some(selected_vec) = vector_fetcher(selected_id) {
                        let dist_candidate_to_selected =
                            crate::vector::calculate_distance(metric, candidate_vec, selected_vec);
                        if dist_candidate_to_selected < dist_to_base {
                            is_closer_to_base_than_selected = false;
                            break;
                        }
                    }
                }
            }

            if is_closer_to_base_than_selected {
                selected.push(candidate_id);
            }
        }

        selected
    }

    pub fn get_neighbors(&self, id: MemoryId, layer: u8) -> Option<&[MemoryId]> {
        self.layers.get(&layer).and_then(|map| map.get(&id).map(|v| v.as_slice()))
    }
}
