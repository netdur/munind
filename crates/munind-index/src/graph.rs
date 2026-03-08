use crate::lookup::VectorLookup;
use crate::search::AnnSearcher;
use munind_core::config::{DistanceMetric, IndexConfig};
use munind_core::domain::MemoryId;
use std::collections::HashMap;

/// A Hierarchical Navigable Small World (HNSW) graph representation.
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

    /// Generates a random level for a new node assignment.
    fn get_random_level(&self) -> u8 {
        let r: f32 = rand::random::<f32>();
        let r = if r == 0.0 { 0.000001 } else { r };
        let raw = (-r.ln() * self.ml).floor().max(0.0);
        raw.min(u8::MAX as f32) as u8
    }

    pub fn insert<L>(
        &mut self,
        id: MemoryId,
        vector: &[f32],
        metric: DistanceMetric,
        ef_construction: usize,
        vector_lookup: &L,
    ) where
        L: VectorLookup,
    {
        let insert_level = self.get_random_level();

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = insert_level;

            for lc in 0..=insert_level {
                self.layers.entry(lc).or_default().insert(id, Vec::new());
            }
            return;
        }

        let mut current_node = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };

        // Phase 1: Top-down routing from upper layers to insertion neighborhood.
        if insert_level < self.max_layer {
            for lc in (insert_level + 1..=self.max_layer).rev() {
                let searcher = AnnSearcher::new(self, vector, &metric, vector_lookup);
                let candidates = searcher.search_layer(Some(current_node), 1, 1, lc);
                if let Some(closest) = candidates.first() {
                    current_node = closest.id;
                }
            }
        }

        // Phase 2: Insert into all layers from min(insert_level, max_layer) down to 0.
        let connect_top = std::cmp::min(insert_level, self.max_layer);

        for lc in (0..=connect_top).rev() {
            self.layers.entry(lc).or_default().entry(id).or_default();

            let searcher = AnnSearcher::new(self, vector, &metric, vector_lookup);
            let max_edges = if lc == 0 { self.m0 } else { self.m };
            let candidates =
                searcher.search_layer(Some(current_node), max_edges, ef_construction, lc);

            for hit in &candidates {
                if hit.id != id {
                    self.add_edge_internal(id, hit.id, lc, max_edges, &metric, vector_lookup);
                    self.add_edge_internal(hit.id, id, lc, max_edges, &metric, vector_lookup);
                }
            }

            if let Some(closest) = candidates.first() {
                current_node = closest.id;
            }
        }

        if insert_level > self.max_layer {
            self.max_layer = insert_level;
            self.entry_point = Some(id);
            for lc in connect_top + 1..=insert_level {
                self.layers.entry(lc).or_default().insert(id, Vec::new());
            }
        }
    }

    pub fn remove_node(&mut self, id: MemoryId) {
        for layer in self.layers.values_mut() {
            layer.remove(&id);
            for neighbors in layer.values_mut() {
                neighbors.retain(|&n| n != id);
            }
        }

        self.layers.retain(|_, layer| !layer.is_empty());
        self.recompute_entry_point();
    }

    fn recompute_entry_point(&mut self) {
        if self.layers.is_empty() {
            self.max_layer = 0;
            self.entry_point = None;
            return;
        }

        let mut levels: Vec<u8> = self.layers.keys().copied().collect();
        levels.sort_unstable_by(|a, b| b.cmp(a));

        for lc in levels {
            if let Some(layer) = self.layers.get(&lc)
                && let Some((&id, _)) = layer.iter().next()
            {
                self.max_layer = lc;
                self.entry_point = Some(id);
                return;
            }
        }

        self.max_layer = 0;
        self.entry_point = None;
    }

    fn add_edge_internal<L>(
        &mut self,
        from: MemoryId,
        to: MemoryId,
        lc: u8,
        max_edges: usize,
        metric: &DistanceMetric,
        vector_lookup: &L,
    ) where
        L: VectorLookup,
    {
        if let Some(layer) = self.layers.get_mut(&lc)
            && let Some(neighbors) = layer.get_mut(&from)
            && !neighbors.contains(&to)
        {
            neighbors.push(to);
            if neighbors.len() > max_edges {
                *neighbors =
                    Self::select_edges_hnsw(metric, from, neighbors, max_edges, vector_lookup);
            }
        }
    }

    /// Evaluates candidate connections and attempts to prevent clustered connections
    /// by ensuring long-distance links are preserved to satisfy small-world properties.
    pub fn select_edges_hnsw<L>(
        metric: &DistanceMetric,
        base_id: MemoryId,
        candidates: &[MemoryId],
        max_edges: usize,
        vector_lookup: &L,
    ) -> Vec<MemoryId>
    where
        L: VectorLookup,
    {
        let base_vec = match vector_lookup.get_vector(base_id) {
            Some(v) => v,
            None => return candidates.iter().take(max_edges).copied().collect(),
        };

        let mut sorted_candidates: Vec<(MemoryId, f32)> = candidates
            .iter()
            .filter_map(|&id| {
                vector_lookup
                    .get_vector(id)
                    .map(|v| (id, crate::vector::calculate_distance(metric, base_vec, v)))
            })
            .collect();

        sorted_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::with_capacity(max_edges);

        for (candidate_id, dist_to_base) in sorted_candidates {
            if selected.len() >= max_edges {
                break;
            }

            let mut is_closer_to_base_than_selected = true;
            if let Some(candidate_vec) = vector_lookup.get_vector(candidate_id) {
                for &selected_id in &selected {
                    if let Some(selected_vec) = vector_lookup.get_vector(selected_id) {
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
        self.layers
            .get(&layer)
            .and_then(|map| map.get(&id).map(Vec::as_slice))
    }
}
