use crate::exact::ScoredHit;
use crate::graph::GraphIndex;
use crate::lookup::VectorLookup;
use crate::vector::calculate_distance;
use munind_core::config::DistanceMetric;
use munind_core::domain::MemoryId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub struct AnnSearcher<'g, 'q, L>
where
    L: VectorLookup,
{
    graph: &'g GraphIndex,
    query: &'q [f32],
    metric: &'g DistanceMetric,
    vector_lookup: &'g L,
}

impl<'g, 'q, L> AnnSearcher<'g, 'q, L>
where
    L: VectorLookup,
{
    pub fn new(
        graph: &'g GraphIndex,
        query: &'q [f32],
        metric: &'g DistanceMetric,
        vector_lookup: &'g L,
    ) -> Self {
        Self {
            graph,
            query,
            metric,
            vector_lookup,
        }
    }

    /// Performs greedy search on a specific layer of the graph
    pub fn search_layer(
        &self,
        entry_point_opt: Option<MemoryId>,
        top_k: usize,
        ef_search: usize,
        layer: u8,
    ) -> Vec<ScoredHit> {
        let entry_point = match entry_point_opt {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        let mut visited = HashSet::new();

        let initial_vec = match self.vector_lookup.get_vector(entry_point) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let dist = calculate_distance(self.metric, self.query, initial_vec);
        let hit = ScoredHit {
            id: entry_point,
            distance: dist,
        };

        // Reverse ordering for min-heap behavior for candidates
        #[derive(Ord, PartialOrd, Eq, PartialEq)]
        struct MinScoredHit(std::cmp::Reverse<ScoredHit>);

        candidates.push(MinScoredHit(std::cmp::Reverse(hit)));
        results.push(hit);
        visited.insert(entry_point);

        while let Some(MinScoredHit(std::cmp::Reverse(current))) = candidates.pop() {
            // Stop if the closest candidate is further than the furthest result
            if let Some(worst_result) = results.peek()
                && current.distance > worst_result.distance
                && results.len() >= ef_search
            {
                break;
            }

            if let Some(neighbors) = self.graph.get_neighbors(current.id, layer) {
                for &neighbor_id in neighbors {
                    if visited.insert(neighbor_id)
                        && let Some(neighbor_vec) = self.vector_lookup.get_vector(neighbor_id)
                    {
                        let dist = calculate_distance(self.metric, self.query, neighbor_vec);
                        let neighbor_hit = ScoredHit {
                            id: neighbor_id,
                            distance: dist,
                        };

                        let should_push = if results.len() < ef_search {
                            true
                        } else {
                            results.peek().map(|h| dist < h.distance).unwrap_or(true)
                        };

                        if should_push {
                            candidates.push(MinScoredHit(std::cmp::Reverse(neighbor_hit)));
                            results.push(neighbor_hit);

                            if results.len() > ef_search {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut final_results = results.into_vec();
        final_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        final_results.truncate(top_k);
        final_results
    }

    /// Performs greedy search on the graph (defaults to base layer 0)
    pub fn search(
        &self,
        entry_point_opt: Option<MemoryId>,
        top_k: usize,
        ef_search: usize,
    ) -> Vec<ScoredHit> {
        self.search_layer(entry_point_opt, top_k, ef_search, 0)
    }
}
