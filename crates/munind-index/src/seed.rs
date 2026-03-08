use crate::lookup::VectorLookup;
use crate::vector::calculate_distance;
use munind_core::config::DistanceMetric;
use munind_core::domain::MemoryId;
use rand::prelude::IndexedRandom;
use std::cmp::Ordering;

pub struct VpNode {
    pub vantage_point: MemoryId,
    pub radius: f32,
    pub inside: Option<Box<VpNode>>,
    pub outside: Option<Box<VpNode>>,
}

pub struct VpTree {
    pub root: Option<Box<VpNode>>,
}

impl Default for VpTree {
    fn default() -> Self {
        Self::new()
    }
}

impl VpTree {
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Batch builds the tree recursively from a set of initial nodes.
    pub fn build<L>(items: Vec<MemoryId>, metric: DistanceMetric, vector_lookup: &L) -> Self
    where
        L: VectorLookup,
    {
        Self {
            root: Self::build_node(items, &metric, vector_lookup),
        }
    }

    fn build_node<L>(
        mut items: Vec<MemoryId>,
        metric: &DistanceMetric,
        vector_lookup: &L,
    ) -> Option<Box<VpNode>>
    where
        L: VectorLookup,
    {
        if items.is_empty() {
            return None;
        }

        // Pick a random vantage point.
        let vp_idx = {
            let mut rng = rand::rng();
            let indices: Vec<usize> = (0..items.len()).collect();
            match indices.choose(&mut rng) {
                Some(idx) => *idx,
                None => 0,
            }
        };
        let vantage_point = items.swap_remove(vp_idx);

        if items.is_empty() {
            return Some(Box::new(VpNode {
                vantage_point,
                radius: 0.0,
                inside: None,
                outside: None,
            }));
        }

        let vp_vec = vector_lookup.get_vector(vantage_point)?;

        let mut distances: Vec<(MemoryId, f32)> = items
            .into_iter()
            .filter_map(|id| {
                vector_lookup
                    .get_vector(id)
                    .map(|v| (id, calculate_distance(metric, vp_vec, v)))
            })
            .collect();

        if distances.is_empty() {
            return Some(Box::new(VpNode {
                vantage_point,
                radius: 0.0,
                inside: None,
                outside: None,
            }));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let median_idx = distances.len() / 2;
        let median_radius = distances[median_idx].1;

        let mut inside_ids = Vec::with_capacity(median_idx);
        let mut outside_ids = Vec::with_capacity(distances.len() - median_idx);

        for (id, dist) in distances {
            if dist <= median_radius {
                inside_ids.push(id);
            } else {
                outside_ids.push(id);
            }
        }

        Some(Box::new(VpNode {
            vantage_point,
            radius: median_radius,
            inside: Self::build_node(inside_ids, metric, vector_lookup),
            outside: Self::build_node(outside_ids, metric, vector_lookup),
        }))
    }

    /// Traverses the tree to find initial candidates (seeds).
    pub fn search_seeds<L>(
        &self,
        query: &[f32],
        max_seeds: usize,
        metric: DistanceMetric,
        vector_lookup: &L,
    ) -> Vec<MemoryId>
    where
        L: VectorLookup,
    {
        let mut seeds = Vec::new();
        self.search_node(
            &self.root,
            query,
            max_seeds,
            &metric,
            vector_lookup,
            &mut seeds,
        );

        seeds.sort_by(|&a, &b| {
            let dist_a = vector_lookup
                .get_vector(a)
                .map(|v| calculate_distance(&metric, query, v))
                .unwrap_or(f32::MAX);
            let dist_b = vector_lookup
                .get_vector(b)
                .map(|v| calculate_distance(&metric, query, v))
                .unwrap_or(f32::MAX);
            dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
        });

        seeds
    }

    fn search_node<L>(
        &self,
        node_opt: &Option<Box<VpNode>>,
        query: &[f32],
        max_seeds: usize,
        metric: &DistanceMetric,
        vector_lookup: &L,
        seeds: &mut Vec<MemoryId>,
    ) where
        L: VectorLookup,
    {
        if seeds.len() >= max_seeds {
            return;
        }

        let node = match node_opt {
            Some(n) => n,
            None => return,
        };

        seeds.push(node.vantage_point);

        let vp_vec = match vector_lookup.get_vector(node.vantage_point) {
            Some(v) => v,
            None => return,
        };

        let dist = calculate_distance(metric, query, vp_vec);

        if dist <= node.radius {
            self.search_node(&node.inside, query, max_seeds, metric, vector_lookup, seeds);
            if seeds.len() < max_seeds {
                self.search_node(
                    &node.outside,
                    query,
                    max_seeds,
                    metric,
                    vector_lookup,
                    seeds,
                );
            }
        } else {
            self.search_node(
                &node.outside,
                query,
                max_seeds,
                metric,
                vector_lookup,
                seeds,
            );
            if seeds.len() < max_seeds {
                self.search_node(&node.inside, query, max_seeds, metric, vector_lookup, seeds);
            }
        }
    }
}
