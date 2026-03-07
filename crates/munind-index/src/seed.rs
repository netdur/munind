use munind_core::domain::MemoryId;
use munind_core::config::DistanceMetric;
use crate::vector::calculate_distance;
use rand::prelude::IndexedRandom;

pub struct VpNode {
    pub vantage_point: MemoryId,
    pub radius: f32, // Distance to median
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

    /// Batch builds the tree recursively from a set of initial nodes
    pub fn build<'a, F>(items: Vec<MemoryId>, metric: DistanceMetric, vector_fetcher: &F) -> Self 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        Self {
            root: Self::build_node(items, &metric, vector_fetcher)
        }
    }

    fn build_node<'a, F>(mut items: Vec<MemoryId>, metric: &DistanceMetric, vector_fetcher: &F) -> Option<Box<VpNode>> 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        if items.is_empty() {
            return None;
        }

        // Pick a random vantage point
        let vp_idx = {
            let mut rng = rand::rng();
            let indices: Vec<usize> = (0..items.len()).collect();
            *indices.choose(&mut rng).unwrap()
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

        let vp_vec = vector_fetcher(vantage_point)?;

        // Compute distances
        let mut distances: Vec<(MemoryId, f32)> = items.into_iter().filter_map(|id| {
            vector_fetcher(id).map(|v| (id, calculate_distance(metric, vp_vec, v)))
        }).collect();

        if distances.is_empty() {
            return Some(Box::new(VpNode {
                vantage_point,
                radius: 0.0,
                inside: None,
                outside: None,
            }));
        }

        // Calculate median
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
            inside: Self::build_node(inside_ids, metric, vector_fetcher),
            outside: Self::build_node(outside_ids, metric, vector_fetcher),
        }))
    }

    /// Traverses the tree to find initial candidates (seeds)
    pub fn search_seeds<'a, F>(&self, query: &[f32], max_seeds: usize, metric: DistanceMetric, vector_fetcher: &F) -> Vec<MemoryId> 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        let mut seeds = Vec::new();
        self.search_node(&self.root, query, max_seeds, &metric, vector_fetcher, &mut seeds);
        
        // Sort seeds by actual distance to query to ensure the best one is first
        seeds.sort_by(|&a, &b| {
            let dist_a = vector_fetcher(a).map(|v| calculate_distance(&metric, query, v)).unwrap_or(f32::MAX);
            let dist_b = vector_fetcher(b).map(|v| calculate_distance(&metric, query, v)).unwrap_or(f32::MAX);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        seeds
    }

    fn search_node<'a, F>(&self, node_opt: &Option<Box<VpNode>>, query: &[f32], max_seeds: usize, metric: &DistanceMetric, vector_fetcher: &F, seeds: &mut Vec<MemoryId>) 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        if seeds.len() >= max_seeds {
            return;
        }

        let node = match node_opt {
            Some(n) => n,
            None => return,
        };

        seeds.push(node.vantage_point);

        let vp_vec = match vector_fetcher(node.vantage_point) {
            Some(v) => v,
            None => return,
        };

        let dist = calculate_distance(metric, query, vp_vec);

        // Simple heuristic search logic (not perfectly exact nearest neighbors, just close seeds)
        if dist <= node.radius {
            // Likely inside
            self.search_node(&node.inside, query, max_seeds, metric, vector_fetcher, seeds);
            if seeds.len() < max_seeds {
                self.search_node(&node.outside, query, max_seeds, metric, vector_fetcher, seeds);
            }
        } else {
            // Likely outside
            self.search_node(&node.outside, query, max_seeds, metric, vector_fetcher, seeds);
            if seeds.len() < max_seeds {
                self.search_node(&node.inside, query, max_seeds, metric, vector_fetcher, seeds);
            }
        }
    }
}
