use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::{Distance, ObjectID};
use crate::node::ObjectDistance;
use crate::object_space::ObjectSpace;

// Min-heap for unchecked nodes
#[derive(Clone, Copy, Debug)]
pub struct MinDistanceNode(pub ObjectDistance);

impl PartialEq for MinDistanceNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance && self.0.id == other.0.id
    }
}
impl Eq for MinDistanceNode {}

impl PartialOrd for MinDistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other
            .0
            .distance
            .partial_cmp(&self.0.distance)
            .map(|ord: Ordering| ord.then_with(|| self.0.id.cmp(&other.0.id)))
    }
}
impl Ord for MinDistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Max-heap for result nodes
#[derive(Clone, Copy, Debug)]
pub struct MaxDistanceNode(pub ObjectDistance);

impl PartialEq for MaxDistanceNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance && self.0.id == other.0.id
    }
}
impl Eq for MaxDistanceNode {}

impl PartialOrd for MaxDistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .map(|ord: Ordering| ord.then_with(|| self.0.id.cmp(&other.0.id)))
    }
}
impl Ord for MaxDistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct SearchContainer<'a> {
    pub object: &'a [f32],
    pub radius: Distance,
    pub size: usize,
    pub exploration_coefficient: f64,
    pub edge_size: isize, // -1 means use property default
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeighborhoodGraph {
    pub edges: Vec<Vec<ObjectDistance>>,
    pub edge_size_for_creation: usize,
    pub edge_size_for_search: usize,
    pub insertion_exploration_coefficient: f64,
    pub dynamic_edge_size_base: usize,
    pub dynamic_edge_size_rate: usize,
}

impl NeighborhoodGraph {
    pub fn new() -> Self {
        Self {
            edges: vec![], // 0-indexed
            edge_size_for_creation: 10,
            edge_size_for_search: 40,
            insertion_exploration_coefficient: 1.1,
            dynamic_edge_size_base: 30,
            dynamic_edge_size_rate: 20,
        }
    }

    pub fn insert_node(&mut self, id: ObjectID) {
        let idx = (id - 1) as usize;
        if idx >= self.edges.len() {
            self.edges.resize(idx + 1, vec![]);
        }
    }

    pub fn get_edge_size(&self, sc: &SearchContainer) -> Result<usize, String> {
        let edge_size = if sc.edge_size == -1 {
            self.edge_size_for_search as i64
        } else {
            sc.edge_size as i64
        };

        if edge_size == 0 {
            return Ok(usize::MAX);
        }
        if edge_size > 0 {
            return Ok(edge_size as usize);
        }
        if edge_size == -2 {
            let add =
                10f64.powf((sc.exploration_coefficient - 1.0) * self.dynamic_edge_size_rate as f64);
            let dynamic = self.dynamic_edge_size_base as f64 + add;
            return Ok(dynamic.min(usize::MAX as f64) as usize);
        }

        Err(format!(
            "NeighborhoodGraph::get_edge_size: invalid edge size {}",
            sc.edge_size
        ))
    }

    pub fn add_edge(
        &mut self,
        target: usize,
        edge: ObjectDistance,
        identity_check: bool,
    ) -> Result<(), String> {
        self.insert_node((target + 1) as u32);
        let node = &mut self.edges[target];
        match node.binary_search_by(|candidate| {
            candidate
                .distance
                .partial_cmp(&edge.distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| candidate.id.cmp(&edge.id))
        }) {
            Ok(existing) => {
                if identity_check && node[existing].id == edge.id {
                    return Err(format!(
                        "NeighborhoodGraph::add_edge: already existed {}",
                        edge.id
                    ));
                }
            }
            Err(position) => node.insert(position, edge),
        }
        if let Some(first_duplicate) = node.windows(2).position(|pair| pair[0].id == pair[1].id) {
            if identity_check {
                return Err(format!(
                    "NeighborhoodGraph::add_edge: already existed {}",
                    node[first_duplicate].id
                ));
            }
            node.remove(first_duplicate + 1);
        }
        Ok(())
    }

    pub fn add_edge_with_deletion(
        &mut self,
        target: usize,
        edge: ObjectDistance,
        k_edge: usize,
        identity_check: bool,
    ) -> Result<(), String> {
        if k_edge == 0 {
            return Ok(());
        }
        self.insert_node((target + 1) as u32);
        while self.edges[target].len() >= k_edge
            && self.edges[target]
                .last()
                .map(|last| last.distance > edge.distance)
                .unwrap_or(false)
        {
            self.edges[target].pop();
        }
        if self.edges[target].len() < k_edge {
            self.add_edge(target, edge, identity_check)?;
            if self.edges[target].len() > k_edge {
                self.edges[target].truncate(k_edge);
            }
        }
        Ok(())
    }

    pub fn search(
        &self,
        object_space: &ObjectSpace,
        sc: &mut SearchContainer,
        seeds: &mut [ObjectDistance],
    ) -> Vec<ObjectDistance> {
        if sc.exploration_coefficient == 0.0 {
            sc.exploration_coefficient = 1.1;
        }

        let edge_size = self
            .get_edge_size(sc)
            .unwrap_or(self.edge_size_for_search.max(1));

        let mut unchecked = BinaryHeap::new();
        let mut distance_checked = vec![0u64; (self.edges.len() + 64) / 64];
        let mut results = BinaryHeap::new();

        // compute initial distances for seeds
        for seed in seeds.iter_mut() {
            seed.distance = object_space
                .compare_to_id(sc.object, seed.id as usize)
                .unwrap_or(f32::MAX);
        }

        seeds.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        if sc.size > 0 {
            for seed in seeds.iter() {
                if (results.len() < sc.size) && (seed.distance <= sc.radius) {
                    results.push(MaxDistanceNode(*seed));
                } else {
                    break;
                }
            }

            if results.len() >= sc.size {
                sc.radius = results.peek().unwrap().0.distance;
            }
        }

        for seed in seeds.iter() {
            if seed.distance == f32::MAX {
                continue;
            }
            let seed_id = seed.id as usize;
            distance_checked[seed_id / 64] |= 1 << (seed_id % 64);
            unchecked.push(MinDistanceNode(*seed));
        }

        let mut exploration_radius = (sc.exploration_coefficient * sc.radius as f64) as Distance;

        while let Some(MinDistanceNode(target)) = unchecked.pop() {
            if target.distance > exploration_radius {
                break;
            }

            let tid = (target.id - 1) as usize;
            let neighbors = if tid < self.edges.len() {
                &self.edges[tid]
            } else {
                continue;
            };

            let neighbor_size = neighbors.len().min(edge_size);

            for neighbor in neighbors.iter().take(neighbor_size) {
                let neighbor_id = neighbor.id as usize;
                let block_idx = neighbor_id / 64;
                let bit_mask = 1 << (neighbor_id % 64);

                if block_idx >= distance_checked.len()
                    || (distance_checked[block_idx] & bit_mask) != 0
                {
                    continue;
                }
                distance_checked[block_idx] |= bit_mask;

                if let Some(distance) = object_space.compare_to_id(sc.object, neighbor.id as usize)
                {
                    if distance <= exploration_radius {
                        let res = ObjectDistance {
                            id: neighbor.id,
                            distance,
                        };
                        unchecked.push(MinDistanceNode(res));

                        if sc.size > 0 && distance <= sc.radius {
                            results.push(MaxDistanceNode(res));
                            if results.len() >= sc.size {
                                if results
                                    .peek()
                                    .map(|top| top.0.distance >= distance)
                                    .unwrap_or(false)
                                {
                                    if results.len() > sc.size {
                                        results.pop();
                                    }
                                    sc.radius = results.peek().unwrap().0.distance;
                                    exploration_radius =
                                        (sc.exploration_coefficient * sc.radius as f64) as Distance;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut final_results = Vec::with_capacity(results.len());
        while let Some(node) = results.pop() {
            final_results.push(node.0);
        }
        final_results.reverse();
        final_results
    }
}
