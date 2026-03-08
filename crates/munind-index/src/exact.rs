use crate::vector::calculate_distance;
use munind_core::config::DistanceMetric;
use munind_core::domain::MemoryId;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct ScoredHit {
    pub id: MemoryId,
    pub distance: f32,
}

impl PartialEq for ScoredHit {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for ScoredHit {}

impl PartialOrd for ScoredHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Exact Brute Force Search
pub struct ExactSearcher<'a> {
    pub query: &'a [f32],
    pub metric: DistanceMetric,
    pub top_k: usize,
    pub heap: BinaryHeap<ScoredHit>,
}

impl<'a> ExactSearcher<'a> {
    pub fn new(query: &'a [f32], metric: DistanceMetric, top_k: usize) -> Self {
        Self {
            query,
            metric,
            top_k,
            heap: BinaryHeap::with_capacity(top_k),
        }
    }

    pub fn push(&mut self, id: MemoryId, vector: &[f32]) {
        let distance = calculate_distance(&self.metric, self.query, vector);

        if self.heap.len() < self.top_k {
            self.heap.push(ScoredHit { id, distance });
        } else if let Some(max) = self.heap.peek()
            && distance < max.distance
        {
            self.heap.pop();
            self.heap.push(ScoredHit { id, distance });
        }
    }

    pub fn take_results(self) -> Vec<ScoredHit> {
        let mut results = self.heap.into_vec();
        // Sort lowest distance first
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        results
    }
}
