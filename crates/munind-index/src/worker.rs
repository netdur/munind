use munind_core::domain::MemoryId;
use munind_core::config::DistanceMetric;
use crate::graph::GraphIndex;
use std::collections::HashSet;

/// Graph repair mechanisms to maintain degree constraints and connectivity 
/// per NGT principles.
pub struct GraphWorker;

impl GraphWorker {
    /// Prunes edges that exceed the maximum degree of the node.
    /// NGT favors longer edges for small world navigation and shorter edges for refinement.
    /// A simple heuristic for MVP is just distance-based culling.
    pub fn prune_degree<'a, F>(graph: &mut GraphIndex, _old_max_degree: usize, metric: DistanceMetric, vector_fetcher: &F) 
    where F: Fn(MemoryId) -> Option<&'a [f32]>
    {
        for (lc, layer) in graph.layers.iter_mut() {
            let max_edges = if *lc == 0 { graph.m0 } else { graph.m };
            for (&id, edges) in layer.iter_mut() {
                if edges.len() > max_edges {
                    *edges = GraphIndex::select_edges_hnsw(&metric, id, edges, max_edges, vector_fetcher);
                }
            }
        }
    }

    /// Basic check for isolated connected components (orphans)
    pub fn check_connectivity(graph: &GraphIndex) -> usize {
        let l0 = match graph.layers.get(&0) {
            Some(l) => l,
            None => return 0,
        };
        let mut unvisited: HashSet<MemoryId> = l0.keys().copied().collect();
        let mut components = 0;

        while let Some(&start) = unvisited.iter().next() {
            components += 1;
            let mut stack = vec![start];
            
            while let Some(current) = stack.pop() {
                if unvisited.remove(&current)
                    && let Some(neighbors) = graph.get_neighbors(current, 0)
                {
                    for &n in neighbors {
                        if unvisited.contains(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
        }
        
        components
    }
}
