use crate::graph::GraphIndex;
use crate::lookup::VectorLookup;
use munind_core::config::DistanceMetric;
use munind_core::domain::MemoryId;
use std::collections::HashSet;

/// Graph repair mechanisms to maintain degree constraints and connectivity.
pub struct GraphWorker;

impl GraphWorker {
    /// Prunes edges that exceed the maximum degree of the node.
    pub fn prune_degree<L>(
        graph: &mut GraphIndex,
        _old_max_degree: usize,
        metric: DistanceMetric,
        vector_lookup: &L,
    ) where
        L: VectorLookup,
    {
        for (lc, layer) in &mut graph.layers {
            let max_edges = if *lc == 0 { graph.m0 } else { graph.m };
            for (&id, edges) in layer.iter_mut() {
                if edges.len() > max_edges {
                    *edges =
                        GraphIndex::select_edges_hnsw(&metric, id, edges, max_edges, vector_lookup);
                }
            }
        }
    }

    /// Basic check for isolated connected components (orphans).
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
