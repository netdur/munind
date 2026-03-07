pub mod vector;
pub mod graph;
pub mod exact;
pub mod search;
pub mod seed;
pub mod worker;
pub mod engine;

pub use graph::GraphIndex;
pub use exact::{ExactSearcher, ScoredHit};
pub use vector::*;
pub use search::AnnSearcher;
pub use seed::VpTree;
pub use worker::GraphWorker;
pub use engine::IndexEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use munind_core::domain::MemoryId;
    use munind_core::config::IndexConfig;

    #[test]
    fn test_graph_add_remove() {
        use munind_core::config::DistanceMetric;
        use std::collections::HashMap;

        let config = IndexConfig {
            m: 2,
            m0: 2,
            ..Default::default()
        };
        let mut graph = GraphIndex::new(&config);
        let mut vectors = HashMap::new();
        
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);
        let id3 = MemoryId(3);
        let id4 = MemoryId(4);

        vectors.insert(id1, vec![1.0, 1.0, 1.0]);
        vectors.insert(id2, vec![0.1, 0.1, 0.1]);
        vectors.insert(id3, vec![5.0, 5.0, 5.0]);
        vectors.insert(id4, vec![0.5, 0.5, 0.5]);

        graph.insert(id1, &vectors[&id1], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));
        graph.insert(id2, &vectors[&id2], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));
        graph.insert(id3, &vectors[&id3], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));
        graph.insert(id4, &vectors[&id4], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));

        let neighbors = graph.get_neighbors(id1, 0).unwrap();
        assert!(neighbors.len() <= 2);
        
        graph.remove_node(id2);
        let neighbors = graph.get_neighbors(id1, 0).unwrap();
        assert!(!neighbors.contains(&id2));
    }

    #[test]
    fn test_exact_search() {
        use munind_core::config::DistanceMetric;

        let query = vec![0.0, 0.0, 0.0];
        let mut searcher = ExactSearcher::new(&query, DistanceMetric::L2, 2);

        searcher.push(MemoryId(1), &[1.0, 1.0, 1.0]); // dist: 3.0
        searcher.push(MemoryId(2), &[0.1, 0.1, 0.1]); // dist: 0.03 (1st)
        searcher.push(MemoryId(3), &[5.0, 5.0, 5.0]); // dist: 75.0
        searcher.push(MemoryId(4), &[0.5, 0.5, 0.5]); // dist: 0.75 (2nd)

        let results = searcher.take_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.0, 2);
        assert_eq!(results[1].id.0, 4);
    }

    #[test]
    fn test_ann_search() {
        use munind_core::config::DistanceMetric;
        use std::collections::HashMap;

        let config = IndexConfig {
            m: 3,
            m0: 3,
            ..Default::default()
        };
        let mut graph = GraphIndex::new(&config);
        let mut vectors = HashMap::new();

        let id1 = MemoryId(1);
        let id2 = MemoryId(2);
        let id3 = MemoryId(3);
        
        vectors.insert(id1, vec![1.0, 1.0, 1.0]);
        vectors.insert(id2, vec![0.1, 0.1, 0.1]);
        vectors.insert(id3, vec![5.0, 5.0, 5.0]);

        graph.insert(id1, &vectors[&id1], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));
        graph.insert(id2, &vectors[&id2], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));
        graph.insert(id3, &vectors[&id3], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));

        let query = vec![0.0, 0.0, 0.0];
        let entry = graph.entry_point;
        let fetcher = |id| vectors.get(&id).map(|v| v.as_slice());
        let searcher = AnnSearcher::new(&graph, &query, &DistanceMetric::L2, &fetcher);
        
        let results = searcher.search(entry, 2, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id2); // closest
    }

    #[test]
    fn test_vp_tree() {
        use munind_core::config::DistanceMetric;
        use std::collections::HashMap;

        let mut vectors = HashMap::new();
        
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);
        let id3 = MemoryId(3);
        let id4 = MemoryId(4);

        vectors.insert(id1, vec![1.0, 1.0, 1.0]);
        vectors.insert(id2, vec![0.1, 0.1, 0.1]);
        vectors.insert(id3, vec![5.0, 5.0, 5.0]);
        vectors.insert(id4, vec![8.0, 8.0, 8.0]);

        let id_list = vec![id1, id2, id3, id4];
        let tree = VpTree::build(id_list, DistanceMetric::L2, &|id| vectors.get(&id).map(|v| v.as_slice()));
        
        // Search should find closest items as seeds
        let query = vec![0.0, 0.0, 0.0];
        let seeds = tree.search_seeds(&query, 2, DistanceMetric::L2, &|id| vectors.get(&id).map(|v| v.as_slice()));
        
        assert!(seeds.len() <= 2);
        assert!(!seeds.is_empty());
    }

    #[test]
    fn test_graph_worker() {
        use munind_core::config::DistanceMetric;
        use std::collections::HashMap;

        let config = IndexConfig {
            m: 5,
            m0: 5,
            ..Default::default()
        };
        let mut graph = GraphIndex::new(&config);
        let mut vectors = HashMap::new();
        
        // Add 6 nodes and connect id1 to all of them
        let id1 = MemoryId(1);
        vectors.insert(id1, vec![0.0, 0.0, 0.0]);
        graph.insert(id1, &vectors[&id1], DistanceMetric::L2, 10, |id| vectors.get(&id).map(|v| v.as_slice()));

        for i in 2..=7 {
            let id = MemoryId(i);
            vectors.insert(id, vec![i as f32, i as f32, i as f32]);
            // Manually forcing degree inflation rather than insert which might reject
            let l0 = graph.layers.entry(0).or_default();
            l0.entry(id).or_default();
            l0.get_mut(&id1).unwrap().push(id);
        }

        assert_eq!(graph.layers.get(&0).unwrap().get(&id1).unwrap().len(), 6);
        
        // Prune to max degree 3
        GraphWorker::prune_degree(&mut graph, 3, DistanceMetric::L2, &|id| vectors.get(&id).map(|v| v.as_slice()));
        
        let pruned_edges = graph.layers.get(&0).unwrap().get(&id1).unwrap();
        // HNSW heuristic: id2 is closest (dist 3). id3 (dist 27 to id1, dist 3 to id2) 
        // is closer to id2 than id1, so it is dropped. Same for the rest.
        assert_eq!(pruned_edges.len(), 1); 
        assert!(pruned_edges.contains(&MemoryId(2)));

        let components = GraphWorker::check_connectivity(&graph);
        assert!(components > 0);
    }

    #[test]
    fn test_index_engine_integration() {
        use munind_core::config::{EngineConfig, DistanceMetric};
        use std::collections::HashMap;

        let mut config = EngineConfig::default();
        config.index.m = 5;
        config.index.m0 = 5;
        config.index.ef_construction = 10;
        config.index.ef_search = 10;
        config.index.metric = DistanceMetric::L2;

        let mut engine = IndexEngine::new(&config);
        let mut vectors = HashMap::new();

        // Insert a line of vectors
        for i in 1..=20 {
            let id = MemoryId(i);
            vectors.insert(id, vec![i as f32, 0.0, 0.0]);
            engine.insert(id, &vectors[&id], &|id| vectors.get(&id).map(|v| v.as_slice()));
        }

        if let Some(l0) = engine.graph.layers.get(&0) {
            for (id, edges) in l0.iter() {
                println!("Node {}: {:?}", id.0, edges);
            }
        }

        // Search for [15.1, 0, 0] -> Should find 15 and 16
        let query = vec![15.1, 0.0, 0.0];
        let results = engine.search(&query, 2, &|id| vectors.get(&id).map(|v| v.as_slice()));

        assert_eq!(results.len(), 2);
        
        let found_ids: Vec<u64> = results.iter().map(|r| r.id.0).collect();
        assert!(found_ids.contains(&15));
        assert!(found_ids.contains(&16));

        // Test deletion
        engine.delete(MemoryId(15));
        let results2 = engine.search(&query, 2, &|id| vectors.get(&id).map(|v| v.as_slice()));
        let found_ids2: Vec<u64> = results2.iter().map(|r| r.id.0).collect();
        assert!(!found_ids2.contains(&15));
        assert!(found_ids2.contains(&14) || found_ids2.contains(&17)); // Should fall back to next neighbors
    }
}
