/// Read-only memory-mapped index.
///
/// Phase 1: loads everything into RAM (memmap2 optimization is Phase 3).
/// Provides search and linear_search on a previously saved index.

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, SearchOptions};
use crate::graph::{GraphProperty, NeighborhoodGraph};
use crate::index::IndexProperty;
use crate::object_space::ObjectSpace;
use crate::tree::DVPTree;

pub struct MmapIndex {
    object_space: ObjectSpace,
    graph: NeighborhoodGraph,
    tree: Option<DVPTree>,
    property: IndexProperty,
}

impl MmapIndex {
    pub fn open(dir: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let property = IndexProperty::import_from(&ps);

        let mut os = ObjectSpace::new(property.dimension, property.to_distance_type());
        os.deserialize(&format!("{}/obj", dir))?;

        let gp = GraphProperty {
            truncation_threshold: property.truncation_threshold,
            edge_size_for_creation: property.edge_size_for_creation,
            edge_size_for_search: property.edge_size_for_search,
            insertion_radius_coefficient: property.insertion_radius_coefficient,
            seed_size: property.seed_size,
            graph_type: property.graph_type,
            batch_size_for_creation: property.batch_size_for_creation,
            outgoing_edge: property.outgoing_edge,
            incoming_edge: property.incoming_edge,
            ..GraphProperty::default()
        };
        let mut graph = NeighborhoodGraph::with_property(gp);
        graph.deserialize_from_file(&format!("{}/grp", dir))?;

        let tree = if std::path::Path::new(&format!("{}/tre", dir)).exists() {
            let mut t = DVPTree::new(property.leaf_node_size, property.internal_children_size);
            t.deserialize_from_file(&format!("{}/tre", dir), property.dimension)?;
            Some(t)
        } else {
            None
        };

        Ok(MmapIndex {
            object_space: os,
            graph,
            tree,
            property,
        })
    }

    pub fn object_count(&self) -> usize {
        self.object_space.count()
    }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        if options.k == 0 {
            return Ok(Vec::new());
        }

        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.object_space.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let mut seeds = self.get_seeds(q)?;
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        let edge_size = match options.edge_size {
            Some(es) => es as i32,
            None => -1,
        };

        let mut results = self.graph.search(
            q,
            &mut seeds,
            options.k,
            options.epsilon,
            edge_size,
            f32::MAX,
            &self.object_space,
        );
        results.truncate(options.k);
        Ok(results)
    }

    pub fn linear_search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        self.object_space.linear_search(query, -1.0, k)
    }

    fn get_seeds(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        if let Some(tree) = &self.tree {
            let leaf_nid = tree.search_leaf(query, &self.object_space)?;
            let seeds = tree.get_object_ids_from_leaf(leaf_nid);
            if !seeds.is_empty() {
                return Ok(seeds);
            }
        }

        let mut seeds = Vec::new();
        for id in 1..self.object_space.size() {
            let oid = id as ObjectID;
            if self.object_space.is_present(oid) {
                seeds.push(ObjectDistance::new(oid, 0.0));
                if seeds.len() >= self.property.seed_size.max(1) {
                    break;
                }
            }
        }
        Ok(seeds)
    }
}
