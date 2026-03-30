/// Public-facing NGT Index — wraps ObjectSpace + DVPTree + NeighborhoodGraph
/// with the API expected by tests/ngt_engine.rs.

use std::io::Write;

use rayon::prelude::*;

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, SearchOptions};
use crate::graph::{GraphProperty, GraphType, NeighborhoodGraph};
use crate::node::NodeId;
use crate::object_space::ObjectSpace;
use crate::primitive_comparator::DistanceType;
use crate::tree::DVPTree;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexDistanceType {
    L1,
    L2,
    Hamming,
    Angle,
    Cosine,
    NormalizedAngle,
    NormalizedCosine,
    Jaccard,
    SparseJaccard,
    NormalizedL2,
    InnerProduct,
    Poincare,
    Lorentz,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    GraphAndTree = 0,
    Graph = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdenticalObjectEdgeType {
    None = 0,
    DirectedEdge = 1,
    UndirectedEdge = 2,
}

// ---------------------------------------------------------------------------
// IndexProperty
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IndexProperty {
    pub dimension: usize,
    pub distance_type: IndexDistanceType,
    pub index_type: IndexType,
    pub edge_size_for_creation: i32,
    pub edge_size_for_search: i32,
    pub thread_pool_size: i32,
    pub seed_size: usize,
    pub truncation_threshold: usize,
    pub batch_size_for_creation: i32,
    pub graph_type: GraphType,
    pub leaf_node_size: usize,
    pub internal_children_size: usize,
    pub outgoing_edge: i32,
    pub incoming_edge: i32,
    pub identical_object_edge_type: IdenticalObjectEdgeType,
    pub(crate) insertion_radius_coefficient: f32,
}

impl IndexProperty {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            distance_type: IndexDistanceType::L2,
            index_type: IndexType::GraphAndTree,
            edge_size_for_creation: 10,
            edge_size_for_search: 0,
            thread_pool_size: 32,
            seed_size: 10,
            truncation_threshold: 50,
            batch_size_for_creation: 200,
            graph_type: GraphType::ANNG,
            leaf_node_size: 100,
            internal_children_size: 5,
            outgoing_edge: 10,
            incoming_edge: 80,
            identical_object_edge_type: IdenticalObjectEdgeType::None,
            insertion_radius_coefficient: 1.1,
        }
    }

    pub fn set_distance_type(&mut self, dt: IndexDistanceType) {
        self.distance_type = dt;
    }

    pub(crate) fn to_distance_type(&self) -> DistanceType {
        match self.distance_type {
            IndexDistanceType::L1 => DistanceType::L1,
            IndexDistanceType::L2 => DistanceType::L2,
            IndexDistanceType::Hamming => DistanceType::Hamming,
            IndexDistanceType::Angle => DistanceType::Angle,
            IndexDistanceType::Cosine => DistanceType::NormalizedCosineSimilarity,
            IndexDistanceType::NormalizedAngle => DistanceType::NormalizedAngle,
            IndexDistanceType::NormalizedCosine => DistanceType::NormalizedCosineSimilarity,
            IndexDistanceType::Jaccard => DistanceType::Jaccard,
            IndexDistanceType::SparseJaccard => DistanceType::SparseJaccard,
            IndexDistanceType::NormalizedL2 => DistanceType::NormalizedL2,
            IndexDistanceType::InnerProduct => DistanceType::InnerProduct,
            IndexDistanceType::Poincare => DistanceType::Poincare,
            IndexDistanceType::Lorentz => DistanceType::Lorentz,
        }
    }

    pub fn export_to(&self, ps: &mut PropertySet) {
        ps.set_str("Dimension", self.dimension);
        ps.set_str("DistanceType", self.to_distance_type() as i32);
        ps.set_str("ObjectType", "Float");
        ps.set_str("IndexType", match self.index_type {
            IndexType::GraphAndTree => "GraphAndTree",
            IndexType::Graph => "Graph",
        });
        ps.set_str("EdgeSizeForCreation", self.edge_size_for_creation);
        ps.set_str("EdgeSizeForSearch", self.edge_size_for_search);
        ps.set_str("EpsilonForCreation", self.insertion_radius_coefficient - 1.0);
        ps.set_str("IncrimentalEdgeSizeLimitForTruncation", self.truncation_threshold);
        ps.set_str("SeedSize", self.seed_size);
        ps.set_str("LeafNodeSize", self.leaf_node_size);
        ps.set_str("InternalChildrenSize", self.internal_children_size);
        ps.set_str("BatchSizeForCreation", self.batch_size_for_creation);
        ps.set_str("OutgoingEdge", self.outgoing_edge);
        ps.set_str("IncomingEdge", self.incoming_edge);
        ps.set_str("GraphType", match self.graph_type {
            GraphType::ANNG => "ANNG",
            GraphType::KNNG => "KNNG",
            GraphType::ONNG => "ONNG",
            GraphType::IANNG => "IANNG",
            _ => "ANNG",
        });
    }

    pub(crate) fn import_from(ps: &PropertySet) -> Self {
        let mut p = IndexProperty::new(0);
        p.dimension = ps.get_i64("Dimension", 0) as usize;
        let dt = ps.get_i64("DistanceType", 1) as i32;
        p.distance_type = match dt {
            0 => IndexDistanceType::L1,
            1 => IndexDistanceType::L2,
            3 => IndexDistanceType::Angle,
            4 => IndexDistanceType::Cosine,
            5 => IndexDistanceType::NormalizedAngle,
            6 => IndexDistanceType::NormalizedCosine,
            9 => IndexDistanceType::NormalizedL2,
            10 => IndexDistanceType::InnerProduct,
            100 => IndexDistanceType::Poincare,
            101 => IndexDistanceType::Lorentz,
            _ => IndexDistanceType::L2,
        };
        if let Some(it) = ps.get_str("IndexType") {
            p.index_type = match it {
                "Graph" => IndexType::Graph,
                _ => IndexType::GraphAndTree,
            };
        }
        p.edge_size_for_creation = ps.get_i64("EdgeSizeForCreation", 10) as i32;
        p.edge_size_for_search = ps.get_i64("EdgeSizeForSearch", 0) as i32;
        let eps = ps.get_f32("EpsilonForCreation", 0.1);
        p.insertion_radius_coefficient = eps + 1.0;
        p.truncation_threshold = ps.get_i64("IncrimentalEdgeSizeLimitForTruncation", 50) as usize;
        p.seed_size = ps.get_i64("SeedSize", 10) as usize;
        p.leaf_node_size = ps.get_i64("LeafNodeSize", 100) as usize;
        p.internal_children_size = ps.get_i64("InternalChildrenSize", 5) as usize;
        p.batch_size_for_creation = ps.get_i64("BatchSizeForCreation", 200) as i32;
        p.outgoing_edge = ps.get_i64("OutgoingEdge", 10) as i32;
        p.incoming_edge = ps.get_i64("IncomingEdge", 80) as i32;
        p
    }
}

// ---------------------------------------------------------------------------
// Graph view (0-indexed edges for public access)
// ---------------------------------------------------------------------------

pub struct Graph {
    /// 0-indexed: `edges[i]` = adjacency list for object `i+1`.
    pub edges: Vec<Vec<ObjectDistance>>,
    inner: NeighborhoodGraph,
}

impl Graph {
    fn new(prop: &IndexProperty) -> Self {
        let gp = GraphProperty {
            truncation_threshold: prop.truncation_threshold,
            edge_size_for_creation: prop.edge_size_for_creation,
            edge_size_for_search: prop.edge_size_for_search,
            insertion_radius_coefficient: prop.insertion_radius_coefficient,
            seed_size: prop.seed_size,
            graph_type: prop.graph_type,
            batch_size_for_creation: prop.batch_size_for_creation,
            outgoing_edge: prop.outgoing_edge,
            incoming_edge: prop.incoming_edge,
            ..GraphProperty::default()
        };
        Graph {
            edges: Vec::new(),
            inner: NeighborhoodGraph::with_property(gp),
        }
    }

    /// Sync `edges` (0-indexed) from `inner.nodes` (1-indexed).
    fn sync_from_inner(&mut self) {
        self.edges.clear();
        self.edges.reserve(self.inner.nodes.len().saturating_sub(1));
        for i in 1..self.inner.nodes.len() {
            match &self.inner.nodes[i] {
                Some(edges) => self.edges.push(edges.clone()),
                None => self.edges.push(Vec::new()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tree view
// ---------------------------------------------------------------------------

pub struct Tree {
    pub(crate) inner: DVPTree,
}

impl Tree {
    fn new(prop: &IndexProperty) -> Self {
        Tree {
            inner: DVPTree::new(prop.leaf_node_size, prop.internal_children_size),
        }
    }

    /// Direct access to leaf nodes (no cloning — references inner storage).
    pub fn leaves(&self) -> &[Option<crate::node::LeafNode>] {
        &self.inner.leaf_nodes
    }

    fn sync_from_inner(&mut self) {
        // No-op: leaves() reads inner.leaf_nodes directly.
    }

    pub fn leaf_for_query(
        &self,
        query: &[f32],
        os: &ObjectSpace,
    ) -> Result<NodeId, NgtError> {
        self.inner.search_leaf(query, os)
    }

    pub fn get_object_ids_from_leaf(&self, nid: NodeId) -> Vec<ObjectDistance> {
        self.inner.get_object_ids_from_leaf(nid)
    }
}

// ---------------------------------------------------------------------------
// Objects view
// ---------------------------------------------------------------------------

pub struct ObjectsView {
    count: usize,
}

impl ObjectsView {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

pub struct Index {
    pub objects: ObjectsView,
    pub graph: Graph,
    pub object_space: Option<ObjectSpace>,
    pub tree: Option<Tree>,
    pub property: IndexProperty,
    path: String,
    batch_auto_build: bool,
}

impl Index {
    pub fn create(path: &str, property: IndexProperty) -> Result<Self, NgtError> {
        if property.dimension == 0 {
            return Err("Index::create: dimension must be > 0".to_string());
        }
        let os = ObjectSpace::new(property.dimension, property.to_distance_type());
        let graph = Graph::new(&property);
        let tree = if property.index_type == IndexType::GraphAndTree {
            Some(Tree::new(&property))
        } else {
            None
        };

        Ok(Index {
            objects: ObjectsView { count: 0 },
            graph,
            object_space: Some(os),
            tree,
            property,
            path: path.to_string(),
            batch_auto_build: true,
        })
    }

    pub fn open(path: &str) -> Result<Self, NgtError> {
        Self::open_directory(path)
    }

    pub fn open_directory(path: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", path))?;
        let property = IndexProperty::import_from(&ps);

        let mut os = ObjectSpace::new(property.dimension, property.to_distance_type());
        os.deserialize(&format!("{}/obj", path))?;

        let mut graph = Graph::new(&property);
        graph.inner.deserialize_from_file(&format!("{}/grp", path))?;
        graph.sync_from_inner();

        let tree = if property.index_type == IndexType::GraphAndTree {
            let mut t = Tree::new(&property);
            t.inner.deserialize_from_file(&format!("{}/tre", path), property.dimension)?;
            t.sync_from_inner();
            Some(t)
        } else {
            None
        };

        let count = os.count();
        Ok(Index {
            objects: ObjectsView { count },
            graph,
            object_space: Some(os),
            tree,
            property,
            path: path.to_string(),
            batch_auto_build: true,
        })
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert a vector. Returns the assigned 1-based ID.
    /// Does NOT build graph/tree — call `build()` after all inserts.
    pub fn insert(&mut self, v: &[f32]) -> Result<ObjectID, NgtError> {
        let os = self.object_space.as_mut().ok_or("no object space")?;
        let id = os.insert(v)?;
        self.objects.count = os.count();
        // Ensure graph has a slot for this ID.
        let idx = id as usize;
        if idx >= self.graph.inner.nodes.len() {
            self.graph.inner.nodes.resize_with(idx + 1, || None);
        }
        if self.graph.inner.nodes[idx].is_none() {
            self.graph.inner.nodes[idx] = Some(Vec::new());
        }
        // Keep edges in sync (add empty entry).
        while self.graph.edges.len() < os.size() - 1 {
            self.graph.edges.push(Vec::new());
        }
        Ok(id)
    }

    /// Batch insert. Returns IDs. Auto-builds if `batch_auto_build` is true.
    pub fn insert_batch(&mut self, vecs: &[Vec<f32>]) -> Result<Vec<ObjectID>, NgtError> {
        let mut ids = Vec::with_capacity(vecs.len());
        for v in vecs {
            ids.push(self.insert(v)?);
        }
        if self.batch_auto_build {
            self.build();
        }
        Ok(ids)
    }

    /// Insert and immediately rebuild graph/tree for this object.
    pub fn insert_and_rebuild(&mut self, v: &[f32]) -> Result<ObjectID, NgtError> {
        let id = self.insert(v)?;
        self.build();
        Ok(id)
    }

    pub fn set_batch_auto_build(&mut self, auto_build: bool) {
        self.batch_auto_build = auto_build;
    }

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    pub fn build(&mut self) {
        let obj_count = match &self.object_space {
            Some(os) => os.size(),
            None => return,
        };

        // Collect IDs that need building.
        let mut to_build: Vec<ObjectID> = Vec::new();
        for id in 1..obj_count {
            let oid = id as ObjectID;
            let present = self.object_space.as_ref().unwrap().is_present(oid);
            if !present {
                continue;
            }
            let has_edges = match self.graph.inner.nodes.get(oid as usize) {
                Some(Some(edges)) => !edges.is_empty(),
                _ => false,
            };
            if !has_edges {
                to_build.push(oid);
            }
        }

        if to_build.is_empty() {
            self.sync_views();
            return;
        }

        let batch_size = (self.property.batch_size_for_creation as usize).max(1);
        let k = self.property.edge_size_for_creation as usize;
        let epsilon = self.property.insertion_radius_coefficient - 1.0;
        let seed_size = self.property.seed_size.max(1);
        let use_directed = self.property.identical_object_edge_type == IdenticalObjectEdgeType::DirectedEdge;
        let use_tree = self.property.index_type == IndexType::GraphAndTree;

        // For small datasets, use sequential build for correctness.
        // Parallel batching only helps for large datasets.
        let parallel_threshold = batch_size * 2;
        if to_build.len() <= parallel_threshold {
            for &oid in &to_build {
                let _ = self.build_single(oid);
            }
            self.sync_views();
            return;
        }

        // Bootstrap: insert first object serially so the graph has at least
        // one node for subsequent parallel batches to use as seeds.
        {
            let first = to_build[0];
            let _ = self.build_single(first);
        }
        let to_build = &to_build[1..];

        for batch in to_build.chunks(batch_size) {
            // Phase 1: parallel search for neighbors (read-only graph + tree + os).
            let os = self.object_space.as_ref().unwrap();
            let graph_ref = &self.graph.inner;
            let tree_ref = self.tree.as_ref().map(|t| &t.inner);

            let search_results: Vec<(ObjectID, Vec<f32>, Vec<ObjectDistance>)> = batch
                .par_iter()
                .map(|&oid| {
                    let obj = match os.get_object(oid) {
                        Ok(o) => o.to_vec(),
                        Err(_) => return (oid, Vec::new(), Vec::new()),
                    };

                    // Get seeds.
                    let mut seeds = Vec::new();
                    if let Some(tree) = tree_ref {
                        if let Ok(leaf_nid) = tree.search_leaf(&obj, os) {
                            seeds = tree.get_object_ids_from_leaf(leaf_nid);
                        }
                    }
                    if seeds.is_empty() {
                        // Fallback: first objects already in graph.
                        for id in 1..os.size() {
                            let sid = id as ObjectID;
                            if os.is_present(sid) && !graph_ref.is_empty_node(sid) {
                                if let Some(edges) = graph_ref.get_node(sid) {
                                    if !edges.is_empty() {
                                        seeds.push(ObjectDistance::new(sid, 0.0));
                                        if seeds.len() >= seed_size {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if seeds.is_empty() {
                        return (oid, obj, Vec::new());
                    }

                    let results = graph_ref.search(
                        &obj, &mut seeds, k, epsilon, 0, f32::MAX, os,
                    );
                    (oid, obj, results)
                })
                .collect();

            // Phase 2: serial insertion of edges + tree (with deferred truncation).
            let mut anng_batch: Vec<(ObjectID, Vec<ObjectDistance>)> = Vec::new();

            for (id, _obj, results) in search_results {
                if results.is_empty() && self.graph.inner.nodes.get(id as usize).map_or(true, |n| n.is_none()) {
                    self.graph.inner.insert_node(id, Vec::new());
                    if use_tree {
                        if let Some(tree) = &mut self.tree {
                            let os = self.object_space.as_ref().unwrap();
                            let _ = tree.inner.insert(id, os);
                        }
                    }
                    continue;
                }

                // Handle identical objects with directed edges.
                if use_directed && !results.is_empty() {
                    let first_dist: f32 = results[0].distance;
                    if first_dist == 0.0 {
                        let first_id: u32 = results[0].id;
                        let os = self.object_space.as_ref().unwrap();
                        if let (Ok(a), Ok(b)) = (os.get_object(id), os.get_object(first_id)) {
                            let l1_dist = crate::primitive_comparator::compare_l1(a, b);
                            if l1_dist == 0.0 {
                                self.graph.inner.insert_node(id, Vec::new());
                                let _ = self.graph.inner.add_edge(first_id, id, 0.0);
                                continue;
                            }
                        }
                    }
                }

                // Collect for batch ANNG insertion (deferred truncation).
                anng_batch.push((id, results.clone()));

                // Insert into tree.
                if use_tree {
                    let skip_tree = !results.is_empty() && {
                        let first_dist: f32 = results[0].distance;
                        first_dist == 0.0
                    };
                    if !skip_tree {
                        if let Some(tree) = &mut self.tree {
                            let os = self.object_space.as_ref().unwrap();
                            let _ = tree.inner.insert(id, os);
                        }
                    }
                }
            }

            // Batch ANNG insert with parallel truncation.
            if !anng_batch.is_empty() {
                let os = self.object_space.as_ref().unwrap();
                self.graph.inner.insert_anng_nodes_batch(&anng_batch, os);
            }
        }
        self.sync_views();
    }

    fn build_single(&mut self, id: ObjectID) -> Result<(), NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;
        let obj_data = os.get_object(id)?.to_vec();

        // Search for nearest neighbors.
        let results = self.search_for_insertion(&obj_data)?;

        // Handle identical objects.
        if self.property.identical_object_edge_type == IdenticalObjectEdgeType::DirectedEdge {
            if !results.is_empty() {
                let first_dist: f32 = results[0].distance;
                if first_dist == 0.0 {
                    let first_id: u32 = results[0].id;
                    // Check if truly identical.
                    let os = self.object_space.as_ref().unwrap();
                    if let (Ok(a), Ok(b)) = (os.get_object(id), os.get_object(first_id)) {
                        let l1_dist = crate::primitive_comparator::compare_l1(a, b);
                        if l1_dist == 0.0 {
                            // Directed edge: empty edges for new node, add FROM existing TO new.
                            self.graph.inner.insert_node(id, Vec::new());
                            let _ = self.graph.inner.add_edge(first_id, id, 0.0);
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Normal ANNG insertion.
        {
            let os = self.object_space.as_ref().unwrap();
            self.graph.inner.insert_anng_node(id, results.clone(), os);
        }

        // Insert into tree if not identical.
        if self.property.index_type == IndexType::GraphAndTree {
            let skip_tree = !results.is_empty() && {
                let first_dist: f32 = results[0].distance;
                first_dist == 0.0
            };
            if !skip_tree {
                if let Some(tree) = &mut self.tree {
                    let os = self.object_space.as_ref().unwrap();
                    let _ = tree.inner.insert(id, os);
                }
            }
        }

        Ok(())
    }

    fn search_for_insertion(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;
        let k = self.property.edge_size_for_creation as usize;
        let epsilon = self.property.insertion_radius_coefficient - 1.0;

        let mut seeds = self.get_seeds(query)?;
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        let results = self.graph.inner.search(
            query, &mut seeds, k, epsilon, 0, f32::MAX, os,
        );
        Ok(results)
    }

    fn get_seeds(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;

        // Try tree first.
        if let Some(tree) = &self.tree {
            let leaf_nid = tree.inner.search_leaf(query, os)?;
            let seeds = tree.inner.get_object_ids_from_leaf(leaf_nid);
            if !seeds.is_empty() {
                return Ok(seeds);
            }
        }

        // Fallback: first available objects that are already in the graph.
        let mut seeds = Vec::new();
        for id in 1..os.size() {
            let oid = id as ObjectID;
            if os.is_present(oid) && !self.graph.inner.is_empty_node(oid) {
                if let Some(edges) = self.graph.inner.get_node(oid) {
                    if !edges.is_empty() {
                        seeds.push(ObjectDistance::new(oid, 0.0));
                        if seeds.len() >= self.property.seed_size.max(1) {
                            break;
                        }
                    }
                }
            }
        }
        Ok(seeds)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;
        if options.k == 0 {
            return Ok(Vec::new());
        }

        // Normalize query if needed.
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if os.normalization {
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

        let mut results = self.graph.inner.search(
            q, &mut seeds, options.k, options.epsilon, edge_size, f32::MAX, os,
        );
        results.truncate(options.k);
        Ok(results)
    }

    pub fn linear_search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;
        os.linear_search(query, -1.0, k)
    }

    // -----------------------------------------------------------------------
    // Remove
    // -----------------------------------------------------------------------

    /// Batch delete with ID compaction. Returns count of actually removed objects.
    pub fn delete_batch(&mut self, ids: &[ObjectID]) -> Result<usize, NgtError> {
        let os = self.object_space.as_ref().ok_or("no object space")?;

        // Validate: ID 0 is out of range.
        for &id in ids {
            if id == 0 || id as usize >= os.size() {
                return Err(format!("delete_batch: id {} is out of range", id));
            }
        }

        // Collect unique IDs to delete.
        let mut to_delete: std::collections::HashSet<ObjectID> = std::collections::HashSet::new();
        for &id in ids {
            if self.object_space.as_ref().unwrap().is_present(id) {
                to_delete.insert(id);
            }
        }
        let removed_count = to_delete.len();

        if removed_count == 0 {
            return Ok(0);
        }

        // Collect surviving objects.
        let os = self.object_space.as_ref().unwrap();
        let mut surviving: Vec<Vec<f32>> = Vec::new();
        for id in 1..os.size() {
            let oid = id as ObjectID;
            if os.is_present(oid) && !to_delete.contains(&oid) {
                surviving.push(os.get_object(oid).unwrap().to_vec());
            }
        }

        // Rebuild from scratch with compacted IDs.
        let dim = self.property.dimension;
        let dt = self.property.to_distance_type();
        let mut new_os = ObjectSpace::new(dim, dt);
        for v in &surviving {
            new_os.insert(v)?;
        }

        self.object_space = Some(new_os);

        // Rebuild graph and tree.
        self.graph = Graph::new(&self.property);
        if self.property.index_type == IndexType::GraphAndTree {
            self.tree = Some(Tree::new(&self.property));
        }
        self.build();

        self.objects.count = self.object_space.as_ref().unwrap().count();
        Ok(removed_count)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn object_count(&self) -> usize {
        self.objects.count
    }

    pub fn all_objects(&self) -> Vec<Vec<f32>> {
        let os = match &self.object_space {
            Some(os) => os,
            None => return Vec::new(),
        };
        let mut result = Vec::new();
        for (_, obj) in os.iter_objects() {
            result.push(obj.to_vec());
        }
        result
    }

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    pub fn save(&self, path: Option<&str>) -> Result<(), NgtError> {
        let p = path.unwrap_or(&self.path);
        self.save_as_directory(p)
    }

    pub fn save_as_directory(&self, dir: &str) -> Result<(), NgtError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("save_as_directory: {}: {}", dir, e))?;

        let mut ps = PropertySet::new();
        self.property.export_to(&mut ps);
        ps.save(&format!("{}/prf", dir))?;

        if let Some(os) = &self.object_space {
            os.serialize(&format!("{}/obj", dir))?;
        }

        self.graph.inner.serialize_to_file(&format!("{}/grp", dir))?;

        if let Some(tree) = &self.tree {
            tree.inner.serialize_to_file(&format!("{}/tre", dir), self.property.dimension)?;
        }

        Ok(())
    }

    pub fn save_as_mmap(&self, dir: &str) -> Result<(), NgtError> {
        // Same format — obj is already flat/mmap-friendly.
        self.save_as_directory(dir)
    }

    // -----------------------------------------------------------------------
    // Internal: sync public views from internal state
    // -----------------------------------------------------------------------

    fn sync_views(&mut self) {
        self.graph.sync_from_inner();
        if let Some(tree) = &mut self.tree {
            tree.sync_from_inner();
        }
        if let Some(os) = &self.object_space {
            self.objects.count = os.count();
        }
    }
}
