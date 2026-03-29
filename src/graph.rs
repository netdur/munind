/// Port of NGT/Graph.h + NGT/Graph.cpp  (NeighborhoodGraph)
///
/// Non-shared-memory, float-only variant (Phase 1).
///
/// Implements the ANNG (Approximate Nearest Neighbor Graph) as the default
/// graph type.  The core greedy best-first search algorithm is ported 1:1.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID};
use crate::object_space::ObjectSpace;

// ---------------------------------------------------------------------------
// BooleanVector — visited set matching C++ NGT::BooleanVector
// ---------------------------------------------------------------------------

struct BooleanVector {
    data: Vec<bool>,
}

impl BooleanVector {
    fn new(size: usize) -> Self {
        Self {
            data: vec![false; size],
        }
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        let idx = id as usize;
        idx < self.data.len() && self.data[idx]
    }

    #[inline]
    fn insert(&mut self, id: u32) {
        let idx = id as usize;
        if idx < self.data.len() {
            self.data[idx] = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Prefetch helper
// ---------------------------------------------------------------------------

#[inline]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // prfm pldl1keep, [ptr]
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr as *const u8, options(nostack, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

// ---------------------------------------------------------------------------
// Constants matching C++ #defines
// ---------------------------------------------------------------------------

const DEFAULT_EXPLORATION_COEFFICIENT: f32 = 1.1;
const DEFAULT_INSERTION_EXPLORATION_COEFFICIENT: f32 = 1.1;
const DEFAULT_TRUNCATION_THRESHOLD: usize = 50;
const DEFAULT_SEED_SIZE: usize = 10;
const DEFAULT_CREATION_EDGE_SIZE: i32 = 10;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphType {
    None   = 0,
    ANNG   = 1,
    KNNG   = 2,
    BKNNG  = 3,
    ONNG   = 4,
    IANNG  = 5,
    DNNG   = 6,
    RANNG  = 7,
    RIANNG = 8,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedType {
    None         = 0,
    RandomNodes  = 1,
    FixedNodes   = 2,
    FirstNode    = 3,
    AllLeafNodes = 4,
}

// ---------------------------------------------------------------------------
// Graph::Property  (NeighborhoodGraph::Property)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GraphProperty {
    pub truncation_threshold: usize,
    pub edge_size_for_creation: i32,
    pub edge_size_for_search: i32,
    pub edge_size_limit_for_creation: i32,
    pub insertion_radius_coefficient: f32,
    pub seed_size: usize,
    pub seed_type: SeedType,
    pub batch_size_for_creation: i32,
    pub graph_type: GraphType,
    pub dynamic_edge_size_base: i32,
    pub dynamic_edge_size_rate: i32,
    pub outgoing_edge: i32,
    pub incoming_edge: i32,
}

impl Default for GraphProperty {
    fn default() -> Self {
        Self {
            truncation_threshold: DEFAULT_TRUNCATION_THRESHOLD,
            edge_size_for_creation: DEFAULT_CREATION_EDGE_SIZE,
            edge_size_for_search: 0,
            edge_size_limit_for_creation: 5,
            insertion_radius_coefficient: DEFAULT_INSERTION_EXPLORATION_COEFFICIENT,
            seed_size: DEFAULT_SEED_SIZE,
            seed_type: SeedType::None,
            batch_size_for_creation: 200,
            graph_type: GraphType::ANNG,
            dynamic_edge_size_base: 30,
            dynamic_edge_size_rate: 20,
            outgoing_edge: 10,
            incoming_edge: 80,
        }
    }
}

// ---------------------------------------------------------------------------
// NeighborhoodGraph
// ---------------------------------------------------------------------------

/// The ANNG (Approximate Nearest Neighbor Graph).
///
/// `nodes[id]` = `Some(Vec<ObjectDistance>)` is the adjacency list for object
/// `id`, sorted by ascending distance.  `nodes[0]` is always `None`.
///
/// `prevsize[id]` tracks the edge count at last truncation, used to decide
/// whether truncation is needed.
pub struct NeighborhoodGraph {
    /// Adjacency lists (1-based).  `nodes[0]` = None.
    pub nodes: Vec<Option<Vec<ObjectDistance>>>,
    /// Per-node previous size (for truncation).  Index 0 unused.
    pub prevsize: Vec<u16>,
    pub property: GraphProperty,
}

impl NeighborhoodGraph {
    pub fn new() -> Self {
        Self {
            nodes: vec![None], // slot 0
            prevsize: vec![0],
            property: GraphProperty::default(),
        }
    }

    pub fn with_property(prop: GraphProperty) -> Self {
        Self {
            nodes: vec![None],
            prevsize: vec![0],
            property: prop,
        }
    }

    /// Total allocated slots (including slot 0).
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty_node(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx >= self.nodes.len() || self.nodes[idx].is_none()
    }

    /// Get the adjacency list for node `id`.
    pub fn get_node(&self, id: ObjectID) -> Option<&Vec<ObjectDistance>> {
        self.nodes.get(id as usize).and_then(|o| o.as_ref())
    }

    /// Get mutable adjacency list for node `id`.
    pub fn get_node_mut(&mut self, id: ObjectID) -> Option<&mut Vec<ObjectDistance>> {
        self.nodes.get_mut(id as usize).and_then(|o| o.as_mut())
    }

    // -----------------------------------------------------------------------
    // Edge manipulation
    // -----------------------------------------------------------------------

    /// Add an edge to `node` (sorted insertion, maintains ascending order).
    /// If `identity_check` and the edge already exists, returns Err.
    /// Maps to `NeighborhoodGraph::addEdge(GraphNode&, ...)`.
    fn add_edge_to_node(
        node: &mut Vec<ObjectDistance>,
        add_id: ObjectID,
        add_distance: f32,
        identity_check: bool,
    ) -> Result<(), NgtError> {
        let obj = ObjectDistance::new(add_id, add_distance);
        // Binary search for insertion point (sorted by (distance, id)).
        let pos = node.partition_point(|e| *e < obj);
        if pos < node.len() {
            let existing_id: u32 = node[pos].id;
            if existing_id == add_id {
                if identity_check {
                    return Err(format!("addEdge: already existed! {}", add_id));
                }
                return Ok(());
            }
        }
        node.insert(pos, obj);
        Ok(())
    }

    /// Add an edge from `target` to `add_id`.
    /// Returns `true` if truncation is needed (edge count exceeds threshold).
    /// Maps to `NeighborhoodGraph::addEdge(ObjectID, ...)`.
    pub fn add_edge(
        &mut self,
        target: ObjectID,
        add_id: ObjectID,
        add_distance: f32,
    ) -> Result<bool, NgtError> {
        // Ensure node exists.
        let idx = target as usize;
        if idx >= self.nodes.len() {
            self.nodes.resize_with(idx + 1, || None);
        }
        if self.nodes[idx].is_none() {
            return Err(format!("addEdge: target node {} does not exist", target));
        }

        let minsize = if idx < self.prevsize.len() {
            self.prevsize[idx] as usize
        } else {
            0
        };

        let node = self.nodes[idx].as_mut().unwrap();
        Self::add_edge_to_node(node, add_id, add_distance, true)?;

        let needs_truncation = self.property.truncation_threshold != 0
            && node.len() - minsize > self.property.truncation_threshold;
        Ok(needs_truncation)
    }

    /// Remove the edge from `fid` to `rmid`.
    pub fn remove_edge(&mut self, fid: ObjectID, rmid: ObjectID) {
        if let Some(node) = self.get_node_mut(fid) {
            if let Some(pos) = node.iter().position(|e| { let eid: u32 = e.id; eid == rmid }) {
                node.remove(pos);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Node insertion
    // -----------------------------------------------------------------------

    /// Store an adjacency list for `id`.
    /// Maps to `GraphRepository::insert`.
    pub fn insert_node(&mut self, id: ObjectID, edges: Vec<ObjectDistance>) {
        let idx = id as usize;
        if idx >= self.nodes.len() {
            self.nodes.resize_with(idx + 1, || None);
        }
        self.nodes[idx] = Some(edges);

        if idx >= self.prevsize.len() {
            self.prevsize.resize(idx + 1, 0);
        }
        self.prevsize[idx] = 0;
    }

    /// ANNG-style insertion: store the edge list for `id`, then add
    /// back-edges from each neighbor to `id`.  Truncate if needed.
    /// Maps to `NeighborhoodGraph::insertANNGNode`.
    pub fn insert_anng_node(&mut self, id: ObjectID, results: Vec<ObjectDistance>, os: &ObjectSpace) {
        self.insert_node(id, results.clone());
        let mut truncate_queue: Vec<ObjectID> = Vec::new();
        for r in &results {
            let r_id: u32 = r.id;
            let r_dist: f32 = r.distance;
            match self.add_edge(r_id, id, r_dist) {
                Ok(needs_truncation) => {
                    if needs_truncation {
                        truncate_queue.push(r_id);
                    }
                }
                Err(_) => {}
            }
        }
        for tid in truncate_queue {
            self.truncate_edges(tid, os);
        }
    }

    /// Batch ANNG insertion: insert multiple nodes, deferring truncation
    /// to a parallel phase at the end.
    pub fn insert_anng_nodes_batch(
        &mut self,
        items: &[(ObjectID, Vec<ObjectDistance>)],
        os: &ObjectSpace,
    ) {
        let mut truncate_queue: Vec<ObjectID> = Vec::new();

        for (id, results) in items {
            self.insert_node(*id, results.clone());
            for r in results {
                let r_id: u32 = r.id;
                let r_dist: f32 = r.distance;
                match self.add_edge(r_id, *id, r_dist) {
                    Ok(needs_truncation) => {
                        if needs_truncation {
                            truncate_queue.push(r_id);
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // Deduplicate truncation targets.
        truncate_queue.sort_unstable();
        truncate_queue.dedup();

        // Parallel truncation: search phase (read-only) then apply phase (write).
        self.truncate_edges_batch(&truncate_queue, os);
    }

    /// Truncate edges for node `id` optimally.
    ///
    /// 1. Remove edges beyond `truncation_size`.
    /// 2. Remove back-edges from deleted neighbors.
    /// 3. Re-route: for each deleted neighbor, search the graph to find
    ///    an alternative node to reconnect it to.
    ///
    /// Maps to `NeighborhoodGraph::truncateEdgesOptimally`.
    /// Batch truncation: for each node in `targets`, truncate its edges and
    /// re-route deleted neighbors in parallel (matching C++ thread pool).
    fn truncate_edges_batch(&mut self, targets: &[ObjectID], os: &ObjectSpace) {
        use rayon::prelude::*;

        let truncation_size = {
            let mut ts = self.property.truncation_threshold;
            if ts < self.property.edge_size_for_creation as usize {
                ts = self.property.edge_size_for_creation as usize;
            }
            ts
        };

        // Phase 1: for each target, compute what needs deleting.
        struct TruncInfo {
            id: ObjectID,
            del_nodes: Vec<ObjectDistance>,
        }
        let mut trunc_infos: Vec<TruncInfo> = Vec::new();

        for &tid in targets {
            let del_nodes = {
                let node = match self.get_node(tid) {
                    Some(n) if n.len() > truncation_size => n,
                    _ => continue,
                };
                node[truncation_size..]
                    .iter()
                    .filter(|e| { let eid: u32 = e.id; eid != tid })
                    .copied()
                    .collect::<Vec<ObjectDistance>>()
            };

            // Truncate edges.
            if let Some(node) = self.get_node_mut(tid) {
                node.truncate(truncation_size);
            }
            // Remove back-edges.
            for del in &del_nodes {
                let del_id: u32 = del.id;
                self.remove_edge(del_id, tid);
            }

            if !del_nodes.is_empty() {
                trunc_infos.push(TruncInfo {
                    id: tid,
                    del_nodes,
                });
            }
        }

        if trunc_infos.is_empty() {
            return;
        }

        // Phase 2: parallel re-routing searches.
        // Collect ALL (source_id, deleted_neighbor) pairs to search.
        struct RerouteJob {
            source_id: ObjectID,
            del: ObjectDistance,
        }
        let mut jobs: Vec<RerouteJob> = Vec::new();
        for info in &trunc_infos {
            for del in &info.del_nodes {
                jobs.push(RerouteJob {
                    source_id: info.id,
                    del: *del,
                });
            }
        }

        // Parallel search: find nearest node for each deleted neighbor.
        let nodes_ref = &self.nodes;
        let property_ref = &self.property;

        let reroute_results: Vec<(ObjectID, ObjectDistance, ObjectDistance)> = jobs
            .par_iter()
            .map(|job| {
                let del_id: u32 = job.del.id;
                let del_dist: f32 = job.del.distance;
                let obj = match os.get_object(del_id) {
                    Ok(o) => o,
                    Err(_) => return (job.source_id, job.del, ObjectDistance::new(del_id, del_dist)),
                };
                let mut seeds = vec![ObjectDistance::new(job.source_id, del_dist)];
                let res_size = 2;
                let epsilon = 0.1;
                // Use standalone search with raw node references.
                let results = search_standalone(
                    nodes_ref, obj, &mut seeds, res_size, epsilon,
                    property_ref.edge_size_for_search, os,
                );
                let nearest = if results.is_empty() {
                    ObjectDistance::new(job.source_id, del_dist)
                } else {
                    results[0]
                };
                (job.source_id, job.del, nearest)
            })
            .collect();

        // Phase 3: serial re-routing mutations.
        for (source_id, del, nearest) in reroute_results {
            let nearest_id: u32 = nearest.id;
            let del_id: u32 = del.id;
            let nearest_dist: f32 = nearest.distance;

            if nearest_id == del_id {
                continue; // Already reachable.
            }

            // Add edge: deleted_neighbor → nearest.
            if let Some(del_node) = self.get_node_mut(del_id) {
                let _ = Self::add_edge_to_node(del_node, nearest_id, nearest_dist, false);
            }

            // Add reverse edge: nearest → deleted_neighbor.
            if nearest_id != source_id {
                if let Some(nearest_node) = self.get_node_mut(nearest_id) {
                    let _ = Self::add_edge_to_node(nearest_node, del_id, nearest_dist, false);
                }
            } else {
                if let Some(id_node) = self.get_node_mut(source_id) {
                    let _ = Self::add_edge_to_node(id_node, del_id, nearest_dist, false);
                }
            }
        }
    }

    pub fn truncate_edges(&mut self, id: ObjectID, os: &ObjectSpace) {
        let truncation_size = {
            let mut ts = self.property.truncation_threshold;
            if ts < self.property.edge_size_for_creation as usize {
                ts = self.property.edge_size_for_creation as usize;
            }
            ts
        };

        let (del_nodes, osize) = {
            let node = match self.get_node(id) {
                Some(n) if !n.is_empty() => n,
                _ => return,
            };
            if node.len() <= truncation_size {
                return;
            }
            let osize = node.len();
            let del: Vec<ObjectDistance> = node[truncation_size..]
                .iter()
                .filter(|e| { let eid: u32 = e.id; eid != id })
                .copied()
                .collect();
            (del, osize)
        };

        // Step 1: truncate the node's edge list.
        if let Some(node) = self.get_node_mut(id) {
            node.truncate(truncation_size);
        }

        // Step 2: remove back-edges from deleted neighbors → this node.
        for del in &del_nodes {
            let del_id: u32 = del.id;
            self.remove_edge(del_id, id);
        }

        // Step 3: re-route each deleted neighbor to its nearest alternative.
        // del_status: Some(od) = still needs re-routing, None = done.
        let mut del_status: Vec<Option<ObjectDistance>> =
            del_nodes.iter().map(|d| Some(*d)).collect();
        let mut res_size: usize = 2;
        let max_res_size = osize * 2;
        let exploration_coeff: f32 = 0.1; // epsilon for truncation search (C++ uses 1.1 as coefficient = 0.1 + 1.0)
        let batch_size: usize = 20;

        loop {
            let mut retry = false;
            let mut node_idx: usize = 0;

            while node_idx < del_status.len() {
                // Collect a batch of unprocessed deleted neighbors.
                let mut batch: Vec<(usize, ObjectDistance)> = Vec::new();
                while node_idx < del_status.len() && batch.len() < batch_size {
                    if let Some(del) = del_status[node_idx] {
                        batch.push((node_idx, del));
                    }
                    node_idx += 1;
                }
                if batch.is_empty() {
                    break;
                }

                // Search for nearest node to each deleted neighbor.
                // Seed = the truncated node, so search starts from there.
                let search_results: Vec<(usize, ObjectDistance, ObjectDistance)> = batch
                    .iter()
                    .map(|&(idx, del)| {
                        let del_id: u32 = del.id;
                        let del_dist: f32 = del.distance;
                        let obj = match os.get_object(del_id) {
                            Ok(o) => o,
                            Err(_) => return (idx, del, ObjectDistance::new(del_id, del_dist)),
                        };
                        let mut seeds = vec![ObjectDistance::new(id, del_dist)];
                        let results = self.search(
                            obj, &mut seeds, res_size, exploration_coeff,
                            0, f32::MAX, os,
                        );
                        let nearest = if results.is_empty() {
                            ObjectDistance::new(id, del_dist)
                        } else {
                            results[0]
                        };
                        (idx, del, nearest)
                    })
                    .collect();

                // Apply re-routing.
                let mut cannot_move_cnt: usize = 0;
                for (idx, del, nearest) in search_results {
                    let nearest_id: u32 = nearest.id;
                    let del_id: u32 = del.id;
                    let nearest_dist: f32 = nearest.distance;

                    if nearest_id == del_id {
                        // Already reachable — done.
                        del_status[idx] = None;
                        continue;
                    } else if nearest_id == id {
                        // Couldn't find alternative — nearest is the truncated node itself.
                        cannot_move_cnt += 1;
                        if res_size < max_res_size && cannot_move_cnt > 1 {
                            retry = true;
                            continue;
                        }
                    }

                    del_status[idx] = None;

                    // Add edge: deleted_neighbor → nearest.
                    if let Some(del_node) = self.get_node_mut(del_id) {
                        let _ = Self::add_edge_to_node(
                            del_node, nearest_id, nearest_dist, false,
                        );
                    }

                    // Add reverse edge: nearest → deleted_neighbor.
                    if nearest_id != id {
                        if let Some(nearest_node) = self.get_node_mut(nearest_id) {
                            let _ = Self::add_edge_to_node(
                                nearest_node, del_id, nearest_dist, false,
                            );
                        }
                    } else {
                        // Re-add to the truncated node's list.
                        if let Some(id_node) = self.get_node_mut(id) {
                            let _ = Self::add_edge_to_node(
                                id_node, del_id, nearest_dist, false,
                            );
                        }
                    }
                }
            }

            if !retry || res_size >= max_res_size {
                break;
            }
            res_size = max_res_size;
        }
    }

    // -----------------------------------------------------------------------
    // Compute edge size for search
    // -----------------------------------------------------------------------

    fn get_edge_size(&self, sc_edge_size: i32) -> usize {
        let esize: i64 = if sc_edge_size == -1 {
            self.property.edge_size_for_search as i64
        } else {
            sc_edge_size as i64
        };
        if esize == 0 {
            usize::MAX
        } else if esize > 0 {
            esize as usize
        } else {
            usize::MAX
        }
    }

    // -----------------------------------------------------------------------
    // Search  (NeighborhoodGraph::search)
    // -----------------------------------------------------------------------

    /// Graph-based greedy best-first search.
    ///
    /// `seeds`: initial entry points with pre-computed distances to query.
    /// `k`: number of results.
    /// `epsilon`: exploration coefficient offset (actual coeff = epsilon + 1.0).
    /// `edge_size`: max edges to follow per node (-1 = use property, 0 = all).
    /// `radius`: initial search radius (f32::MAX for unlimited).
    ///
    /// Returns results sorted by ascending distance.
    pub fn search(
        &self,
        query: &[f32],
        seeds: &mut Vec<ObjectDistance>,
        k: usize,
        epsilon: f32,
        edge_size: i32,
        radius: f32,
        os: &ObjectSpace,
    ) -> Vec<ObjectDistance> {
        let exploration_coefficient = if epsilon == 0.0 {
            DEFAULT_EXPLORATION_COEFFICIENT
        } else {
            epsilon + 1.0
        };

        let edge_size = self.get_edge_size(edge_size);

        // Compute distances for seeds.
        self.setup_distances(query, seeds, os);

        // Sort seeds by distance.
        seeds.sort_unstable_by(|a, b| a.cmp(b));

        // Setup results (max-heap) and unchecked (min-heap).
        let mut results: BinaryHeap<ObjectDistance> = BinaryHeap::new();
        let mut unchecked: BinaryHeap<Reverse<ObjectDistance>> = BinaryHeap::new();
        // BooleanVector matching C++ BooleanVector — O(1) lookup, no hashing.
        let mut distance_checked = BooleanVector::new(self.nodes.len());
        let mut current_radius = radius;

        // Prefetch parameters matching C++ ObjectSpace formulas.
        let padded_dim = ((os.dim.saturating_sub(1)) / 16 + 1) * 16;
        let prefetch_offset: usize =
            (300.0 / (padded_dim as f32 + 30.0) + 1.0).floor() as usize;

        // Setup seeds.
        for s in seeds.iter() {
            let s_dist: f32 = s.distance;
            let s_id: u32 = s.id;
            if results.len() < k && s_dist <= current_radius {
                results.push(*s);
            }
            if s_dist < f32::MAX {
                distance_checked.insert(s_id);
                unchecked.push(Reverse(*s));
            }
        }

        if results.len() >= k {
            if let Some(top) = results.peek() {
                current_radius = top.distance;
            }
        }

        let mut exploration_radius = exploration_coefficient * current_radius;

        // Main search loop — 1:1 with C++ NeighborhoodGraph::search.
        while let Some(Reverse(target)) = unchecked.pop() {
            let target_dist: f32 = target.distance;
            if target_dist > exploration_radius {
                break;
            }

            let target_id: u32 = target.id;
            let neighbors = match self.get_node(target_id) {
                Some(n) => n,
                None => continue,
            };
            if neighbors.is_empty() {
                continue;
            }

            let neighbor_size = neighbors.len().min(edge_size);

            // Prefetch first batch of neighbor objects.
            let poft = prefetch_offset.min(neighbor_size);
            for i in 0..poft {
                let nid: u32 = neighbors[i].id;
                if !distance_checked.contains(nid) {
                    if let Ok(obj) = os.get_object(nid) {
                        prefetch_read(obj.as_ptr());
                    }
                }
            }

            for ni in 0..neighbor_size {
                // Sliding prefetch window.
                if ni + prefetch_offset < neighbor_size {
                    let ahead_id: u32 = neighbors[ni + prefetch_offset].id;
                    if !distance_checked.contains(ahead_id) {
                        if let Ok(obj) = os.get_object(ahead_id) {
                            prefetch_read(obj.as_ptr());
                        }
                    }
                }

                let neighbor_id: u32 = neighbors[ni].id;

                if distance_checked.contains(neighbor_id) {
                    continue;
                }
                distance_checked.insert(neighbor_id);

                if !os.is_present(neighbor_id) {
                    continue;
                }

                let stored = match os.get_object(neighbor_id) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let distance = os.distance(query, stored);

                if distance <= exploration_radius {
                    let result = ObjectDistance::new(neighbor_id, distance);
                    unchecked.push(Reverse(result));

                    if distance <= current_radius {
                        results.push(result);
                        if results.len() >= k {
                            if let Some(top) = results.peek() {
                                let top_dist: f32 = top.distance;
                                if top_dist >= distance {
                                    if results.len() > k {
                                        results.pop();
                                    }
                                    current_radius = results.peek().unwrap().distance;
                                    exploration_radius =
                                        exploration_coefficient * current_radius;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert results (max-heap) to sorted vec (ascending distance).
        let mut result_vec: Vec<ObjectDistance> = results.into_vec();
        result_vec.sort_unstable_by(|a, b| a.cmp(b));
        result_vec
    }

    /// Compute distances from `query` to each seed's object.
    /// Maps to `NeighborhoodGraph::setupDistances`.
    fn setup_distances(
        &self,
        query: &[f32],
        seeds: &mut Vec<ObjectDistance>,
        os: &ObjectSpace,
    ) {
        for seed in seeds.iter_mut() {
            let seed_id: u32 = seed.id;
            if !os.is_present(seed_id) {
                seed.distance = f32::MAX;
                continue;
            }
            match os.get_object(seed_id) {
                Ok(obj) => {
                    seed.distance = os.distance(query, obj);
                }
                Err(_) => {
                    seed.distance = f32::MAX;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Binary serialization
    // -----------------------------------------------------------------------
    //
    // Format (matching GraphRepository::serialize):
    //   [Repository<GraphNode> serialization]:
    //     [8 bytes] u64 slot count
    //     Per slot: '-' or '+' + GraphNode data
    //       GraphNode data: [4 bytes] u32 edge count +
    //                       edge_count × [8 bytes] (u32 id + f32 distance)
    //   [prevsize vector]:
    //     [4 bytes] u32 length
    //     length × [2 bytes] u16 values

    pub fn serialize<W: Write>(&self, w: &mut W) -> Result<(), NgtError> {
        // Write node repository.
        let slot_count = self.nodes.len() as u64;
        w.write_all(&slot_count.to_le_bytes())
            .map_err(|e| format!("Graph::serialize slot count: {}", e))?;

        for slot in &self.nodes {
            match slot {
                None => {
                    w.write_all(&[b'-'])
                        .map_err(|e| format!("Graph::serialize: {}", e))?;
                }
                Some(edges) => {
                    w.write_all(&[b'+'])
                        .map_err(|e| format!("Graph::serialize: {}", e))?;
                    // Write edge count as u32.
                    let edge_count = edges.len() as u32;
                    w.write_all(&edge_count.to_le_bytes())
                        .map_err(|e| format!("Graph::serialize edge count: {}", e))?;
                    // Write each edge as raw ObjectDistance (u32 id + f32 dist).
                    for e in edges {
                        let eid: u32 = e.id;
                        let edist: f32 = e.distance;
                        w.write_all(&eid.to_le_bytes())
                            .map_err(|e| format!("Graph::serialize edge id: {}", e))?;
                        w.write_all(&edist.to_le_bytes())
                            .map_err(|e| format!("Graph::serialize edge dist: {}", e))?;
                    }
                }
            }
        }

        // Write prevsize vector.
        let ps_len = self.prevsize.len() as u32;
        w.write_all(&ps_len.to_le_bytes())
            .map_err(|e| format!("Graph::serialize prevsize len: {}", e))?;
        for &v in &self.prevsize {
            w.write_all(&v.to_le_bytes())
                .map_err(|e| format!("Graph::serialize prevsize: {}", e))?;
        }

        Ok(())
    }

    pub fn deserialize<R: Read>(&mut self, r: &mut R) -> Result<(), NgtError> {
        // Read node repository.
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("Graph::deserialize slot count: {}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;

        self.nodes.clear();
        let mut buf4 = [0u8; 4];
        let mut buf2 = [0u8; 2];

        for _i in 0..slot_count {
            let mut type_byte = [0u8; 1];
            r.read_exact(&mut type_byte)
                .map_err(|e| format!("Graph::deserialize type: {}", e))?;
            match type_byte[0] {
                b'-' => {
                    self.nodes.push(None);
                }
                b'+' => {
                    // Read edge count (u32).
                    r.read_exact(&mut buf4)
                        .map_err(|e| format!("Graph::deserialize edge count: {}", e))?;
                    let edge_count = u32::from_le_bytes(buf4) as usize;

                    let mut edges = Vec::with_capacity(edge_count);
                    for _ in 0..edge_count {
                        r.read_exact(&mut buf4)
                            .map_err(|e| format!("Graph::deserialize edge id: {}", e))?;
                        let id = u32::from_le_bytes(buf4);
                        r.read_exact(&mut buf4)
                            .map_err(|e| format!("Graph::deserialize edge dist: {}", e))?;
                        let dist = f32::from_le_bytes(buf4);
                        edges.push(ObjectDistance::new(id, dist));
                    }
                    self.nodes.push(Some(edges));
                }
                _ => {
                    return Err(format!(
                        "Graph::deserialize: unexpected type byte {:?}",
                        type_byte[0] as char
                    ));
                }
            }
        }

        // Read prevsize vector.
        r.read_exact(&mut buf4)
            .map_err(|e| format!("Graph::deserialize prevsize len: {}", e))?;
        let ps_len = u32::from_le_bytes(buf4) as usize;

        self.prevsize.clear();
        self.prevsize.reserve(ps_len);
        for _ in 0..ps_len {
            r.read_exact(&mut buf2)
                .map_err(|e| format!("Graph::deserialize prevsize: {}", e))?;
            self.prevsize.push(u16::from_le_bytes(buf2));
        }

        Ok(())
    }

    pub fn serialize_to_file(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("Graph::serialize_to_file: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        self.serialize(&mut w)
    }

    pub fn deserialize_from_file(&mut self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("Graph::deserialize_from_file: {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        self.deserialize(&mut r)
    }
}

impl Default for NeighborhoodGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Standalone search — used for parallel truncation (avoids &self borrow)
// ---------------------------------------------------------------------------

fn search_standalone(
    nodes: &[Option<Vec<ObjectDistance>>],
    query: &[f32],
    seeds: &mut Vec<ObjectDistance>,
    k: usize,
    epsilon: f32,
    edge_size_for_search: i32,
    os: &ObjectSpace,
) -> Vec<ObjectDistance> {
    let exploration_coefficient = if epsilon == 0.0 {
        DEFAULT_EXPLORATION_COEFFICIENT
    } else {
        epsilon + 1.0
    };

    let edge_size: usize = if edge_size_for_search <= 0 {
        usize::MAX
    } else {
        edge_size_for_search as usize
    };

    // Compute distances for seeds.
    for seed in seeds.iter_mut() {
        let sid: u32 = seed.id;
        match os.get_object(sid) {
            Ok(obj) => seed.distance = os.distance(query, obj),
            Err(_) => seed.distance = f32::MAX,
        }
    }

    seeds.sort_unstable_by(|a, b| a.cmp(b));

    let mut results: BinaryHeap<ObjectDistance> = BinaryHeap::new();
    let mut unchecked: BinaryHeap<Reverse<ObjectDistance>> = BinaryHeap::new();
    // Use HashSet for small k (truncation searches) to avoid allocating
    // a full BooleanVector(1M+) for a search that visits ~20 nodes.
    let mut distance_checked = std::collections::HashSet::<u32>::with_capacity(64);
    let mut current_radius = f32::MAX;

    for s in seeds.iter() {
        let s_dist: f32 = s.distance;
        let s_id: u32 = s.id;
        if results.len() < k && s_dist <= current_radius {
            results.push(*s);
        }
        if s_dist < f32::MAX {
            distance_checked.insert(s_id);
            unchecked.push(Reverse(*s));
        }
    }

    if results.len() >= k {
        if let Some(top) = results.peek() {
            current_radius = top.distance;
        }
    }

    let mut exploration_radius = exploration_coefficient * current_radius;

    while let Some(Reverse(target)) = unchecked.pop() {
        let target_dist: f32 = target.distance;
        if target_dist > exploration_radius {
            break;
        }
        let target_id: u32 = target.id;
        let neighbors = match nodes.get(target_id as usize).and_then(|n| n.as_ref()) {
            Some(n) => n,
            None => continue,
        };
        if neighbors.is_empty() {
            continue;
        }
        let neighbor_size = neighbors.len().min(edge_size);

        for ni in 0..neighbor_size {
            let neighbor_id: u32 = neighbors[ni].id;
            if distance_checked.contains(&neighbor_id) {
                continue;
            }
            distance_checked.insert(neighbor_id);

            if !os.is_present(neighbor_id) {
                continue;
            }
            let stored = match os.get_object(neighbor_id) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let distance = os.distance(query, stored);
            if distance <= exploration_radius {
                let result = ObjectDistance::new(neighbor_id, distance);
                unchecked.push(Reverse(result));
                if distance <= current_radius {
                    results.push(result);
                    if results.len() >= k {
                        if let Some(top) = results.peek() {
                            let top_dist: f32 = top.distance;
                            if top_dist >= distance {
                                if results.len() > k {
                                    results.pop();
                                }
                                current_radius = results.peek().unwrap().distance;
                                exploration_radius = exploration_coefficient * current_radius;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result_vec: Vec<ObjectDistance> = results.into_vec();
    result_vec.sort_unstable_by(|a, b| a.cmp(b));
    result_vec
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive_comparator::DistanceType;

    fn make_os(dim: usize, vecs: &[Vec<f32>]) -> ObjectSpace {
        let mut os = ObjectSpace::new(dim, DistanceType::L2);
        for v in vecs {
            os.insert(v).unwrap();
        }
        os
    }

    #[test]
    fn test_insert_and_get_node() {
        let mut g = NeighborhoodGraph::new();
        let edges = vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(3, 2.0),
        ];
        g.insert_node(1, edges);
        let node = g.get_node(1).unwrap();
        assert_eq!(node.len(), 2);
        let id0: u32 = node[0].id;
        assert_eq!(id0, 2);
    }

    #[test]
    fn test_add_edge_sorted() {
        let mut g = NeighborhoodGraph::new();
        g.insert_node(1, vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(4, 3.0),
        ]);
        g.add_edge(1, 3, 2.0).unwrap();
        let node = g.get_node(1).unwrap();
        assert_eq!(node.len(), 3);
        let ids: Vec<u32> = node.iter().map(|e| e.id).collect();
        // Should be sorted by distance: 2(1.0), 3(2.0), 4(3.0)
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn test_insert_anng_node() {
        let os = make_os(2, &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut g = NeighborhoodGraph::new();
        // Pre-create nodes 2 and 3 so back-edges can be added.
        g.insert_node(2, vec![]);
        g.insert_node(3, vec![]);

        let results = vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(3, 2.0),
        ];
        g.insert_anng_node(1, results, &os);

        // Node 1 should have edges to 2 and 3.
        assert_eq!(g.get_node(1).unwrap().len(), 2);
        // Node 2 should have back-edge to 1.
        assert_eq!(g.get_node(2).unwrap().len(), 1);
        let back_id: u32 = g.get_node(2).unwrap()[0].id;
        assert_eq!(back_id, 1);
    }

    #[test]
    fn test_search_basic() {
        // Build a small graph manually.
        let vecs: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ];
        let os = make_os(2, &vecs);

        let mut g = NeighborhoodGraph::new();
        // Object IDs 1..=5.  Build a fully connected graph.
        for id in 1..=5u32 {
            let mut edges = Vec::new();
            for other in 1..=5u32 {
                if other != id {
                    let a = os.get_object(id).unwrap();
                    let b = os.get_object(other).unwrap();
                    let d = os.distance(a, b);
                    edges.push(ObjectDistance::new(other, d));
                }
            }
            edges.sort_by(|a, b| a.cmp(b));
            g.insert_node(id, edges);
        }

        // Search for [0.9, 0.1] — closest should be object 2 [1.0, 0.0].
        let mut seeds = vec![ObjectDistance::new(1, 0.0)];
        let results = g.search(&[0.9, 0.1], &mut seeds, 1, 0.1, 0, f32::MAX, &os);
        assert!(!results.is_empty());
        let best_id: u32 = results[0].id;
        assert_eq!(best_id, 2);
    }

    #[test]
    fn test_remove_edge() {
        let mut g = NeighborhoodGraph::new();
        g.insert_node(1, vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(3, 2.0),
        ]);
        g.remove_edge(1, 2);
        assert_eq!(g.get_node(1).unwrap().len(), 1);
        let id0: u32 = g.get_node(1).unwrap()[0].id;
        assert_eq!(id0, 3);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut g = NeighborhoodGraph::new();
        g.insert_node(1, vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(3, 2.0),
        ]);
        g.insert_node(2, vec![ObjectDistance::new(1, 1.0)]);

        let mut buf = Vec::new();
        g.serialize(&mut buf).unwrap();

        let mut g2 = NeighborhoodGraph::new();
        let mut cursor = std::io::Cursor::new(&buf);
        g2.deserialize(&mut cursor).unwrap();

        assert_eq!(g2.nodes.len(), g.nodes.len());
        let n1 = g2.get_node(1).unwrap();
        assert_eq!(n1.len(), 2);
        let id0: u32 = n1[0].id;
        assert_eq!(id0, 2);
    }

    #[test]
    fn test_truncate_edges() {
        let os = make_os(2, &[
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 1.0], vec![2.0, 2.0],
        ]);
        let mut g = NeighborhoodGraph::new();
        g.property.truncation_threshold = 2;
        g.property.edge_size_for_creation = 2;

        // Create node 1 with 4 edges.
        g.insert_node(1, vec![
            ObjectDistance::new(2, 1.0),
            ObjectDistance::new(3, 2.0),
            ObjectDistance::new(4, 3.0),
            ObjectDistance::new(5, 4.0),
        ]);
        // Create targets so back-edge removal doesn't panic.
        g.insert_node(2, vec![ObjectDistance::new(1, 1.0)]);
        g.insert_node(3, vec![ObjectDistance::new(1, 2.0)]);
        g.insert_node(4, vec![ObjectDistance::new(1, 3.0)]);
        g.insert_node(5, vec![ObjectDistance::new(1, 4.0)]);

        g.truncate_edges(1, &os);
        // Node 1 should be truncated, but re-routing may add some edges back.
        // At minimum: back-edges from 4 and 5 to node 1 should be removed.
        assert!(g.get_node(1).unwrap().len() >= 2);
        let n4_has_back = g.get_node(4).unwrap().iter().any(|e| { let eid: u32 = e.id; eid == 1 });
        let n5_has_back = g.get_node(5).unwrap().iter().any(|e| { let eid: u32 = e.id; eid == 1 });
        assert!(!n4_has_back, "back-edge 4->1 should be removed");
        assert!(!n5_has_back, "back-edge 5->1 should be removed");
    }
}
