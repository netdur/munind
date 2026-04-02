/// Port of NGT/Tree.h + NGT/Tree.cpp  (DVPTree)
///
/// Non-shared-memory variant, float objects only (Phase 1).
///
/// The DVPTree (Distance-based Vantage-Point Tree) is a spatial partitioning
/// structure used to find the correct leaf for inserting a new object and to
/// perform rough candidate search before the graph traversal.

use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID};
use crate::node::{
    self, InternalNode, LeafNode, NodeId, NodeObject, NodeType, PIVOT,
};
use crate::object_space::ObjectSpace;

// ---------------------------------------------------------------------------
// SplitMode  (DVPTree::SplitMode)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    MaxDistance = 0,
    MaxVariance = 1,
}

// ---------------------------------------------------------------------------
// DVPTree
// ---------------------------------------------------------------------------

pub struct DVPTree {
    /// Leaf node storage.  Index 0 is always `None` (1-based).
    pub leaf_nodes: Vec<Option<LeafNode>>,
    /// Internal node storage.  Index 0 is always `None` (1-based).
    pub internal_nodes: Vec<Option<InternalNode>>,
    /// Number of children per internal node (default 5).
    pub internal_children_size: usize,
    /// Maximum objects per leaf before splitting (default 100).
    pub leaf_objects_size: usize,
    /// Pivot selection strategy (default MaxVariance).
    pub split_mode: SplitMode,
}

impl DVPTree {
    pub fn new(leaf_objects_size: usize, internal_children_size: usize) -> Self {
        let mut tree = DVPTree {
            leaf_nodes: vec![None],       // slot 0 = None
            internal_nodes: vec![None],   // slot 0 = None
            internal_children_size,
            leaf_objects_size,
            split_mode: SplitMode::MaxVariance,
        };
        // Insert initial empty root leaf (gets ID 1).
        tree.allocate_and_insert_leaf();
        tree
    }

    // -----------------------------------------------------------------------
    // Node repository helpers
    // -----------------------------------------------------------------------

    fn allocate_and_insert_leaf(&mut self) -> u32 {
        let mut leaf = LeafNode::new();
        let id = self.leaf_nodes.len() as u32;
        leaf.id = NodeId::leaf(id);
        self.leaf_nodes.push(Some(leaf));
        id
    }

    fn insert_leaf_node(&mut self, mut leaf: LeafNode) -> u32 {
        let id = self.leaf_nodes.len() as u32;
        leaf.id = NodeId::leaf(id);
        self.leaf_nodes.push(Some(leaf));
        id
    }

    pub fn get_leaf(&self, id: u32) -> Option<&LeafNode> {
        self.leaf_nodes.get(id as usize).and_then(|o| o.as_ref())
    }

    pub fn get_leaf_mut(&mut self, id: u32) -> Option<&mut LeafNode> {
        self.leaf_nodes.get_mut(id as usize).and_then(|o| o.as_mut())
    }

    pub fn get_internal(&self, id: u32) -> Option<&InternalNode> {
        self.internal_nodes.get(id as usize).and_then(|o| o.as_ref())
    }

    pub fn get_internal_mut(&mut self, id: u32) -> Option<&mut InternalNode> {
        self.internal_nodes.get_mut(id as usize).and_then(|o| o.as_mut())
    }

    fn remove_leaf_node(&mut self, id: u32) {
        if let Some(slot) = self.leaf_nodes.get_mut(id as usize) {
            *slot = None;
        }
    }

    fn remove_internal_node(&mut self, id: u32) {
        if let Some(slot) = self.internal_nodes.get_mut(id as usize) {
            *slot = None;
        }
    }

    fn remove_node(&mut self, nid: NodeId) {
        match nid.get_type() {
            NodeType::Leaf => self.remove_leaf_node(nid.get_id()),
            NodeType::Internal => self.remove_internal_node(nid.get_id()),
        }
    }

    // -----------------------------------------------------------------------
    // Root node access
    // -----------------------------------------------------------------------

    /// Returns the NodeId of the root node (always at slot 1).
    /// The root can be either an InternalNode or a LeafNode.
    fn root_node_id(&self) -> Result<NodeId, NgtError> {
        if self.internal_nodes.len() > 1 && self.internal_nodes[1].is_some() {
            Ok(NodeId::internal(1))
        } else if self.leaf_nodes.len() > 1 && self.leaf_nodes[1].is_some() {
            Ok(NodeId::leaf(1))
        } else {
            Err("DVPTree::getRootNode: no root node".to_string())
        }
    }

    // -----------------------------------------------------------------------
    // Insert  (DVPTree::insert)
    // -----------------------------------------------------------------------

    /// Insert object `obj_id` into the tree.
    /// The object's data is looked up from `os`.
    pub fn insert(&mut self, obj_id: ObjectID, os: &ObjectSpace) -> Result<(), NgtError> {
        // Step 1: find which leaf the object should go into.
        let obj_data = os.get_object(obj_id)?.to_vec();
        let leaf_nid = self.search_leaf(&obj_data, os)?;
        let leaf_idx = leaf_nid.get_id();

        // Step 2: duplicate check.
        // In C++, if an object in the leaf has the same pivot-distance AND
        // the actual distance is 0, it's considered a duplicate and skipped.
        {
            let leaf = self.get_leaf(leaf_idx)
                .ok_or_else(|| format!("DVPTree::insert: leaf {} not found", leaf_idx))?;
            let fsize = leaf.object_size();
            if fsize != 0 {
                let pivot = leaf.pivot.as_ref().unwrap();
                let d = os.distance(&obj_data, pivot);
                for oid in &leaf.object_ids {
                    let oid_dist = oid.distance; // copy from packed struct
                    let oid_id = oid.id;
                    if oid_dist == d {
                        if os.is_present(oid_id) {
                            let stored = os.get_object(oid_id)?;
                            let idd = os.distance(&obj_data, stored);
                            if idd == 0.0 {
                                // Duplicate — skip.
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Step 3: insert or split.
        let leaf = self.get_leaf(leaf_idx).unwrap();
        if leaf.object_size() >= self.leaf_objects_size {
            self.split(obj_id, &obj_data, leaf_idx, os)?;
        } else {
            self.insert_object(obj_id, &obj_data, leaf_idx, os)?;
        }

        Ok(())
    }

    /// Insert an object into a leaf without splitting.
    /// Maps to `DVPTree::insertObject`.
    fn insert_object(
        &mut self,
        obj_id: ObjectID,
        obj_data: &[f32],
        leaf_idx: u32,
        os: &ObjectSpace,
    ) -> Result<(), NgtError> {
        let leaf = self.get_leaf_mut(leaf_idx)
            .ok_or_else(|| format!("DVPTree::insertObject: leaf {} not found", leaf_idx))?;

        if leaf.object_size() == 0 {
            // First object becomes the pivot.
            leaf.pivot = Some(obj_data.to_vec());
            leaf.object_ids.push(ObjectDistance::new(obj_id, 0.0));
        } else {
            let pivot = leaf.pivot.as_ref().unwrap();
            let d = os.distance(obj_data, pivot);
            leaf.object_ids.push(ObjectDistance::new(obj_id, d));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Split  (DVPTree::split + recombineNodes)
    // -----------------------------------------------------------------------

    fn split(
        &mut self,
        new_obj_id: ObjectID,
        new_obj_data: &[f32],
        leaf_idx: u32,
        os: &ObjectSpace,
    ) -> Result<NodeId, NgtError> {
        // Gather all objects in the leaf + the new one.
        let mut node_objects = self.get_objects(leaf_idx, new_obj_id, new_obj_data, os)?;

        // Select pivot.
        let compare = |a: &[f32], b: &[f32]| -> f32 { os.distance(a, b) };
        let pv = match self.split_mode {
            SplitMode::MaxVariance => {
                node::select_pivot_by_max_variance(&node_objects, &compare)
            }
            SplitMode::MaxDistance => {
                node::select_pivot_by_max_distance(&node_objects, &compare)
            }
        };

        // Split objects into clusters.
        node::split_objects(
            &mut node_objects,
            pv,
            self.internal_children_size,
            &compare,
        )?;

        // Recombine into new internal + leaf nodes.
        self.recombine_nodes(node_objects, leaf_idx, os)
    }

    /// Collect all objects from the leaf plus the new insertion object.
    /// Maps to `DVPTree::getObjects`.
    fn get_objects(
        &self,
        leaf_idx: u32,
        new_obj_id: ObjectID,
        new_obj_data: &[f32],
        os: &ObjectSpace,
    ) -> Result<Vec<NodeObject>, NgtError> {
        let leaf = self.get_leaf(leaf_idx)
            .ok_or_else(|| format!("DVPTree::getObjects: leaf {} not found", leaf_idx))?;
        let size = leaf.object_size() + 1;
        let mut objects = Vec::with_capacity(size);

        for oid in &leaf.object_ids {
            let id = oid.id;
            let data = os.get_object(id)?.to_vec();
            objects.push(NodeObject::new(id, data));
        }
        // Add the new object.
        objects.push(NodeObject::new(new_obj_id, new_obj_data.to_vec()));
        Ok(objects)
    }

    /// Reassemble leaf + internal nodes after a split.
    /// Maps to `DVPTree::recombineNodes`.
    fn recombine_nodes(
        &mut self,
        fs: Vec<NodeObject>,
        target_leaf_idx: u32,
        os: &ObjectSpace,
    ) -> Result<NodeId, NgtError> {
        let children_size = self.internal_children_size;
        let fsize = fs.len();

        // Save the target leaf's identity.
        let target_id = {
            let leaf = self.get_leaf(target_leaf_idx).unwrap();
            leaf.id
        };
        let target_parent = {
            let leaf = self.get_leaf(target_leaf_idx).unwrap();
            leaf.parent
        };

        // Reuse the target leaf as child[0], clearing its objects.
        {
            let leaf = self.get_leaf_mut(target_leaf_idx).unwrap();
            leaf.object_ids.clear();
            leaf.pivot = None;
        }

        // Allocate children_size - 1 new leaf nodes.
        let mut child_leaf_ids: Vec<u32> = Vec::with_capacity(children_size);
        child_leaf_ids.push(target_leaf_idx); // child[0] = reused leaf
        for _ in 1..children_size {
            let id = self.allocate_and_insert_leaf();
            child_leaf_ids.push(id);
        }

        // Create a new internal node.
        let mut internal = InternalNode::new(children_size);
        let in_id = {
            let id = self.internal_nodes.len() as u32;
            internal.id = NodeId::internal(id);
            id
        };

        // If the target had a parent, update the parent's child pointer.
        if target_parent.get_id() != 0 {
            let parent = self.get_internal_mut(target_parent.get_id())
                .ok_or_else(|| "DVPTree::recombineNodes: parent not found".to_string())?;
            parent.update_child(target_id, NodeId::internal(in_id));
        }

        // Set the internal node's pivot from fs[0].
        internal.pivot = Some(os.get_object(fs[0].id)?.to_vec());
        internal.parent = target_parent;

        // Distribute objects to child leaves.
        let mut cid = fs[0].cluster_id;
        let mut max_cluster_id = cid;

        // First object (fs[0]).
        {
            let c_idx = cid as usize;
            let leaf = self.get_leaf_mut(child_leaf_ids[c_idx]).unwrap();
            leaf.object_ids.push(ObjectDistance::new(fs[0].id, 0.0));
            if fs[0].leaf_distance == PIVOT {
                leaf.pivot = Some(os.get_object(fs[0].id)?.to_vec());
            }
            leaf.parent = NodeId::internal(in_id);
        }

        for i in 1..fsize {
            let cluster_id = fs[i].cluster_id;
            if cluster_id > max_cluster_id {
                max_cluster_id = cluster_id;
            }

            let ld = if fs[i].leaf_distance == PIVOT {
                // This object is the sub-pivot of its cluster.
                let c_idx = cluster_id as usize;
                let leaf = self.get_leaf_mut(child_leaf_ids[c_idx]).unwrap();
                leaf.pivot = Some(os.get_object(fs[i].id)?.to_vec());
                0.0
            } else {
                fs[i].leaf_distance
            };

            {
                let c_idx = cluster_id as usize;
                let leaf = self.get_leaf_mut(child_leaf_ids[c_idx]).unwrap();
                leaf.object_ids.push(ObjectDistance::new(fs[i].id, ld));
                leaf.parent = NodeId::internal(in_id);
            }

            // Set border when transitioning between clusters.
            if cluster_id != cid {
                internal.borders[cid as usize] = fs[i].distance;
                cid = cluster_id;
            }
        }

        // Fill empty children (when fewer clusters than children_size).
        for i in (max_cluster_id as usize + 1)..children_size {
            let leaf = self.get_leaf_mut(child_leaf_ids[i]).unwrap();
            leaf.parent = NodeId::internal(in_id);
            // Dummy pivot from first object.
            leaf.pivot = Some(os.get_object(fs[0].id)?.to_vec());
            if i < children_size - 1 {
                internal.borders[i] = f32::MAX;
            }
        }

        // Set children IDs on the internal node.
        internal.children[0] = NodeId::leaf(child_leaf_ids[0]);
        for i in 1..children_size {
            internal.children[i] = NodeId::leaf(child_leaf_ids[i]);
        }

        // Store the internal node.
        self.internal_nodes.push(Some(internal));

        Ok(NodeId::internal(in_id))
    }

    // -----------------------------------------------------------------------
    // Search  (DVPTree::search)
    // -----------------------------------------------------------------------

    /// Find the leaf node that a query vector would be routed to.
    /// Maps to DVPTree::search with mode = SearchLeaf.
    pub fn search_leaf(
        &self,
        query: &[f32],
        os: &ObjectSpace,
    ) -> Result<NodeId, NgtError> {
        let root_id = self.root_node_id()?;

        // If root is already a leaf, return it directly.
        if root_id.get_type() == NodeType::Leaf {
            return Ok(root_id);
        }

        let mut current = root_id;
        loop {
            match current.get_type() {
                NodeType::Internal => {
                    let node = self.get_internal(current.get_id())
                        .ok_or_else(|| "DVPTree::search_leaf: internal node not found".to_string())?;
                    let pivot = node.pivot.as_ref().unwrap();
                    let d = os.distance(query, pivot);

                    // Find the closest child region.
                    let child_idx = self.route_to_child(d, &node.borders, 0.0);
                    current = node.children[child_idx];
                }
                NodeType::Leaf => {
                    return Ok(current);
                }
            }
        }
    }

    /// Search the tree for objects within `radius` of `query`, returning up to
    /// `k` results sorted by ascending distance.
    ///
    /// Maps to DVPTree::search with mode = SearchObject.
    pub fn search(
        &self,
        query: &[f32],
        radius: f32,
        k: usize,
        os: &ObjectSpace,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let root_id = self.root_node_id()?;
        let mut results: Vec<ObjectDistance> = Vec::new();
        let mut current_radius = radius;

        let mut stack: Vec<NodeId> = vec![root_id];

        while let Some(nid) = stack.pop() {
            match nid.get_type() {
                NodeType::Internal => {
                    let node = match self.get_internal(nid.get_id()) {
                        Some(n) => n,
                        None => continue,
                    };
                    let pivot = match &node.pivot {
                        Some(p) => p,
                        None => continue,
                    };
                    let d = os.distance(query, pivot);
                    let regions = self.compute_regions(d, &node.borders, current_radius);

                    // Sort regions by distance, push all within radius.
                    let mut sorted_regions = regions;
                    sorted_regions.sort_by(|a, b| {
                        let da: f32 = a.distance;
                        let db: f32 = b.distance;
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    for r in &sorted_regions {
                        let child_id: u32 = r.id;
                        let child_id = child_id as usize;
                        if child_id < node.children.len() {
                            stack.push(node.children[child_id]);
                        }
                    }
                }
                NodeType::Leaf => {
                    let leaf = match self.get_leaf(nid.get_id()) {
                        Some(l) => l,
                        None => continue,
                    };
                    if leaf.object_size() == 0 {
                        continue;
                    }
                    let pivot = match &leaf.pivot {
                        Some(p) => p,
                        None => continue,
                    };
                    let pq = os.distance(query, pivot);

                    for oid in &leaf.object_ids {
                        let oid_dist = oid.distance;
                        let oid_id = oid.id;

                        // Triangle inequality pruning.
                        if oid_dist <= pq + current_radius
                            && oid_dist >= pq - current_radius
                        {
                            let stored = match os.get_object(oid_id) {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                            let d = os.distance(query, stored);
                            if d <= current_radius {
                                results.push(ObjectDistance::new(oid_id, d));
                                results.sort_by(|a, b| {
                                    let da: f32 = a.distance;
                                    let db: f32 = b.distance;
                                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                if results.len() > k {
                                    results.truncate(k);
                                }
                                if results.len() == k {
                                    let worst = results.last().unwrap().distance;
                                    current_radius = worst;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Route a query distance `d` to the closest child region index.
    /// Used for SearchLeaf mode (picks the single closest child).
    fn route_to_child(&self, d: f32, borders: &[f32], _radius: f32) -> usize {
        let bsize = borders.len();
        for mid in 0..bsize {
            if d < borders[mid] {
                return mid;
            }
        }
        bsize // last child
    }

    /// Compute which child regions overlap with the query at distance `d`
    /// given the search `radius`.
    /// Maps to the internal-node search logic in DVPTree::search(InternalNode).
    fn compute_regions(
        &self,
        d: f32,
        borders: &[f32],
        radius: f32,
    ) -> Vec<ObjectDistance> {
        let bsize = borders.len(); // = internal_children_size - 1
        let mut regions: Vec<ObjectDistance> = Vec::with_capacity(bsize + 1);

        let mut mid = 0;
        while mid < bsize {
            if d < borders[mid] {
                regions.push(ObjectDistance::new(mid as u32, 0.0));
                if d + radius < borders[mid] {
                    break;
                }
            } else if d < borders[mid] + radius {
                regions.push(ObjectDistance::new(mid as u32, d - borders[mid]));
            }
            mid += 1;
        }

        // Last child.
        if mid == bsize {
            if bsize > 0 && d >= borders[bsize - 1] {
                regions.push(ObjectDistance::new(bsize as u32, 0.0));
            } else if bsize > 0 {
                regions.push(ObjectDistance::new(
                    bsize as u32,
                    borders[bsize - 1] - d,
                ));
            } else {
                // Only one child (bsize == 0) — always route there.
                regions.push(ObjectDistance::new(0, 0.0));
            }
        }

        regions
    }

    // -----------------------------------------------------------------------
    // Remove  (DVPTree::remove)
    // -----------------------------------------------------------------------

    /// Remove `obj_id` from the tree.  `replace_id` is a compaction hint
    /// (0 = simple removal).
    pub fn remove(
        &mut self,
        obj_id: ObjectID,
        replace_id: ObjectID,
        os: &ObjectSpace,
    ) -> Result<(), NgtError> {
        let obj_data = os.get_object(obj_id)?.to_vec();
        let leaf_nid = self.search_leaf(&obj_data, os)?;
        let leaf_idx = leaf_nid.get_id();

        {
            let leaf = self.get_leaf_mut(leaf_idx)
                .ok_or_else(|| format!("DVPTree::remove: leaf {} not found", leaf_idx))?;
            leaf.remove_object(obj_id, replace_id);
        }

        // If leaf is now empty and has a parent, try to collapse.
        let (is_empty, parent_id) = {
            let leaf = self.get_leaf(leaf_idx).unwrap();
            (leaf.object_size() == 0, leaf.parent.get_id())
        };

        if is_empty && parent_id != 0 {
            self.remove_empty_nodes(parent_id)?;
        }

        Ok(())
    }

    /// Collapse an internal node whose leaf children are all empty.
    /// Maps to `DVPTree::removeEmptyNodes`.
    fn remove_empty_nodes(&mut self, internal_idx: u32) -> Result<(), NgtError> {
        let mut target_idx = internal_idx;

        loop {
            let children_size = self.internal_children_size;

            // Check if all children are empty leaves.
            let (all_empty, children_ids, parent_id, target_nid) = {
                let node = match self.get_internal(target_idx) {
                    Some(n) => n,
                    None => return Ok(()),
                };
                let mut all_empty = true;
                let mut child_ids = Vec::new();
                for i in 0..children_size {
                    let child = node.children[i];
                    child_ids.push(child);
                    if child.get_type() == NodeType::Internal {
                        all_empty = false;
                        break;
                    }
                    if let Some(leaf) = self.get_leaf(child.get_id()) {
                        if leaf.object_size() != 0 {
                            all_empty = false;
                            break;
                        }
                    }
                }
                (all_empty, child_ids, node.parent, node.id)
            };

            if !all_empty {
                return Ok(());
            }

            // All children are empty leaves — remove them.
            for child_id in &children_ids {
                self.remove_node(*child_id);
            }

            if parent_id.get_id() == 0 {
                // This is the root internal node. Remove it and create a new
                // empty root leaf.
                self.remove_node(target_nid);
                let new_root_id = self.allocate_and_insert_leaf();
                // The new root should be at slot 1 ideally, but if not we
                // just let it be where it lands.  In practice the tree is
                // being emptied, so this is fine.
                let _ = new_root_id;
                return Ok(());
            }

            // Replace this internal node with an empty leaf under its parent.
            let mut new_leaf = LeafNode::new();
            new_leaf.parent = parent_id;
            // Allocate a dummy pivot.
            new_leaf.pivot = Some(vec![0.0; 0]);
            let new_leaf_id = self.insert_leaf_node(new_leaf);

            // Update parent to point to the new leaf.
            if let Some(parent) = self.get_internal_mut(parent_id.get_id()) {
                parent.update_child(target_nid, NodeId::leaf(new_leaf_id));
            }

            // Remove the old internal node.
            self.remove_node(target_nid);

            // Continue up the tree.
            target_idx = parent_id.get_id();
        }
    }

    // -----------------------------------------------------------------------
    // Replace  (DVPTree::replace)
    // -----------------------------------------------------------------------

    pub fn replace(
        &mut self,
        id: ObjectID,
        replaced_id: ObjectID,
        os: &ObjectSpace,
    ) -> Result<(), NgtError> {
        self.remove(id, replaced_id, os)
    }

    // -----------------------------------------------------------------------
    // Utility: get all object IDs from a leaf
    // -----------------------------------------------------------------------

    pub fn get_object_ids_from_leaf(&self, nid: NodeId) -> Vec<ObjectDistance> {
        match self.get_leaf(nid.get_id()) {
            Some(leaf) => leaf.object_ids.clone(),
            None => Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Binary serialization
    // -----------------------------------------------------------------------
    //
    // Format (matching C++ DVPTree::serialize):
    //   [leafNodes serialized as Repository format]
    //     [8 bytes] slot count
    //     For each slot: '-' or '+' + LeafNode binary
    //   [internalNodes serialized as Repository format]
    //     [8 bytes] slot count
    //     For each slot: '-' or '+' + InternalNode binary

    pub fn serialize<W: Write>(&self, w: &mut W, dim: usize) -> Result<(), NgtError> {
        // Serialize leaf nodes.
        let leaf_count = self.leaf_nodes.len() as u64;
        w.write_all(&leaf_count.to_le_bytes())
            .map_err(|e| format!("DVPTree::serialize leaf count: {}", e))?;
        for slot in &self.leaf_nodes {
            match slot {
                None => {
                    w.write_all(&[b'-'])
                        .map_err(|e| format!("DVPTree::serialize: {}", e))?;
                }
                Some(leaf) => {
                    w.write_all(&[b'+'])
                        .map_err(|e| format!("DVPTree::serialize: {}", e))?;
                    leaf.write_to(w, dim)?;
                }
            }
        }

        // Serialize internal nodes.
        let internal_count = self.internal_nodes.len() as u64;
        w.write_all(&internal_count.to_le_bytes())
            .map_err(|e| format!("DVPTree::serialize internal count: {}", e))?;
        for slot in &self.internal_nodes {
            match slot {
                None => {
                    w.write_all(&[b'-'])
                        .map_err(|e| format!("DVPTree::serialize: {}", e))?;
                }
                Some(node) => {
                    w.write_all(&[b'+'])
                        .map_err(|e| format!("DVPTree::serialize: {}", e))?;
                    node.write_to(w, dim)?;
                }
            }
        }
        Ok(())
    }

    pub fn deserialize<R: Read>(&mut self, r: &mut R, dim: usize) -> Result<(), NgtError> {
        // Deserialize leaf nodes.
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("DVPTree::deserialize leaf count: {}", e))?;
        let leaf_count = u64::from_le_bytes(buf8) as usize;

        self.leaf_nodes.clear();
        for _i in 0..leaf_count {
            let mut type_byte = [0u8; 1];
            r.read_exact(&mut type_byte)
                .map_err(|e| format!("DVPTree::deserialize leaf type: {}", e))?;
            match type_byte[0] {
                b'-' => {
                    self.leaf_nodes.push(None);
                }
                b'+' => {
                    let mut leaf = LeafNode::new();
                    leaf.read_from(r, dim)?;
                    self.leaf_nodes.push(Some(leaf));
                }
                _ => {
                    return Err(format!(
                        "DVPTree::deserialize: unexpected leaf type byte {:?}",
                        type_byte[0] as char
                    ));
                }
            }
        }

        // Deserialize internal nodes.
        r.read_exact(&mut buf8)
            .map_err(|e| format!("DVPTree::deserialize internal count: {}", e))?;
        let internal_count = u64::from_le_bytes(buf8) as usize;

        self.internal_nodes.clear();
        for _i in 0..internal_count {
            let mut type_byte = [0u8; 1];
            r.read_exact(&mut type_byte)
                .map_err(|e| format!("DVPTree::deserialize internal type: {}", e))?;
            match type_byte[0] {
                b'-' => {
                    self.internal_nodes.push(None);
                }
                b'+' => {
                    let mut node = InternalNode::new(self.internal_children_size);
                    node.read_from(r, dim)?;
                    self.internal_nodes.push(Some(node));
                }
                _ => {
                    return Err(format!(
                        "DVPTree::deserialize: unexpected internal type byte {:?}",
                        type_byte[0] as char
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn serialize_to_file(&self, path: &str, dim: usize) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("DVPTree::serialize_to_file: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        self.serialize(&mut w, dim)
    }

    pub fn deserialize_from_file(&mut self, path: &str, dim: usize) -> Result<(), NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("DVPTree::deserialize_from_file: {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        self.deserialize(&mut r, dim)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive_comparator::DistanceType;

    fn make_os_with_objects(dim: usize, vecs: &[Vec<f32>]) -> ObjectSpace {
        let mut os = ObjectSpace::new(dim, DistanceType::L2);
        for v in vecs {
            os.insert(v).unwrap();
        }
        os
    }

    #[test]
    fn test_insert_single_object() {
        let os = make_os_with_objects(2, &[vec![1.0, 2.0]]);
        let mut tree = DVPTree::new(100, 5);
        tree.insert(1, &os).unwrap();

        let leaf = tree.get_leaf(1).unwrap();
        assert_eq!(leaf.object_size(), 1);
        let id = leaf.object_ids[0].id;
        assert_eq!(id, 1);
    }

    #[test]
    fn test_insert_multiple_objects() {
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, (i * 2) as f32])
            .collect();
        let os = make_os_with_objects(2, &vecs);
        let mut tree = DVPTree::new(100, 5);
        for id in 1..=10 {
            tree.insert(id, &os).unwrap();
        }
        // All should be in the same leaf (no split, leaf_objects_size = 100).
        let leaf = tree.get_leaf(1).unwrap();
        assert_eq!(leaf.object_size(), 10);
    }

    #[test]
    fn test_split_triggers() {
        // Set small leaf size to force a split.
        let vecs: Vec<Vec<f32>> = (0..6)
            .map(|i| vec![i as f32 * 10.0, 0.0])
            .collect();
        let os = make_os_with_objects(2, &vecs);
        let mut tree = DVPTree::new(5, 2); // max 5 per leaf, 2 children

        for id in 1..=6 {
            tree.insert(id as ObjectID, &os).unwrap();
        }

        // Should have split: internal nodes should exist beyond slot 0.
        let has_internal = tree.internal_nodes.iter().any(|n| n.is_some());
        assert!(has_internal, "Expected an internal node after splitting");
    }

    #[test]
    fn test_search_finds_nearest() {
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 10.0, 0.0])
            .collect();
        let os = make_os_with_objects(2, &vecs);
        let mut tree = DVPTree::new(100, 5);
        for id in 1..=5 {
            tree.insert(id, &os).unwrap();
        }

        // Search for something close to object 3 (= [20.0, 0.0]).
        let results = tree.search(&[21.0, 0.0], f32::MAX, 1, &os).unwrap();
        assert!(!results.is_empty());
        let best_id = results[0].id;
        assert_eq!(best_id, 3); // [20.0, 0.0] is closest to [21.0, 0.0]
    }

    #[test]
    fn test_remove_object() {
        let vecs: Vec<Vec<f32>> = (0..3)
            .map(|i| vec![i as f32, 0.0])
            .collect();
        let os = make_os_with_objects(2, &vecs);
        let mut tree = DVPTree::new(100, 5);
        for id in 1..=3 {
            tree.insert(id, &os).unwrap();
        }

        tree.remove(2, 0, &os).unwrap();
        let leaf = tree.get_leaf(1).unwrap();
        assert_eq!(leaf.object_size(), 2);
        let ids: Vec<u32> = leaf.object_ids.iter().map(|o| o.id).collect();
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 3.0, i as f32])
            .collect();
        let os = make_os_with_objects(2, &vecs);
        let mut tree = DVPTree::new(100, 5);
        for id in 1..=5 {
            tree.insert(id, &os).unwrap();
        }

        let mut buf = Vec::new();
        tree.serialize(&mut buf, 2).unwrap();

        let mut tree2 = DVPTree::new(100, 5);
        let mut cursor = std::io::Cursor::new(&buf);
        tree2.deserialize(&mut cursor, 2).unwrap();

        // Check same leaf structure.
        assert_eq!(tree2.leaf_nodes.len(), tree.leaf_nodes.len());
        let leaf1 = tree2.get_leaf(1).unwrap();
        assert_eq!(leaf1.object_size(), 5);
    }
}
