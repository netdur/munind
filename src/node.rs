/// Port of NGT/Node.h + NGT/Node.cpp
///
/// Non-shared-memory, non-NGT_NODE_USE_VECTOR variant.
/// Phase 1: float objects only (pivots stored as Vec<f32>).

use std::io::{Read, Write};

use crate::common::{Distance, NgtError, ObjectDistance, ObjectID};

// ---------------------------------------------------------------------------
// NodeId — NGT::Node::ID
// ---------------------------------------------------------------------------
//
// The raw `NodeID` (u32) encodes both the node type and the slot index:
//   bit 31 = 1  →  Leaf node
//   bit 31 = 0  →  Internal node
//   bits 0-30   →  actual index into the leaf / internal repository

/// Node type discriminant stored in bit 31 of the raw ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    Internal = 0,
    Leaf     = 1,
}

/// A packed node identifier that encodes both type and index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NULL: NodeId = NodeId(0);

    /// Construct a null ID (id = 0, type = Internal).
    #[inline]
    pub fn null() -> Self {
        Self(0)
    }

    /// Is this the null sentinel?
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    /// Decode the type bit.
    /// Maps to `Node::ID::getType()`.
    #[inline]
    pub fn get_type(&self) -> NodeType {
        if (self.0 & 0x80000000) != 0 {
            NodeType::Leaf
        } else {
            NodeType::Internal
        }
    }

    /// Decode the slot index (bits 0-30).
    /// Maps to `Node::ID::getID()`.
    #[inline]
    pub fn get_id(&self) -> u32 {
        self.0 & 0x7fffffff
    }

    /// Return the raw packed value.
    /// Maps to `Node::ID::get()`.
    #[inline]
    pub fn get_raw(&self) -> u32 {
        self.0
    }

    /// Build a Leaf NodeId from a slot index.
    #[inline]
    pub fn leaf(idx: u32) -> Self {
        Self(0x80000000 | idx)
    }

    /// Build an Internal NodeId from a slot index.
    #[inline]
    pub fn internal(idx: u32) -> Self {
        Self(idx & 0x7fffffff)
    }

    /// Overwrite only the type bit, preserving the index.
    /// Maps to `Node::ID::setType(t)`.
    pub fn set_type(&mut self, t: NodeType) {
        let idx = self.get_id();
        self.0 = ((t as u32) << 31) | idx;
    }

    /// Overwrite only the index bits, preserving the type.
    /// Maps to `Node::ID::setID(i)`.
    pub fn set_id(&mut self, i: u32) {
        self.0 = (0x80000000 & self.0) | (i & 0x7fffffff);
    }

    /// Replace the entire packed value.
    /// Maps to `Node::ID::setRaw(i)`.
    pub fn set_raw(&mut self, i: u32) {
        self.0 = i;
    }

    // --- binary I/O ---

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), NgtError> {
        w.write_all(&self.0.to_le_bytes())
            .map_err(|e| format!("NodeId::write_to: {}", e))
    }

    pub fn read_from<R: Read>(&mut self, r: &mut R) -> Result<(), NgtError> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)
            .map_err(|e| format!("NodeId::read_from: {}", e))?;
        self.0 = u32::from_le_bytes(buf);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NodeObject — NGT::Node::Object  (temporary, used only during tree build)
// ---------------------------------------------------------------------------

/// A temporary container used during DVPTree insertion to carry an object's
/// data alongside its pivot-distance and cluster assignment.
///
/// Maps to `NGT::Node::Object`.  In the C++ code, `object` is a raw pointer
/// into the ObjectSpace; here we store a borrowed slice reference for the
/// duration of the split operation.
///
/// `PIVOT = -1.0` is used as a sentinel for `leaf_distance` to indicate that
/// this object is the sub-pivot of its cluster.
pub const PIVOT: f32 = -1.0;

#[derive(Clone, Debug)]
pub struct NodeObject {
    /// 1-based object ID in the ObjectSpace.
    pub id: ObjectID,
    /// The float vector for this object (borrowed or cloned from ObjectSpace).
    pub data: Vec<f32>,
    /// Distance from the global pivot to this object (computed in splitObjects).
    pub distance: f32,
    /// `PIVOT` (-1.0) if this is the cluster sub-pivot; otherwise distance from
    /// the cluster sub-pivot.
    pub leaf_distance: f32,
    /// Which child cluster this object belongs to (0-based).
    pub cluster_id: i32,
}

impl NodeObject {
    pub fn new(id: ObjectID, data: Vec<f32>) -> Self {
        Self {
            id,
            data,
            distance: 0.0,
            leaf_distance: 0.0,
            cluster_id: 0,
        }
    }
}

/// `Node::Object::operator<` — sort ascending by distance.
impl PartialOrd for NodeObject {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for NodeObject {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

// ---------------------------------------------------------------------------
// InternalNode — NGT::InternalNode
// ---------------------------------------------------------------------------

/// An internal node in the DVPTree.
///
/// `children`: `children_size` child IDs (each may be a leaf or internal node).
/// `borders`:  `children_size - 1` distance thresholds used to route queries.
/// `pivot`:    representative object for this subtree (a float vector).
pub struct InternalNode {
    pub id:            NodeId,
    pub parent:        NodeId,
    pub pivot:         Option<Vec<f32>>,
    pub children_size: usize,
    pub children:      Vec<NodeId>,
    pub borders:       Vec<Distance>,
}

impl InternalNode {
    /// Default `children_size` matching C++ `InternalNode(ObjectSpace* = 0)`.
    pub const DEFAULT_CHILDREN_SIZE: usize = 5;

    pub fn new(children_size: usize) -> Self {
        let mut node = InternalNode {
            id:            NodeId::null(),
            parent:        NodeId::null(),
            pivot:         None,
            children_size,
            children:      vec![NodeId::null(); children_size],
            borders:       vec![0.0f32; children_size.saturating_sub(1)],
        };
        node.id.set_type(NodeType::Internal);
        node
    }

    /// Replace child `src` with `dst`.
    /// Maps to `InternalNode::updateChild`.
    pub fn update_child(&mut self, src: NodeId, dst: NodeId) {
        for child in self.children.iter_mut() {
            if *child == src {
                *child = dst;
                return;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Binary serialization
    // -----------------------------------------------------------------------

    /// Write to a binary stream.
    ///
    /// Format (all little-endian):
    ///   [4]  id (u32)
    ///   [4]  parent (u32)
    ///   [dim*4]  pivot float array  (must be Some)
    ///   [8]  children_size (u64 = size_t)
    ///   [4 * children_size]  child IDs
    ///   [4 * (children_size - 1)]  border distances
    pub fn write_to<W: Write>(&self, w: &mut W, dim: usize) -> Result<(), NgtError> {
        self.id.write_to(w)?;
        self.parent.write_to(w)?;

        let pivot = self.pivot.as_ref().ok_or_else(|| {
            "InternalNode::write_to: pivot is null".to_string()
        })?;
        for &f in pivot.iter().take(dim) {
            w.write_all(&f.to_le_bytes())
                .map_err(|e| format!("InternalNode::write_to pivot: {}", e))?;
        }

        // children_size as size_t (8 bytes)
        w.write_all(&(self.children_size as u64).to_le_bytes())
            .map_err(|e| format!("InternalNode::write_to children_size: {}", e))?;

        for child in &self.children {
            child.write_to(w)?;
        }
        for &b in &self.borders {
            w.write_all(&b.to_le_bytes())
                .map_err(|e| format!("InternalNode::write_to border: {}", e))?;
        }
        Ok(())
    }

    /// Read from a binary stream.
    pub fn read_from<R: Read>(&mut self, r: &mut R, dim: usize) -> Result<(), NgtError> {
        self.id.read_from(r)?;
        self.parent.read_from(r)?;

        // Read pivot (dim * 4 bytes)
        let mut pivot = vec![0.0f32; dim];
        let mut buf4 = [0u8; 4];
        for f in pivot.iter_mut() {
            r.read_exact(&mut buf4)
                .map_err(|e| format!("InternalNode::read_from pivot: {}", e))?;
            *f = f32::from_le_bytes(buf4);
        }
        self.pivot = Some(pivot);

        // Read children_size (u64)
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("InternalNode::read_from children_size: {}", e))?;
        let cs = u64::from_le_bytes(buf8) as usize;
        self.children_size = cs;

        self.children.resize(cs, NodeId::null());
        for child in self.children.iter_mut() {
            child.read_from(r)?;
        }

        self.borders.resize(cs.saturating_sub(1), 0.0);
        for b in self.borders.iter_mut() {
            r.read_exact(&mut buf4)
                .map_err(|e| format!("InternalNode::read_from border: {}", e))?;
            *b = f32::from_le_bytes(buf4);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LeafNode — NGT::LeafNode  (non-NGT_NODE_USE_VECTOR variant)
// ---------------------------------------------------------------------------

/// A leaf node in the DVPTree.
///
/// `object_ids`:  the ObjectDistance entries stored in this leaf (id + dist).
/// `pivot`:       representative object for this leaf (float vector).
///
/// In the C++ code `objectSize` is `unsigned short` (u16).
#[derive(Clone)]
pub struct LeafNode {
    pub id:         NodeId,
    pub parent:     NodeId,
    pub pivot:      Option<Vec<f32>>,
    pub object_ids: Vec<ObjectDistance>,
}

impl LeafNode {
    /// Max objects per leaf — matches C++ `LeafObjectsSizeMax`.
    pub const LEAF_OBJECTS_SIZE_MAX: usize = 100;

    pub fn new() -> Self {
        let mut node = LeafNode {
            id:         NodeId::null(),
            parent:     NodeId::null(),
            pivot:      None,
            object_ids: Vec::new(),
        };
        node.id.set_type(NodeType::Leaf);
        node
    }

    pub fn object_size(&self) -> usize {
        self.object_ids.len()
    }

    // -----------------------------------------------------------------------
    // removeObject  (NGT::LeafNode::removeObject)
    // -----------------------------------------------------------------------

    /// Remove `id` from this leaf's object list.
    ///
    /// `replace_id` is a compaction hint (the last valid ID that was swapped
    /// into the removed object's slot by the Repository).  Three cases:
    ///
    /// 1. `replace_id == 0`:  simple removal of `id`.
    /// 2. `replace_id != 0` and `replace_id` is **not** in this leaf:
    ///    overwrite `id`'s slot with `replace_id` (no size change).
    /// 3. `replace_id != 0` and `replace_id` is **already** in this leaf:
    ///    both happen to be in the same leaf; just remove `id`.
    pub fn remove_object(&mut self, id: ObjectID, mut replace_id: ObjectID) {
        let fsize = self.object_ids.len();
        let mut replace_id_found = false;

        if replace_id != 0 {
            for oid in self.object_ids.iter() {
                let oid_id = oid.id; // copy out of packed struct
                if oid_id == replace_id {
                    replace_id = 0;          // signal: found it
                    replace_id_found = true;
                    break;
                }
            }
        }

        let mut found_idx = fsize; // sentinel: not found
        for (i, oid) in self.object_ids.iter().enumerate() {
            let oid_id = oid.id;
            if oid_id == id {
                if replace_id != 0 {
                    // Case 2: replace_id not in this leaf — overwrite and return.
                    self.object_ids[i].id = replace_id;
                    return;
                } else {
                    // Case 1 or 3: remove id from list.
                    if replace_id_found {
                        // Both id and replace_id in same leaf — just remove id.
                        // (This mirrors the C++ warning path.)
                    }
                    found_idx = i;
                    break;
                }
            }
        }

        if found_idx == fsize {
            // id not found in this leaf.  In C++ this can happen for the root
            // leaf (pivot == null) and is silently tolerated.
            return;
        }

        // Shift elements down and shrink.
        self.object_ids.remove(found_idx);
    }

    // -----------------------------------------------------------------------
    // Binary serialization
    // -----------------------------------------------------------------------

    /// Write to a binary stream.
    ///
    /// Format:
    ///   [4]  id (u32)
    ///   [4]  parent (u32)
    ///   [2]  objectSize (u16 = unsigned short)
    ///   [8 * objectSize]  ObjectDistance entries: [4 id][4 dist] each
    ///   [dim*4]  pivot float array — written only when pivot is Some
    ///            (empty index: parent.getID()==0 && objectSize==0 → skip)
    pub fn write_to<W: Write>(&self, w: &mut W, dim: usize) -> Result<(), NgtError> {
        self.id.write_to(w)?;
        self.parent.write_to(w)?;

        let obj_size = self.object_ids.len() as u16;
        w.write_all(&obj_size.to_le_bytes())
            .map_err(|e| format!("LeafNode::write_to objectSize: {}", e))?;

        for od in &self.object_ids {
            let id_val = od.id;           // copy from packed struct
            let dist_val = od.distance;
            w.write_all(&id_val.to_le_bytes())
                .map_err(|e| format!("LeafNode::write_to object id: {}", e))?;
            w.write_all(&dist_val.to_le_bytes())
                .map_err(|e| format!("LeafNode::write_to object dist: {}", e))?;
        }

        if let Some(pivot) = &self.pivot {
            for &f in pivot.iter().take(dim) {
                w.write_all(&f.to_le_bytes())
                    .map_err(|e| format!("LeafNode::write_to pivot: {}", e))?;
            }
        }
        // else: empty index leaf (parent.getID()==0 && object_size==0), no pivot bytes written.
        Ok(())
    }

    /// Read from a binary stream.
    pub fn read_from<R: Read>(&mut self, r: &mut R, dim: usize) -> Result<(), NgtError> {
        self.id.read_from(r)?;
        self.parent.read_from(r)?;

        // objectSize as u16
        let mut buf2 = [0u8; 2];
        r.read_exact(&mut buf2)
            .map_err(|e| format!("LeafNode::read_from objectSize: {}", e))?;
        let obj_size = u16::from_le_bytes(buf2) as usize;

        self.object_ids.clear();
        self.object_ids.reserve(obj_size);
        let mut buf4 = [0u8; 4];
        for _ in 0..obj_size {
            r.read_exact(&mut buf4)
                .map_err(|e| format!("LeafNode::read_from object id: {}", e))?;
            let id = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)
                .map_err(|e| format!("LeafNode::read_from object dist: {}", e))?;
            let dist = f32::from_le_bytes(buf4);
            self.object_ids.push(ObjectDistance::new(id, dist));
        }

        // Check empty-index case (matches C++ deserialize logic).
        let parent_id = self.parent.get_id();
        if parent_id == 0 && obj_size == 0 {
            self.pivot = None;
            return Ok(());
        }

        // Read pivot (dim * 4 bytes)
        let mut pivot = vec![0.0f32; dim];
        for f in pivot.iter_mut() {
            r.read_exact(&mut buf4)
                .map_err(|e| format!("LeafNode::read_from pivot: {}", e))?;
            *f = f32::from_le_bytes(buf4);
        }
        self.pivot = Some(pivot);
        Ok(())
    }
}

impl Default for LeafNode {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Static split algorithms  (NGT::LeafNode static methods)
// ---------------------------------------------------------------------------

/// Choose a pivot index using the "max distance" heuristic.
///
/// Algorithm (mirrors `LeafNode::selectPivotByMaxDistance`):
/// 1. From `objects[0]`, find the farthest object → `aid`
/// 2. From `objects[aid]`, find the farthest (excluding `aid`) → `bid`
/// 3. From `objects[bid]`, find the farthest (excluding `bid`) → return that index
///
/// `compare`: distance function `Fn(&[f32], &[f32]) -> f32`
pub fn select_pivot_by_max_distance<F>(objects: &[NodeObject], compare: &F) -> usize
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let fsize = objects.len();
    if fsize == 0 {
        return 0;
    }

    // Step 1: farthest from objects[0]
    let mut max_d = 0.0f32;
    let mut aid = 0usize;
    for i in 1..fsize {
        let d = compare(&objects[0].data, &objects[i].data);
        if d >= max_d {
            max_d = d;
            aid = i;
        }
    }

    // Step 2: farthest from objects[aid]
    max_d = 0.0;
    let mut bid = 0usize;
    for i in 0..fsize {
        if i == aid {
            continue;
        }
        let d = compare(&objects[aid].data, &objects[i].data);
        if d >= max_d {
            max_d = d;
            bid = i;
        }
    }

    // Step 3: farthest from objects[bid]
    max_d = 0.0;
    let mut pivot_idx = 0usize;
    for i in 0..fsize {
        if i == bid {
            continue;
        }
        let d = compare(&objects[bid].data, &objects[i].data);
        if d >= max_d {
            max_d = d;
            pivot_idx = i;
        }
    }

    pivot_idx
}

/// Choose a pivot index using the "max variance" heuristic.
///
/// Computes all pairwise distances, then picks the object with the highest
/// variance of distances to others.
///
/// Maps to `LeafNode::selectPivotByMaxVariance`.
pub fn select_pivot_by_max_variance<F>(objects: &[NodeObject], compare: &F) -> usize
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let fsize = objects.len();
    if fsize == 0 {
        return 0;
    }

    // Build full pairwise distance matrix.
    let mut dist = vec![0.0f32; fsize * fsize];
    for i in 0..fsize {
        for j in (i + 1)..fsize {
            let d = compare(&objects[i].data, &objects[j].data);
            dist[i * fsize + j] = d;
            dist[j * fsize + i] = d;
        }
    }

    // Compute variance for each row.
    let mut max_v = f64::NEG_INFINITY;
    let mut max_id = 0usize;
    for i in 0..fsize {
        let avg: f64 = (0..fsize)
            .map(|j| dist[i * fsize + j] as f64)
            .sum::<f64>()
            / fsize as f64;
        let v: f64 = (0..fsize)
            .map(|j| {
                let d = dist[i * fsize + j] as f64 - avg;
                d * d
            })
            .sum::<f64>()
            / fsize as f64;
        if v > max_v {
            max_v = v;
            max_id = i;
        }
    }

    max_id
}

/// Partition `objects` into `children_size` clusters around the pivot at
/// index `pv`.
///
/// After this call:
/// - `objects` is sorted by distance to `objects[pv]`.
/// - Each entry has its `cluster_id` assigned (0-based, 0 .. children_size-1).
/// - The first object in each cluster has `leaf_distance = PIVOT` (-1.0).
/// - Other objects in a cluster have `leaf_distance` = distance from the
///   cluster sub-pivot.
///
/// Maps to `LeafNode::splitObjects`.
///
/// Returns `Err` if too many objects share the same distance (cannot split).
pub fn split_objects<F>(
    objects: &mut Vec<NodeObject>,
    pv: usize,
    children_size: usize,
    compare: &F,
) -> Result<(), NgtError>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let fsize = objects.len();

    // Compute distances from pivot pv to all others.
    for i in 0..fsize {
        if i == pv {
            objects[i].distance = 0.0;
        } else {
            let d = compare(&objects[pv].data, &objects[i].data);
            objects[i].distance = d;
        }
    }

    // Sort by distance (ascending).  Stable sort to match C++ stable_sort behaviour;
    // std::sort is not guaranteed stable in C++ but the algorithm works either way.
    objects.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign cluster IDs (0 .. children_size-1) from back to front.
    let mut cid = (children_size - 1) as i32;
    let mut cms = ((fsize as i32) * cid) / children_size as i32;

    objects[fsize - 1].cluster_id = cid;
    let mut i = fsize as i32 - 2;
    while i >= 0 {
        let idx = i as usize;
        if i < cms && cid > 0 {
            let next_dist = objects[idx + 1].distance;
            let cur_dist  = objects[idx].distance;
            if cur_dist != next_dist {
                cid -= 1;
                cms = ((fsize as i32) * cid) / children_size as i32;
            }
        }
        objects[idx].cluster_id = cid;
        i -= 1;
    }

    if cid != 0 {
        // Could not fill all children — too many identical distances.
        if objects[fsize - 1].cluster_id == cid {
            return Err(format!(
                "LeafNode::splitObjects: All of the object distances are the same! \
                 internalChildrenSize={} size={} pivot={}",
                children_size, fsize, pv
            ));
        }
        // Partial split: shift cluster IDs down so they start from 0.
        for obj in objects.iter_mut() {
            obj.cluster_id -= cid;
        }
    }

    // Find sub-pivot for each cluster and compute leafDistance.
    let mut sub_pivots: Vec<i64> = vec![-1; children_size];
    // First pass: mark sub-pivots (first element of each cluster).
    for i in 0..fsize {
        let cid_idx = objects[i].cluster_id as usize;
        if sub_pivots[cid_idx] == -1 {
            sub_pivots[cid_idx] = i as i64;
            objects[i].leaf_distance = PIVOT; // -1.0
        }
    }

    // Second pass: compute leafDistance for non-sub-pivots.
    // Clone sub-pivot data to avoid borrow conflict.
    let sub_pivot_vecs: Vec<Option<Vec<f32>>> = sub_pivots
        .iter()
        .map(|&sp| {
            if sp == -1 {
                None
            } else {
                Some(objects[sp as usize].data.clone())
            }
        })
        .collect();

    for i in 0..fsize {
        if objects[i].leaf_distance == PIVOT {
            continue; // already set as sub-pivot
        }
        let cid_idx = objects[i].cluster_id as usize;
        if let Some(sp_data) = &sub_pivot_vecs[cid_idx] {
            let d = compare(sp_data, &objects[i].data);
            objects[i].leaf_distance = d;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_round_trip() {
        let leaf = NodeId::leaf(42);
        assert_eq!(leaf.get_type(), NodeType::Leaf);
        assert_eq!(leaf.get_id(), 42);

        let internal = NodeId::internal(7);
        assert_eq!(internal.get_type(), NodeType::Internal);
        assert_eq!(internal.get_id(), 7);

        let null = NodeId::null();
        assert!(null.is_null());
    }

    #[test]
    fn test_leaf_node_remove_simple() {
        let mut leaf = LeafNode::new();
        leaf.object_ids = vec![
            ObjectDistance::new(1, 0.1),
            ObjectDistance::new(2, 0.2),
            ObjectDistance::new(3, 0.3),
        ];
        // Remove id=2, no replacement.
        leaf.remove_object(2, 0);
        assert_eq!(leaf.object_ids.len(), 2);
        let ids: Vec<u32> = leaf.object_ids.iter().map(|o| o.id).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn test_leaf_node_remove_with_replace() {
        let mut leaf = LeafNode::new();
        leaf.object_ids = vec![
            ObjectDistance::new(1, 0.1),
            ObjectDistance::new(2, 0.2),
        ];
        // Remove id=1, replace with id=5 (5 is not in leaf → overwrite slot).
        leaf.remove_object(1, 5);
        assert_eq!(leaf.object_ids.len(), 2);
        let ids: Vec<u32> = leaf.object_ids.iter().map(|o| o.id).collect();
        assert_eq!(ids, vec![5, 2]);
    }

    #[test]
    fn test_internal_node_update_child() {
        let mut n = InternalNode::new(3);
        n.children[0] = NodeId::leaf(1);
        n.children[1] = NodeId::leaf(2);
        n.children[2] = NodeId::leaf(3);

        n.update_child(NodeId::leaf(2), NodeId::internal(9));
        assert_eq!(n.children[1].get_id(), 9);
        assert_eq!(n.children[1].get_type(), NodeType::Internal);
    }

    #[test]
    fn test_select_pivot_max_distance() {
        // Simple 1-D objects; pivot should be the most extreme.
        let objects: Vec<NodeObject> = (0..4u32)
            .map(|i| NodeObject::new(i + 1, vec![i as f32]))
            .collect();

        let pivot = select_pivot_by_max_distance(&objects, &|a, b| (a[0] - b[0]).abs());
        // Should be an extreme element.
        assert!(pivot < 4);
    }

    #[test]
    fn test_split_objects_two_clusters() {
        let mut objects: Vec<NodeObject> = vec![
            NodeObject::new(1, vec![0.0, 0.0]),
            NodeObject::new(2, vec![0.0, 1.0]),
            NodeObject::new(3, vec![10.0, 10.0]),
            NodeObject::new(4, vec![10.0, 11.0]),
        ];

        let compare = |a: &[f32], b: &[f32]| -> f32 {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            (dx * dx + dy * dy).sqrt()
        };

        // Pivot = object 0 (near cluster); children_size = 2
        split_objects(&mut objects, 0, 2, &compare).unwrap();

        // Objects should be sorted and clustered.
        let cluster_ids: Vec<i32> = objects.iter().map(|o| o.cluster_id).collect();
        // Two distinct cluster values.
        let distinct: std::collections::HashSet<i32> = cluster_ids.iter().copied().collect();
        assert_eq!(distinct.len(), 2);
    }

    #[test]
    fn test_internal_node_serialize_roundtrip() {
        let dim = 2usize;
        let mut n = InternalNode::new(3);
        n.id = NodeId::internal(1);
        n.parent = NodeId::null();
        n.pivot = Some(vec![1.0, 2.0]);
        n.children = vec![NodeId::leaf(1), NodeId::leaf(2), NodeId::leaf(3)];
        n.borders = vec![0.5, 1.5];

        let mut buf = Vec::new();
        n.write_to(&mut buf, dim).unwrap();

        let mut n2 = InternalNode::new(3);
        let mut cursor = std::io::Cursor::new(&buf);
        n2.read_from(&mut cursor, dim).unwrap();

        assert_eq!(n2.id.get_id(), 1);
        assert_eq!(n2.children_size, 3);
        assert_eq!(n2.children[0].get_id(), 1);
        assert_eq!(n2.borders[0], 0.5);
        assert_eq!(n2.pivot.as_ref().unwrap(), &vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_leaf_node_serialize_roundtrip() {
        let dim = 2usize;
        let mut leaf = LeafNode::new();
        leaf.id = NodeId::leaf(1);
        leaf.parent = NodeId::internal(0);
        leaf.parent.set_id(1); // non-zero parent so pivot IS written
        leaf.object_ids = vec![
            ObjectDistance::new(1, 0.0),
            ObjectDistance::new(2, 0.5),
        ];
        leaf.pivot = Some(vec![0.5, 0.5]);

        let mut buf = Vec::new();
        leaf.write_to(&mut buf, dim).unwrap();

        let mut leaf2 = LeafNode::new();
        let mut cursor = std::io::Cursor::new(&buf);
        leaf2.read_from(&mut cursor, dim).unwrap();

        assert_eq!(leaf2.object_ids.len(), 2);
        let id0 = leaf2.object_ids[0].id;
        assert_eq!(id0, 1);
        assert_eq!(leaf2.pivot.as_ref().unwrap(), &vec![0.5f32, 0.5]);
    }
}
