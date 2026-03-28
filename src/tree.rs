use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::node::ObjectDistance;
use crate::object_space::ObjectSpace;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeNodeRef {
    Leaf(usize),
    Internal(usize),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DvpLeaf {
    pub parent: Option<TreeNodeRef>,
    pub pivot_id: Option<u32>,
    pub object_ids: Vec<ObjectDistance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DvpInternal {
    pub parent: Option<TreeNodeRef>,
    pub pivot_id: u32,
    pub children: Vec<TreeNodeRef>,
    pub borders: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DvpTree {
    pub leaf_objects_size: usize,
    pub internal_children_size: usize,
    pub root: TreeNodeRef,
    pub leaves: Vec<Option<DvpLeaf>>,
    pub internals: Vec<Option<DvpInternal>>,
    pub object_to_leaf: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SplitEntry {
    id: u32,
    distance: f32,
    leaf_distance: f32,
    cluster_id: usize,
}

#[derive(Clone, Copy, Debug)]
struct PendingNode {
    distance: f32,
    node: TreeNodeRef,
}

impl PartialEq for PendingNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl Eq for PendingNode {}

impl PartialOrd for PendingNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance).map(|ord| {
            ord.then_with(|| format!("{:?}", self.node).cmp(&format!("{:?}", other.node)))
        })
    }
}

impl Ord for PendingNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl DvpTree {
    pub fn new(leaf_objects_size: usize, internal_children_size: usize) -> Self {
        let mut tree = Self {
            leaf_objects_size: leaf_objects_size.max(1),
            internal_children_size: internal_children_size.max(2),
            root: TreeNodeRef::Leaf(1),
            leaves: vec![None],
            internals: vec![None],
            object_to_leaf: vec![0],
        };
        let root = DvpLeaf {
            parent: None,
            pivot_id: None,
            object_ids: Vec::with_capacity(tree.leaf_objects_size),
        };
        tree.leaves.push(Some(root));
        tree
    }

    pub fn build(
        objects: &[Vec<f32>],
        object_space: &ObjectSpace,
        leaf_objects_size: usize,
        internal_children_size: usize,
    ) -> Self {
        let mut tree = Self::new(leaf_objects_size, internal_children_size);
        tree.object_to_leaf.resize(objects.len() + 1, 0);
        for object_id in 1..=objects.len() {
            tree.insert(object_id as u32, object_space);
        }
        tree
    }

    pub fn is_empty(&self) -> bool {
        self.leaf(self.root_leaf_id()).object_ids.is_empty()
    }

    pub fn insert(&mut self, object_id: u32, object_space: &ObjectSpace) {
        if self.object_to_leaf.len() <= object_id as usize {
            self.object_to_leaf.resize(object_id as usize + 1, 0);
        }
        let leaf_id = self.search_leaf_id_for_object(object_id, object_space);
        let should_split = self.leaf(leaf_id).object_ids.len() >= self.leaf_objects_size;
        if should_split {
            self.split_and_insert(leaf_id, object_id, object_space);
        } else {
            self.insert_into_leaf(leaf_id, object_id, object_space);
        }
    }

    pub fn leaf_for_query(&self, query: &[f32], object_space: &ObjectSpace) -> Option<usize> {
        self.greedy_leaf_for_query(query, object_space)
    }

    pub fn greedy_leaf_for_query(
        &self,
        query: &[f32],
        object_space: &ObjectSpace,
    ) -> Option<usize> {
        let mut current = self.root;
        loop {
            match current {
                TreeNodeRef::Leaf(leaf_id) => return Some(leaf_id),
                TreeNodeRef::Internal(internal_id) => {
                    let node = self.internal(internal_id);
                    let pivot = object_space.materialize_object(node.pivot_id as usize)?;
                    let distance = object_space.compare(query, &pivot) as f32;
                    let (region, _) = self
                        .search_regions(distance, 0.0, &node.borders)
                        .into_iter()
                        .next()?;
                    current = node.children[region];
                }
            }
        }
    }

    pub fn nearest_leaves_for_query(
        &self,
        query: &[f32],
        object_space: &ObjectSpace,
        limit: usize,
    ) -> Vec<usize> {
        if limit == 0 {
            return Vec::new();
        }
        let mut unchecked = vec![self.root];
        let mut pending = BinaryHeap::new();
        let mut leaves = Vec::with_capacity(limit);
        while let Some(current) = unchecked.pop() {
            match current {
                TreeNodeRef::Leaf(leaf_id) => {
                    leaves.push(leaf_id);
                    if leaves.len() >= limit {
                        break;
                    }
                    while unchecked.is_empty() {
                        if let Some(PendingNode { node, .. }) = pending.pop() {
                            unchecked.push(node);
                        } else {
                            break;
                        }
                    }
                }
                TreeNodeRef::Internal(internal_id) => {
                    let node = self.internal(internal_id);
                    let Some(pivot) = object_space.materialize_object(node.pivot_id as usize)
                    else {
                        continue;
                    };
                    let distance = object_space.compare(query, &pivot) as f32;
                    let regions = self.search_regions(distance, 0.0, &node.borders);
                    let Some((first_region, _first_distance)) = regions.first().copied() else {
                        continue;
                    };
                    match node.children[first_region] {
                        child @ TreeNodeRef::Leaf(_) | child @ TreeNodeRef::Internal(_) => {
                            unchecked.push(child)
                        }
                    }
                    for (region, region_distance) in regions.into_iter().skip(1) {
                        pending.push(PendingNode {
                            distance: region_distance,
                            node: node.children[region],
                        });
                    }
                }
            }
        }
        leaves
    }

    pub fn get_object_ids_from_leaf(&self, leaf_id: usize) -> Vec<ObjectDistance> {
        self.leaves
            .get(leaf_id)
            .and_then(|leaf| leaf.as_ref())
            .map(|leaf| leaf.object_ids.clone())
            .unwrap_or_default()
    }

    pub fn get_object_ids_from_leaf_for_object(&self, object_id: u32) -> Vec<ObjectDistance> {
        let leaf_id = self
            .object_to_leaf
            .get(object_id as usize)
            .copied()
            .unwrap_or(0);
        self.get_object_ids_from_leaf(leaf_id)
    }

    fn root_leaf_id(&self) -> usize {
        match self.root {
            TreeNodeRef::Leaf(leaf_id) => leaf_id,
            TreeNodeRef::Internal(_) => 1,
        }
    }

    fn search_leaf_id_for_object(&self, object_id: u32, object_space: &ObjectSpace) -> usize {
        let Some(object) = object_space.materialize_object(object_id as usize) else {
            return self.root_leaf_id();
        };
        self.leaf_for_query(&object, object_space)
            .unwrap_or(self.root_leaf_id())
    }

    fn insert_into_leaf(&mut self, leaf_id: usize, object_id: u32, object_space: &ObjectSpace) {
        let pivot_id = {
            let leaf = self.leaf(leaf_id);
            leaf.pivot_id.unwrap_or(object_id)
        };
        let distance = if pivot_id == object_id {
            0.0
        } else {
            let pivot = object_space
                .materialize_object(pivot_id as usize)
                .expect("pivot object missing");
            let object = object_space
                .materialize_object(object_id as usize)
                .expect("inserted object missing");
            object_space.compare(&object, &pivot) as f32
        };

        let leaf = self.leaf_mut(leaf_id);
        if leaf.pivot_id.is_none() {
            leaf.pivot_id = Some(object_id);
        }
        leaf.object_ids.push(ObjectDistance {
            id: object_id,
            distance,
        });
        leaf.object_ids.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        self.object_to_leaf[object_id as usize] = leaf_id;
    }

    fn split_and_insert(&mut self, leaf_id: usize, object_id: u32, object_space: &ObjectSpace) {
        let parent = self.leaf(leaf_id).parent;
        let existing_ids: Vec<u32> = self
            .leaf(leaf_id)
            .object_ids
            .iter()
            .map(|entry| entry.id)
            .collect();
        let mut entries = existing_ids
            .into_iter()
            .map(|id| SplitEntry {
                id,
                distance: 0.0,
                leaf_distance: 0.0,
                cluster_id: 0,
            })
            .collect::<Vec<_>>();
        entries.push(SplitEntry {
            id: object_id,
            distance: 0.0,
            leaf_distance: 0.0,
            cluster_id: 0,
        });

        let pivot_idx = self.select_pivot_by_max_variance(&entries, object_space);
        self.split_objects(&mut entries, pivot_idx, object_space);

        let mut cluster_children = Vec::with_capacity(self.internal_children_size);
        let mut borders = vec![f32::MAX; self.internal_children_size.saturating_sub(1)];
        let mut current_cluster = 0usize;
        let internal_id = self.allocate_internal(DvpInternal {
            parent,
            pivot_id: entries[0].id,
            children: Vec::with_capacity(self.internal_children_size),
            borders: borders.clone(),
        });

        for cluster in 0..self.internal_children_size {
            let child_leaf_id = if cluster == 0 {
                leaf_id
            } else {
                self.allocate_leaf(DvpLeaf {
                    parent: Some(TreeNodeRef::Internal(internal_id)),
                    pivot_id: None,
                    object_ids: Vec::with_capacity(self.leaf_objects_size),
                })
            };
            {
                let child_leaf = self.leaf_mut(child_leaf_id);
                child_leaf.parent = Some(TreeNodeRef::Internal(internal_id));
                child_leaf.pivot_id = None;
                child_leaf.object_ids.clear();
            }
            cluster_children.push(TreeNodeRef::Leaf(child_leaf_id));
        }

        for (idx, entry) in entries.iter().enumerate() {
            let cluster = entry.cluster_id;
            let child_leaf_id = match cluster_children[cluster] {
                TreeNodeRef::Leaf(leaf_id) => leaf_id,
                TreeNodeRef::Internal(_) => unreachable!(),
            };
            let child_leaf = self.leaf_mut(child_leaf_id);
            if child_leaf.pivot_id.is_none() {
                child_leaf.pivot_id = Some(entry.id);
                child_leaf.object_ids.push(ObjectDistance {
                    id: entry.id,
                    distance: 0.0,
                });
            } else {
                child_leaf.object_ids.push(ObjectDistance {
                    id: entry.id,
                    distance: entry.leaf_distance,
                });
            }
            self.object_to_leaf[entry.id as usize] = child_leaf_id;

            if idx + 1 < entries.len()
                && entries[idx + 1].cluster_id != cluster
                && cluster < borders.len()
            {
                borders[cluster] = entries[idx + 1].distance;
                current_cluster = entries[idx + 1].cluster_id;
            }
        }

        for (cluster, child_ref) in cluster_children.iter().enumerate() {
            let child_leaf_id = match child_ref {
                TreeNodeRef::Leaf(leaf_id) => *leaf_id,
                TreeNodeRef::Internal(_) => unreachable!(),
            };
            let child_leaf = self.leaf_mut(child_leaf_id);
            if child_leaf.pivot_id.is_none() {
                child_leaf.pivot_id = Some(entries[0].id);
            }
            child_leaf.object_ids.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            if cluster < borders.len() && child_leaf.object_ids.is_empty() {
                borders[cluster] = f32::MAX;
            }
        }

        let internal = self.internal_mut(internal_id);
        internal.pivot_id = entries[0].id;
        internal.children = cluster_children;
        internal.borders = borders;

        let internal_ref = TreeNodeRef::Internal(internal_id);
        match parent {
            None => {
                self.root = internal_ref;
            }
            Some(TreeNodeRef::Internal(parent_id)) => {
                let parent_node = self.internal_mut(parent_id);
                for child in &mut parent_node.children {
                    if *child == TreeNodeRef::Leaf(leaf_id) {
                        *child = internal_ref;
                        break;
                    }
                }
            }
            Some(TreeNodeRef::Leaf(_)) => {}
        }
        if current_cluster == 0 {
            let _ = current_cluster;
        }
    }

    fn select_pivot_by_max_variance(
        &self,
        entries: &[SplitEntry],
        object_space: &ObjectSpace,
    ) -> usize {
        let fsize = entries.len();
        let mut distances = vec![0.0f32; fsize * fsize];
        for i in 0..fsize {
            for j in (i + 1)..fsize {
                let distance = object_space
                    .compare_ids(entries[i].id as usize, entries[j].id as usize)
                    .expect("object missing");
                distances[i * fsize + j] = distance;
                distances[j * fsize + i] = distance;
            }
        }

        let mut max_idx = 0usize;
        let mut max_variance = f32::MIN;
        for i in 0..fsize {
            let mut avg = 0.0f32;
            for j in 0..fsize {
                avg += distances[i * fsize + j];
            }
            avg /= fsize as f32;
            let mut variance = 0.0f32;
            for j in 0..fsize {
                let delta = distances[i * fsize + j] - avg;
                variance += delta * delta;
            }
            variance /= fsize as f32;
            if variance > max_variance {
                max_variance = variance;
                max_idx = i;
            }
        }
        max_idx
    }

    fn split_objects(
        &self,
        entries: &mut [SplitEntry],
        pivot_idx: usize,
        object_space: &ObjectSpace,
    ) {
        let pivot_id = entries[pivot_idx].id;
        for entry in entries.iter_mut() {
            if entry.id == pivot_id {
                entry.distance = 0.0;
            } else {
                entry.distance = object_space
                    .compare_ids(pivot_id as usize, entry.id as usize)
                    .expect("object missing");
            }
        }
        entries.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        let fsize = entries.len();
        let mut cluster_id = self.internal_children_size - 1;
        let mut cluster_min_start = (fsize * cluster_id) / self.internal_children_size;
        entries[fsize - 1].cluster_id = cluster_id;
        for idx in (0..(fsize - 1)).rev() {
            if idx < cluster_min_start
                && cluster_id > 0
                && entries[idx].distance != entries[idx + 1].distance
            {
                cluster_id -= 1;
                cluster_min_start = (fsize * cluster_id) / self.internal_children_size;
            }
            entries[idx].cluster_id = cluster_id;
        }

        let shift = entries
            .iter()
            .map(|entry| entry.cluster_id)
            .min()
            .unwrap_or(0);
        for entry in entries.iter_mut() {
            entry.cluster_id -= shift;
        }

        let mut pivot_for_cluster = vec![None; self.internal_children_size];
        for idx in 0..entries.len() {
            let cluster = entries[idx].cluster_id;
            let pivot_idx = match pivot_for_cluster[cluster] {
                Some(pivot_idx) => pivot_idx,
                None => {
                    pivot_for_cluster[cluster] = Some(idx);
                    entries[idx].leaf_distance = -1.0;
                    continue;
                }
            };
            entries[idx].leaf_distance = object_space
                .compare_ids(entries[pivot_idx].id as usize, entries[idx].id as usize)
                .expect("object missing");
        }
    }

    fn search_regions(&self, distance: f32, radius: f32, borders: &[f32]) -> Vec<(usize, f32)> {
        let bsize = borders.len();
        let mut regions: Vec<(usize, f32)> = Vec::with_capacity(self.internal_children_size);
        let mut mid = 0usize;
        while mid < bsize {
            if distance < borders[mid] {
                regions.push((mid, 0.0));
                if distance + radius < borders[mid] {
                    break;
                }
            } else if distance < borders[mid] + radius {
                regions.push((mid, distance - borders[mid]));
            }
            mid += 1;
        }

        if mid == bsize {
            let region_distance = if bsize == 0 || distance >= borders[bsize - 1] {
                0.0
            } else {
                borders[bsize - 1] - distance
            };
            regions.push((mid, region_distance));
        }

        regions.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        regions
    }

    fn allocate_leaf(&mut self, leaf: DvpLeaf) -> usize {
        self.leaves.push(Some(leaf));
        self.leaves.len() - 1
    }

    fn allocate_internal(&mut self, node: DvpInternal) -> usize {
        self.internals.push(Some(node));
        self.internals.len() - 1
    }

    fn leaf(&self, leaf_id: usize) -> &DvpLeaf {
        self.leaves[leaf_id].as_ref().expect("leaf node missing")
    }

    fn leaf_mut(&mut self, leaf_id: usize) -> &mut DvpLeaf {
        self.leaves[leaf_id].as_mut().expect("leaf node missing")
    }

    fn internal(&self, internal_id: usize) -> &DvpInternal {
        self.internals[internal_id]
            .as_ref()
            .expect("internal node missing")
    }

    fn internal_mut(&mut self, internal_id: usize) -> &mut DvpInternal {
        self.internals[internal_id]
            .as_mut()
            .expect("internal node missing")
    }
}
