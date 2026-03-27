use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::graph::{MaxDistanceNode, MinDistanceNode};
use crate::index::{IndexProperty, SearchOptions};
use crate::node::ObjectDistance;
use crate::object_space::ObjectSpace;
use crate::tree::{DvpTree, TreeNodeRef};
use memmap2::Mmap;

pub const OBJECT_MAGIC: &[u8; 8] = b"MOBJ0001";
pub const GRAPH_MAGIC: &[u8; 8] = b"MGRF0001";
pub const OBJECT_HEADER_SIZE: usize = 36;
const GRAPH_HEADER_SIZE: usize = 32;
const EDGE_RECORD_SIZE: usize = 8;

pub struct MmapIndex {
    pub property: IndexProperty,
    object_space: ObjectSpace,
    objects_mmap: Mmap,
    graph_mmap: Mmap,
    tree: Option<DvpTree>,
    object_count: usize,
    graph_offsets_start: usize,
    graph_edges_start: usize,
}

impl MmapIndex {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let property = crate::index::Index::load_property_file(path.join("prf"))?;
        let tree = Self::load_tree_sidecar(path.join("tre.bin"))?;

        let objects_file = File::open(path.join("obj.mmap")).map_err(|e| e.to_string())?;
        let graph_file = File::open(path.join("grp.mmap")).map_err(|e| e.to_string())?;
        let objects_mmap = unsafe { Mmap::map(&objects_file).map_err(|e| e.to_string())? };
        let graph_mmap = unsafe { Mmap::map(&graph_file).map_err(|e| e.to_string())? };

        let (object_count, dimension, max_magnitude) = Self::parse_object_header(&objects_mmap)?;
        if dimension != property.dimension {
            return Err(format!(
                "Mmap object dimension mismatch: property={} mmap={dimension}",
                property.dimension
            ));
        }
        let (node_count, graph_offsets_start, graph_edges_start) =
            Self::parse_graph_header(&graph_mmap)?;
        if node_count != object_count {
            return Err(format!(
                "Mmap graph/object count mismatch: graph={node_count} objects={object_count}"
            ));
        }

        let mut object_space = ObjectSpace::new(
            property.dimension,
            property.distance_type.into(),
            property.object_type,
        );
        object_space.max_magnitude = max_magnitude;

        Ok(Self {
            property,
            object_space,
            objects_mmap,
            graph_mmap,
            tree,
            object_count,
            graph_offsets_start,
            graph_edges_start,
        })
    }

    pub fn object_count(&self) -> usize {
        self.object_count
    }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, String> {
        let prepared_query = self.object_space.prepare_query(query)?;
        let mut seeds = self.get_seeds(&prepared_query, options.k)?;
        let mut radius = f32::MAX;
        let edge_size = self.get_edge_size(
            options.edge_size.unwrap_or(-1),
            1.0 + options.epsilon as f64,
        )?;
        let mut unchecked = BinaryHeap::new();
        let mut checked = vec![0u64; (self.object_count + 64) / 64];
        let mut results = BinaryHeap::new();

        for seed in seeds.iter_mut() {
            seed.distance = self
                .get_object(seed.id as usize)
                .map(|object| self.object_space.compare(&prepared_query, object))
                .unwrap_or(f32::MAX);
        }
        seeds.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        for seed in seeds.iter() {
            if seed.distance == f32::MAX {
                continue;
            }
            if results.len() < options.k && seed.distance <= radius {
                results.push(MaxDistanceNode(*seed));
            }
        }
        if results.len() >= options.k && options.k > 0 {
            radius = results
                .peek()
                .map(|node| node.0.distance)
                .unwrap_or(f32::MAX);
        }

        for seed in seeds.iter() {
            if seed.distance == f32::MAX {
                continue;
            }
            self.mark_checked(&mut checked, seed.id as usize);
            unchecked.push(MinDistanceNode(*seed));
        }

        let mut exploration_radius = ((1.0 + options.epsilon as f64) * radius as f64) as f32;
        while let Some(MinDistanceNode(target)) = unchecked.pop() {
            if target.distance > exploration_radius {
                break;
            }

            for neighbor in self.neighbors(target.id as usize)?.take(edge_size) {
                let neighbor_id = neighbor.id as usize;
                if self.is_checked(&checked, neighbor_id) {
                    continue;
                }
                self.mark_checked(&mut checked, neighbor_id);
                let Some(object) = self.get_object(neighbor_id) else {
                    continue;
                };
                let distance = self.object_space.compare(&prepared_query, object);
                if distance > exploration_radius {
                    continue;
                }

                let result = ObjectDistance {
                    id: neighbor.id,
                    distance,
                };
                unchecked.push(MinDistanceNode(result));
                if options.k > 0 && distance <= radius {
                    results.push(MaxDistanceNode(result));
                    if results.len() >= options.k
                        && results
                            .peek()
                            .map(|top| top.0.distance >= distance)
                            .unwrap_or(false)
                    {
                        if results.len() > options.k {
                            results.pop();
                        }
                        radius = results
                            .peek()
                            .map(|node| node.0.distance)
                            .unwrap_or(f32::MAX);
                        exploration_radius =
                            ((1.0 + options.epsilon as f64) * radius as f64) as f32;
                    }
                }
            }
        }

        let mut final_results = Vec::with_capacity(results.len());
        while let Some(node) = results.pop() {
            final_results.push(node.0);
        }
        final_results.reverse();
        Ok(final_results)
    }

    pub fn linear_search(&self, query: &[f32], k: usize) -> Result<Vec<ObjectDistance>, String> {
        let prepared_query = self.object_space.prepare_query(query)?;
        let mut results = BinaryHeap::new();
        for id in 1..=self.object_count {
            let Some(object) = self.get_object(id) else {
                continue;
            };
            let distance = self.object_space.compare(&prepared_query, object);
            results.push(MaxDistanceNode(ObjectDistance {
                id: id as u32,
                distance,
            }));
            if results.len() > k {
                results.pop();
            }
        }
        let mut final_results = Vec::with_capacity(results.len());
        while let Some(node) = results.pop() {
            final_results.push(node.0);
        }
        final_results.reverse();
        Ok(final_results)
    }

    fn load_tree_sidecar<P: AsRef<Path>>(path: P) -> Result<Option<DvpTree>, String> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| e.to_string())
    }

    fn parse_object_header(bytes: &[u8]) -> Result<(usize, usize, f32), String> {
        if bytes.len() < OBJECT_HEADER_SIZE {
            return Err("Mmap object file is too small".to_string());
        }
        if &bytes[..8] != OBJECT_MAGIC {
            return Err("Invalid mmap object magic".to_string());
        }
        let object_count = read_u64(bytes, 8)? as usize;
        let dimension = read_u64(bytes, 16)? as usize;
        let max_magnitude = read_f32(bytes, 24)?;
        Ok((object_count, dimension, max_magnitude))
    }

    fn parse_graph_header(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
        if bytes.len() < GRAPH_HEADER_SIZE {
            return Err("Mmap graph file is too small".to_string());
        }
        if &bytes[..8] != GRAPH_MAGIC {
            return Err("Invalid mmap graph magic".to_string());
        }
        let node_count = read_u64(bytes, 8)? as usize;
        let offsets_start = GRAPH_HEADER_SIZE;
        let offsets_bytes = (node_count + 1) * 8;
        let edges_start = offsets_start + offsets_bytes;
        if bytes.len() < edges_start {
            return Err("Mmap graph file is truncated".to_string());
        }
        Ok((node_count, offsets_start, edges_start))
    }

    fn get_object(&self, id: usize) -> Option<&[f32]> {
        if id == 0 || id > self.object_count {
            return None;
        }
        let dim = self.property.dimension;
        let start = OBJECT_HEADER_SIZE + (id - 1) * dim * 4;
        let end = start + dim * 4;
        if end > self.objects_mmap.len() {
            return None;
        }
        let bytes = &self.objects_mmap[start..end];
        let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() || values.len() != dim {
            return None;
        }
        Some(values)
    }

    fn neighbors(&self, id: usize) -> Result<NeighborIter<'_>, String> {
        if id == 0 || id > self.object_count {
            return Ok(NeighborIter { bytes: &[] });
        }
        let start_edges = self.offset_value(id - 1)? as usize;
        let end_edges = self.offset_value(id)? as usize;
        let start = self.graph_edges_start + start_edges * EDGE_RECORD_SIZE;
        let end = self.graph_edges_start + end_edges * EDGE_RECORD_SIZE;
        if end > self.graph_mmap.len() || start > end {
            return Err("Mmap graph edge slice is out of bounds".to_string());
        }
        Ok(NeighborIter {
            bytes: &self.graph_mmap[start..end],
        })
    }

    fn offset_value(&self, index: usize) -> Result<u64, String> {
        read_u64(&self.graph_mmap, self.graph_offsets_start + index * 8)
    }

    fn get_seeds(&self, prepared_query: &[f32], k: usize) -> Result<Vec<ObjectDistance>, String> {
        if matches!(
            self.property.index_type,
            crate::index::IndexType::GraphAndTree
        ) {
            let tree_seeds = self.get_seeds_from_tree(prepared_query, k)?;
            if !tree_seeds.is_empty() {
                return Ok(tree_seeds);
            }
        }
        self.get_seeds_from_graph(k)
    }

    fn get_seeds_from_tree(
        &self,
        prepared_query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, String> {
        let Some(tree) = &self.tree else {
            return Ok(Vec::new());
        };
        let Some(leaf_id) = self.greedy_leaf_for_query(tree, prepared_query) else {
            return Ok(Vec::new());
        };
        let mut seeds = tree.get_object_ids_from_leaf(leaf_id);
        self.thin_tree_seeds(&mut seeds, k);
        Ok(seeds)
    }

    fn get_seeds_from_graph(&self, k: usize) -> Result<Vec<ObjectDistance>, String> {
        let repository_size = self.object_count;
        if repository_size == 0 {
            return Ok(Vec::new());
        }
        let seed_size = self.effective_seed_count(k).min(repository_size);
        let mut seeds = Vec::new();
        match self.property.seed_type {
            crate::index::SeedType::FixedNodes => {
                for id in 1..=seed_size {
                    seeds.push(ObjectDistance {
                        id: id as u32,
                        distance: 0.0,
                    });
                }
            }
            crate::index::SeedType::FirstNode => {
                seeds.push(ObjectDistance {
                    id: 1,
                    distance: 0.0,
                });
            }
            _ => {
                for id in 1..=seed_size {
                    seeds.push(ObjectDistance {
                        id: id as u32,
                        distance: 0.0,
                    });
                }
            }
        }
        Ok(seeds)
    }

    fn effective_seed_count(&self, k: usize) -> usize {
        if self.property.seed_size == 0 {
            k.max(1)
        } else {
            self.property.seed_size
        }
    }

    fn thin_tree_seeds(&self, seeds: &mut Vec<ObjectDistance>, k: usize) {
        if seeds.is_empty()
            || matches!(
                self.property.seed_type,
                crate::index::SeedType::AllLeafNodes
            )
        {
            return;
        }
        let target = match self.property.seed_type {
            crate::index::SeedType::FirstNode => 1,
            _ => self.effective_seed_count(k).min(k.max(1)),
        };
        if seeds.len() > target {
            seeds.truncate(target);
        }
    }

    fn greedy_leaf_for_query(&self, tree: &DvpTree, query: &[f32]) -> Option<usize> {
        let mut current = tree.root;
        loop {
            match current {
                TreeNodeRef::Leaf(leaf_id) => return Some(leaf_id),
                TreeNodeRef::Internal(internal_id) => {
                    let node = tree
                        .internals
                        .get(internal_id)
                        .and_then(|node| node.as_ref())?;
                    let pivot = self.get_object(node.pivot_id as usize)?;
                    let distance = self.object_space.compare(query, pivot);
                    let region = select_region(distance, &node.borders)?;
                    current = node.children[region];
                }
            }
        }
    }

    fn get_edge_size(
        &self,
        edge_size: isize,
        exploration_coefficient: f64,
    ) -> Result<usize, String> {
        let edge_size = if edge_size == -1 {
            self.property.edge_size_for_search as i64
        } else {
            edge_size as i64
        };
        if edge_size == 0 {
            return Ok(usize::MAX);
        }
        if edge_size > 0 {
            return Ok(edge_size as usize);
        }
        if edge_size == -2 {
            let add = 10f64.powf(
                (exploration_coefficient - 1.0) * self.property.dynamic_edge_size_rate as f64,
            );
            return Ok((self.property.dynamic_edge_size_base as f64 + add) as usize);
        }
        Err(format!("Invalid edge size: {edge_size}"))
    }

    fn is_checked(&self, checked: &[u64], id: usize) -> bool {
        let block = id / 64;
        let bit = 1u64 << (id % 64);
        block < checked.len() && (checked[block] & bit) != 0
    }

    fn mark_checked(&self, checked: &mut [u64], id: usize) {
        let block = id / 64;
        if block < checked.len() {
            checked[block] |= 1u64 << (id % 64);
        }
    }
}

pub struct NeighborIter<'a> {
    bytes: &'a [u8],
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = ObjectDistance;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.len() < EDGE_RECORD_SIZE {
            return None;
        }
        let id = u32::from_le_bytes(self.bytes[..4].try_into().ok()?);
        let distance = f32::from_le_bytes(self.bytes[4..8].try_into().ok()?);
        self.bytes = &self.bytes[EDGE_RECORD_SIZE..];
        Some(ObjectDistance { id, distance })
    }
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "Unexpected EOF while reading u64".to_string())?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, String> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "Unexpected EOF while reading f32".to_string())?;
    Ok(f32::from_le_bytes(slice.try_into().unwrap()))
}

fn select_region(distance: f32, borders: &[f32]) -> Option<usize> {
    for (idx, border) in borders.iter().enumerate() {
        if distance < *border {
            return Some(idx);
        }
    }
    Some(borders.len())
}
