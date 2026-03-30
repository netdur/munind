/// TurboQuant — block-based vector quantization for compressed ANN search.
///
/// Based on TurboQuant (arXiv 2504.19874) with MNN implementation insights:
/// - Block-based WHT (32 values per block, 5 butterfly stages)
/// - Hardcoded Lloyd-Max codebooks (no training)
/// - Per-block RMS scale
/// - LUT-based search in rotated domain

pub mod codebook;
#[cfg(target_os = "macos")]
pub mod metal_kernel;
pub mod quantizer;
pub mod rotation;

use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, SearchOptions};
use crate::graph::{BooleanVector, GraphProperty, NeighborhoodGraph};
use crate::index::IndexProperty;
use crate::node::NodeType;
use crate::object_space::ObjectSpace;
use crate::primitive_comparator::{self, DistanceType};
use crate::tree::DVPTree;

use self::quantizer::TqQuantizer;
use self::rotation::BLOCK_SIZE;

// ---------------------------------------------------------------------------
// TqObjectSpace — quantized object storage
// ---------------------------------------------------------------------------

pub struct TqObjectSpace {
    pub dim: usize,
    pub distance_type: DistanceType,
    pub normalization: bool,
    pub slot_count: usize,
    pub live_count: usize,
    /// Per-object codes: `codes[id * padded_dim .. (id+1) * padded_dim]`.
    /// Flat contiguous storage.
    codes: Vec<u8>,
    /// Per-object per-block scales: `scales[id * num_blocks .. (id+1) * num_blocks]`.
    scales: Vec<f32>,
    /// Presence bitmap.
    present: Vec<bool>,
    pub quantizer: TqQuantizer,
}

impl TqObjectSpace {
    pub fn new(quantizer: TqQuantizer, distance_type: DistanceType) -> Self {
        let pd = quantizer.padded_dim();
        let nb = quantizer.num_blocks();
        TqObjectSpace {
            dim: quantizer.dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count: 1,
            live_count: 0,
            codes: vec![0u8; pd],      // slot 0
            scales: vec![0.0f32; nb],   // slot 0
            present: vec![false],
            quantizer,
        }
    }

    /// Quantize and insert.
    pub fn insert(&mut self, v: &[f32]) -> ObjectID {
        let enc = self.quantizer.encode(v);
        let id = self.slot_count as ObjectID;
        self.codes.extend_from_slice(&enc.codes);
        self.scales.extend_from_slice(&enc.scales);
        self.present.push(true);
        self.slot_count += 1;
        self.live_count += 1;
        id
    }

    /// Insert placeholder for absent slot.
    fn insert_empty(&mut self) {
        let pd = self.quantizer.padded_dim();
        let nb = self.quantizer.num_blocks();
        self.codes.extend(std::iter::repeat(0u8).take(pd));
        self.scales.extend(std::iter::repeat(0.0f32).take(nb));
        self.present.push(false);
        self.slot_count += 1;
    }

    /// Get codes slice for object `id`.
    #[inline]
    fn get_codes(&self, id: ObjectID) -> &[u8] {
        let pd = self.quantizer.padded_dim();
        let off = id as usize * pd;
        &self.codes[off..off + pd]
    }

    /// Get scales slice for object `id`.
    #[inline]
    fn get_scales(&self, id: ObjectID) -> &[f32] {
        let nb = self.quantizer.num_blocks();
        let off = id as usize * nb;
        &self.scales[off..off + nb]
    }

    pub fn is_present(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx > 0 && idx < self.slot_count && self.present[idx]
    }

    pub fn count(&self) -> usize { self.live_count }
    pub fn size(&self) -> usize { self.slot_count }

    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("TqObjectSpace::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);

        let pd = self.quantizer.padded_dim();
        let nb = self.quantizer.num_blocks();
        w.write_all(&(self.slot_count as u64).to_le_bytes()).map_err(|e| format!("{}", e))?;
        w.write_all(&(pd as u32).to_le_bytes()).map_err(|e| format!("{}", e))?;
        w.write_all(&(nb as u32).to_le_bytes()).map_err(|e| format!("{}", e))?;
        w.write_all(&self.quantizer.bits.to_le_bytes()).map_err(|e| format!("{}", e))?;

        // Codes: flat contiguous.
        w.write_all(&self.codes).map_err(|e| format!("{}", e))?;

        // Scales: flat contiguous.
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const u8, self.scales.len() * 4)
        };
        w.write_all(scale_bytes).map_err(|e| format!("{}", e))?;

        // Presence bitmap.
        let bitmap: Vec<u8> = self.present.iter().map(|&p| if p { 1u8 } else { 0u8 }).collect();
        w.write_all(&bitmap).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load(path: &str, quantizer: TqQuantizer, distance_type: DistanceType) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("TqObjectSpace::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let pd = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let nb = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let _bits = u32::from_le_bytes(buf4);

        // Codes.
        let mut codes = vec![0u8; slot_count * pd];
        r.read_exact(&mut codes).map_err(|e| format!("{}", e))?;

        // Scales.
        let mut scales = vec![0.0f32; slot_count * nb];
        let scale_bytes = unsafe {
            std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, scales.len() * 4)
        };
        r.read_exact(scale_bytes).map_err(|e| format!("{}", e))?;

        // Presence.
        let mut bitmap = vec![0u8; slot_count];
        r.read_exact(&mut bitmap).map_err(|e| format!("{}", e))?;
        let present: Vec<bool> = bitmap.iter().map(|&b| b != 0).collect();
        let live_count = present.iter().skip(1).filter(|&&p| p).count();

        Ok(TqObjectSpace {
            dim: quantizer.dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count,
            live_count,
            codes,
            scales,
            present,
            quantizer,
        })
    }
}

// ---------------------------------------------------------------------------
// TqIndex
// ---------------------------------------------------------------------------

pub struct TqIndex {
    pub graph: NeighborhoodGraph,
    pub tree: Option<DVPTree>,
    pub tq_objects: TqObjectSpace,
    pub property: IndexProperty,
    pub bits: u32,
}

impl TqIndex {
    pub fn build_from_index(index_dir: &str, bits: u32) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", index_dir))?;
        let property = IndexProperty::import_from(&ps);
        let dt = property.to_distance_type();

        let mut os = ObjectSpace::new(property.dimension, dt);
        os.deserialize(&format!("{}/obj", index_dir))?;

        let quantizer = TqQuantizer::new(property.dimension, bits);
        let mut tq_objects = TqObjectSpace::new(
            TqQuantizer::new(property.dimension, bits),
            dt,
        );
        // Rebuild with same rotation.
        tq_objects = TqObjectSpace::new(quantizer, dt);

        for id in 1..os.size() {
            let oid = id as ObjectID;
            if os.is_present(oid) {
                let obj = os.get_object(oid)?;
                tq_objects.insert(obj);
            } else {
                tq_objects.insert_empty();
            }
        }

        let gp = GraphProperty {
            edge_size_for_search: property.edge_size_for_search,
            ..GraphProperty::default()
        };
        let mut graph = NeighborhoodGraph::with_property(gp);
        graph.deserialize_from_file(&format!("{}/grp", index_dir))?;

        let tree = if std::path::Path::new(&format!("{}/tre", index_dir)).exists() {
            let mut t = DVPTree::new(property.leaf_node_size, property.internal_children_size);
            t.deserialize_from_file(&format!("{}/tre", index_dir), property.dimension)?;
            Some(t)
        } else {
            None
        };

        Ok(TqIndex { graph, tree, tq_objects, property, bits })
    }

    pub fn save(&self, dir: &str) -> Result<(), NgtError> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}", e))?;
        let mut ps = PropertySet::new();
        self.property.export_to(&mut ps);
        ps.set_str("TqBits", self.bits);
        ps.save(&format!("{}/prf", dir))?;
        self.graph.serialize_to_file(&format!("{}/grp", dir))?;
        if let Some(tree) = &self.tree {
            tree.serialize_to_file(&format!("{}/tre", dir), self.property.dimension)?;
        }
        self.tq_objects.quantizer.save(dir)?;
        self.tq_objects.save(&format!("{}/obj.tq", dir))?;
        Ok(())
    }

    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let property = IndexProperty::import_from(&ps);
        let bits = ps.get_i64("TqBits", 8) as u32;
        let dt = property.to_distance_type();

        let quantizer = TqQuantizer::load(dir)?;
        let tq_objects = TqObjectSpace::load(&format!("{}/obj.tq", dir), quantizer, dt)?;

        let gp = GraphProperty {
            edge_size_for_search: property.edge_size_for_search,
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

        Ok(TqIndex { graph, tree, tq_objects, property, bits })
    }

    pub fn is_tq_index(dir: &str) -> bool {
        std::path::Path::new(&format!("{}/obj.tq", dir)).exists()
    }

    pub fn object_count(&self) -> usize { self.tq_objects.count() }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        if options.k == 0 { return Ok(Vec::new()); }

        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.tq_objects.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let mut seeds = self.get_seeds(q)?;
        if seeds.is_empty() { return Ok(Vec::new()); }

        let edge_size = match options.edge_size {
            Some(es) => es as i32,
            None => -1,
        };

        Ok(self.search_lut(q, &mut seeds, options.k, options.epsilon, edge_size))
    }

    pub fn linear_search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.tq_objects.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let pd = self.tq_objects.quantizer.padded_dim();
        let dt = self.tq_objects.distance_type;
        let mut q_rot = vec![0.0f32; pd];
        self.tq_objects.quantizer.rotation.forward(q, &mut q_rot);

        let mut dequant = vec![0.0f32; pd];
        let mut results = crate::common::ResultSet::with_capacity(k + 1);

        for id in 1..self.tq_objects.slot_count {
            let oid = id as ObjectID;
            if !self.tq_objects.is_present(oid) { continue; }
            self.tq_objects.quantizer.dequantize_rotated(
                self.tq_objects.get_codes(oid),
                self.tq_objects.get_scales(oid),
                &mut dequant,
            );
            let d = primitive_comparator::compare(&q_rot, &dequant, dt);
            results.push(ObjectDistance::new(oid, d));
            if results.len() > k { results.pop(); }
        }

        let mut v = results.into_sorted_vec();
        v.truncate(k);
        Ok(v)
    }

    fn get_seeds(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        if let Some(tree) = &self.tree {
            if let Ok(leaf_nid) = self.search_tree_leaf(tree, query) {
                let seeds = tree.get_object_ids_from_leaf(leaf_nid);
                if !seeds.is_empty() { return Ok(seeds); }
            }
        }
        let mut seeds = Vec::new();
        for id in 1..self.tq_objects.size() {
            let oid = id as ObjectID;
            if self.tq_objects.is_present(oid) {
                seeds.push(ObjectDistance::new(oid, 0.0));
                if seeds.len() >= 10 { break; }
            }
        }
        Ok(seeds)
    }

    fn search_tree_leaf(&self, tree: &DVPTree, query: &[f32]) -> Result<crate::node::NodeId, NgtError> {
        let root_id = if tree.internal_nodes.len() > 1 && tree.internal_nodes[1].is_some() {
            crate::node::NodeId::internal(1)
        } else if tree.leaf_nodes.len() > 1 && tree.leaf_nodes[1].is_some() {
            crate::node::NodeId::leaf(1)
        } else {
            return Err("no root".to_string());
        };
        if root_id.get_type() == NodeType::Leaf { return Ok(root_id); }
        let mut current = root_id;
        loop {
            match current.get_type() {
                NodeType::Internal => {
                    let node = tree.get_internal(current.get_id()).ok_or("not found")?;
                    let pivot = node.pivot.as_ref().unwrap();
                    let d = primitive_comparator::compare(query, pivot, self.tq_objects.distance_type);
                    let bsize = node.borders.len();
                    let mut child_idx = bsize;
                    for mid in 0..bsize {
                        if d < node.borders[mid] { child_idx = mid; break; }
                    }
                    current = node.children[child_idx];
                }
                NodeType::Leaf => return Ok(current),
            }
        }
    }

    /// LUT-based search in rotated domain.
    fn search_lut(
        &self,
        query: &[f32],
        seeds: &mut Vec<ObjectDistance>,
        k: usize,
        epsilon: f32,
        edge_size: i32,
    ) -> Vec<ObjectDistance> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let exploration_coefficient = if epsilon == 0.0 { 1.1 } else { epsilon + 1.0 };
        let edge_size: usize = {
            let es = if edge_size == -1 {
                self.graph.property.edge_size_for_search as i64
            } else { edge_size as i64 };
            if es <= 0 { usize::MAX } else { es as usize }
        };

        let pd = self.tq_objects.quantizer.padded_dim();
        let dt = self.tq_objects.distance_type;

        // Rotate query ONCE.
        let mut q_rot = vec![0.0f32; pd];
        self.tq_objects.quantizer.rotation.forward(query, &mut q_rot);

        // Dequant buffer for distance computation.
        let mut dequant = vec![0.0f32; pd];

        // Compute seed distances.
        for seed in seeds.iter_mut() {
            let sid: u32 = seed.id;
            if self.tq_objects.is_present(sid) {
                self.tq_objects.quantizer.dequantize_rotated(
                    self.tq_objects.get_codes(sid),
                    self.tq_objects.get_scales(sid),
                    &mut dequant,
                );
                seed.distance = primitive_comparator::compare(&q_rot, &dequant, dt);
            } else {
                seed.distance = f32::MAX;
            }
        }
        seeds.sort_unstable_by(|a, b| a.cmp(b));

        let mut results: BinaryHeap<ObjectDistance> = BinaryHeap::new();
        let mut unchecked: BinaryHeap<Reverse<ObjectDistance>> = BinaryHeap::new();
        let mut checked = BooleanVector::new(self.graph.nodes.len());
        let mut current_radius = f32::MAX;

        for s in seeds.iter() {
            let sd: f32 = s.distance;
            let si: u32 = s.id;
            if results.len() < k && sd <= current_radius { results.push(*s); }
            if sd < f32::MAX { checked.insert(si); unchecked.push(Reverse(*s)); }
        }
        if results.len() >= k {
            current_radius = results.peek().unwrap().distance;
        }
        let mut exploration_radius = exploration_coefficient * current_radius;

        while let Some(Reverse(target)) = unchecked.pop() {
            let td: f32 = target.distance;
            if td > exploration_radius { break; }
            let tid: u32 = target.id;
            let neighbors = match self.graph.get_node(tid) {
                Some(n) => n,
                None => continue,
            };
            if neighbors.is_empty() { continue; }
            let nsize = neighbors.len().min(edge_size);

            for ni in 0..nsize {
                let nid: u32 = neighbors[ni].id;
                if checked.contains(nid) { continue; }
                checked.insert(nid);
                if !self.tq_objects.is_present(nid) { continue; }

                // Dequantize in rotated domain + distance.
                self.tq_objects.quantizer.dequantize_rotated(
                    self.tq_objects.get_codes(nid),
                    self.tq_objects.get_scales(nid),
                    &mut dequant,
                );
                let distance = primitive_comparator::compare(&q_rot, &dequant, dt);

                if distance <= exploration_radius {
                    let result = ObjectDistance::new(nid, distance);
                    unchecked.push(Reverse(result));
                    if distance <= current_radius {
                        results.push(result);
                        if results.len() >= k {
                            if let Some(top) = results.peek() {
                                let topd: f32 = top.distance;
                                if topd >= distance {
                                    if results.len() > k { results.pop(); }
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
}
