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
    codes: Vec<u8>,
    /// Per-object per-block scales: `scales[id * num_blocks .. (id+1) * num_blocks]`.
    scales: Vec<f32>,
    /// QJL sign bits: `qjl_signs[id * sign_bytes .. (id+1) * sign_bytes]`.
    qjl_signs: Vec<u8>,
    /// QJL residual norms: `gammas[id]`.
    gammas: Vec<f32>,
    /// Presence bitmap.
    present: Vec<bool>,
    pub quantizer: TqQuantizer,
}

impl TqObjectSpace {
    pub fn new(quantizer: TqQuantizer, distance_type: DistanceType) -> Self {
        let pd = quantizer.padded_dim();
        let nb = quantizer.num_blocks();
        let sb = (pd + 7) / 8;
        TqObjectSpace {
            dim: quantizer.dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count: 1,
            live_count: 0,
            codes: vec![0u8; pd],       // slot 0
            scales: vec![0.0f32; nb],    // slot 0
            qjl_signs: vec![0u8; sb],    // slot 0
            gammas: vec![0.0f32],        // slot 0
            present: vec![false],
            quantizer,
        }
    }

    fn sign_bytes(&self) -> usize {
        (self.quantizer.padded_dim() + 7) / 8
    }

    /// Quantize and insert.
    pub fn insert(&mut self, v: &[f32]) -> ObjectID {
        let enc = self.quantizer.encode(v);
        let id = self.slot_count as ObjectID;
        self.codes.extend_from_slice(&enc.codes);
        self.scales.extend_from_slice(&enc.scales);
        self.qjl_signs.extend_from_slice(&enc.qjl_signs);
        self.gammas.push(enc.gamma);
        self.present.push(true);
        self.slot_count += 1;
        self.live_count += 1;
        id
    }

    /// Insert placeholder for absent slot.
    fn insert_empty(&mut self) {
        let pd = self.quantizer.padded_dim();
        let nb = self.quantizer.num_blocks();
        let sb = self.sign_bytes();
        self.codes.extend(std::iter::repeat(0u8).take(pd));
        self.scales.extend(std::iter::repeat(0.0f32).take(nb));
        self.qjl_signs.extend(std::iter::repeat(0u8).take(sb));
        self.gammas.push(0.0);
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

    /// Get QJL sign bits for object `id`.
    #[inline]
    fn get_qjl_signs(&self, id: ObjectID) -> &[u8] {
        let sb = self.sign_bytes();
        let off = id as usize * sb;
        &self.qjl_signs[off..off + sb]
    }

    /// Get QJL gamma (residual norm) for object `id`.
    #[inline]
    fn get_gamma(&self, id: ObjectID) -> f32 {
        self.gammas[id as usize]
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

        let sb = self.sign_bytes();
        w.write_all(&(sb as u32).to_le_bytes()).map_err(|e| format!("{}", e))?;

        // Codes.
        w.write_all(&self.codes).map_err(|e| format!("{}", e))?;

        // Scales.
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const u8, self.scales.len() * 4)
        };
        w.write_all(scale_bytes).map_err(|e| format!("{}", e))?;

        // QJL sign bits.
        w.write_all(&self.qjl_signs).map_err(|e| format!("{}", e))?;

        // QJL gammas.
        let gamma_bytes = unsafe {
            std::slice::from_raw_parts(self.gammas.as_ptr() as *const u8, self.gammas.len() * 4)
        };
        w.write_all(gamma_bytes).map_err(|e| format!("{}", e))?;

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
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let sb = u32::from_le_bytes(buf4) as usize;

        // Codes.
        let mut codes = vec![0u8; slot_count * pd];
        r.read_exact(&mut codes).map_err(|e| format!("{}", e))?;

        // Scales.
        let mut scales = vec![0.0f32; slot_count * nb];
        let scale_bytes = unsafe {
            std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, scales.len() * 4)
        };
        r.read_exact(scale_bytes).map_err(|e| format!("{}", e))?;

        // QJL sign bits.
        let mut qjl_signs = vec![0u8; slot_count * sb];
        r.read_exact(&mut qjl_signs).map_err(|e| format!("{}", e))?;

        // QJL gammas.
        let mut gammas = vec![0.0f32; slot_count];
        let gamma_bytes = unsafe {
            std::slice::from_raw_parts_mut(gammas.as_mut_ptr() as *mut u8, gammas.len() * 4)
        };
        r.read_exact(gamma_bytes).map_err(|e| format!("{}", e))?;

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
            qjl_signs,
            gammas,
            present,
            quantizer,
        })
    }
}

// ---------------------------------------------------------------------------
// TqIndex
// ---------------------------------------------------------------------------

/// Precomputed QJL query state: total sum + pointer to q_qjl data.
/// `dot(q, ±1) = 2 * sum_where_bit_is_1(q) - sum_all(q)`
/// `sum_all` is precomputed once. Per neighbor: just compute the masked sum.
struct QjlQueryState {
    /// Sum of all q_qjl values (precomputed once per query).
    sum_all: f32,
    qjl_scale: f32,
}

impl QjlQueryState {
    fn new(q_qjl: &[f32], qjl_scale: f32) -> Self {
        let sum_all: f32 = q_qjl.iter().sum();
        QjlQueryState { sum_all, qjl_scale }
    }
}

/// Compute the QJL dot-product correction for one neighbor.
/// Uses the identity: dot(q, ±1) = 2 * masked_sum - sum_all
#[inline]
fn qjl_dot_correction(
    q_qjl: &[f32],
    signs: &[u8],
    gamma: f32,
    state: &QjlQueryState,
    pd: usize,
) -> f32 {
    if gamma == 0.0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    let masked_sum = unsafe { qjl_masked_sum_neon(q_qjl, signs, pd) };

    #[cfg(not(target_arch = "aarch64"))]
    let masked_sum = qjl_masked_sum_scalar(q_qjl, signs, pd);

    let dot = 2.0 * masked_sum - state.sum_all;
    gamma * state.qjl_scale * dot
}

/// Scalar masked sum: sum q_qjl[i] where sign bit i is 1.
#[inline]
fn qjl_masked_sum_scalar(q_qjl: &[f32], signs: &[u8], pd: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..pd {
        if (signs[i / 8] >> (i % 8)) & 1 != 0 {
            sum += q_qjl[i];
        }
    }
    sum
}

/// NEON masked sum: process 4 floats per sign-byte bit group.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn qjl_masked_sum_neon(q_qjl: &[f32], signs: &[u8], pd: usize) -> f32 { unsafe {
    use std::arch::aarch64::*;

    let qptr = q_qjl.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let mut byte_idx = 0usize;

    // Process 8 floats (1 byte of signs) per iteration.
    while i + 8 <= pd {
        let byte = *signs.get_unchecked(byte_idx);

        // Lower 4 bits → 4 floats.
        let q0 = vld1q_f32(qptr.add(i));
        // Create mask from bits 0-3: expand each bit to a full 32-bit lane.
        let mask0 = vmvnq_u32(vsubq_u32(
            vandq_u32(
                vdupq_n_u32(byte as u32),
                vld1q_u32([1u32, 2, 4, 8].as_ptr()),
            ),
            vdupq_n_u32(1),
        ));
        // Bit-select: keep q0 where mask is all-1s, else 0.
        let selected0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(q0), mask0));
        acc0 = vaddq_f32(acc0, selected0);

        // Upper 4 bits → next 4 floats.
        let q1 = vld1q_f32(qptr.add(i + 4));
        let mask1 = vmvnq_u32(vsubq_u32(
            vandq_u32(
                vdupq_n_u32((byte >> 4) as u32),
                vld1q_u32([1u32, 2, 4, 8].as_ptr()),
            ),
            vdupq_n_u32(1),
        ));
        let selected1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(q1), mask1));
        acc1 = vaddq_f32(acc1, selected1);

        i += 8;
        byte_idx += 1;
    }

    acc0 = vaddq_f32(acc0, acc1);
    let mut sum = vaddvq_f32(acc0);

    // Scalar tail.
    while i < pd {
        if (signs[i / 8] >> (i % 8)) & 1 != 0 {
            sum += *qptr.add(i);
        }
        i += 1;
    }
    sum
}}

/// Compute distance with QJL correction applied to the inner product.
/// For NormalizedCosine: |1 - (dot_mse + qjl_correction)|
/// For L2: uses the MSE distance directly (QJL correction is for inner product only).
#[inline]
fn distance_with_qjl_correction(
    q_rot: &[f32],
    dequant: &[f32],
    qjl_corr: f32,
    dt: DistanceType,
) -> f32 {
    match dt {
        DistanceType::NormalizedCosineSimilarity | DistanceType::CosineSimilarity => {
            // dot(q_rot, dequant) + qjl_correction
            let mut dot = 0.0f32;
            for i in 0..q_rot.len() {
                dot += q_rot[i] * dequant[i];
            }
            (1.0 - (dot + qjl_corr)).abs()
        }
        _ => {
            // For L2 and other metrics, use standard distance (QJL correction
            // is designed for inner product / cosine).
            primitive_comparator::compare(q_rot, dequant, dt)
        }
    }
}

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
        let mut tq_objects = TqObjectSpace::new(quantizer, dt);

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
        let dim = self.tq_objects.dim;
        let dt = self.tq_objects.distance_type;
        let qjl_scale = (std::f32::consts::FRAC_PI_2 as f32).sqrt() / (pd as f32);

        let mut q_rot = vec![0.0f32; pd];
        self.tq_objects.quantizer.rotation.forward(q, &mut q_rot);

        let mut q_qjl = vec![0.0f32; pd];
        self.tq_objects.quantizer.qjl_rotation.forward(&q_rot[..dim], &mut q_qjl);

        let qjl_state = QjlQueryState::new(&q_qjl, qjl_scale);

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
            let qjl_corr = qjl_dot_correction(
                &q_qjl, self.tq_objects.get_qjl_signs(oid),
                self.tq_objects.get_gamma(oid), &qjl_state, pd,
            );
            let d = distance_with_qjl_correction(&q_rot, &dequant, qjl_corr, dt);
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
        let dim = self.tq_objects.dim;
        let dt = self.tq_objects.distance_type;
        let qjl_scale = (std::f32::consts::FRAC_PI_2 as f32).sqrt() / (pd as f32);

        // Rotate query ONCE via WHT1.
        let mut q_rot = vec![0.0f32; pd];
        self.tq_objects.quantizer.rotation.forward(query, &mut q_rot);

        // QJL: also rotate query via WHT2 (once per query).
        let mut q_qjl = vec![0.0f32; pd];
        self.tq_objects.quantizer.qjl_rotation.forward(&q_rot[..dim], &mut q_qjl);

        let qjl_state = QjlQueryState::new(&q_qjl, qjl_scale);
        let mut dequant = vec![0.0f32; pd];

        // Compute seed distances with QJL correction.
        for seed in seeds.iter_mut() {
            let sid: u32 = seed.id;
            if self.tq_objects.is_present(sid) {
                self.tq_objects.quantizer.dequantize_rotated(
                    self.tq_objects.get_codes(sid),
                    self.tq_objects.get_scales(sid),
                    &mut dequant,
                );
                // Apply QJL correction to dot product.
                let qjl_corr = qjl_dot_correction(
                    &q_qjl, self.tq_objects.get_qjl_signs(sid),
                    self.tq_objects.get_gamma(sid), &qjl_state, pd,
                );
                seed.distance = distance_with_qjl_correction(
                    &q_rot, &dequant, qjl_corr, dt,
                );
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

                // Dequantize in rotated domain + QJL correction.
                self.tq_objects.quantizer.dequantize_rotated(
                    self.tq_objects.get_codes(nid),
                    self.tq_objects.get_scales(nid),
                    &mut dequant,
                );
                let qjl_corr = qjl_dot_correction(
                    &q_qjl, self.tq_objects.get_qjl_signs(nid),
                    self.tq_objects.get_gamma(nid), &qjl_state, pd,
                );
                let distance = distance_with_qjl_correction(
                    &q_rot, &dequant, qjl_corr, dt,
                );

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
