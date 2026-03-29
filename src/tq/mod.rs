/// TurboQuant — data-oblivious vector quantization for compressed ANN search.
///
/// Usage:
///   Build:  `-tq <bits>` flag on `munind create`
///   Search: auto-detected from saved files

pub mod codebook;
pub mod quantizer;
pub mod rotation;

use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, SearchOptions};
use crate::graph::{GraphProperty, NeighborhoodGraph};
use crate::index::{Graph, IndexProperty, IndexType, Tree};
use crate::mmap_index::ObjectAccessor;
use crate::object_space::ObjectSpace;
use crate::primitive_comparator::{self, DistanceType};
use crate::tree::DVPTree;

use self::quantizer::TqQuantizer;

// ---------------------------------------------------------------------------
// TqObjectSpace — quantized object storage with on-the-fly decode
// ---------------------------------------------------------------------------

/// Stores quantized codes + norms.  Decodes on the fly for distance computation.
pub struct TqObjectSpace {
    pub dim: usize,
    pub distance_type: DistanceType,
    pub normalization: bool,
    pub slot_count: usize,
    pub live_count: usize,
    /// Per-object: quantized codes.  `codes[id]` = Vec<u32> of length `dim`.
    codes: Vec<Option<Vec<u32>>>,
    /// Per-object L2 norm.
    norms: Vec<f32>,
    /// Presence bitmap.
    present: Vec<bool>,
    /// The quantizer (for decode).
    quantizer: TqQuantizer,
}

impl TqObjectSpace {
    pub fn new(quantizer: TqQuantizer, distance_type: DistanceType) -> Self {
        let dim = quantizer.dim;
        TqObjectSpace {
            dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count: 1,
            live_count: 0,
            codes: vec![None], // slot 0
            norms: vec![0.0],
            present: vec![false],
            quantizer,
        }
    }

    /// Quantize and insert a full-precision vector.
    pub fn insert(&mut self, v: &[f32]) -> ObjectID {
        let enc = self.quantizer.encode_mse(v);
        let id = self.slot_count as ObjectID;
        self.codes.push(Some(enc.codes));
        self.norms.push(enc.norm);
        self.present.push(true);
        self.slot_count += 1;
        self.live_count += 1;
        id
    }

    /// Decode object `id` into `out` buffer.
    #[inline]
    pub fn decode_into(&self, id: ObjectID, out: &mut [f32]) -> Result<(), NgtError> {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count || !self.present[idx] {
            return Err(format!("TqObjectSpace: invalid id {}", id));
        }
        let codes = self.codes[idx].as_ref().unwrap();
        self.quantizer
            .decode_mse_into(codes, self.norms[idx], out);
        Ok(())
    }

    pub fn is_present(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx > 0 && idx < self.slot_count && self.present[idx]
    }

    pub fn count(&self) -> usize {
        self.live_count
    }

    pub fn size(&self) -> usize {
        self.slot_count
    }

    /// Serialize quantized objects.
    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("TqObjectSpace::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);

        w.write_all(&(self.slot_count as u64).to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        w.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        w.write_all(&self.quantizer.bits.to_le_bytes())
            .map_err(|e| format!("{}", e))?;

        // Norms.
        for i in 0..self.slot_count {
            w.write_all(&self.norms[i].to_le_bytes())
                .map_err(|e| format!("{}", e))?;
        }

        // Codes: for simplicity, store as u8 per code (works for bits <= 8).
        for i in 0..self.slot_count {
            if let Some(codes) = &self.codes[i] {
                w.write_all(&[1u8]).map_err(|e| format!("{}", e))?;
                for &c in codes {
                    w.write_all(&[c as u8]).map_err(|e| format!("{}", e))?;
                }
            } else {
                w.write_all(&[0u8]).map_err(|e| format!("{}", e))?;
            }
        }

        // Presence bitmap.
        for i in 0..self.slot_count {
            let b: u8 = if self.present[i] { 1 } else { 0 };
            w.write_all(&[b]).map_err(|e| format!("{}", e))?;
        }

        Ok(())
    }

    /// Deserialize quantized objects.
    pub fn load(path: &str, quantizer: TqQuantizer, distance_type: DistanceType) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("TqObjectSpace::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let _bits = u32::from_le_bytes(buf4);

        // Norms.
        let mut norms = vec![0.0f32; slot_count];
        for i in 0..slot_count {
            r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
            norms[i] = f32::from_le_bytes(buf4);
        }

        // Codes.
        let mut codes: Vec<Option<Vec<u32>>> = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let mut flag = [0u8; 1];
            r.read_exact(&mut flag).map_err(|e| format!("{}", e))?;
            if flag[0] == 1 {
                let mut c = vec![0u32; dim];
                for j in 0..dim {
                    let mut b = [0u8; 1];
                    r.read_exact(&mut b).map_err(|e| format!("{}", e))?;
                    c[j] = b[0] as u32;
                }
                codes.push(Some(c));
            } else {
                codes.push(None);
            }
        }

        // Presence.
        let mut present = vec![false; slot_count];
        let mut live_count = 0;
        for i in 0..slot_count {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| format!("{}", e))?;
            present[i] = b[0] != 0;
            if present[i] && i > 0 {
                live_count += 1;
            }
        }

        Ok(TqObjectSpace {
            dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count,
            live_count,
            codes,
            norms,
            present,
            quantizer,
        })
    }
}

// Implement ObjectAccessor for TqObjectSpace so graph search works.
impl ObjectAccessor for TqObjectSpace {
    fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        // Can't return a slice — we decode on the fly.
        // This trait doesn't fit perfectly; we'll use a different search path.
        Err("TqObjectSpace: use decode_into instead".to_string())
    }

    fn is_present(&self, id: ObjectID) -> bool {
        self.is_present(id)
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        primitive_comparator::compare(a, b, self.distance_type)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn size(&self) -> usize {
        self.slot_count
    }
}

// ---------------------------------------------------------------------------
// TqIndex — graph + tree + quantized objects
// ---------------------------------------------------------------------------

pub struct TqIndex {
    pub graph: NeighborhoodGraph,
    pub tree: Option<DVPTree>,
    pub tq_objects: TqObjectSpace,
    pub property: IndexProperty,
    pub bits: u32,
    // Decode buffer (reused across searches to avoid allocation).
    decode_buf: Vec<f32>,
}

impl TqIndex {
    /// Build a TQ index from a full-precision index directory.
    /// Reads the existing graph/tree, quantizes all objects.
    pub fn build_from_index(
        index_dir: &str,
        bits: u32,
    ) -> Result<Self, NgtError> {
        // Load property.
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", index_dir))?;
        let property = IndexProperty::import_from(&ps);
        let dt = property.to_distance_type();

        // Load full-precision objects to quantize.
        let mut os = ObjectSpace::new(property.dimension, dt);
        os.deserialize(&format!("{}/obj", index_dir))?;

        // Create quantizer.
        let quantizer = TqQuantizer::new(property.dimension, bits, false);

        // Quantize all objects.
        let mut tq_objects = TqObjectSpace::new(
            TqQuantizer::new(property.dimension, bits, false),
            dt,
        );
        // We need to use the SAME quantizer instance for both storage and later decode.
        // Rebuild with the real quantizer.
        tq_objects = TqObjectSpace::new(quantizer, dt);

        for id in 1..os.size() {
            let oid = id as ObjectID;
            if os.is_present(oid) {
                let obj = os.get_object(oid)?;
                // Normalize if needed (for cosine, objects are already normalized in os).
                tq_objects.insert(obj);
            } else {
                // Push placeholder.
                tq_objects.codes.push(None);
                tq_objects.norms.push(0.0);
                tq_objects.present.push(false);
                tq_objects.slot_count += 1;
            }
        }

        // Load graph.
        let gp = GraphProperty {
            edge_size_for_search: property.edge_size_for_search,
            ..GraphProperty::default()
        };
        let mut graph = NeighborhoodGraph::with_property(gp);
        graph.deserialize_from_file(&format!("{}/grp", index_dir))?;

        // Load tree.
        let tree = if std::path::Path::new(&format!("{}/tre", index_dir)).exists() {
            let mut t = DVPTree::new(property.leaf_node_size, property.internal_children_size);
            t.deserialize_from_file(&format!("{}/tre", index_dir), property.dimension)?;
            Some(t)
        } else {
            None
        };

        let dim = property.dimension;
        Ok(TqIndex {
            graph,
            tree,
            tq_objects,
            property,
            bits,
            decode_buf: vec![0.0f32; dim],
        })
    }

    /// Save TQ index to directory.
    pub fn save(&self, dir: &str) -> Result<(), NgtError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("TqIndex::save: {}: {}", dir, e))?;

        // Property (with TQ marker).
        let mut ps = PropertySet::new();
        self.property.export_to(&mut ps);
        ps.set_str("TqBits", self.bits);
        ps.save(&format!("{}/prf", dir))?;

        // Graph.
        self.graph.serialize_to_file(&format!("{}/grp", dir))?;

        // Tree.
        if let Some(tree) = &self.tree {
            tree.serialize_to_file(&format!("{}/tre", dir), self.property.dimension)?;
        }

        // Quantizer (rotation + codebook).
        self.tq_objects.quantizer.save(dir)?;

        // Quantized objects.
        self.tq_objects.save(&format!("{}/obj.tq", dir))?;

        Ok(())
    }

    /// Load a TQ index.
    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let property = IndexProperty::import_from(&ps);
        let bits = ps.get_i64("TqBits", 4) as u32;
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

        let dim = property.dimension;
        Ok(TqIndex {
            graph,
            tree,
            tq_objects,
            property,
            bits,
            decode_buf: vec![0.0f32; dim],
        })
    }

    /// Detect if a directory contains a TQ index.
    pub fn is_tq_index(dir: &str) -> bool {
        std::path::Path::new(&format!("{}/obj.tq", dir)).exists()
    }

    /// Search the TQ index.
    pub fn search(
        &mut self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        if options.k == 0 {
            return Ok(Vec::new());
        }

        // Normalize query if needed.
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.tq_objects.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        // Get seeds from tree.
        let mut seeds = self.get_seeds(q)?;
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        let edge_size = match options.edge_size {
            Some(es) => es as i32,
            None => -1,
        };

        // Search with asymmetric distance: query is full precision,
        // database vectors are decoded from quantized codes.
        let results = self.search_asymmetric(
            q,
            &mut seeds,
            options.k,
            options.epsilon,
            edge_size,
        );

        Ok(results)
    }

    /// Linear search over all quantized objects (in rotated domain).
    pub fn linear_search(
        &mut self,
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

        let dim = self.tq_objects.dim;
        let dt = self.tq_objects.distance_type;

        // Rotate query once.
        let mut q_rot = vec![0.0f32; dim];
        self.tq_objects.quantizer.rotation.mul(q, &mut q_rot);

        let mut results = crate::common::ResultSet::with_capacity(k + 1);
        let mut buf = vec![0.0f32; dim];

        for id in 1..self.tq_objects.slot_count {
            let oid = id as ObjectID;
            if !self.tq_objects.is_present(oid) {
                continue;
            }
            self.dequantize_rotated(oid, &mut buf)?;
            let d = primitive_comparator::compare(&q_rot, &buf, dt);
            results.push(ObjectDistance::new(oid, d));
            if results.len() > k {
                results.pop();
            }
        }

        let mut v = results.into_sorted_vec();
        v.truncate(k);
        Ok(v)
    }

    fn get_seeds(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        if let Some(tree) = &self.tree {
            // Manual tree traversal (avoids needing ObjectSpace).
            if let Ok(leaf_nid) = self.search_tree_leaf(tree, query) {
                let seeds = tree.get_object_ids_from_leaf(leaf_nid);
                if !seeds.is_empty() {
                    return Ok(seeds);
                }
            }
        }
        // Fallback.
        let mut seeds = Vec::new();
        for id in 1..self.tq_objects.size() {
            let oid = id as ObjectID;
            if self.tq_objects.is_present(oid) {
                seeds.push(ObjectDistance::new(oid, 0.0));
                if seeds.len() >= 10 {
                    break;
                }
            }
        }
        Ok(seeds)
    }

    fn search_tree_leaf(
        &self,
        tree: &DVPTree,
        query: &[f32],
    ) -> Result<crate::node::NodeId, NgtError> {
        use crate::node::NodeType;
        let root_id = if tree.internal_nodes.len() > 1 && tree.internal_nodes[1].is_some() {
            crate::node::NodeId::internal(1)
        } else if tree.leaf_nodes.len() > 1 && tree.leaf_nodes[1].is_some() {
            crate::node::NodeId::leaf(1)
        } else {
            return Err("no root".to_string());
        };
        if root_id.get_type() == NodeType::Leaf {
            return Ok(root_id);
        }
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
                        if d < node.borders[mid] {
                            child_idx = mid;
                            break;
                        }
                    }
                    current = node.children[child_idx];
                }
                NodeType::Leaf => return Ok(current),
            }
        }
    }

    /// Dequantize in rotated domain (no matrix multiply — just scalar lookups).
    /// Returns the rotated, scaled vector: ỹ * norm.
    #[inline]
    fn dequantize_rotated(&self, id: ObjectID, out: &mut [f32]) -> Result<(), NgtError> {
        let idx = id as usize;
        let tq = &self.tq_objects;
        if idx == 0 || idx >= tq.slot_count || !tq.present[idx] {
            return Err(format!("invalid id {}", id));
        }
        let codes = tq.codes[idx].as_ref().unwrap();
        let norm = tq.norms[idx];
        let cb = &tq.quantizer.codebook;
        for i in 0..tq.dim {
            out[i] = cb.dequantize(codes[i]) * norm;
        }
        Ok(())
    }

    /// Asymmetric graph search in the ROTATED domain.
    ///
    /// Key optimization: rotate the query ONCE, then compare directly against
    /// dequantized codes (scalar lookups only — no d×d matrix multiply per object).
    /// This works because rotation preserves distances: ‖Πa - Πb‖ = ‖a - b‖.
    fn search_asymmetric(
        &mut self,
        query: &[f32],
        seeds: &mut Vec<ObjectDistance>,
        k: usize,
        epsilon: f32,
        edge_size: i32,
    ) -> Vec<ObjectDistance> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        use crate::graph::BooleanVector;

        let exploration_coefficient = if epsilon == 0.0 { 1.1 } else { epsilon + 1.0 };
        let edge_size: usize = {
            let es = if edge_size == -1 {
                self.graph.property.edge_size_for_search as i64
            } else {
                edge_size as i64
            };
            if es <= 0 { usize::MAX } else { es as usize }
        };

        let dim = self.tq_objects.dim;
        let dt = self.tq_objects.distance_type;

        // Rotate query ONCE: q_rot = Π · query.
        // All subsequent distances are computed in the rotated domain.
        let mut q_rot = vec![0.0f32; dim];
        self.tq_objects.quantizer.rotation.mul(query, &mut q_rot);

        let mut dequant_buf = vec![0.0f32; dim];

        // Compute seed distances in rotated domain.
        for seed in seeds.iter_mut() {
            let sid: u32 = seed.id;
            if self.tq_objects.is_present(sid) {
                if self.dequantize_rotated(sid, &mut dequant_buf).is_ok() {
                    seed.distance = primitive_comparator::compare(&q_rot, &dequant_buf, dt);
                } else {
                    seed.distance = f32::MAX;
                }
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
            if results.len() < k && sd <= current_radius {
                results.push(*s);
            }
            if sd < f32::MAX {
                checked.insert(si);
                unchecked.push(Reverse(*s));
            }
        }
        if results.len() >= k {
            current_radius = results.peek().unwrap().distance;
        }
        let mut exploration_radius = exploration_coefficient * current_radius;

        while let Some(Reverse(target)) = unchecked.pop() {
            let td: f32 = target.distance;
            if td > exploration_radius {
                break;
            }
            let tid: u32 = target.id;
            let neighbors = match self.graph.get_node(tid) {
                Some(n) => n,
                None => continue,
            };
            if neighbors.is_empty() {
                continue;
            }
            let nsize = neighbors.len().min(edge_size);

            for ni in 0..nsize {
                let nid: u32 = neighbors[ni].id;
                if checked.contains(nid) {
                    continue;
                }
                checked.insert(nid);
                if !self.tq_objects.is_present(nid) {
                    continue;
                }
                // Dequantize in rotated domain: O(d) scalar lookups, no matrix multiply.
                if self.dequantize_rotated(nid, &mut dequant_buf).is_err() {
                    continue;
                }
                let distance = primitive_comparator::compare(&q_rot, &dequant_buf, dt);

                if distance <= exploration_radius {
                    let result = ObjectDistance::new(nid, distance);
                    unchecked.push(Reverse(result));
                    if distance <= current_radius {
                        results.push(result);
                        if results.len() >= k {
                            if let Some(top) = results.peek() {
                                let topd: f32 = top.distance;
                                if topd >= distance {
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

    pub fn object_count(&self) -> usize {
        self.tq_objects.count()
    }
}
