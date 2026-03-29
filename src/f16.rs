/// Float16 compressed object storage.
///
/// Stores each f32 coordinate as IEEE 754 half-precision (16-bit).
/// 2× compression with near-zero recall loss (~3 decimal digits precision).
///
/// Format: same header as ObjectSpace flat format, but data is f16 (2 bytes/float).
///
/// File: `obj.f16` — auto-detected on search.

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, ResultSet, SearchOptions};
use crate::graph::{BooleanVector, GraphProperty, NeighborhoodGraph};
use crate::index::IndexProperty;
use crate::node::NodeType;
use crate::object_space::ObjectSpace;
use crate::primitive_comparator::{self, DistanceType};
use crate::tree::DVPTree;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// f32 <-> f16 conversion (IEEE 754 half-precision)
// ---------------------------------------------------------------------------

#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Denorm or zero
        if new_exp < -10 {
            return sign as u16;
        }
        let m = (man | 0x800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (man >> 13)) as u16
}

#[inline]
pub fn f16_to_f32(val: u16) -> f32 {
    let sign = ((val & 0x8000) as u32) << 16;
    let exp = ((val >> 10) & 0x1F) as u32;
    let man = (val & 0x3FF) as u32;

    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Denormalized
        let mut e = 1u32;
        let mut m = man;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) << 23;
        let man32 = (m & 0x3FF) << 13;
        return f32::from_bits(sign | exp32 | man32);
    }
    if exp == 31 {
        // Inf/NaN
        let man32 = man << 13;
        return f32::from_bits(sign | 0x7F800000 | man32);
    }

    let exp32 = (exp + 127 - 15) << 23;
    let man32 = man << 13;
    f32::from_bits(sign | exp32 | man32)
}

// ---------------------------------------------------------------------------
// save_f16 — convert existing obj to obj.f16
// ---------------------------------------------------------------------------

/// Read the flat obj file, convert to f16, write obj.f16.
pub fn save_f16(dir: &str) -> Result<(), NgtError> {
    let obj_path = format!("{}/obj", dir);
    let f16_path = format!("{}/obj.f16", dir);

    let f = std::fs::File::open(&obj_path)
        .map_err(|e| format!("save_f16: cannot open {}: {}", obj_path, e))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);

    // Read header.
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
    let slot_count = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
    let dim = u64::from_le_bytes(buf8) as usize;

    // Read all f32 data.
    let total_floats = slot_count * dim;
    let mut data_f32 = vec![0.0f32; total_floats];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(data_f32.as_mut_ptr() as *mut u8, total_floats * 4)
    };
    r.read_exact(bytes).map_err(|e| format!("{}", e))?;

    // Read bitmap.
    let mut bitmap = vec![0u8; slot_count];
    r.read_exact(&mut bitmap).map_err(|e| format!("{}", e))?;

    // Write f16 file.
    let f = std::fs::File::create(&f16_path)
        .map_err(|e| format!("save_f16: cannot create {}: {}", f16_path, e))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);

    // Same header.
    w.write_all(&(slot_count as u64).to_le_bytes()).map_err(|e| format!("{}", e))?;
    w.write_all(&(dim as u64).to_le_bytes()).map_err(|e| format!("{}", e))?;

    // Convert and write f16 data.
    for &v in &data_f32 {
        let h = f32_to_f16(v);
        w.write_all(&h.to_le_bytes()).map_err(|e| format!("{}", e))?;
    }

    // Same bitmap.
    w.write_all(&bitmap).map_err(|e| format!("{}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// F16ObjectSpace — read-only, decodes f16→f32 on access
// ---------------------------------------------------------------------------

pub struct F16ObjectSpace {
    pub dim: usize,
    pub distance_type: DistanceType,
    pub normalization: bool,
    slot_count: usize,
    live_count: usize,
    /// f16 data as u16, flat: data[id * dim + j].
    data: Vec<u16>,
    present: Vec<bool>,
}

impl F16ObjectSpace {
    pub fn load(path: &str, distance_type: DistanceType) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("F16ObjectSpace::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);

        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;

        // Read f16 data.
        let total = slot_count * dim;
        let mut data = vec![0u16; total];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total * 2)
        };
        r.read_exact(bytes).map_err(|e| format!("{}", e))?;

        // Read bitmap.
        let mut bitmap = vec![0u8; slot_count];
        r.read_exact(&mut bitmap).map_err(|e| format!("{}", e))?;
        let present: Vec<bool> = bitmap.iter().map(|&b| b != 0).collect();
        let live_count = present.iter().skip(1).filter(|&&p| p).count();

        Ok(F16ObjectSpace {
            dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
            slot_count,
            live_count,
            data,
            present,
        })
    }

    /// Decode f16 object into f32 buffer.
    #[inline]
    pub fn decode_into(&self, id: ObjectID, out: &mut [f32]) -> Result<(), NgtError> {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count || !self.present[idx] {
            return Err(format!("F16ObjectSpace: invalid id {}", id));
        }
        let start = idx * self.dim;
        for i in 0..self.dim {
            out[i] = f16_to_f32(self.data[start + i]);
        }
        Ok(())
    }

    pub fn is_present(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx > 0 && idx < self.slot_count && self.present[idx]
    }

    pub fn count(&self) -> usize { self.live_count }
    pub fn size(&self) -> usize { self.slot_count }
}

// ---------------------------------------------------------------------------
// F16Index — graph + tree + f16 objects
// ---------------------------------------------------------------------------

pub struct F16Index {
    pub graph: NeighborhoodGraph,
    pub tree: Option<DVPTree>,
    pub objects: F16ObjectSpace,
    pub property: IndexProperty,
}

impl F16Index {
    pub fn is_f16_index(dir: &str) -> bool {
        std::path::Path::new(&format!("{}/obj.f16", dir)).exists()
    }

    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let property = IndexProperty::import_from(&ps);
        let dt = property.to_distance_type();

        let objects = F16ObjectSpace::load(&format!("{}/obj.f16", dir), dt)?;

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

        Ok(F16Index { graph, tree, objects, property })
    }

    pub fn object_count(&self) -> usize { self.objects.count() }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        if options.k == 0 { return Ok(Vec::new()); }

        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.objects.normalization {
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

        Ok(self.search_f16(q, &mut seeds, options.k, options.epsilon, edge_size))
    }

    pub fn linear_search(&self, query: &[f32], k: usize) -> Result<Vec<ObjectDistance>, NgtError> {
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.objects.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let dt = self.objects.distance_type;
        let mut results = ResultSet::with_capacity(k + 1);
        let mut buf = vec![0.0f32; self.objects.dim];
        for id in 1..self.objects.slot_count {
            let oid = id as ObjectID;
            if !self.objects.is_present(oid) { continue; }
            self.objects.decode_into(oid, &mut buf)?;
            let d = primitive_comparator::compare(q, &buf, dt);
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
        for id in 1..self.objects.size() {
            let oid = id as ObjectID;
            if self.objects.is_present(oid) {
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
                    let d = primitive_comparator::compare(query, pivot, self.objects.distance_type);
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

    fn search_f16(
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

        let dim = self.objects.dim;
        let dt = self.objects.distance_type;
        let mut buf = vec![0.0f32; dim];

        // Compute seed distances.
        for seed in seeds.iter_mut() {
            let sid: u32 = seed.id;
            if self.objects.is_present(sid) {
                if self.objects.decode_into(sid, &mut buf).is_ok() {
                    seed.distance = primitive_comparator::compare(query, &buf, dt);
                } else { seed.distance = f32::MAX; }
            } else { seed.distance = f32::MAX; }
        }
        seeds.sort_unstable_by(|a, b| a.cmp(b));

        let mut results: BinaryHeap<ObjectDistance> = BinaryHeap::new();
        let mut unchecked: BinaryHeap<Reverse<ObjectDistance>> = BinaryHeap::new();
        let mut checked = BooleanVector::new(self.graph.nodes.len());
        let mut current_radius = f32::MAX;

        let padded_dim = ((dim.saturating_sub(1)) / 16 + 1) * 16;
        let prefetch_offset = (300.0 / (padded_dim as f32 + 30.0) + 1.0).floor() as usize;

        for s in seeds.iter() {
            let sd: f32 = s.distance;
            let si: u32 = s.id;
            if results.len() < k && sd <= current_radius { results.push(*s); }
            if sd < f32::MAX { checked.insert(si); unchecked.push(Reverse(*s)); }
        }
        if results.len() >= k { current_radius = results.peek().unwrap().distance; }
        let mut exploration_radius = exploration_coefficient * current_radius;

        while let Some(Reverse(target)) = unchecked.pop() {
            let td: f32 = target.distance;
            if td > exploration_radius { break; }
            let tid: u32 = target.id;
            let neighbors = match self.graph.get_node(tid) {
                Some(n) => n, None => continue,
            };
            if neighbors.is_empty() { continue; }
            let nsize = neighbors.len().min(edge_size);

            // Prefetch.
            let poft = prefetch_offset.min(nsize);
            for i in 0..poft {
                let nid: u32 = neighbors[i].id;
                if !checked.contains(nid) {
                    let idx = nid as usize;
                    if idx < self.objects.slot_count {
                        let ptr = unsafe { self.objects.data.as_ptr().add(idx * dim) };
                        crate::graph::prefetch_read(ptr);
                    }
                }
            }

            for ni in 0..nsize {
                if ni + prefetch_offset < nsize {
                    let ahead: u32 = neighbors[ni + prefetch_offset].id;
                    if !checked.contains(ahead) {
                        let idx = ahead as usize;
                        if idx < self.objects.slot_count {
                            let ptr = unsafe { self.objects.data.as_ptr().add(idx * dim) };
                            crate::graph::prefetch_read(ptr);
                        }
                    }
                }

                let nid: u32 = neighbors[ni].id;
                if checked.contains(nid) { continue; }
                checked.insert(nid);
                if !self.objects.is_present(nid) { continue; }
                if self.objects.decode_into(nid, &mut buf).is_err() { continue; }
                let distance = primitive_comparator::compare(query, &buf, dt);

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
