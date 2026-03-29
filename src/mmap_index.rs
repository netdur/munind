/// Read-only memory-mapped index.
///
/// Objects are memory-mapped from a flat binary file — zero-copy, instant open.
/// Graph and tree are loaded into RAM (small relative to objects).

use memmap2::Mmap;

use crate::common::{NgtError, ObjectDistance, ObjectID, PropertySet, ResultSet, SearchOptions};
use crate::graph::{GraphProperty, NeighborhoodGraph};
use crate::index::IndexProperty;
use crate::object_space::ObjectSpace;
use crate::primitive_comparator::{self, DistanceType};
use crate::tree::DVPTree;

// ---------------------------------------------------------------------------
// MmapObjectSpace — read-only object access backed by mmap
// ---------------------------------------------------------------------------

/// Zero-copy object storage: the flat f32 array is memory-mapped directly.
/// No deserialization, no heap allocation for objects.
struct MmapObjectSpace {
    /// Memory-mapped file (kept alive for the lifetime of the index).
    _mmap: Mmap,
    /// Pointer to the start of the f32 data (after the 16-byte header).
    data: *const f32,
    /// Presence bitmap (after the f32 data).
    present: *const u8,
    /// Number of slots (including slot 0).
    slot_count: usize,
    /// Dimensions per object.
    dim: usize,
    /// Distance type.
    distance_type: DistanceType,
    /// Normalization flag.
    normalization: bool,
}

// Safety: the mmap is read-only and the pointers are derived from it.
// The Mmap is owned by MmapObjectSpace, so the pointers remain valid.
unsafe impl Send for MmapObjectSpace {}
unsafe impl Sync for MmapObjectSpace {}

impl MmapObjectSpace {
    fn open(path: &str, distance_type: DistanceType) -> Result<Self, NgtError> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("MmapObjectSpace::open: {}: {}", path, e))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| format!("MmapObjectSpace::open: mmap failed: {}", e))?
        };

        let bytes = &mmap[..];
        if bytes.len() < 16 {
            return Err("MmapObjectSpace: file too small for header".to_string());
        }

        let slot_count =
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let dim =
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

        let data_bytes = slot_count * dim * 4;
        let bitmap_bytes = slot_count;
        let expected = 16 + data_bytes + bitmap_bytes;
        if bytes.len() < expected {
            return Err(format!(
                "MmapObjectSpace: file too small: {} < {}",
                bytes.len(),
                expected
            ));
        }

        let data_ptr = bytes[16..].as_ptr() as *const f32;
        let present_ptr = bytes[16 + data_bytes..].as_ptr();

        Ok(MmapObjectSpace {
            _mmap: mmap,
            data: data_ptr,
            present: present_ptr,
            slot_count,
            dim,
            distance_type,
            normalization: primitive_comparator::requires_normalization(distance_type),
        })
    }

    #[inline]
    fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count {
            return Err(format!("MmapObjectSpace: invalid id {}", id));
        }
        unsafe {
            if *self.present.add(idx) == 0 {
                return Err(format!("MmapObjectSpace: removed id {}", id));
            }
            let ptr = self.data.add(idx * self.dim);
            Ok(std::slice::from_raw_parts(ptr, self.dim))
        }
    }

    #[inline]
    fn is_present(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count {
            return false;
        }
        unsafe { *self.present.add(idx) != 0 }
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        primitive_comparator::compare(a, b, self.distance_type)
    }

    fn count(&self) -> usize {
        let mut c = 0;
        for i in 1..self.slot_count {
            if unsafe { *self.present.add(i) != 0 } {
                c += 1;
            }
        }
        c
    }

    fn size(&self) -> usize {
        self.slot_count
    }
}

// ---------------------------------------------------------------------------
// MmapIndex
// ---------------------------------------------------------------------------

pub struct MmapIndex {
    objects: MmapObjectSpace,
    graph: NeighborhoodGraph,
    tree: Option<DVPTree>,
    property: IndexProperty,
}

impl MmapIndex {
    pub fn open(dir: &str) -> Result<Self, NgtError> {
        let mut ps = PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let property = IndexProperty::import_from(&ps);

        let dt = property.to_distance_type();

        // Mmap the obj file directly — same flat format as ObjectSpace.
        let obj_path = format!("{}/obj", dir);
        let objects = MmapObjectSpace::open(&obj_path, dt)?;

        let gp = GraphProperty {
            truncation_threshold: property.truncation_threshold,
            edge_size_for_creation: property.edge_size_for_creation,
            edge_size_for_search: property.edge_size_for_search,
            insertion_radius_coefficient: property.insertion_radius_coefficient,
            seed_size: property.seed_size,
            graph_type: property.graph_type,
            batch_size_for_creation: property.batch_size_for_creation,
            outgoing_edge: property.outgoing_edge,
            incoming_edge: property.incoming_edge,
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

        Ok(MmapIndex {
            objects,
            graph,
            tree,
            property,
        })
    }

    pub fn object_count(&self) -> usize {
        self.objects.count()
    }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        if options.k == 0 {
            return Ok(Vec::new());
        }

        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.objects.normalization {
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

        let mut results = self.graph.search_with_mmap(
            q,
            &mut seeds,
            options.k,
            options.epsilon,
            edge_size,
            f32::MAX,
            &self.objects,
        );
        results.truncate(options.k);
        Ok(results)
    }

    pub fn linear_search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.objects.normalization {
            q_buf = query.to_vec();
            ObjectSpace::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let mut results = ResultSet::with_capacity(k + 1);
        for idx in 1..self.objects.slot_count {
            if !self.objects.is_present(idx as ObjectID) {
                continue;
            }
            let obj = self.objects.get_object(idx as ObjectID)?;
            let d = self.objects.distance(q, obj);
            let od = ObjectDistance::new(idx as ObjectID, d);
            results.push(od);
            if results.len() > k {
                results.pop();
            }
        }

        let mut v = results.into_sorted_vec();
        v.truncate(k);
        Ok(v)
    }

    fn get_seeds(&self, query: &[f32]) -> Result<Vec<ObjectDistance>, NgtError> {
        // Use a temporary ObjectSpace shim for tree search.
        // The tree needs ObjectSpace::distance and ObjectSpace::get_object.
        // We wrap MmapObjectSpace calls.
        if let Some(tree) = &self.tree {
            if let Ok(leaf_nid) = self.search_tree_leaf(tree, query) {
                let seeds = tree.get_object_ids_from_leaf(leaf_nid);
                if !seeds.is_empty() {
                    return Ok(seeds);
                }
            }
        }

        let mut seeds = Vec::new();
        for id in 1..self.objects.size() {
            let oid = id as ObjectID;
            if self.objects.is_present(oid) {
                seeds.push(ObjectDistance::new(oid, 0.0));
                if seeds.len() >= self.property.seed_size.max(1) {
                    break;
                }
            }
        }
        Ok(seeds)
    }

    /// Manually traverse tree to find leaf (avoids needing &ObjectSpace).
    fn search_tree_leaf(
        &self,
        tree: &DVPTree,
        query: &[f32],
    ) -> Result<crate::node::NodeId, NgtError> {
        use crate::node::NodeType;

        let root_id = if tree.internal_nodes.len() > 1
            && tree.internal_nodes[1].is_some()
        {
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
                    let node = tree
                        .get_internal(current.get_id())
                        .ok_or("internal node not found")?;
                    let pivot = node.pivot.as_ref().unwrap();
                    let d = self.objects.distance(query, pivot);

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
}

// ---------------------------------------------------------------------------
// MmapObjectAccessor — trait to let NeighborhoodGraph::search work with
// both ObjectSpace and MmapObjectSpace without generics overhead.
// ---------------------------------------------------------------------------

/// Trait for read-only object access during graph search.
pub trait ObjectAccessor {
    fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError>;
    fn is_present(&self, id: ObjectID) -> bool;
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn dim(&self) -> usize;
    fn size(&self) -> usize;
}

impl ObjectAccessor for ObjectSpace {
    #[inline]
    fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        self.get_object(id)
    }
    #[inline]
    fn is_present(&self, id: ObjectID) -> bool {
        self.is_present(id)
    }
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance(a, b)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn size(&self) -> usize {
        self.size()
    }
}

impl ObjectAccessor for MmapObjectSpace {
    #[inline]
    fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        self.get_object(id)
    }
    #[inline]
    fn is_present(&self, id: ObjectID) -> bool {
        self.is_present(id)
    }
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance(a, b)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn size(&self) -> usize {
        self.size()
    }
}
