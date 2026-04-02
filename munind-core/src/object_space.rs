/// Port of NGT/ObjectSpace.h + NGT/ObjectRepository.h + NGT/ObjectSpaceRepository.h
///
/// Flat contiguous object storage for cache-friendly SIMD access.
/// Objects are stored in a single `Vec<f32>` with stride = dim.
/// A presence bitmap tracks which slots are live vs removed.

use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID, ResultSet};
use crate::primitive_comparator::{self, DistanceType};

// ---------------------------------------------------------------------------
// ObjectType  (NGT::ObjectSpace::ObjectType)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectType {
    None    = 0,
    Uint8   = 1,
    Float   = 2,
    Unset   = 127,
}

// ---------------------------------------------------------------------------
// ObjectSpace — flat contiguous storage
// ---------------------------------------------------------------------------

pub struct ObjectSpace {
    /// Number of float dimensions per object.
    pub dim: usize,
    /// Active distance type.
    pub distance_type: DistanceType,
    /// True when objects and queries must be unit-normalized.
    pub normalization: bool,

    /// Flat contiguous storage: `data[id * dim .. (id+1) * dim]`.
    /// Slot 0 is reserved (zeroed, never used).
    data: Vec<f32>,
    /// Presence bitmap: `present[id] = true` means slot `id` is live.
    present: Vec<bool>,
    /// Number of allocated slots (including slot 0).
    slot_count: usize,
    /// Number of live objects (excludes slot 0 and removed).
    live_count: usize,
}

impl ObjectSpace {
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        let mut os = ObjectSpace {
            dim,
            distance_type: DistanceType::None,
            normalization: false,
            data: vec![0.0f32; dim], // slot 0
            present: vec![false],    // slot 0 = not present
            slot_count: 1,
            live_count: 0,
        };
        os.set_distance_type(distance_type);
        os
    }

    pub fn set_distance_type(&mut self, t: DistanceType) {
        self.distance_type = t;
        self.normalization = primitive_comparator::requires_normalization(t);
    }

    // -----------------------------------------------------------------------
    // Dimension helpers
    // -----------------------------------------------------------------------

    #[inline]
    pub fn byte_size(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn padded_dim(&self) -> usize {
        ((self.dim.saturating_sub(1)) / 16 + 1) * 16
    }

    // -----------------------------------------------------------------------
    // Normalize
    // -----------------------------------------------------------------------

    pub fn normalize(v: &mut [f32]) -> Result<(), NgtError> {
        let sum: f32 = v.iter().map(|x| x * x).sum();
        if sum == 0.0 {
            return Err(
                "ObjectSpace::normalize: the object is an invalid zero vector".to_string(),
            );
        }
        let inv = 1.0 / sum.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
        Ok(())
    }

    fn allocate_normalized(&self, src: &[f32]) -> Result<Vec<f32>, NgtError> {
        if src.len() != self.dim {
            return Err(format!(
                "ObjectSpace::allocate_normalized: dimension mismatch: expected {} got {}",
                self.dim,
                src.len()
            ));
        }
        let mut obj: Vec<f32> = src.to_vec();
        if self.normalization {
            Self::normalize(&mut obj)?;
        }
        Ok(obj)
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    pub fn insert(&mut self, v: &[f32]) -> Result<ObjectID, NgtError> {
        let obj = self.allocate_normalized(v)?;
        let id = self.slot_count as ObjectID;
        // Append to flat storage.
        self.data.extend_from_slice(&obj);
        self.present.push(true);
        self.slot_count += 1;
        self.live_count += 1;
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Access — zero-copy slice into flat array
    // -----------------------------------------------------------------------

    /// Get a reference to the stored object. Returns a slice into the
    /// contiguous flat array — no pointer indirection.
    #[inline]
    pub fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count || !self.present[idx] {
            return Err(format!(
                "ObjectSpace::get_object: invalid or removed id {}",
                id
            ));
        }
        let start = idx * self.dim;
        Ok(&self.data[start..start + self.dim])
    }

    #[inline]
    pub fn is_present(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx > 0 && idx < self.slot_count && self.present[idx]
    }

    pub fn is_removed(&self, id: ObjectID) -> bool {
        let idx = id as usize;
        idx > 0 && idx < self.slot_count && !self.present[idx]
    }

    pub fn count(&self) -> usize {
        self.live_count
    }

    pub fn size(&self) -> usize {
        self.slot_count
    }

    // -----------------------------------------------------------------------
    // Remove
    // -----------------------------------------------------------------------

    pub fn remove(&mut self, id: ObjectID) -> Result<(), NgtError> {
        let idx = id as usize;
        if idx == 0 || idx >= self.slot_count || !self.present[idx] {
            return Err(format!("ObjectSpace::remove: invalid id {}", id));
        }
        self.present[idx] = false;
        self.live_count -= 1;
        // Zero the slot data (optional, for safety).
        let start = idx * self.dim;
        self.data[start..start + self.dim].fill(0.0);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Distance
    // -----------------------------------------------------------------------

    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        primitive_comparator::compare(a, b, self.distance_type)
    }

    // -----------------------------------------------------------------------
    // Linear search
    // -----------------------------------------------------------------------

    pub fn linear_search(
        &self,
        query: &[f32],
        radius: f64,
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.normalization {
            q_buf = query.to_vec();
            Self::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let mut results = ResultSet::with_capacity(k + 1);

        for idx in 1..self.slot_count {
            if !self.present[idx] {
                continue;
            }
            let start = idx * self.dim;
            let obj = &self.data[start..start + self.dim];
            let d = primitive_comparator::compare(q, obj, self.distance_type) as f64;
            if radius < 0.0 || d <= radius {
                let od = ObjectDistance::new(idx as ObjectID, d as f32);
                results.push(od);
                if results.len() > k {
                    results.pop();
                }
            }
        }

        let mut v = results.into_sorted_vec();
        v.truncate(k);
        Ok(v)
    }

    // -----------------------------------------------------------------------
    // Serialization — flat mmap-friendly format
    // -----------------------------------------------------------------------
    //
    // Layout (all little-endian):
    //   [8 bytes]  u64  slot_count
    //   [8 bytes]  u64  dim
    //   [slot_count * dim * 4 bytes]  f32 data (contiguous, slot 0 = zeroed)
    //   [slot_count bytes]  presence bitmap (0 = absent, 1 = present)
    //
    // This is the exact in-memory layout — write/read is a bulk memcpy.
    // MmapIndex can mmap this file directly (zero-copy).

    pub fn serialize(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("ObjectSpace::serialize: cannot create {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        self.write_to(&mut w)
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), NgtError> {
        // Header.
        w.write_all(&(self.slot_count as u64).to_le_bytes())
            .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;
        w.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;

        // Bulk write flat f32 data.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * 4,
            )
        };
        w.write_all(byte_slice)
            .map_err(|e| format!("ObjectSpace::write_to data: {}", e))?;

        // Presence bitmap.
        let bitmap: Vec<u8> = self.present.iter().map(|&p| if p { 1u8 } else { 0u8 }).collect();
        w.write_all(&bitmap)
            .map_err(|e| format!("ObjectSpace::write_to bitmap: {}", e))?;

        Ok(())
    }

    pub fn deserialize(&mut self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("ObjectSpace::deserialize: cannot open {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        self.read_from(&mut r)
    }

    pub fn read_from<R: Read>(&mut self, r: &mut R) -> Result<(), NgtError> {
        // Header.
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("ObjectSpace::read_from: header: {}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)
            .map_err(|e| format!("ObjectSpace::read_from: header: {}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;

        if dim != self.dim {
            return Err(format!(
                "ObjectSpace::read_from: dimension mismatch: file={} expected={}",
                dim, self.dim
            ));
        }

        // Bulk read flat f32 data.
        let total_floats = slot_count * dim;
        self.data = vec![0.0f32; total_floats];
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                total_floats * 4,
            )
        };
        r.read_exact(byte_slice)
            .map_err(|e| format!("ObjectSpace::read_from data: {}", e))?;

        // Presence bitmap.
        let mut bitmap = vec![0u8; slot_count];
        r.read_exact(&mut bitmap)
            .map_err(|e| format!("ObjectSpace::read_from bitmap: {}", e))?;

        self.present = bitmap.iter().map(|&b| b != 0).collect();
        self.slot_count = slot_count;
        self.live_count = self.present.iter().skip(1).filter(|&&p| p).count();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------

    pub fn iter_objects(&self) -> impl Iterator<Item = (ObjectID, &[f32])> {
        let dim = self.dim;
        (1..self.slot_count).filter_map(move |idx| {
            if self.present[idx] {
                let start = idx * dim;
                Some((idx as ObjectID, &self.data[start..start + dim]))
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive_comparator::DistanceType;

    #[test]
    fn test_insert_and_get() {
        let mut os = ObjectSpace::new(3, DistanceType::L2);
        let id = os.insert(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(id, 1);
        let stored = os.get_object(id).unwrap();
        assert_eq!(stored, &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn test_normalization_on_insert() {
        let mut os = ObjectSpace::new(2, DistanceType::CosineSimilarity);
        assert!(os.normalization);
        os.insert(&[3.0, 0.0]).unwrap();
        let stored = os.get_object(1).unwrap();
        assert!((stored[0] - 1.0).abs() < 1e-6);
        assert!(stored[1].abs() < 1e-6);
    }

    #[test]
    fn test_linear_search_l2() {
        let mut os = ObjectSpace::new(2, DistanceType::L2);
        os.insert(&[0.0, 0.0]).unwrap();
        os.insert(&[1.0, 0.0]).unwrap();
        os.insert(&[0.0, 1.0]).unwrap();

        let results = os.linear_search(&[0.9, 0.1], -1.0, 1).unwrap();
        let id = results[0].id;
        assert_eq!(id, 2);
    }

    #[test]
    fn test_remove_and_count() {
        let mut os = ObjectSpace::new(2, DistanceType::L2);
        os.insert(&[0.0, 0.0]).unwrap();
        os.insert(&[1.0, 0.0]).unwrap();
        assert_eq!(os.count(), 2);
        os.remove(1).unwrap();
        assert_eq!(os.count(), 1);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let dir = "./target/test_os_roundtrip";
        std::fs::create_dir_all(dir).unwrap();
        let path = format!("{}/obj", dir);

        let mut os = ObjectSpace::new(2, DistanceType::L2);
        os.insert(&[1.0, 2.0]).unwrap();
        os.insert(&[3.0, 4.0]).unwrap();
        os.serialize(&path).unwrap();

        let mut os2 = ObjectSpace::new(2, DistanceType::L2);
        os2.deserialize(&path).unwrap();
        assert_eq!(os2.count(), 2);
        assert_eq!(os2.get_object(1).unwrap(), &[1.0_f32, 2.0]);
        assert_eq!(os2.get_object(2).unwrap(), &[3.0_f32, 4.0]);
    }
}
