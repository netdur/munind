/// Port of NGT/ObjectSpace.h + NGT/ObjectRepository.h + NGT/ObjectSpaceRepository.h
///
/// Phase 1: float (f32) objects only.  The DistanceType enum and distance
/// functions live in `primitive_comparator`; this module wires them to a
/// `Repository<Vec<f32>>` and provides insert / search / serialize logic.

use std::io::{Read, Write};

use crate::common::{NgtError, ObjectDistance, ObjectID, Repository, ResultSet};
use crate::primitive_comparator::{self, DistanceType};

// ---------------------------------------------------------------------------
// ObjectType  (NGT::ObjectSpace::ObjectType)
// ---------------------------------------------------------------------------

/// The element type of stored objects.
/// Phase 1 only supports `Float` (= 2, matching C++ `ObjectType::Float`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectType {
    None    = 0,
    Uint8   = 1,
    Float   = 2,
    Unset   = 127,
}

// ---------------------------------------------------------------------------
// ObjectSpace
// ---------------------------------------------------------------------------

/// Combines C++ `ObjectSpace` + `ObjectRepository` + `ObjectSpaceRepository<float,…>`
/// into a single struct.
///
/// Responsibilities:
/// - Own the `Repository<Vec<f32>>` (1-based sparse, slot-0 = null sentinel).
/// - Apply normalization on insert when the distance type requires it.
/// - Dispatch distance computations to `primitive_comparator`.
/// - Implement linear-search brute-force.
/// - Binary serialize / deserialize matching the NGT on-disk format.
pub struct ObjectSpace {
    /// Number of float dimensions per object.
    pub dim: usize,
    /// Active distance type.
    pub distance_type: DistanceType,
    /// True when objects and queries must be unit-normalized (set by
    /// `set_distance_type` for Cosine / NormalizedCosine / NormalizedAngle /
    /// NormalizedL2).
    pub normalization: bool,
    /// 1-based sparse storage.  `objects.data[0]` is always `None`.
    pub objects: Repository<Vec<f32>>,
}

impl ObjectSpace {
    /// Create a new, empty ObjectSpace.
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        let mut os = ObjectSpace {
            dim,
            distance_type: DistanceType::None,
            normalization: false,
            objects: Repository::new(),
        };
        os.set_distance_type(distance_type);
        os
    }

    /// Set (or change) the distance type, updating the normalization flag.
    /// Maps to `ObjectSpaceRepository::setDistanceType`.
    pub fn set_distance_type(&mut self, t: DistanceType) {
        self.distance_type = t;
        self.normalization = primitive_comparator::requires_normalization(t);
    }

    // -----------------------------------------------------------------------
    // Dimension helpers
    // -----------------------------------------------------------------------

    /// Byte size of a serialized object = `dim * sizeof(float)`.
    /// Maps to `ObjectRepository::getByteSize()` / `getByteSizeOfObject()`.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }

    /// Padded dimension: rounded up to the next multiple of 16.
    /// Maps to `ObjectSpace::getPaddedDimension()`.
    #[inline]
    pub fn padded_dim(&self) -> usize {
        ((self.dim.saturating_sub(1)) / 16 + 1) * 16
    }

    // -----------------------------------------------------------------------
    // Object allocation helpers
    // -----------------------------------------------------------------------

    /// Normalize `v` in-place.  Maps to `ObjectSpace::normalize`.
    ///
    /// If the vector is the zero vector, returns an error (matching C++ behaviour
    /// unless `NGT_DISABLE_NORMALIZATION_ERROR_CHECK` is defined).
    pub fn normalize(v: &mut [f32]) -> Result<(), NgtError> {
        let sum: f32 = v.iter().map(|x| x * x).sum();
        if sum == 0.0 {
            // Zero-vector check: if all elements really are 0 return error.
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

    /// Allocate a (possibly normalized) copy of `src` matching `dim`.
    /// Maps to `ObjectSpaceRepository::allocateNormalizedObject(const vector<float>&)`.
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

    /// Insert a float vector, normalizing if required.
    /// Returns the 1-based object ID.
    pub fn insert(&mut self, v: &[f32]) -> Result<ObjectID, NgtError> {
        let obj = self.allocate_normalized(v)?;
        let id = self.objects.insert(obj);
        Ok(id as ObjectID)
    }

    // -----------------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------------

    /// Get a reference to the stored object with the given 1-based ID.
    pub fn get_object(&self, id: ObjectID) -> Result<&[f32], NgtError> {
        if id == 0 {
            return Err("ObjectSpace::get_object: id 0 is reserved".to_string());
        }
        self.objects
            .get(id as usize)
            .map(Vec::as_slice)
    }

    /// True when the slot for `id` is occupied.
    pub fn is_present(&self, id: ObjectID) -> bool {
        !self.objects.is_empty_slot(id as usize)
    }

    /// True when `id` is in range [1, size) and the slot is null.
    pub fn is_removed(&self, id: ObjectID) -> bool {
        id as usize > 0
            && (id as usize) < self.objects.size()
            && self.objects.is_empty_slot(id as usize)
    }

    /// Number of live objects (excluding slot 0 and removed slots).
    pub fn count(&self) -> usize {
        self.objects.count()
    }

    /// Total allocated slots (including slot 0 and removed slots).
    pub fn size(&self) -> usize {
        self.objects.size()
    }

    // -----------------------------------------------------------------------
    // Remove
    // -----------------------------------------------------------------------

    /// Remove the object with the given 1-based ID.
    pub fn remove(&mut self, id: ObjectID) -> Result<(), NgtError> {
        if id == 0 {
            return Err("ObjectSpace::remove: id 0 is reserved".to_string());
        }
        self.objects.remove(id as usize).map(|_| ())
    }

    // -----------------------------------------------------------------------
    // Distance
    // -----------------------------------------------------------------------

    /// Compute the distance between two float slices using the current
    /// distance type.  Maps to the Comparator functors in C++.
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        primitive_comparator::compare(a, b, self.distance_type)
    }

    // -----------------------------------------------------------------------
    // Linear search  (ObjectSpaceRepository::linearSearch)
    // -----------------------------------------------------------------------

    /// Brute-force linear search over all live objects.
    ///
    /// `radius < 0` means unlimited (all objects within `k`-nearest).
    /// Results are returned sorted by ascending distance, truncated to `k`.
    ///
    /// Maps to `ObjectSpaceRepository::linearSearch(Object&, double, size_t,
    /// ResultSet&)`.
    pub fn linear_search(
        &self,
        query: &[f32],
        radius: f64,
        k: usize,
    ) -> Result<Vec<ObjectDistance>, NgtError> {
        // Normalise query copy if required (allocateNormalizedObject for the query).
        let mut q_buf: Vec<f32>;
        let q: &[f32] = if self.normalization {
            q_buf = query.to_vec();
            Self::normalize(&mut q_buf)?;
            &q_buf
        } else {
            query
        };

        let mut results = ResultSet::with_capacity(k + 1);
        let rep = &self.objects;

        for idx in 1..rep.size() {
            let obj = match rep.get_unchecked(idx) {
                Some(o) => o.as_slice(),
                None => continue,
            };
            let d = primitive_comparator::compare(q, obj, self.distance_type) as f64;
            if radius < 0.0 || d <= radius {
                let od = ObjectDistance::new(idx as ObjectID, d as f32);
                results.push(od);
                if results.len() > k {
                    results.pop(); // eject farthest
                }
            }
        }

        // Drain heap → ascending-distance vec (best first).
        let mut v = results.into_sorted_vec();
        v.truncate(k);
        Ok(v)
    }

    // -----------------------------------------------------------------------
    // Serialization — NGT binary format
    // -----------------------------------------------------------------------
    //
    // Format (little-endian on all supported platforms):
    //   [8 bytes]  u64  total slot count  (= objects.size())
    //   For each slot 0 .. count-1:
    //     [1 byte]  '-'  (slot is null / sentinel)
    //     OR
    //     [1 byte]  '+'  followed by  [dim * 4 bytes]  f32 array (LE)
    //
    // This matches:
    //   NGT::Repository<Object>::serialize  →  Serializer::write(size_t),
    //                                          '-' / '+',
    //                                          Object::serialize(os, ospace)
    //   BaseObject::serialize              →  Serializer::write(uint8*, byteSize)
    //   where byteSize = dim * sizeof(float)

    /// Serialize to a binary file.
    pub fn serialize(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("ObjectSpace::serialize: cannot create {}: {}", path, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        self.write_to(&mut w)
    }

    /// Write the binary representation to any `Write` sink.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), NgtError> {
        let slot_count = self.objects.size() as u64;
        w.write_all(&slot_count.to_le_bytes())
            .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;

        for idx in 0..self.objects.size() {
            match self.objects.data[idx].as_ref() {
                None => {
                    w.write_all(&[b'-'])
                        .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;
                }
                Some(obj) => {
                    w.write_all(&[b'+'])
                        .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;
                    // Write dim * 4 bytes (raw f32 LE, matching byteSize not paddedByteSize)
                    for &f in obj.iter().take(self.dim) {
                        w.write_all(&f.to_le_bytes())
                            .map_err(|e| format!("ObjectSpace::write_to: {}", e))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Deserialize from a binary file.
    pub fn deserialize(&mut self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("ObjectSpace::deserialize: cannot open {}: {}", path, e))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        self.read_from(&mut r)
    }

    /// Read the binary representation from any `Read` source.
    pub fn read_from<R: Read>(&mut self, r: &mut R) -> Result<(), NgtError> {
        // Read slot count (size_t = 8 bytes on 64-bit).
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("ObjectSpace::read_from: reading count: {}", e))?;
        let slot_count = u64::from_le_bytes(buf8) as usize;

        self.objects.delete_all(); // clears data, restores slot 0

        for i in 0..slot_count {
            let mut type_byte = [0u8; 1];
            r.read_exact(&mut type_byte)
                .map_err(|e| format!("ObjectSpace::read_from: reading slot {} type: {}", i, e))?;

            match type_byte[0] {
                b'-' => {
                    // Null slot: push None.  For slot 0 it's already there via
                    // delete_all(); for subsequent slots grow the data vec.
                    if i == 0 {
                        // slot 0 already exists as None after delete_all()
                    } else {
                        self.objects.data.push(None);
                    }
                }
                b'+' => {
                    // Read dim * 4 bytes of f32 data.
                    let byte_size = self.byte_size();
                    let mut raw = vec![0u8; byte_size];
                    r.read_exact(&mut raw).map_err(|e| {
                        format!("ObjectSpace::read_from: reading slot {} data: {}", i, e)
                    })?;

                    let mut obj = Vec::with_capacity(self.dim);
                    for chunk in raw.chunks_exact(4) {
                        obj.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }

                    if i == 0 {
                        // Slot 0 should be null sentinel; if the file has '+' at 0,
                        // replace the None we just created.
                        self.objects.data[0] = Some(obj);
                    } else {
                        self.objects.data.push(Some(obj));
                    }
                }
                other => {
                    return Err(format!(
                        "ObjectSpace::read_from: unexpected slot type byte {:?} at slot {}",
                        other as char, i
                    ));
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // mmap-compatible raw access helpers (used by MmapIndex in Step 8)
    // -----------------------------------------------------------------------

    /// Return all live objects as `(id, slice)` pairs.
    pub fn iter_objects(&self) -> impl Iterator<Item = (ObjectID, &[f32])> {
        self.objects
            .data
            .iter()
            .enumerate()
            .skip(1) // skip slot 0
            .filter_map(|(idx, opt)| opt.as_ref().map(|v| (idx as ObjectID, v.as_slice())))
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
        assert!((stored[0] - 1.0).abs() < 1e-6, "should be normalized to unit vec");
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
