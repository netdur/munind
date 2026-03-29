/// Port of NGT/Common.h — foundational types, containers, and search state.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Primitive type aliases (NGT::ObjectID, NGT::Distance)
// ---------------------------------------------------------------------------

/// 1-based object identifier.  0 is the sentinel "invalid" value.
/// Maps to `unsigned int` in C++.
pub type ObjectID = u32;

/// Distance between two objects.
/// Maps to `float` in C++.
pub type Distance = f32;

/// Error type.  The test suite calls `.unwrap_err()` and then
/// `err.contains("…")`, which requires `String`.
pub type NgtError = String;

// ---------------------------------------------------------------------------
// ObjectDistance  (NGT::ObjectDistance, #pragma pack(2))
// ---------------------------------------------------------------------------

/// An object paired with its distance from the query.
///
/// The C++ struct is declared `#pragma pack(2)` so its alignment is 2 bytes
/// (both members are 4 bytes, so the total is still 8 bytes; only the struct's
/// *own* alignment changes).  We mirror that with `#[repr(C, packed(2))]` so
/// that direct binary round-trips have the same layout.
///
/// Ordering follows the C++ `operator<` / `operator>`:
///   compare by distance first, then by id as tie-breaker.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, packed(2))]
pub struct ObjectDistance {
    pub id: u32,
    pub distance: f32,
}

impl ObjectDistance {
    #[inline]
    pub fn new(id: ObjectID, distance: Distance) -> Self {
        Self { id, distance }
    }
}

impl Eq for ObjectDistance {}

/// Natural ordering: smaller distance is "less" (better result).
/// Tie-break by id (smaller id is "less").
/// This mirrors the C++ `operator<`.
impl PartialOrd for ObjectDistance {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ObjectDistance {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Copy fields to locals first — required because the struct is packed(2)
        // and Rust forbids unaligned references to fields.
        let (d_self, d_other) = (self.distance, other.distance);
        let (id_self, id_other) = (self.id, other.id);
        match d_self.total_cmp(&d_other) {
            Ordering::Equal => id_self.cmp(&id_other),
            ord => ord,
        }
    }
}

/// Convenience alias used as the public search-result type.
pub type SearchResult = ObjectDistance;

// ---------------------------------------------------------------------------
// ResultSet  (NGT::ResultSet / NGT::ResultPriorityQueue)
// ---------------------------------------------------------------------------

/// A **max**-heap of `ObjectDistance`.
///
/// The element with the **largest** distance is at the top so that, when the
/// heap holds exactly `k` results, the worst (farthest) result can be ejected
/// in O(log k) to make room for a better candidate.
///
/// Maps to `NGT::ResultSet` which wraps `std::vector<ObjectDistance>` with
/// `std::push_heap` / `std::pop_heap` (default C++ comparator = max-heap via
/// `operator<`).
pub struct ResultSet {
    heap: BinaryHeap<ObjectDistance>,
}

impl ResultSet {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(cap),
        }
    }

    /// The worst (largest-distance) element currently in the set.
    #[inline]
    pub fn top(&self) -> Option<&ObjectDistance> {
        self.heap.peek()
    }

    /// Add an element (O(log n)).
    #[inline]
    pub fn push(&mut self, o: ObjectDistance) {
        self.heap.push(o);
    }

    /// Remove and return the worst element (O(log n)).
    #[inline]
    pub fn pop(&mut self) -> Option<ObjectDistance> {
        self.heap.pop()
    }

    /// Add an element then immediately remove the worst one.
    /// This keeps the set bounded without a separate size check.
    /// Maps to `NGT::ResultSet::push_pop`.
    #[inline]
    pub fn push_pop(&mut self, o: ObjectDistance) {
        self.heap.push(o);
        self.heap.pop();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain the heap into a `Vec` sorted by ascending distance (best first).
    pub fn into_sorted_vec(self) -> Vec<ObjectDistance> {
        self.heap.into_sorted_vec()
    }
}

impl Default for ResultSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BooleanSet  (NGT::BooleanSet)
// ---------------------------------------------------------------------------

/// Bit-vector visited set.
///
/// Size is rounded up to a multiple of 4 × 64 bits so that the inner loop
/// can be unrolled.  Maps directly to `NGT::BooleanSet`.
pub struct BooleanSet {
    bitvec: Vec<u64>,
}

impl BooleanSet {
    pub fn new(capacity: usize) -> Self {
        // NGT formula: size = ((s>>6)+1), then pad to next multiple of 4, +4
        let size = (capacity >> 6) + 1;
        let size = ((size >> 2) << 2) + 4;
        Self {
            bitvec: vec![0u64; size],
        }
    }

    #[inline]
    fn entry(&mut self, i: usize) -> &mut u64 {
        &mut self.bitvec[i >> 6]
    }

    #[inline]
    fn bit(i: usize) -> u64 {
        1u64 << (i & 63)
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        let bit = Self::bit(i);
        *self.entry(i) |= bit;
    }

    /// Alias used in NGT source.
    #[inline]
    pub fn insert(&mut self, i: usize) {
        self.set(i);
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.bitvec[i >> 6] & Self::bit(i)) != 0
    }

    #[inline]
    pub fn reset_bit(&mut self, i: usize) {
        let bit = Self::bit(i);
        *self.entry(i) &= !bit;
    }
}

impl std::ops::Index<usize> for BooleanSet {
    type Output = bool;
    fn index(&self, i: usize) -> &bool {
        // Return a reference to a static bool.  Used only in boolean contexts.
        if self.get(i) {
            &true
        } else {
            &false
        }
    }
}

// ---------------------------------------------------------------------------
// BooleanVectorByEpoch  (NGT::BooleanVectorByEpoch<uint8_t>)
// ---------------------------------------------------------------------------

/// Epoch-based visited tracker.
///
/// Instead of clearing the array after every search, an epoch counter is
/// incremented.  An element is considered "visited this epoch" if its stored
/// value equals the current epoch.  When the epoch wraps (255 → 0 → 1) the
/// array *is* cleared, but that only happens every 255 searches.
///
/// The array size is always a power of two; access uses a bitmask.
/// Maps to `NGT::BooleanVectorByEpoch<uint8_t>`.
pub struct BooleanVectorByEpoch {
    array: Vec<u8>,
    epoch: u8,
    mask: usize,
}

impl BooleanVectorByEpoch {
    /// `capacity` is a hint for the expected max object ID.
    /// The internal array is sized to the next power of two ≥ `capacity`.
    pub fn new(capacity: usize) -> Self {
        // C++: size = 1u << (32 - __builtin_clz(s - 1))
        // i.e., next power of two ≥ capacity.
        let size = capacity.next_power_of_two();
        Self {
            array: vec![0u8; size],
            epoch: 1,
            mask: size - 1,
        }
    }

    /// Advance the epoch (equivalent to clearing the visited set).
    /// When epoch reaches 255, clear the backing array and restart.
    /// Maps to `NGT::BooleanVectorByEpoch::reset`.
    #[inline]
    pub fn reset(&mut self) {
        const MAX: u8 = u8::MAX; // sizeof(uint8_t) * 0x100 - 1 = 255
        if self.epoch == MAX {
            self.array.fill(0);
            self.epoch = 0;
        }
        self.epoch += 1;
    }

    /// Mark `id` as visited.  Returns `true` if it was already visited
    /// in the current epoch (i.e., a duplicate).
    /// Maps to `NGT::BooleanVectorByEpoch::visit`.
    #[inline]
    pub fn visit(&mut self, id: u32) -> bool {
        let idx = id as usize & self.mask;
        let already = self.array[idx] == self.epoch;
        self.array[idx] = self.epoch;
        already
    }
}

// ---------------------------------------------------------------------------
// CompactVector<T>  (NGT::CompactVector<TYPE>)
// ---------------------------------------------------------------------------

/// A compact growable vector whose length and capacity are stored as `u16`.
///
/// The maximum number of elements is `u16::MAX` (65 535).  Exceeding this
/// limit causes a panic, exactly as the C++ code does (`abort()`).
///
/// This is used for graph-node edge lists, where the 16-bit limit is a
/// deliberate design choice to reduce memory per node.
///
/// Maps to `NGT::CompactVector<TYPE>`.
#[derive(Clone, Debug, Default)]
pub struct CompactVector<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> CompactVector<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        assert!(
            cap <= u16::MAX as usize,
            "CompactVector: requested capacity {} exceeds u16::MAX",
            cap
        );
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    /// Number of elements (always fits in `u16`).
    #[inline]
    pub fn size(&self) -> u16 {
        self.data.len() as u16
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn push_back(&mut self, val: T) {
        if self.data.len() >= u16::MAX as usize {
            panic!(
                "CompactVector is too big (> {})",
                u16::MAX
            );
        }
        self.data.push(val);
    }

    #[inline]
    pub fn pop_back(&mut self) {
        self.data.pop();
    }

    pub fn reserve(&mut self, s: usize) {
        assert!(
            s <= u16::MAX as usize,
            "CompactVector: reserve {} exceeds u16::MAX",
            s
        );
        self.data.reserve(s.saturating_sub(self.data.len()));
    }

    pub fn resize(&mut self, s: usize, val: T) {
        assert!(
            s <= u16::MAX as usize,
            "CompactVector: resize to {} exceeds u16::MAX",
            s
        );
        self.data.resize(s, val);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn erase(&mut self, idx: usize) {
        self.data.remove(idx);
    }

    pub fn insert(&mut self, idx: usize, val: T) {
        if self.data.len() >= u16::MAX as usize {
            panic!("CompactVector is too big");
        }
        self.data.insert(idx, val);
    }

    #[inline]
    pub fn front(&self) -> Option<&T> {
        self.data.first()
    }

    #[inline]
    pub fn back(&self) -> Option<&T> {
        self.data.last()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: Clone + Default> std::ops::Index<usize> for CompactVector<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

impl<T: Clone + Default> std::ops::IndexMut<usize> for CompactVector<T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }
}

impl<'a, T: Clone + Default> IntoIterator for &'a CompactVector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// ---------------------------------------------------------------------------
// Repository<T>  (NGT::Repository<TYPE>)
// ---------------------------------------------------------------------------

/// Sparse, 1-based pointer repository with slot reuse.
///
/// Slot 0 is always `None`.  IDs start at 1.  When an entry is removed its
/// slot index is pushed onto a min-heap (`removedList`) so that the smallest
/// freed slot is reused first — exactly as the C++ priority_queue with
/// `std::greater<size_t>` behaves.
///
/// Maps to `NGT::Repository<TYPE>` (non-shared-memory variant).
pub struct Repository<T> {
    /// `data[0]` is always `None`; real objects start at `data[1]`.
    pub data: Vec<Option<T>>,
    /// Min-heap of freed slot indices (`Reverse` turns BinaryHeap into min-heap).
    removed_list: BinaryHeap<Reverse<usize>>,
}

impl<T> Repository<T> {
    pub fn new() -> Self {
        let mut data = Vec::new();
        data.push(None); // slot 0 is always the null sentinel
        Self {
            data,
            removed_list: BinaryHeap::new(),
        }
    }

    /// Number of live entries (excludes slot 0 and removed slots).
    pub fn count(&self) -> usize {
        // C++: size() == 0 ? 0 : size() - removedList.size() - 1
        if self.data.is_empty() {
            0
        } else {
            self.data.len().saturating_sub(1 + self.removed_list.len())
        }
    }

    /// Total allocated slots (including slot 0 and removed slots).
    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty_slot(&self, idx: usize) -> bool {
        idx >= self.data.len() || self.data[idx].is_none()
    }

    /// Insert `val`, reusing a freed slot if available.
    /// Returns the assigned ID.
    pub fn insert(&mut self, val: T) -> usize {
        if let Some(Reverse(idx)) = self.removed_list.pop() {
            // Reuse the smallest freed slot.
            self.data[idx] = Some(val);
            idx
        } else {
            self.push(val)
        }
    }

    /// Always appends to the end.  Returns the new ID.
    fn push(&mut self, val: T) -> usize {
        if self.data.is_empty() {
            self.data.push(None); // ensure slot 0 exists
        }
        self.data.push(Some(val));
        self.data.len() - 1
    }

    /// Insert at a specific index (must be empty).
    pub fn insert_at(&mut self, idx: usize, val: T) -> Result<usize, NgtError> {
        // Grow if needed.
        if self.data.len() <= idx {
            self.data.resize_with(idx + 1, || None);
        }
        if self.data[idx].is_some() {
            return Err(format!("Repository::insert_at: slot {} is not empty", idx));
        }
        // Remove from freed list if it was there.
        self.removed_list_remove(idx);
        self.data[idx] = Some(val);
        Ok(idx)
    }

    pub fn get(&self, idx: usize) -> Result<&T, NgtError> {
        self.data
            .get(idx)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| {
                format!(
                    "Repository::get: not in-memory or invalid offset. idx={} size={}",
                    idx,
                    self.data.len()
                )
            })
    }

    pub fn get_mut(&mut self, idx: usize) -> Result<&mut T, NgtError> {
        let len = self.data.len();
        self.data
            .get_mut(idx)
            .and_then(|opt| opt.as_mut())
            .ok_or_else(|| {
                format!(
                    "Repository::get_mut: not in-memory or invalid offset. idx={} size={}",
                    idx, len
                )
            })
    }

    pub fn get_unchecked(&self, idx: usize) -> Option<&T> {
        self.data.get(idx).and_then(|o| o.as_ref())
    }

    /// Remove the entry at `idx`, returning it, and push `idx` onto the freed list.
    pub fn remove(&mut self, idx: usize) -> Result<T, NgtError> {
        if self.is_empty_slot(idx) {
            return Err(format!(
                "Repository::remove: slot {} is empty or out of range",
                idx
            ));
        }
        let val = self.data[idx].take().unwrap();
        self.removed_list.push(Reverse(idx));
        Ok(val)
    }

    /// Delete all entries and clear the freed list.
    pub fn delete_all(&mut self) {
        self.data.clear();
        self.data.push(None); // restore slot 0
        while self.removed_list.pop().is_some() {}
    }

    fn removed_list_remove(&mut self, target: usize) {
        // Rebuild the heap without `target`.  O(n) but called rarely.
        let items: Vec<_> = std::mem::take(&mut self.removed_list).into_iter().collect();
        self.removed_list = items
            .into_iter()
            .filter(|&Reverse(id)| id != target)
            .collect();
    }
}

impl<T> Default for Repository<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PropertySet  (NGT::PropertySet)
// ---------------------------------------------------------------------------

/// String key-value store used to persist index properties.
/// Maps to `NGT::PropertySet` (inherits `std::map<std::string, std::string>`).
#[derive(Clone, Debug, Default)]
pub struct PropertySet {
    map: std::collections::BTreeMap<String, String>,
}

impl PropertySet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_str(&mut self, key: impl Into<String>, value: impl ToString) {
        self.map.insert(key.into(), value.to_string());
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.map
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    pub fn get_i64(&self, key: &str, default: i64) -> i64 {
        self.map
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Save to a tab-separated text stream (key TAB value NEWLINE).
    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)
            .map_err(|e| format!("PropertySet::save: cannot open {}: {}", path, e))?;
        for (k, v) in &self.map {
            writeln!(f, "{}\t{}", k, v)
                .map_err(|e| format!("PropertySet::save: write error: {}", e))?;
        }
        Ok(())
    }

    /// Load from a tab-separated text file.
    pub fn load(&mut self, path: &str) -> Result<(), NgtError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("PropertySet::load: cannot open {}: {}", path, e))?;
        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() == 2 {
                self.map.insert(parts[0].to_string(), parts[1].to_string());
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SearchContainer  (NGT::SearchContainer)
// ---------------------------------------------------------------------------

/// Public search options (user-facing, matches the test imports).
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// Number of nearest neighbours to return.
    pub k: usize,
    /// Exploration coefficient offset.  The internal
    /// `explorationCoefficient = epsilon + 1.0`.
    /// `0.0` → exact graph traversal; `0.1` → 10 % wider search.
    pub epsilon: f32,
    /// Override for the number of edges to follow per node.
    /// `None` → use the index's `edge_size_for_search` property.
    /// `Some(0)` → use all edges.
    pub edge_size: Option<usize>,
}

/// Internal search state, created from `SearchOptions` per-call.
/// Maps to `NGT::SearchContainer`.
pub struct SearchContainer {
    // ---- parameters set before search ----
    /// k (result count).
    pub size: usize,
    /// Search radius.  `f32::MAX` means unlimited.
    pub radius: Distance,
    /// `epsilon + 1.0`.  Default 1.1 (epsilon = 0.1).
    pub exploration_coefficient: f32,
    /// Edge-size override.  -1 = use index property, 0 = all edges.
    pub edge_size: i32,
    /// Whether to return all objects in the seed leaf regardless of `k`.
    pub use_all_nodes_in_leaf: bool,
    /// Target accuracy hint (-1.0 = disabled).
    pub expected_accuracy: f32,
    /// True when this search is driving an insertion (not a user query).
    pub insertion: bool,

    // ---- state updated during search ----
    pub working_result: ResultSet,
    pub distance_computation_count: usize,
    pub visit_count: usize,
}

impl SearchContainer {
    /// Build from user-facing `SearchOptions`.
    pub fn from_options(opts: &SearchOptions) -> Self {
        Self {
            size: opts.k,
            radius: f32::MAX,
            exploration_coefficient: opts.epsilon + 1.0,
            edge_size: opts.edge_size.map(|e| e as i32).unwrap_or(-1),
            use_all_nodes_in_leaf: false,
            expected_accuracy: -1.0,
            insertion: false,
            working_result: ResultSet::with_capacity(opts.k + 1),
            distance_computation_count: 0,
            visit_count: 0,
        }
    }

    /// Equivalent to `NGT::SearchContainer::getEpsilon()`.
    #[inline]
    pub fn get_epsilon(&self) -> f32 {
        self.exploration_coefficient - 1.0
    }

    /// Equivalent to `NGT::SearchContainer::setEpsilon(e)`.
    #[inline]
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.exploration_coefficient = epsilon + 1.0;
    }
}

// ---------------------------------------------------------------------------
// Serializer helpers  (NGT::Serializer namespace)
// ---------------------------------------------------------------------------

/// Binary read/write helpers that match NGT's on-disk format.
///
/// NGT writes each primitive with `os.write((const char*)&v, sizeof(v))`.
/// That is little-endian on all platforms NGT supports.  We use Rust's
/// `to_le_bytes` / `from_le_bytes` explicitly.
pub mod serializer {
    use super::NgtError;
    use std::io::{Read, Write};

    pub fn write_u8(w: &mut impl Write, v: u8) -> Result<(), NgtError> {
        w.write_all(&[v]).map_err(|e| e.to_string())
    }
    pub fn read_u8(r: &mut impl Read) -> Result<u8, NgtError> {
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf[0])
    }

    pub fn write_u32(w: &mut impl Write, v: u32) -> Result<(), NgtError> {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    pub fn read_u32(r: &mut impl Read) -> Result<u32, NgtError> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn write_u64(w: &mut impl Write, v: u64) -> Result<(), NgtError> {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    pub fn read_u64(r: &mut impl Read) -> Result<u64, NgtError> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(u64::from_le_bytes(buf))
    }

    pub fn write_f32(w: &mut impl Write, v: f32) -> Result<(), NgtError> {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    pub fn read_f32(r: &mut impl Read) -> Result<f32, NgtError> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(f32::from_le_bytes(buf))
    }

    pub fn write_char(w: &mut impl Write, c: char) -> Result<(), NgtError> {
        write_u8(w, c as u8)
    }
    pub fn read_char(r: &mut impl Read) -> Result<char, NgtError> {
        Ok(read_u8(r)? as char)
    }

    pub fn write_usize(w: &mut impl Write, v: usize) -> Result<(), NgtError> {
        write_u64(w, v as u64)
    }
    pub fn read_usize(r: &mut impl Read) -> Result<usize, NgtError> {
        Ok(read_u64(r)? as usize)
    }
}
