/// C FFI for munind.
///
/// Build with: `cargo build --release`
/// Produces: `target/release/libmunind.dylib` (macOS) / `.so` (Linux)
///
/// Header: see `include/munind.h`

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use crate::api::{Distance, Index, IndexConfig, SearchResult};

/// Opaque index handle for C callers.
pub struct MunindIndex {
    index: Index,
}

/// Search result for C callers.
#[repr(C)]
pub struct MunindResult {
    pub id: u32,
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a new empty index.
/// `distance`: "l2", "cosine", "ip" (inner product), "l1".
/// Returns null on error.
#[unsafe(no_mangle)]
pub extern "C" fn munind_create(dim: u32, distance: *const c_char) -> *mut MunindIndex {
    let dist_str = if distance.is_null() {
        "l2"
    } else {
        match unsafe { CStr::from_ptr(distance) }.to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };
    let dist = match dist_str {
        "l1" => Distance::L1,
        "l2" => Distance::L2,
        "cosine" => Distance::Cosine,
        "ip" | "inner_product" => Distance::InnerProduct,
        "angle" => Distance::Angle,
        _ => Distance::L2,
    };
    let config = IndexConfig::new(dim as usize, dist);
    match Index::create(config) {
        Ok(index) => Box::into_raw(Box::new(MunindIndex { index })),
        Err(_) => ptr::null_mut(),
    }
}

/// Open an existing index from a directory path.
/// Returns null on error.
#[unsafe(no_mangle)]
pub extern "C" fn munind_open(path: *const c_char) -> *mut MunindIndex {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match Index::open(path) {
        Ok(index) => Box::into_raw(Box::new(MunindIndex { index })),
        Err(_) => ptr::null_mut(),
    }
}

/// Free an index.
#[unsafe(no_mangle)]
pub extern "C" fn munind_free(idx: *mut MunindIndex) {
    if !idx.is_null() {
        unsafe { drop(Box::from_raw(idx)) };
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

/// Insert a vector. Returns the assigned ID, or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn munind_insert(
    idx: *mut MunindIndex,
    vec: *const f32,
    dim: u32,
) -> u32 {
    if idx.is_null() || vec.is_null() {
        return 0;
    }
    let idx = unsafe { &mut *idx };
    let slice = unsafe { std::slice::from_raw_parts(vec, dim as usize) };
    match idx.index.insert(slice) {
        Ok(id) => id,
        Err(_) => 0,
    }
}

/// Build the graph index. Call after all inserts.
#[unsafe(no_mangle)]
pub extern "C" fn munind_build(idx: *mut MunindIndex) {
    if idx.is_null() {
        return;
    }
    let idx = unsafe { &mut *idx };
    let _ = idx.index.build();
}

/// Save the index to a directory. Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn munind_save(idx: *const MunindIndex, path: *const c_char) -> i32 {
    if idx.is_null() || path.is_null() {
        return -1;
    }
    let idx = unsafe { &*idx };
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match idx.index.save(path) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search for k nearest neighbors.
/// Results are written to `results` (caller-allocated, size >= k).
/// `result_count` is set to the actual number of results found.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn munind_search(
    idx: *const MunindIndex,
    query: *const f32,
    dim: u32,
    k: u32,
    epsilon: f32,
    results: *mut MunindResult,
    result_count: *mut u32,
) -> i32 {
    if idx.is_null() || query.is_null() || results.is_null() || result_count.is_null() {
        return -1;
    }
    let idx = unsafe { &*idx };
    let query = unsafe { std::slice::from_raw_parts(query, dim as usize) };
    let eps = if epsilon <= 0.0 { 0.1 } else { epsilon };
    match idx.index.search_with(query, k as usize, eps, None) {
        Ok(res) => {
            let n = res.len().min(k as usize);
            for i in 0..n {
                unsafe {
                    *results.add(i) = MunindResult {
                        id: res[i].id,
                        distance: res[i].distance,
                    };
                }
            }
            unsafe { *result_count = n as u32 };
            0
        }
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Return the number of objects in the index.
#[unsafe(no_mangle)]
pub extern "C" fn munind_len(idx: *const MunindIndex) -> u32 {
    if idx.is_null() {
        return 0;
    }
    let idx = unsafe { &*idx };
    idx.index.len() as u32
}

/// Return the dimension of the index.
#[unsafe(no_mangle)]
pub extern "C" fn munind_dimension(idx: *const MunindIndex) -> u32 {
    if idx.is_null() {
        return 0;
    }
    let idx = unsafe { &*idx };
    idx.index.dimension() as u32
}
