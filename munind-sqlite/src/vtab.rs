/// MunindVTab — SQLite virtual table backed by munind-core ANN index.

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_int;

use rusqlite::ffi;
use rusqlite::types::ValueRef;
use rusqlite::vtab::{
    Context, CreateVTab, IndexConstraintOp, IndexInfo, UpdateVTab, VTab, VTabConnection,
    VTabCursor, VTabKind, Values,
};
use rusqlite::{Error, Result};

use munind_core::api::{Distance, Index, IndexConfig};

use crate::convert;
use crate::plan::*;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA: &str = "\
    CREATE TABLE x(\
        vector BLOB,\
        distance REAL HIDDEN,\
        k INTEGER HIDDEN,\
        epsilon REAL HIDDEN\
    )";

// ---------------------------------------------------------------------------
// MunindVTab
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct MunindVTab {
    base: ffi::sqlite3_vtab,
    index: Index,
    dim: usize,
    dirty: bool,
    index_path: String,
    /// munind internal ID → user rowid
    rowid_map: Vec<Option<i64>>,
    /// user rowid → munind internal ID
    reverse_map: HashMap<i64, u32>,
}

impl MunindVTab {
    fn parse_args(args: &[&[u8]]) -> Result<(usize, Distance)> {
        let mut dim: Option<usize> = None;
        let mut metric = Distance::L2;

        for &arg in args.iter().skip(3) {
            let s = std::str::from_utf8(arg)
                .map_err(|_| Error::ModuleError("munind: invalid UTF-8 in argument".into()))?;
            let s = s.trim();
            if let Some(val) = s.strip_prefix("dim=") {
                dim = Some(val.parse::<usize>().map_err(|_| {
                    Error::ModuleError(format!("munind: invalid dim value: {}", val))
                })?);
            } else if let Some(val) = s.strip_prefix("metric=") {
                metric = match val {
                    "l1" => Distance::L1,
                    "l2" => Distance::L2,
                    "cosine" => Distance::Cosine,
                    "ip" | "inner_product" => Distance::InnerProduct,
                    "angle" => Distance::Angle,
                    _ => {
                        return Err(Error::ModuleError(format!(
                            "munind: unknown metric: {}",
                            val
                        )));
                    }
                };
            }
        }

        let dim = dim.ok_or_else(|| {
            Error::ModuleError("munind: dim=N is required (e.g. dim=512)".into())
        })?;
        Ok((dim, metric))
    }

    fn derive_index_path(db: &mut VTabConnection, args: &[&[u8]]) -> Result<String> {
        let table_name = std::str::from_utf8(args[2]).unwrap_or("munind_vtab");
        let db_name = std::str::from_utf8(args[1]).unwrap_or("main");

        // Get the database file path via sqlite3_db_filename.
        let db_path = unsafe {
            let handle = db.handle();
            let c_db_name = std::ffi::CString::new(db_name).unwrap_or_default();
            let filename_ptr = ffi::sqlite3_db_filename(handle, c_db_name.as_ptr());
            if filename_ptr.is_null() {
                None
            } else {
                CStr::from_ptr(filename_ptr).to_str().ok().map(String::from)
            }
        };

        let base = match db_path {
            Some(ref p) if !p.is_empty() => p.clone(),
            _ => {
                // In-memory database: use temp directory
                let tmp = std::env::temp_dir();
                tmp.join("munind_sqlite").to_string_lossy().into_owned()
            }
        };

        Ok(format!("{}-munind-{}", base, table_name))
    }

    fn save_rowid_map(&self) -> Result<()> {
        let path = format!("{}/rowmap", self.index_path);
        let data = bincode_encode_rowid_map(&self.rowid_map, &self.reverse_map);
        std::fs::write(&path, &data)
            .map_err(|e| Error::ModuleError(format!("munind: failed to save rowid map: {}", e)))?;
        Ok(())
    }

    fn load_rowid_map(&mut self) -> Result<()> {
        let path = format!("{}/rowmap", self.index_path);
        if let Ok(data) = std::fs::read(&path) {
            if let Some((rmap, revmap)) = bincode_decode_rowid_map(&data) {
                self.rowid_map = rmap;
                self.reverse_map = revmap;
            }
        }
        Ok(())
    }
}

// Simple encoding for the rowid map.
fn bincode_encode_rowid_map(
    rowid_map: &[Option<i64>],
    reverse_map: &HashMap<i64, u32>,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(rowid_map.len() as u64).to_le_bytes());
    for entry in rowid_map {
        match entry {
            Some(v) => {
                buf.push(1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            None => {
                buf.push(0);
            }
        }
    }
    buf.extend_from_slice(&(reverse_map.len() as u64).to_le_bytes());
    for (&k, &v) in reverse_map {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn bincode_decode_rowid_map(data: &[u8]) -> Option<(Vec<Option<i64>>, HashMap<i64, u32>)> {
    let mut pos = 0;
    if data.len() < 8 {
        return None;
    }

    let map_len = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?) as usize;
    pos += 8;

    let mut rowid_map = Vec::with_capacity(map_len);
    for _ in 0..map_len {
        if pos >= data.len() {
            return None;
        }
        let tag = data[pos];
        pos += 1;
        if tag == 1 {
            if pos + 8 > data.len() {
                return None;
            }
            let v = i64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
            pos += 8;
            rowid_map.push(Some(v));
        } else {
            rowid_map.push(None);
        }
    }

    if pos + 8 > data.len() {
        return None;
    }
    let rev_len = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?) as usize;
    pos += 8;

    let mut reverse_map = HashMap::with_capacity(rev_len);
    for _ in 0..rev_len {
        if pos + 12 > data.len() {
            return None;
        }
        let k = i64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let v = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        reverse_map.insert(k, v);
    }

    Some((rowid_map, reverse_map))
}

// ---------------------------------------------------------------------------
// VTab trait
// ---------------------------------------------------------------------------

unsafe impl<'vtab> VTab<'vtab> for MunindVTab {
    type Aux = ();
    type Cursor = MunindCursor;

    fn connect(
        db: &mut VTabConnection,
        _aux: Option<&()>,
        args: &[&[u8]],
    ) -> Result<(String, Self)> {
        let (dim, metric) = Self::parse_args(args)?;
        let index_path = Self::derive_index_path(db, args)?;

        // Try to open existing index, else create new.
        let index = if std::path::Path::new(&format!("{}/obj", index_path)).exists() {
            Index::open(&index_path).map_err(|e| {
                Error::ModuleError(format!("munind: failed to open index: {}", e))
            })?
        } else {
            std::fs::create_dir_all(&index_path).map_err(|e| {
                Error::ModuleError(format!("munind: failed to create directory: {}", e))
            })?;
            let config = IndexConfig::new(dim, metric);
            Index::create(config).map_err(|e| {
                Error::ModuleError(format!("munind: failed to create index: {}", e))
            })?
        };

        let mut vtab = MunindVTab {
            base: unsafe { std::mem::zeroed() },
            index,
            dim,
            dirty: false,
            index_path,
            rowid_map: vec![None], // slot 0 reserved
            reverse_map: HashMap::new(),
        };
        vtab.load_rowid_map()?;

        Ok((SCHEMA.to_string(), vtab))
    }

    fn best_index(&self, info: &mut IndexInfo) -> Result<()> {
        let mut match_idx: Option<usize> = None;
        let mut k_idx: Option<usize> = None;
        let mut limit_idx: Option<usize> = None;
        let mut epsilon_idx: Option<usize> = None;

        for (i, constraint) in info.constraints().enumerate() {
            if !constraint.is_usable() {
                continue;
            }
            match (constraint.column(), constraint.operator()) {
                (0, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_MATCH) => {
                    match_idx = Some(i);
                }
                (2, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ) => {
                    k_idx = Some(i);
                }
                (3, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ) => {
                    epsilon_idx = Some(i);
                }
                (_, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_LIMIT) => {
                    limit_idx = Some(i);
                }
                _ => {}
            }
        }

        if let Some(mi) = match_idx {
            let k_source = k_idx.or(limit_idx);
            if let Some(ki) = k_source {
                let mut argv_index: c_int = 1;
                let mut idx_str = String::new();

                let mut match_set = false;
                let mut k_set = false;
                let mut eps_set = false;

                for (i, (constraint, mut usage)) in
                    info.constraints_and_usages().enumerate()
                {
                    if !constraint.is_usable() {
                        continue;
                    }
                    if i == mi && !match_set {
                        usage.set_argv_index(argv_index);
                        usage.set_omit(true);
                        idx_str.push(ARG_MATCH_VECTOR as char);
                        argv_index += 1;
                        match_set = true;
                    }
                    if i == ki && !k_set {
                        usage.set_argv_index(argv_index);
                        usage.set_omit(true);
                        let ch = if k_idx.is_some() && i == k_idx.unwrap() {
                            ARG_K_VALUE
                        } else {
                            ARG_LIMIT
                        };
                        idx_str.push(ch as char);
                        argv_index += 1;
                        k_set = true;
                    }
                    if epsilon_idx == Some(i) && !eps_set {
                        usage.set_argv_index(argv_index);
                        usage.set_omit(true);
                        idx_str.push(ARG_EPSILON as char);
                        argv_index += 1;
                        eps_set = true;
                    }
                }

                info.set_idx_num(PLAN_KNN);
                info.set_idx_str(&idx_str);
                info.set_estimated_cost(30.0);
                info.set_estimated_rows(10);

                // Check if ORDER BY is on distance column ascending
                let order_bys: Vec<_> = info.order_bys().collect();
                if order_bys.len() == 1
                    && order_bys[0].column() == 1
                    && !order_bys[0].is_order_by_desc()
                {
                    info.set_order_by_consumed(true);
                }

                return Ok(());
            }
        }

        // Check for rowid = ? (point lookup, needed for DELETE WHERE rowid = ?)
        let mut rowid_idx: Option<usize> = None;
        for (i, constraint) in info.constraints().enumerate() {
            if !constraint.is_usable() {
                continue;
            }
            if constraint.column() == -1
                && constraint.operator() == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ
            {
                rowid_idx = Some(i);
                break;
            }
        }

        if let Some(ri) = rowid_idx {
            for (i, (_constraint, mut usage)) in info.constraints_and_usages().enumerate() {
                if i == ri {
                    usage.set_argv_index(1);
                    break;
                }
            }
            info.set_idx_num(PLAN_POINT);
            info.set_idx_str("R");
            info.set_estimated_cost(10.0);
            info.set_estimated_rows(1);
            return Ok(());
        }

        // Fallback: full scan
        info.set_idx_num(PLAN_FULLSCAN);
        info.set_estimated_cost(3_000_000.0);
        info.set_estimated_rows(self.index.len().max(1) as i64);
        Ok(())
    }

    fn open(&'vtab mut self) -> Result<Self::Cursor> {
        Ok(MunindCursor {
            base: unsafe { std::mem::zeroed() },
            results: Vec::new(),
            pos: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// CreateVTab
// ---------------------------------------------------------------------------

impl<'vtab> CreateVTab<'vtab> for MunindVTab {
    const KIND: VTabKind = VTabKind::Default;

    fn create(
        db: &mut VTabConnection,
        aux: Option<&()>,
        args: &[&[u8]],
    ) -> Result<(String, Self)> {
        // create delegates to connect, which handles both fresh and existing
        <Self as VTab>::connect(db, aux, args)
    }

    fn destroy(&self) -> Result<()> {
        let _ = std::fs::remove_dir_all(&self.index_path);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UpdateVTab
// ---------------------------------------------------------------------------

impl<'vtab> UpdateVTab<'vtab> for MunindVTab {
    fn delete(&mut self, arg: ValueRef<'_>) -> Result<()> {
        let user_rowid = match arg {
            ValueRef::Integer(i) => i,
            _ => return Err(Error::ModuleError("munind: expected integer rowid".into())),
        };

        if let Some(&internal_id) = self.reverse_map.get(&user_rowid) {
            self.index.remove(internal_id).map_err(|e| {
                Error::ModuleError(format!("munind: delete failed: {}", e))
            })?;
            if (internal_id as usize) < self.rowid_map.len() {
                self.rowid_map[internal_id as usize] = None;
            }
            self.reverse_map.remove(&user_rowid);
            self.dirty = true;
        }

        Ok(())
    }

    fn insert(&mut self, args: &Values<'_>) -> Result<i64> {
        // args layout for INSERT:
        //   args[0] = NULL (signals INSERT)
        //   args[1] = new rowid (or NULL for auto)
        //   args[2] = column 0 (vector)
        //   args[3] = column 1 (distance — HIDDEN, ignored)
        //   args[4] = column 2 (k — HIDDEN, ignored)

        let user_rowid_opt: Option<i64> = args.get(1)?;

        // Get the vector value via iterator (Values doesn't have get_ref)
        let vector_ref = args.iter().nth(2).ok_or_else(|| {
            Error::ModuleError("munind: missing vector column".into())
        })?;

        let vec = convert::value_to_f32_vec(vector_ref, self.dim)?;

        // Insert into munind
        let internal_id = self.index.insert(&vec).map_err(|e| {
            Error::ModuleError(format!("munind: insert failed: {}", e))
        })?;

        let user_rowid = user_rowid_opt.unwrap_or(internal_id as i64);

        if self.reverse_map.contains_key(&user_rowid) {
            return Err(Error::ModuleError(format!(
                "munind: rowid {} already exists",
                user_rowid
            )));
        }

        // Store mapping
        while self.rowid_map.len() <= internal_id as usize {
            self.rowid_map.push(None);
        }
        self.rowid_map[internal_id as usize] = Some(user_rowid);
        self.reverse_map.insert(user_rowid, internal_id);

        self.dirty = true;

        Ok(user_rowid)
    }

    fn update(&mut self, _args: &Values<'_>) -> Result<()> {
        Err(Error::ModuleError(
            "munind: UPDATE is not supported (use DELETE + INSERT)".into(),
        ))
    }
}

// Save on disconnect (but not if directory was removed by xDestroy)
impl Drop for MunindVTab {
    fn drop(&mut self) {
        if !std::path::Path::new(&self.index_path).exists() {
            return; // xDestroy already cleaned up
        }
        if self.dirty {
            let _ = self.index.build();
            self.dirty = false;
        }
        let _ = self.index.save(&self.index_path);
        let _ = self.save_rowid_map();
    }
}

// ---------------------------------------------------------------------------
// MunindCursor
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct MunindCursor {
    base: ffi::sqlite3_vtab_cursor,
    /// Buffered results: (user_rowid, distance)
    results: Vec<(i64, f32)>,
    pos: usize,
}

unsafe impl VTabCursor for MunindCursor {
    fn filter(&mut self, idx_num: c_int, idx_str: Option<&str>, args: &Values<'_>) -> Result<()> {
        self.results.clear();
        self.pos = 0;

        // Access the vtab via the cursor's pVtab pointer.
        let vtab: &mut MunindVTab = unsafe {
            let vtab_ptr = self.base.pVtab as *mut MunindVTab;
            &mut *vtab_ptr
        };

        match idx_num {
            PLAN_KNN => {
                let idx_bytes = idx_str.unwrap_or("").as_bytes();

                let mut query_vec: Option<Vec<f32>> = None;
                let mut k: usize = 10;
                let mut epsilon: f32 = 0.1;

                let arg_values: Vec<ValueRef<'_>> = args.iter().collect();

                for (i, &code) in idx_bytes.iter().enumerate() {
                    if i >= arg_values.len() {
                        break;
                    }
                    match code {
                        ARG_MATCH_VECTOR => {
                            query_vec =
                                Some(convert::value_to_f32_vec(arg_values[i], vtab.dim)?);
                        }
                        ARG_K_VALUE | ARG_LIMIT => {
                            k = match arg_values[i] {
                                ValueRef::Integer(v) => v as usize,
                                _ => {
                                    return Err(Error::ModuleError(
                                        "munind: k must be an integer".into(),
                                    ))
                                }
                            };
                        }
                        ARG_EPSILON => {
                            epsilon = match arg_values[i] {
                                ValueRef::Real(v) => v as f32,
                                ValueRef::Integer(v) => v as f32,
                                _ => {
                                    return Err(Error::ModuleError(
                                        "munind: epsilon must be a number".into(),
                                    ))
                                }
                            };
                        }
                        _ => {}
                    }
                }

                let query = query_vec.ok_or_else(|| {
                    Error::ModuleError("munind: missing query vector in MATCH".into())
                })?;

                // Lazy build if dirty
                if vtab.dirty {
                    vtab.index.build().map_err(|e| {
                        Error::ModuleError(format!("munind: build failed: {}", e))
                    })?;
                    vtab.dirty = false;
                }

                // Execute search
                let raw_results = vtab.index.search_with(&query, k, epsilon, None).map_err(|e| {
                    Error::ModuleError(format!("munind: search failed: {}", e))
                })?;

                // Map internal IDs to user rowids
                self.results = raw_results
                    .into_iter()
                    .filter_map(|r| {
                        let user_rowid = vtab
                            .rowid_map
                            .get(r.id as usize)
                            .and_then(|opt| *opt)?;
                        Some((user_rowid, r.distance))
                    })
                    .collect();
            }
            PLAN_POINT => {
                // Point lookup by rowid
                let arg_values: Vec<ValueRef<'_>> = args.iter().collect();
                if let Some(ValueRef::Integer(rowid)) = arg_values.first() {
                    if vtab.reverse_map.contains_key(rowid) {
                        self.results.push((*rowid, 0.0));
                    }
                }
            }
            PLAN_FULLSCAN => {
                // Iterate all present objects
                for (&user_rowid, _) in &vtab.reverse_map {
                    self.results.push((user_rowid, 0.0));
                }
                self.results.sort_by_key(|&(rid, _)| rid);
            }
            _ => {
                return Err(Error::ModuleError(
                    "munind: unknown query plan".into(),
                ));
            }
        }

        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        self.pos += 1;
        Ok(())
    }

    fn eof(&self) -> bool {
        self.pos >= self.results.len()
    }

    fn column(&self, ctx: &mut Context, col: c_int) -> Result<()> {
        if self.pos >= self.results.len() {
            return Ok(());
        }

        let (user_rowid, distance) = self.results[self.pos];

        match col {
            0 => {
                // vector column: retrieve stored vector
                let vtab: &MunindVTab = unsafe {
                    let vtab_ptr = self.base.pVtab as *const MunindVTab;
                    &*vtab_ptr
                };
                if let Some(&internal_id) = vtab.reverse_map.get(&user_rowid) {
                    match vtab.index.get(internal_id) {
                        Ok(vec) => {
                            let bytes: Vec<u8> =
                                bytemuck::cast_slice::<f32, u8>(&vec).to_vec();
                            ctx.set_result(&bytes)?;
                        }
                        Err(_) => {
                            ctx.set_result(&rusqlite::types::Null)?;
                        }
                    }
                } else {
                    ctx.set_result(&rusqlite::types::Null)?;
                }
            }
            1 => {
                ctx.set_result(&(distance as f64))?;
            }
            _ => {
                ctx.set_result(&rusqlite::types::Null)?;
            }
        }
        Ok(())
    }

    fn rowid(&self) -> Result<i64> {
        if self.pos >= self.results.len() {
            return Ok(0);
        }
        Ok(self.results[self.pos].0)
    }
}
