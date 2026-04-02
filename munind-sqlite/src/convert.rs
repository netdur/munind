/// Conversion utilities between SQLite types and f32 vectors.

use rusqlite::types::ValueRef;
use rusqlite::Error;

/// Convert a SQLite value (BLOB or JSON text) to Vec<f32>.
pub fn value_to_f32_vec(value: ValueRef<'_>, dim: usize) -> Result<Vec<f32>, Error> {
    match value {
        ValueRef::Blob(b) => {
            if b.len() != dim * 4 {
                return Err(Error::ModuleError(format!(
                    "munind: vector blob has {} bytes, expected {} (dim={})",
                    b.len(),
                    dim * 4,
                    dim
                )));
            }
            Ok(bytemuck::cast_slice::<u8, f32>(b).to_vec())
        }
        ValueRef::Text(t) => {
            let text = std::str::from_utf8(t).map_err(|e| {
                Error::ModuleError(format!("munind: invalid UTF-8 in vector text: {}", e))
            })?;
            let arr: Vec<f32> = serde_json::from_str(text).map_err(|e| {
                Error::ModuleError(format!("munind: invalid JSON vector: {}", e))
            })?;
            if arr.len() != dim {
                return Err(Error::ModuleError(format!(
                    "munind: JSON vector has {} elements, expected {}",
                    arr.len(),
                    dim
                )));
            }
            Ok(arr)
        }
        _ => Err(Error::ModuleError(
            "munind: vector must be a BLOB or JSON text".to_string(),
        )),
    }
}

/// Convert a SQLite value (BLOB or JSON text) to Vec<f32>, inferring dimension.
pub fn value_to_f32_vec_infer(value: ValueRef<'_>) -> Result<Vec<f32>, Error> {
    match value {
        ValueRef::Blob(b) => {
            if b.len() % 4 != 0 {
                return Err(Error::ModuleError(format!(
                    "munind: vector blob has {} bytes (not a multiple of 4)",
                    b.len()
                )));
            }
            Ok(bytemuck::cast_slice::<u8, f32>(b).to_vec())
        }
        ValueRef::Text(t) => {
            let text = std::str::from_utf8(t).map_err(|e| {
                Error::ModuleError(format!("munind: invalid UTF-8 in vector text: {}", e))
            })?;
            let arr: Vec<f32> = serde_json::from_str(text).map_err(|e| {
                Error::ModuleError(format!("munind: invalid JSON vector: {}", e))
            })?;
            Ok(arr)
        }
        _ => Err(Error::ModuleError(
            "munind: vector must be a BLOB or JSON text".to_string(),
        )),
    }
}

/// Convert &[f32] to &[u8] for SQLite BLOB output.
#[allow(dead_code)]
pub fn f32_slice_to_bytes(slice: &[f32]) -> &[u8] {
    bytemuck::cast_slice(slice)
}
