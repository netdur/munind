/// munind SQLite loadable extension.

mod convert;
mod plan;
mod vtab;

use rusqlite::functions::FunctionFlags;
use rusqlite::vtab::update_module;
use rusqlite::{Connection, Result};
use std::os::raw::{c_char, c_int};

use vtab::MunindVTab;

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_munind_init(
    db: *mut rusqlite::ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *mut rusqlite::ffi::sqlite3_api_routines,
) -> c_int {
    unsafe { Connection::extension_init2(db, pz_err_msg, p_api, extension_init) }
}

fn extension_init(db: Connection) -> Result<bool> {
    register_functions(&db)?;
    db.create_module("munind", update_module::<MunindVTab>(), None)?;
    Ok(false)
}

// ---------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------

fn register_functions(db: &Connection) -> Result<()> {
    // munind_version() → TEXT
    db.create_scalar_function(
        "munind_version",
        0,
        FunctionFlags::SQLITE_DETERMINISTIC,
        |_ctx: &rusqlite::functions::Context<'_>| Ok(env!("CARGO_PKG_VERSION").to_string()),
    )?;

    // munind_vector(json_text_or_blob) → BLOB
    db.create_scalar_function(
        "munind_vector",
        1,
        FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx: &rusqlite::functions::Context<'_>| {
            let raw = ctx.get_raw(0);
            let floats = convert::value_to_f32_vec_infer(raw)?;
            let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&floats).to_vec();
            Ok(bytes)
        },
    )?;

    // munind_vector_json(blob) → TEXT
    db.create_scalar_function(
        "munind_vector_json",
        1,
        FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx: &rusqlite::functions::Context<'_>| {
            let blob = ctx.get::<Vec<u8>>(0)?;
            if blob.len() % 4 != 0 {
                return Err(rusqlite::Error::ModuleError(
                    "munind: blob size not a multiple of 4".into(),
                ));
            }
            let floats: &[f32] = bytemuck::cast_slice(&blob);
            let json = serde_json::to_string(floats).map_err(|e| {
                rusqlite::Error::ModuleError(format!("munind: JSON encode error: {}", e))
            })?;
            Ok(json)
        },
    )?;

    // munind_distance(blob1, blob2, metric) → REAL
    db.create_scalar_function(
        "munind_distance",
        3,
        FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx: &rusqlite::functions::Context<'_>| {
            let blob1 = ctx.get::<Vec<u8>>(0)?;
            let blob2 = ctx.get::<Vec<u8>>(1)?;
            let metric_str = ctx.get::<String>(2)?;

            if blob1.len() % 4 != 0 || blob2.len() % 4 != 0 {
                return Err(rusqlite::Error::ModuleError(
                    "munind: blob size not a multiple of 4".into(),
                ));
            }
            let a: &[f32] = bytemuck::cast_slice(&blob1);
            let b: &[f32] = bytemuck::cast_slice(&blob2);

            if a.len() != b.len() {
                return Err(rusqlite::Error::ModuleError(format!(
                    "munind: dimension mismatch: {} vs {}",
                    a.len(),
                    b.len()
                )));
            }

            let distance: f64 = match metric_str.as_str() {
                "l2" => {
                    let sum: f64 = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| {
                            let d = (*x as f64) - (*y as f64);
                            d * d
                        })
                        .sum();
                    sum.sqrt()
                }
                "cosine" => {
                    let dot: f64 =
                        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                    let na: f64 =
                        a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                    let nb: f64 =
                        b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                    if na == 0.0 || nb == 0.0 {
                        1.0
                    } else {
                        1.0 - dot / (na * nb)
                    }
                }
                "ip" | "inner_product" => {
                    let dot: f64 =
                        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                    -dot
                }
                "l1" => a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
                    .sum(),
                _ => {
                    return Err(rusqlite::Error::ModuleError(format!(
                        "munind: unknown metric: {}",
                        metric_str
                    )));
                }
            };

            Ok(distance)
        },
    )?;

    Ok(())
}
