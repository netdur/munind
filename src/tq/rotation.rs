/// Random orthogonal rotation matrix via QR decomposition.
///
/// TurboQuant's first step: rotate all vectors so that each coordinate
/// follows a known Beta distribution, enabling a precomputed scalar quantizer.

use crate::common::NgtError;
use rand::Rng;
use std::io::{Read, Write};

/// A d×d orthogonal rotation matrix stored row-major.
pub struct RotationMatrix {
    pub dim: usize,
    /// Row-major: `data[i * dim + j]` = element (i, j).
    pub data: Vec<f32>,
}

impl RotationMatrix {
    /// Generate a random orthogonal matrix via QR decomposition of a
    /// random Gaussian matrix.  Uses the Gram-Schmidt process.
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = vec![0.0f32; dim * dim];

        // Fill with N(0,1) entries.
        for v in data.iter_mut() {
            // Box-Muller transform for normal distribution.
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            *v = ((-2.0 * u1.ln()).sqrt() * u2.cos()) as f32;
        }

        // Gram-Schmidt QR decomposition (modified, numerically stable).
        // We only need Q, not R.
        let mut q = vec![0.0f32; dim * dim];
        for i in 0..dim {
            // Copy column i from data into q.
            for k in 0..dim {
                q[k * dim + i] = data[k * dim + i];
            }
            // Orthogonalize against previous columns.
            for j in 0..i {
                let mut dot = 0.0f64;
                for k in 0..dim {
                    dot += (q[k * dim + i] as f64) * (q[k * dim + j] as f64);
                }
                for k in 0..dim {
                    q[k * dim + i] -= (dot as f32) * q[k * dim + j];
                }
            }
            // Normalize column i.
            let mut norm = 0.0f64;
            for k in 0..dim {
                norm += (q[k * dim + i] as f64) * (q[k * dim + i] as f64);
            }
            let inv = 1.0 / norm.sqrt();
            for k in 0..dim {
                q[k * dim + i] = (q[k * dim + i] as f64 * inv) as f32;
            }
        }

        RotationMatrix { dim, data: q }
    }

    /// Multiply: `out = Π · x` (rotate forward).
    #[inline]
    pub fn mul(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let d = self.dim;
        for i in 0..d {
            let mut sum = 0.0f32;
            let row = &self.data[i * d..(i + 1) * d];
            for j in 0..d {
                sum += row[j] * x[j];
            }
            out[i] = sum;
        }
    }

    /// Multiply: `out = Π^T · y` (rotate inverse).
    #[inline]
    pub fn mul_transpose(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let d = self.dim;
        // Π^T: column i of Π becomes row i of Π^T.
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += self.data[i * d + j] * y[i];
            }
            out[j] = sum;
        }
    }

    /// Serialize to file.
    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("RotationMatrix::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| format!("RotationMatrix::save: {}", e))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        w.write_all(bytes)
            .map_err(|e| format!("RotationMatrix::save: {}", e))?;
        Ok(())
    }

    /// Deserialize from file.
    pub fn load(path: &str) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("RotationMatrix::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::new(f);
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)
            .map_err(|e| format!("RotationMatrix::load: {}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0.0f32; dim * dim];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
        };
        r.read_exact(bytes)
            .map_err(|e| format!("RotationMatrix::load: {}", e))?;
        Ok(RotationMatrix { dim, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonality() {
        let dim = 16;
        let rot = RotationMatrix::random(dim);

        // Check Π^T · Π ≈ I.
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f64;
                for k in 0..dim {
                    dot += (rot.data[k * dim + i] as f64) * (rot.data[k * dim + j] as f64);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Π^T·Π[{},{}] = {}, expected {}",
                    i, j, dot, expected
                );
            }
        }
    }

    #[test]
    fn test_preserves_norm() {
        let dim = 32;
        let rot = RotationMatrix::random(dim);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let norm_before: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();

        let mut y = vec![0.0f32; dim];
        rot.mul(&x, &mut y);
        let norm_after: f64 = y.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();

        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm changed: {} -> {}",
            norm_before, norm_after
        );
    }

    #[test]
    fn test_inverse_roundtrip() {
        let dim = 16;
        let rot = RotationMatrix::random(dim);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 2.0).collect();

        let mut y = vec![0.0f32; dim];
        let mut x2 = vec![0.0f32; dim];
        rot.mul(&x, &mut y);
        rot.mul_transpose(&y, &mut x2);

        for i in 0..dim {
            assert!(
                (x[i] - x2[i]).abs() < 1e-4,
                "roundtrip failed at {}: {} vs {}",
                i, x[i], x2[i]
            );
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = "./target/test_rotation";
        std::fs::create_dir_all(dir).unwrap();
        let path = format!("{}/rot.bin", dir);

        let rot = RotationMatrix::random(8);
        rot.save(&path).unwrap();
        let rot2 = RotationMatrix::load(&path).unwrap();

        assert_eq!(rot.dim, rot2.dim);
        assert_eq!(rot.data, rot2.data);
    }
}
