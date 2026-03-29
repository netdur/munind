/// Randomized Walsh-Hadamard Transform (WHT) rotation.
///
/// Replaces the full d×d random orthogonal matrix with HD·x where:
///   D = diagonal of random ±1 signs (d values)
///   H = normalized Hadamard transform (in-place butterfly, O(d log d))
///
/// This is an orthogonal transform that maps unit-sphere vectors to
/// coordinates with near-Gaussian distribution — same effect as a random
/// rotation matrix but O(d log d) instead of O(d²).
///
/// Requires dim to be a power of 2. If the original dim is not power-of-2,
/// we pad to the next power of 2 and zero-fill.

use crate::common::NgtError;
use rand::Rng;
use std::io::{Read, Write};

pub struct RotationMatrix {
    /// Original dimension (may not be power of 2).
    pub dim: usize,
    /// Padded dimension (next power of 2 ≥ dim).
    padded_dim: usize,
    /// Random ±1 sign flips, length = padded_dim.
    signs: Vec<f32>,
}

impl RotationMatrix {
    /// Generate a new randomized Hadamard rotation for dimension `dim`.
    pub fn random(dim: usize) -> Self {
        let padded_dim = dim.next_power_of_two();
        let mut rng = rand::thread_rng();
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rng.gen_bool(0.5) { 1.0f32 } else { -1.0f32 })
            .collect();
        RotationMatrix { dim, padded_dim, signs }
    }

    /// Padded dimension (power of 2).
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Forward transform: `out = (1/√n) · H · D · x`.
    /// `x` has length `dim`, `out` has length `padded_dim`.
    /// The extra coordinates carry real energy and must be preserved.
    #[inline]
    pub fn mul(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.padded_dim);
        let n = self.padded_dim;

        // Apply sign flips (D · x), zero-pad the rest.
        for i in 0..self.dim {
            out[i] = x[i] * self.signs[i];
        }
        for i in self.dim..n {
            out[i] = 0.0;
        }

        // In-place Walsh-Hadamard transform (butterfly).
        wht_in_place(out);

        // Normalize: 1/√n.
        let inv = 1.0 / (n as f32).sqrt();
        for v in out.iter_mut() {
            *v *= inv;
        }
    }

    /// Inverse transform: `out = D · H · (1/√n) · y`.
    /// `y` has length `padded_dim`, `out` has length `dim` (truncated).
    #[inline]
    pub fn mul_transpose(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.padded_dim);
        debug_assert_eq!(out.len(), self.dim);
        let n = self.padded_dim;

        // Copy to work buffer.
        let mut buf = y.to_vec();

        // H (same butterfly — H is symmetric).
        wht_in_place(&mut buf);

        // Normalize and apply D (signs are self-inverse), truncate to dim.
        let inv = 1.0 / (n as f32).sqrt();
        for i in 0..self.dim {
            out[i] = buf[i] * inv * self.signs[i];
        }
    }

    /// Serialize: just the signs vector + dims.
    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("RotationMatrix::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        w.write_all(&(self.padded_dim as u64).to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        // Write signs as bytes: +1 → 1, -1 → 0.
        let sign_bytes: Vec<u8> = self.signs.iter().map(|&s| if s > 0.0 { 1u8 } else { 0u8 }).collect();
        w.write_all(&sign_bytes)
            .map_err(|e| format!("{}", e))?;
        Ok(())
    }

    /// Deserialize.
    pub fn load(path: &str) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("RotationMatrix::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::new(f);
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let padded_dim = u64::from_le_bytes(buf8) as usize;
        let mut sign_bytes = vec![0u8; padded_dim];
        r.read_exact(&mut sign_bytes).map_err(|e| format!("{}", e))?;
        let signs: Vec<f32> = sign_bytes.iter().map(|&b| if b != 0 { 1.0f32 } else { -1.0f32 }).collect();
        Ok(RotationMatrix { dim, padded_dim, signs })
    }
}

/// In-place Walsh-Hadamard Transform (iterative butterfly).
/// `buf` length must be a power of 2.
#[inline]
fn wht_in_place(buf: &mut [f32]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserves_norm() {
        let dim = 100;
        let rot = RotationMatrix::random(dim);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let norm_before: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();

        let mut y = vec![0.0f32; rot.padded_dim()];
        rot.mul(&x, &mut y);
        // Norm is preserved across all padded_dim coordinates.
        let norm_after: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();

        assert!(
            (norm_before - norm_after).abs() / norm_before < 0.01,
            "norm changed: {} -> {}",
            norm_before, norm_after
        );
    }

    #[test]
    fn test_inverse_roundtrip() {
        let dim = 64; // Power of 2 — no padding.
        let rot = RotationMatrix::random(dim);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 10.0).collect();

        let mut y = vec![0.0f32; rot.padded_dim()];
        let mut x2 = vec![0.0f32; dim];
        rot.mul(&x, &mut y);
        rot.mul_transpose(&y, &mut x2);

        for i in 0..dim {
            assert!(
                (x[i] - x2[i]).abs() < 0.01,
                "roundtrip failed at {}: {} vs {}",
                i, x[i], x2[i]
            );
        }
    }

    #[test]
    fn test_inverse_roundtrip_non_power_of_2() {
        let dim = 100; // Pads to 128.
        let rot = RotationMatrix::random(dim);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 5.0).collect();

        let mut y = vec![0.0f32; rot.padded_dim()];
        let mut x2 = vec![0.0f32; dim];
        rot.mul(&x, &mut y);
        rot.mul_transpose(&y, &mut x2);

        let error: f64 = x.iter().zip(x2.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        assert!(
            error / norm < 0.01,
            "roundtrip relative error too high: {:.6}",
            error / norm
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = "./target/test_rotation_wht";
        std::fs::create_dir_all(dir).unwrap();
        let path = format!("{}/rot.bin", dir);

        let rot = RotationMatrix::random(100);
        rot.save(&path).unwrap();
        let rot2 = RotationMatrix::load(&path).unwrap();

        assert_eq!(rot.dim, rot2.dim);
        assert_eq!(rot.padded_dim, rot2.padded_dim);
        assert_eq!(rot.signs, rot2.signs);
    }

    #[test]
    fn test_wht_butterfly() {
        // Known WHT of [1, 1, 1, 1] = [4, 0, 0, 0].
        let mut buf = vec![1.0f32, 1.0, 1.0, 1.0];
        wht_in_place(&mut buf);
        assert_eq!(buf, vec![4.0, 0.0, 0.0, 0.0]);

        // WHT of [1, 0, 0, 0] = [1, 1, 1, 1].
        let mut buf = vec![1.0f32, 0.0, 0.0, 0.0];
        wht_in_place(&mut buf);
        assert_eq!(buf, vec![1.0, 1.0, 1.0, 1.0]);
    }
}
