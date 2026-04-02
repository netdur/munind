/// Block-based Walsh-Hadamard Transform (WHT) with random sign flips.
///
/// Processes vectors in blocks of 32 (matching 5 butterfly stages).
/// Each block is independently: sign-flip → WHT → normalize by 1/√32.
///
/// Based on MNN/TurboQuant: deterministic sign pattern per block position,
/// seeded from a random seed stored with the index.

use crate::common::NgtError;
use std::io::{Read, Write};

pub const BLOCK_SIZE: usize = 32;
const INV_SQRT_32: f32 = 1.0 / 5.656854; // 1/√32

pub struct Rotation {
    /// Original vector dimension.
    pub dim: usize,
    /// Number of blocks = ceil(dim / 32).
    pub num_blocks: usize,
    /// Padded dimension = num_blocks * 32.
    pub padded_dim: usize,
    /// Sign flips per padded dimension (+1.0 or -1.0).
    signs: Vec<f32>,
}

impl Rotation {
    pub fn new(dim: usize) -> Self {
        let num_blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let padded_dim = num_blocks * BLOCK_SIZE;
        let mut rng = rand::thread_rng();
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rand::Rng::gen_bool(&mut rng, 0.5) { 1.0f32 } else { -1.0f32 })
            .collect();
        Rotation { dim, num_blocks, padded_dim, signs }
    }

    /// Forward WHT: sign-flip → butterfly → normalize. Block-based.
    /// `x` has length `dim`, `out` has length `padded_dim`.
    #[inline]
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.padded_dim);

        // Copy + sign-flip + zero-pad.
        for i in 0..self.dim {
            out[i] = x[i] * self.signs[i];
        }
        for i in self.dim..self.padded_dim {
            out[i] = 0.0;
        }

        // WHT butterfly per block of 32.
        for b in 0..self.num_blocks {
            let off = b * BLOCK_SIZE;
            wht32(&mut out[off..off + BLOCK_SIZE]);
            // Normalize.
            for i in off..off + BLOCK_SIZE {
                out[i] *= INV_SQRT_32;
            }
        }
    }

    /// Inverse WHT: butterfly → normalize → sign-flip. Block-based.
    /// `y` has length `padded_dim`, `out` has length `dim`.
    #[inline]
    pub fn inverse(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.padded_dim);
        debug_assert_eq!(out.len(), self.dim);

        let mut buf = y.to_vec();
        for b in 0..self.num_blocks {
            let off = b * BLOCK_SIZE;
            wht32(&mut buf[off..off + BLOCK_SIZE]);
            for i in off..off + BLOCK_SIZE {
                buf[i] *= INV_SQRT_32;
            }
        }
        // Sign-flip and truncate to dim.
        for i in 0..self.dim {
            out[i] = buf[i] * self.signs[i];
        }
    }

    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("Rotation::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&(self.dim as u64).to_le_bytes()).map_err(|e| format!("{}", e))?;
        w.write_all(&(self.padded_dim as u64).to_le_bytes()).map_err(|e| format!("{}", e))?;
        let sign_bytes: Vec<u8> = self.signs.iter().map(|&s| if s > 0.0 { 1u8 } else { 0u8 }).collect();
        w.write_all(&sign_bytes).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("Rotation::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::new(f);
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let dim = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8).map_err(|e| format!("{}", e))?;
        let padded_dim = u64::from_le_bytes(buf8) as usize;
        let num_blocks = padded_dim / BLOCK_SIZE;
        let mut sign_bytes = vec![0u8; padded_dim];
        r.read_exact(&mut sign_bytes).map_err(|e| format!("{}", e))?;
        let signs: Vec<f32> = sign_bytes.iter().map(|&b| if b != 0 { 1.0f32 } else { -1.0f32 }).collect();
        Ok(Rotation { dim, num_blocks, padded_dim, signs })
    }
}

/// In-place WHT on exactly 32 elements (5 butterfly stages).
#[inline]
fn wht32(buf: &mut [f32]) {
    debug_assert_eq!(buf.len(), 32);
    // Stage 1: stride 1
    for i in (0..32).step_by(2) {
        let a = buf[i];
        let b = buf[i + 1];
        buf[i] = a + b;
        buf[i + 1] = a - b;
    }
    // Stage 2: stride 2
    for i in (0..32).step_by(4) {
        for j in 0..2 {
            let a = buf[i + j];
            let b = buf[i + j + 2];
            buf[i + j] = a + b;
            buf[i + j + 2] = a - b;
        }
    }
    // Stage 3: stride 4
    for i in (0..32).step_by(8) {
        for j in 0..4 {
            let a = buf[i + j];
            let b = buf[i + j + 4];
            buf[i + j] = a + b;
            buf[i + j + 4] = a - b;
        }
    }
    // Stage 4: stride 8
    for i in (0..32).step_by(16) {
        for j in 0..8 {
            let a = buf[i + j];
            let b = buf[i + j + 8];
            buf[i + j] = a + b;
            buf[i + j + 8] = a - b;
        }
    }
    // Stage 5: stride 16
    for j in 0..16 {
        let a = buf[j];
        let b = buf[j + 16];
        buf[j] = a + b;
        buf[j + 16] = a - b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let rot = Rotation::new(100);
        let x: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let mut y = vec![0.0f32; rot.padded_dim];
        let mut x2 = vec![0.0f32; 100];
        rot.forward(&x, &mut y);
        rot.inverse(&y, &mut x2);

        let error: f64 = x.iter().zip(x2.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        assert!(error / norm < 0.01, "roundtrip error: {:.6}", error / norm);
    }

    #[test]
    fn test_preserves_norm() {
        let rot = Rotation::new(64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let norm_before: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let mut y = vec![0.0f32; rot.padded_dim];
        rot.forward(&x, &mut y);
        let norm_after: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        assert!((norm_before - norm_after).abs() / norm_before < 0.01);
    }

    #[test]
    fn test_save_load() {
        let dir = "./target/test_rot_block";
        std::fs::create_dir_all(dir).unwrap();
        let path = format!("{}/rot.bin", dir);
        let rot = Rotation::new(100);
        rot.save(&path).unwrap();
        let rot2 = Rotation::load(&path).unwrap();
        assert_eq!(rot.dim, rot2.dim);
        assert_eq!(rot.signs, rot2.signs);
    }
}
