/// Block-based TurboQuant encoder/decoder.
///
/// Each vector is processed as ceil(dim/32) blocks of 32.
/// Per block: RMS normalize → WHT → scalar quantize → pack.
/// Storage: packed codes + per-block f32 scale.

use super::codebook;
use super::rotation::{Rotation, BLOCK_SIZE};
use crate::common::NgtError;

/// Encoded vector.
pub struct Encoded {
    /// Quantized codes, one u8 per padded dimension.
    pub codes: Vec<u8>,
    /// RMS scale per block (num_blocks values).
    pub scales: Vec<f32>,
}

pub struct TqQuantizer {
    pub dim: usize,
    pub bits: u32,
    pub rotation: Rotation,
}

impl TqQuantizer {
    pub fn new(dim: usize, bits: u32) -> Self {
        TqQuantizer {
            dim,
            bits,
            rotation: Rotation::new(dim),
        }
    }

    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let rotation = Rotation::load(&format!("{}/rotation.tq", dir))?;
        // Read bits from property.
        let mut ps = crate::common::PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let bits = ps.get_i64("TqBits", 8) as u32;
        let dim = rotation.dim;
        Ok(TqQuantizer { dim, bits, rotation })
    }

    pub fn save(&self, dir: &str) -> Result<(), NgtError> {
        self.rotation.save(&format!("{}/rotation.tq", dir))
    }

    pub fn padded_dim(&self) -> usize {
        self.rotation.padded_dim
    }

    pub fn num_blocks(&self) -> usize {
        self.rotation.num_blocks
    }

    /// Encode a vector.
    pub fn encode(&self, x: &[f32]) -> Encoded {
        let pd = self.rotation.padded_dim;
        let nb = self.rotation.num_blocks;

        // Forward WHT.
        let mut rotated = vec![0.0f32; pd];
        self.rotation.forward(x, &mut rotated);

        // Per-block: compute RMS scale, normalize, quantize.
        let mut codes = vec![0u8; pd];
        let mut scales = vec![0.0f32; nb];

        for b in 0..nb {
            let off = b * BLOCK_SIZE;
            let block = &rotated[off..off + BLOCK_SIZE];

            // RMS = sqrt(mean(x²)).
            let rms: f32 = (block.iter().map(|&v| v * v).sum::<f32>()
                / BLOCK_SIZE as f32)
                .sqrt();
            scales[b] = rms;

            let inv_rms = if rms > 0.0 { 1.0 / rms } else { 1.0 };

            // Normalize and quantize each value.
            for i in 0..BLOCK_SIZE {
                let normalized = block[i] * inv_rms;
                codes[off + i] = match self.bits {
                    3 => codebook::quantize_tq3(normalized),
                    4 => codebook::quantize_tq4(normalized),
                    _ => codebook::quantize_tq8(normalized),
                };
            }
        }

        Encoded { codes, scales }
    }

    /// Decode: unpack → dequantize → inverse WHT.
    pub fn decode(&self, enc: &Encoded) -> Vec<f32> {
        let pd = self.rotation.padded_dim;
        let nb = self.rotation.num_blocks;

        let mut rotated = vec![0.0f32; pd];
        for b in 0..nb {
            let off = b * BLOCK_SIZE;
            let scale = enc.scales[b];
            for i in 0..BLOCK_SIZE {
                let val = match self.bits {
                    3 => codebook::dequantize_tq3(enc.codes[off + i]),
                    4 => codebook::dequantize_tq4(enc.codes[off + i]),
                    _ => codebook::dequantize_tq8(enc.codes[off + i]),
                };
                rotated[off + i] = val * scale;
            }
        }

        let mut out = vec![0.0f32; self.dim];
        self.rotation.inverse(&rotated, &mut out);
        out
    }

    /// Dequantize in the rotated domain (no inverse WHT).
    /// Returns padded_dim-length vector = scale * centroid[code].
    /// This is the fast path for search — avoids the inverse WHT per object.
    #[inline]
    pub fn dequantize_rotated(&self, codes: &[u8], scales: &[f32], out: &mut [f32]) {
        let nb = self.rotation.num_blocks;
        for b in 0..nb {
            let off = b * BLOCK_SIZE;
            let scale = scales[b];
            for i in 0..BLOCK_SIZE {
                let val = match self.bits {
                    3 => codebook::dequantize_tq3(codes[off + i]),
                    4 => codebook::dequantize_tq4(codes[off + i]),
                    _ => codebook::dequantize_tq8(codes[off + i]),
                };
                out[off + i] = val * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let q = TqQuantizer::new(64, 8);
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.7 - 20.0).collect();
        let enc = q.encode(&x);
        let x_hat = q.decode(&enc);

        let norm: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let error: f64 = x.iter().zip(x_hat.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        assert!(error / norm < 0.2, "relative error too high: {:.4}", error / norm);
    }

    #[test]
    fn test_higher_bits_lower_error() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 1.3 - 40.0) * 0.05).collect();
        let norm: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();

        let mut prev_error = f64::MAX;
        for bits in [3, 4, 8] {
            let q = TqQuantizer::new(64, bits);
            let enc = q.encode(&x);
            let x_hat = q.decode(&enc);
            let error: f64 = x.iter().zip(x_hat.iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt() / norm;
            assert!(error < prev_error, "bits={}: error {:.6} >= prev {:.6}", bits, error, prev_error);
            prev_error = error;
        }
    }
}
