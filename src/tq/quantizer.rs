/// TurboQuant encoder/decoder.
///
/// TurboQuant_mse: rotate → scalar quantize each coordinate → store codes + norm.
/// TurboQuant_prod: TurboQuant_mse + 1-bit QJL on residual for unbiased inner product.

use crate::common::NgtError;
use super::codebook::ScalarCodebook;
use super::rotation::RotationMatrix;

/// Encoded vector (TurboQuant_mse).
pub struct EncodedMse {
    /// Quantized coordinate indices, one per dimension.  Each in [0, 2^bits).
    pub codes: Vec<u32>,
    /// Original L2 norm (preserved separately).
    pub norm: f32,
}

/// Encoded vector (TurboQuant_prod = mse + QJL residual).
pub struct EncodedProd {
    pub mse: EncodedMse,
    /// Sign bits of S·residual.  Bit i of byte i/8 = sign of (S·r)[i].
    pub qjl_signs: Vec<u8>,
    /// L2 norm of residual.
    pub gamma: f32,
}

pub struct TqQuantizer {
    pub dim: usize,
    pub bits: u32,
    pub rotation: RotationMatrix,
    pub codebook: ScalarCodebook,
    /// Random Gaussian matrix for QJL (d×d, row-major).  None if mse-only.
    pub qjl_matrix: Option<Vec<f32>>,
}

impl TqQuantizer {
    /// Create a new quantizer.  If `use_prod` is true, also generates the QJL matrix.
    pub fn new(dim: usize, bits: u32, use_prod: bool) -> Self {
        let rotation = RotationMatrix::random(dim);
        // Build codebook for padded_dim (WHT operates at power-of-2 size).
        let codebook = ScalarCodebook::build(rotation.padded_dim(), bits);
        let qjl_matrix = if use_prod {
            let mut rng = rand::thread_rng();
            let mut m = vec![0.0f32; dim * dim];
            for v in m.iter_mut() {
                let u1: f64 = rand::Rng::gen_range(&mut rng, 1e-10..1.0);
                let u2: f64 = rand::Rng::gen_range(&mut rng, 0.0..std::f64::consts::TAU);
                *v = ((-2.0 * u1.ln()).sqrt() * u2.cos()) as f32;
            }
            Some(m)
        } else {
            None
        };

        TqQuantizer {
            dim,
            bits,
            rotation,
            codebook,
            qjl_matrix,
        }
    }

    /// Load a quantizer from saved files.
    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let rotation = RotationMatrix::load(&format!("{}/rotation.tq", dir))?;
        let codebook = ScalarCodebook::load(&format!("{}/codebook.tq", dir))?;
        let dim = rotation.dim;
        let bits = codebook.bits;

        let qjl_path = format!("{}/qjl.tq", dir);
        let qjl_matrix = if std::path::Path::new(&qjl_path).exists() {
            let f = std::fs::File::open(&qjl_path)
                .map_err(|e| format!("TqQuantizer::load qjl: {}", e))?;
            let mut r = std::io::BufReader::new(f);
            let mut buf8 = [0u8; 8];
            std::io::Read::read_exact(&mut r, &mut buf8)
                .map_err(|e| format!("{}", e))?;
            let d = u64::from_le_bytes(buf8) as usize;
            assert_eq!(d, dim);
            let mut data = vec![0.0f32; d * d];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
            };
            std::io::Read::read_exact(&mut r, bytes)
                .map_err(|e| format!("{}", e))?;
            Some(data)
        } else {
            None
        };

        Ok(TqQuantizer {
            dim,
            bits,
            rotation,
            codebook,
            qjl_matrix,
        })
    }

    /// Save quantizer state to directory.
    pub fn save(&self, dir: &str) -> Result<(), NgtError> {
        self.rotation.save(&format!("{}/rotation.tq", dir))?;
        self.codebook.save(&format!("{}/codebook.tq", dir))?;

        if let Some(ref m) = self.qjl_matrix {
            let f = std::fs::File::create(&format!("{}/qjl.tq", dir))
                .map_err(|e| format!("save qjl: {}", e))?;
            let mut w = std::io::BufWriter::new(f);
            use std::io::Write;
            w.write_all(&(self.dim as u64).to_le_bytes())
                .map_err(|e| format!("{}", e))?;
            let bytes = unsafe {
                std::slice::from_raw_parts(m.as_ptr() as *const u8, m.len() * 4)
            };
            w.write_all(bytes).map_err(|e| format!("{}", e))?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // TurboQuant_mse: encode / decode
    // -----------------------------------------------------------------------

    /// The padded dimension used by the WHT rotation.
    pub fn padded_dim(&self) -> usize {
        self.rotation.padded_dim()
    }

    /// Encode a vector using TurboQuant_mse.
    /// Returns codes of length `padded_dim` (WHT operates on padded size).
    pub fn encode_mse(&self, x: &[f32]) -> EncodedMse {
        debug_assert_eq!(x.len(), self.dim);
        let pd = self.rotation.padded_dim();

        // Compute norm and normalize.
        let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 1.0 };

        let mut unit = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            unit[i] = x[i] * inv_norm;
        }

        // Rotate: y = WHT(D · x), output has padded_dim coordinates.
        let mut y = vec![0.0f32; pd];
        self.rotation.mul(&unit, &mut y);

        // Scalar quantize each coordinate (all padded_dim of them).
        let codes: Vec<u32> = y.iter().map(|&v| self.codebook.quantize(v)).collect();

        EncodedMse { codes, norm }
    }

    /// Decode a TurboQuant_mse encoded vector.
    pub fn decode_mse(&self, enc: &EncodedMse) -> Vec<f32> {
        let pd = enc.codes.len();

        // Dequantize each coordinate (padded_dim).
        let y: Vec<f32> = enc.codes.iter().map(|&c| self.codebook.dequantize(c)).collect();

        // Inverse rotate: x̃ = norm · Π^T · ỹ → output has original dim.
        let mut out = vec![0.0f32; self.dim];
        self.rotation.mul_transpose(&y, &mut out);

        for v in out.iter_mut() {
            *v *= enc.norm;
        }
        out
    }

    /// Decode in-place into a provided buffer (avoids allocation in hot path).
    #[inline]
    pub fn decode_mse_into(&self, codes: &[u32], norm: f32, out: &mut [f32]) {
        let pd = codes.len();
        debug_assert_eq!(out.len(), self.dim);

        let mut y = vec![0.0f32; pd];
        for i in 0..pd {
            y[i] = self.codebook.dequantize(codes[i]);
        }

        self.rotation.mul_transpose(&y, out);

        for v in out.iter_mut() {
            *v *= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip_low_error() {
        let dim = 64;
        let bits = 4;
        let q = TqQuantizer::new(dim, bits, false);

        let x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.7 - 20.0) * 0.1).collect();
        let enc = q.encode_mse(&x);
        let x_hat = q.decode_mse(&enc);

        // Compute relative error.
        let norm_x: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let error: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative = error / norm_x;

        // 4-bit quantization should give < 20% relative error.
        assert!(
            relative < 0.2,
            "relative error too high: {:.4}",
            relative
        );
    }

    #[test]
    fn test_norm_preserved() {
        let dim = 32;
        let q = TqQuantizer::new(dim, 4, false);
        let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let enc = q.encode_mse(&x);

        let norm_x: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (enc.norm - norm_x).abs() < 1e-5,
            "norm mismatch: {} vs {}",
            enc.norm, norm_x
        );
    }

    #[test]
    fn test_higher_bits_lower_error() {
        let dim = 64;
        let x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 1.3 - 40.0) * 0.05).collect();
        let norm_x: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();

        let mut prev_error = f64::MAX;
        for bits in 1..=4 {
            let q = TqQuantizer::new(dim, bits, false);
            let enc = q.encode_mse(&x);
            let x_hat = q.decode_mse(&enc);
            let error: f64 = x
                .iter()
                .zip(x_hat.iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm_x;

            assert!(
                error < prev_error,
                "bits={}: error {:.6} >= prev {:.6}",
                bits, error, prev_error
            );
            prev_error = error;
        }
    }
}
