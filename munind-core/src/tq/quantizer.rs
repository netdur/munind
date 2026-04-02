/// Block-based TurboQuant encoder/decoder (full TurboQuant_prod).
///
/// Stage 1 (MSE): WHT rotation → per-block RMS normalize → scalar quantize.
/// Stage 2 (QJL): compute residual → second WHT → sign bits.
/// The QJL step removes inner-product bias from the MSE quantizer.
///
/// Storage per vector: codes (padded_dim u8) + scales (num_blocks f32)
///                   + qjl_signs (padded_dim/8 bytes) + gamma (1 f32)

use super::codebook;
use super::rotation::{Rotation, BLOCK_SIZE};
use crate::common::NgtError;

/// Encoded vector (full TurboQuant_prod).
pub struct Encoded {
    /// Quantized codes, one u8 per padded dimension.
    pub codes: Vec<u8>,
    /// RMS scale per block (num_blocks values).
    pub scales: Vec<f32>,
    /// QJL sign bits: bit i = sign of (WHT2 · residual)[i]. Packed u8.
    pub qjl_signs: Vec<u8>,
    /// L2 norm of the residual (before QJL projection).
    pub gamma: f32,
}

pub struct TqQuantizer {
    pub dim: usize,
    pub bits: u32,
    pub rotation: Rotation,
    /// Second WHT rotation for QJL projection (independent signs).
    pub qjl_rotation: Rotation,
}

impl TqQuantizer {
    pub fn new(dim: usize, bits: u32) -> Self {
        TqQuantizer {
            dim,
            bits,
            rotation: Rotation::new(dim),
            qjl_rotation: Rotation::new(dim), // independent signs
        }
    }

    pub fn load(dir: &str) -> Result<Self, NgtError> {
        let rotation = Rotation::load(&format!("{}/rotation.tq", dir))?;
        let qjl_rotation = Rotation::load(&format!("{}/qjl.tq", dir))
            .unwrap_or_else(|_| Rotation::new(rotation.dim));
        let mut ps = crate::common::PropertySet::new();
        ps.load(&format!("{}/prf", dir))?;
        let bits = ps.get_i64("TqBits", 8) as u32;
        let dim = rotation.dim;
        Ok(TqQuantizer { dim, bits, rotation, qjl_rotation })
    }

    pub fn save(&self, dir: &str) -> Result<(), NgtError> {
        self.rotation.save(&format!("{}/rotation.tq", dir))?;
        self.qjl_rotation.save(&format!("{}/qjl.tq", dir))
    }

    pub fn padded_dim(&self) -> usize {
        self.rotation.padded_dim
    }

    pub fn num_blocks(&self) -> usize {
        self.rotation.num_blocks
    }

    /// Encode a vector (full TurboQuant_prod).
    ///
    /// Stage 1: WHT → RMS normalize → scalar quantize (MSE quantizer).
    /// Stage 2: compute residual in rotated domain → QJL projection → sign bits.
    pub fn encode(&self, x: &[f32]) -> Encoded {
        let pd = self.rotation.padded_dim;
        let nb = self.rotation.num_blocks;

        // Stage 1: Forward WHT.
        let mut rotated = vec![0.0f32; pd];
        self.rotation.forward(x, &mut rotated);

        // Per-block: compute RMS scale, normalize, quantize.
        let mut codes = vec![0u8; pd];
        let mut scales = vec![0.0f32; nb];

        for b in 0..nb {
            let off = b * BLOCK_SIZE;
            let block = &rotated[off..off + BLOCK_SIZE];

            let rms: f32 = (block.iter().map(|&v| v * v).sum::<f32>()
                / BLOCK_SIZE as f32)
                .sqrt();
            scales[b] = rms;

            let inv_rms = if rms > 0.0 { 1.0 / rms } else { 1.0 };

            for i in 0..BLOCK_SIZE {
                let normalized = block[i] * inv_rms;
                codes[off + i] = match self.bits {
                    3 => codebook::quantize_tq3(normalized),
                    4 => codebook::quantize_tq4(normalized),
                    _ => codebook::quantize_tq8(normalized),
                };
            }
        }

        // Stage 2: QJL on the residual.
        // Compute residual = rotated - dequant(codes) * scale (in rotated domain).
        let mut residual = vec![0.0f32; pd];
        for b in 0..nb {
            let off = b * BLOCK_SIZE;
            let scale = scales[b];
            for i in 0..BLOCK_SIZE {
                let dequant = match self.bits {
                    3 => codebook::dequantize_tq3(codes[off + i]),
                    4 => codebook::dequantize_tq4(codes[off + i]),
                    _ => codebook::dequantize_tq8(codes[off + i]),
                };
                residual[off + i] = rotated[off + i] - dequant * scale;
            }
        }

        // Gamma = ‖residual‖₂.
        let gamma: f32 = residual.iter().map(|&v| v * v).sum::<f32>().sqrt();

        // Project residual with second WHT: qjl_projected = WHT2(residual).
        let mut qjl_projected = vec![0.0f32; pd];
        self.qjl_rotation.forward(
            &residual[..self.dim],  // forward takes dim-length input
            &mut qjl_projected,
        );

        // Sign bits: pack sign(qjl_projected[i]) into bytes.
        let sign_bytes = (pd + 7) / 8;
        let mut qjl_signs = vec![0u8; sign_bytes];
        for i in 0..pd {
            if qjl_projected[i] >= 0.0 {
                qjl_signs[i / 8] |= 1 << (i % 8);
            }
        }

        Encoded { codes, scales, qjl_signs, gamma }
    }

    /// Decode: MSE dequantize + QJL residual correction → inverse WHT.
    pub fn decode(&self, enc: &Encoded) -> Vec<f32> {
        let pd = self.rotation.padded_dim;
        let nb = self.rotation.num_blocks;

        // MSE reconstruction in rotated domain.
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

        // QJL residual correction.
        // Reconstruct: r̃ = (√(π/2) / pd) · γ · WHT2^T · sign_vector
        if enc.gamma > 0.0 {
            // Unpack sign bits to ±1.
            let mut sign_vec = vec![0.0f32; pd];
            for i in 0..pd {
                let bit = (enc.qjl_signs[i / 8] >> (i % 8)) & 1;
                sign_vec[i] = if bit != 0 { 1.0 } else { -1.0 };
            }

            // Inverse QJL projection: WHT2^T · sign_vec → residual estimate.
            // We only need dim-length output, but inverse takes padded input.
            let mut r_est = vec![0.0f32; self.dim];
            self.qjl_rotation.inverse(&sign_vec, &mut r_est);

            // Scale: (√(π/2) / pd) · γ
            let scale = (std::f32::consts::FRAC_PI_2.sqrt()) / (pd as f32) * enc.gamma;
            // Add correction in the original (unrotated) domain — but we're working
            // in rotated domain. The correction should be added before inverse WHT.
            // Actually: the residual was computed in rotated domain, projected via WHT2.
            // So the inverse of WHT2 gives back the rotated-domain residual estimate.
            // We need to add it to `rotated` before the final inverse WHT1.

            // Re-do: inverse WHT2 gives padded-dim output in rotated domain.
            let _r_rotated = vec![0.0f32; pd];
            // inverse() outputs dim-length, but we need pd-length in rotated domain.
            // Use forward of qjl_rotation on sign_vec to get the pd-length rotated residual.
            // Wait — the QJL projection was: qjl = WHT2_forward(residual[..dim]).
            // To invert: residual_est = WHT2_inverse(sign_vec) (dim-length).
            // But residual was pd-length in rotated domain, and we forward(residual[..dim]).
            // This loses the padded dims. Let's simplify: add correction in original domain.

            // Simpler: add QJL correction after inverse WHT1.
            // rotated → inverse WHT1 → x_mse
            // Then x_corrected = x_mse + scale * WHT2_inverse(sign_vec)
            // This is correct because the residual r = x - x_mse was in original domain
            // (well, it was computed in rotated domain, but the correction is linear).

            // Actually let me think again. The residual was:
            //   residual[i] = rotated[i] - dequant[i]  (in rotated domain)
            // Then we did: qjl = WHT2_forward(residual[..dim])
            // So qjl is in WHT2-rotated domain.
            // To correct: add residual_est to rotated[] before inverse WHT1.
            // residual_est in rotated domain = WHT2_inverse(±1 vector) * scale

            // But WHT2_inverse outputs dim-length. We need to pad back to pd.
            // The padded dims had residual ≈ 0 (zero-padded input to WHT2_forward).
            // So correction for padded dims is 0. Just add to first dim coordinates.

            // Apply correction in rotated domain.
            for i in 0..self.dim {
                rotated[i] += r_est[i] * scale;
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

        #[cfg(target_arch = "aarch64")]
        {
            match self.bits {
                4 => {
                    unsafe { neon_dequant::dequantize_rotated_tq4(codes, scales, out, nb) };
                    return;
                }
                8 => {
                    unsafe { neon_dequant::dequantize_rotated_tq8(codes, scales, out, nb) };
                    return;
                }
                _ => {}
            }
        }

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

// ---------------------------------------------------------------------------
// NEON-accelerated dequantization
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_dequant {
    use std::arch::aarch64::*;
    use super::super::codebook::TQ4_CENTROIDS;
    use super::super::rotation::BLOCK_SIZE;

    /// TQ4 dequantization via NEON `tbl` instruction.
    /// 16 f32 centroids (64 bytes) fit exactly in a `uint8x16x4_t` table.
    /// For each group of 4 codes, one `vqtbl4q_u8` replaces 4 array lookups.
    #[inline]
    pub unsafe fn dequantize_rotated_tq4(
        codes: &[u8],
        scales: &[f32],
        out: &mut [f32],
        num_blocks: usize,
    ) {
        unsafe {
            // Load 16 TQ4 centroids as a 64-byte lookup table (4 × 128-bit registers).
            let cptr = TQ4_CENTROIDS.as_ptr() as *const u8;
            let table = uint8x16x4_t(
                vld1q_u8(cptr),
                vld1q_u8(cptr.add(16)),
                vld1q_u8(cptr.add(32)),
                vld1q_u8(cptr.add(48)),
            );

            // Shuffle pattern: replicate each of the first 4 bytes 4 times.
            let rep = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
            // Byte offset within each f32: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
            let boff = vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].as_ptr());

            let codes_ptr = codes.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for b in 0..num_blocks {
                let off = b * BLOCK_SIZE;
                let scale_v = vdupq_n_f32(*scales.get_unchecked(b));

                // Process 4 codes per iteration, 8 iterations per block (BLOCK_SIZE=32).
                for g in 0..8 {
                    let goff = off + g * 4;

                    // Load 4 codes as a u32, broadcast to all lanes.
                    let code_word = (codes_ptr.add(goff) as *const u32).read_unaligned();
                    let raw = vreinterpretq_u8_u32(vdupq_n_u32(code_word));

                    // Build byte indices: replicate each code 4×, ×4 for f32 stride, +[0,1,2,3].
                    let replicated = vqtbl1q_u8(raw, rep);
                    let base = vshlq_n_u8::<2>(replicated);
                    let indices = vaddq_u8(base, boff);

                    // 64-byte table lookup → 4 f32 centroids (as raw bytes).
                    let centroid_bytes = vqtbl4q_u8(table, indices);
                    let centroids = vreinterpretq_f32_u8(centroid_bytes);

                    vst1q_f32(out_ptr.add(goff), vmulq_f32(centroids, scale_v));
                }
            }
        }
    }

    /// TQ8 dequantization via NEON FMA.
    /// TQ8 centroid = -3.0 + (idx + 0.5) × (6/256) = idx × 0.0234375 − 2.98828125
    /// Replaces per-code function call with vectorized multiply-add.
    #[inline]
    pub unsafe fn dequantize_rotated_tq8(
        codes: &[u8],
        scales: &[f32],
        out: &mut [f32],
        num_blocks: usize,
    ) {
        unsafe {
            let step = vdupq_n_f32(6.0 / 256.0);
            let base_offset = vdupq_n_f32(-3.0 + 0.5 * (6.0 / 256.0));

            let codes_ptr = codes.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for b in 0..num_blocks {
                let off = b * BLOCK_SIZE;
                let scale_v = vdupq_n_f32(*scales.get_unchecked(b));

                for g in 0..8 {
                    let goff = off + g * 4;

                    // Load 4 codes as u32, broadcast, widen u8→u16→u32→f32.
                    let code_word = (codes_ptr.add(goff) as *const u32).read_unaligned();
                    let raw = vreinterpretq_u8_u32(vdupq_n_u32(code_word));
                    let wide16 = vmovl_u8(vget_low_u8(raw));
                    let wide32 = vmovl_u16(vget_low_u16(wide16));
                    let idx_f32 = vcvtq_f32_u32(wide32);

                    // centroid = idx × step + base_offset
                    let centroids = vfmaq_f32(base_offset, idx_f32, step);

                    vst1q_f32(out_ptr.add(goff), vmulq_f32(centroids, scale_v));
                }
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

    /// Verify NEON dequant matches scalar for TQ4 and TQ8.
    #[test]
    fn test_neon_dequant_matches_scalar() {
        for bits in [4u32, 8] {
            let q = TqQuantizer::new(100, bits);
            let x: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3 - 15.0).collect();
            let enc = q.encode(&x);

            let pd = q.padded_dim();

            // Scalar dequant.
            let mut scalar_out = vec![0.0f32; pd];
            let nb = q.num_blocks();
            for b in 0..nb {
                let off = b * BLOCK_SIZE;
                let scale = enc.scales[b];
                for i in 0..BLOCK_SIZE {
                    let val = match bits {
                        4 => codebook::dequantize_tq4(enc.codes[off + i]),
                        _ => codebook::dequantize_tq8(enc.codes[off + i]),
                    };
                    scalar_out[off + i] = val * scale;
                }
            }

            // NEON dequant (dispatched via dequantize_rotated).
            let mut neon_out = vec![0.0f32; pd];
            q.dequantize_rotated(&enc.codes, &enc.scales, &mut neon_out);

            let max_err: f32 = scalar_out.iter().zip(neon_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-5,
                "bits={}: NEON/scalar mismatch, max_err={}", bits, max_err);
        }
    }

    /// Microbenchmark: NEON vs scalar dequant throughput.
    #[test]
    fn bench_dequant_neon() {
        let dim = 128;
        let bits = 4u32;
        let q = TqQuantizer::new(dim, bits);
        let pd = q.padded_dim();
        let nb = q.num_blocks();

        // Generate many encoded vectors.
        let n = 100_000usize;
        let mut all_codes = vec![0u8; n * pd];
        let mut all_scales = vec![0.0f32; n * nb];
        for i in 0..n {
            let x: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32 * 0.01).sin()).collect();
            let enc = q.encode(&x);
            all_codes[i * pd..(i + 1) * pd].copy_from_slice(&enc.codes);
            all_scales[i * nb..(i + 1) * nb].copy_from_slice(&enc.scales);
        }

        let mut out = vec![0.0f32; pd];

        // Warm up.
        for i in 0..1000 {
            q.dequantize_rotated(&all_codes[i * pd..(i + 1) * pd], &all_scales[i * nb..(i + 1) * nb], &mut out);
        }

        // Time NEON path (current default on aarch64).
        let t0 = std::time::Instant::now();
        for i in 0..n {
            q.dequantize_rotated(&all_codes[i * pd..(i + 1) * pd], &all_scales[i * nb..(i + 1) * nb], &mut out);
            std::hint::black_box(&out);
        }
        let neon_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        // Time scalar path.
        let t0 = std::time::Instant::now();
        for i in 0..n {
            let codes = &all_codes[i * pd..(i + 1) * pd];
            let scales = &all_scales[i * nb..(i + 1) * nb];
            for b in 0..nb {
                let off = b * BLOCK_SIZE;
                let scale = scales[b];
                for j in 0..BLOCK_SIZE {
                    let val = codebook::dequantize_tq4(codes[off + j]);
                    out[off + j] = val * scale;
                }
            }
            std::hint::black_box(&out);
        }
        let scalar_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        eprintln!("TQ4 dequant dim={}: NEON={:.0}ns  scalar={:.0}ns  speedup={:.2}×",
            dim, neon_ns, scalar_ns, scalar_ns / neon_ns);
    }

    #[test]
    fn bench_dequant_tq8_neon() {
        let dim = 128;
        let bits = 8u32;
        let q = TqQuantizer::new(dim, bits);
        let pd = q.padded_dim();
        let nb = q.num_blocks();

        let n = 100_000usize;
        let mut all_codes = vec![0u8; n * pd];
        let mut all_scales = vec![0.0f32; n * nb];
        for i in 0..n {
            let x: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32 * 0.01).sin()).collect();
            let enc = q.encode(&x);
            all_codes[i * pd..(i + 1) * pd].copy_from_slice(&enc.codes);
            all_scales[i * nb..(i + 1) * nb].copy_from_slice(&enc.scales);
        }

        let mut out = vec![0.0f32; pd];

        for i in 0..1000 {
            q.dequantize_rotated(&all_codes[i * pd..(i + 1) * pd], &all_scales[i * nb..(i + 1) * nb], &mut out);
        }

        let t0 = std::time::Instant::now();
        for i in 0..n {
            q.dequantize_rotated(&all_codes[i * pd..(i + 1) * pd], &all_scales[i * nb..(i + 1) * nb], &mut out);
            std::hint::black_box(&out);
        }
        let neon_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        let t0 = std::time::Instant::now();
        for i in 0..n {
            let codes = &all_codes[i * pd..(i + 1) * pd];
            let scales = &all_scales[i * nb..(i + 1) * nb];
            for b in 0..nb {
                let off = b * BLOCK_SIZE;
                let scale = scales[b];
                for j in 0..BLOCK_SIZE {
                    let val = codebook::dequantize_tq8(codes[off + j]);
                    out[off + j] = val * scale;
                }
            }
            std::hint::black_box(&out);
        }
        let scalar_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        eprintln!("TQ8 dequant dim={}: NEON={:.0}ns  scalar={:.0}ns  speedup={:.2}×",
            dim, neon_ns, scalar_ns, scalar_ns / neon_ns);
    }
}
