/// Max-Lloyd optimal scalar quantizer for the Beta distribution that arises
/// after random rotation of unit-sphere vectors.
///
/// After rotating x ∈ S^{d-1} by a random orthogonal Π, each coordinate
/// follows Beta((d-1)/2, (d-1)/2) on [-1, 1], which in high dimensions
/// concentrates around 0 with scale ~1/√d.
///
/// TurboQuant precomputes optimal centroids for this distribution using the
/// Max-Lloyd algorithm, then uses a simple nearest-centroid lookup per
/// coordinate.

use crate::common::NgtError;
use std::io::{Read, Write};

/// Precomputed scalar codebook for b-bit quantization.
pub struct ScalarCodebook {
    /// Bits per dimension.
    pub bits: u32,
    /// Number of levels = 2^bits.
    pub num_levels: usize,
    /// Centroid values, sorted ascending.  Length = num_levels.
    pub centroids: Vec<f32>,
    /// Decision boundaries (midpoints between adjacent centroids).
    /// Length = num_levels - 1.
    pub boundaries: Vec<f32>,
}

impl ScalarCodebook {
    /// Build the codebook for dimension `d` and `bits` per coordinate.
    ///
    /// Uses the Max-Lloyd algorithm on the Beta((d-1)/2, (d-1)/2)
    /// distribution.  For high d, this is well-approximated by N(0, 1/d).
    pub fn build(dim: usize, bits: u32) -> Self {
        let num_levels = 1usize << bits;
        let sigma = 1.0 / (dim as f64).sqrt();

        // Initialize centroids uniformly in [-3σ, 3σ].
        let lo = -3.0 * sigma;
        let hi = 3.0 * sigma;
        let mut centroids: Vec<f64> = (0..num_levels)
            .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / num_levels as f64)
            .collect();

        // Max-Lloyd iterations (converges fast for Gaussian-like distributions).
        for _ in 0..100 {
            // Compute boundaries (midpoints).
            let mut bounds: Vec<f64> = Vec::with_capacity(num_levels - 1);
            for i in 0..num_levels - 1 {
                bounds.push((centroids[i] + centroids[i + 1]) / 2.0);
            }

            // Update centroids: E[X | boundary_left < X < boundary_right]
            // under N(0, σ²).
            let mut new_centroids = vec![0.0f64; num_levels];
            let n_samples = 10000;
            let step = 6.0 * sigma / n_samples as f64;
            for i in 0..num_levels {
                let left = if i == 0 { lo - sigma } else { bounds[i - 1] };
                let right = if i == num_levels - 1 { hi + sigma } else { bounds[i] };
                let mut sum_xw = 0.0f64;
                let mut sum_w = 0.0f64;
                let mut x = left;
                while x < right {
                    let w = (-x * x / (2.0 * sigma * sigma)).exp();
                    sum_xw += x * w;
                    sum_w += w;
                    x += step;
                }
                new_centroids[i] = if sum_w > 0.0 { sum_xw / sum_w } else { centroids[i] };
            }
            centroids = new_centroids;
        }

        let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();
        let boundaries: Vec<f32> = (0..num_levels - 1)
            .map(|i| (centroids_f32[i] + centroids_f32[i + 1]) / 2.0)
            .collect();

        ScalarCodebook {
            bits,
            num_levels,
            centroids: centroids_f32,
            boundaries,
        }
    }

    /// Quantize a single scalar value.  Returns the centroid index (0-based).
    #[inline]
    pub fn quantize(&self, value: f32) -> u32 {
        // Binary search on boundaries.
        let mut lo = 0usize;
        let mut hi = self.boundaries.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if value > self.boundaries[mid] {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as u32
    }

    /// Dequantize: return the centroid value for the given index.
    #[inline]
    pub fn dequantize(&self, index: u32) -> f32 {
        self.centroids[index as usize]
    }

    /// Serialize to file.
    pub fn save(&self, path: &str) -> Result<(), NgtError> {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("ScalarCodebook::save: {}: {}", path, e))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&self.bits.to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        w.write_all(&(self.num_levels as u32).to_le_bytes())
            .map_err(|e| format!("{}", e))?;
        for &c in &self.centroids {
            w.write_all(&c.to_le_bytes()).map_err(|e| format!("{}", e))?;
        }
        Ok(())
    }

    /// Deserialize from file.
    pub fn load(path: &str) -> Result<Self, NgtError> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("ScalarCodebook::load: {}: {}", path, e))?;
        let mut r = std::io::BufReader::new(f);
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let bits = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
        let num_levels = u32::from_le_bytes(buf4) as usize;
        let mut centroids = vec![0.0f32; num_levels];
        for c in centroids.iter_mut() {
            r.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
            *c = f32::from_le_bytes(buf4);
        }
        let boundaries: Vec<f32> = (0..num_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();
        Ok(ScalarCodebook {
            bits,
            num_levels,
            centroids,
            boundaries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_1bit() {
        let cb = ScalarCodebook::build(100, 1);
        assert_eq!(cb.num_levels, 2);

        // Negative value → index 0, positive → index 1.
        assert_eq!(cb.quantize(-0.05), 0);
        assert_eq!(cb.quantize(0.05), 1);

        // Dequantize gives centroid values.
        let c0 = cb.dequantize(0);
        let c1 = cb.dequantize(1);
        assert!(c0 < 0.0);
        assert!(c1 > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_4bit() {
        let cb = ScalarCodebook::build(100, 4);
        assert_eq!(cb.num_levels, 16);

        // Centroids should be sorted.
        for i in 1..cb.num_levels {
            assert!(cb.centroids[i] > cb.centroids[i - 1]);
        }

        // Quantize near a centroid should return that centroid's index.
        for (i, &c) in cb.centroids.iter().enumerate() {
            assert_eq!(cb.quantize(c), i as u32);
        }
    }

    #[test]
    fn test_distortion_decreases_with_bits() {
        // Higher bits should give lower quantization error.
        let dim = 100;
        let sigma = 1.0 / (dim as f32).sqrt();
        let test_vals: Vec<f32> = (-50..=50).map(|i| i as f32 * sigma * 0.06).collect();

        let mut prev_error = f64::MAX;
        for bits in 1..=4 {
            let cb = ScalarCodebook::build(dim, bits);
            let error: f64 = test_vals
                .iter()
                .map(|&v| {
                    let q = cb.dequantize(cb.quantize(v));
                    ((v - q) as f64).powi(2)
                })
                .sum::<f64>()
                / test_vals.len() as f64;

            assert!(error < prev_error, "bits={}: error {} >= prev {}", bits, error, prev_error);
            prev_error = error;
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = "./target/test_codebook";
        std::fs::create_dir_all(dir).unwrap();
        let path = format!("{}/cb.bin", dir);

        let cb = ScalarCodebook::build(100, 3);
        cb.save(&path).unwrap();
        let cb2 = ScalarCodebook::load(&path).unwrap();

        assert_eq!(cb.bits, cb2.bits);
        assert_eq!(cb.centroids, cb2.centroids);
    }
}
