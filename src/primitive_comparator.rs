pub struct PrimitiveComparator;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

impl PrimitiveComparator {
    pub fn compare_l2_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            let limit = a.len() & !3; // Multiple of 4

            while i < limit {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vdiff = vsubq_f32(va, vb);
                sum_vec = vfmaq_f32(sum_vec, vdiff, vdiff);
                i += 4;
            }

            let mut d = vaddvq_f32(sum_vec);
            while i < a.len() {
                let diff = a[i] - b[i];
                d += diff * diff;
                i += 1;
            }
            d.sqrt()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum::<f32>()
                .sqrt()
        }
    }

    pub fn compare_l1_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
    }

    pub fn compare_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            let limit = a.len() & !3;

            while i < limit {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum_vec = vfmaq_f32(sum_vec, va, vb);
                i += 4;
            }

            let mut sum = vaddvq_f32(sum_vec);
            while i < a.len() {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
        }
    }

    pub fn compare_cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut norm_a_vec = vdupq_n_f32(0.0);
            let mut norm_b_vec = vdupq_n_f32(0.0);
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            let limit = a.len() & !3;

            while i < limit {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                norm_a_vec = vfmaq_f32(norm_a_vec, va, va);
                norm_b_vec = vfmaq_f32(norm_b_vec, vb, vb);
                sum_vec = vfmaq_f32(sum_vec, va, vb);
                i += 4;
            }

            let mut norm_a = vaddvq_f32(norm_a_vec);
            let mut norm_b = vaddvq_f32(norm_b_vec);
            let mut sum = vaddvq_f32(sum_vec);

            while i < a.len() {
                let x = a[i];
                let y = b[i];
                norm_a += x * x;
                norm_b += y * y;
                sum += x * y;
                i += 1;
            }

            sum / (norm_a * norm_b).sqrt()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (norm_a, norm_b, sum) = a
                .iter()
                .zip(b.iter())
                .fold((0.0f32, 0.0f32, 0.0f32), |(na, nb, s), (&x, &y)| {
                    (na + x * x, nb + y * y, s + x * y)
                });
            sum / (norm_a * norm_b).sqrt()
        }
    }

    pub fn compare_normalized_l2_f32(a: &[f32], b: &[f32]) -> f32 {
        let v = 2.0 - 2.0 * Self::compare_dot_product_f32(a, b);
        if v < 0.0 { 0.0 } else { v.sqrt() }
    }

    pub fn compare_normalized_cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        let v = 1.0 - Self::compare_dot_product_f32(a, b);
        if v < 0.0 { -v } else { v }
    }

    pub fn compare_angle_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        let cosine = Self::compare_cosine_f32(a, b);
        if cosine >= 1.0 {
            0.0
        } else if cosine <= -1.0 {
            std::f32::consts::PI
        } else {
            cosine.acos()
        }
    }

    pub fn compare_normalized_angle_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        let cosine = Self::compare_dot_product_f32(a, b);
        if cosine >= 1.0 {
            0.0
        } else if cosine <= -1.0 {
            std::f32::consts::PI
        } else {
            cosine.acos()
        }
    }
}
