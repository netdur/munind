/// Port of NGT/PrimitiveComparatorNoArch.h + NGT/PrimitiveComparator.h
///
/// Scalar baseline + NEON SIMD (aarch64) for hot-path distance functions.
/// The C++ template functions use a `double` accumulator internally; the SIMD
/// paths use `f32` NEON lanes for throughput and promote to `f64` only for
/// the final reduction — this is sufficient precision for ANN search.

// ---------------------------------------------------------------------------
// DistanceType  (NGT::ObjectSpace::DistanceType)
// ---------------------------------------------------------------------------

/// Distance metric.  Numeric values match the C++ enum so that the
/// `PropertySet` text representation round-trips correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DistanceType {
    None                      = -1,
    L1                        = 0,
    L2                        = 1,
    Hamming                   = 2,
    Angle                     = 3,
    CosineSimilarity          = 4,
    NormalizedAngle           = 5,
    NormalizedCosineSimilarity = 6,
    Jaccard                   = 7,
    SparseJaccard             = 8,
    NormalizedL2              = 9,
    InnerProduct              = 10,
    DotProduct                = 11,
    Poincare                  = 100,
    Lorentz                   = 101,
}

impl DistanceType {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1  => Some(Self::None),
            0   => Some(Self::L1),
            1   => Some(Self::L2),
            2   => Some(Self::Hamming),
            3   => Some(Self::Angle),
            4   => Some(Self::CosineSimilarity),
            5   => Some(Self::NormalizedAngle),
            6   => Some(Self::NormalizedCosineSimilarity),
            7   => Some(Self::Jaccard),
            8   => Some(Self::SparseJaccard),
            9   => Some(Self::NormalizedL2),
            100 => Some(Self::Poincare),
            101 => Some(Self::Lorentz),
            102 => Some(Self::InnerProduct),
            _   => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Euclidean norm of a float vector.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32
}

/// Normalize `v` in-place (divide by its L2 norm).
/// If the norm is zero the vector is left unchanged (matches NGT behaviour).
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Whether this distance type requires objects to be stored pre-normalized.
/// Matches the C++ logic in `GraphAndTreeIndex::insert`.
#[inline]
pub fn requires_normalization(dt: DistanceType) -> bool {
    matches!(
        dt,
        DistanceType::CosineSimilarity
            | DistanceType::NormalizedCosineSimilarity
            | DistanceType::Angle
            | DistanceType::NormalizedAngle
            | DistanceType::NormalizedL2
    )
}

// ---------------------------------------------------------------------------
// compareDotProduct  (PrimitiveComparator::compareDotProduct<float>)
// ---------------------------------------------------------------------------

/// `Σ a[i] * b[i]` using `f64` accumulator.
/// Maps to `PrimitiveComparator::compareDotProduct<float>`.
#[inline]
pub fn compare_dot_product(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_product_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f64;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            sum += (ai as f64) * (bi as f64);
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// compareL2  (PrimitiveComparator::compareL2<float, double>)
// ---------------------------------------------------------------------------

/// `sqrt( Σ (a[i] - b[i])² )` with `f64` accumulator, unrolled by 4.
/// Maps to `PrimitiveComparator::compareL2(const float*, const float*, size_t)`.
pub fn compare_l2(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        neon::l2_neon(a, b).sqrt()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let size = a.len();
        let mut d = 0.0f64;
        let mut i = 0usize;
        while i + 4 <= size {
            let diff0 = (a[i]     as f64) - (b[i]     as f64);
            let diff1 = (a[i + 1] as f64) - (b[i + 1] as f64);
            let diff2 = (a[i + 2] as f64) - (b[i + 2] as f64);
            let diff3 = (a[i + 3] as f64) - (b[i + 3] as f64);
            d += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            i += 4;
        }
        while i < size {
            let diff = (a[i] as f64) - (b[i] as f64);
            d += diff * diff;
            i += 1;
        }
        d.sqrt()
    }
}

// ---------------------------------------------------------------------------
// compareNormalizedL2  (PrimitiveComparator::compareNormalizedL2<float>)
// ---------------------------------------------------------------------------

/// For pre-normalized vectors: `sqrt(max(0, 2 - 2·dot(a, b)))`.
/// Using the identity |a-b|² = |a|² + |b|² - 2·dot = 2 - 2·dot when |a|=|b|=1.
/// Maps to `PrimitiveComparator::compareNormalizedL2<float>`.
#[inline]
pub fn compare_normalized_l2(a: &[f32], b: &[f32]) -> f64 {
    let v = 2.0 - 2.0 * compare_dot_product(a, b);
    if v < 0.0 { 0.0 } else { v.sqrt() }
}

// ---------------------------------------------------------------------------
// compareL1  (PrimitiveComparator::compareL1<float, double>)
// ---------------------------------------------------------------------------

/// `Σ |a[i] - b[i]|` with `f64` accumulator, unrolled by 4.
/// Maps to `PrimitiveComparator::compareL1(const float*, const float*, size_t)`.
pub fn compare_l1(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let size = a.len();
    let mut d = 0.0f64;
    let mut i = 0usize;

    while i + 4 <= size {
        d += ((a[i]     as f64) - (b[i]     as f64)).abs();
        d += ((a[i + 1] as f64) - (b[i + 1] as f64)).abs();
        d += ((a[i + 2] as f64) - (b[i + 2] as f64)).abs();
        d += ((a[i + 3] as f64) - (b[i + 3] as f64)).abs();
        i += 4;
    }
    while i < size {
        d += ((a[i] as f64) - (b[i] as f64)).abs();
        i += 1;
    }
    d
}

// ---------------------------------------------------------------------------
// Hamming  (PrimitiveComparator::compareHammingDistance<uint8_t>)
// ---------------------------------------------------------------------------

/// Hamming distance: number of differing bits.
/// Operates on the raw bytes of both slices cast to `u32` words, using
/// popcount on `a XOR b`, exactly as in the C++ template.
/// Maps to `PrimitiveComparator::compareHammingDistance<uint8_t>`.
pub fn compare_hamming(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() % 4 == 0, "Hamming: size must be a multiple of 4 bytes");

    let mut count = 0u64;
    let words_a = bytecast_u32(a);
    let words_b = bytecast_u32(b);
    for (&wa, &wb) in words_a.iter().zip(words_b.iter()) {
        count += (wa ^ wb).count_ones() as u64;
    }
    count as f64
}

// ---------------------------------------------------------------------------
// Jaccard  (PrimitiveComparator::compareJaccardDistance<uint8_t>)
// ---------------------------------------------------------------------------

/// Jaccard distance: `1 - |A∩B| / |A∪B|` over bit-sets, unrolled by 2.
/// Maps to `PrimitiveComparator::compareJaccardDistance<uint8_t>`.
pub fn compare_jaccard(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() % 8 == 0, "Jaccard: size must be a multiple of 8 bytes");

    let mut count = 0u64;    // |A ∩ B|
    let mut count_de = 0u64; // |A ∪ B|
    let words_a = bytecast_u32(a);
    let words_b = bytecast_u32(b);
    let mut i = 0;
    while i + 2 <= words_a.len() {
        count    += (words_a[i]     & words_b[i]    ).count_ones() as u64;
        count_de += (words_a[i]     | words_b[i]    ).count_ones() as u64;
        count    += (words_a[i + 1] & words_b[i + 1]).count_ones() as u64;
        count_de += (words_a[i + 1] | words_b[i + 1]).count_ones() as u64;
        i += 2;
    }
    1.0 - (count as f64) / (count_de as f64)
}

// ---------------------------------------------------------------------------
// SparseJaccard  (PrimitiveComparator::compareSparseJaccardDistance<float>)
// ---------------------------------------------------------------------------

/// Sparse Jaccard on a sorted, zero-terminated list of IDs packed as `f32`.
/// Maps to `PrimitiveComparator::compareSparseJaccardDistance(const float*, ...)`.
pub fn compare_sparse_jaccard(a: &[f32], b: &[f32]) -> f64 {
    let ai = a.as_ptr() as *const u32;
    let bi = b.as_ptr() as *const u32;
    let size = b.len();

    let mut loca = 0usize;
    let mut locb = 0usize;
    let mut count = 0i64;

    unsafe {
        while locb < size && *ai.add(loca) != 0 && *bi.add(locb) != 0 {
            let sub = (*ai.add(loca) as i64) - (*bi.add(locb) as i64);
            if sub == 0 { count += 1; }
            if sub <= 0 { loca += 1; }
            if sub >= 0 { locb += 1; }
        }
        while *ai.add(loca) != 0 { loca += 1; }
        while locb < size && *bi.add(locb) != 0 { locb += 1; }
    }

    1.0 - (count as f64) / ((loca + locb - count as usize) as f64)
}

// ---------------------------------------------------------------------------
// Cosine helpers
// ---------------------------------------------------------------------------

/// `compareCosine`: returns the cosine value (not distance).
/// `cos(θ) = dot(a,b) / (|a| · |b|)`
/// Maps to `PrimitiveComparator::compareCosine<float>`.
#[inline]
fn compare_cosine_value(a: &[f32], b: &[f32]) -> f64 {
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut sum    = 0.0f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let af = ai as f64;
        let bf = bi as f64;
        norm_a += af * af;
        norm_b += bf * bf;
        sum    += af * bf;
    }
    sum / (norm_a * norm_b).sqrt()
}

/// `compareCosineSimilarity`: `|1 - cosine(a, b)|`.
/// Maps to `PrimitiveComparator::compareCosineSimilarity<float>`.
#[inline]
pub fn compare_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let v = 1.0 - compare_cosine_value(a, b);
    v.abs()
}

/// `compareNormalizedCosineSimilarity` (float): `|1 - dot(a, b)|`.
/// For pre-normalized vectors dot(a,b) = cosine(a,b).
/// Maps to `PrimitiveComparator::compareNormalizedCosineSimilarity(const float*, ...)`.
#[inline]
pub fn compare_normalized_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let v = 1.0 - compare_dot_product(a, b);
    v.abs()
}

// ---------------------------------------------------------------------------
// Angle
// ---------------------------------------------------------------------------

/// `compareAngleDistance`: `acos(cosine(a, b))`, clamped to `[0, π]`.
/// Maps to `PrimitiveComparator::compareAngleDistance<float>`.
#[inline]
pub fn compare_angle_distance(a: &[f32], b: &[f32]) -> f64 {
    let cosine = compare_cosine_value(a, b);
    if cosine >= 1.0 {
        0.0
    } else if cosine <= -1.0 {
        std::f64::consts::PI
    } else {
        cosine.acos()
    }
}

/// `compareNormalizedAngleDistance`: `acos(dot(a, b))` for pre-normalized vectors.
/// Maps to `PrimitiveComparator::compareNormalizedAngleDistance<float>`.
#[inline]
pub fn compare_normalized_angle_distance(a: &[f32], b: &[f32]) -> f64 {
    let cosine = compare_dot_product(a, b);
    if cosine >= 1.0 {
        0.0
    } else if cosine <= -1.0 {
        std::f64::consts::PI
    } else {
        cosine.acos()
    }
}

// ---------------------------------------------------------------------------
// Poincaré  (PrimitiveComparator::comparePoincareDistance<float>)
// ---------------------------------------------------------------------------

/// Poincaré hyperbolic distance.
/// `acosh(1 + 2·|a-b|²  / ((1-|a|²)·(1-|b|²)))`
/// Maps to `PrimitiveComparator::comparePoincareDistance<float>`.
pub fn compare_poincare(a: &[f32], b: &[f32]) -> f64 {
    let mut a2 = 0.0f64;
    let mut b2 = 0.0f64;
    for &ai in a { a2 += (ai as f64) * (ai as f64); }
    for &bi in b { b2 += (bi as f64) * (bi as f64); }
    let c2 = compare_l2(a, b);
    (1.0 + 2.0 * c2 * c2 / (1.0 - a2) / (1.0 - b2)).acosh()
}

// ---------------------------------------------------------------------------
// Lorentz  (PrimitiveComparator::compareLorentzDistance<float>)
// ---------------------------------------------------------------------------

/// Lorentz (Minkowski hyperboloid) distance.
/// `acosh(a[0]*b[0] - Σ_{i>0} a[i]*b[i])`
/// Maps to `PrimitiveComparator::compareLorentzDistance<float>`.
pub fn compare_lorentz(a: &[f32], b: &[f32]) -> f64 {
    debug_assert!(!a.is_empty());
    let mut sum = (a[0] as f64) * (b[0] as f64);
    for i in 1..a.len() {
        sum -= (a[i] as f64) * (b[i] as f64);
    }
    sum.acosh()
}

// ---------------------------------------------------------------------------
// Dispatch  (single entry-point used by ObjectSpace)
// ---------------------------------------------------------------------------

/// Compute the distance between `a` and `b` using the given metric.
/// Returns `f32` (NGT::Distance).
///
/// This is the single dispatch function used by `ObjectSpace::distance_from_vec`
/// and `ObjectSpace::distance`, mirroring the C++ virtual `compare()` call.
pub fn compare(a: &[f32], b: &[f32], dt: DistanceType) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match dt {
        DistanceType::L1                        => compare_l1(a, b) as f32,
        DistanceType::L2                        => compare_l2(a, b) as f32,
        DistanceType::NormalizedL2              => compare_normalized_l2(a, b) as f32,
        DistanceType::CosineSimilarity          => compare_cosine_similarity(a, b) as f32,
        DistanceType::NormalizedCosineSimilarity => compare_normalized_cosine_similarity(a, b) as f32,
        DistanceType::Angle                     => compare_angle_distance(a, b) as f32,
        DistanceType::NormalizedAngle           => compare_normalized_angle_distance(a, b) as f32,
        DistanceType::Poincare                  => compare_poincare(a, b) as f32,
        DistanceType::Lorentz                   => compare_lorentz(a, b) as f32,
        DistanceType::InnerProduct              => {
            // -dot(a, b) as distance (maps to ComparatorInnerProduct: return -compareDotProduct)
            -compare_dot_product(a, b) as f32
        }
        DistanceType::DotProduct                => {
            // magnitude - dot(a, b); magnitude not stored here — callers that need
            // DotProduct should use the Index-level comparator which knows magnitude.
            // For plain compare(), fall back to -dot (same as InnerProduct).
            -compare_dot_product(a, b) as f32
        }
        DistanceType::Hamming | DistanceType::Jaccard | DistanceType::SparseJaccard => {
            // These operate on u8 slices — callers must use compare_hamming /
            // compare_jaccard / compare_sparse_jaccard directly.
            panic!(
                "PrimitiveComparator::compare: {:?} is not valid for f32 slices",
                dt
            );
        }
        DistanceType::None => {
            panic!("PrimitiveComparator::compare: DistanceType::None");
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reinterpret a `&[u8]` as `&[u32]` (zero-copy).
/// The slice length must be a multiple of 4.
#[inline]
fn bytecast_u32(bytes: &[u8]) -> &[u32] {
    debug_assert!(bytes.len() % 4 == 0);
    // SAFETY: u8 has alignment 1; u32 has alignment 4.  We check that the
    // pointer is 4-byte aligned and that the byte count is a multiple of 4.
    // NGT always passes aligned, properly-sized buffers here.
    let (prefix, words, suffix) = unsafe { bytes.align_to::<u32>() };
    debug_assert!(prefix.is_empty(), "bytecast_u32: unaligned input");
    debug_assert!(suffix.is_empty(), "bytecast_u32: length not a multiple of 4");
    words
}

// ---------------------------------------------------------------------------
// NEON SIMD implementations (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Dot product using NEON: process 16 f32s per iteration (4 accumulators).
    #[inline]
    pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f64 {
        let size = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut i = 0usize;

        unsafe {
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);

            // Process 16 elements per iteration.
            while i + 16 <= size {
                let a0 = vld1q_f32(ap.add(i));
                let b0 = vld1q_f32(bp.add(i));
                sum0 = vfmaq_f32(sum0, a0, b0);

                let a1 = vld1q_f32(ap.add(i + 4));
                let b1 = vld1q_f32(bp.add(i + 4));
                sum1 = vfmaq_f32(sum1, a1, b1);

                let a2 = vld1q_f32(ap.add(i + 8));
                let b2 = vld1q_f32(bp.add(i + 8));
                sum2 = vfmaq_f32(sum2, a2, b2);

                let a3 = vld1q_f32(ap.add(i + 12));
                let b3 = vld1q_f32(bp.add(i + 12));
                sum3 = vfmaq_f32(sum3, a3, b3);

                i += 16;
            }

            // Process 4 elements at a time.
            while i + 4 <= size {
                let a0 = vld1q_f32(ap.add(i));
                let b0 = vld1q_f32(bp.add(i));
                sum0 = vfmaq_f32(sum0, a0, b0);
                i += 4;
            }

            // Reduce: sum all 4 accumulators into one.
            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            let mut result = vaddvq_f32(sum0) as f64;

            // Scalar tail.
            while i < size {
                result += (*ap.add(i) as f64) * (*bp.add(i) as f64);
                i += 1;
            }
            result
        }
    }

    /// L2 squared distance using NEON: process 16 f32s per iteration.
    /// Returns the SQUARED distance (caller takes sqrt).
    #[inline]
    pub fn l2_neon(a: &[f32], b: &[f32]) -> f64 {
        let size = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut i = 0usize;

        unsafe {
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);

            while i + 16 <= size {
                let d0 = vsubq_f32(vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
                sum0 = vfmaq_f32(sum0, d0, d0);

                let d1 = vsubq_f32(vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
                sum1 = vfmaq_f32(sum1, d1, d1);

                let d2 = vsubq_f32(vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
                sum2 = vfmaq_f32(sum2, d2, d2);

                let d3 = vsubq_f32(vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
                sum3 = vfmaq_f32(sum3, d3, d3);

                i += 16;
            }

            while i + 4 <= size {
                let d0 = vsubq_f32(vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
                sum0 = vfmaq_f32(sum0, d0, d0);
                i += 4;
            }

            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            let mut result = vaddvq_f32(sum0) as f64;

            while i < size {
                let diff = (*ap.add(i) as f64) - (*bp.add(i) as f64);
                result += diff * diff;
                i += 1;
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_l2_basic() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        // sqrt(1 + 1)
        assert!(approx_eq(compare_l2(&a, &b) as f32, 2.0f32.sqrt()));
    }

    #[test]
    fn test_l1_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 0.0, 1.0];
        // |1-4| + |2-0| + |3-1| = 3 + 2 + 2 = 7
        assert!(approx_eq(compare_l1(&a, &b) as f32, 7.0));
    }

    #[test]
    fn test_normalized_l2_orthogonal() {
        // Two orthogonal unit vectors: L2 = sqrt(2), NormalizedL2 = sqrt(2)
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(approx_eq(compare_normalized_l2(&a, &b) as f32, 2.0f32.sqrt()));
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        // cosine = 0, distance = |1 - 0| = 1
        assert!(approx_eq(compare_cosine_similarity(&a, &b) as f32, 1.0));
    }

    #[test]
    fn test_cosine_identical() {
        let a = [1.0f32, 1.0];
        let b = [1.0f32, 1.0];
        // cosine = 1, distance = 0
        assert!(approx_eq(compare_cosine_similarity(&a, &b) as f32, 0.0));
    }

    #[test]
    fn test_normalized_cosine_similarity() {
        // Pre-normalized vectors
        let s = 2.0f32.sqrt();
        let a = [1.0 / s, 1.0 / s];
        let b = [1.0f32, 0.0];
        // dot = 1/sqrt(2), distance = |1 - 1/sqrt(2)|
        let expected = (1.0 - 1.0f32 / s).abs();
        assert!(approx_eq(
            compare_normalized_cosine_similarity(&a, &b) as f32,
            expected
        ));
    }

    #[test]
    fn test_angle_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        // angle = π/2
        assert!(approx_eq(
            compare_angle_distance(&a, &b) as f32,
            std::f32::consts::FRAC_PI_2
        ));
    }

    #[test]
    fn test_angle_identical() {
        let a = [1.0f32, 0.0];
        assert!(approx_eq(compare_angle_distance(&a, &a) as f32, 0.0));
    }

    #[test]
    fn test_normalize_unit_vector() {
        let mut v = [3.0f32, 4.0]; // norm = 5
        normalize(&mut v);
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
        assert!(approx_eq(norm(&v), 1.0));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = [0.0f32, 0.0];
        normalize(&mut v); // must not panic
        assert_eq!(v, [0.0, 0.0]);
    }

    #[test]
    fn test_requires_normalization() {
        assert!(requires_normalization(DistanceType::CosineSimilarity));
        assert!(requires_normalization(DistanceType::NormalizedAngle));
        assert!(!requires_normalization(DistanceType::L2));
        assert!(!requires_normalization(DistanceType::L1));
    }

    #[test]
    fn test_dispatch_l2() {
        let a = [0.0f32, 0.0];
        let b = [3.0f32, 4.0];
        assert!(approx_eq(compare(&a, &b, DistanceType::L2), 5.0));
    }
}
