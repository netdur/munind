/// Hardcoded Lloyd-Max optimal codebooks for N(0,1) distribution.
///
/// These are the exact same centroids used by MNN/TurboQuant.
/// No per-dataset training needed — the WHT ensures coordinates are ~N(0,1).

/// TQ3: 8 centroids (3-bit, indices 0-7).
pub const TQ3_CENTROIDS: [f32; 8] = [
    -2.1519, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1519,
];

/// TQ3 decision boundaries (midpoints between adjacent centroids).
pub const TQ3_BOUNDARIES: [f32; 7] = [
    -1.7479, -1.0500, -0.5006, 0.0,
     0.5006,  1.0500,  1.7479,
];

/// TQ4: 16 centroids (4-bit, indices 0-15).
pub const TQ4_CENTROIDS: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562,
    -0.9424, -0.6568, -0.3880, -0.1284,
     0.1284,  0.3880,  0.6568,  0.9424,
     1.2562,  1.6180,  2.0690,  2.7326,
];

/// TQ4 decision boundaries.
pub const TQ4_BOUNDARIES: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993,
    -0.7996, -0.5224, -0.2582, 0.0,
     0.2582,  0.5224,  0.7996,  1.0993,
     1.4371,  1.8435,  2.4008,
];

/// TQ8: 256 centroids (8-bit). Computed from Lloyd-Max for N(0,1).
/// For 8-bit we use uniform quantization in [-3, 3] which is near-optimal.
pub fn tq8_centroid(index: u8) -> f32 {
    -3.0 + (index as f32 + 0.5) * (6.0 / 256.0)
}

/// Quantize a single value using TQ3 codebook.
#[inline]
pub fn quantize_tq3(val: f32) -> u8 {
    // Threshold comparisons (faster than binary search for 8 levels).
    if val < TQ3_BOUNDARIES[3] {
        if val < TQ3_BOUNDARIES[1] {
            if val < TQ3_BOUNDARIES[0] { 0 } else { 1 }
        } else {
            if val < TQ3_BOUNDARIES[2] { 2 } else { 3 }
        }
    } else {
        if val < TQ3_BOUNDARIES[5] {
            if val < TQ3_BOUNDARIES[4] { 4 } else { 5 }
        } else {
            if val < TQ3_BOUNDARIES[6] { 6 } else { 7 }
        }
    }
}

/// Quantize a single value using TQ4 codebook.
#[inline]
pub fn quantize_tq4(val: f32) -> u8 {
    if val < TQ4_BOUNDARIES[7] {
        if val < TQ4_BOUNDARIES[3] {
            if val < TQ4_BOUNDARIES[1] {
                if val < TQ4_BOUNDARIES[0] { 0 } else { 1 }
            } else {
                if val < TQ4_BOUNDARIES[2] { 2 } else { 3 }
            }
        } else {
            if val < TQ4_BOUNDARIES[5] {
                if val < TQ4_BOUNDARIES[4] { 4 } else { 5 }
            } else {
                if val < TQ4_BOUNDARIES[6] { 6 } else { 7 }
            }
        }
    } else {
        if val < TQ4_BOUNDARIES[11] {
            if val < TQ4_BOUNDARIES[9] {
                if val < TQ4_BOUNDARIES[8] { 8 } else { 9 }
            } else {
                if val < TQ4_BOUNDARIES[10] { 10 } else { 11 }
            }
        } else {
            if val < TQ4_BOUNDARIES[13] {
                if val < TQ4_BOUNDARIES[12] { 12 } else { 13 }
            } else {
                if val < TQ4_BOUNDARIES[14] { 14 } else { 15 }
            }
        }
    }
}

/// Quantize a single value using TQ8 (uniform in [-3, 3]).
#[inline]
pub fn quantize_tq8(val: f32) -> u8 {
    let clamped = val.clamp(-3.0, 3.0);
    let idx = ((clamped + 3.0) * (256.0 / 6.0)) as u32;
    idx.min(255) as u8
}

/// Dequantize TQ3.
#[inline]
pub fn dequantize_tq3(idx: u8) -> f32 {
    TQ3_CENTROIDS[idx as usize & 7]
}

/// Dequantize TQ4.
#[inline]
pub fn dequantize_tq4(idx: u8) -> f32 {
    TQ4_CENTROIDS[idx as usize & 15]
}

/// Dequantize TQ8.
#[inline]
pub fn dequantize_tq8(idx: u8) -> f32 {
    tq8_centroid(idx)
}
