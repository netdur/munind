/// Metal GPU fused dequant+distance kernel for TurboQuant search.
///
/// Dispatches one threadgroup per neighbor vector. Each thread handles
/// a subset of dimensions. The kernel does:
///   1. Read quantized code for this dimension
///   2. Look up centroid value from codebook
///   3. Multiply by norm → dequantized coordinate
///   4. Compute partial distance vs rotated query
///   5. Threadgroup reduction → final distance per neighbor

#[cfg(target_os = "macos")]
use metal::*;

/// The Metal Shading Language (MSL) source for the fused kernel.
#[cfg(target_os = "macos")]
const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused TQ dequant + NormalizedCosine distance kernel.
//
// Each threadgroup computes distance for ONE neighbor.
// Threads within the group split the dimensions.
//
// Buffers:
//   0: query_rot [padded_dim]      — rotated query (f32)
//   1: codes    [max_neighbors * padded_dim]  — u8 codes for all neighbors
//   2: norms    [max_neighbors]    — f32 norms per neighbor
//   3: centroids [num_levels]      — f32 codebook centroids
//   4: distances [max_neighbors]   — f32 output distances
//   5: params   [4]                — {padded_dim, num_levels, num_neighbors, distance_type}

kernel void tq_batch_distance(
    device const float*   query_rot  [[buffer(0)]],
    device const uchar*   codes      [[buffer(1)]],
    device const float*   norms      [[buffer(2)]],
    device const float*   centroids  [[buffer(3)]],
    device float*         distances  [[buffer(4)]],
    device const uint*    params     [[buffer(5)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tcount  [[threads_per_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]]
) {
    uint padded_dim    = params[0];
    uint num_levels    = params[1];
    uint num_neighbors = params[2];
    uint dist_type     = params[3];  // 0=L2, 1=NormalizedCosine

    if (gid >= num_neighbors) return;

    // Pointer to this neighbor's codes.
    device const uchar* my_codes = codes + gid * padded_dim;
    float my_norm = norms[gid];

    // Each thread accumulates partial dot product over its dimensions.
    float partial_dot = 0.0f;
    float partial_csq = 0.0f;  // for L2: sum of centroid²

    for (uint i = tid; i < padded_dim; i += tcount) {
        uint code = my_codes[i];
        float centroid_val = centroids[code];
        float dequant = centroid_val * my_norm;
        float q = query_rot[i];

        if (dist_type == 1) {
            // NormalizedCosine: accumulate dot product.
            partial_dot += q * dequant;
        } else {
            // L2: accumulate (q - dequant)².
            float diff = q - dequant;
            partial_dot += diff * diff;
        }
    }

    // Threadgroup reduction using shared memory.
    threadgroup float shared_dot[256];
    threadgroup float shared_csq[256];
    shared_dot[tid] = partial_dot;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final distance.
    if (tid == 0) {
        if (dist_type == 1) {
            // NormalizedCosine: |1 - dot|
            distances[gid] = fabs(1.0f - shared_dot[0]);
        } else {
            // L2: sqrt(sum_sq)
            float sq = shared_dot[0];
            distances[gid] = sq > 0.0f ? sqrt(sq) : 0.0f;
        }
    }
}
"#;

/// Persistent Metal context — created once, reused across searches.
#[cfg(target_os = "macos")]
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    // Reusable buffers (resized as needed).
    buf_query: Buffer,
    buf_codes: Buffer,
    buf_norms: Buffer,
    buf_centroids: Buffer,
    buf_distances: Buffer,
    buf_params: Buffer,
    // Capacity tracking.
    max_neighbors: usize,
    padded_dim: usize,
    num_levels: usize,
}

#[cfg(target_os = "macos")]
impl MetalContext {
    pub fn new(padded_dim: usize, num_levels: usize) -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let library = device
            .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
            .ok()?;
        let function = library.get_function("tq_batch_distance", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        let initial_neighbors = 64;

        let buf_query = device.new_buffer(
            (padded_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_codes = device.new_buffer(
            (initial_neighbors * padded_dim) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_norms = device.new_buffer(
            (initial_neighbors * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_centroids = device.new_buffer(
            (num_levels * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_distances = device.new_buffer(
            (initial_neighbors * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_params = device.new_buffer(
            16u64, // 4 uint32s
            MTLResourceOptions::StorageModeShared,
        );

        Some(MetalContext {
            device,
            queue,
            pipeline,
            buf_query,
            buf_codes,
            buf_norms,
            buf_centroids,
            buf_distances,
            buf_params,
            max_neighbors: initial_neighbors,
            padded_dim,
            num_levels,
        })
    }

    /// Ensure buffers are large enough for `n` neighbors.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.max_neighbors {
            return;
        }
        let new_cap = n.next_power_of_two();
        self.buf_codes = self.device.new_buffer(
            (new_cap * self.padded_dim) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.buf_norms = self.device.new_buffer(
            (new_cap * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.buf_distances = self.device.new_buffer(
            (new_cap * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.max_neighbors = new_cap;
    }

    /// Upload rotated query and centroids (once per query).
    pub fn set_query(&self, query_rot: &[f32], centroids: &[f32]) {
        unsafe {
            let ptr = self.buf_query.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(query_rot.as_ptr(), ptr, query_rot.len());

            let ptr = self.buf_centroids.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(centroids.as_ptr(), ptr, centroids.len());
        }
    }

    /// Compute distances for a batch of neighbors.
    ///
    /// `neighbor_codes`: slice of code arrays (each padded_dim u8s).
    /// `neighbor_norms`: slice of norms (f32).
    /// `dist_type`: 0=L2, 1=NormalizedCosine.
    ///
    /// Returns distances (f32 per neighbor).
    pub fn batch_distance(
        &mut self,
        neighbor_codes: &[&[u32]],
        neighbor_norms: &[f32],
        dist_type: u32,
    ) -> Vec<f32> {
        let n = neighbor_codes.len();
        if n == 0 {
            return Vec::new();
        }
        self.ensure_capacity(n);

        let pd = self.padded_dim;

        // Upload codes as u8.
        unsafe {
            let ptr = self.buf_codes.contents() as *mut u8;
            for (i, codes) in neighbor_codes.iter().enumerate() {
                for j in 0..pd {
                    *ptr.add(i * pd + j) = codes[j] as u8;
                }
            }
        }

        // Upload norms.
        unsafe {
            let ptr = self.buf_norms.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(neighbor_norms.as_ptr(), ptr, n);
        }

        // Upload params.
        unsafe {
            let ptr = self.buf_params.contents() as *mut u32;
            *ptr.add(0) = pd as u32;
            *ptr.add(1) = self.num_levels as u32;
            *ptr.add(2) = n as u32;
            *ptr.add(3) = dist_type;
        }

        // Dispatch.
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.buf_query), 0);
        encoder.set_buffer(1, Some(&self.buf_codes), 0);
        encoder.set_buffer(2, Some(&self.buf_norms), 0);
        encoder.set_buffer(3, Some(&self.buf_centroids), 0);
        encoder.set_buffer(4, Some(&self.buf_distances), 0);
        encoder.set_buffer(5, Some(&self.buf_params), 0);

        // One threadgroup per neighbor, 128 threads per group (covers padded_dim).
        let threads_per_group = 128.min(pd).next_power_of_two();
        let threadgroup_size = MTLSize::new(threads_per_group as u64, 1, 1);
        let grid_size = MTLSize::new(threads_per_group as u64 * n as u64, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back distances.
        let mut result = vec![0.0f32; n];
        unsafe {
            let ptr = self.buf_distances.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), n);
        }
        result
    }
}

/// Stub for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub fn new(_padded_dim: usize, _num_levels: usize) -> Option<Self> {
        None
    }
}
