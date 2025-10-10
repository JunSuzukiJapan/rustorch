//! Metal Performance Shaders kernel implementations for GPU acceleration
//! GPU加速のためのMetal Performance Shadersカーネル実装

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
// Metal GPU kernel implementations
use std::ffi::c_void;
use std::marker::PhantomData;
use std::cell::RefCell;
use std::time::Instant;

#[cfg(feature = "metal")]
use metal::foreign_types::ForeignType;
#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
    MTLSize,
};
#[cfg(feature = "metal")]
use lazy_static::lazy_static;
#[cfg(feature = "metal")]
use std::sync::Mutex;

/// Metal kernel types
/// Metalカーネルタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalKernelType {
    /// Element-wise operations (add, mul, etc.)
    /// 要素ごとの演算（加算、乗算など）
    ElementWise,
    /// Matrix multiplication operations
    /// 行列乗算演算
    MatMul,
    /// Reduction operations (sum, mean, etc.)
    /// リダクション演算（合計、平均など）
    Reduction,
    /// Convolution operations
    /// 畳み込み演算
    Convolution,
    /// Batch normalization operations
    /// バッチ正規化演算
    BatchNorm,
}

/// Metal kernel parameters
/// Metalカーネルパラメータ
#[derive(Debug, Clone)]
pub struct MetalKernelParams {
    /// Threads per threadgroup for Metal kernel execution
    /// Metalカーネル実行のスレッドグループあたりのスレッド数
    pub threads_per_threadgroup: (u32, u32, u32),
    /// Threadgroups per grid for Metal kernel execution
    /// Metalカーネル実行のグリッドあたりのスレッドグループ数
    pub threadgroups_per_grid: (u32, u32, u32),
}

impl Default for MetalKernelParams {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: (1, 1, 1),
            threadgroups_per_grid: (1, 1, 1),
        }
    }
}

/// Metal buffer wrapper
/// Metalバッファラッパー
pub struct MetalBuffer<T> {
    _buffer: *mut c_void,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> MetalBuffer<T> {
    /// Create a new Metal buffer
    /// 新しいMetalバッファを作成
    #[cfg(feature = "metal")]
    pub fn new(size: usize, device: &Device) -> RusTorchResult<Self> {
        let buffer_size = size * std::mem::size_of::<T>();
        let buffer = device.new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            _buffer: buffer.as_ptr() as *mut c_void,
            size,
            _phantom: PhantomData,
        })
    }

    #[cfg(not(feature = "metal"))]
    /// Create a new Metal buffer with specified size
    /// 指定されたサイズで新しいMetalバッファを作成
    pub fn new(_size: usize, _device: &()) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }

    /// Copy data from host to Metal buffer
    /// ホストからMetalバッファへデータをコピー
    pub fn copy_from_host(&mut self, host_data: &[T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidOperation(
                "Size mismatch in host-to-device copy".to_string(),
            ));
        }

        #[cfg(feature = "metal")]
        {
            unsafe {
                let buffer_ptr = self._buffer as *mut T;
                std::ptr::copy_nonoverlapping(host_data.as_ptr(), buffer_ptr, self.size);
            }
            Ok(())
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "Metal not available".to_string(),
            ))
        }
    }

    /// Copy data from Metal buffer to host
    /// Metalバッファからホストへデータをコピー
    pub fn copy_to_host(&self, host_data: &mut [T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidOperation(
                "Size mismatch in device-to-host copy".to_string(),
            ));
        }

        #[cfg(feature = "metal")]
        {
            unsafe {
                let buffer_ptr = self._buffer as *const T;
                std::ptr::copy_nonoverlapping(buffer_ptr, host_data.as_mut_ptr(), self.size);
            }
            Ok(())
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "Metal not available".to_string(),
            ))
        }
    }
}

/// Cached Metal buffer for reuse
/// 再利用可能なキャッシュされたMetalバッファ
#[cfg(feature = "metal")]
struct CachedBuffer {
    buffer: Buffer,
    size_bytes: usize,
    last_used: Instant,
}

/// Metal kernel executor for high-performance GPU operations
/// 高性能GPU演算のためのMetalカーネル実行器
#[cfg(feature = "metal")]
pub struct MetalKernelExecutor {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    pipeline_states: HashMap<String, ComputePipelineState>,
    // Buffer cache for reuse to avoid repeated allocation/deallocation
    // 繰り返しの割り当て/解放を避けるためのバッファキャッシュ
    buffer_cache: RefCell<Vec<CachedBuffer>>,
    // RAII-based matmul executor for proper memory management
    // 適切なメモリ管理のためのRAIIベースmatmulエグゼキュータ
    raii_matmul: Option<crate::gpu::metal_matmul_raii::MetalMatMulExecutor>,
}

#[cfg(feature = "metal")]
lazy_static! {
    /// Global singleton Metal kernel executor
    /// グローバルシングルトンMetalカーネル実行器
    static ref METAL_EXECUTOR: Mutex<Option<MetalKernelExecutor>> = Mutex::new(None);
}

#[cfg(feature = "metal")]
impl Drop for MetalKernelExecutor {
    fn drop(&mut self) {
        // Metal Device and CommandQueue are automatically managed by ARC
        // but we explicitly log cleanup for debugging
        eprintln!("🧹 Cleaning up Metal kernel executor resources");
    }
}

#[cfg(feature = "metal")]
impl MetalKernelExecutor {
    /// Get or create the singleton Metal kernel executor
    /// シングルトンMetalカーネル実行器を取得または作成
    pub fn get() -> RusTorchResult<&'static Mutex<Option<MetalKernelExecutor>>> {
        // Ensure executor is initialized
        eprintln!("🔍 [METAL GET] Acquiring lock on METAL_EXECUTOR...");
        let mut executor_guard = METAL_EXECUTOR.lock().unwrap();
        eprintln!("🔍 [METAL GET] Lock acquired, checking if executor exists...");
        if executor_guard.is_none() {
            eprintln!("🔍 [METAL GET] Executor not initialized, calling new_internal()...");
            *executor_guard = Some(Self::new_internal()?);
            eprintln!("🚀 Initialized Metal kernel executor singleton");
        } else {
            eprintln!("✅ [METAL GET] Using existing Metal kernel executor singleton");
        }
        eprintln!("🔍 [METAL GET] Releasing lock...");
        drop(executor_guard);
        eprintln!("🔍 [METAL GET] Lock released, returning reference");
        Ok(&METAL_EXECUTOR)
    }

    /// Create a new Metal kernel executor (deprecated - use get() instead)
    /// 新しいMetalカーネル実行器を作成（非推奨 - get()を使用してください）
    #[deprecated(note = "Use MetalKernelExecutor::get() instead to avoid context leaks")]
    pub fn new() -> RusTorchResult<Self> {
        Self::new_internal()
    }

    /// Create a new Metal kernel executor (internal use only)
    /// 新しいMetalカーネル実行器を作成（内部使用のみ）
    fn new_internal() -> RusTorchResult<Self> {
        eprintln!("🔍 [METAL INIT] Step 1: Starting MetalKernelExecutor initialization...");

        // CRITICAL: Wrap ALL Metal initialization in autoreleasepool
        // すべてのMetal初期化をautoreleasepoolで囲む（重要）
        use crate::gpu::objc_bridge::with_autoreleasepool;

        with_autoreleasepool(|| {
            eprintln!("🔍 [METAL INIT] Step 2: Inside autoreleasepool, getting default device...");
            let device = Device::system_default()
                .ok_or_else(|| RusTorchError::tensor_op("No Metal device available"))?;

            eprintln!("🔍 [METAL INIT] Step 3: Device obtained, creating command queue...");
            let command_queue = device.new_command_queue();
            eprintln!("🔍 [METAL INIT] Step 4: Command queue created");

            Self::new_internal_with_device_and_queue(device, command_queue)
        })
    }

    fn new_internal_with_device_and_queue(device: Device, command_queue: CommandQueue) -> RusTorchResult<Self> {
        eprintln!("🔍 [METAL INIT] Step 5: Compiling Metal shaders...");

        // Metal shader library source optimized for AMD Radeon Pro Vega 56
        let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        // Optimized element-wise addition with vectorization
        kernel void elementwise_add_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            result[index] = a[index] + b[index];
        }
        
        // Optimized element-wise multiplication with vectorization
        kernel void elementwise_mul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            result[index] = a[index] * b[index];
        }
        
        // High-performance matrix multiplication optimized for Vega 56 architecture
        kernel void matmul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* c [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;
            
            if (row >= M || col >= N) return;
            
            float sum = 0.0;
            // Unroll loop for better performance on Vega architecture
            for (uint k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }

        // Matrix multiplication with B transposed: C = A @ B^T
        // B is stored as [n, k] but treated as [k, n]^T
        kernel void matmul_transposed_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* c [[buffer(2)]],
            constant uint& m [[buffer(3)]],
            constant uint& n [[buffer(4)]],
            constant uint& k [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;

            if (row >= m || col >= n) return;

            float value = 0.0;
            for (uint i = 0; i < k; i++) {
                // B is [n, k], so B^T[i, col] = B[col, i] = b[col * k + i]
                value += a[row * k + i] * b[col * k + i];
            }
            c[row * n + col] = value;
        }

        // Optimized tiled matrix multiplication for large matrices
        kernel void tiled_matrix_multiply_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* c [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            threadgroup float* tile_a [[threadgroup(0)]],
            threadgroup float* tile_b [[threadgroup(1)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 lid [[thread_position_in_threadgroup]]
        ) {
            const uint TILE_SIZE = 16; // Optimized for Vega 56 memory hierarchy
            
            uint row = gid.y;
            uint col = gid.x;
            uint local_row = lid.y;
            uint local_col = lid.x;
            
            float sum = 0.0;
            
            // Tiled computation for better memory bandwidth utilization
            for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
                // Load tiles into shared memory
                uint a_idx = row * K + tile * TILE_SIZE + local_col;
                uint b_idx = (tile * TILE_SIZE + local_row) * N + col;
                
                tile_a[local_row * TILE_SIZE + local_col] = 
                    (row < M && tile * TILE_SIZE + local_col < K) ? a[a_idx] : 0.0;
                tile_b[local_row * TILE_SIZE + local_col] = 
                    (tile * TILE_SIZE + local_row < K && col < N) ? b[b_idx] : 0.0;
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Compute partial result
                for (uint k = 0; k < TILE_SIZE; k++) {
                    sum += tile_a[local_row * TILE_SIZE + k] * 
                           tile_b[k * TILE_SIZE + local_col];
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (row < M && col < N) {
                c[row * N + col] = sum;
            }
        }
        
        kernel void reduce_sum_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            threadgroup float* shared_data [[threadgroup(0)]],
            uint tid [[thread_index_in_threadgroup]],
            uint bid [[threadgroup_position_in_grid]],
            uint block_size [[threads_per_threadgroup]],
            uint grid_size [[threadgroups_per_grid]]
        ) {
            uint index = bid * block_size + tid;

            // Load data into shared memory
            shared_data[tid] = (index < grid_size * block_size) ? input[index] : 0.0;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduction in shared memory
            for (uint s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write result for this block
            if (tid == 0) {
                output[bid] = shared_data[0];
            }
        }

        // Activation functions optimized for Vega 56 architecture

        // ReLU activation: max(0, x)
        kernel void relu_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            output[index] = fmax(0.0f, input[index]);
        }

        // Sigmoid activation: 1 / (1 + exp(-x))
        kernel void sigmoid_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            output[index] = 1.0f / (1.0f + exp(-input[index]));
        }

        // Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        kernel void tanh_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            output[index] = tanh(input[index]);
        }

        // GELU activation: x * 0.5 * (1 + erf(x / sqrt(2))) - using Abramowitz and Stegun approximation
        kernel void gelu_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            float x = input[index];
            float sqrt_2_inv = 0.70710678118f; // 1 / sqrt(2)
            float t = x * sqrt_2_inv;

            // Abramowitz and Stegun approximation for erf function
            float a1 =  0.254829592f;
            float a2 = -0.284496736f;
            float a3 =  1.421413741f;
            float a4 = -1.453152027f;
            float a5 =  1.061405429f;
            float p  =  0.3275911f;

            float sign = (t >= 0.0f) ? 1.0f : -1.0f;
            t = fabs(t);

            float t_p = 1.0f / (1.0f + p * t);
            float erf_approx = 1.0f - (((((a5 * t_p + a4) * t_p) + a3) * t_p + a2) * t_p + a1) * t_p * exp(-t * t);
            erf_approx *= sign;

            output[index] = x * 0.5f * (1.0f + erf_approx);
        }

        // Softmax activation (per element, requires separate reduction for proper implementation)
        kernel void softmax_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            constant float& max_val [[buffer(3)]],
            constant float& sum_exp [[buffer(4)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            float exp_val = exp(input[index] - max_val);
            output[index] = exp_val / sum_exp;
        }

        // Leaky ReLU activation: max(alpha * x, x)
        kernel void leaky_relu_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            constant float& alpha [[buffer(3)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            float x = input[index];
            output[index] = (x > 0.0f) ? x : alpha * x;
        }

        // ELU activation: x if x > 0, alpha * (exp(x) - 1) if x <= 0
        kernel void elu_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            constant float& alpha [[buffer(3)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            float x = input[index];
            output[index] = (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
        }

        // Swish activation: x * sigmoid(x)
        kernel void swish_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            if (index >= n) return;
            float x = input[index];
            float sigmoid_x = 1.0f / (1.0f + exp(-x));
            output[index] = x * sigmoid_x;
        }

        // LayerNorm: normalize over last dimension with learned affine parameters
        // Optimized for GPT transformer models
        kernel void layer_norm_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            device const float* gamma [[buffer(2)]],
            device const float* beta [[buffer(3)]],
            constant uint& batch_size [[buffer(4)]],
            constant uint& seq_len [[buffer(5)]],
            constant uint& features [[buffer(6)]],
            constant float& eps [[buffer(7)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint b = gid.y;  // batch index
            uint s = gid.x;  // sequence position index

            if (b >= batch_size || s >= seq_len) return;

            uint offset = (b * seq_len + s) * features;

            // Calculate mean
            float sum = 0.0f;
            for (uint f = 0; f < features; f++) {
                sum += input[offset + f];
            }
            float mean = sum / float(features);

            // Calculate variance
            float var_sum = 0.0f;
            for (uint f = 0; f < features; f++) {
                float diff = input[offset + f] - mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / float(features);
            float std = sqrt(variance + eps);

            // Normalize and apply affine transformation
            for (uint f = 0; f < features; f++) {
                float normalized = (input[offset + f] - mean) / std;
                output[offset + f] = gamma[f] * normalized + beta[f];
            }
        }

        // LayerNorm f64 version - DISABLED: Metal does not support f64 (double precision)
        // Use f32 version or CPU fallback instead
        /*
        kernel void layer_norm_f64(
            device const double* input [[buffer(0)]],
            device double* output [[buffer(1)]],
            device const double* gamma [[buffer(2)]],
            device const double* beta [[buffer(3)]],
            constant uint& batch_size [[buffer(4)]],
            constant uint& seq_len [[buffer(5)]],
            constant uint& features [[buffer(6)]],
            constant double& eps [[buffer(7)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint b = gid.y;  // batch index
            uint s = gid.x;  // sequence position index

            if (b >= batch_size || s >= seq_len) return;

            uint offset = (b * seq_len + s) * features;

            // Calculate mean
            double sum = 0.0;
            for (uint f = 0; f < features; f++) {
                sum += input[offset + f];
            }
            double mean = sum / double(features);

            // Calculate variance
            double var_sum = 0.0;
            for (uint f = 0; f < features; f++) {
                double diff = input[offset + f] - mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / double(features);
            double std = sqrt(variance + eps);

            // Normalize and apply affine transformation
            for (uint f = 0; f < features; f++) {
                double normalized = (input[offset + f] - mean) / std;
                output[offset + f] = gamma[f] * normalized + beta[f];
            }
        }
        */

        // RoPE (Rotary Position Embedding) kernel
        // RoPE（回転位置埋め込み）カーネル
        kernel void apply_rope_f32(
            device float* x [[buffer(0)]],              // Input/output tensor [seq_len, num_heads, head_dim]
            constant uint& start_pos [[buffer(1)]],     // Starting position for RoPE
            constant uint& seq_len [[buffer(2)]],       // Sequence length
            constant uint& num_heads [[buffer(3)]],     // Number of heads
            constant uint& head_dim [[buffer(4)]],      // Head dimension
            constant float& rope_theta [[buffer(5)]],   // RoPE theta parameter
            uint3 gid [[thread_position_in_grid]]       // (pos, head, dim_pair)
        ) {
            uint pos = gid.x;
            uint head = gid.y;
            uint dim_pair = gid.z;

            if (pos >= seq_len || head >= num_heads || dim_pair >= head_dim / 2) {
                return;
            }

            // Compute absolute position
            uint absolute_pos = start_pos + pos;

            // Compute frequency: 1 / (theta ^ (2 * dim / head_dim))
            uint dim = dim_pair * 2;
            float freq = 1.0f / pow(rope_theta, float(dim) / float(head_dim));
            float angle = float(absolute_pos) * freq;

            float cos_val = cos(angle);
            float sin_val = sin(angle);

            // Compute offsets
            uint head_offset = pos * (num_heads * head_dim) + head * head_dim;

            // Rotate (x[dim], x[dim+1]) pair
            float x0 = x[head_offset + dim];
            float x1 = x[head_offset + dim + 1];

            x[head_offset + dim] = x0 * cos_val - x1 * sin_val;
            x[head_offset + dim + 1] = x0 * sin_val + x1 * cos_val;
        }

        // Attention Score Computation: scores = Q @ K^T / sqrt(head_dim)
        kernel void compute_attention_scores_f32(
            device const float* q [[buffer(0)]],
            device const float* k [[buffer(1)]],
            device float* scores [[buffer(2)]],
            constant uint& q_len [[buffer(3)]],
            constant uint& kv_len [[buffer(4)]],
            constant uint& num_heads [[buffer(5)]],
            constant uint& head_dim [[buffer(6)]],
            constant float& scale [[buffer(7)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint q_pos = gid.x;
            uint kv_pos = gid.y;
            uint head = gid.z;

            if (q_pos >= q_len || kv_pos >= kv_len || head >= num_heads) {
                return;
            }

            uint q_offset = q_pos * (num_heads * head_dim) + head * head_dim;
            uint k_offset = kv_pos * (num_heads * head_dim) + head * head_dim;

            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += q[q_offset + d] * k[k_offset + d];
            }

            uint score_idx = head * q_len * kv_len + q_pos * kv_len + kv_pos;
            scores[score_idx] = dot * scale;
        }

        // Softmax: Find max value per row
        kernel void softmax_max_f32(
            device const float* input [[buffer(0)]],
            device float* max_vals [[buffer(1)]],
            constant uint& q_len [[buffer(2)]],
            constant uint& kv_len [[buffer(3)]],
            constant uint& num_heads [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint q_pos = gid.x;
            uint head = gid.y;

            if (q_pos >= q_len || head >= num_heads) {
                return;
            }

            uint row_offset = head * q_len * kv_len + q_pos * kv_len;
            float max_val = input[row_offset];

            for (uint j = 1; j < kv_len; j++) {
                float val = input[row_offset + j];
                if (val > max_val) {
                    max_val = val;
                }
            }

            max_vals[head * q_len + q_pos] = max_val;
        }

        // Softmax: Compute exp and sum
        kernel void softmax_exp_sum_f32(
            device float* scores [[buffer(0)]],
            device const float* max_vals [[buffer(1)]],
            device float* sum_exp [[buffer(2)]],
            constant uint& q_len [[buffer(3)]],
            constant uint& kv_len [[buffer(4)]],
            constant uint& num_heads [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint q_pos = gid.x;
            uint head = gid.y;

            if (q_pos >= q_len || head >= num_heads) {
                return;
            }

            uint row_offset = head * q_len * kv_len + q_pos * kv_len;
            float max_val = max_vals[head * q_len + q_pos];
            float sum = 0.0f;

            for (uint j = 0; j < kv_len; j++) {
                float exp_val = exp(scores[row_offset + j] - max_val);
                scores[row_offset + j] = exp_val;
                sum += exp_val;
            }

            sum_exp[head * q_len + q_pos] = sum;
        }

        // Softmax: Normalize by sum
        kernel void softmax_normalize_f32(
            device float* scores [[buffer(0)]],
            device const float* sum_exp [[buffer(1)]],
            constant uint& q_len [[buffer(2)]],
            constant uint& kv_len [[buffer(3)]],
            constant uint& num_heads [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint q_pos = gid.x;
            uint head = gid.y;

            if (q_pos >= q_len || head >= num_heads) {
                return;
            }

            uint row_offset = head * q_len * kv_len + q_pos * kv_len;
            float sum = sum_exp[head * q_len + q_pos];

            for (uint j = 0; j < kv_len; j++) {
                scores[row_offset + j] /= sum;
            }
        }

        // Apply attention to values: output = scores @ V
        kernel void apply_attention_to_values_f32(
            device const float* scores [[buffer(0)]],
            device const float* v [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& q_len [[buffer(3)]],
            constant uint& kv_len [[buffer(4)]],
            constant uint& num_heads [[buffer(5)]],
            constant uint& head_dim [[buffer(6)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint q_pos = gid.x;
            uint head = gid.y;
            uint dim = gid.z;

            if (q_pos >= q_len || head >= num_heads || dim >= head_dim) {
                return;
            }

            uint score_row_offset = head * q_len * kv_len + q_pos * kv_len;
            uint out_offset = q_pos * (num_heads * head_dim) + head * head_dim + dim;

            float sum = 0.0f;
            for (uint j = 0; j < kv_len; j++) {
                uint v_offset = j * (num_heads * head_dim) + head * head_dim + dim;
                sum += scores[score_row_offset + j] * v[v_offset];
            }

            output[out_offset] = sum;
        }
        "#;

        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to compile Metal shaders: {:?}", e))
            })?;

        let mut pipeline_states = HashMap::new();

        // Create compute pipeline states
        let add_function = library
            .get_function("elementwise_add_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get add function: {:?}", e))
            })?;
        let add_pipeline = device
            .new_compute_pipeline_state_with_function(&add_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create add pipeline: {:?}", e))
            })?;
        pipeline_states.insert("elementwise_add_f32".to_string(), add_pipeline);

        let matmul_function = library
            .get_function("matmul_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get matmul function: {:?}", e))
            })?;
        let matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&matmul_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create matmul pipeline: {:?}", e))
            })?;
        pipeline_states.insert("matmul_f32".to_string(), matmul_pipeline);

        // Use the same matmul_f32 kernel for tiled operations (for now)
        // TODO: Add optimized tiled kernel to metal_shaders.metal
        let tiled_matmul_function = library
            .get_function("matmul_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get tiled matmul function: {:?}", e))
            })?;
        let tiled_matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&tiled_matmul_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!(
                    "Failed to create tiled matmul pipeline: {:?}",
                    e
                ))
            })?;
        pipeline_states.insert("tiled_matmul_f32".to_string(), tiled_matmul_pipeline);

        let reduce_function = library.get_function("reduce_sum_f32", None).map_err(|e| {
            RusTorchError::KernelError(format!("Failed to get reduce function: {:?}", e))
        })?;
        let reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create reduce pipeline: {:?}", e))
            })?;
        pipeline_states.insert("reduce_sum_f32".to_string(), reduce_pipeline);

        // Register matmul_transposed_f32 kernel
        let matmul_transposed_function = library
            .get_function("matmul_transposed_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get matmul_transposed function: {:?}", e))
            })?;
        let matmul_transposed_pipeline = device
            .new_compute_pipeline_state_with_function(&matmul_transposed_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create matmul_transposed pipeline: {:?}", e))
            })?;
        pipeline_states.insert("matmul_transposed_f32".to_string(), matmul_transposed_pipeline);

        // Register apply_rope_f32 kernel
        let rope_function = library
            .get_function("apply_rope_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get apply_rope function: {:?}", e))
            })?;
        let rope_pipeline = device
            .new_compute_pipeline_state_with_function(&rope_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create apply_rope pipeline: {:?}", e))
            })?;
        pipeline_states.insert("apply_rope_f32".to_string(), rope_pipeline);

        // Register compute_attention_scores_f32 kernel
        let attn_scores_function = library
            .get_function("compute_attention_scores_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get compute_attention_scores function: {:?}", e))
            })?;
        let attn_scores_pipeline = device
            .new_compute_pipeline_state_with_function(&attn_scores_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create compute_attention_scores pipeline: {:?}", e))
            })?;
        pipeline_states.insert("compute_attention_scores_f32".to_string(), attn_scores_pipeline);

        // Register softmax_max_f32 kernel
        let softmax_max_function = library
            .get_function("softmax_max_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get softmax_max function: {:?}", e))
            })?;
        let softmax_max_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_max_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create softmax_max pipeline: {:?}", e))
            })?;
        pipeline_states.insert("softmax_max_f32".to_string(), softmax_max_pipeline);

        // Register softmax_exp_sum_f32 kernel
        let softmax_exp_sum_function = library
            .get_function("softmax_exp_sum_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get softmax_exp_sum function: {:?}", e))
            })?;
        let softmax_exp_sum_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_exp_sum_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create softmax_exp_sum pipeline: {:?}", e))
            })?;
        pipeline_states.insert("softmax_exp_sum_f32".to_string(), softmax_exp_sum_pipeline);

        // Register softmax_normalize_f32 kernel
        let softmax_normalize_function = library
            .get_function("softmax_normalize_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get softmax_normalize function: {:?}", e))
            })?;
        let softmax_normalize_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_normalize_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create softmax_normalize pipeline: {:?}", e))
            })?;
        pipeline_states.insert("softmax_normalize_f32".to_string(), softmax_normalize_pipeline);

        // Register apply_attention_to_values_f32 kernel
        let attn_values_function = library
            .get_function("apply_attention_to_values_f32", None)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to get apply_attention_to_values function: {:?}", e))
            })?;
        let attn_values_pipeline = device
            .new_compute_pipeline_state_with_function(&attn_values_function)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create apply_attention_to_values pipeline: {:?}", e))
            })?;
        pipeline_states.insert("apply_attention_to_values_f32".to_string(), attn_values_pipeline);

        eprintln!("🔍 [METAL INIT] Step 6: All pipelines created successfully");

        // Initialize RAII matmul executor for proper memory management
        // 適切なメモリ管理のためのRAII matmulエグゼキュータを初期化
        eprintln!("🔍 [METAL INIT] Step 7: Initializing RAII matmul executor...");
        let raii_matmul = crate::gpu::metal_matmul_raii::MetalMatMulExecutor::new().ok();
        if raii_matmul.is_none() {
            eprintln!("⚠️ [METAL] Failed to initialize RAII matmul executor, falling back to legacy implementation");
        } else {
            eprintln!("✅ [METAL INIT] RAII matmul executor initialized successfully");
        }

        eprintln!("🔍 [METAL INIT] Step 8: Creating MetalKernelExecutor instance...");

        Ok(Self {
            device,
            command_queue,
            library,
            pipeline_states,
            buffer_cache: RefCell::new(Vec::new()),
            raii_matmul,
        })
    }

    /// Execute element-wise addition using Metal Performance Shaders
    /// Metal Performance Shadersを使用して要素ごと加算を実行
    pub fn elementwise_add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> RusTorchResult<()> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(RusTorchError::InvalidOperation(
                "Array size mismatch in element-wise addition".to_string(),
            ));
        }

        // Create Metal buffers
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline state
        let pipeline_state = self
            .pipeline_states
            .get("elementwise_add_f32")
            .ok_or_else(|| {
                RusTorchError::KernelError("ElementWise pipeline not found".to_string())
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups_per_grid = MTLSize::new(((size + 255) / 256) as u64, 1, 1);

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), size);
        }

        Ok(())
    }

    /// Execute element-wise multiplication using Metal Performance Shaders
    /// Metal Performance Shadersを使用して要素ごと乗算を実行
    pub fn elementwise_mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> RusTorchResult<()> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(RusTorchError::InvalidOperation(
                "Array size mismatch in element-wise multiplication".to_string(),
            ));
        }

        // Create Metal buffers
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline state for element-wise multiplication
        // Note: We need a separate pipeline for elementwise_mul_f32 kernel
        let function_name = "elementwise_mul_f32";
        let function = self.library.get_function(function_name, None).map_err(|e| {
            RusTorchError::KernelError(format!("Failed to get function {}: {:?}", function_name, e))
        })?;

        let pipeline_descriptor = metal::ComputePipelineDescriptor::new();
        pipeline_descriptor.set_compute_function(Some(&function));

        let pipeline_state = self
            .device
            .new_compute_pipeline_state(&pipeline_descriptor)
            .map_err(|e| {
                RusTorchError::KernelError(format!("Failed to create pipeline: {}", e))
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups_per_grid = MTLSize::new(((size + 255) / 256) as u64, 1, 1);

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), size);
        }

        Ok(())
    }

    /// Execute tiled matrix multiplication using Metal for large matrices
    /// 大行列に対してMetalを使用してタイル化行列乗算を実行
    pub fn tiled_matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        const TILE_SIZE: usize = 16; // Optimized for Vega 56

        // Debug: Log tiled matmul parameters
        if std::env::var("RUSTORCH_DEBUG").is_ok() {
            eprintln!("🔧 [TILED MATMUL] m={}, n={}, k={}, a.len()={}, b.len()={}, c.len()={}",
                      m, n, k, a.len(), b.len(), c.len());
        }

        // Validate buffer sizes
        if a.len() != m * k {
            return Err(RusTorchError::InvalidOperation(
                format!("Matrix A size mismatch: expected {}, got {}", m * k, a.len())
            ));
        }
        if b.len() != k * n {
            return Err(RusTorchError::InvalidOperation(
                format!("Matrix B size mismatch: expected {}, got {}", k * n, b.len())
            ));
        }
        if c.len() != m * n {
            return Err(RusTorchError::InvalidOperation(
                format!("Matrix C size mismatch: expected {}, got {}", m * n, c.len())
            ));
        }

        // Create Metal buffers
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (m * k * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (k * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create parameter buffers
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let m_buffer = self.device.new_buffer_with_data(
            &m_u32 as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let n_buffer = self.device.new_buffer_with_data(
            &n_u32 as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let k_buffer = self.device.new_buffer_with_data(
            &k_u32 as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get tiled pipeline state (use MatMul kernel for tiled version)
        let pipeline_state = self
            .pipeline_states
            .get("tiled_matmul_f32")
            .ok_or_else(|| {
                RusTorchError::KernelError("Tiled MatMul pipeline not found".to_string())
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);
        compute_encoder.set_buffer(3, Some(&m_buffer), 0);
        compute_encoder.set_buffer(4, Some(&n_buffer), 0);
        compute_encoder.set_buffer(5, Some(&k_buffer), 0);

        // Set threadgroup memory for tiles (2 tiles * TILE_SIZE * TILE_SIZE * sizeof(f32))
        let threadgroup_memory_size = 2 * TILE_SIZE * TILE_SIZE * std::mem::size_of::<f32>();
        compute_encoder.set_threadgroup_memory_length(0, threadgroup_memory_size as u64);
        compute_encoder.set_threadgroup_memory_length(1, threadgroup_memory_size as u64);

        // Calculate thread configuration for tiled execution
        let threads_per_threadgroup = MTLSize::new(TILE_SIZE as u64, TILE_SIZE as u64, 1);
        let threadgroups_per_grid = MTLSize::new(
            n.div_ceil(TILE_SIZE) as u64,
            m.div_ceil(TILE_SIZE) as u64,
            1,
        );

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), m * n);
        }

        Ok(())
    }

    /// Execute matrix multiplication with B transposed: C = A @ B^T using Metal
    /// B is stored as [n, k] and treated as transposed [k, n]
    pub fn matmul_transposed_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Create Metal buffers
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (m * k * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (n * k * std::mem::size_of::<f32>()) as u64,  // B is [n, k]
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create dimension buffers
        let m_val = m as u32;
        let n_val = n as u32;
        let k_val = k as u32;

        let m_buffer = self.device.new_buffer_with_data(
            &m_val as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_buffer = self.device.new_buffer_with_data(
            &n_val as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let k_buffer = self.device.new_buffer_with_data(
            &k_val as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline state
        let pipeline_state = self
            .pipeline_states
            .get("matmul_transposed_f32")
            .ok_or_else(|| RusTorchError::KernelError("MatMul transposed pipeline not found".to_string()))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);
        compute_encoder.set_buffer(3, Some(&m_buffer), 0);
        compute_encoder.set_buffer(4, Some(&n_buffer), 0);
        compute_encoder.set_buffer(5, Some(&k_buffer), 0);

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(16, 16, 1);
        let threadgroups_per_grid = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), m * n);
        }

        Ok(())
    }

    // ============================================================================
    // Buffer Health Check
    // バッファヘルスチェック
    // ============================================================================

    /// Validate Metal buffer integrity
    /// Metalバッファの整合性を検証
    fn validate_buffer_health(&self, buffer: &Buffer, expected_size: usize, label: &str) -> RusTorchResult<()> {
        // Check if buffer pointer is valid
        // バッファポインタが有効か確認
        let contents_ptr = buffer.contents();
        if contents_ptr.is_null() {
            return Err(RusTorchError::KernelError(format!(
                "[BUFFER HEALTH] {} buffer has NULL contents pointer!", label
            )));
        }

        // Check if buffer length matches expected size
        // バッファ長が期待値と一致するか確認
        let actual_length = buffer.length() as usize;
        if actual_length != expected_size {
            return Err(RusTorchError::KernelError(format!(
                "[BUFFER HEALTH] {} buffer size mismatch: expected {} bytes, got {} bytes",
                label, expected_size, actual_length
            )));
        }

        if std::env::var("RUSTORCH_DEBUG").is_ok() {
            eprintln!("✅ [BUFFER HEALTH] {} buffer OK: {} bytes, ptr={:?}",
                label, actual_length, contents_ptr);
        }

        Ok(())
    }

    // ============================================================================
    // Buffer Cache Management
    // バッファキャッシュ管理
    // ============================================================================

    /// Get a buffer from cache or create a new one
    /// キャッシュからバッファを取得するか、新規作成
    fn get_or_create_buffer(&self, size_bytes: usize) -> Buffer {
        // TEMPORARILY DISABLED: Buffer caching causes memory corruption in sequential matmul
        // 一時的に無効化：バッファキャッシュが連続matmulでメモリ破壊を引き起こす
        // TODO: Implement proper buffer lifetime management

        if std::env::var("RUSTORCH_DEBUG").is_ok() {
            eprintln!("🆕 [BUFFER] Creating new buffer: {} bytes (CACHE DISABLED)", size_bytes);
        }

        self.device.new_buffer(
            size_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )

        /* ORIGINAL CACHE IMPLEMENTATION - DISABLED
        const MAX_CACHE_SIZE: usize = 20;
        const CACHE_TOLERANCE_BYTES: usize = 1024;

        let mut cache = self.buffer_cache.borrow_mut();

        if let Some(idx) = cache.iter().position(|cached| {
            cached.size_bytes >= size_bytes && cached.size_bytes <= size_bytes + CACHE_TOLERANCE_BYTES
        }) {
            let mut cached = cache.swap_remove(idx);
            cached.last_used = Instant::now();

            if std::env::var("RUSTORCH_DEBUG").is_ok() {
                eprintln!("♻️  [BUFFER] Reusing cached buffer: {} bytes", cached.size_bytes);
            }

            return cached.buffer;
        }

        if std::env::var("RUSTORCH_DEBUG").is_ok() {
            eprintln!("🆕 [BUFFER] Creating new buffer: {} bytes (cache size: {})", size_bytes, cache.len());
        }

        self.device.new_buffer(
            size_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
        */
    }

    /// Return a buffer to cache for reuse
    /// バッファをキャッシュに返却して再利用可能にする
    fn return_buffer_to_cache(&self, _buffer: Buffer, _size_bytes: usize) {
        // TEMPORARILY DISABLED: Let buffers be dropped immediately
        // 一時的に無効化：バッファを即座にdropさせる
        // No-op: buffer will be dropped when this function returns
        // 何もしない：この関数が戻る時にバッファがdropされる
    }

    /// Force cleanup of accumulated Metal resources
    /// 蓄積されたMetalリソースを強制的にクリーンアップ
    pub fn force_cleanup(&self) {
        // Drain the autoreleasepool by creating and immediately dropping a new one
        // 新しいautoreleasepoolを作成して即座にdropすることで既存のプールをドレインする
        objc::rc::autoreleasepool(|| {
            // Empty autoreleasepool to drain accumulated objects
            // 蓄積されたオブジェクトをドレインするための空のautoreleasepool
            if std::env::var("RUSTORCH_DEBUG").is_ok() {
                eprintln!("🧹 [CLEANUP] Draining Metal autoreleasepool");
            }
        });

        // Also clear buffer cache if needed
        // 必要に応じてバッファキャッシュもクリア
        let mut cache = self.buffer_cache.borrow_mut();
        let cache_size = cache.len();
        cache.clear();

        if std::env::var("RUSTORCH_DEBUG").is_ok() && cache_size > 0 {
            eprintln!("🧹 [CLEANUP] Cleared {} cached buffers", cache_size);
        }
    }

    // ============================================================================
    // Matrix Operations
    // 行列演算
    // ============================================================================

    /// Execute standard matrix multiplication using Metal
    /// Metalを使用して標準行列乗算を実行
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Use RAII matmul executor if available (proper memory management)
        // RAII matmulエグゼキュータが利用可能なら使用（適切なメモリ管理）
        if let Some(ref raii_matmul) = self.raii_matmul {
            eprintln!("✅ [METAL RAII] Using RAII matmul executor ({}x{}x{})", m, n, k);
            raii_matmul.matmul_f32(a, b, c, m, n, k)
        } else {
            // Fallback to legacy implementation
            // レガシー実装にフォールバック
            eprintln!("⚠️ [METAL] Using legacy matmul implementation (RAII unavailable)");
            self.matmul_f32_legacy(a, b, c, m, n, k)
        }
    }

    /// Legacy matmul implementation (fallback)
    /// レガシーmatmul実装（フォールバック）
    fn matmul_f32_legacy(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // matmul_f32_impl wraps everything in autoreleasepool for immediate cleanup
        // matmul_f32_implがすべてをautoreleasepoolで囲んで即座にクリーンアップ
        self.matmul_f32_impl(a, b, c, m, n, k)
    }

    fn matmul_f32_impl(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // CRITICAL: Use direct Objective-C autoreleasepool FFI
        // 重要：Objective-CのautoreleasepoolFFIを直接使用
        use crate::gpu::objc_bridge::with_autoreleasepool;

        with_autoreleasepool(|| {
            self.matmul_f32_inner(a, b, c, m, n, k)
        })
    }

    fn matmul_f32_inner(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Calculate buffer sizes
        // バッファサイズを計算
        let a_size_bytes = m * k * std::mem::size_of::<f32>();
        let b_size_bytes = k * n * std::mem::size_of::<f32>();
        let c_size_bytes = m * n * std::mem::size_of::<f32>();
        let param_size_bytes = std::mem::size_of::<u32>();

        // Get buffers from cache or create new ones
        // キャッシュからバッファを取得、または新規作成
        let a_buffer = self.get_or_create_buffer(a_size_bytes);
        let b_buffer = self.get_or_create_buffer(b_size_bytes);
        let c_buffer = self.get_or_create_buffer(c_size_bytes);
        let m_buffer = self.get_or_create_buffer(param_size_bytes);
        let n_buffer = self.get_or_create_buffer(param_size_bytes);
        let k_buffer = self.get_or_create_buffer(param_size_bytes);

        // Validate buffer health after creation
        // バッファ作成後のヘルスチェック
        self.validate_buffer_health(&a_buffer, a_size_bytes, "A (input)")?;
        self.validate_buffer_health(&b_buffer, b_size_bytes, "B (weight)")?;
        self.validate_buffer_health(&c_buffer, c_size_bytes, "C (output)")?;
        self.validate_buffer_health(&m_buffer, param_size_bytes, "M (param)")?;
        self.validate_buffer_health(&n_buffer, param_size_bytes, "N (param)")?;
        self.validate_buffer_health(&k_buffer, param_size_bytes, "K (param)")?;

        // Copy input data to buffers
        // 入力データをバッファにコピー
        unsafe {
            std::ptr::copy_nonoverlapping(
                a.as_ptr(),
                a_buffer.contents() as *mut f32,
                m * k,
            );
            std::ptr::copy_nonoverlapping(
                b.as_ptr(),
                b_buffer.contents() as *mut f32,
                k * n,
            );

            // Copy parameters
            let m_u32 = m as u32;
            let n_u32 = n as u32;
            let k_u32 = k as u32;
            std::ptr::copy_nonoverlapping(
                &m_u32 as *const u32,
                m_buffer.contents() as *mut u32,
                1,
            );
            std::ptr::copy_nonoverlapping(
                &n_u32 as *const u32,
                n_buffer.contents() as *mut u32,
                1,
            );
            std::ptr::copy_nonoverlapping(
                &k_u32 as *const u32,
                k_buffer.contents() as *mut u32,
                1,
            );
        }

        // Validate buffers again after data copy
        // データコピー後の再検証
        self.validate_buffer_health(&a_buffer, a_size_bytes, "A (after copy)")?;
        self.validate_buffer_health(&b_buffer, b_size_bytes, "B (after copy)")?;
        self.validate_buffer_health(&c_buffer, c_size_bytes, "C (after copy)")?;

        // Get pipeline state
        let pipeline_state = self
            .pipeline_states
            .get("matmul_f32")
            .ok_or_else(|| RusTorchError::KernelError("MatMul pipeline not found".to_string()))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);
        compute_encoder.set_buffer(3, Some(&m_buffer), 0);
        compute_encoder.set_buffer(4, Some(&n_buffer), 0);
        compute_encoder.set_buffer(5, Some(&k_buffer), 0);

        // Final validation before GPU execution
        // GPU実行前の最終検証
        self.validate_buffer_health(&a_buffer, a_size_bytes, "A (before GPU)")?;
        self.validate_buffer_health(&b_buffer, b_size_bytes, "B (before GPU)")?;
        self.validate_buffer_health(&c_buffer, c_size_bytes, "C (before GPU)")?;

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(16, 16, 1);
        let threadgroups_per_grid = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        // 結果をホストにコピー
        unsafe {
            let result_ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), m * n);
        }

        // Buffers will be automatically released when autoreleasepool exits
        // バッファはautoreleasepool終了時に自動的に解放される
        Ok(())
    }

    /// Perform matrix multiplication using Metal with result return
    /// Metalを使用して行列乗算を実行し結果を返す  
    pub fn matrix_multiply_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(RusTorchError::InvalidOperation(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![0.0f32; m * n];
        self.matmul_f32(a, b, &mut result, m, n, k)?;
        Ok(result)
    }

    /// Execute 2D convolution using Metal GPU
    /// Metal GPUを使用して2D畳み込みを実行
    pub fn conv2d_f32(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        input_height: usize,
        input_width: usize,
        input_channels: usize,
        output_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> RusTorchResult<()> {
        // Calculate output dimensions
        let output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;

        // For now, implement as optimized matrix multiplication (im2col + GEMM approach)
        // This is a common approach in high-performance convolution implementations

        // Convert convolution to matrix multiplication using im2col
        let patch_size = kernel_height * kernel_width * input_channels;
        let num_patches = output_height * output_width;

        // Create im2col matrix: [patch_size, num_patches]
        let mut im2col_matrix = vec![0.0f32; patch_size * num_patches];

        // Extract patches (im2col operation)
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                let patch_idx = out_h * output_width + out_w;
                let start_h = out_h * stride_h;
                let start_w = out_w * stride_w;

                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        for ch in 0..input_channels {
                            let in_h = start_h + kh;
                            let in_w = start_w + kw;

                            let im2col_row =
                                ch * kernel_height * kernel_width + kh * kernel_width + kw;
                            let im2col_idx = im2col_row * num_patches + patch_idx;

                            if in_h >= pad_h
                                && in_h < input_height + pad_h
                                && in_w >= pad_w
                                && in_w < input_width + pad_w
                            {
                                let actual_in_h = in_h - pad_h;
                                let actual_in_w = in_w - pad_w;
                                let input_idx = ch * input_height * input_width
                                    + actual_in_h * input_width
                                    + actual_in_w;
                                im2col_matrix[im2col_idx] = input[input_idx];
                            } else {
                                im2col_matrix[im2col_idx] = 0.0; // Padding
                            }
                        }
                    }
                }
            }
        }

        // Use Metal matrix multiplication: kernel [output_channels, patch_size] * im2col [patch_size, num_patches]
        let mut result_matrix = vec![0.0f32; output_channels * num_patches];
        self.matmul_f32(
            kernel,
            &im2col_matrix,
            &mut result_matrix,
            output_channels,
            num_patches,
            patch_size,
        )?;

        // Copy result to output with proper layout
        for out_ch in 0..output_channels {
            for out_h in 0..output_height {
                for out_w in 0..output_width {
                    let result_idx = out_ch * num_patches + out_h * output_width + out_w;
                    let output_idx =
                        out_ch * output_height * output_width + out_h * output_width + out_w;
                    output[output_idx] = result_matrix[result_idx];
                }
            }
        }

        Ok(())
    }

    /// Execute reduction operation (sum) using Metal
    /// Metalを使用してリダクション演算（合計）を実行
    pub fn reduce_sum_f32(&self, input: &[f32]) -> RusTorchResult<f32> {
        let size = input.len();
        let block_size = 256;
        let grid_size = size.div_ceil(block_size);

        // Create Metal buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (grid_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline state
        let pipeline_state = self
            .pipeline_states
            .get("reduce_sum_f32")
            .ok_or_else(|| {
                RusTorchError::KernelError("Reduction pipeline not found".to_string())
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(pipeline_state);
        compute_encoder.set_buffer(0, Some(&input_buffer), 0);
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);

        // Set threadgroup memory
        compute_encoder
            .set_threadgroup_memory_length(0, (block_size * std::mem::size_of::<f32>()) as u64);

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(block_size as u64, 1, 1);
        let threadgroups_per_grid = MTLSize::new(grid_size as u64, 1, 1);

        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy partial results back and sum on CPU
        let mut partial_results = vec![0.0f32; grid_size];
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, partial_results.as_mut_ptr(), grid_size);
        }

        Ok(partial_results.iter().sum())
    }

    /// Execute ReLU activation using Metal GPU
    /// Metal GPUを使用してReLU活性化を実行
    pub fn relu_f32(&self, input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for ReLU
        let function = self
            .library
            .get_function("relu_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get ReLU function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create ReLU pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute Sigmoid activation using Metal GPU
    /// Metal GPUを使用してSigmoid活性化を実行
    pub fn sigmoid_f32(&self, input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for Sigmoid
        let function = self
            .library
            .get_function("sigmoid_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get Sigmoid function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                RusTorchError::gpu(&format!("Failed to create Sigmoid pipeline: {}", e))
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute Tanh activation using Metal GPU
    /// Metal GPUを使用してTanh活性化を実行
    pub fn tanh_f32(&self, input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for Tanh
        let function = self
            .library
            .get_function("tanh_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get Tanh function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create Tanh pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute GELU activation using Metal GPU
    /// Metal GPUを使用してGELU活性化を実行
    pub fn gelu_f32(&self, input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for GELU
        let function = self
            .library
            .get_function("gelu_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get GELU function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create GELU pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute Leaky ReLU activation using Metal GPU
    /// Metal GPUを使用してLeaky ReLU活性化を実行
    pub fn leaky_relu_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        alpha: f32,
    ) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for Leaky ReLU
        let function = self
            .library
            .get_function("leaky_relu_f32", None)
            .map_err(|e| {
                RusTorchError::gpu(&format!("Failed to get Leaky ReLU function: {}", e))
            })?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                RusTorchError::gpu(&format!("Failed to create Leaky ReLU pipeline: {}", e))
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &alpha as *const f32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute ELU activation using Metal GPU
    /// Metal GPUを使用してELU活性化を実行
    pub fn elu_f32(&self, input: &[f32], output: &mut [f32], alpha: f32) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for ELU
        let function = self
            .library
            .get_function("elu_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get ELU function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create ELU pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &alpha as *const f32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute Swish activation using Metal GPU
    /// Metal GPUを使用してSwish活性化を実行
    pub fn swish_f32(&self, input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
        if input.len() != output.len() {
            return Err(RusTorchError::InvalidOperation(
                "Input and output lengths must match".to_string(),
            ));
        }

        let n = input.len();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for Swish
        let function = self
            .library
            .get_function("swish_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get Swish function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create Swish pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const c_void,
        );

        // Dispatch threads
        let threads_per_threadgroup = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(((n + 255) / 256).try_into().unwrap(), 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), n);
        }

        Ok(())
    }

    /// Execute LayerNorm using Metal GPU (f32 version)
    /// Metal GPUを使用してLayerNormを実行（f32版）
    pub fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        gamma: &[f32],
        beta: &[f32],
        batch_size: usize,
        seq_len: usize,
        features: usize,
        eps: f32,
    ) -> RusTorchResult<()> {
        if input.len() != batch_size * seq_len * features {
            return Err(RusTorchError::InvalidOperation(
                "Input size mismatch".to_string(),
            ));
        }
        if output.len() != input.len() {
            return Err(RusTorchError::InvalidOperation(
                "Output size mismatch".to_string(),
            ));
        }
        if gamma.len() != features || beta.len() != features {
            return Err(RusTorchError::InvalidOperation(
                "Gamma/Beta size mismatch".to_string(),
            ));
        }

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (input.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (output.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let gamma_buffer = self.device.new_buffer_with_data(
            gamma.as_ptr() as *const c_void,
            (features * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let beta_buffer = self.device.new_buffer_with_data(
            beta.as_ptr() as *const c_void,
            (features * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for LayerNorm
        let function = self
            .library
            .get_function("layer_norm_f32", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get LayerNorm function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create LayerNorm pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&gamma_buffer), 0);
        encoder.set_buffer(3, Some(&beta_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(seq_len as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &(features as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const c_void,
        );

        // Dispatch threads: one thread per (batch, sequence) position
        let threads_per_threadgroup = metal::MTLSize::new(16, 16, 1);
        let threadgroups = metal::MTLSize::new(
            seq_len.div_ceil(16).try_into().unwrap(),
            batch_size.div_ceil(16).try_into().unwrap(),
            1,
        );

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), output.len());
        }

        Ok(())
    }

    /// Execute LayerNorm using Metal GPU (f64 version)
    /// Metal GPUを使用してLayerNormを実行（f64版）
    pub fn layer_norm_f64(
        &self,
        input: &[f64],
        output: &mut [f64],
        gamma: &[f64],
        beta: &[f64],
        batch_size: usize,
        seq_len: usize,
        features: usize,
        eps: f64,
    ) -> RusTorchResult<()> {
        if input.len() != batch_size * seq_len * features {
            return Err(RusTorchError::InvalidOperation(
                "Input size mismatch".to_string(),
            ));
        }
        if output.len() != input.len() {
            return Err(RusTorchError::InvalidOperation(
                "Output size mismatch".to_string(),
            ));
        }
        if gamma.len() != features || beta.len() != features {
            return Err(RusTorchError::InvalidOperation(
                "Gamma/Beta size mismatch".to_string(),
            ));
        }

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (input.len() * std::mem::size_of::<f64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (output.len() * std::mem::size_of::<f64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let gamma_buffer = self.device.new_buffer_with_data(
            gamma.as_ptr() as *const c_void,
            (features * std::mem::size_of::<f64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let beta_buffer = self.device.new_buffer_with_data(
            beta.as_ptr() as *const c_void,
            (features * std::mem::size_of::<f64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get or create pipeline state for LayerNorm
        let function = self
            .library
            .get_function("layer_norm_f64", None)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to get LayerNorm function: {}", e)))?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RusTorchError::gpu(&format!("Failed to create LayerNorm pipeline: {}", e)))?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&gamma_buffer), 0);
        encoder.set_buffer(3, Some(&beta_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(seq_len as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &(features as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<f64>() as u64,
            &eps as *const f64 as *const c_void,
        );

        // Dispatch threads: one thread per (batch, sequence) position
        let threads_per_threadgroup = metal::MTLSize::new(16, 16, 1);
        let threadgroups = metal::MTLSize::new(
            seq_len.div_ceil(16).try_into().unwrap(),
            batch_size.div_ceil(16).try_into().unwrap(),
            1,
        );

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f64;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), output.len());
        }

        Ok(())
    }

    /// Apply RoPE (Rotary Position Embedding) using Metal GPU
    /// Metal GPUを使用してRoPE（回転位置埋め込み）を適用
    pub fn apply_rope_f32(
        &self,
        x: &mut [f32],
        start_pos: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> RusTorchResult<()> {
        let expected_size = seq_len * num_heads * head_dim;
        if x.len() != expected_size {
            return Err(RusTorchError::InvalidOperation(format!(
                "Input size mismatch: expected {}, got {}",
                expected_size,
                x.len()
            )));
        }

        // Get pipeline state for apply_rope_f32
        let pipeline_state = self
            .pipeline_states
            .get("apply_rope_f32")
            .ok_or_else(|| RusTorchError::KernelError("apply_rope_f32 pipeline not found".to_string()))?;

        // Create Metal buffer for x (input/output)
        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&x_buffer), 0);

        // Set scalar parameters
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &(start_pos as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(seq_len as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(num_heads as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(head_dim as u32) as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<f32>() as u64,
            &rope_theta as *const f32 as *const c_void,
        );

        // Dispatch 3D thread grid: (seq_len, num_heads, head_dim/2)
        let threads_per_threadgroup = metal::MTLSize::new(8, 8, 4);
        let threadgroups = metal::MTLSize::new(
            seq_len.div_ceil(8).try_into().unwrap(),
            num_heads.div_ceil(8).try_into().unwrap(),
            (head_dim / 2).div_ceil(4).try_into().unwrap(),
        );

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = x_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, x.as_mut_ptr(), x.len());
        }

        Ok(())
    }

    /// Compute attention scores and apply softmax+values in one call
    /// Attention scoreを計算し、softmax+valuesを一括適用
    pub fn compute_attention_with_softmax_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        q_len: usize,
        kv_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> RusTorchResult<()> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores_size = num_heads * q_len * kv_len;
        let max_vals_size = num_heads * q_len;
        let sum_exp_size = num_heads * q_len;

        // Get pipeline states
        let attn_scores_pipeline = self
            .pipeline_states
            .get("compute_attention_scores_f32")
            .ok_or_else(|| RusTorchError::KernelError("compute_attention_scores_f32 pipeline not found".to_string()))?;
        let softmax_max_pipeline = self
            .pipeline_states
            .get("softmax_max_f32")
            .ok_or_else(|| RusTorchError::KernelError("softmax_max_f32 pipeline not found".to_string()))?;
        let softmax_exp_sum_pipeline = self
            .pipeline_states
            .get("softmax_exp_sum_f32")
            .ok_or_else(|| RusTorchError::KernelError("softmax_exp_sum_f32 pipeline not found".to_string()))?;
        let softmax_normalize_pipeline = self
            .pipeline_states
            .get("softmax_normalize_f32")
            .ok_or_else(|| RusTorchError::KernelError("softmax_normalize_f32 pipeline not found".to_string()))?;
        let attn_values_pipeline = self
            .pipeline_states
            .get("apply_attention_to_values_f32")
            .ok_or_else(|| RusTorchError::KernelError("apply_attention_to_values_f32 pipeline not found".to_string()))?;

        // Create Metal buffers
        let q_buffer = self.device.new_buffer_with_data(
            q.as_ptr() as *const c_void,
            (q.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let k_buffer = self.device.new_buffer_with_data(
            k.as_ptr() as *const c_void,
            (k.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let v_buffer = self.device.new_buffer_with_data(
            v.as_ptr() as *const c_void,
            (v.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scores_buffer = self.device.new_buffer(
            (scores_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let max_vals_buffer = self.device.new_buffer(
            (max_vals_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let sum_exp_buffer = self.device.new_buffer(
            (sum_exp_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (output.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();

        // Step 1: Compute attention scores
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(attn_scores_pipeline);
            encoder.set_buffer(0, Some(&q_buffer), 0);
            encoder.set_buffer(1, Some(&k_buffer), 0);
            encoder.set_buffer(2, Some(&scores_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(q_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(kv_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(num_heads as u32) as *const u32 as *const c_void);
            encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &(head_dim as u32) as *const u32 as *const c_void);
            encoder.set_bytes(7, std::mem::size_of::<f32>() as u64, &scale as *const f32 as *const c_void);

            let threads_per_threadgroup = metal::MTLSize::new(8, 8, 4);
            let threadgroups = metal::MTLSize::new(
                q_len.div_ceil(8).try_into().unwrap(),
                kv_len.div_ceil(8).try_into().unwrap(),
                num_heads.div_ceil(4).try_into().unwrap(),
            );
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
        }

        // Step 2: Softmax max
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(softmax_max_pipeline);
            encoder.set_buffer(0, Some(&scores_buffer), 0);
            encoder.set_buffer(1, Some(&max_vals_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &(q_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(kv_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(num_heads as u32) as *const u32 as *const c_void);

            let threads_per_threadgroup = metal::MTLSize::new(16, 16, 1);
            let threadgroups = metal::MTLSize::new(
                q_len.div_ceil(16).try_into().unwrap(),
                num_heads.div_ceil(16).try_into().unwrap(),
                1,
            );
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
        }

        // Step 3: Softmax exp and sum
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(softmax_exp_sum_pipeline);
            encoder.set_buffer(0, Some(&scores_buffer), 0);
            encoder.set_buffer(1, Some(&max_vals_buffer), 0);
            encoder.set_buffer(2, Some(&sum_exp_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(q_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(kv_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(num_heads as u32) as *const u32 as *const c_void);

            let threads_per_threadgroup = metal::MTLSize::new(16, 16, 1);
            let threadgroups = metal::MTLSize::new(
                q_len.div_ceil(16).try_into().unwrap(),
                num_heads.div_ceil(16).try_into().unwrap(),
                1,
            );
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
        }

        // Step 4: Softmax normalize
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(softmax_normalize_pipeline);
            encoder.set_buffer(0, Some(&scores_buffer), 0);
            encoder.set_buffer(1, Some(&sum_exp_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &(q_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(kv_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(num_heads as u32) as *const u32 as *const c_void);

            let threads_per_threadgroup = metal::MTLSize::new(16, 16, 1);
            let threadgroups = metal::MTLSize::new(
                q_len.div_ceil(16).try_into().unwrap(),
                num_heads.div_ceil(16).try_into().unwrap(),
                1,
            );
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
        }

        // Step 5: Apply attention to values
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(attn_values_pipeline);
            encoder.set_buffer(0, Some(&scores_buffer), 0);
            encoder.set_buffer(1, Some(&v_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(q_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(kv_len as u32) as *const u32 as *const c_void);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(num_heads as u32) as *const u32 as *const c_void);
            encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &(head_dim as u32) as *const u32 as *const c_void);

            let threads_per_threadgroup = metal::MTLSize::new(8, 8, 4);
            let threadgroups = metal::MTLSize::new(
                q_len.div_ceil(8).try_into().unwrap(),
                num_heads.div_ceil(8).try_into().unwrap(),
                head_dim.div_ceil(4).try_into().unwrap(),
            );
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to host
        unsafe {
            let result_ptr = output_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(result_ptr, output.as_mut_ptr(), output.len());
        }

        Ok(())
    }
}

/// Non-Metal fallback executor for compatibility
/// 互換性のための非Metalフォールバック実行器
#[cfg(not(feature = "metal"))]
pub struct MetalKernelExecutor;

#[cfg(not(feature = "metal"))]
impl MetalKernelExecutor {
    /// Create a new Metal kernel executor (fallback implementation)
    /// 新しいMetalカーネル実行器を作成（フォールバック実装）
    pub fn new() -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }

    /// Perform element-wise addition using Metal
    /// Metalを使用して要素ごとの加算を実行
    pub fn elementwise_add_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }

    /// Perform matrix multiplication using Metal
    /// Metalを使用して行列乗算を実行
    pub fn matmul_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }

    /// Perform reduction sum using Metal
    /// Metalを使用してリダクション合計を実行
    pub fn reduce_sum_f32(&self, _input: &[f32]) -> RusTorchResult<f32> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
}

/// Public interface functions for Metal operations
/// Metal演算のためのパブリックインターフェース関数
///
/// Execute Metal matrix multiplication
/// Metal行列乗算を実行
pub fn metal_matmul_f32(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "metal")]
    {
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        if let Some(ref executor) = *executor_guard {
            executor.matmul_f32(_a, _b, _c, _m, _n, _k)
        } else {
            Err(RusTorchError::tensor_op("Metal executor not initialized"))
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
}

/// Execute Metal element-wise addition
/// Metal要素ごと加算を実行
pub fn metal_elementwise_add_f32(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> RusTorchResult<()> {
    #[cfg(feature = "metal")]
    {
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        if let Some(ref executor) = *executor_guard {
            executor.elementwise_add_f32(_a, _b, _c)
        } else {
            Err(RusTorchError::tensor_op("Metal executor not initialized"))
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
}

/// Execute Metal reduction sum
/// Metalリダクション合計を実行
pub fn metal_reduce_sum_f32(_input: &[f32]) -> RusTorchResult<f32> {
    #[cfg(feature = "metal")]
    {
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        if let Some(ref executor) = *executor_guard {
            executor.reduce_sum_f32(_input)
        } else {
            Err(RusTorchError::tensor_op("Metal executor not initialized"))
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
}

/// Execute Metal 2D convolution
/// Metal 2D畳み込みを実行
pub fn metal_conv2d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    input_height: usize,
    input_width: usize,
    input_channels: usize,
    output_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "metal")]
    {
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        if let Some(ref executor) = *executor_guard {
            executor.conv2d_f32(
                input,
                kernel,
                output,
                input_height,
                input_width,
                input_channels,
                output_channels,
                kernel_height,
                kernel_width,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
            )
        } else {
            Err(RusTorchError::tensor_op("Metal executor not initialized"))
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
}

/// Execute ReLU activation using Metal GPU
/// Metal GPUを使用してReLU活性化を実行
#[cfg(feature = "metal")]
pub fn metal_relu_f32(input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.relu_f32(input, output)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_relu_f32(_input: &[f32], _output: &mut [f32]) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute Sigmoid activation using Metal GPU
/// Metal GPUを使用してSigmoid活性化を実行
#[cfg(feature = "metal")]
pub fn metal_sigmoid_f32(input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.sigmoid_f32(input, output)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_sigmoid_f32(_input: &[f32], _output: &mut [f32]) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute Tanh activation using Metal GPU
/// Metal GPUを使用してTanh活性化を実行
#[cfg(feature = "metal")]
pub fn metal_tanh_f32(input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.tanh_f32(input, output)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_tanh_f32(_input: &[f32], _output: &mut [f32]) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute GELU activation using Metal GPU
/// Metal GPUを使用してGELU活性化を実行
#[cfg(feature = "metal")]
pub fn metal_gelu_f32(input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.gelu_f32(input, output)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_gelu_f32(_input: &[f32], _output: &mut [f32]) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute Leaky ReLU activation using Metal GPU
/// Metal GPUを使用してLeaky ReLU活性化を実行
#[cfg(feature = "metal")]
pub fn metal_leaky_relu_f32(input: &[f32], output: &mut [f32], alpha: f32) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.leaky_relu_f32(input, output, alpha)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_leaky_relu_f32(
    _input: &[f32],
    _output: &mut [f32],
    _alpha: f32,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute ELU activation using Metal GPU
/// Metal GPUを使用してELU活性化を実行
#[cfg(feature = "metal")]
pub fn metal_elu_f32(input: &[f32], output: &mut [f32], alpha: f32) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.elu_f32(input, output, alpha)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_elu_f32(_input: &[f32], _output: &mut [f32], _alpha: f32) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute Swish activation using Metal GPU
/// Metal GPUを使用してSwish活性化を実行
#[cfg(feature = "metal")]
pub fn metal_swish_f32(input: &[f32], output: &mut [f32]) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.swish_f32(input, output)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_swish_f32(_input: &[f32], _output: &mut [f32]) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute LayerNorm using Metal GPU (f32 version)
/// Metal GPUを使用してLayerNormを実行（f32版）
#[cfg(feature = "metal")]
pub fn metal_layer_norm_f32(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    seq_len: usize,
    features: usize,
    eps: f32,
) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.layer_norm_f32(input, output, gamma, beta, batch_size, seq_len, features, eps)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_layer_norm_f32(
    _input: &[f32],
    _output: &mut [f32],
    _gamma: &[f32],
    _beta: &[f32],
    _batch_size: usize,
    _seq_len: usize,
    _features: usize,
    _eps: f32,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Execute LayerNorm using Metal GPU (f64 version)
/// Metal GPUを使用してLayerNormを実行（f64版）
#[cfg(feature = "metal")]
pub fn metal_layer_norm_f64(
    input: &[f64],
    output: &mut [f64],
    gamma: &[f64],
    beta: &[f64],
    batch_size: usize,
    seq_len: usize,
    features: usize,
    eps: f64,
) -> RusTorchResult<()> {
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    if let Some(ref executor) = *executor_guard {
        executor.layer_norm_f64(input, output, gamma, beta, batch_size, seq_len, features, eps)
    } else {
        Err(RusTorchError::tensor_op("Metal executor not initialized"))
    }
}

#[cfg(not(feature = "metal"))]
pub fn metal_layer_norm_f64(
    _input: &[f64],
    _output: &mut [f64],
    _gamma: &[f64],
    _beta: &[f64],
    _batch_size: usize,
    _seq_len: usize,
    _features: usize,
    _eps: f64,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_kernel_params() {
        let params = MetalKernelParams::default();
        assert_eq!(params.threads_per_threadgroup, (1, 1, 1));
        assert_eq!(params.threadgroups_per_grid, (1, 1, 1));
    }

    #[test]
    fn test_metal_executor_creation() {
        let result = MetalKernelExecutor::new();
        #[cfg(not(feature = "metal"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_metal_kernel_types() {
        assert_eq!(MetalKernelType::ElementWise, MetalKernelType::ElementWise);
        assert_ne!(MetalKernelType::ElementWise, MetalKernelType::MatMul);
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_matmul_transposed_small() {
        // テストケース: 3x4 @ (5x4)^T = 3x5
        let m = 3;
        let n = 5;
        let k = 4;

        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ];

        let b: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0
        ];

        let c_expected: Vec<f32> = vec![
            3.0, 7.0, 11.0, 15.0, 19.0,
            7.0, 17.4, 27.8, 38.2, 48.6,
            11.0, 27.8, 44.6, 61.4, 78.2
        ];

        let mut c_result = vec![0.0f32; m * n];

        // Metal実行器を取得
        let executor_mutex = MetalKernelExecutor::get().expect("Metal executor initialization failed");
        let executor_guard = executor_mutex.lock().unwrap();
        let executor = executor_guard.as_ref().expect("Metal executor not initialized");

        // 転置行列乗算を実行
        executor.matmul_transposed_f32(&a, &b, &mut c_result, m, n, k)
            .expect("matmul_transposed_f32 failed");

        // 結果を検証 (相対誤差 < 1e-5)
        eprintln!("\n=== 転置行列乗算テスト結果 ===");
        eprintln!("Metal結果 C [{}x{}]:", m, n);
        for i in 0..m {
            eprint!("  ");
            for j in 0..n {
                eprint!("{:6.2} ", c_result[i * n + j]);
            }
            eprintln!();
        }

        let mut max_rel_error = 0.0f32;
        for i in 0..m * n {
            let expected = c_expected[i];
            let actual = c_result[i];
            let abs_error = (expected - actual).abs();
            let rel_error = if expected.abs() > 1e-6 {
                abs_error / expected.abs()
            } else {
                abs_error
            };

            max_rel_error = max_rel_error.max(rel_error);

            assert!(
                rel_error < 1e-5,
                "要素 {} の相対誤差が大きすぎます: 期待値={}, 実際={}, 相対誤差={}",
                i, expected, actual, rel_error
            );
        }

        eprintln!("最大相対誤差: {:.2e}", max_rel_error);
        eprintln!("✅ テスト成功: すべての要素が期待値と一致");
    }
}
