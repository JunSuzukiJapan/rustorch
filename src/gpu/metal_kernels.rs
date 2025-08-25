//! Metal Performance Shaders kernel implementations for GPU acceleration
//! GPU加速のためのMetal Performance Shadersカーネル実装

use crate::error::{RusTorchError, RusTorchResult};
// Metal GPU kernel implementations
use std::ffi::c_void;
use std::marker::PhantomData;

#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    Library, MTLResourceOptions, MTLSize,
};
#[cfg(feature = "metal")]
use objc::runtime::Object;

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
                let buffer_ptr = self.buffer as *mut T;
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
                let buffer_ptr = self.buffer as *const T;
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

/// Metal kernel executor for high-performance GPU operations
/// 高性能GPU演算のためのMetalカーネル実行器
#[cfg(feature = "metal")]
pub struct MetalKernelExecutor {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    pipeline_states: HashMap<MetalKernelType, ComputePipelineState>,
}

#[cfg(feature = "metal")]
impl MetalKernelExecutor {
    /// Create a new Metal kernel executor
    /// 新しいMetalカーネル実行器を作成
    pub fn new() -> RusTorchResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            RusTorchError::InitializationError("No Metal device available".to_string())
        })?;

        let command_queue = device.new_command_queue();

        // Metal shader library source
        let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void elementwise_add_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            result[index] = a[index] + b[index];
        }
        
        kernel void elementwise_mul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint index [[thread_position_in_grid]]
        ) {
            result[index] = a[index] * b[index];
        }
        
        kernel void matrix_multiply_f32(
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
            for (uint k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
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
        "#;

        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to compile Metal shaders: {:?}",
                    e
                ))
            })?;

        let mut pipeline_states = HashMap::new();

        // Create compute pipeline states
        let add_function = library
            .get_function("elementwise_add_f32", None)
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to get add function: {:?}",
                    e
                ))
            })?;
        let add_pipeline = device
            .new_compute_pipeline_state_with_function(&add_function)
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to create add pipeline: {:?}",
                    e
                ))
            })?;
        pipeline_states.insert(MetalKernelType::ElementWise, add_pipeline);

        let matmul_function = library
            .get_function("matrix_multiply_f32", None)
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to get matmul function: {:?}",
                    e
                ))
            })?;
        let matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&matmul_function)
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to create matmul pipeline: {:?}",
                    e
                ))
            })?;
        pipeline_states.insert(MetalKernelType::MatMul, matmul_pipeline);

        let reduce_function = library.get_function("reduce_sum_f32", None).map_err(|e| {
            RusTorchError::KernelCompilationError(format!("Failed to get reduce function: {:?}", e))
        })?;
        let reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce_function)
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to create reduce pipeline: {:?}",
                    e
                ))
            })?;
        pipeline_states.insert(MetalKernelType::Reduction, reduce_pipeline);

        Ok(Self {
            device,
            command_queue,
            library,
            pipeline_states,
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
            .get(&MetalKernelType::ElementWise)
            .ok_or_else(|| {
                RusTorchError::KernelExecutionError("ElementWise pipeline not found".to_string())
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
        let threadgroups_per_grid = MTLSize::new((size + 255) / 256, 1, 1);

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

    /// Execute matrix multiplication using Metal
    /// Metalを使用して行列乗算を実行
    pub fn matmul_f32(
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
            (k * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create parameter buffers
        let m_buffer = self.device.new_buffer_with_data(
            &m as *const usize as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let n_buffer = self.device.new_buffer_with_data(
            &n as *const usize as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let k_buffer = self.device.new_buffer_with_data(
            &k as *const usize as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline state
        let pipeline_state = self
            .pipeline_states
            .get(&MetalKernelType::MatMul)
            .ok_or_else(|| {
                RusTorchError::KernelExecutionError("MatMul pipeline not found".to_string())
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

        // Calculate thread configuration
        let threads_per_threadgroup = MTLSize::new(16, 16, 1);
        let threadgroups_per_grid = MTLSize::new((n + 15) / 16, (m + 15) / 16, 1);

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

    /// Execute reduction operation (sum) using Metal
    /// Metalを使用してリダクション演算（合計）を実行
    pub fn reduce_sum_f32(&self, input: &[f32]) -> RusTorchResult<f32> {
        let size = input.len();
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;

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
            .get(&MetalKernelType::Reduction)
            .ok_or_else(|| {
                RusTorchError::KernelExecutionError("Reduction pipeline not found".to_string())
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
        let threads_per_threadgroup = MTLSize::new(block_size, 1, 1);
        let threadgroups_per_grid = MTLSize::new(grid_size, 1, 1);

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
        let executor = MetalKernelExecutor::new()?;
        executor.matmul_f32(_a, _b, _c, _m, _n, _k)
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
        let executor = MetalKernelExecutor::new()?;
        executor.elementwise_add_f32(_a, _b, _c)
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
        let executor = MetalKernelExecutor::new()?;
        executor.reduce_sum_f32(_input)
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }
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
}
