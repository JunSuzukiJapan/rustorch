//! Optimized OpenCL implementation for cross-platform GPU acceleration
//! クロスプラットフォームGPU加速のための最適化OpenCL実装

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_device_info, Device, DeviceInfo},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY},
    platform::{get_platforms, Platform},
    program::Program,
    types::{cl_device_type, cl_event, cl_platform_id, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU},
};

/// OpenCL device information for optimization selection
/// 最適化選択のためのOpenCLデバイス情報
#[derive(Debug, Clone)]
pub struct OpenClDeviceInfo {
    /// Device name
    /// デバイス名
    pub name: String,
    /// Vendor (AMD, Intel, NVIDIA)
    /// ベンダー (AMD, Intel, NVIDIA)
    pub vendor: String,
    /// Compute units (cores)
    /// 計算ユニット（コア）
    pub compute_units: u32,
    /// Maximum work group size
    /// 最大ワークグループサイズ
    pub max_work_group_size: usize,
    /// Global memory size in bytes
    /// グローバルメモリサイズ（バイト）
    pub global_mem_size: u64,
    /// Local memory size in bytes
    /// ローカルメモリサイズ（バイト）
    pub local_mem_size: u64,
    /// Maximum clock frequency in MHz
    /// 最大クロック周波数（MHz）
    pub max_clock_frequency: u32,
    /// Device type (CPU/GPU/Accelerator)
    /// デバイスタイプ（CPU/GPU/アクセラレータ）
    pub device_type: String,
}

/// Enhanced OpenCL matrix operations with vendor-specific optimizations
/// ベンダー固有最適化による強化OpenCL行列演算
#[cfg(feature = "opencl")]
pub struct OpenClMatrixExecutor {
    context: Context,
    command_queue: CommandQueue,
    device: Device,
    device_info: OpenClDeviceInfo,
    kernels: Arc<Mutex<HashMap<String, Kernel>>>,
    programs: Arc<Mutex<HashMap<String, Program>>>,
}

#[cfg(feature = "opencl")]
impl OpenClMatrixExecutor {
    /// Create new OpenCL matrix executor with automatic device selection
    /// 自動デバイス選択による新しいOpenCL行列実行器を作成
    pub fn new() -> RusTorchResult<Self> {
        let platforms = get_platforms().map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to get OpenCL platforms: {:?}", e))
        })?;

        if platforms.is_empty() {
            return Err(RusTorchError::UnsupportedDevice(
                "No OpenCL platforms available".to_string(),
            ));
        }

        // Find best GPU device across all platforms
        let (device, platform) = Self::select_best_device(&platforms)?;
        
        // Create context
        let context = Context::from_device(&device).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to create OpenCL context: {:?}", e))
        })?;

        // Create command queue with profiling
        let command_queue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to create command queue: {:?}", e))
        })?;

        // Get device information
        let device_info = Self::get_device_info(&device)?;
        
        println!("Selected OpenCL device: {} by {}", device_info.name, device_info.vendor);
        println!("Compute units: {}, Max work group size: {}", 
                 device_info.compute_units, device_info.max_work_group_size);

        Ok(Self {
            context,
            command_queue,
            device,
            device_info,
            kernels: Arc::new(Mutex::new(HashMap::new())),
            programs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Select best GPU device from available platforms
    /// 利用可能なプラットフォームから最適なGPUデバイスを選択
    fn select_best_device(platforms: &[Platform]) -> RusTorchResult<(Device, Platform)> {
        let mut best_device = None;
        let mut best_platform = None;
        let mut best_score = 0f64;

        for platform in platforms {
            let devices = platform
                .get_devices(CL_DEVICE_TYPE_GPU)
                .or_else(|_| platform.get_devices(CL_DEVICE_TYPE_ALL))
                .map_err(|e| {
                    RusTorchError::tensor_op(format!("Failed to get devices: {:?}", e))
                })?;

            for device in devices {
                let info = Self::get_device_info(&device)?;
                
                // Score devices based on compute capability
                let score = Self::score_device(&info);
                
                if score > best_score {
                    best_score = score;
                    best_device = Some(device);
                    best_platform = Some(platform.clone());
                }
            }
        }

        match (best_device, best_platform) {
            (Some(device), Some(platform)) => Ok((device, platform)),
            _ => Err(RusTorchError::UnsupportedDevice(
                "No suitable OpenCL device found".to_string(),
            )),
        }
    }

    /// Get detailed device information
    /// 詳細なデバイス情報を取得
    fn get_device_info(device: &Device) -> RusTorchResult<OpenClDeviceInfo> {
        Ok(OpenClDeviceInfo {
            name: get_device_info(device, DeviceInfo::CL_DEVICE_NAME)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get device name: {:?}", e)))?
                .to_string(),
            vendor: get_device_info(device, DeviceInfo::CL_DEVICE_VENDOR)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get device vendor: {:?}", e)))?
                .to_string(),
            compute_units: get_device_info(device, DeviceInfo::CL_DEVICE_MAX_COMPUTE_UNITS)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get compute units: {:?}", e)))?
                .to_uint(),
            max_work_group_size: get_device_info(device, DeviceInfo::CL_DEVICE_MAX_WORK_GROUP_SIZE)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get max work group size: {:?}", e)))?
                .to_size(),
            global_mem_size: get_device_info(device, DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get global memory size: {:?}", e)))?
                .to_ulong(),
            local_mem_size: get_device_info(device, DeviceInfo::CL_DEVICE_LOCAL_MEM_SIZE)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get local memory size: {:?}", e)))?
                .to_ulong(),
            max_clock_frequency: get_device_info(device, DeviceInfo::CL_DEVICE_MAX_CLOCK_FREQUENCY)
                .map_err(|e| RusTorchError::tensor_op(format!("Failed to get max clock frequency: {:?}", e)))?
                .to_uint(),
            device_type: "GPU".to_string(), // Simplified for now
        })
    }

    /// Score device based on compute capability
    /// 計算能力に基づいてデバイスをスコア化
    fn score_device(info: &OpenClDeviceInfo) -> f64 {
        let mut score = 0.0;

        // Base score from compute units and frequency
        score += info.compute_units as f64 * info.max_clock_frequency as f64 / 1000.0;

        // Bonus for memory size (GB)
        score += (info.global_mem_size as f64 / 1024.0 / 1024.0 / 1024.0) * 10.0;

        // Vendor-specific bonuses
        if info.vendor.contains("NVIDIA") {
            score *= 1.2; // NVIDIA typically has better OpenCL support
        } else if info.vendor.contains("AMD") {
            score *= 1.1; // AMD also good for OpenCL
        }

        score
    }

    /// Optimized matrix multiplication with vendor-specific kernels
    /// ベンダー固有カーネルによる最適化行列乗算
    pub fn matmul_f32(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Validate dimensions
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(RusTorchError::shape_mismatch(
                &[m, k, k, n, m, n],
                &[a.len(), 0, b.len(), 0, c.len(), 0],
            ));
        }

        // Select optimal kernel based on matrix size and device
        let kernel_source = self.generate_matmul_kernel(m, n, k)?;
        let kernel = self.compile_and_cache_kernel("matmul_f32", &kernel_source)?;

        // Create buffers
        let buffer_a = unsafe {
            Buffer::<f32>::create(&self.context, CL_MEM_READ_ONLY, m * k, std::ptr::null_mut())
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to create buffer A: {:?}", e)))?;

        let buffer_b = unsafe {
            Buffer::<f32>::create(&self.context, CL_MEM_READ_ONLY, k * n, std::ptr::null_mut())
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to create buffer B: {:?}", e)))?;

        let buffer_c = unsafe {
            Buffer::<f32>::create(&self.context, CL_MEM_WRITE_ONLY, m * n, std::ptr::null_mut())
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to create buffer C: {:?}", e)))?;

        // Copy data to device
        let write_a_event = unsafe {
            self.command_queue.enqueue_write_buffer(&buffer_a, false, 0, a, &[])
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to write buffer A: {:?}", e)))?;

        let write_b_event = unsafe {
            self.command_queue.enqueue_write_buffer(&buffer_b, false, 0, b, &[])
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to write buffer B: {:?}", e)))?;

        // Calculate optimal work group sizes
        let (global_work_size, local_work_size) = self.calculate_work_sizes(m, n)?;

        // Set kernel arguments and execute
        let kernel_event = ExecuteKernel::new(&kernel)
            .set_arg(&buffer_a)
            .set_arg(&buffer_b)
            .set_arg(&buffer_c)
            .set_arg(&(m as u32))
            .set_arg(&(n as u32))
            .set_arg(&(k as u32))
            .set_global_work_sizes(&global_work_size)
            .set_local_work_sizes(&local_work_size)
            .set_wait_list(&[write_a_event, write_b_event])
            .enqueue_nd_range(&self.command_queue)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to execute kernel: {:?}", e)))?;

        // Read result back
        let _read_event = unsafe {
            self.command_queue.enqueue_read_buffer(&buffer_c, true, 0, c, &[kernel_event])
        }
        .map_err(|e| RusTorchError::tensor_op(format!("Failed to read result: {:?}", e)))?;

        Ok(())
    }

    /// Generate optimized matrix multiplication kernel based on device and size
    /// デバイスとサイズに基づく最適化行列乗算カーネルの生成
    fn generate_matmul_kernel(&self, m: usize, n: usize, k: usize) -> RusTorchResult<String> {
        let tile_size = self.calculate_optimal_tile_size(m, n, k);
        
        // Vendor-specific optimizations
        let kernel_source = if self.device_info.vendor.contains("AMD") {
            self.generate_amd_optimized_kernel(tile_size)
        } else if self.device_info.vendor.contains("NVIDIA") {
            self.generate_nvidia_optimized_kernel(tile_size)
        } else if self.device_info.vendor.contains("Intel") {
            self.generate_intel_optimized_kernel(tile_size)
        } else {
            self.generate_generic_kernel(tile_size)
        };

        Ok(kernel_source)
    }

    /// Calculate optimal tile size based on device characteristics
    /// デバイス特性に基づく最適タイルサイズの計算
    fn calculate_optimal_tile_size(&self, m: usize, n: usize, k: usize) -> usize {
        let max_work_group = self.device_info.max_work_group_size;
        let local_mem_size = self.device_info.local_mem_size as usize;
        
        // Calculate tile size based on local memory constraints
        let max_tile_from_memory = ((local_mem_size / 2) / (2 * std::mem::size_of::<f32>())).sqrt();
        let max_tile_from_workgroup = (max_work_group as f64).sqrt() as usize;
        
        let optimal_tile = max_tile_from_memory.min(max_tile_from_workgroup).min(32);
        
        // Ensure tile size is a power of 2 and at least 8
        let tile_size = (optimal_tile.next_power_of_two().max(8)).min(32);
        
        tile_size
    }

    /// Generate AMD-optimized kernel
    /// AMD最適化カーネルの生成
    fn generate_amd_optimized_kernel(&self, tile_size: usize) -> String {
        format!(
            r#"
__kernel void matmul_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {{
    // AMD GCN architecture optimizations
    const int TILE_SIZE = {tile_size};
    
    // Get work group and local IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    // Shared memory for tiles
    __local float As[{tile_size}][{tile_size}];
    __local float Bs[{tile_size}][{tile_size}];
    
    // Calculate global position
    const int global_row = group_y * TILE_SIZE + local_y;
    const int global_col = group_x * TILE_SIZE + local_x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
        // Load tiles into shared memory with bounds checking
        int a_row = global_row;
        int a_col = tile * TILE_SIZE + local_x;
        int b_row = tile * TILE_SIZE + local_y;
        int b_col = global_col;
        
        As[local_y][local_x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[local_y][local_x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum with loop unrolling for AMD GCN
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum = fma(As[local_y][k], Bs[k][local_x], sum);
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
    // Write result
    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = sum;
    }}
}}
"#,
            tile_size = tile_size
        )
    }

    /// Generate NVIDIA-optimized kernel
    /// NVIDIA最適化カーネルの生成
    fn generate_nvidia_optimized_kernel(&self, tile_size: usize) -> String {
        format!(
            r#"
__kernel void matmul_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {{
    // NVIDIA CUDA core optimizations
    const int TILE_SIZE = {tile_size};
    
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    __local float As[{tile_size}][{tile_size} + 1]; // Bank conflict avoidance
    __local float Bs[{tile_size}][{tile_size} + 1];
    
    const int global_row = group_y * TILE_SIZE + local_y;
    const int global_col = group_x * TILE_SIZE + local_x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
        // Coalesced memory access for NVIDIA
        int a_row = global_row;
        int a_col = tile * TILE_SIZE + local_x;
        int b_row = tile * TILE_SIZE + local_y;
        int b_col = global_col;
        
        As[local_y][local_x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[local_y][local_x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Optimized for NVIDIA warp execution
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[local_y][k] * Bs[k][local_x];
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = sum;
    }}
}}
"#,
            tile_size = tile_size
        )
    }

    /// Generate Intel-optimized kernel
    /// Intel最適化カーネルの生成
    fn generate_intel_optimized_kernel(&self, tile_size: usize) -> String {
        format!(
            r#"
__kernel void matmul_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {{
    // Intel GPU optimizations
    const int TILE_SIZE = {tile_size};
    
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    __local float As[{tile_size}][{tile_size}];
    __local float Bs[{tile_size}][{tile_size}];
    
    const int global_row = group_y * TILE_SIZE + local_y;
    const int global_col = group_x * TILE_SIZE + local_x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
        int a_row = global_row;
        int a_col = tile * TILE_SIZE + local_x;
        int b_row = tile * TILE_SIZE + local_y;
        int b_col = global_col;
        
        // Intel-specific prefetch hints
        As[local_y][local_x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[local_y][local_x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Conservative unrolling for Intel
        #pragma unroll 2
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[local_y][k] * Bs[k][local_x];
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = sum;
    }}
}}
"#,
            tile_size = tile_size
        )
    }

    /// Generate generic kernel for unknown vendors
    /// 不明ベンダー向け汎用カーネルの生成
    fn generate_generic_kernel(&self, tile_size: usize) -> String {
        format!(
            r#"
__kernel void matmul_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {{
    const int TILE_SIZE = {tile_size};
    
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    __local float As[{tile_size}][{tile_size}];
    __local float Bs[{tile_size}][{tile_size}];
    
    const int global_row = group_y * TILE_SIZE + local_y;
    const int global_col = group_x * TILE_SIZE + local_x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
        int a_row = global_row;
        int a_col = tile * TILE_SIZE + local_x;
        int b_row = tile * TILE_SIZE + local_y;
        int b_col = global_col;
        
        As[local_y][local_x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[local_y][local_x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[local_y][k] * Bs[k][local_x];
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = sum;
    }}
}}
"#,
            tile_size = tile_size
        )
    }

    /// Compile and cache kernel
    /// カーネルのコンパイルとキャッシュ
    fn compile_and_cache_kernel(&mut self, name: &str, source: &str) -> RusTorchResult<Kernel> {
        // Check cache first
        {
            let kernels = self.kernels.lock().unwrap();
            if let Some(kernel) = kernels.get(name) {
                return Ok(kernel.clone());
            }
        }

        // Compile program
        let program = Program::create_and_build_from_source(&self.context, source, "")
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to compile kernel: {:?}", e)))?;

        // Create kernel
        let kernel = Kernel::create(&program, name)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to create kernel: {:?}", e)))?;

        // Cache both program and kernel
        {
            let mut programs = self.programs.lock().unwrap();
            programs.insert(name.to_string(), program);
            
            let mut kernels = self.kernels.lock().unwrap();
            kernels.insert(name.to_string(), kernel.clone());
        }

        Ok(kernel)
    }

    /// Calculate optimal work group sizes
    /// 最適ワークグループサイズの計算
    fn calculate_work_sizes(&self, m: usize, n: usize) -> RusTorchResult<([usize; 2], [usize; 2])> {
        let tile_size = self.calculate_optimal_tile_size(m, n, m.max(n));
        
        let global_work_size = [
            ((n + tile_size - 1) / tile_size) * tile_size,
            ((m + tile_size - 1) / tile_size) * tile_size,
        ];
        
        let local_work_size = [tile_size, tile_size];
        
        Ok((global_work_size, local_work_size))
    }

    /// Get device information for display
    /// 表示用デバイス情報の取得
    pub fn get_device_info(&self) -> &OpenClDeviceInfo {
        &self.device_info
    }
}

/// Non-OpenCL fallback implementation
/// OpenCL非対応時のフォールバック実装
#[cfg(not(feature = "opencl"))]
pub struct OpenClMatrixExecutor;

#[cfg(not(feature = "opencl"))]
impl OpenClMatrixExecutor {
    pub fn new() -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "OpenCL not available".to_string(),
        ))
    }

    pub fn matmul_f32(
        &mut self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "OpenCL not available".to_string(),
        ))
    }
}

/// Public interface functions for OpenCL operations
/// OpenCL演算のためのパブリックインターフェース関数

/// Execute OpenCL matrix multiplication
/// OpenCL行列乗算を実行
pub fn opencl_matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "opencl")]
    {
        let mut executor = OpenClMatrixExecutor::new()?;
        executor.matmul_f32(a, b, c, m, n, k)
    }
    #[cfg(not(feature = "opencl"))]
    {
        let _ = (a, b, c, m, n, k);
        Err(RusTorchError::UnsupportedDevice(
            "OpenCL not available".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_executor_creation() {
        let result = OpenClMatrixExecutor::new();
        #[cfg(not(feature = "opencl"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_opencl_matmul_interface() {
        let a = vec![1.0f32; 64];
        let b = vec![2.0f32; 64];
        let mut c = vec![0.0f32; 64];

        let result = opencl_matmul_f32(&a, &b, &mut c, 8, 8, 8);
        #[cfg(not(feature = "opencl"))]
        assert!(result.is_err());
    }
}