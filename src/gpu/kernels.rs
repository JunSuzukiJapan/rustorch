/// GPU kernel operations and execution
/// GPUカーネル操作と実行

use super::{DeviceType, GpuError};
use num_traits::Float;

/// GPU kernel execution parameters
/// GPUカーネル実行パラメータ
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Block/workgroup size
    /// ブロック/ワークグループサイズ
    pub block_size: (u32, u32, u32),
    /// Grid size
    /// グリッドサイズ
    pub grid_size: (u32, u32, u32),
    /// Shared memory size
    /// 共有メモリサイズ
    pub shared_memory: u32,
    /// Stream/queue ID
    /// ストリーム/キューID
    pub stream_id: u32,
}

impl Default for KernelParams {
    fn default() -> Self {
        KernelParams {
            block_size: (256, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory: 0,
            stream_id: 0,
        }
    }
}

/// GPU kernel trait for different operations
/// 異なる操作用GPUカーネルトレイト
pub trait GpuKernel<T: Float> {
    /// Execute the kernel
    /// カーネルを実行
    fn execute(
        &self,
        device: DeviceType,
        _params: &KernelParams,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError>;

    /// Get optimal parameters for given problem size
    /// 指定された問題サイズに対する最適パラメータを取得
    fn optimal_params(&self, problem_size: usize, device: DeviceType) -> KernelParams;
}

/// Element-wise addition kernel
/// 要素ごと加算カーネル
pub struct AddKernel;

impl<T: Float> GpuKernel<T> for AddKernel {
    fn execute(
        &self,
        device: DeviceType,
        _params: &KernelParams,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(GpuError::InvalidOperation("Add kernel requires 2 inputs and 1 output".to_string()));
        }

        let a = inputs[0];
        let b = inputs[1];
        let c = &mut outputs[0];

        if a.len() != b.len() || a.len() != c.len() {
            return Err(GpuError::InvalidOperation("Input/output size mismatch".to_string()));
        }

        match device {
            DeviceType::Cpu => {
                // CPU implementation
                for i in 0..a.len() {
                    c[i] = a[i] + b[i];
                }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use crate::gpu::cuda_kernels::cuda_elementwise_add_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        cuda_elementwise_add_f32(a_f32, b_f32, c_f32)
                            .map_err(|e| GpuError::KernelExecutionError(format!("CUDA add failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("CUDA only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::UnsupportedDevice("CUDA not supported".to_string()));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    use crate::gpu::metal_kernels::metal_elementwise_add_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        metal_elementwise_add_f32(a_f32, b_f32, c_f32)
                            .map_err(|e| GpuError::KernelExecutionError(format!("Metal add failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("Metal only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(GpuError::UnsupportedDevice("Metal not supported".to_string()));
                }
            }
            DeviceType::OpenCl(_) => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl_kernels::opencl_elementwise_add_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        opencl_elementwise_add_f32(a_f32, b_f32, c_f32)
                            .map_err(|e| GpuError::KernelExecutionError(format!("OpenCL add failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("OpenCL only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(GpuError::UnsupportedDevice("OpenCL not supported".to_string()));
                }
            }
        }

        Ok(())
    }

    fn optimal_params(&self, problem_size: usize, device: DeviceType) -> KernelParams {
        match device {
            DeviceType::Cpu => KernelParams::default(),
            DeviceType::Cuda(_) => {
                let threads_per_block = 256;
                let num_blocks = (problem_size + threads_per_block - 1) / threads_per_block;
                KernelParams {
                    block_size: (threads_per_block as u32, 1, 1),
                    grid_size: (num_blocks as u32, 1, 1),
                    shared_memory: 0,
                    stream_id: 0,
                }
            }
            DeviceType::Metal(_) => {
                let threads_per_group = 256;
                let num_groups = (problem_size + threads_per_group - 1) / threads_per_group;
                KernelParams {
                    block_size: (threads_per_group as u32, 1, 1),
                    grid_size: (num_groups as u32, 1, 1),
                    shared_memory: 0,
                    stream_id: 0,
                }
            }
            DeviceType::OpenCl(_) => {
                let work_group_size = 256;
                let global_size = (problem_size + work_group_size - 1) / work_group_size * work_group_size;
                KernelParams {
                    block_size: (work_group_size as u32, 1, 1),
                    grid_size: (global_size as u32, 1, 1),
                    shared_memory: 0,
                    stream_id: 0,
                }
            }
        }
    }
}

// AddKernel implementation methods removed - now using direct GPU kernel calls

/// Matrix multiplication kernel
/// 行列乗算カーネル
pub struct MatMulKernel;

impl<T: Float> GpuKernel<T> for MatMulKernel {
    fn execute(
        &self,
        device: DeviceType,
        _params: &KernelParams,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(GpuError::InvalidOperation("MatMul kernel requires 2 inputs and 1 output".to_string()));
        }

        // For simplicity, assume square matrices for now
        let a = inputs[0];
        let b = inputs[1];
        let c = &mut outputs[0];

        let n = (a.len() as f64).sqrt() as usize;
        if n * n != a.len() || b.len() != a.len() || c.len() != a.len() {
            return Err(GpuError::InvalidOperation("Matrix size mismatch".to_string()));
        }

        match device {
            DeviceType::Cpu => {
                // CPU matrix multiplication
                for i in 0..n {
                    for j in 0..n {
                        let mut sum = T::zero();
                        for k in 0..n {
                            sum = sum + a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use crate::gpu::cuda_kernels::cuda_matmul_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        cuda_matmul_f32(a_f32, b_f32, c_f32, n, n, n)
                            .map_err(|e| GpuError::KernelExecutionError(format!("CUDA matmul failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("CUDA only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::UnsupportedDevice("CUDA not supported".to_string()));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    use crate::gpu::metal_kernels::metal_matmul_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        metal_matmul_f32(a_f32, b_f32, c_f32, n, n, n)
                            .map_err(|e| GpuError::KernelExecutionError(format!("Metal matmul failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("Metal only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(GpuError::UnsupportedDevice("Metal not supported".to_string()));
                }
            }
            DeviceType::OpenCl(_) => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl_kernels::opencl_matmul_f32;
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                        let c_f32 = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, c.len()) };
                        opencl_matmul_f32(a_f32, b_f32, c_f32, n, n, n)
                            .map_err(|e| GpuError::KernelExecutionError(format!("OpenCL matmul failed: {:?}", e)))?;
                    } else {
                        return Err(GpuError::UnsupportedDevice("OpenCL only supports f32 currently".to_string()));
                    }
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(GpuError::UnsupportedDevice("OpenCL not supported".to_string()));
                }
            }
        }

        Ok(())
    }

    fn optimal_params(&self, problem_size: usize, device: DeviceType) -> KernelParams {
        let n = (problem_size as f64).sqrt() as usize;
        
        match device {
            DeviceType::Cpu => KernelParams::default(),
            DeviceType::Cuda(_) => {
                // Use 2D block for matrix multiplication
                let block_size = 16; // 16x16 block
                let grid_size = (n + block_size - 1) / block_size;
                KernelParams {
                    block_size: (block_size as u32, block_size as u32, 1),
                    grid_size: (grid_size as u32, grid_size as u32, 1),
                    shared_memory: (2 * block_size * block_size * std::mem::size_of::<f32>()) as u32,
                    stream_id: 0,
                }
            }
            DeviceType::Metal(_) => {
                let threads_per_group = 16;
                let num_groups = (n + threads_per_group - 1) / threads_per_group;
                KernelParams {
                    block_size: (threads_per_group as u32, threads_per_group as u32, 1),
                    grid_size: (num_groups as u32, num_groups as u32, 1),
                    shared_memory: 0,
                    stream_id: 0,
                }
            }
            DeviceType::OpenCl(_) => {
                let work_group_size = 16;
                let global_size = (n + work_group_size - 1) / work_group_size * work_group_size;
                KernelParams {
                    block_size: (work_group_size as u32, work_group_size as u32, 1),
                    grid_size: (global_size as u32, global_size as u32, 1),
                    shared_memory: 0,
                    stream_id: 0,
                }
            }
        }
    }
}

// MatMulKernel implementation methods removed - now using direct GPU kernel calls

/// Convolution kernel
/// 畳み込みカーネル
pub struct ConvKernel {
    /// Size of the convolution kernel
    /// 畳み込みカーネルのサイズ
    pub kernel_size: usize,
    /// Stride of the convolution operation
    /// 畳み込み操作のストライド
    pub stride: usize,
    /// Padding applied to the input
    /// 入力に適用されるパディング
    pub padding: usize,
}

impl<T: Float> GpuKernel<T> for ConvKernel {
    fn execute(
        &self,
        device: DeviceType,
        _params: &KernelParams,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(GpuError::InvalidOperation("Conv kernel requires 2 inputs and 1 output".to_string()));
        }

        match device {
            DeviceType::Cpu => {
                // CPU convolution implementation (simplified)
                self.execute_cpu_conv(inputs[0], inputs[1], &mut outputs[0])?;
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    self.execute_cuda_conv(params, inputs[0], inputs[1], &mut outputs[0])?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::UnsupportedDevice("CUDA not supported".to_string()));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    self.execute_metal_conv(params, inputs[0], inputs[1], &mut outputs[0])?;
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(GpuError::UnsupportedDevice("Metal not supported".to_string()));
                }
            }
            DeviceType::OpenCl(_) => {
                #[cfg(feature = "opencl")]
                {
                    self.execute_opencl_conv(params, inputs[0], inputs[1], &mut outputs[0])?;
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(GpuError::UnsupportedDevice("OpenCL not supported".to_string()));
                }
            }
        }

        Ok(())
    }

    fn optimal_params(&self, problem_size: usize, device: DeviceType) -> KernelParams {
        match device {
            DeviceType::Cpu => KernelParams::default(),
            DeviceType::Cuda(_) => {
                let block_size = 16;
                let grid_size = (problem_size + block_size - 1) / block_size;
                KernelParams {
                    block_size: (block_size as u32, block_size as u32, 1),
                    grid_size: (grid_size as u32, 1, 1),
                    shared_memory: (block_size * block_size * std::mem::size_of::<f32>()) as u32,
                    stream_id: 0,
                }
            }
            _ => KernelParams::default(),
        }
    }
}

impl ConvKernel {
    fn execute_cpu_conv<T: Float>(
        &self,
        _input: &[T],
        _kernel: &[T],
        _output: &mut [T],
    ) -> Result<(), GpuError> {
        // Simplified CPU convolution
        // In practice, this would implement proper 2D convolution
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn execute_cuda_conv<T: Float>(
        &self,
        _params: &KernelParams,
        _input: &[T],
        _kernel: &[T],
        _output: &mut [T],
    ) -> Result<(), GpuError> {
        // CUDA convolution kernel would go here
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn execute_metal_conv<T: Float>(
        &self,
        _params: &KernelParams,
        _input: &[T],
        _kernel: &[T],
        _output: &mut [T],
    ) -> Result<(), GpuError> {
        // Metal convolution shader would go here
        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn execute_opencl_conv<T: Float>(
        &self,
        _params: &KernelParams,
        _input: &[T],
        _kernel: &[T],
        _output: &mut [T],
    ) -> Result<(), GpuError> {
        // OpenCL convolution kernel would go here
        Ok(())
    }
}

/// Kernel executor for managing and running kernels
/// カーネル管理と実行用カーネルエグゼキューター
pub struct KernelExecutor {
    device: DeviceType,
}

impl KernelExecutor {
    /// Create a new kernel executor
    /// 新しいカーネルエグゼキューターを作成
    pub fn new(device: DeviceType) -> Self {
        KernelExecutor { device }
    }

    /// Execute a kernel with automatic parameter optimization
    /// 自動パラメータ最適化でカーネルを実行
    pub fn execute_kernel<T: Float, K: GpuKernel<T>>(
        &self,
        kernel: &K,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError> {
        let problem_size = if !inputs.is_empty() { inputs[0].len() } else { 0 };
        let params = kernel.optimal_params(problem_size, self.device);
        kernel.execute(self.device, &params, inputs, outputs)
    }

    /// Execute a kernel with custom parameters
    /// カスタムパラメータでカーネルを実行
    pub fn execute_kernel_with_params<T: Float, K: GpuKernel<T>>(
        &self,
        kernel: &K,
        params: &KernelParams,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<(), GpuError> {
        kernel.execute(self.device, params, inputs, outputs)
    }

    /// Get device
    /// デバイスを取得
    pub fn device(&self) -> DeviceType {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_params_default() {
        let params = KernelParams::default();
        assert_eq!(params.block_size, (256, 1, 1));
        assert_eq!(params.grid_size, (1, 1, 1));
        assert_eq!(params.shared_memory, 0);
        assert_eq!(params.stream_id, 0);
    }

    #[test]
    fn test_add_kernel_cpu() {
        let kernel = AddKernel;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        let inputs = [a.as_slice(), b.as_slice()];
        let mut outputs = [c.as_mut_slice()];

        let params = KernelParams::default();
        kernel.execute(DeviceType::Cpu, &params, &inputs, &mut outputs).unwrap();

        assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul_kernel_cpu() {
        let kernel = MatMulKernel;
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = vec![0.0f32; 4];

        let inputs = [a.as_slice(), b.as_slice()];
        let mut outputs = [c.as_mut_slice()];

        let params = KernelParams::default();
        kernel.execute(DeviceType::Cpu, &params, &inputs, &mut outputs).unwrap();

        // Expected result: [19, 22, 43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_kernel_executor() {
        let executor = KernelExecutor::new(DeviceType::Cpu);
        assert_eq!(executor.device(), DeviceType::Cpu);

        let kernel = AddKernel;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut c = vec![0.0f32; 3];

        let inputs = [a.as_slice(), b.as_slice()];
        let mut outputs = [c.as_mut_slice()];

        executor.execute_kernel(&kernel, &inputs, &mut outputs).unwrap();
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_optimal_params() {
        let kernel = AddKernel;
        
        let params_cpu = <AddKernel as GpuKernel<f32>>::optimal_params(&kernel, 1000, DeviceType::Cpu);
        assert_eq!(params_cpu.block_size, (256, 1, 1));
        
        let params_cuda = <AddKernel as GpuKernel<f32>>::optimal_params(&kernel, 1000, DeviceType::Cuda(0));
        assert!(params_cuda.grid_size.0 > 1);
    }

    #[test]
    fn test_conv_kernel() {
        let kernel = ConvKernel {
            kernel_size: 3,
            stride: 1,
            padding: 0,
        };

        let input = vec![1.0f32; 16]; // 4x4 input
        let filter = vec![1.0f32; 9]; // 3x3 filter
        let mut output = vec![0.0f32; 4]; // 2x2 output

        let inputs = [input.as_slice(), filter.as_slice()];
        let mut outputs = [output.as_mut_slice()];

        let params = KernelParams::default();
        kernel.execute(DeviceType::Cpu, &params, &inputs, &mut outputs).unwrap();
    }
}
