//! Unified kernel interface for cross-platform GPU acceleration
//! クロスプラットフォームGPU加速のための統一カーネルインターフェース
//!
//! This module provides a unified interface for GPU kernels across CUDA, Metal, and OpenCL,
//! enabling seamless execution on different GPU backends with automatic optimization.
//! このモジュールは、CUDA、Metal、OpenCL間でGPUカーネルの統一インターフェースを提供し、
//! 異なるGPUバックエンドでのシームレスな実行と自動最適化を可能にします。

use super::{DeviceType, GpuError, GpuResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Unified kernel operation types
/// 統一カーネル操作タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    /// Element-wise addition
    /// 要素ごとの加算
    Add,
    /// Element-wise multiplication
    /// 要素ごとの乗算
    Mul,
    /// Element-wise subtraction
    /// 要素ごとの減算
    Sub,
    /// Element-wise division
    /// 要素ごとの除算
    Div,
    /// Matrix multiplication
    /// 行列乗算
    MatMul,
    /// Convolution 2D
    /// 2D畳み込み
    Conv2D,
    /// Batch normalization
    /// バッチ正規化
    BatchNorm,
    /// Reduction sum
    /// リダクション合計
    ReduceSum,
    /// Reduction mean
    /// リダクション平均
    ReduceMean,
    /// Activation ReLU
    /// ReLU活性化
    ReLU,
    /// Activation Softmax
    /// Softmax活性化
    Softmax,
}

/// Kernel execution parameters
/// カーネル実行パラメータ
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Input tensor shapes for optimization
    /// 最適化のための入力テンソル形状
    pub input_shapes: Vec<Vec<usize>>,
    /// Expected output shape
    /// 期待される出力形状
    pub output_shape: Vec<usize>,
    /// Additional operation parameters
    /// 追加の操作パラメータ
    pub extra_params: HashMap<String, f64>,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            input_shapes: Vec::new(),
            output_shape: Vec::new(),
            extra_params: HashMap::new(),
        }
    }
}

/// Kernel performance metrics
/// カーネル性能メトリクス
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    /// Execution time
    /// 実行時間
    pub execution_time: Duration,
    /// Memory bandwidth utilization
    /// メモリ帯域利用率
    pub memory_bandwidth: f64,
    /// GPU occupancy percentage
    /// GPU占有率パーセンテージ
    pub occupancy: f64,
    /// FLOPS (floating point operations per second)
    /// FLOPS（1秒あたりの浮動小数点演算数）
    pub flops: f64,
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::ZERO,
            memory_bandwidth: 0.0,
            occupancy: 0.0,
            flops: 0.0,
        }
    }
}

/// Unified kernel executor trait
/// 統一カーネル実行者トレイト
pub trait UnifiedKernelExecutor: Send + Sync {
    /// Execute a kernel operation for f32 tensors
    /// f32テンソル用カーネル操作を実行
    fn execute_f32(&self, op: KernelOp, inputs: &[&Tensor<f32>], params: &KernelParams) -> GpuResult<Tensor<f32>>;

    /// Execute a kernel operation for f64 tensors
    /// f64テンソル用カーネル操作を実行
    fn execute_f64(&self, op: KernelOp, inputs: &[&Tensor<f64>], params: &KernelParams) -> GpuResult<Tensor<f64>>;

    /// Check if operation is supported
    /// 操作がサポートされているかチェック
    fn supports_operation(&self, op: KernelOp) -> bool;

    /// Get device type
    /// デバイスタイプを取得
    fn device_type(&self) -> DeviceType;

    /// Get performance metrics for last execution
    /// 最後の実行の性能メトリクスを取得
    fn get_metrics(&self) -> KernelMetrics;

    /// Optimize kernel parameters for given operation
    /// 指定された操作に対してカーネルパラメータを最適化
    fn optimize_params(&self, op: KernelOp, params: &mut KernelParams) -> GpuResult<()>;
}

/// Generic execute helper trait for internal use
/// 内部使用向けジェネリック実行ヘルパートレイト
trait ExecuteGeneric<T: Float + 'static + Send + Sync> {
    fn execute(&self, op: KernelOp, inputs: &[&Tensor<T>], params: &KernelParams) -> GpuResult<Tensor<T>>;
}

/// CUDA kernel executor
/// CUDAカーネル実行者
#[cfg(feature = "cuda")]
pub struct CudaUnifiedExecutor {
    device_id: usize,
    metrics: Arc<Mutex<KernelMetrics>>,
}

#[cfg(feature = "cuda")]
impl CudaUnifiedExecutor {
    pub fn new(device_id: usize) -> GpuResult<Self> {
        // Validate CUDA device
        use crate::gpu::cuda_kernels::CudaKernelExecutor;
        let _executor = CudaKernelExecutor::new(device_id)
            .map_err(|_| GpuError::DeviceNotFound(device_id))?;

        Ok(Self {
            device_id,
            metrics: Arc::new(Mutex::new(KernelMetrics::default())),
        })
    }
}

#[cfg(feature = "cuda")]
impl ExecuteGeneric<f32> for CudaUnifiedExecutor {
    fn execute(&self, op: KernelOp, inputs: &[&Tensor<f32>], params: &KernelParams) -> GpuResult<Tensor<f32>> {
        let start_time = Instant::now();
        
        // Execute CUDA-specific operation
        let result = match op {
            KernelOp::Add => self.execute_cuda_add(inputs, params),
            KernelOp::Mul => self.execute_cuda_mul(inputs, params),
            KernelOp::MatMul => self.execute_cuda_matmul(inputs, params),
            KernelOp::Conv2D => self.execute_cuda_conv2d(inputs, params),
            _ => Err(GpuError::UnsupportedOperation(format!("Operation {:?} not implemented for CUDA", op))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;
            // TODO: Calculate actual metrics from CUDA profiling
            metrics.memory_bandwidth = 100.0; // Placeholder
            metrics.occupancy = 80.0; // Placeholder
            metrics.flops = 1000.0; // Placeholder
        }

        result
    }
}

#[cfg(feature = "cuda")]
impl ExecuteGeneric<f64> for CudaUnifiedExecutor {
    fn execute(&self, op: KernelOp, inputs: &[&Tensor<f64>], params: &KernelParams) -> GpuResult<Tensor<f64>> {
        let start_time = Instant::now();
        
        // Execute CUDA-specific operation
        let result = match op {
            KernelOp::Add => self.execute_cuda_add(inputs, params),
            KernelOp::Mul => self.execute_cuda_mul(inputs, params),
            KernelOp::MatMul => self.execute_cuda_matmul(inputs, params),
            KernelOp::Conv2D => self.execute_cuda_conv2d(inputs, params),
            _ => Err(GpuError::UnsupportedOperation(format!("Operation {:?} not implemented for CUDA", op))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;
            metrics.memory_bandwidth = 100.0;
            metrics.occupancy = 80.0;
            metrics.flops = 1000.0;
        }

        result
    }
}

#[cfg(feature = "cuda")]
impl UnifiedKernelExecutor for CudaUnifiedExecutor {
    fn execute_f32(&self, op: KernelOp, inputs: &[&Tensor<f32>], params: &KernelParams) -> GpuResult<Tensor<f32>> {
        <Self as ExecuteGeneric<f32>>::execute(self, op, inputs, params)
    }

    fn execute_f64(&self, op: KernelOp, inputs: &[&Tensor<f64>], params: &KernelParams) -> GpuResult<Tensor<f64>> {
        <Self as ExecuteGeneric<f64>>::execute(self, op, inputs, params)
    }

    fn supports_operation(&self, op: KernelOp) -> bool {
        matches!(op, KernelOp::Add | KernelOp::Mul | KernelOp::MatMul | KernelOp::Conv2D | KernelOp::ReLU)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.device_id)
    }

    fn get_metrics(&self) -> KernelMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn optimize_params(&self, op: KernelOp, params: &mut KernelParams) -> GpuResult<()> {
        // CUDA-specific parameter optimization
        match op {
            KernelOp::MatMul => {
                // Optimize for CUDA matrix multiplication
                params.extra_params.insert("cuda_tile_size".to_string(), 32.0);
                params.extra_params.insert("cuda_use_cublas".to_string(), 1.0);
            },
            KernelOp::Conv2D => {
                // Optimize for CUDA convolution
                params.extra_params.insert("cuda_use_cudnn".to_string(), 1.0);
                params.extra_params.insert("cuda_workspace_size".to_string(), 64.0 * 1024.0 * 1024.0);
            },
            _ => {}
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaUnifiedExecutor {
    fn execute_cuda_add<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Add operation requires exactly 2 inputs".to_string()));
        }

        // Placeholder implementation - would use actual CUDA kernels
        inputs[0].add(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_cuda_mul<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Mul operation requires exactly 2 inputs".to_string()));
        }

        // Placeholder implementation
        inputs[0].mul(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_cuda_matmul<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("MatMul operation requires exactly 2 inputs".to_string()));
        }

        // Placeholder implementation
        inputs[0].matmul(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_cuda_conv2d<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        // Placeholder implementation for convolution
        Err(GpuError::UnsupportedOperation("Conv2D not yet implemented".to_string()))
    }
}

/// Metal kernel executor
/// Metalカーネル実行者
#[cfg(feature = "metal")]
pub struct MetalUnifiedExecutor {
    device_id: usize,
    metrics: Arc<Mutex<KernelMetrics>>,
}

#[cfg(feature = "metal")]
impl MetalUnifiedExecutor {
    pub fn new(device_id: usize) -> GpuResult<Self> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;
        let _executor = MetalKernelExecutor::new()
            .map_err(|_| GpuError::DeviceNotFound(device_id))?;

        Ok(Self {
            device_id,
            metrics: Arc::new(Mutex::new(KernelMetrics::default())),
        })
    }
}

#[cfg(feature = "metal")]
impl UnifiedKernelExecutor for MetalUnifiedExecutor {
    fn execute<T>(&self, op: KernelOp, inputs: &[&Tensor<T>], params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Execute Metal-specific operation
        let result = match op {
            KernelOp::Add => self.execute_metal_add(inputs, params),
            KernelOp::Mul => self.execute_metal_mul(inputs, params),
            KernelOp::MatMul => self.execute_metal_matmul(inputs, params),
            _ => Err(GpuError::UnsupportedOperation(format!("Operation {:?} not implemented for Metal", op))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;
            metrics.memory_bandwidth = 90.0; // Placeholder for Metal
            metrics.occupancy = 75.0; // Placeholder
            metrics.flops = 800.0; // Placeholder
        }

        result
    }

    fn supports_operation(&self, op: KernelOp) -> bool {
        matches!(op, KernelOp::Add | KernelOp::Mul | KernelOp::MatMul | KernelOp::ReLU)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Metal(self.device_id)
    }

    fn get_metrics(&self) -> KernelMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn optimize_params(&self, op: KernelOp, params: &mut KernelParams) -> GpuResult<()> {
        // Metal-specific parameter optimization
        match op {
            KernelOp::MatMul => {
                // Optimize for Metal matrix multiplication
                params.extra_params.insert("metal_threads_per_group".to_string(), 64.0);
                params.extra_params.insert("metal_use_mps".to_string(), 1.0);
            },
            _ => {}
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
impl MetalUnifiedExecutor {
    fn execute_metal_add<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Add operation requires exactly 2 inputs".to_string()));
        }

        // Placeholder implementation - would use actual Metal shaders
        inputs[0].add(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_metal_mul<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Mul operation requires exactly 2 inputs".to_string()));
        }

        inputs[0].mul(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_metal_matmul<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("MatMul operation requires exactly 2 inputs".to_string()));
        }

        inputs[0].matmul(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }
}

/// OpenCL kernel executor
/// OpenCLカーネル実行者
#[cfg(feature = "opencl")]
pub struct OpenClUnifiedExecutor {
    device_id: usize,
    metrics: Arc<Mutex<KernelMetrics>>,
}

#[cfg(feature = "opencl")]
impl OpenClUnifiedExecutor {
    pub fn new(device_id: usize) -> GpuResult<Self> {
        use crate::gpu::opencl_kernels::OpenClKernelExecutor;
        let _executor = OpenClKernelExecutor::new(device_id)
            .map_err(|_| GpuError::DeviceNotFound(device_id))?;

        Ok(Self {
            device_id,
            metrics: Arc::new(Mutex::new(KernelMetrics::default())),
        })
    }
}

#[cfg(feature = "opencl")]
impl UnifiedKernelExecutor for OpenClUnifiedExecutor {
    fn execute<T>(&self, op: KernelOp, inputs: &[&Tensor<T>], params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Execute OpenCL-specific operation
        let result = match op {
            KernelOp::Add => self.execute_opencl_add(inputs, params),
            KernelOp::Mul => self.execute_opencl_mul(inputs, params),
            _ => Err(GpuError::UnsupportedOperation(format!("Operation {:?} not implemented for OpenCL", op))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;
            metrics.memory_bandwidth = 70.0; // Placeholder for OpenCL
            metrics.occupancy = 60.0; // Placeholder
            metrics.flops = 600.0; // Placeholder
        }

        result
    }

    fn supports_operation(&self, op: KernelOp) -> bool {
        matches!(op, KernelOp::Add | KernelOp::Mul)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::OpenCL(self.device_id)
    }

    fn get_metrics(&self) -> KernelMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn optimize_params(&self, op: KernelOp, params: &mut KernelParams) -> GpuResult<()> {
        // OpenCL-specific parameter optimization
        match op {
            KernelOp::Add | KernelOp::Mul => {
                params.extra_params.insert("opencl_local_size".to_string(), 256.0);
            },
            _ => {}
        }
        Ok(())
    }
}

#[cfg(feature = "opencl")]
impl OpenClUnifiedExecutor {
    fn execute_opencl_add<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Add operation requires exactly 2 inputs".to_string()));
        }

        inputs[0].add(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }

    fn execute_opencl_mul<T>(&self, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        if inputs.len() != 2 {
            return Err(GpuError::InvalidOperation("Mul operation requires exactly 2 inputs".to_string()));
        }

        inputs[0].mul(inputs[1])
            .map_err(|e| GpuError::KernelExecutionError(e))
    }
}

/// CPU fallback executor
/// CPUフォールバック実行者
pub struct CpuFallbackExecutor {
    metrics: Arc<Mutex<KernelMetrics>>,
}

impl CpuFallbackExecutor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(KernelMetrics::default())),
        }
    }
}

impl Default for CpuFallbackExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedKernelExecutor for CpuFallbackExecutor {
    fn execute<T>(&self, op: KernelOp, inputs: &[&Tensor<T>], _params: &KernelParams) -> GpuResult<Tensor<T>>
    where
        T: Float + 'static + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Execute CPU implementation
        let result = match op {
            KernelOp::Add => {
                if inputs.len() != 2 {
                    return Err(GpuError::InvalidOperation("Add operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].add(inputs[1])
                    .map_err(|e| GpuError::KernelExecutionError(e))
            },
            KernelOp::Mul => {
                if inputs.len() != 2 {
                    return Err(GpuError::InvalidOperation("Mul operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].mul(inputs[1])
                    .map_err(|e| GpuError::KernelExecutionError(e))
            },
            KernelOp::Sub => {
                if inputs.len() != 2 {
                    return Err(GpuError::InvalidOperation("Sub operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].sub(inputs[1])
                    .map_err(|e| GpuError::KernelExecutionError(e))
            },
            KernelOp::MatMul => {
                if inputs.len() != 2 {
                    return Err(GpuError::InvalidOperation("MatMul operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].matmul(inputs[1])
                    .map_err(|e| GpuError::KernelExecutionError(e))
            },
            _ => Err(GpuError::UnsupportedOperation(format!("Operation {:?} not implemented for CPU fallback", op))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.execution_time = execution_time;
            metrics.memory_bandwidth = 50.0; // CPU memory bandwidth
            metrics.occupancy = 100.0; // CPU always "fully occupied"
            metrics.flops = 100.0; // Lower FLOPS for CPU
        }

        result
    }

    fn supports_operation(&self, op: KernelOp) -> bool {
        matches!(op, KernelOp::Add | KernelOp::Mul | KernelOp::Sub | KernelOp::MatMul)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn get_metrics(&self) -> KernelMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn optimize_params(&self, _op: KernelOp, _params: &mut KernelParams) -> GpuResult<()> {
        // CPU doesn't need special parameter optimization
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_cpu_fallback_executor() {
        let executor = CpuFallbackExecutor::new();
        assert_eq!(executor.device_type(), DeviceType::Cpu);
        assert!(executor.supports_operation(KernelOp::Add));
        assert!(executor.supports_operation(KernelOp::Mul));
        assert!(!executor.supports_operation(KernelOp::Conv2D));
    }

    #[test]
    fn test_cpu_add_execution() {
        let executor = CpuFallbackExecutor::new();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = executor.execute(KernelOp::Add, &[&a, &b], &params).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        // Check metrics were recorded
        let metrics = executor.get_metrics();
        assert!(metrics.execution_time > Duration::ZERO);
    }

    #[test]
    fn test_kernel_params() {
        let mut params = KernelParams::default();
        params.input_shapes = vec![vec![3, 3], vec![3, 3]];
        params.output_shape = vec![3, 3];
        params.extra_params.insert("test_param".to_string(), 42.0);

        assert_eq!(params.input_shapes.len(), 2);
        assert_eq!(params.extra_params.get("test_param"), Some(&42.0));
    }

    #[test]
    fn test_kernel_metrics() {
        let metrics = KernelMetrics {
            execution_time: Duration::from_millis(10),
            memory_bandwidth: 100.0,
            occupancy: 80.0,
            flops: 1000.0,
        };

        assert_eq!(metrics.execution_time, Duration::from_millis(10));
        assert_eq!(metrics.memory_bandwidth, 100.0);
        assert_eq!(metrics.occupancy, 80.0);
        assert_eq!(metrics.flops, 1000.0);
    }
}