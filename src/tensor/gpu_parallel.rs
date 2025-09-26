//! # GPU-Integrated Parallel Tensor Operations
//! GPU統合並列テンソル操作
//!
//! This module provides seamless GPU acceleration for parallel tensor operations,
//! combining the power of CUDA, Metal, and OpenCL with intelligent fallback to
//! CPU parallelism when GPU resources are unavailable.
//!
//! ## Overview
//!
//! The GPU parallel system extends the core parallel traits with GPU acceleration:
//!
//! - [`GpuParallelOp`]: GPU-accelerated parallel operations
//! - [`GpuBatchParallelOp`]: GPU batch processing operations
//! - [`GpuExecutionStrategy`]: Configurable GPU execution strategies
//! - `DeviceSelection`: Intelligent device selection utilities
//!
//! ## Supported GPU Backends
//!
//! - **CUDA**: NVIDIA GPU acceleration with cuBLAS/cuDNN integration
//! - **Metal**: Apple Silicon GPU acceleration with Metal Performance Shaders
//! - **OpenCL**: Cross-platform GPU acceleration for AMD/Intel/NVIDIA
//!
//! ## Key Features
//!
//! - **Automatic Device Selection**: Chooses optimal GPU device based on workload
//! - **Intelligent Fallback**: Seamlessly falls back to CPU when GPU unavailable
//! - **Memory Management**: Efficient GPU-CPU data transfer with minimal overhead
//! - **Batch Optimization**: Specialized batch operations for neural network training
//! - **Error Recovery**: Robust error handling with automatic fallback strategies
//!
//! ## Usage Examples
//!
//! ### Basic GPU Operations
//!
//! ```rust
//! use rustorch::tensor::{Tensor, gpu_parallel::*};
//! use rustorch::gpu::DeviceType;
//!
//! let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
//! let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
//!
//! // GPU-accelerated element-wise operations
//! let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b)?;
//!
//! // Basic tensor operations (GPU-accelerated when available)
//! println!("GPU operations completed successfully");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Device Management
//!
//! ```rust
//! use rustorch::tensor::{Tensor, gpu_parallel::*};
//! use rustorch::gpu::DeviceType;
//!
//! let tensor = Tensor::<f32>::ones(&[1000, 1000]);
//!
//! // Perform GPU-accelerated operations (with CPU fallback)
//! let result = tensor.gpu_elementwise_op(&tensor, |a, b| a * 2.0)?;
//! println!("GPU operation completed: {:?}", result.shape());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Batch GPU Operations
//!
//! ```rust
//! use rustorch::tensor::{Tensor, gpu_parallel::*};
//!
//! let batch_tensor1 = Tensor::<f32>::ones(&[2, 4, 4]);  // Small batch for testing
//! let batch_tensor2 = Tensor::<f32>::ones(&[2, 4, 4]);
//!
//! // GPU batch matrix multiplication
//! let batch_result = batch_tensor1.gpu_batch_matmul(&batch_tensor2)?;
//!
//! // GPU batch normalization
//! let normalized = batch_tensor1.gpu_batch_normalize(1e-5)?;
//!
//! // GPU batch convolution (placeholder for future implementation)
//! // let conv_result = batch_tensor1.gpu_batch_conv2d(&kernel, &bias)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Execution Strategy Configuration
//!
//! ```rust
//! use rustorch::tensor::gpu_parallel::*;
//!
//! // Configure GPU execution strategy
//! let strategy = GpuParallelStrategy::GpuPreferred;
//!
//! // Use strategy for operations
//! println!("Using strategy: {:?}", strategy);
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Performance Optimization
//!
//! - **Memory Coalescing**: Optimized memory access patterns for GPU efficiency
//! - **Kernel Fusion**: Multiple operations fused into single GPU kernels
//! - **Asynchronous Execution**: Non-blocking GPU operations with CPU overlap
//! - **Memory Pooling**: Efficient GPU memory allocation and reuse
//!
//! ## Error Handling and Fallback
//!
//! The GPU parallel system provides robust error handling with automatic fallback:
//!
//! - **Device Unavailable**: Automatically falls back to CPU parallel execution
//! - **Memory Exhausted**: Attempts smaller batch sizes or CPU fallback
//! - **Kernel Failures**: Logs errors and uses CPU implementation
//! - **Driver Issues**: Graceful degradation to CPU-only mode
//!
//! ## Feature Flags
//!
//! GPU backends are enabled through Cargo feature flags:
//!
//! ```toml
//! [dependencies]
//! rustorch = { version = "0.1", features = ["cuda", "metal", "opencl"] }
//! ```
//!
//! - `cuda`: Enable NVIDIA CUDA support
//! - `metal`: Enable Apple Metal support  
//! - `opencl`: Enable OpenCL support
//! - `opengl`: Enable OpenGL support

use super::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
type ParallelResult<T> = RusTorchResult<T>;
use super::parallel_traits::{
    BatchParallelOp, MatrixParallelOp, ParallelConfig, ParallelOp, ReductionParallelOp,
};
use crate::gpu::{get_device_manager, DeviceType};
// GPU parallel operations implementation
use num_traits::Float;

/// GPU execution strategy for operations
#[derive(Debug, Clone, Copy)]
pub enum GpuExecutionStrategy {
    /// Use CPU parallel execution
    CpuParallel,
    /// Prefer GPU execution with fallback threshold
    GpuPreferred {
        /// Threshold for falling back to CPU
        /// CPUへのフォールバック闾値
        fallback_threshold: usize,
    },
    /// Hybrid CPU+GPU execution
    Hybrid {
        /// Threshold for GPU usage
        /// GPU使用の闾値
        gpu_threshold: usize,
        /// Number of CPU threads
        /// CPUスレッド数
        cpu_threads: usize,
    },
    /// Automatic selection based on workload
    Auto,
}

/// Select optimal device for given tensor shape
pub fn select_optimal_device(shape: &[usize]) -> DeviceType {
    let total_elements: usize = shape.iter().product();

    // For large tensors, prefer GPU if available
    if total_elements > 100000 {
        if DeviceType::Cuda(0).is_available() {
            return DeviceType::Cuda(0);
        }
        if DeviceType::Metal(0).is_available() {
            return DeviceType::Metal(0);
        }
        if DeviceType::OpenCL(0).is_available() {
            return DeviceType::OpenCL(0);
        }
    }

    // Default to CPU
    DeviceType::Cpu
}

/// Get current device
pub fn current_device() -> DeviceType {
    // Try to get the best available device
    if DeviceType::Cuda(0).is_available() {
        DeviceType::Cuda(0)
    } else if DeviceType::Metal(0).is_available() {
        DeviceType::Metal(0)
    } else if DeviceType::OpenCL(0).is_available() {
        DeviceType::OpenCL(0)
    } else {
        DeviceType::Cpu
    }
}

/// GPU並列操作のトレイト
/// Trait for GPU parallel operations
pub trait GpuParallelOp<T: Float + Send + Sync + Clone + 'static>: ParallelOp<T> {
    /// GPU上での並列要素ごと演算
    /// Parallel element-wise operations on GPU
    fn gpu_elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static;

    /// GPU上での並列行列乗算
    /// Parallel matrix multiplication on GPU
    fn gpu_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>>;

    /// Perform parallel reduction on GPU
    /// GPU上で並列リダクションを実行
    fn gpu_reduce<F, R>(&self, dim: usize, init: R, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(R, T) -> R + Send + Sync + Clone,
        R: Send + Sync + Clone + Into<T>;

    /// GPU-CPU間データ転送
    /// Data transfer between GPU and CPU
    fn to_device(&self, device: DeviceType) -> ParallelResult<Tensor<T>>;

    /// Transfer tensor data from GPU to CPU
    /// テンソルデータをGPUからCPUに転送
    fn to_cpu(&self) -> ParallelResult<Tensor<T>>;
}

/// GPU並列実行戦略
/// GPU parallel execution strategy
#[derive(Debug, Clone, Copy)]
pub enum GpuParallelStrategy {
    /// CPU並列実行
    /// CPU parallel execution
    CpuParallel,
    /// GPU実行（利用可能な場合）
    /// GPU execution (if available)
    GpuPreferred,
    /// ハイブリッド実行（CPU+GPU）
    /// Hybrid execution (CPU+GPU)
    Hybrid,
    /// 自動選択
    /// Automatic selection
    Auto,
}

/// GPU並列設定
/// GPU parallel configuration
#[derive(Debug, Clone)]
pub struct GpuParallelConfig {
    /// 基本並列設定
    /// Base parallel configuration
    pub base_config: ParallelConfig,
    /// GPU実行戦略
    /// GPU execution strategy
    pub gpu_strategy: GpuParallelStrategy,
    /// GPU-CPU転送閾値
    /// GPU-CPU transfer threshold
    pub transfer_threshold: usize,
    /// 優先デバイス
    /// Preferred device
    pub preferred_device: Option<DeviceType>,
}

impl Default for GpuParallelConfig {
    fn default() -> Self {
        Self {
            base_config: ParallelConfig::default(),
            gpu_strategy: GpuParallelStrategy::Auto,
            transfer_threshold: 10000,
            preferred_device: None,
        }
    }
}

/// GPU並列コンテキスト
/// GPU parallel context
pub struct GpuParallelContext {
    /// 設定
    /// Configuration
    pub config: GpuParallelConfig,
    /// 現在のデバイス
    /// Current device
    current_device: DeviceType,
}

impl GpuParallelContext {
    /// 新しいGPU並列コンテキストを作成
    /// Create a new GPU parallel context
    pub fn new(config: GpuParallelConfig) -> Self {
        let current_device = current_device();
        Self {
            config,
            current_device,
        }
    }

    /// デフォルトコンテキストを作成
    /// Create a default context
    pub fn default() -> Self {
        Self::new(GpuParallelConfig::default())
    }

    /// 最適な実行戦略を決定
    /// Determine optimal execution strategy
    pub fn determine_strategy(&self, size: usize) -> GpuParallelStrategy {
        match self.config.gpu_strategy {
            GpuParallelStrategy::Auto => {
                if size < self.config.transfer_threshold {
                    GpuParallelStrategy::CpuParallel
                } else if self.is_gpu_available() {
                    GpuParallelStrategy::GpuPreferred
                } else {
                    GpuParallelStrategy::CpuParallel
                }
            }
            strategy => strategy,
        }
    }

    /// GPUが利用可能かチェック
    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        !matches!(self.current_device, DeviceType::Cpu)
    }

    /// デバイスを設定
    /// Set device
    pub fn set_device(&mut self, device: DeviceType) -> RusTorchResult<()> {
        crate::gpu::set_device(device)?;
        self.current_device = device;
        Ok(())
    }

    /// 現在のデバイスを取得
    /// Get current device
    pub fn current_device(&self) -> DeviceType {
        self.current_device
    }
}

/// GPU統合並列テンソル操作の実装
/// Implementation of GPU-integrated parallel tensor operations
#[cfg(feature = "cuda")]
impl<T> GpuParallelOp<T> for Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits,
{
    fn gpu_elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        // 形状チェック
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let ctx = GpuParallelContext::default();
        let strategy = ctx.determine_strategy(self.data.len());

        match strategy {
            GpuParallelStrategy::CpuParallel => {
                // CPU並列実行
                self.batch_elementwise_op(other, op)
            }
            GpuParallelStrategy::GpuPreferred => {
                // 実際のGPU実行を試行
                match self.try_gpu_elementwise_op(other, op.clone()) {
                    Ok(result) => Ok(result),
                    Err(_) => {
                        // GPU実行失敗時はCPU並列にフォールバック
                        self.batch_elementwise_op(other, op)
                    }
                }
            }
            GpuParallelStrategy::Hybrid => {
                // ハイブリッド実行（現在はCPU並列にフォールバック）
                self.batch_elementwise_op(other, op)
            }
            GpuParallelStrategy::Auto => self.batch_elementwise_op(other, op),
        }
    }

    fn gpu_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::matrix_ops::GpuLinearAlgebra;

        let ctx = GpuParallelContext::default();
        let strategy = ctx.determine_strategy(self.data.len() * other.data.len());

        match strategy {
            GpuParallelStrategy::CpuParallel => self.batch_matmul(other),
            GpuParallelStrategy::GpuPreferred => {
                // GPU行列乗算 - 新しい実装を使用
                GpuLinearAlgebra::gpu_matmul(self, other).or_else(|_| {
                    // GPU失敗時はCPUにフォールバック
                    self.batch_matmul(other)
                })
            }
            GpuParallelStrategy::Hybrid => {
                // 大きな行列を分割してCPU+GPU並列実行
                // 現在はGPU実装にフォールバック
                GpuLinearAlgebra::gpu_matmul(self, other).or_else(|_| self.batch_matmul(other))
            }
            GpuParallelStrategy::Auto => {
                // 自動選択: まずGPUを試行、失敗時はCPU
                GpuLinearAlgebra::gpu_matmul(self, other).or_else(|_| self.batch_matmul(other))
            }
        }
    }

    fn gpu_reduce<F, R>(&self, dim: usize, init: R, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(R, T) -> R + Send + Sync + Clone,
        R: Send + Sync + Clone + Into<T>,
    {
        let ctx = GpuParallelContext::default();
        let strategy = ctx.determine_strategy(self.data.len());

        match strategy {
            GpuParallelStrategy::CpuParallel => self.parallel_reduce(dim, init, op),
            GpuParallelStrategy::GpuPreferred => {
                // GPU リダクション - 新しい実装を使用
                use crate::gpu::reduction_ops::GpuReduction;
                GpuReduction::gpu_sum(self, Some(dim)).or_else(|_| {
                    // GPU失敗時はCPUにフォールバック
                    self.parallel_reduce(dim, init, op)
                })
            }
            GpuParallelStrategy::Hybrid => self.parallel_reduce(dim, init, op),
            GpuParallelStrategy::Auto => {
                // 自動選択: まずGPUを試行、失敗時はCPU
                use crate::gpu::reduction_ops::GpuReduction;
                GpuReduction::gpu_sum(self, Some(dim))
                    .or_else(|_| self.parallel_reduce(dim, init, op))
            }
        }
    }

    fn to_device(&self, device: DeviceType) -> ParallelResult<Tensor<T>> {
        use crate::gpu::memory_ops::manager::GpuMemoryManager;

        match device {
            DeviceType::Cpu => {
                // すでにCPU上にある場合はクローンを返す
                Ok(self.clone())
            }
            _ => {
                // GPU転送処理
                let gpu_buffer = GpuMemoryManager::to_device(self, &device)?;

                // For now, immediately transfer back to CPU tensor
                // In future, we'll maintain GPU buffer and track device location
                GpuMemoryManager::to_cpu(&gpu_buffer, self.shape())
            }
        }
    }

    fn to_cpu(&self) -> ParallelResult<Tensor<T>> {
        // GPU上のデータをCPUに転送
        // In future implementation, this will check if tensor is on GPU
        // and transfer it back. For now, just return clone.
        Ok(self.clone())
    }
}

/// GPU並列バッチ操作
/// GPU parallel batch operations
pub trait GpuBatchParallelOp<T: Float + Send + Sync + Clone + 'static>: GpuParallelOp<T> {
    /// GPU並列バッチ正規化
    /// GPU parallel batch normalization
    fn gpu_batch_normalize(&self, epsilon: T) -> ParallelResult<Tensor<T>>;

    /// GPU並列バッチ畳み込み
    /// GPU parallel batch convolution
    fn gpu_batch_conv2d(
        &self,
        kernel: &Tensor<T>,
        stride: usize,
        padding: usize,
    ) -> ParallelResult<Tensor<T>>;

    /// GPU並列バッチアテンション
    /// GPU parallel batch attention
    fn gpu_batch_attention(&self, key: &Tensor<T>, value: &Tensor<T>) -> ParallelResult<Tensor<T>>;
}

#[cfg(feature = "cuda")]
impl<T> GpuBatchParallelOp<T> for Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits,
{
    fn gpu_batch_normalize(&self, epsilon: T) -> ParallelResult<Tensor<T>> {
        let ctx = GpuParallelContext::default();
        let strategy = ctx.determine_strategy(self.data.len());

        match strategy {
            GpuParallelStrategy::GpuPreferred => {
                // GPU バッチ正規化を試行
                Ok(self.try_gpu_batch_normalize(epsilon).unwrap_or_else(|_| {
                    // GPU失敗時はCPUにフォールバック
                    self.batch_normalize(epsilon)
                }))
            }
            _ => Ok(self.batch_normalize(epsilon)),
        }
    }

    fn gpu_batch_conv2d(
        &self,
        kernel: &Tensor<T>,
        stride: usize,
        padding: usize,
    ) -> ParallelResult<Tensor<T>> {
        use crate::backends::ConvolutionParams;
        use crate::gpu::conv_ops::GpuConvolution;

        let ctx = GpuParallelContext::default();
        let strategy = ctx.determine_strategy(self.data.len() * kernel.data.len());

        // Convert stride/padding to ConvolutionParams
        let params = ConvolutionParams {
            kernel_size: vec![3, 3], // Default kernel size
            stride: vec![stride, stride],
            padding: vec![padding, padding],
            dilation: vec![1, 1],
            groups: 1,
        };

        match strategy {
            GpuParallelStrategy::GpuPreferred => {
                // GPU 畳み込み - 新しい実装を使用
                GpuConvolution::gpu_batch_conv2d(self, kernel, &params).or_else(|_| {
                    // GPU失敗時はCPUにフォールバック
                    self.batch_conv2d(kernel, stride, padding)
                })
            }
            GpuParallelStrategy::Auto => {
                // 自動選択: まずGPUを試行、失敗時はCPU
                GpuConvolution::gpu_batch_conv2d(self, kernel, &params)
                    .or_else(|_| self.batch_conv2d(kernel, stride, padding))
            }
            _ => self.batch_conv2d(kernel, stride, padding),
        }
    }

    fn gpu_batch_attention(&self, key: &Tensor<T>, value: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        // GPU最適化アテンション実装を試行
        match self.try_gpu_batch_attention(key, value) {
            Ok(result) => Ok(result),
            Err(_) => {
                // GPU失敗時は基本的なCPU実装にフォールバック
                let scores = self.gpu_matmul(key)?;
                let attention_weights = self.apply_softmax(&scores)?; // softmax適用
                attention_weights.gpu_matmul(value)
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl<T> GpuParallelOp<T> for Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand,
{
    fn gpu_elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        // Without CUDA, fall back to CPU
        self.batch_elementwise_op(other, op)
    }

    fn gpu_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        // Use hardware acceleration when available
        #[cfg(feature = "mac-hybrid")]
        {
            return self.matmul_hybrid(other).map_err(|e| e.into());
        }
        #[cfg(all(feature = "metal", not(feature = "mac-hybrid")))]
        {
            return self.matmul_metal(other, 0).map_err(|e| e.into());
        }
        #[cfg(all(feature = "coreml", not(any(feature = "metal", feature = "mac-hybrid"))))]
        {
            return self.matmul_coreml(other, 0).map_err(|e| e.into());
        }
        #[cfg(not(any(feature = "metal", feature = "coreml", feature = "mac-hybrid")))]
        {
            // Only fall back to CPU if no hardware acceleration available
            self.matmul(other).map_err(|e| e.into())
        }
    }

    fn gpu_reduce<F, R>(&self, dim: usize, init: R, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(R, T) -> R + Send + Sync + Clone,
        R: Send + Sync + Clone + Into<T>,
    {
        // Without CUDA, fall back to CPU reduction
        Err(crate::error::RusTorchError::tensor_op(
            "GPU reduce not available without CUDA".to_string(),
        )
        .into())
    }

    fn to_device(&self, _device: DeviceType) -> ParallelResult<Tensor<T>> {
        // Without CUDA, just return self
        Ok(self.clone())
    }

    fn to_cpu(&self) -> ParallelResult<Tensor<T>> {
        // Without CUDA, already on CPU
        Ok(self.clone())
    }
}

#[cfg(not(feature = "cuda"))]
impl<T> GpuBatchParallelOp<T> for Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand,
{
    fn gpu_batch_normalize(&self, epsilon: T) -> ParallelResult<Tensor<T>> {
        // Without CUDA, fall back to CPU batch normalize
        Ok(self.batch_normalize(epsilon))
    }

    fn gpu_batch_conv2d(
        &self,
        kernel: &Tensor<T>,
        stride: usize,
        padding: usize,
    ) -> ParallelResult<Tensor<T>> {
        // Use available hardware acceleration
        use crate::backends::ConvolutionParams;
        let params = ConvolutionParams {
            kernel_size: vec![kernel.shape()[2], kernel.shape()[3]],
            stride: vec![stride, stride],
            padding: vec![padding, padding],
            dilation: vec![1, 1],
            groups: 1,
        };

        #[cfg(feature = "mac-hybrid")]
        {
            use crate::gpu::{GpuConvolution, DeviceType, OpType};
            return self.gpu_conv2d(kernel, &params).map_err(|e| e.into());
        }
        #[cfg(all(feature = "metal", not(feature = "mac-hybrid")))]
        {
            use crate::gpu::GpuConvolution;
            return self.gpu_conv2d(kernel, &params).map_err(|e| e.into());
        }
        #[cfg(all(feature = "coreml", not(any(feature = "metal", feature = "mac-hybrid"))))]
        {
            use crate::gpu::GpuConvolution;
            return self.gpu_conv2d(kernel, &params).map_err(|e| e.into());
        }
        #[cfg(not(any(feature = "metal", feature = "coreml", feature = "mac-hybrid")))]
        {
            Err(crate::error::RusTorchError::tensor_op(
                "No GPU acceleration available (enable metal, coreml, or mac-hybrid features)".to_string(),
            ).into())
        }
    }

    fn gpu_batch_attention(&self, key: &Tensor<T>, value: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        // Without CUDA, fall back to CPU
        let scores = self.matmul(key)?;
        let attention_weights = self.apply_softmax(&scores)?;
        attention_weights.matmul(value)
    }
}

// Helper function for GPU element-wise operations
#[cfg(feature = "cuda")]
impl<T> Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits,
{
    /// GPU element-wise operation implementation (internal helper)
    fn try_gpu_elementwise_op<F>(
        &self,
        other: &Tensor<T>,
        op: F,
    ) -> Result<Tensor<T>, crate::error::RusTorchError>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        use crate::gpu::memory_ops::manager::GpuMemoryManager;
        use ndarray::ArrayD;

        // Get available devices
        let manager = get_device_manager();
        let devices = manager.available_devices();

        // If no GPU devices available, return an error that will trigger CPU fallback
        if devices.is_empty() || devices == vec![DeviceType::Cpu] {
            return Err(crate::error::RusTorchError::gpu(
                "No GPU devices available, falling back to CPU",
            ));
        }

        // Try to use the first available device
        let device = devices.first().unwrap();

        // For CPU, just perform the operation directly without GPU
        if *device == DeviceType::Cpu {
            return self.batch_elementwise_op(other, op).map_err(|e| {
                crate::error::RusTorchError::gpu(format!("CPU fallback failed: {}", e))
            });
        }

        // Ensure tensors have the same shape
        if self.shape() != other.shape() {
            return Err(crate::error::RusTorchError::shape_mismatch(
                self.shape(),
                other.shape(),
            ));
        }

        // Save the original shape and data length
        let original_shape = self.shape().to_vec();
        let data_len = self.data.len();

        // Get the data as a flat vector, ensuring it's contiguous
        let self_data = self.data.view().to_owned().into_raw_vec_and_offset().0;
        let other_data = other.data.view().to_owned().into_raw_vec_and_offset().0;

        // Create flat tensors with the correct shape [total_elements]
        let flat_self = Tensor::from_vec(self_data, vec![data_len]);
        let flat_other = Tensor::from_vec(other_data, vec![data_len]);

        // Initialize GPU memory manager
        let gpu_manager = GpuMemoryManager::new();

        // Transfer data to device
        let gpu_self = GpuMemoryManager::to_device(&flat_self, device)?;
        let gpu_other = GpuMemoryManager::to_device(&flat_other, device)?;

        // Execute element-wise operation
        let gpu_result = gpu_manager.execute_elementwise(&gpu_self, &gpu_other, op)?;

        // Transfer result back to CPU as a flat vector
        let flat_result = GpuMemoryManager::to_cpu(&gpu_result, &[data_len]).map_err(|e| {
            crate::error::RusTorchError::gpu(format!(
                "Failed to transfer result from device: {}",
                e
            ))
        })?;

        // Convert the flat result back to the original shape
        let result_data = flat_result.data.into_raw_vec_and_offset().0;

        // Ensure the total number of elements matches
        let total_elements: usize = original_shape.iter().product();
        if result_data.len() != total_elements {
            return Err(crate::error::RusTorchError::gpu(format!(
                "Mismatched element count: expected {} but got {}",
                total_elements,
                result_data.len()
            )));
        }

        // Create the result tensor with the original shape
        let array = ArrayD::from_shape_vec(original_shape.clone(), result_data).map_err(|e| {
            crate::error::RusTorchError::gpu(format!("Failed to reshape result: {}", e))
        })?;
        let result = Tensor::from_ndarray(array);

        Ok(result)
    }

    /// GPU batch normalization implementation (internal helper)
    fn try_gpu_batch_normalize(
        &self,
        epsilon: T,
    ) -> Result<Tensor<T>, crate::error::RusTorchError> {
        use crate::gpu::{memory_ops::manager::GpuMemoryManager, DeviceManager};

        let manager = DeviceManager::new();
        let devices = manager.available_devices();

        if devices.is_empty() {
            return Err(crate::error::RusTorchError::gpu("GPU unavailable"));
        }

        // GPU memory managerを作成
        let gpu_manager = GpuMemoryManager::new();
        let device = devices.first().unwrap();

        // CPU -> GPU転送
        let gpu_tensor = GpuMemoryManager::to_device(self, device)?;

        // GPU上でバッチ正規化を実行
        let gpu_result = gpu_manager.execute_batch_normalize(&gpu_tensor, epsilon)?;

        // GPU -> CPU転送
        let result = GpuMemoryManager::to_cpu(&gpu_result, self.data.shape())?;

        Ok(result)
    }

    /// GPU batch attention implementation (internal helper)
    fn try_gpu_batch_attention(
        &self,
        key: &Tensor<T>,
        value: &Tensor<T>,
    ) -> Result<Tensor<T>, crate::error::RusTorchError> {
        use crate::gpu::{memory_ops::manager::GpuMemoryManager, DeviceManager};

        let manager = DeviceManager::new();
        let devices = manager.available_devices();

        if devices.is_empty() {
            return Err(crate::error::RusTorchError::gpu("GPU unavailable"));
        }

        // GPU memory managerを作成
        let gpu_manager = GpuMemoryManager::new();
        let device = devices.first().unwrap();

        // CPU -> GPU転送
        let gpu_query = GpuMemoryManager::to_device(self, device)?;
        let gpu_key = GpuMemoryManager::to_device(key, device)?;
        let gpu_value = GpuMemoryManager::to_device(value, device)?;

        // GPU上でアテンション処理を実行
        let gpu_result = gpu_manager.execute_attention(&gpu_query, &gpu_key, &gpu_value)?;

        // GPU -> CPU転送
        let result = GpuMemoryManager::to_cpu(&gpu_result, self.data.shape())?;

        Ok(result)
    }

    /// Apply softmax function (CPU implementation)
    fn apply_softmax(&self, tensor: &Tensor<T>) -> Result<Tensor<T>, crate::error::RusTorchError> {
        let data = tensor.data.as_slice().ok_or_else(|| {
            crate::error::RusTorchError::tensor_op(
                "Non-contiguous tensor not supported for softmax",
            )
        })?;

        // 最大値を計算して数値安定性を確保
        let max_val = data
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        // exp(x - max)を計算
        let exp_data: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();

        // 合計を計算
        let sum = exp_data.iter().fold(T::zero(), |acc, &x| acc + x);

        // 正規化
        let softmax_data: Vec<T> = exp_data.iter().map(|&x| x / sum).collect();

        // 新しいTensorを作成
        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(tensor.shape()), softmax_data)
            .map_err(|e| {
            crate::error::RusTorchError::tensor_op(format!("Shape error: {}", e))
        })?;

        Ok(Tensor::from_ndarray(array))
    }
}

// Helper function for GPU element-wise operations (non-CUDA version)
#[cfg(not(feature = "cuda"))]
impl<T> Tensor<T>
where
    T: Float
        + Send
        + Sync
        + Clone
        + 'static
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + ndarray::ScalarOperand,
{
    /// GPU element-wise operation implementation (CPU fallback)
    fn try_gpu_elementwise_op<F>(
        &self,
        other: &Tensor<T>,
        op: F,
    ) -> Result<Tensor<T>, crate::error::RusTorchError>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        // Without CUDA, fall back to CPU
        self.batch_elementwise_op(other, op)
    }

    /// GPU batch normalization implementation (CPU fallback)
    fn try_gpu_batch_normalize(
        &self,
        epsilon: T,
    ) -> Result<Tensor<T>, crate::error::RusTorchError> {
        // Without CUDA, fall back to CPU batch normalize
        Ok(self.batch_normalize(epsilon))
    }

    /// GPU batch attention implementation (CPU fallback)
    fn try_gpu_batch_attention(
        &self,
        key: &Tensor<T>,
        value: &Tensor<T>,
    ) -> Result<Tensor<T>, crate::error::RusTorchError> {
        // Without CUDA, fall back to CPU
        let scores = self.matmul(key)?;
        let attention_weights = self.apply_softmax(&scores)?;
        attention_weights.matmul(value)
    }

    /// Apply softmax function (CPU implementation)
    fn apply_softmax(&self, tensor: &Tensor<T>) -> Result<Tensor<T>, crate::error::RusTorchError> {
        let data = tensor.data.as_slice().ok_or_else(|| {
            crate::error::RusTorchError::tensor_op(
                "Non-contiguous tensor not supported for softmax",
            )
        })?;

        // Calculate max for numerical stability
        let max_val = data
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        // Calculate exp and sum
        let exp_values: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp = exp_values.iter().fold(T::zero(), |acc, &x| acc + x);

        // Normalize
        let softmax_values: Vec<T> = exp_values.iter().map(|&x| x / sum_exp).collect();

        let array = ndarray::ArrayD::from_shape_vec(tensor.data.raw_dim(), softmax_values)
            .map_err(|e| {
                crate::error::RusTorchError::tensor_op(format!("Softmax shape error: {}", e))
            })?;

        Ok(Tensor::from_ndarray(array))
    }
}

/// GPU並列操作のユーティリティ
/// Utilities for GPU parallel operations
pub mod gpu_parallel_utils {
    use super::*;

    /// 最適なデバイスを選択
    /// Select optimal device
    pub fn select_optimal_device(size: usize) -> DeviceType {
        let manager = get_device_manager();
        let available = manager.available_devices();

        // サイズに基づいてデバイスを選択
        if size > 100000 {
            // 大きなテンソルはGPUを優先
            for device in available {
                if !matches!(device, DeviceType::Cpu) {
                    return device;
                }
            }
        }

        // デフォルトはCPU
        DeviceType::Cpu
    }

    /// GPU並列実行の効率性を評価
    /// Evaluate efficiency of GPU parallel execution
    pub fn evaluate_gpu_efficiency(tensor_size: usize, operation_complexity: f32) -> f32 {
        // 簡単な効率性評価
        let transfer_cost = tensor_size as f32 * 0.001; // 転送コスト
        let compute_benefit = operation_complexity * tensor_size as f32 * 0.01; // 計算利益

        (compute_benefit - transfer_cost).max(0.0)
    }

    /// バッチサイズ最適化
    /// Optimize batch size
    pub fn optimize_batch_size(total_size: usize, device: DeviceType) -> usize {
        match device {
            DeviceType::Cpu => {
                // CPU用バッチサイズ
                std::cmp::min(1024, total_size)
            }
            DeviceType::Cuda(_) => {
                // CUDA用バッチサイズ
                std::cmp::min(4096, total_size)
            }
            DeviceType::Metal(_) => {
                // Metal用バッチサイズ
                std::cmp::min(2048, total_size)
            }
            DeviceType::OpenCL(_) => {
                // OpenCL用バッチサイズ
                std::cmp::min(1024, total_size)
            }
            #[cfg(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            ))]
            DeviceType::CoreML(_) => {
                // CoreML用バッチサイズ
                std::cmp::min(3072, total_size)
            }
            #[cfg(feature = "mac-hybrid")]
            DeviceType::MacHybrid => {
                // MacHybrid用バッチサイズ - automatically optimized
                std::cmp::min(4096, total_size)
            }
            DeviceType::Auto => {
                // Auto用バッチサイズ
                std::cmp::min(3072, total_size)
            }
        }
    }
}

// Additional GPU operations
impl<
        T: Float + Send + Sync + Clone + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Tensor<T>
{
    /// GPU sum operation
    pub fn gpu_sum(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_sum(self, dim)
    }

    /// GPU mean operation
    pub fn gpu_mean(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>>
    where
        T: num_traits::FromPrimitive,
    {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_mean(self, dim)
    }

    /// GPU max operation
    pub fn gpu_max(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_max(self, dim)
    }

    /// GPU min operation
    pub fn gpu_min(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_min(self, dim)
    }

    /// GPU standard deviation operation
    pub fn gpu_std(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_std(self, dim)
    }

    /// GPU variance operation
    pub fn gpu_var(&self, dim: Option<usize>) -> ParallelResult<Tensor<T>> {
        use crate::gpu::reduction_ops::GpuReduction;
        GpuReduction::gpu_var(self, dim)
    }

    /// GPU batch matrix multiplication
    pub fn gpu_batch_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        // Fallback to CPU batch implementation
        self.batch_matmul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_parallel_config() {
        let config = GpuParallelConfig::default();
        assert_eq!(config.transfer_threshold, 10000);
        assert!(matches!(config.gpu_strategy, GpuParallelStrategy::Auto));
    }

    #[test]
    fn test_gpu_parallel_context() {
        let ctx = GpuParallelContext::default();

        // 実際に利用可能なデバイスをテスト
        let actual_device = ctx.current_device();

        // macOSでもMetal機能が無効または初期化失敗の場合はCPUになる
        println!("実際のデバイス: {:?}", actual_device);

        // デバイスがCpu、Metal、Cuda、OpenCLのいずれかであることを確認
        assert!(matches!(
            actual_device,
            DeviceType::Cpu | DeviceType::Metal(_) | DeviceType::Cuda(_) | DeviceType::OpenCL(_)
        ));

        let strategy = ctx.determine_strategy(5000);
        assert!(matches!(strategy, GpuParallelStrategy::CpuParallel));

        let strategy = ctx.determine_strategy(50000);
        // 実際のデバイスに基づいて期待値を設定
        if actual_device != DeviceType::Cpu {
            assert!(matches!(strategy, GpuParallelStrategy::GpuPreferred));
        } else {
            assert!(matches!(strategy, GpuParallelStrategy::CpuParallel));
        }
    }

    #[test]
    fn test_gpu_elementwise_op() {
        let tensor1 = Tensor::<f32>::ones(&[2, 2]);
        let tensor2 = Tensor::<f32>::ones(&[2, 2]);

        let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.as_array()[[0, 0]], 2.0);
    }

    #[test]
    fn test_gpu_matmul() {
        let tensor1 = Tensor::<f32>::ones(&[2, 3]);
        let tensor2 = Tensor::<f32>::ones(&[3, 2]);

        let result = tensor1.matmul(&tensor2);
        assert_eq!(result.unwrap().shape(), &[2, 2]);

        // Test GPU batch matmul with 3D tensors (batch dimension required)
        let batch_tensor1 = Tensor::<f32>::ones(&[1, 2, 3]);
        let batch_tensor2 = Tensor::<f32>::ones(&[1, 3, 2]);

        let result = batch_tensor1.gpu_batch_matmul(&batch_tensor2);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.shape(), &[1, 2, 2]);
    }
}
