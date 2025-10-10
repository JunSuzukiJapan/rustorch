//! # GPU Acceleration Module
//! RusTorchのGPU加速モジュール
//!
//! This module provides comprehensive GPU acceleration support for RusTorch,
//! including device management, memory allocation, and kernel execution
//! across multiple GPU backends (CUDA, Metal, OpenCL).
//!
//! ## Supported GPU Backends
//!
//! - **CUDA**: NVIDIA GPU acceleration with cuBLAS/cuDNN integration
//! - **Metal**: Apple Silicon GPU acceleration with Metal Performance Shaders
//! - **OpenCL**: Cross-platform GPU acceleration for AMD/Intel/NVIDIA
//!
//! ## Core Components
//!
//! - `DeviceType`: GPU device type enumeration (CUDA, Metal, OpenCL)
//! - `GpuContext`: GPU context management and device state
//! - `DeviceManager`: Global device management and selection
//! - `GpuMemoryPool`: Efficient GPU memory allocation and pooling
//! - `cuda_kernels`: CUDA-specific kernel implementations
//! - `metal_kernels`: Metal-specific kernel implementations  
//! - `opencl_kernels`: OpenCL-specific kernel implementations
//!
//! ## Key Features
//!
//! - **Automatic Device Detection**: Discovers available GPU devices at runtime
//! - **Intelligent Device Selection**: Chooses optimal device based on workload
//! - **Memory Management**: Efficient GPU memory allocation with pooling
//! - **Error Handling**: Robust error handling with automatic fallback
//! - **Cross-Platform**: Unified API across different GPU vendors
//!
//! ## Usage Examples
//!
//! ### Device Management
//!
//! ```no_run
//! use rustorch::gpu::get_device_manager;
//! // Example GPU usage (implementation dependent on backend)
//!
//! // Get available devices
//! let manager = get_device_manager();
//! let devices = manager.available_devices();
//! println!("Available devices: {:?}", devices);
//!
//! // Check current device
//! let current = manager.current_device();
//! println!("Current device: {:?}", current);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### GPU Context Creation
//!
//! ```rust
//! use rustorch::gpu::{GpuContext, DeviceType};
//!
//! // Create GPU context - will fallback to CPU if CUDA unavailable
//! let context_result = GpuContext::new(DeviceType::Cuda(0));
//! if let Ok(context) = context_result {
//!     println!("Created context for device: {:?}", context.device());
//!     // GPU context created successfully
//! } else {
//!     println!("CUDA not available, using CPU fallback");
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Memory Pool Operations
//!
//! ```rust
//! use rustorch::gpu::{DeviceType, memory::GpuMemoryPool};
//!
//! // Try to create GPU memory pool - fallback if CUDA unavailable
//! let pool_result = GpuMemoryPool::new(DeviceType::Cuda(0), 1024 * 1024 * 100);
//! if let Ok(mut pool) = pool_result {
//!     if let Ok(buffer) = pool.allocate(1024 * 1024) {
//!         println!("Allocated buffer size: {}", buffer.size);
//!     }
//! } else {
//!     println!("GPU memory pool unavailable, using system memory");
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Feature Flags
//!
//! GPU backends are controlled by Cargo feature flags:
//!
//! ```toml
//! [dependencies]
//! rustorch = { version = "0.1", features = ["cuda", "metal", "opencl"] }
//! ```
//!
//! - `cuda`: Enable NVIDIA CUDA support
//! - `metal`: Enable Apple Metal support
//! - `opencl`: Enable OpenCL support
//!
//! ## Error Handling
//!
//! The GPU module provides comprehensive error handling through `GpuError`:
//!
//! - **DeviceNotFound**: Requested GPU device is not available
//! - **OutOfMemory**: Insufficient GPU memory for operation
//! - **KernelLaunchFailed**: GPU kernel execution failed
//! - **DriverError**: GPU driver or runtime error
//! - **UnsupportedOperation**: Operation not supported on current device
//!
//! ## Performance Considerations
//!
//! - **Memory Coalescing**: Optimize memory access patterns for GPU efficiency
//! - **Kernel Occupancy**: Balance thread blocks for optimal GPU utilization
//! - **Memory Bandwidth**: Minimize GPU-CPU data transfers
//! - **Asynchronous Execution**: Use streams for overlapping computation and transfer

/// GPU activation operations and optimization
/// GPU活性化関数演算と最適化
pub mod activation_ops;
/// Performance benchmark suite for GPU operations
/// GPU演算用パフォーマンスベンチマークスイート
pub mod benchmark;
/// GPU convolution operations and cuDNN/MPS integration
/// GPU畳み込み演算とcuDNN/MPS統合
pub mod conv_ops;
pub mod cuda_enhanced;

/// CUDA kernel implementations
/// CUDAカーネル実装
pub mod cuda_kernels;
pub mod custom_kernels;
/// Device management module for GPU operations
/// GPU操作のためのデバイス管理モジュール
pub mod device;
/// Device caching module for optimized initialization
/// 最適化された初期化用デバイスキャッシュモジュール
pub mod device_cache;
/// Integration tests for unified GPU kernel system
/// 統一GPUカーネルシステムの統合テスト
#[cfg(test)]
pub mod integration_tests;
/// GPU kernel execution and management
/// GPUカーネル実行と管理
pub mod kernels;
/// GPU matrix operations and BLAS integration
/// GPU行列演算とBLAS統合
pub mod matrix_ops;
/// GPU memory management and allocation
/// GPUメモリ管理とアロケーション
pub mod memory;
/// GPU memory operations (modular implementation)
/// GPUメモリ操作（モジュラー実装）
pub mod memory_ops;
/// GPU memory transfer operations
/// GPUメモリ転送操作
pub mod memory_transfer;
pub mod metal_kernels;
#[cfg(feature = "metal")]
pub mod metal_matmul_raii;
#[cfg(feature = "metal")]
pub mod metal_objc_bridge;
#[cfg(feature = "metal")]
pub mod objc_bridge;
/// Batch processing kernel wrappers for Metal GPU
/// Metal GPU用バッチ処理カーネルラッパー
#[cfg(feature = "metal")]
pub mod batch_kernels;
pub mod opencl_kernels;
pub mod opencl_optimized;
pub mod performance_benchmark;
/// GPU performance optimizer
/// GPUパフォーマンス最適化器
pub mod performance_optimizer;
/// GPU reduction operations and optimizations
/// GPUリダクション演算と最適化
pub mod reduction_ops;
/// Simple Metal GPU testing and benchmarking
/// シンプルなMetal GPUテストとベンチマーク
pub mod simple_metal_test;
/// Unified kernel interface for cross-platform GPU acceleration (simplified)
/// クロスプラットフォームGPU加速のための統一カーネルインターフェース（簡潔版）
pub mod unified_kernel_simple;
/// GPU kernel validation and testing
/// GPUカーネル検証とテスト
pub mod validation;

/// Smart device selection module for optimized operation routing
/// 最適化された操作ルーティング用スマートデバイス選択モジュール
pub mod smart_device_selector;

/// GPU vs CPU verification tests
/// GPU vs CPU検証テスト
pub mod verification_tests;

/// Multi-GPU distributed processing and communication
/// マルチGPU分散処理と通信
pub mod multi_gpu;

/// GPU synchronization primitives for multi-GPU operations
/// マルチGPU操作用GPU同期プリミティブ
pub mod sync_primitives;

/// Distributed training infrastructure for multi-GPU learning
/// マルチGPU学習用分散学習インフラストラクチャ
pub mod distributed_training;

/// Multi-GPU performance profiling and benchmarking
/// マルチGPUパフォーマンスプロファイリングとベンチマーキング
pub mod multi_gpu_profiler;

/// Hybrid execution engine for CoreML + GPU fallback
/// CoreML + GPU フォールバック用ハイブリッド実行エンジン
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
pub mod hybrid_executor;

/// Unified CoreML module for Apple Neural Engine integration
/// Apple Neural Engine統合用統一CoreMLモジュール
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
pub mod coreml;

// Re-export GPU traits
pub use activation_ops::GpuActivation;
pub use conv_ops::GpuConvolution;
pub use matrix_ops::GpuLinearAlgebra;

use std::fmt;
// use crate::error::{RusTorchError, RusTorchResult}; // Currently unused

/// GPU device types with CoreML and hybrid support
/// GPU デバイスタイプ（CoreMLとハイブリッド対応）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    /// CPUデバイス
    Cpu,
    /// CUDA GPU device
    /// CUDA GPUデバイス
    Cuda(usize), // device_id
    /// Metal GPU device (for Apple Silicon)
    /// Metal GPUデバイス（Apple Silicon用）
    Metal(usize),
    /// OpenCL GPU device
    /// OpenCL GPUデバイス
    OpenCL(usize),
    /// CoreML device (Apple Neural Engine + GPU)
    /// CoreMLデバイス（Apple Neural Engine + GPU）
    #[cfg(feature = "coreml")]
    CoreML(usize),
    /// Hybrid CoreML with GPU fallback
    /// GPU フォールバック付きハイブリッドCoreML
    #[cfg(feature = "coreml-hybrid")]
    CoreMLHybrid {
        coreml_id: usize,
        fallback_gpu: Option<GpuDevice>,
    },
    /// Auto-select best available device
    /// 利用可能な最高性能デバイスを自動選択
    Auto,
    /// Mac hybrid: Intelligent Metal/CoreML selection (mac-hybrid feature only)
    /// Mac ハイブリッド: Metal/CoreML インテリジェント選択（mac-hybrid フィーチャーのみ）
    #[cfg(feature = "mac-hybrid")]
    MacHybrid,
}

/// GPU device types for fallback
/// フォールバック用GPUデバイスタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDevice {
    Cuda(usize),
    Metal(usize),
    OpenCL(usize),
}
/// Operation types for device capability checking
/// デバイス能力チェック用演算タイプ
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum OpType {
    /// Linear algebra operations (matmul, etc.)
    LinearAlgebra,
    /// Convolution operations
    Convolution,
    /// Activation functions
    Activation,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
    /// Normalization operations
    Normalization,
    /// Complex number operations (CoreML unsupported)
    ComplexMath,
    /// Statistical distributions (CoreML unsupported)
    Distribution,
    /// Custom kernel operations (CoreML unsupported)
    CustomKernel,
    /// Distributed operations (CoreML unsupported)
    DistributedOps,
}

/// Device capability information
/// デバイス能力情報
#[derive(Debug)]
pub struct DeviceCapability {
    pub device_type: DeviceType,
    pub supports_f16: bool,
    pub supports_f32: bool,
    pub supports_f64: bool,
    pub supports_complex: bool,
    pub supports_distributed: bool,
    pub max_memory_gb: f32,
    pub supported_operations: std::collections::HashSet<OpType>,
}

impl DeviceCapability {
    /// Check if device supports specific operation
    /// デバイスが特定の演算をサポートするかチェック
    pub fn supports_operation(&self, op_type: &OpType) -> bool {
        self.supported_operations.contains(op_type)
    }

    /// Get CoreML capability
    /// CoreML能力を取得
    #[cfg(feature = "coreml")]
    pub fn coreml_capability() -> Self {
        let mut supported_ops = std::collections::HashSet::new();
        supported_ops.insert(OpType::LinearAlgebra);
        supported_ops.insert(OpType::Convolution);
        supported_ops.insert(OpType::Activation);
        supported_ops.insert(OpType::Reduction);
        supported_ops.insert(OpType::Normalization);

        Self {
            device_type: DeviceType::CoreML(0),
            supports_f16: true,
            supports_f32: true,
            supports_f64: false,         // CoreML limitation
            supports_complex: false,     // CoreML limitation
            supports_distributed: false, // CoreML limitation
            max_memory_gb: 8.0,          // Typical Apple Silicon unified memory
            supported_operations: supported_ops,
        }
    }
}

#[cfg(feature = "mac-hybrid")]
impl DeviceType {
    /// Intelligent device selection for Mac hybrid feature
    /// Mac ハイブリッドフィーチャー用インテリジェント・デバイス選択
    pub fn select_best_for_operation(op_type: &OpType, tensor_size: Option<usize>) -> DeviceType {
        use crate::backends::DeviceManager;
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        // Check if both backends are available
        let coreml_available = DeviceManager::is_coreml_available();
        let metal_available = MetalKernelExecutor::new().is_ok();

        // If only one backend is available, use it
        if coreml_available && !metal_available {
            return DeviceType::CoreML(0);
        }
        if !coreml_available && metal_available {
            return DeviceType::Metal(0);
        }
        if !coreml_available && !metal_available {
            // mac-hybrid feature should not fall back to CPU!
            // This is a configuration error - user enabled mac-hybrid but no hardware available
            panic!("mac-hybrid feature enabled but neither CoreML nor Metal available. Check system configuration.");
        }

        // Both available: intelligent selection based on operation type
        match op_type {
            // CoreML preferred for optimized ML operations
            OpType::Convolution | OpType::Activation if tensor_size.unwrap_or(0) > 1000 => {
                DeviceType::CoreML(0)
            }
            // Metal preferred for custom computations and large matrices
            OpType::LinearAlgebra if tensor_size.unwrap_or(0) > 10000 => DeviceType::Metal(0),
            // CoreML unsupported operations → Metal
            OpType::ComplexMath
            | OpType::Distribution
            | OpType::CustomKernel
            | OpType::DistributedOps => DeviceType::Metal(0),
            // Default: prefer CoreML for power efficiency
            _ => DeviceType::CoreML(0),
        }
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        // Auto-select best available device
        #[cfg(feature = "coreml")]
        if crate::backends::DeviceManager::is_coreml_available() {
            return DeviceType::CoreML(0);
        }

        #[cfg(feature = "cuda")]
        if crate::backends::DeviceManager::is_cuda_available() {
            return DeviceType::Cuda(0);
        }

        #[cfg(feature = "metal")]
        {
            use crate::gpu::metal_kernels::MetalKernelExecutor;
            if MetalKernelExecutor::get().is_ok() {
                return DeviceType::Metal(0);
            }
        }

        DeviceType::Cpu
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda(id) => write!(f, "cuda:{}", id),
            DeviceType::Metal(id) => write!(f, "metal:{}", id),
            DeviceType::OpenCL(id) => write!(f, "opencl:{}", id),
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(id) => write!(f, "coreml:{}", id),
            #[cfg(feature = "coreml-hybrid")]
            DeviceType::CoreMLHybrid {
                coreml_id,
                fallback_gpu,
            } => {
                write!(f, "coreml_hybrid:{}:{:?}", coreml_id, fallback_gpu)
            }
            #[cfg(feature = "mac-hybrid")]
            DeviceType::MacHybrid => write!(f, "mac_hybrid"),
            DeviceType::Auto => write!(f, "auto"),
        }
    }
}

impl DeviceType {
    /// Check if the device is available
    /// デバイスが利用可能かチェック
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::Cpu => true,
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use crate::gpu::cuda_kernels::CudaKernelExecutor;
                    CudaKernelExecutor::new(0).is_ok()
                }
                #[cfg(not(feature = "cuda"))]
                false
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    use crate::gpu::metal_kernels::MetalKernelExecutor;
                    MetalKernelExecutor::get().is_ok()
                }
                #[cfg(not(feature = "metal"))]
                false
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl_kernels::OpenClKernelExecutor;
                    OpenClKernelExecutor::new(0).is_ok()
                }
                #[cfg(not(feature = "opencl"))]
                false
            }
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(_) => {
                // Check if CoreML is available on this platform
                cfg!(target_os = "macos")
            }
            #[cfg(feature = "coreml-hybrid")]
            DeviceType::CoreMLHybrid { fallback_gpu, .. } => {
                // Check if CoreML is available, or if fallback GPU is available
                cfg!(target_os = "macos")
                    || fallback_gpu.map_or(false, |gpu| match gpu {
                        GpuDevice::Cuda(id) => DeviceType::Cuda(id).is_available(),
                        GpuDevice::Metal(id) => DeviceType::Metal(id).is_available(),
                        GpuDevice::OpenCL(id) => DeviceType::OpenCL(id).is_available(),
                    })
            }
            #[cfg(feature = "mac-hybrid")]
            DeviceType::MacHybrid => {
                // MacHybrid is available if either Metal or CoreML is available
                cfg!(target_os = "macos")
                    && (DeviceType::Metal(0).is_available() || DeviceType::CoreML(0).is_available())
            }
            DeviceType::Auto => true, // Auto always "available" - selects best device
        }
    }
}

/// GPU memory layout
/// GPUメモリレイアウト
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    /// 行優先（C形式）レイアウト
    RowMajor,
    /// Column-major (Fortran-style) layout
    /// 列優先（Fortran形式）レイアウト
    ColMajor,
}

/// GPU operation context
/// GPU演算コンテキスト
pub struct GpuContext {
    device: DeviceType,
    memory_pool_size: usize,
    stream_count: usize,
}

impl GpuContext {
    /// Create a new GPU context
    /// 新しいGPUコンテキストを作成
    pub fn new(device: DeviceType) -> crate::error::RusTorchResult<Self> {
        match device {
            DeviceType::Cpu => Ok(GpuContext {
                device,
                memory_pool_size: 0,
                stream_count: 1,
            }),
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 1024 * 1024 * 1024, // 1GB default
                        stream_count: 4,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(crate::error::RusTorchError::gpu(
                        "CUDA support not compiled",
                    ))
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 512 * 1024 * 1024, // 512MB default
                        stream_count: 2,
                    })
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(crate::error::RusTorchError::gpu(
                        "Metal support not compiled",
                    ))
                }
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 256 * 1024 * 1024, // 256MB default
                        stream_count: 2,
                    })
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(crate::error::RusTorchError::gpu(
                        "OpenCL support not compiled",
                    ))
                }
            }
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(_) => {
                Ok(GpuContext {
                    device,
                    memory_pool_size: 1024 * 1024 * 1024, // 1GB default
                    stream_count: 1,
                })
            }
            #[cfg(feature = "coreml-hybrid")]
            DeviceType::CoreMLHybrid { .. } => {
                Ok(GpuContext {
                    device,
                    memory_pool_size: 1024 * 1024 * 1024, // 1GB default
                    stream_count: 2,
                })
            }
            #[cfg(feature = "mac-hybrid")]
            DeviceType::MacHybrid => {
                Ok(GpuContext {
                    device,
                    memory_pool_size: 1024 * 1024 * 1024, // 1GB default for hybrid
                    stream_count: 4, // Support multiple streams for hybrid operations
                })
            }
            DeviceType::Auto => {
                // Auto-select best available device
                let best_device = DeviceType::default();
                Self::new(best_device)
            }
        }
    }

    /// Get the device type
    /// デバイスタイプを取得
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Check if GPU is available
    /// GPUが利用可能かチェック
    pub fn is_gpu_available(&self) -> bool {
        !matches!(self.device, DeviceType::Cpu)
    }

    /// Get memory pool size
    /// メモリプールサイズを取得
    pub fn memory_pool_size(&self) -> usize {
        self.memory_pool_size
    }

    /// Get number of streams
    /// ストリーム数を取得
    pub fn stream_count(&self) -> usize {
        self.stream_count
    }
}

// GPU error types
// GPUエラータイプ
// GpuError enum removed - now using unified RusTorchError system
// GpuErrorエナム削除 - 統一RusTorchErrorシステムを使用

// Result type for GPU operations (now unified) - using global RusTorchResult
// GPU演算の結果型（統一済み）- グローバルRusTorchResultを使用

// Re-export simplified unified kernel system components
// 簡潔な統一カーネルシステムコンポーネントを再エクスポート
pub use unified_kernel_simple::{
    KernelMetrics, KernelOp, KernelParams, KernelSelector, UnifiedKernelExecutor,
};

/// GPU device manager
/// GPUデバイスマネージャー
pub struct DeviceManager {
    contexts: Vec<GpuContext>,
    current_device: usize,
}

impl DeviceManager {
    /// Create a new device manager
    /// 新しいデバイスマネージャーを作成
    pub fn new() -> Self {
        let mut contexts = Vec::new();

        // Always add CPU context
        if let Ok(cpu_context) = GpuContext::new(DeviceType::Cpu) {
            contexts.push(cpu_context);
        }

        // Try to add GPU contexts
        #[cfg(feature = "cuda")]
        {
            for device_id in 0..Self::get_cuda_device_count() {
                if let Ok(cuda_context) = GpuContext::new(DeviceType::Cuda(device_id)) {
                    contexts.push(cuda_context);
                }
            }
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(metal_context) = GpuContext::new(DeviceType::Metal(0)) {
                contexts.push(metal_context);
            }
        }

        DeviceManager {
            contexts,
            current_device: 0,
        }
    }

    /// Get available devices
    /// 利用可能なデバイスを取得
    pub fn available_devices(&self) -> Vec<DeviceType> {
        self.contexts.iter().map(|ctx| ctx.device()).collect()
    }

    /// Set current device
    /// 現在のデバイスを設定
    pub fn set_device(&mut self, device: DeviceType) -> crate::error::RusTorchResult<()> {
        if let Some(index) = self.contexts.iter().position(|ctx| ctx.device() == device) {
            self.current_device = index;
            Ok(())
        } else {
            Err(crate::error::RusTorchError::device_not_available(
                device.to_string(),
            ))
        }
    }

    /// Get current device
    /// 現在のデバイスを取得
    pub fn current_device(&self) -> DeviceType {
        self.contexts[self.current_device].device()
    }

    /// Get current context
    /// 現在のコンテキストを取得
    pub fn current_context(&self) -> &GpuContext {
        &self.contexts[self.current_device]
    }

    /// Check if CUDA is available
    /// CUDAが利用可能かチェック
    pub fn is_cuda_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            Self::get_cuda_device_count() > 0
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Check if Metal is available
    /// Metalが利用可能かチェック
    pub fn is_metal_available() -> bool {
        #[cfg(feature = "metal")]
        {
            use crate::gpu::metal_kernels::MetalKernelExecutor;
            cfg!(target_os = "macos") && MetalKernelExecutor::get().is_ok()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> usize {
        use crate::gpu::cuda_kernels::CudaKernelExecutor;
        // Try to create CUDA executors for devices 0-7 and count successful ones
        (0..8)
            .filter(|&i| CudaKernelExecutor::new(i).is_ok())
            .count()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global device manager instance
/// グローバルデバイスマネージャーインスタンス
static mut DEVICE_MANAGER: Option<DeviceManager> = None;
static DEVICE_MANAGER_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global device manager
/// グローバルデバイスマネージャーを取得
pub fn get_device_manager() -> &'static mut DeviceManager {
    unsafe {
        DEVICE_MANAGER_INIT.call_once(|| {
            DEVICE_MANAGER = Some(DeviceManager::new());
        });
        #[allow(static_mut_refs)]
        {
            DEVICE_MANAGER.as_mut().unwrap()
        }
    }
}

/// Set the current device globally
/// グローバルに現在のデバイスを設定
pub fn set_device(device: DeviceType) -> crate::error::RusTorchResult<()> {
    get_device_manager().set_device(device)
}

/// Get the current device
/// 現在のデバイスを取得
pub fn current_device() -> DeviceType {
    get_device_manager().current_device()
}

/// Check if any GPU is available
/// 何らかのGPUが利用可能かチェック
pub fn is_gpu_available() -> bool {
    DeviceManager::is_cuda_available() || DeviceManager::is_metal_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Cuda(0).to_string(), "cuda:0");
        assert_eq!(DeviceType::Metal(1).to_string(), "metal:1");
        assert_eq!(DeviceType::OpenCL(2).to_string(), "opencl:2");
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(!manager.available_devices().is_empty());
        assert_eq!(manager.current_device(), DeviceType::Cpu);
    }

    #[test]
    fn test_gpu_context_cpu() {
        let context = GpuContext::new(DeviceType::Cpu).unwrap();
        assert_eq!(context.device(), DeviceType::Cpu);
        assert!(!context.is_gpu_available());
        assert_eq!(context.stream_count(), 1);
    }

    #[test]
    fn test_global_device_manager() {
        let device = current_device();
        assert_eq!(device, DeviceType::Cpu);

        let available = get_device_manager().available_devices();
        assert!(!available.is_empty());
    }

    #[test]
    fn test_gpu_availability() {
        // This will depend on the compilation features
        let _cuda_available = DeviceManager::is_cuda_available();
        let _metal_available = DeviceManager::is_metal_available();
        let _any_gpu = is_gpu_available();
    }
}
