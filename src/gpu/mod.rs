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

/// Device management module for GPU operations
/// GPU操作のためのデバイス管理モジュール
pub mod device;
/// GPU memory management and allocation
/// GPUメモリ管理とアロケーション
pub mod memory;
/// GPU kernel execution and management
/// GPUカーネル実行と管理
pub mod kernels;
/// CUDA kernel implementations
/// CUDAカーネル実装
pub mod cuda_kernels;
pub mod metal_kernels;
pub mod opencl_kernels;
pub mod custom_kernels;
/// GPU kernel validation and testing
/// GPUカーネル検証とテスト
pub mod validation;
/// Unified kernel interface for cross-platform GPU acceleration (simplified)
/// クロスプラットフォームGPU加速のための統一カーネルインターフェース（簡潔版）
pub mod unified_kernel_simple;
/// Integration tests for unified GPU kernel system
/// 統一GPUカーネルシステムの統合テスト
#[cfg(test)]
pub mod integration_tests;

use std::fmt;

/// GPU device types
/// GPU デバイスタイプ
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
}

impl Default for DeviceType {
    fn default() -> Self {
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
            },
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    use crate::gpu::metal_kernels::MetalKernelExecutor;
                    MetalKernelExecutor::new().is_ok()
                }
                #[cfg(not(feature = "metal"))]
                false
            },
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl_kernels::OpenClKernelExecutor;
                    OpenClKernelExecutor::new(0).is_ok()
                }
                #[cfg(not(feature = "opencl"))]
                false
            },
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
    pub fn new(device: DeviceType) -> Result<Self, GpuError> {
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
                    Err(GpuError::UnsupportedDevice("CUDA support not compiled".to_string()))
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
                    Err(GpuError::UnsupportedDevice("Metal support not compiled".to_string()))
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
                    Err(GpuError::UnsupportedDevice("OpenCL support not compiled".to_string()))
                }
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

/// GPU error types
/// GPUエラータイプ
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU device not found
    /// GPUデバイスが見つからない
    DeviceNotFound(usize),
    /// GPU device not available
    /// GPUデバイスが利用できない
    DeviceNotAvailable(String),
    /// Unsupported GPU device
    /// サポートされていないGPUデバイス
    UnsupportedDevice(String),
    /// GPU memory allocation failed
    /// GPUメモリ割り当て失敗
    MemoryAllocationFailed(usize),
    /// GPU memory allocation error
    /// GPUメモリ割り当てエラー
    MemoryAllocationError(String),
    /// GPU kernel compilation failed
    /// GPUカーネルコンパイル失敗
    KernelCompilationFailed(String),
    /// GPU kernel execution failed
    /// GPUカーネル実行失敗
    KernelExecutionFailed(String),
    /// GPU kernel execution error
    /// GPUカーネル実行エラー
    KernelExecutionError(String),
    /// Invalid GPU device
    /// 無効なGPUデバイス
    InvalidDevice(String),
    /// GPU out of memory
    /// GPUメモリ不足
    OutOfMemory,
    /// GPU context creation failed
    /// GPUコンテキスト作成失敗
    ContextCreationFailed(String),
    /// GPU driver error
    /// GPUドライバーエラー
    DriverError(String),
    /// Unsupported GPU operation
    /// サポートされていないGPU操作
    UnsupportedOperation(String),
    /// GPU kernel error
    /// GPUカーネルエラー
    KernelError(String),
    /// Data transfer error
    /// データ転送エラー
    DataTransferError(String),
    /// Invalid operation
    /// 無効な操作
    InvalidOperation(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceNotFound(id) => write!(f, "GPU device {} not found", id),
            GpuError::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            GpuError::UnsupportedDevice(msg) => write!(f, "Unsupported device: {}", msg),
            GpuError::MemoryAllocationFailed(size) => write!(f, "GPU memory allocation failed: {} bytes", size),
            GpuError::MemoryAllocationError(msg) => write!(f, "Memory allocation error: {}", msg),
            GpuError::KernelCompilationFailed(msg) => write!(f, "GPU kernel compilation failed: {}", msg),
            GpuError::KernelExecutionFailed(msg) => write!(f, "GPU kernel execution failed: {}", msg),
            GpuError::KernelExecutionError(msg) => write!(f, "Kernel execution error: {}", msg),
            GpuError::InvalidDevice(device) => write!(f, "Invalid GPU device: {}", device),
            GpuError::OutOfMemory => write!(f, "GPU out of memory"),
            GpuError::ContextCreationFailed(msg) => write!(f, "GPU context creation failed: {}", msg),
            GpuError::DriverError(msg) => write!(f, "GPU driver error: {}", msg),
            GpuError::UnsupportedOperation(msg) => write!(f, "Unsupported GPU operation: {}", msg),
            GpuError::KernelError(msg) => write!(f, "GPU kernel error: {}", msg),
            GpuError::DataTransferError(msg) => write!(f, "Data transfer error: {}", msg),
            GpuError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// Result type for GPU operations (now unified)
/// GPU演算の結果型（統一済み）
pub type GpuResult<T> = crate::error::RusTorchResult<T>;

// Re-export simplified unified kernel system components
// 簡潔な統一カーネルシステムコンポーネントを再エクスポート
pub use unified_kernel_simple::{UnifiedKernelExecutor, KernelSelector, KernelOp, KernelParams, KernelMetrics};

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
    pub fn set_device(&mut self, device: DeviceType) -> Result<(), GpuError> {
        if let Some(index) = self.contexts.iter().position(|ctx| ctx.device() == device) {
            self.current_device = index;
            Ok(())
        } else {
            Err(GpuError::DeviceNotAvailable(format!("Device {} not found", device)))
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
            cfg!(target_os = "macos") && MetalKernelExecutor::new().is_ok()
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
        (0..8).filter(|&i| CudaKernelExecutor::new(i).is_ok()).count()
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
        { DEVICE_MANAGER.as_mut().unwrap() }
    }
}

/// Set the current device globally
/// グローバルに現在のデバイスを設定
pub fn set_device(device: DeviceType) -> Result<(), GpuError> {
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
