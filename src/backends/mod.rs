//! Unified compute backend abstraction for RusTorch v0.4.0
//! RusTorch v0.4.0の統一計算バックエンド抽象化
//!
//! This module provides a unified interface for different compute backends,
//! enabling seamless switching between CPU, CUDA, Metal, and OpenCL implementations.
//!
//! このモジュールは異なる計算バックエンドの統一インターフェースを提供し、
//! CPU、CUDA、Metal、OpenCLの実装間でのシームレスな切り替えを可能にします。
//!
//! ## Unified Backend Architecture / 統一バックエンドアーキテクチャ
//!
//! The new architecture provides:
//! 新しいアーキテクチャは以下を提供:
//! - Unified ComputeBackend trait for all device types
//!   全デバイス種別用の統一ComputeBackendトレイト
//! - Automatic device selection with performance tracking
//!   パフォーマンス追跡付き自動デバイス選択
//! - Operation abstraction with type-safe execution
//!   型安全実行付き操作抽象化

// New unified backend system
pub mod compute_backend;
pub mod cpu_unified;

// Legacy backend modules (temporarily disabled during refactoring)
// #[allow(dead_code)]
// pub mod cpu;

// Re-export unified types with new names to avoid conflicts
pub use compute_backend::{
    global_device_manager, initialize_backends, ComputeBackend as UnifiedComputeBackend,
    DeviceManager, DeviceType as UnifiedDeviceType, Operation, PerformanceMetrics,
    SelectionStrategy, TransferDirection,
};
pub use cpu_unified::CpuBackend as UnifiedCpuBackend;

use crate::tensor::Tensor;
use std::any::Any;
use std::fmt;

/// Result type for backend operations (now unified)
/// バックエンド操作用の結果型（統一済み）
pub type BackendResult<T> = crate::error::RusTorchResult<T>;

/// Device memory buffer abstraction
/// デバイスメモリバッファ抽象化
pub trait DeviceBuffer: Send + Sync {
    /// Size of the buffer in bytes
    /// バッファのバイトサイズ
    fn size(&self) -> usize;

    /// Copy data from host to device
    /// ホストからデバイスへデータをコピー
    fn copy_from_host(&mut self, data: &[u8]) -> BackendResult<()>;

    /// Copy data from device to host
    /// デバイスからホストへデータをコピー
    fn copy_to_host(&self, data: &mut [u8]) -> BackendResult<()>;

    /// Get raw pointer for device operations (unsafe)
    /// デバイス操作用の生ポインタを取得（unsafe）
    fn as_ptr(&self) -> *mut u8;
}

/// Device information and capabilities
/// デバイス情報と機能
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    /// デバイス名
    pub name: String,
    /// Device type
    /// デバイスタイプ
    pub device_type: DeviceType,
    /// Total memory in bytes
    /// 総メモリ（バイト）
    pub total_memory: usize,
    /// Available memory in bytes
    /// 利用可能メモリ（バイト）
    pub available_memory: usize,
    /// Maximum number of threads/cores
    /// 最大スレッド/コア数
    pub max_threads: usize,
    /// Supports double precision
    /// 倍精度サポート
    pub supports_f64: bool,
    /// Supports half precision
    /// 半精度サポート
    pub supports_f16: bool,
}

/// Types of compute devices
/// 計算デバイスの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU computation
    /// CPU計算
    Cpu,
    /// NVIDIA CUDA GPU
    /// NVIDIA CUDA GPU
    Cuda,
    /// Apple Metal GPU  
    /// Apple Metal GPU
    Metal,
    /// OpenCL compatible device
    /// OpenCL互換デバイス
    OpenCL,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Cuda => write!(f, "CUDA"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::OpenCL => write!(f, "OpenCL"),
        }
    }
}

/// Parameters for convolution operations
/// 畳み込み演算のパラメータ
#[derive(Debug, Clone)]
pub struct ConvolutionParams {
    /// Kernel size [height, width] or [depth, height, width]
    /// カーネルサイズ [height, width] または [depth, height, width]
    pub kernel_size: Vec<usize>,
    /// Stride [height, width] or [depth, height, width]
    /// ストライド [height, width] または [depth, height, width]
    pub stride: Vec<usize>,
    /// Padding [height, width] or [depth, height, width]
    /// パディング [height, width] または [depth, height, width]
    pub padding: Vec<usize>,
    /// Dilation [height, width] or [depth, height, width]
    /// 膨張 [height, width] または [depth, height, width]
    pub dilation: Vec<usize>,
    /// Number of groups for grouped convolution
    /// グループ化畳み込みのグループ数
    pub groups: usize,
}

impl Default for ConvolutionParams {
    fn default() -> Self {
        Self {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        }
    }
}

/// Core compute backend trait that all backends must implement
/// 全バックエンドが実装すべきコア計算バックエンドトレイト
pub trait ComputeBackend: Send + Sync {
    /// Get device information
    /// デバイス情報を取得
    fn device_info(&self) -> &DeviceInfo;

    /// Allocate memory on the device
    /// デバイスでメモリを割り当て
    fn allocate_memory(&self, size: usize) -> BackendResult<Box<dyn DeviceBuffer>>;

    /// Synchronize device execution (wait for all operations to complete)
    /// デバイス実行を同期（全操作の完了を待機）
    fn synchronize(&self) -> BackendResult<()>;

    /// Check if this backend is available on the current system
    /// このバックエンドが現在のシステムで利用可能かチェック
    fn is_available() -> bool
    where
        Self: Sized;

    // === Core Tensor Operations ===
    // === コアテンソル操作 ===

    /// Element-wise addition: a + b
    /// 要素ごと加算: a + b
    fn add<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Element-wise subtraction: a - b
    /// 要素ごと減算: a - b
    fn sub<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Element-wise multiplication: a * b
    /// 要素ごと乗算: a * b
    fn mul<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Element-wise division: a / b
    /// 要素ごと除算: a / b
    fn div<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Matrix multiplication: a @ b
    /// 行列乗算: a @ b
    fn matmul<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    // === Reduction Operations ===
    // === リダクション操作 ===

    /// Sum all elements
    /// 全要素の合計
    fn sum<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Mean of all elements
    /// 全要素の平均
    fn mean<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Maximum element
    /// 最大要素
    fn max<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Minimum element
    /// 最小要素
    fn min<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    // === Shape Operations ===
    // === 形状操作 ===

    /// Reshape tensor to new shape
    /// テンソルを新しい形状に変形
    fn reshape<T>(&self, tensor: &Tensor<T>, new_shape: &[usize]) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Transpose tensor along specified axes
    /// 指定された軸でテンソルを転置
    fn transpose<T>(&self, tensor: &Tensor<T>, axes: &[usize]) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    // === Advanced Operations ===
    // === 高度な操作 ===

    /// Convolution operation
    /// 畳み込み操作
    fn convolution<T>(
        &self,
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        params: &ConvolutionParams,
    ) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Batch normalization
    /// バッチ正規化
    fn batch_norm<T>(
        &self,
        input: &Tensor<T>,
        weight: &Tensor<T>,
        bias: &Tensor<T>,
        running_mean: &Tensor<T>,
        running_var: &Tensor<T>,
        training: bool,
        momentum: T,
        eps: T,
    ) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Activation functions
    /// 活性化関数
    fn relu<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Apply sigmoid activation function
    /// シグモイド活性化関数を適用
    fn sigmoid<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    /// Apply hyperbolic tangent activation function
    /// ハイパボリックタンジェント活性化関数を適用
    fn tanh<T>(&self, tensor: &Tensor<T>) -> BackendResult<Tensor<T>>
    where
        T: num_traits::Float + 'static;

    // === Backend-specific Operations ===
    // === バックエンド固有の操作 ===

    /// Get backend-specific context for advanced operations
    /// 高度な操作のためのバックエンド固有のコンテキストを取得
    fn backend_context(&self) -> &dyn Any;

    /// Execute custom kernel/operation (backend-specific)
    /// カスタムカーネル/操作を実行（バックエンド固有）
    fn execute_custom_op(
        &self,
        op_name: &str,
        inputs: &[&dyn Any],
        params: &dyn Any,
    ) -> BackendResult<Box<dyn Any>>;
}

/// Backend factory for creating and managing compute backends
/// 計算バックエンドの作成と管理のためのバックエンドファクトリ
pub struct BackendFactory;

impl BackendFactory {
    /// Get all available backends on the current system
    /// 現在のシステムで利用可能な全バックエンドを取得
    pub fn available_backends() -> Vec<DeviceType> {
        // CPU is always available
        let backends = vec![DeviceType::Cpu];

        // Additional backends would be checked here when implemented
        // #[cfg(feature = "cuda")]
        // if cuda::CudaBackend::is_available() {
        //     backends.push(DeviceType::Cuda);
        // }

        // #[cfg(feature = "metal")]
        // if metal::MetalBackend::is_available() {
        //     backends.push(DeviceType::Metal);
        // }

        // #[cfg(feature = "opencl")]
        // if opencl::OpenClBackend::is_available() {
        //     backends.push(DeviceType::OpenCL);
        // }

        backends
    }

    /// Create the best available CPU backend for the current system
    /// 現在のシステムで最適な利用可能CPUバックエンドを作成
    pub fn create_cpu_backend() -> BackendResult<cpu_unified::CpuBackend> {
        cpu_unified::CpuBackend::new()
    }
}

// Legacy re-exports temporarily disabled
// pub use cpu::CpuBackend;

// #[cfg(feature = "cuda")]
// pub use cuda::CudaBackend; // TODO: Implement CudaBackend

// #[cfg(feature = "metal")]
// pub use metal::MetalBackend; // TODO: Implement MetalBackend

// OpenCL backend not yet implemented
// #[cfg(feature = "opencl")]
// pub use opencl::OpenClBackend;
