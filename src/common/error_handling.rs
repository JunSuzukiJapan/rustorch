//! Unified error handling for RusTorch
//! RusTorchの統一エラーハンドリング

use std::fmt;

/// Common result type for RusTorch operations
/// RusTorch操作の共通結果型
pub type RusTorchResult<T> = Result<T, RusTorchError>;

/// Unified error type for all RusTorch operations
/// 全RusTorch操作の統一エラー型
#[derive(Debug)]
pub enum RusTorchError {
    /// Tensor operation errors
    /// テンソル操作エラー
    TensorError(TensorError),
    
    /// GPU operation errors
    /// GPU操作エラー
    GpuError(GpuError),
    
    /// Distributed operation errors
    /// 分散操作エラー
    DistributedError(DistributedError),
    
    /// Neural network operation errors
    /// ニューラルネットワーク操作エラー
    NeuralNetworkError(NeuralNetworkError),
    
    /// Optimization errors
    /// 最適化エラー
    OptimizationError(OptimizationError),
    
    /// Data loading errors
    /// データローディングエラー
    DataError(DataError),
    
    /// Memory management errors
    /// メモリ管理エラー
    MemoryError(MemoryError),
    
    /// I/O errors
    /// I/Oエラー
    IoError(std::io::Error),
    
    /// Generic errors
    /// 汎用エラー
    Generic(String),
}

/// Tensor-specific errors
/// テンソル固有エラー
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Tensor shape mismatch error
    /// テンソル形状不一致エラー
    ShapeMismatch { 
        /// Expected shape
        /// 期待される形状
        expected: Vec<usize>, 
        /// Actual shape
        /// 実際の形状
        actual: Vec<usize> 
    },
    /// Tensor dimension mismatch error
    /// テンソル次元不一致エラー
    DimensionMismatch { 
        /// Left tensor shape
        /// 左テンソルの形状
        lhs: Vec<usize>, 
        /// Right tensor shape
        /// 右テンソルの形状
        rhs: Vec<usize> 
    },
    /// Insufficient tensor dimensions error
    /// テンソル次元不足エラー
    InsufficientDimensions { 
        /// Required dimensions
        /// 必要な次元数
        required: usize, 
        /// Actual dimensions
        /// 実際の次元数
        actual: usize 
    },
    /// Invalid tensor shape
    /// 無効なテンソル形状
    InvalidShape(Vec<usize>),
    /// Invalid tensor index
    /// 無効なテンソルインデックス
    InvalidIndex(Vec<usize>),
    /// Invalid tensor operation
    /// 無効なテンソル操作
    InvalidOperation(String),
    /// Empty tensor error
    /// 空のテンソルエラー
    EmptyTensor,
    /// Data type error
    /// データ型エラー
    DataTypeError(String),
}

/// GPU-specific errors
/// GPU固有エラー
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU device not found
    /// GPUデバイスが見つからない
    DeviceNotFound(usize),
    /// GPU device not supported
    /// GPUデバイスがサポートされていない
    DeviceNotSupported(String),
    /// GPU memory allocation failed
    /// GPUメモリ割り当て失敗
    MemoryAllocationFailed(usize),
    /// GPU memory transfer failed
    /// GPUメモリ転送失敗
    MemoryTransferFailed(String),
    /// GPU kernel compilation failed
    /// GPUカーネルコンパイル失敗
    KernelCompilationFailed(String),
    /// GPU kernel execution failed
    /// GPUカーネル実行失敗
    KernelExecutionFailed(String),
    /// GPU context creation failed
    /// GPUコンテキスト作成失敗
    ContextCreationFailed(String),
    /// Invalid GPU device
    /// 無効なGPUデバイス
    InvalidDevice(String),
    /// GPU out of memory
    /// GPUメモリ不足
    OutOfMemory,
    /// GPU driver error
    /// GPUドライバーエラー
    DriverError(String),
}

/// Distributed computing errors
/// 分散コンピューティングエラー
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// Distributed backend not supported
    /// 分散バックエンドがサポートされていない
    BackendNotSupported(String),
    /// Communication failed
    /// 通信失敗
    CommunicationFailed(String),
    /// Process group error
    /// プロセスグループエラー
    ProcessGroupError(String),
    /// Synchronization failed
    /// 同期失敗
    SynchronizationFailed(String),
    /// Node connection failed
    /// ノード接続失敗
    NodeConnectionFailed(String),
    /// Invalid rank
    /// 無効なランク
    InvalidRank(i32),
    /// Invalid world size
    /// 無効なワールドサイズ
    InvalidWorldSize(i32),
    /// Timeout error
    /// タイムアウトエラー
    TimeoutError(String),
    /// Network error
    /// ネットワークエラー
    NetworkError(String),
}

/// Neural network errors
/// ニューラルネットワークエラー
#[derive(Debug, Clone)]
pub enum NeuralNetworkError {
    /// Neural network layer error
    /// ニューラルネットワークレイヤーエラー
    LayerError(String),
    /// Activation function error
    /// 活性化関数エラー
    ActivationError(String),
    /// Loss function error
    /// 損失関数エラー
    LossError(String),
    /// Forward pass error
    /// 順伝播エラー
    ForwardPassError(String),
    /// Backward pass error
    /// 逆伝播エラー
    BackwardPassError(String),
    /// Parameter error
    /// パラメータエラー
    ParameterError(String),
    /// Model error
    /// モデルエラー
    ModelError(String),
}

/// Optimization errors
/// 最適化エラー
#[derive(Debug, Clone)]
pub enum OptimizationError {
    /// Optimizer error
    /// オプティマイザーエラー
    OptimizerError(String),
    /// Learning rate scheduler error
    /// 学習率スケジューラーエラー
    SchedulerError(String),
    /// Gradient calculation error
    /// 勾配計算エラー
    GradientError(String),
    /// Convergence error
    /// 収束エラー
    ConvergenceError(String),
    /// Learning rate error
    /// 学習率エラー
    LearningRateError(String),
}

/// Data loading errors
/// データローディングエラー
#[derive(Debug, Clone)]
pub enum DataError {
    /// Dataset error
    /// データセットエラー
    DatasetError(String),
    /// Data loader error
    /// データローダーエラー
    DataLoaderError(String),
    /// Batch processing error
    /// バッチ処理エラー
    BatchError(String),
    /// Data transformation error
    /// データ変換エラー
    TransformError(String),
    /// File operation error
    /// ファイル操作エラー
    FileError(String),
}

/// Memory management errors
/// メモリ管理エラー
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Memory allocation failed
    /// メモリ割り当て失敗
    AllocationFailed(usize),
    /// Memory deallocation failed
    /// メモリ解放失敗
    DeallocationFailed(String),
    /// Memory alignment error
    /// メモリアライメントエラー
    AlignmentError(usize),
    /// Memory pool exhausted
    /// メモリプール枯渇
    PoolExhausted,
    /// Invalid memory pointer
    /// 無効なメモリポインタ
    InvalidPointer,
    /// Memory leak detected
    /// メモリリーク検出
    MemoryLeak(String),
}

impl fmt::Display for RusTorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RusTorchError::TensorError(e) => write!(f, "Tensor error: {}", e),
            RusTorchError::GpuError(e) => write!(f, "GPU error: {}", e),
            RusTorchError::DistributedError(e) => write!(f, "Distributed error: {}", e),
            RusTorchError::NeuralNetworkError(e) => write!(f, "Neural network error: {}", e),
            RusTorchError::OptimizationError(e) => write!(f, "Optimization error: {}", e),
            RusTorchError::DataError(e) => write!(f, "Data error: {}", e),
            RusTorchError::MemoryError(e) => write!(f, "Memory error: {}", e),
            RusTorchError::IoError(e) => write!(f, "I/O error: {}", e),
            RusTorchError::Generic(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, actual)
            }
            TensorError::DimensionMismatch { lhs, rhs } => {
                write!(f, "Dimension mismatch: {:?} vs {:?}", lhs, rhs)
            }
            TensorError::InsufficientDimensions { required, actual } => {
                write!(f, "Insufficient dimensions: required {}, got {}", required, actual)
            }
            TensorError::InvalidShape(shape) => write!(f, "Invalid shape: {:?}", shape),
            TensorError::InvalidIndex(index) => write!(f, "Invalid index: {:?}", index),
            TensorError::InvalidOperation(op) => write!(f, "Invalid operation: {}", op),
            TensorError::EmptyTensor => write!(f, "Operation on empty tensor"),
            TensorError::DataTypeError(msg) => write!(f, "Data type error: {}", msg),
        }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceNotFound(id) => write!(f, "GPU device {} not found", id),
            GpuError::DeviceNotSupported(device) => write!(f, "GPU device not supported: {}", device),
            GpuError::MemoryAllocationFailed(size) => write!(f, "GPU memory allocation failed: {} bytes", size),
            GpuError::MemoryTransferFailed(msg) => write!(f, "GPU memory transfer failed: {}", msg),
            GpuError::KernelCompilationFailed(msg) => write!(f, "GPU kernel compilation failed: {}", msg),
            GpuError::KernelExecutionFailed(msg) => write!(f, "GPU kernel execution failed: {}", msg),
            GpuError::ContextCreationFailed(msg) => write!(f, "GPU context creation failed: {}", msg),
            GpuError::InvalidDevice(device) => write!(f, "Invalid GPU device: {}", device),
            GpuError::OutOfMemory => write!(f, "GPU out of memory"),
            GpuError::DriverError(msg) => write!(f, "GPU driver error: {}", msg),
        }
    }
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributedError::BackendNotSupported(backend) => {
                write!(f, "Distributed backend not supported: {}", backend)
            }
            DistributedError::CommunicationFailed(msg) => {
                write!(f, "Distributed communication failed: {}", msg)
            }
            DistributedError::ProcessGroupError(msg) => {
                write!(f, "Process group error: {}", msg)
            }
            DistributedError::SynchronizationFailed(msg) => {
                write!(f, "Synchronization failed: {}", msg)
            }
            DistributedError::NodeConnectionFailed(msg) => {
                write!(f, "Node connection failed: {}", msg)
            }
            DistributedError::InvalidRank(rank) => write!(f, "Invalid rank: {}", rank),
            DistributedError::InvalidWorldSize(size) => write!(f, "Invalid world size: {}", size),
            DistributedError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            DistributedError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::LayerError(msg) => write!(f, "Layer error: {}", msg),
            NeuralNetworkError::ActivationError(msg) => write!(f, "Activation error: {}", msg),
            NeuralNetworkError::LossError(msg) => write!(f, "Loss error: {}", msg),
            NeuralNetworkError::ForwardPassError(msg) => write!(f, "Forward pass error: {}", msg),
            NeuralNetworkError::BackwardPassError(msg) => write!(f, "Backward pass error: {}", msg),
            NeuralNetworkError::ParameterError(msg) => write!(f, "Parameter error: {}", msg),
            NeuralNetworkError::ModelError(msg) => write!(f, "Model error: {}", msg),
        }
    }
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationError::OptimizerError(msg) => write!(f, "Optimizer error: {}", msg),
            OptimizationError::SchedulerError(msg) => write!(f, "Scheduler error: {}", msg),
            OptimizationError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
            OptimizationError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            OptimizationError::LearningRateError(msg) => write!(f, "Learning rate error: {}", msg),
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::DatasetError(msg) => write!(f, "Dataset error: {}", msg),
            DataError::DataLoaderError(msg) => write!(f, "DataLoader error: {}", msg),
            DataError::BatchError(msg) => write!(f, "Batch error: {}", msg),
            DataError::TransformError(msg) => write!(f, "Transform error: {}", msg),
            DataError::FileError(msg) => write!(f, "File error: {}", msg),
        }
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::AllocationFailed(size) => write!(f, "Memory allocation failed: {} bytes", size),
            MemoryError::DeallocationFailed(msg) => write!(f, "Memory deallocation failed: {}", msg),
            MemoryError::AlignmentError(alignment) => write!(f, "Memory alignment error: {} bytes", alignment),
            MemoryError::PoolExhausted => write!(f, "Memory pool exhausted"),
            MemoryError::InvalidPointer => write!(f, "Invalid memory pointer"),
            MemoryError::MemoryLeak(msg) => write!(f, "Memory leak detected: {}", msg),
        }
    }
}

impl std::error::Error for RusTorchError {}
impl std::error::Error for TensorError {}
impl std::error::Error for GpuError {}
impl std::error::Error for DistributedError {}
impl std::error::Error for NeuralNetworkError {}
impl std::error::Error for OptimizationError {}
impl std::error::Error for DataError {}
impl std::error::Error for MemoryError {}

// Conversion implementations for backward compatibility
impl From<std::io::Error> for RusTorchError {
    fn from(err: std::io::Error) -> Self {
        RusTorchError::IoError(err)
    }
}

impl From<TensorError> for RusTorchError {
    fn from(err: TensorError) -> Self {
        RusTorchError::TensorError(err)
    }
}

impl From<GpuError> for RusTorchError {
    fn from(err: GpuError) -> Self {
        RusTorchError::GpuError(err)
    }
}

impl From<DistributedError> for RusTorchError {
    fn from(err: DistributedError) -> Self {
        RusTorchError::DistributedError(err)
    }
}

impl From<NeuralNetworkError> for RusTorchError {
    fn from(err: NeuralNetworkError) -> Self {
        RusTorchError::NeuralNetworkError(err)
    }
}

impl From<OptimizationError> for RusTorchError {
    fn from(err: OptimizationError) -> Self {
        RusTorchError::OptimizationError(err)
    }
}

impl From<DataError> for RusTorchError {
    fn from(err: DataError) -> Self {
        RusTorchError::DataError(err)
    }
}

impl From<MemoryError> for RusTorchError {
    fn from(err: MemoryError) -> Self {
        RusTorchError::MemoryError(err)
    }
}

impl From<crate::tensor::parallel_errors::ParallelError> for RusTorchError {
    fn from(err: crate::tensor::parallel_errors::ParallelError) -> Self {
        RusTorchError::TensorError(TensorError::InvalidOperation(format!("{:?}", err)))
    }
}

/// Helper macros for error creation
/// エラー作成用ヘルパーマクロ
/// Macro for creating tensor errors easily
/// テンソルエラーを簡単に作成するためのマクロ
#[macro_export]
macro_rules! tensor_error {
    ($variant:ident) => {
        RusTorchError::TensorError(TensorError::$variant)
    };
    ($variant:ident, $($arg:expr),+) => {
        RusTorchError::TensorError(TensorError::$variant { $($arg),+ })
    };
}

/// Macro for creating GPU errors easily
/// GPUエラーを簡単に作成するためのマクロ
#[macro_export]
macro_rules! gpu_error {
    ($variant:ident) => {
        RusTorchError::GpuError(GpuError::$variant)
    };
    ($variant:ident, $arg:expr) => {
        RusTorchError::GpuError(GpuError::$variant($arg))
    };
}

/// Macro for creating distributed errors easily
/// 分散エラーを簡単に作成するためのマクロ
#[macro_export]
macro_rules! distributed_error {
    ($variant:ident, $arg:expr) => {
        RusTorchError::DistributedError(DistributedError::$variant($arg))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let tensor_err = RusTorchError::TensorError(TensorError::EmptyTensor);
        assert!(tensor_err.to_string().contains("empty tensor"));

        let gpu_err = RusTorchError::GpuError(GpuError::OutOfMemory);
        assert!(gpu_err.to_string().contains("out of memory"));
    }

    #[test]
    fn test_error_conversion() {
        let tensor_err = TensorError::EmptyTensor;
        let rustorch_err: RusTorchError = tensor_err.into();
        matches!(rustorch_err, RusTorchError::TensorError(_));
    }

    #[test]
    fn test_error_macros() {
        let err = tensor_error!(EmptyTensor);
        matches!(err, RusTorchError::TensorError(TensorError::EmptyTensor));

        let err = gpu_error!(OutOfMemory);
        matches!(err, RusTorchError::GpuError(GpuError::OutOfMemory));
    }
}
