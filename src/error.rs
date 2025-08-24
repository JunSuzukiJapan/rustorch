//! Unified error handling system for RusTorch
//! RusTorch用統一エラーハンドリングシステム

use std::fmt;
use std::error::Error;

pub mod context;

/// Main error type for RusTorch operations
/// RusTorch操作のメインエラー型
#[derive(Debug, thiserror::Error)]
pub enum RusTorchError {
    /// Tensor operation failed
    /// テンソル操作失敗
    #[error("Tensor operation failed: {message}")]
    TensorOp { 
        message: String, 
        #[source] 
        source: Option<Box<dyn Error + Send + Sync>> 
    },
    
    /// Shape mismatch between tensors
    /// テンソル間の形状不一致
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { 
        expected: Vec<usize>, 
        actual: Vec<usize> 
    },
    
    /// Device operation failed
    /// デバイス操作失敗
    #[error("Device error on {device}: {message}")]
    Device { 
        device: String, 
        message: String 
    },
    
    /// Backend not available
    /// バックエンドが利用不可
    #[error("Backend not available: {backend}")]
    BackendUnavailable { 
        backend: String 
    },
    
    /// Memory allocation failed
    /// メモリ割り当て失敗
    #[error("Memory allocation failed: {size} bytes on {device}")]
    MemoryAllocation { 
        size: usize, 
        device: String 
    },
    
    /// Invalid operation parameters
    /// 無効な操作パラメータ
    #[error("Invalid parameters for {operation}: {message}")]
    InvalidParameters { 
        operation: String, 
        message: String 
    },
    
    /// Neural network layer error
    /// ニューラルネットワークレイヤーエラー
    #[error("Neural network error in {layer}: {message}")]
    NeuralNetwork { 
        layer: String, 
        message: String 
    },
    
    /// Automatic differentiation error
    /// 自動微分エラー
    #[error("Autograd error: {message}")]
    Autograd { 
        message: String 
    },
    
    /// Model import/export error
    /// モデルインポート/エクスポートエラー
    #[error("Model I/O error: {message}")]
    ModelIo { 
        message: String,
        #[source] 
        source: Option<Box<dyn Error + Send + Sync>> 
    },
    
    /// Import-specific error
    /// インポート固有エラー
    #[error("Import error: {message}")]
    Import {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>
    },
    
    /// Data loading error
    /// データ読み込みエラー
    #[error("Data loading error: {message}")]
    DataLoading { 
        message: String,
        #[source] 
        source: Option<Box<dyn Error + Send + Sync>> 
    },

    /// Legacy tensor operation errors for backward compatibility
    /// 後方互換性のためのレガシーテンソル操作エラー
    #[error("Tensor error: {0}")]
    Tensor(TensorError),
    
    /// GPU and parallel processing errors
    /// GPU・並列処理エラー
    #[error("Parallel processing error: {0}")]
    Parallel(ParallelError),
    
    /// GPU-specific errors
    /// GPU固有エラー
    #[error("GPU error: {message}")]
    Gpu {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>
    },
    
    /// Vision processing errors
    /// 視覚処理エラー
    #[error("Vision processing error: {message}")]
    Vision {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>
    },
    
    /// Distributed computing errors
    /// 分散コンピューティングエラー
    #[error("Distributed computing error: {message}")]
    Distributed {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>
    },
    
    /// Visualization errors
    /// 可視化エラー
    #[error("Visualization error: {message}")]
    Visualization {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>
    },
    
    /// Input/Output and serialization errors
    /// 入出力・シリアライゼーションエラー
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}

/// Tensor-specific errors
/// テンソル固有エラー
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch between tensors
    /// テンソル間の形状不一致
    ShapeMismatch { 
        /// Expected shape
        /// 期待される形状
        expected: Vec<usize>, 
        /// Actual shape
        /// 実際の形状
        actual: Vec<usize> 
    },
    
    /// Invalid dimension index
    /// 無効な次元インデックス
    InvalidDimension { 
        /// Dimension index
        /// 次元インデックス
        dim: usize, 
        /// Maximum allowed dimension
        /// 最大許可次元
        max_dim: usize 
    },
    
    /// Index out of bounds
    /// インデックス範囲外
    IndexOutOfBounds { 
        /// Index that was out of bounds
        /// 境界外のインデックス
        index: Vec<usize>, 
        /// Shape of the tensor
        /// テンソルの形状
        shape: Vec<usize> 
    },
    
    /// Empty tensor operation
    /// 空テンソル操作
    EmptyTensor,
    
    /// Type conversion error
    /// 型変換エラー
    TypeConversion(String),
    
    /// Mathematical operation error (e.g., division by zero)
    /// 数学演算エラー（ゼロ除算など）
    MathError(String),
}

/// Neural network layer errors
/// ニューラルネットワークレイヤーエラー
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralNetworkError {
    /// Layer configuration error
    /// レイヤー設定エラー
    InvalidConfiguration(String),
    
    /// Forward pass error
    /// 順伝播エラー
    ForwardError(String),
    
    /// Backward pass error
    /// 逆伝播エラー
    BackwardError(String),
    
    /// Parameter initialization error
    /// パラメータ初期化エラー
    InitializationError(String),
}

/// Automatic differentiation errors
/// 自動微分エラー
#[derive(Debug, Clone, PartialEq)]
pub enum AutogradError {
    /// Gradient computation error
    /// 勾配計算エラー
    GradientError(String),
    
    /// Computational graph error
    /// 計算グラフエラー
    GraphError(String),
    
    /// Variable state error
    /// 変数状態エラー
    VariableError(String),
}

/// Optimizer errors
/// オプティマイザーエラー
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    /// Invalid learning rate
    /// 無効な学習率
    InvalidLearningRate(f64),
    
    /// Parameter update error
    /// パラメータ更新エラー
    UpdateError(String),
    
    /// Optimizer state error
    /// オプティマイザー状態エラー
    StateError(String),
}

/// Data handling errors
/// データハンドリングエラー
#[derive(Debug, Clone, PartialEq)]
pub enum DataError {
    /// Dataset loading error
    /// データセット読み込みエラー
    LoadError(String),
    
    /// Data format error
    /// データフォーマットエラー
    FormatError(String),
    
    /// Batch processing error
    /// バッチ処理エラー
    BatchError(String),
}

/// Parallel processing errors
/// 並列処理エラー
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelError {
    /// Thread pool error
    /// スレッドプールエラー
    ThreadError(String),
    
    /// GPU operation error
    /// GPU操作エラー
    GpuError(String),
    
    /// Synchronization error
    /// 同期エラー
    SyncError(String),
}

/// Unified Result type for all RusTorch operations
/// 全RusTorch操作用統一Result型
pub type RusTorchResult<T> = Result<T, RusTorchError>;

/// Legacy type alias for backward compatibility
/// 後方互換性のためのレガシータイプエイリアス
pub type BackendResult<T> = RusTorchResult<T>;
pub type GpuResult<T> = RusTorchResult<T>;
pub type VisionResult<T> = RusTorchResult<T>;
pub type DistributedResult<T> = RusTorchResult<T>;
pub type ImportResult<T> = RusTorchResult<T>;
pub type VisualizationResult<T> = RusTorchResult<T>;
pub type NNResult<T> = RusTorchResult<T>;
pub type ParallelResult<T> = RusTorchResult<T>;


impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, actual)
            }
            TensorError::InvalidDimension { dim, max_dim } => {
                write!(f, "Invalid dimension {} (max: {})", dim, max_dim)
            }
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "Index {:?} out of bounds for shape {:?}", index, shape)
            }
            TensorError::EmptyTensor => write!(f, "Operation on empty tensor"),
            TensorError::TypeConversion(msg) => write!(f, "Type conversion error: {}", msg),
            TensorError::MathError(msg) => write!(f, "Math error: {}", msg),
        }
    }
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            NeuralNetworkError::ForwardError(msg) => write!(f, "Forward pass error: {}", msg),
            NeuralNetworkError::BackwardError(msg) => write!(f, "Backward pass error: {}", msg),
            NeuralNetworkError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutogradError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
            AutogradError::GraphError(msg) => write!(f, "Graph error: {}", msg),
            AutogradError::VariableError(msg) => write!(f, "Variable error: {}", msg),
        }
    }
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::InvalidLearningRate(lr) => write!(f, "Invalid learning rate: {}", lr),
            OptimizerError::UpdateError(msg) => write!(f, "Update error: {}", msg),
            OptimizerError::StateError(msg) => write!(f, "State error: {}", msg),
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::LoadError(msg) => write!(f, "Load error: {}", msg),
            DataError::FormatError(msg) => write!(f, "Format error: {}", msg),
            DataError::BatchError(msg) => write!(f, "Batch error: {}", msg),
        }
    }
}

impl fmt::Display for ParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParallelError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
            ParallelError::GpuError(msg) => write!(f, "GPU error: {}", msg),
            ParallelError::SyncError(msg) => write!(f, "Sync error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}
impl std::error::Error for NeuralNetworkError {}
impl std::error::Error for AutogradError {}
impl std::error::Error for OptimizerError {}
impl std::error::Error for DataError {}
impl std::error::Error for ParallelError {}

impl From<TensorError> for RusTorchError {
    fn from(error: TensorError) -> Self {
        RusTorchError::Tensor(error)
    }
}

impl From<ParallelError> for RusTorchError {
    fn from(error: ParallelError) -> Self {
        RusTorchError::Parallel(error)
    }
}

impl From<NeuralNetworkError> for RusTorchError {
    fn from(error: NeuralNetworkError) -> Self {
        RusTorchError::NeuralNetwork {
            layer: "unknown".to_string(),
            message: error.to_string(),
        }
    }
}

impl From<AutogradError> for RusTorchError {
    fn from(error: AutogradError) -> Self {
        RusTorchError::Autograd {
            message: error.to_string(),
        }
    }
}

impl From<OptimizerError> for RusTorchError {
    fn from(error: OptimizerError) -> Self {
        RusTorchError::InvalidParameters {
            operation: "optimizer".to_string(),
            message: error.to_string(),
        }
    }
}

impl From<DataError> for RusTorchError {
    fn from(error: DataError) -> Self {
        RusTorchError::DataLoading {
            message: error.to_string(),
            source: None,
        }
    }
}

// Missing From implementations
use crate::gpu::GpuError;

impl From<GpuError> for RusTorchError {
    fn from(error: GpuError) -> Self {
        RusTorchError::Gpu {
            message: error.to_string(),
            source: None,
        }
    }
}

// Convenience functions for common error creation
impl TensorError {
    /// Create a shape mismatch error
    /// 形状不一致エラーを作成
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        TensorError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }
    
    /// Create an invalid dimension error
    /// 無効な次元エラーを作成
    pub fn invalid_dimension(dim: usize, max_dim: usize) -> Self {
        TensorError::InvalidDimension { dim, max_dim }
    }
    
    /// Create an index out of bounds error
    /// インデックス境界外エラーを作成
    pub fn index_out_of_bounds(index: &[usize], shape: &[usize]) -> Self {
        TensorError::IndexOutOfBounds {
            index: index.to_vec(),
            shape: shape.to_vec(),
        }
    }
}

impl ParallelError {
    /// Create a shape mismatch error with operation context
    /// 操作コンテキスト付きの形状不一致エラーを作成
    pub fn shape_mismatch(expected: &[usize], actual: &[usize], operation: &str) -> Self {
        ParallelError::SyncError(format!(
            "Shape mismatch in {}: expected {:?}, got {:?}",
            operation, expected, actual
        ))
    }
}

// Convenience constructors for unified error handling
impl RusTorchError {
    /// Create a tensor operation error with context
    /// コンテキスト付きのテンソル操作エラーを作成
    pub fn tensor_op(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: message.into(),
            source: None,
        }
    }
    
    /// Create a tensor operation error with source
    /// ソース付きのテンソル操作エラーを作成
    pub fn tensor_op_with_source(message: impl Into<String>, source: Box<dyn Error + Send + Sync>) -> Self {
        RusTorchError::TensorOp {
            message: message.into(),
            source: Some(source),
        }
    }
    
    /// Create a shape mismatch error
    /// 形状不一致エラーを作成
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        RusTorchError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }
    
    /// Create a device error
    /// デバイスエラーを作成
    pub fn device(device: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::Device {
            device: device.into(),
            message: message.into(),
        }
    }
    
    /// Create a GPU error with context
    /// コンテキスト付きのGPUエラーを作成
    pub fn gpu(message: impl Into<String>) -> Self {
        RusTorchError::Gpu {
            message: message.into(),
            source: None,
        }
    }
    
    /// Create a GPU error with source
    /// ソース付きのGPUエラーを作成
    pub fn gpu_with_source(message: impl Into<String>, source: Box<dyn Error + Send + Sync>) -> Self {
        RusTorchError::Gpu {
            message: message.into(),
            source: Some(source),
        }
    }
    
    /// Create an invalid parameters error
    /// 無効なパラメータエラーを作成
    pub fn invalid_params(operation: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::InvalidParameters {
            operation: operation.into(),
            message: message.into(),
        }
    }
}

// Additional From implementations for module-specific error types

impl From<crate::vision::VisionError> for RusTorchError {
    fn from(error: crate::vision::VisionError) -> Self {
        RusTorchError::Vision {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<crate::visualization::VisualizationError> for RusTorchError {
    fn from(error: crate::visualization::VisualizationError) -> Self {
        RusTorchError::Visualization {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<crate::model_import::ImportError> for RusTorchError {
    fn from(error: crate::model_import::ImportError) -> Self {
        RusTorchError::Import {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<crate::distributed::DistributedError> for RusTorchError {
    fn from(error: crate::distributed::DistributedError) -> Self {
        RusTorchError::Distributed {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<crate::nn::conv_base::NNError> for RusTorchError {
    fn from(error: crate::nn::conv_base::NNError) -> Self {
        RusTorchError::NeuralNetwork {
            layer: "conv_base".to_string(),
            message: error.to_string(),
        }
    }
}

impl From<crate::tensor::parallel_errors::ParallelError> for RusTorchError {
    fn from(error: crate::tensor::parallel_errors::ParallelError) -> Self {
        // Convert to the ParallelError type defined in this module
        let parallel_error = match error {
            crate::tensor::parallel_errors::ParallelError::ShapeMismatch { .. } => {
                ParallelError::SyncError(error.to_string())
            },
            crate::tensor::parallel_errors::ParallelError::ParallelExecutionError { .. } => {
                ParallelError::ThreadError(error.to_string())
            },
            crate::tensor::parallel_errors::ParallelError::SimdError { .. } => {
                ParallelError::GpuError(error.to_string())
            },
            _ => ParallelError::SyncError(error.to_string()),
        };
        RusTorchError::Parallel(parallel_error)
    }
}

// Format-specific error conversions
#[cfg(feature = "onnx")]
impl From<crate::formats::onnx::OnnxError> for RusTorchError {
    fn from(error: crate::formats::onnx::OnnxError) -> Self {
        RusTorchError::ModelIo {
            message: format!("ONNX error: {}", error),
            source: Some(Box::new(error)),
        }
    }
}

#[cfg(feature = "safetensors")]
impl From<crate::formats::safetensors::SafetensorsError> for RusTorchError {
    fn from(error: crate::formats::safetensors::SafetensorsError) -> Self {
        RusTorchError::ModelIo {
            message: format!("Safetensors error: {}", error),
            source: Some(Box::new(error)),
        }
    }
}