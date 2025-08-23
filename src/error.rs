//! Unified error handling system for RusTorch
//! RusTorch用統一エラーハンドリングシステム

use std::fmt;

/// Main error type for RusTorch operations
/// RusTorch操作のメインエラー型
#[derive(Debug, Clone, PartialEq)]
pub enum RusTorchError {
    /// Tensor operation errors
    /// テンソル操作エラー
    Tensor(TensorError),
    
    /// Neural network layer errors  
    /// ニューラルネットワークレイヤーエラー
    NeuralNetwork(NeuralNetworkError),
    
    /// Automatic differentiation errors
    /// 自動微分エラー
    Autograd(AutogradError),
    
    /// Optimizer errors
    /// オプティマイザーエラー
    Optimizer(OptimizerError),
    
    /// Data loading and processing errors
    /// データ読み込み・処理エラー
    Data(DataError),
    
    /// GPU and parallel processing errors
    /// GPU・並列処理エラー
    Parallel(ParallelError),
    
    /// Input/Output and serialization errors
    /// 入出力・シリアライゼーションエラー
    IO(std::io::Error),
}

/// Tensor-specific errors
/// テンソル固有エラー
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch between tensors
    /// テンソル間の形状不一致
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    /// Invalid dimension index
    /// 無効な次元インデックス
    InvalidDimension { dim: usize, max_dim: usize },
    
    /// Index out of bounds
    /// インデックス範囲外
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    
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

/// Result type for RusTorch operations
/// RusTorch操作用Result型
pub type RusTorchResult<T> = Result<T, RusTorchError>;

impl fmt::Display for RusTorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RusTorchError::Tensor(e) => write!(f, "Tensor error: {}", e),
            RusTorchError::NeuralNetwork(e) => write!(f, "Neural network error: {}", e),
            RusTorchError::Autograd(e) => write!(f, "Autograd error: {}", e),
            RusTorchError::Optimizer(e) => write!(f, "Optimizer error: {}", e),
            RusTorchError::Data(e) => write!(f, "Data error: {}", e),
            RusTorchError::Parallel(e) => write!(f, "Parallel processing error: {}", e),
            RusTorchError::IO(e) => write!(f, "IO error: {}", e),
        }
    }
}

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

impl std::error::Error for RusTorchError {}
impl std::error::Error for TensorError {}
impl std::error::Error for NeuralNetworkError {}
impl std::error::Error for AutogradError {}
impl std::error::Error for OptimizerError {}
impl std::error::Error for DataError {}
impl std::error::Error for ParallelError {}

impl From<std::io::Error> for RusTorchError {
    fn from(error: std::io::Error) -> Self {
        RusTorchError::IO(error)
    }
}

impl From<TensorError> for RusTorchError {
    fn from(error: TensorError) -> Self {
        RusTorchError::Tensor(error)
    }
}

impl From<NeuralNetworkError> for RusTorchError {
    fn from(error: NeuralNetworkError) -> Self {
        RusTorchError::NeuralNetwork(error)
    }
}

impl From<AutogradError> for RusTorchError {
    fn from(error: AutogradError) -> Self {
        RusTorchError::Autograd(error)
    }
}

impl From<OptimizerError> for RusTorchError {
    fn from(error: OptimizerError) -> Self {
        RusTorchError::Optimizer(error)
    }
}

impl From<DataError> for RusTorchError {
    fn from(error: DataError) -> Self {
        RusTorchError::Data(error)
    }
}

impl From<ParallelError> for RusTorchError {
    fn from(error: ParallelError) -> Self {
        RusTorchError::Parallel(error)
    }
}

// Convenience functions for common error creation
impl TensorError {
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        TensorError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }
    
    pub fn invalid_dimension(dim: usize, max_dim: usize) -> Self {
        TensorError::InvalidDimension { dim, max_dim }
    }
    
    pub fn index_out_of_bounds(index: &[usize], shape: &[usize]) -> Self {
        TensorError::IndexOutOfBounds {
            index: index.to_vec(),
            shape: shape.to_vec(),
        }
    }
}

impl ParallelError {
    pub fn shape_mismatch(expected: &[usize], actual: &[usize], operation: &str) -> Self {
        ParallelError::SyncError(format!(
            "Shape mismatch in {}: expected {:?}, got {:?}",
            operation, expected, actual
        ))
    }
}