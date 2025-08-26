//! Unified error handling system for RusTorch
//! RusTorch用統一エラーハンドリングシステム

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
        /// Error message describing the tensor operation failure
        /// テンソル操作失敗を説明するエラーメッセージ
        message: String,
        /// Optional underlying error cause
        /// オプションの根本的なエラー原因
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Shape mismatch between tensors
    /// テンソル間の形状不一致
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected tensor shape
        /// 期待されるテンソル形状
        expected: Vec<usize>,
        /// Actual tensor shape that was provided
        /// 提供された実際のテンソル形状
        actual: Vec<usize>,
    },

    /// Device operation failed
    /// デバイス操作失敗
    #[error("Device error on {device}: {message}")]
    Device {
        /// Device identifier where the error occurred
        /// エラーが発生したデバイス識別子
        device: String,
        /// Error message describing the device issue
        /// デバイスの問題を説明するエラーメッセージ
        message: String,
    },

    /// Backend not available
    /// バックエンドが利用不可
    #[error("Backend not available: {backend}")]
    BackendUnavailable {
        /// Name of the unavailable backend
        /// 利用不可なバックエンド名
        backend: String,
    },

    /// Memory allocation failed
    /// メモリ割り当て失敗
    #[error("Memory allocation failed: {size} bytes on {device}")]
    MemoryAllocation {
        /// Size of the failed allocation in bytes
        /// 失敗したアロケーションのサイズ（バイト）
        size: usize,
        /// Device where allocation failed
        /// アロケーションが失敗したデバイス
        device: String,
    },

    /// Invalid operation parameters
    /// 無効な操作パラメータ
    #[error("Invalid parameters for {operation}: {message}")]
    InvalidParameters {
        /// Name of the operation with invalid parameters
        /// 無効なパラメータを持つ操作名
        operation: String,
        /// Description of the parameter issue
        /// パラメータの問題の説明
        message: String,
    },

    /// Invalid operation
    /// 無効な操作
    #[error("{operation}: {message}")]
    InvalidOperation {
        /// Name of the invalid operation
        /// 無効な操作名
        operation: String,
        /// Detailed error message
        /// 詳細なエラーメッセージ
        message: String,
    },

    /// Neural network layer error
    /// ニューラルネットワークレイヤーエラー
    #[error("Neural network error in {layer}: {message}")]
    NeuralNetwork {
        /// Name of the neural network layer
        /// ニューラルネットワークレイヤー名
        layer: String,
        /// Error message from the layer
        /// レイヤーからのエラーメッセージ
        message: String,
    },

    /// Automatic differentiation error
    /// 自動微分エラー
    #[error("Autograd error: {message}")]
    Autograd {
        /// Autograd error description
        /// 自動微分エラーの説明
        message: String,
    },

    /// Model import/export error
    /// モデルインポート/エクスポートエラー
    #[error("Model I/O error: {message}")]
    ModelIo {
        /// Model I/O error description
        /// モデルI/Oエラーの説明
        message: String,
        /// Optional underlying I/O error
        /// オプションの根本的なI/Oエラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Import-specific error
    /// インポート固有エラー
    #[error("Import error: {message}")]
    Import {
        /// Import operation error message
        /// インポート操作エラーメッセージ
        message: String,
        /// Optional underlying import error
        /// オプションの根本的なインポートエラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Data loading error
    /// データ読み込みエラー
    #[error("Data loading error: {message}")]
    DataLoading {
        /// Data loading error description
        /// データ読み込みエラーの説明
        message: String,
        /// Optional underlying data loading error
        /// オプションの根本的なデータ読み込みエラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// GPU-specific errors
    /// GPU固有エラー
    #[error("GPU error: {message}")]
    Gpu {
        /// GPU operation error description
        /// GPU操作エラーの説明
        message: String,
        /// Optional underlying GPU error
        /// オプションの根本的なGPUエラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Vision processing errors
    /// 視覚処理エラー
    #[error("Vision processing error: {message}")]
    Vision {
        /// Vision processing error description
        /// 視覚処理エラーの説明
        message: String,
        /// Optional underlying vision error
        /// オプションの根本的な視覚エラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Distributed computing errors
    /// 分散コンピューティングエラー
    #[error("Distributed computing error: {message}")]
    Distributed {
        /// Distributed computing error description
        /// 分散コンピューティングエラーの説明
        message: String,
        /// Optional underlying distributed error
        /// オプションの根本的な分散エラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Visualization errors
    /// 可視化エラー
    #[error("Visualization error: {message}")]
    Visualization {
        /// Visualization error description
        /// 可視化エラーの説明
        message: String,
        /// Optional underlying visualization error
        /// オプションの根本的な可視化エラー
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Input/Output and serialization errors
    /// 入出力・シリアライゼーションエラー
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}

// All individual error types removed - only RusTorchError is used now
// 個別エラー型は全て削除 - RusTorchErrorのみ使用

/// Unified Result type for all RusTorch operations - the ONLY Result type used
/// 全RusTorch操作用統一Result型 - 唯一使用されるResult型
pub type RusTorchResult<T> = Result<T, RusTorchError>;

// All Display implementations and From traits removed - RusTorchError handles everything directly
// 全DisplayとFromトレイト削除 - RusTorchErrorが直接全てを処理

// Comprehensive convenience constructors for unified error handling
#[allow(non_snake_case)] // Allow PascalCase for error constructor methods
impl RusTorchError {
    // === Tensor Operations ===
    /// Create a tensor operation error
    pub fn tensor_op(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: message.into(),
            source: None,
        }
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        RusTorchError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create empty tensor error
    pub fn empty_tensor() -> Self {
        RusTorchError::TensorOp {
            message: "Operation on empty tensor".into(),
            source: None,
        }
    }

    /// Create index out of bounds error
    pub fn index_out_of_bounds(index: &[usize], shape: &[usize]) -> Self {
        RusTorchError::TensorOp {
            message: format!("Index {:?} out of bounds for shape {:?}", index, shape),
            source: None,
        }
    }

    /// Create invalid dimension error
    pub fn invalid_dimension(dim: usize, max_dim: usize) -> Self {
        RusTorchError::TensorOp {
            message: format!("Invalid dimension {} (max: {})", dim, max_dim),
            source: None,
        }
    }

    // === Device & GPU Operations ===
    /// Create a device error
    pub fn device(device: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::Device {
            device: device.into(),
            message: message.into(),
        }
    }

    /// Create a GPU error
    pub fn gpu(message: impl Into<String>) -> Self {
        RusTorchError::Gpu {
            message: message.into(),
            source: None,
        }
    }

    /// Create device not available error
    pub fn device_not_available(device: impl Into<String>) -> Self {
        RusTorchError::Device {
            device: device.into(),
            message: "Device not available".into(),
        }
    }

    /// Create memory allocation error
    pub fn memory_alloc(size: usize, device: impl Into<String>) -> Self {
        RusTorchError::MemoryAllocation {
            size,
            device: device.into(),
        }
    }

    // === Neural Network Operations ===
    /// Create neural network error
    pub fn neural_network(layer: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::NeuralNetwork {
            layer: layer.into(),
            message: message.into(),
        }
    }

    /// Create invalid configuration error
    pub fn invalid_config(layer: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::NeuralNetwork {
            layer: layer.into(),
            message: format!("Invalid configuration: {}", message.into()),
        }
    }

    /// Create forward pass error
    pub fn forward_error(layer: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::NeuralNetwork {
            layer: layer.into(),
            message: format!("Forward pass error: {}", message.into()),
        }
    }

    /// Create backward pass error
    pub fn backward_error(layer: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::NeuralNetwork {
            layer: layer.into(),
            message: format!("Backward pass error: {}", message.into()),
        }
    }

    // === Data Operations ===
    /// Create data loading error
    pub fn data_loading(message: impl Into<String>) -> Self {
        RusTorchError::DataLoading {
            message: message.into(),
            source: None,
        }
    }

    /// Create dataset error
    pub fn dataset_error(message: impl Into<String>) -> Self {
        RusTorchError::DataLoading {
            message: format!("Dataset error: {}", message.into()),
            source: None,
        }
    }

    // === Vision Operations ===
    /// Create vision processing error
    pub fn vision(message: impl Into<String>) -> Self {
        RusTorchError::Vision {
            message: message.into(),
            source: None,
        }
    }

    /// Create invalid image format error
    pub fn invalid_image_format(format: impl Into<String>) -> Self {
        RusTorchError::Vision {
            message: format!("Invalid image format: {}", format.into()),
            source: None,
        }
    }

    /// Create invalid image shape error
    pub fn invalid_image_shape(shape: &[usize]) -> Self {
        RusTorchError::Vision {
            message: format!("Invalid image shape: {:?}", shape),
            source: None,
        }
    }

    // === Distributed Operations ===
    /// Create distributed error
    pub fn distributed(message: impl Into<String>) -> Self {
        RusTorchError::Distributed {
            message: message.into(),
            source: None,
        }
    }

    /// Create process group error
    pub fn process_group_error(message: impl Into<String>) -> Self {
        RusTorchError::Distributed {
            message: format!("Process group error: {}", message.into()),
            source: None,
        }
    }

    // === Visualization Operations ===
    /// Create visualization error
    pub fn visualization(message: impl Into<String>) -> Self {
        RusTorchError::Visualization {
            message: message.into(),
            source: None,
        }
    }

    /// Create plotting error
    pub fn plotting_error(message: impl Into<String>) -> Self {
        RusTorchError::Visualization {
            message: format!("Plotting error: {}", message.into()),
            source: None,
        }
    }

    // === Autograd Operations ===
    /// Create autograd error
    pub fn autograd(message: impl Into<String>) -> Self {
        RusTorchError::Autograd {
            message: message.into(),
        }
    }

    /// Create gradient computation error
    pub fn gradient_error(message: impl Into<String>) -> Self {
        RusTorchError::Autograd {
            message: format!("Gradient error: {}", message.into()),
        }
    }

    // === Model I/O Operations ===
    /// Create model I/O error
    pub fn model_io(message: impl Into<String>) -> Self {
        RusTorchError::ModelIo {
            message: message.into(),
            source: None,
        }
    }

    /// Create import error
    pub fn import_error(message: impl Into<String>) -> Self {
        RusTorchError::Import {
            message: message.into(),
            source: None,
        }
    }

    /// Create unsupported format error
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        RusTorchError::Import {
            message: format!("Unsupported format: {}", format.into()),
            source: None,
        }
    }

    // === Parameter Operations ===
    /// Create invalid parameters error
    pub fn invalid_params(operation: impl Into<String>, message: impl Into<String>) -> Self {
        RusTorchError::InvalidParameters {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create backend unavailable error
    pub fn backend_unavailable(backend: impl Into<String>) -> Self {
        RusTorchError::BackendUnavailable {
            backend: backend.into(),
        }
    }

    // === Type and Numeric Operations ===
    /// Create type error
    pub fn type_error(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: message.into(),
            source: None,
        }
    }

    /// Create numeric error
    pub fn numeric(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: message.into(),
            source: None,
        }
    }

    // === Additional Helper Functions ===
    /// Create unsupported device error
    pub fn UnsupportedDevice(message: impl Into<String>) -> Self {
        RusTorchError::device("gpu", message)
    }

    /// Create parallel error
    pub fn parallel(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Parallel error: {}", message.into()),
            source: None,
        }
    }

    /// Create domain error
    pub fn DomainError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Domain error: {}", message.into()),
            source: None,
        }
    }

    /// Create overflow error
    pub fn OverflowError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Overflow error: {}", message.into()),
            source: None,
        }
    }

    /// Create invalid dimensions error
    pub fn InvalidDimensions(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Invalid dimensions: {}", message.into()),
            source: None,
        }
    }

    /// Create invalid data format error
    pub fn InvalidDataFormat(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Invalid data format: {}", message.into()),
            source: None,
        }
    }

    /// Create invalid operation error
    pub fn InvalidOperation(message: impl Into<String>) -> Self {
        RusTorchError::invalid_params("operation", message)
    }

    /// Create kernel execution error
    pub fn KernelExecutionError(message: impl Into<String>) -> Self {
        RusTorchError::gpu(format!("Kernel execution error: {}", message.into()))
    }

    /// Create communication error
    pub fn CommunicationError(message: impl Into<String>) -> Self {
        RusTorchError::distributed(format!("Communication error: {}", message.into()))
    }

    // === More Helper Functions ===
    /// Create unsupported operation error
    pub fn UnsupportedOperation(message: impl Into<String>) -> Self {
        RusTorchError::invalid_params(
            "operation",
            format!("Unsupported operation: {}", message.into()),
        )
    }

    /// Create IO error
    pub fn IoError(message: impl Into<String>) -> Self {
        RusTorchError::model_io(message)
    }

    /// Create kernel error
    pub fn KernelError(message: impl Into<String>) -> Self {
        RusTorchError::gpu(format!("Kernel error: {}", message.into()))
    }

    /// Create invalid transform params error
    pub fn InvalidTransformParams(message: impl Into<String>) -> Self {
        RusTorchError::vision(format!("Invalid transform params: {}", message.into()))
    }

    /// Create process group error
    pub fn ProcessGroupError(message: impl Into<String>) -> Self {
        RusTorchError::process_group_error(message)
    }

    /// Create memory error
    pub fn MemoryError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Memory error: {}", message.into()),
            source: None,
        }
    }

    /// Create plotting error
    pub fn PlottingError(message: impl Into<String>) -> Self {
        RusTorchError::plotting_error(message)
    }

    // === Final Helper Functions ===
    /// Create memory error (simple)
    pub fn memory(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Memory: {}", message.into()),
            source: None,
        }
    }

    /// Create invalid model error
    pub fn InvalidModel(message: impl Into<String>) -> Self {
        RusTorchError::model_io(format!("Invalid model: {}", message.into()))
    }

    /// Create dataset error
    pub fn DatasetError(message: impl Into<String>) -> Self {
        RusTorchError::dataset_error(message)
    }

    /// Create configuration error
    pub fn ConfigurationError(message: impl Into<String>) -> Self {
        RusTorchError::invalid_params("config", format!("Configuration error: {}", message.into()))
    }

    // === Broadcasting and Shape Operations ===
    /// Create incompatible shapes error
    pub fn IncompatibleShapes(_message: impl Into<String>) -> Self {
        RusTorchError::shape_mismatch(&[], &[])
    }

    /// Create invalid dimension error
    pub fn InvalidDimension(message: impl Into<String>) -> Self {
        RusTorchError::InvalidDimensions(message)
    }

    /// Create reshape error
    pub fn ReshapeError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Reshape error: {}", message.into()),
            source: None,
        }
    }

    /// Create not singleton dimension error
    pub fn NotSingletonDimension(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Not singleton dimension: {}", message.into()),
            source: None,
        }
    }

    /// Create mismatched dimensions error
    pub fn MismatchedDimensions(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Mismatched dimensions: {}", message.into()),
            source: None,
        }
    }

    /// Create computation error
    pub fn ComputationError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Computation error: {}", message.into()),
            source: None,
        }
    }

    // === Additional Missing Functions ===
    /// Create config error
    pub fn ConfigError(message: impl Into<String>) -> Self {
        RusTorchError::invalid_params("config", message)
    }

    /// Create invalid image shape error
    pub fn InvalidImageShape(_message: impl Into<String>) -> Self {
        RusTorchError::invalid_image_shape(&[])
    }

    /// Create convergence error
    pub fn ConvergenceError(message: impl Into<String>) -> Self {
        RusTorchError::TensorOp {
            message: format!("Convergence error: {}", message.into()),
            source: None,
        }
    }

    /// Create serialization error
    pub fn SerializationError(message: impl Into<String>) -> Self {
        RusTorchError::model_io(format!("Serialization error: {}", message.into()))
    }

    /// Create parse error
    pub fn ParseError(message: impl Into<String>) -> Self {
        RusTorchError::model_io(format!("Parse error: {}", message.into()))
    }

    /// Create validation error
    pub fn ValidationError(message: impl Into<String>) -> Self {
        RusTorchError::tensor_op(format!("Validation error: {}", message.into()))
    }

    /// Create device not available error
    pub fn DeviceNotAvailable(message: impl Into<String>) -> Self {
        RusTorchError::device_not_available(message.into())
    }

    /// Create file not found error
    pub fn FileNotFound(message: impl Into<String>) -> Self {
        RusTorchError::model_io(format!("File not found: {}", message.into()))
    }

    /// Create cluster error
    pub fn ClusterError(message: impl Into<String>) -> Self {
        RusTorchError::distributed(format!("Cluster error: {}", message.into()))
    }

    /// Create invalid rank error
    pub fn InvalidRank(message: impl Into<String>) -> Self {
        RusTorchError::distributed(format!("Invalid rank: {}", message.into()))
    }
}

// All individual From implementations removed - using unified RusTorchError only
// 全ての個別From実装削除 - 統一RusTorchErrorのみ使用

// Format-specific error conversions removed - using unified error handling
// フォーマット固有エラー変換削除 - 統一エラーハンドリング使用
