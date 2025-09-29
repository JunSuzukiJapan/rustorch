//! hybrid_f32専用エラー処理
//! hybrid_f32-specific error handling

use thiserror::Error;
use std::fmt;

/// hybrid_f32のResult型エイリアス
/// Result type alias for hybrid_f32
pub type F32Result<T> = Result<T, F32Error>;

/// hybrid_f32専用エラー型
/// hybrid_f32-specific error type
#[derive(Error, Debug, Clone)]
pub enum F32Error {
    /// 形状不一致エラー
    /// Shape mismatch error
    #[error("Shape mismatch: {message}")]
    ShapeMismatch { message: String },

    /// インデックスエラー
    /// Index error
    #[error("Index error: {message}")]
    IndexError { message: String },

    /// 次元エラー
    /// Dimension error
    #[error("Dimension error: expected {expected}, got {actual}")]
    DimensionError { expected: usize, actual: usize },

    /// 数値演算エラー
    /// Numerical operation error
    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    /// デバイスエラー
    /// Device error
    #[error("Device error: {message}")]
    DeviceError { message: String },

    /// メモリエラー
    /// Memory error
    #[error("Memory error: {message}")]
    MemoryError { message: String },

    /// 無効な操作
    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    /// GPU操作エラー
    /// GPU operation error
    #[error("GPU error: {message}")]
    GpuError { message: String },

    /// Neural Engineエラー
    /// Neural Engine error
    #[error("Neural Engine error: {message}")]
    NeuralEngineError { message: String },

    /// 変換エラー
    /// Conversion error
    #[error("Conversion error: {message}")]
    ConversionError { message: String },

    /// データ検証エラー
    /// Data validation error
    #[error("Validation error: {message}")]
    ValidationError { message: String },
}

impl F32Error {
    /// 形状不一致エラーを作成
    /// Create shape mismatch error
    pub fn shape_mismatch<S: Into<String>>(message: S) -> Self {
        Self::ShapeMismatch { message: message.into() }
    }

    /// インデックスエラーを作成
    /// Create index error
    pub fn index_error<S: Into<String>>(message: S) -> Self {
        Self::IndexError { message: message.into() }
    }

    /// 次元エラーを作成
    /// Create dimension error
    pub fn dimension_error(expected: usize, actual: usize) -> Self {
        Self::DimensionError { expected, actual }
    }

    /// 数値演算エラーを作成
    /// Create numerical error
    pub fn numerical_error<S: Into<String>>(message: S) -> Self {
        Self::NumericalError { message: message.into() }
    }

    /// デバイスエラーを作成
    /// Create device error
    pub fn device_error<S: Into<String>>(message: S) -> Self {
        Self::DeviceError { message: message.into() }
    }

    /// メモリエラーを作成
    /// Create memory error
    pub fn memory_error<S: Into<String>>(message: S) -> Self {
        Self::MemoryError { message: message.into() }
    }

    /// 無効な操作エラーを作成
    /// Create invalid operation error
    pub fn invalid_operation<S: Into<String>>(message: S) -> Self {
        Self::InvalidOperation { message: message.into() }
    }

    /// GPUエラーを作成
    /// Create GPU error
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuError { message: message.into() }
    }

    /// Neural Engineエラーを作成
    /// Create Neural Engine error
    pub fn neural_engine_error<S: Into<String>>(message: S) -> Self {
        Self::NeuralEngineError { message: message.into() }
    }

    /// 変換エラーを作成
    /// Create conversion error
    pub fn conversion_error<S: Into<String>>(message: S) -> Self {
        Self::ConversionError { message: message.into() }
    }

    /// データ検証エラーを作成
    /// Create validation error
    pub fn validation_error<S: Into<String>>(message: S) -> Self {
        Self::ValidationError { message: message.into() }
    }

    /// エラーがリカバリ可能かどうか判定
    /// Determine if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            F32Error::ShapeMismatch { .. } => false,
            F32Error::IndexError { .. } => false,
            F32Error::DimensionError { .. } => false,
            F32Error::NumericalError { .. } => true,  // 一部回復可能
            F32Error::DeviceError { .. } => true,     // デバイス切替可能
            F32Error::MemoryError { .. } => true,     // メモリ解放後再試行可能
            F32Error::InvalidOperation { .. } => false,
            F32Error::GpuError { .. } => true,        // CPU fallback可能
            F32Error::NeuralEngineError { .. } => true, // GPU fallback可能
            F32Error::ConversionError { .. } => false,
            F32Error::ValidationError { .. } => false,
        }
    }

    /// エラーの重要度を取得
    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            F32Error::ShapeMismatch { .. } => ErrorSeverity::High,
            F32Error::IndexError { .. } => ErrorSeverity::High,
            F32Error::DimensionError { .. } => ErrorSeverity::High,
            F32Error::NumericalError { .. } => ErrorSeverity::Medium,
            F32Error::DeviceError { .. } => ErrorSeverity::Medium,
            F32Error::MemoryError { .. } => ErrorSeverity::High,
            F32Error::InvalidOperation { .. } => ErrorSeverity::High,
            F32Error::GpuError { .. } => ErrorSeverity::Low,     // fallback可能
            F32Error::NeuralEngineError { .. } => ErrorSeverity::Low, // fallback可能
            F32Error::ConversionError { .. } => ErrorSeverity::Medium,
            F32Error::ValidationError { .. } => ErrorSeverity::High,
        }
    }

    /// ユーザー向けメッセージを取得
    /// Get user-friendly message
    pub fn user_message(&self) -> String {
        match self {
            F32Error::ShapeMismatch { .. } => 
                "テンソルの形状が一致しません。演算を行うには同じ形状である必要があります。".to_string(),
            F32Error::IndexError { .. } => 
                "インデックスが範囲外です。有効な範囲内のインデックスを指定してください。".to_string(),
            F32Error::DimensionError { expected, actual } => 
                format!("次元数が不正です。{}次元が必要ですが、{}次元が指定されました。", expected, actual),
            F32Error::NumericalError { .. } => 
                "数値計算でエラーが発生しました。入力値を確認してください。".to_string(),
            F32Error::DeviceError { .. } => 
                "デバイス操作でエラーが発生しました。別のデバイスを試してください。".to_string(),
            F32Error::MemoryError { .. } => 
                "メモリ不足です。データサイズを小さくするか、メモリを解放してください。".to_string(),
            F32Error::InvalidOperation { .. } => 
                "無効な操作です。操作の前提条件を確認してください。".to_string(),
            F32Error::GpuError { .. } => 
                "GPU操作でエラーが発生しました。CPUモードで再試行します。".to_string(),
            F32Error::NeuralEngineError { .. } => 
                "Neural Engine操作でエラーが発生しました。GPUモードで再試行します。".to_string(),
            F32Error::ConversionError { .. } => 
                "データ変換でエラーが発生しました。入力データの形式を確認してください。".to_string(),
            F32Error::ValidationError { .. } => 
                "データ検証でエラーが発生しました。入力データが仕様に適合しているか確認してください。".to_string(),
        }
    }
}

/// エラーの重要度
/// Error severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
        }
    }
}

/// メインRusTorchErrorからの変換
/// Conversion from main RusTorchError
impl From<crate::error::RusTorchError> for F32Error {
    fn from(error: crate::error::RusTorchError) -> Self {
        use crate::error::RusTorchError;

        match error {
            // 既存のエラータイプマッピング
            RusTorchError::TensorOp { message, .. } => F32Error::invalid_operation(message),
            RusTorchError::ShapeMismatch { expected, actual } =>
                F32Error::shape_mismatch(format!("expected {:?}, got {:?}", expected, actual)),
            RusTorchError::Device { device, message } =>
                F32Error::device_error(format!("{}: {}", device, message)),
            RusTorchError::BackendUnavailable { backend } =>
                F32Error::device_error(format!("Backend unavailable: {}", backend)),
            RusTorchError::MemoryAllocation { size, device } =>
                F32Error::memory_error(format!("Failed to allocate {} bytes on {}", size, device)),
            RusTorchError::InvalidParameters { operation, message } =>
                F32Error::invalid_operation(format!("{}: {}", operation, message)),
            RusTorchError::InvalidOperation { operation, message } =>
                F32Error::invalid_operation(format!("{}: {}", operation, message)),
            RusTorchError::NeuralNetwork { layer, message } =>
                F32Error::neural_engine_error(format!("{}: {}", layer, message)),
            RusTorchError::Autograd { message } =>
                F32Error::numerical_error(message),
            RusTorchError::ModelIo { message, .. } =>
                F32Error::conversion_error(message),
            RusTorchError::Import { message, .. } =>
                F32Error::conversion_error(message),
            RusTorchError::DataLoading { message, .. } =>
                F32Error::validation_error(message),
            RusTorchError::Gpu { message, .. } =>
                F32Error::gpu_error(message),
            RusTorchError::Vision { message, .. } =>
                F32Error::invalid_operation(message),
            RusTorchError::Distributed { message, .. } =>
                F32Error::device_error(message),
            RusTorchError::Visualization { message, .. } =>
                F32Error::invalid_operation(message),
            RusTorchError::Profiling { message } =>
                F32Error::invalid_operation(message),
            RusTorchError::Validation { message } =>
                F32Error::validation_error(message),
            RusTorchError::Debug { message } =>
                F32Error::invalid_operation(message),
            RusTorchError::KernelCompilation { message } =>
                F32Error::gpu_error(message),
            RusTorchError::Dataloader { message } =>
                F32Error::validation_error(message),
            RusTorchError::SerializationError { operation, message } =>
                F32Error::conversion_error(format!("{}: {}", operation, message)),
            RusTorchError::IO(err) =>
                F32Error::invalid_operation(err.to_string()),
            RusTorchError::NotImplemented { feature } =>
                F32Error::invalid_operation(format!("Not implemented: {}", feature)),
            RusTorchError::OutOfMemory { requested, available } =>
                F32Error::memory_error(format!("Out of memory: requested {} bytes, available {} bytes", requested, available)),
            RusTorchError::Serialization { message } =>
                F32Error::conversion_error(message),
        }
    }
}

/// F32ErrorからメインRusTorchErrorへの変換
/// Conversion to main RusTorchError
impl From<F32Error> for crate::error::RusTorchError {
    fn from(error: F32Error) -> Self {
        use crate::error::RusTorchError;

        match error {
            F32Error::ShapeMismatch { message } =>
                RusTorchError::ShapeMismatch {
                    expected: vec![],  // デフォルト値として空のベクター
                    actual: vec![],    // 詳細は message に含まれる
                },
            F32Error::IndexError { message } =>
                RusTorchError::InvalidOperation {
                    operation: "index".to_string(),
                    message,
                },
            F32Error::DimensionError { expected, actual } =>
                RusTorchError::InvalidOperation {
                    operation: "dimension_check".to_string(),
                    message: format!("expected {}, got {}", expected, actual),
                },
            F32Error::NumericalError { message } =>
                RusTorchError::TensorOp {
                    message,
                    source: None,
                },
            F32Error::DeviceError { message } =>
                RusTorchError::Device {
                    device: "hybrid_f32".to_string(),
                    message,
                },
            F32Error::MemoryError { message } =>
                RusTorchError::OutOfMemory {
                    requested: 0,  // 詳細情報は message に含まれる
                    available: 0,
                },
            F32Error::InvalidOperation { message } =>
                RusTorchError::InvalidOperation {
                    operation: "hybrid_f32".to_string(),
                    message,
                },
            F32Error::GpuError { message } =>
                RusTorchError::Gpu {
                    message,
                    source: None,
                },
            F32Error::NeuralEngineError { message } =>
                RusTorchError::NeuralNetwork {
                    layer: "neural_engine".to_string(),
                    message,
                },
            F32Error::ConversionError { message } =>
                RusTorchError::Serialization { message },
            F32Error::ValidationError { message } =>
                RusTorchError::Validation { message },
        }
    }
}

/// エラーコンテキスト用のトレイト
/// Trait for error context
pub trait F32ErrorContext<T> {
    /// エラーにコンテキストを追加
    /// Add context to error
    fn with_context<F, S>(self, f: F) -> F32Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;

    /// エラーにコンテキスト文字列を追加
    /// Add context string to error
    fn context<S: Into<String>>(self, msg: S) -> F32Result<T>;
}

impl<T> F32ErrorContext<T> for F32Result<T> {
    fn with_context<F, S>(self, f: F) -> F32Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|err| {
            let context = f().into();
            match err {
                F32Error::ShapeMismatch { message } => 
                    F32Error::shape_mismatch(format!("{}: {}", context, message)),
                F32Error::IndexError { message } => 
                    F32Error::index_error(format!("{}: {}", context, message)),
                F32Error::InvalidOperation { message } => 
                    F32Error::invalid_operation(format!("{}: {}", context, message)),
                other => other,
            }
        })
    }

    fn context<S: Into<String>>(self, msg: S) -> F32Result<T> {
        let msg = msg.into();
        self.with_context(|| msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = F32Error::shape_mismatch("test message");
        assert!(matches!(err, F32Error::ShapeMismatch { .. }));
        assert_eq!(err.to_string(), "Shape mismatch: test message");
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(F32Error::shape_mismatch("test").severity(), ErrorSeverity::High);
        assert_eq!(F32Error::gpu_error("test").severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_error_recoverability() {
        assert!(!F32Error::shape_mismatch("test").is_recoverable());
        assert!(F32Error::gpu_error("test").is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let result: F32Result<i32> = Err(F32Error::shape_mismatch("original"));
        let with_context = result.context("tensor operation");
        
        assert!(with_context.is_err());
        let err_msg = with_context.unwrap_err().to_string();
        assert!(err_msg.contains("tensor operation"));
        assert!(err_msg.contains("original"));
    }
}