//! CoreML operation implementations
//! CoreML演算実装

use super::common::*;
use crate::error::RusTorchResult;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// CoreML specific errors
/// CoreML固有エラー
#[derive(Debug, thiserror::Error)]
pub enum CoreMLError {
    /// Operation not supported by CoreML
    /// CoreMLでサポートされていない操作
    #[error("CoreML does not support operation: {operation}")]
    UnsupportedOperation {
        /// Operation name that is not supported
        /// サポートされていない操作名
        operation: String,
    },
    /// CoreML not available on this platform
    /// このプラットフォームでCoreMLが利用不可
    #[error("CoreML not available")]
    NotAvailable,
    /// CoreML backend error
    /// CoreMLバックエンドエラー
    #[error("CoreML backend error: {message}")]
    Backend {
        /// Backend error message
        /// バックエンドエラーメッセージ
        message: String,
    },
    /// Invalid input for CoreML operation
    /// CoreML操作の無効な入力
    #[error("Invalid input for CoreML operation: {0}")]
    InvalidInput(String),
    /// Error during tensor conversion
    /// テンソル変換中のエラー
    #[error("Conversion error: {0}")]
    ConversionError(String),
}

impl From<CoreMLError> for crate::error::RusTorchError {
    fn from(err: CoreMLError) -> Self {
        match err {
            CoreMLError::UnsupportedOperation { operation } => {
                crate::error::RusTorchError::InvalidOperation {
                    operation,
                    message: "CoreML does not support this operation".to_string(),
                }
            }
            CoreMLError::NotAvailable => crate::error::RusTorchError::BackendUnavailable {
                backend: "CoreML".to_string(),
            },
            CoreMLError::Backend { message } => crate::error::RusTorchError::Device {
                device: "CoreML".to_string(),
                message,
            },
            CoreMLError::InvalidInput(message) => {
                crate::error::RusTorchError::InvalidParameters {
                    operation: "CoreML".to_string(),
                    message,
                }
            },
            CoreMLError::ConversionError(message) => {
                crate::error::RusTorchError::TensorOp {
                    message,
                    source: None,
                }
            },
        }
    }
}

/// Linear algebra operations module
/// 線形代数演算モジュール
pub mod linear_algebra;

/// Convolution operations module
/// 畳み込み演算モジュール
pub mod convolution;

/// Activation functions module
/// 活性化関数モジュール
pub mod activation;

// Re-export operation traits
pub use activation::*;
pub use convolution::*;
pub use linear_algebra::*;

/// Base trait for CoreML operations
/// CoreML演算のベーストレイト
pub trait CoreMLOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// Execute the operation using CoreML
    /// CoreMLを使用して演算を実行
    fn execute_coreml(&self, device_id: usize) -> CoreMLResult<Tensor<T>>;

    /// Check if operation is supported by CoreML
    /// 演算がCoreMLでサポートされているかチェック
    fn is_supported_by_coreml(&self) -> bool;

    /// Get estimated execution time for the operation
    /// 演算の推定実行時間を取得
    fn estimated_execution_time(&self) -> Option<std::time::Duration> {
        None // Default implementation
    }
}

/// Unified CoreML executor for all operation types
/// 全演算タイプ用統一CoreML実行器
pub struct CoreMLExecutor {
    device: super::device::CoreMLDevice,
}

impl CoreMLExecutor {
    /// Create new CoreML executor
    /// 新しいCoreML実行器を作成
    pub fn new(device_id: usize) -> CoreMLResult<Self> {
        let device = super::device::CoreMLDevice::new(device_id)?;
        Ok(Self { device })
    }

    /// Execute any CoreML operation
    /// 任意のCoreML演算を実行
    pub fn execute<T, Op>(&self, operation: &Op) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
    {
        if !operation.is_supported_by_coreml() {
            return Err(super::error_helpers::unsupported_operation(
                "Operation not supported by CoreML",
            ));
        }

        operation.execute_coreml(self.device.device_id())
    }

    /// Get device capabilities
    /// デバイス機能を取得
    pub fn capabilities(&self) -> &CoreMLCapabilities {
        self.device.capabilities()
    }
}

/// Macro to implement CoreML fallback pattern
/// CoreMLフォールバックパターン実装用マクロ
#[macro_export]
macro_rules! coreml_fallback {
    ($operation:expr, $cpu_fallback:expr) => {
        coreml_feature! {
            match $operation {
                Ok(result) => Ok(result),
                Err(CoreMLError::UnsupportedOperation(_)) => $cpu_fallback,
                Err(e) => Err(e.into()),
            }
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        $cpu_fallback
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_availability() {
        // Test availability check without panicking
        let available = is_coreml_available();
        println!("CoreML available: {}", available);
    }

    #[test]
    fn test_executor_creation() {
        let result = CoreMLExecutor::new(0);

        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            // On CoreML-enabled builds, creation may succeed or fail depending on platform
            match result {
                Ok(executor) => {
                    assert!(executor.capabilities().supports_f32);
                }
                Err(e) => {
                    // Expected on non-macOS or when CoreML is not available
                    println!("CoreML executor creation failed (expected): {}", e);
                }
            }
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // On non-CoreML builds, should always fail
            assert!(result.is_err());
        }
    }
}
