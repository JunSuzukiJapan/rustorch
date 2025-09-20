//! CoreML operation implementations
//! CoreML演算実装

use super::common::*;
use crate::tensor::Tensor;
use crate::error::RusTorchResult;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

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
pub use linear_algebra::*;
pub use convolution::*;
pub use activation::*;

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
                "Operation not supported by CoreML"
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

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
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

        #[cfg(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback"))]
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

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
        {
            // On non-CoreML builds, should always fail
            assert!(result.is_err());
        }
    }
}