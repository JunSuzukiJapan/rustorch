//! Common CoreML definitions, macros, and utilities
//! 共通CoreML定義、マクロ、ユーティリティ

/// Macro for conditional CoreML compilation
/// CoreML条件付きコンパイル用マクロ
#[macro_export]
macro_rules! coreml_feature {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback"))]
            $item
        )*
    };
    ({ $($stmt:stmt)* }) => {
        #[cfg(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback"))]
        {
            $($stmt)*
        }
    };
}

// Re-export the macro
pub use coreml_feature;

/// Macro for CoreML availability check
/// CoreML可用性チェック用マクロ
#[macro_export]
macro_rules! coreml_available {
    () => {
        cfg!(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )) && cfg!(target_os = "macos")
    };
}

/// Result type for CoreML operations using unified RusTorchError
/// 統一RusTorchErrorを使用するCoreML演算用結果型
pub type CoreMLResult<T> = crate::error::RusTorchResult<T>;

/// CoreML operation types that are supported
/// サポートされているCoreML演算タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreMLOpType {
    /// Matrix multiplication operations
    /// 行列乗算演算
    MatrixMultiplication,

    /// Convolution operations
    /// 畳み込み演算
    Convolution,

    /// Activation functions
    /// 活性化関数
    Activation,

    /// Element-wise operations
    /// 要素ごとの演算
    ElementWise,
}

/// CoreML activation function types
/// CoreML活性化関数タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    LeakyReLU,
    ELU,
    Swish,
}

/// CoreML device capabilities and limitations
/// CoreMLデバイスの機能と制限
#[derive(Debug, Clone)]
pub struct CoreMLCapabilities {
    /// Maximum supported tensor size in bytes
    /// サポートされる最大テンソルサイズ（バイト）
    pub max_tensor_size: usize,

    /// Supports float32 operations
    /// float32演算をサポート
    pub supports_f32: bool,

    /// Supports float64 operations (limited)
    /// float64演算をサポート（制限あり）
    pub supports_f64: bool,

    /// Supports complex number operations
    /// 複素数演算をサポート
    pub supports_complex: bool,

    /// Apple Neural Engine availability
    /// Apple Neural Engine可用性
    pub neural_engine_available: bool,

    /// GPU acceleration availability
    /// GPUアクセラレーション可用性
    pub gpu_acceleration_available: bool,
}

impl Default for CoreMLCapabilities {
    fn default() -> Self {
        Self {
            max_tensor_size: 100 * 1024 * 1024, // 100MB default limit
            supports_f32: true,
            supports_f64: false,               // CoreML limitation
            supports_complex: false,           // CoreML limitation
            neural_engine_available: false,    // Detected at runtime
            gpu_acceleration_available: false, // Detected at runtime
        }
    }
}

/// Standard error messages for consistency
/// 一貫性のための標準エラーメッセージ
pub const COREML_NOT_AVAILABLE: &str = "CoreML not available on this system";
pub const COREML_FEATURE_DISABLED: &str = "CoreML feature not enabled at compile time";
pub const COREML_MACOS_ONLY: &str = "CoreML is only available on macOS";
pub const COREML_UNSUPPORTED_OP: &str = "Operation not supported by CoreML";

/// Helper functions to create consistent CoreML errors
/// 一貫したCoreMLエラーを作成するヘルパー関数
pub mod error_helpers {
    use crate::error::RusTorchError;

    /// Create a CoreML not available error
    /// CoreML利用不可エラーを作成
    pub fn not_available() -> RusTorchError {
        RusTorchError::BackendUnavailable {
            backend: "CoreML".to_string(),
        }
    }

    /// Create a CoreML feature disabled error
    /// CoreML機能無効エラーを作成
    pub fn feature_disabled() -> RusTorchError {
        RusTorchError::BackendUnavailable {
            backend: "CoreML (feature disabled)".to_string(),
        }
    }

    /// Create a CoreML unsupported operation error
    /// CoreML非対応演算エラーを作成
    pub fn unsupported_operation(operation: &str) -> RusTorchError {
        RusTorchError::UnsupportedOperation(format!(
            "{}: {}",
            super::COREML_UNSUPPORTED_OP,
            operation
        ))
    }

    /// Create a CoreML device error
    /// CoreMLデバイスエラーを作成
    pub fn device_error(message: &str) -> RusTorchError {
        RusTorchError::Device {
            device: "CoreML".to_string(),
            message: message.to_string(),
        }
    }

    /// Create a CoreML tensor operation error
    /// CoreMLテンソル演算エラーを作成
    pub fn tensor_op_error(message: &str) -> RusTorchError {
        RusTorchError::TensorOp {
            message: format!("CoreML: {}", message),
            source: None,
        }
    }
}

/// Helper function to check CoreML availability
/// CoreML可用性チェックヘルパー関数
pub fn is_coreml_available() -> bool {
    coreml_available!() && check_runtime_availability()
}

/// Runtime availability check (implementation dependent)
/// ランタイム可用性チェック（実装依存）
fn check_runtime_availability() -> bool {
    #[cfg(all(
        any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ),
        target_os = "macos"
    ))]
    {
        // Actual runtime check would go here
        // 実際のランタイムチェックはここに入る
        use crate::backends::DeviceManager;
        DeviceManager::is_coreml_available()
    }

    #[cfg(not(all(
        any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ),
        target_os = "macos"
    )))]
    false
}
