//! 並列テンソル操作のエラー型定義
//! Error types for parallel tensor operations

use std::fmt;

/// 並列テンソル操作のエラー型
/// Error type for parallel tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelError {
    /// 形状不一致エラー
    /// Shape mismatch error
    ShapeMismatch {
        /// Expected tensor shape
        /// 期待されるテンソル形状
        expected: Vec<usize>,
        /// Actual tensor shape
        /// 実際のテンソル形状
        actual: Vec<usize>,
        /// Operation that caused the error
        /// エラーを引き起こした演算
        operation: String,
    },
    
    /// 次元エラー
    /// Dimension error
    DimensionError {
        /// Invalid dimension index
        /// 無効な次元インデックス
        dimension: usize,
        /// Maximum allowed dimension
        /// 許可される最大次元
        max_dimension: usize,
        /// Operation that caused the error
        /// エラーを引き起こした演算
        operation: String,
    },
    
    /// バッチサイズ不一致エラー
    /// Batch size mismatch error
    BatchSizeMismatch {
        /// Expected batch size
        /// 期待されるバッチサイズ
        expected: usize,
        /// Actual batch size
        /// 実際のバッチサイズ
        actual: usize,
    },
    
    /// 行列乗算の次元不一致エラー
    /// Matrix multiplication dimension mismatch error
    MatmulDimensionMismatch {
        /// Shape of left matrix
        /// 左行列の形状
        left_shape: Vec<usize>,
        /// Shape of right matrix
        /// 右行列の形状
        right_shape: Vec<usize>,
    },
    
    /// 畳み込み演算のパラメータエラー
    /// Convolution parameter error
    ConvolutionError {
        /// Number of input channels
        /// 入力チャンネル数
        input_channels: usize,
        /// Number of kernel channels
        /// カーネルチャンネル数
        kernel_channels: usize,
        /// Error message
        /// エラーメッセージ
        message: String,
    },
    
    /// テンソルの次元数不足エラー
    /// Insufficient tensor dimensions error
    InsufficientDimensions {
        /// Required number of dimensions
        /// 必要な次元数
        required: usize,
        /// Actual number of dimensions
        /// 実際の次元数
        actual: usize,
        /// Operation that caused the error
        /// エラーを引き起こした演算
        operation: String,
    },
    
    /// 空のテンソルリストエラー
    /// Empty tensor list error
    EmptyTensorList {
        /// Operation that caused the error
        /// エラーを引き起こした演算
        operation: String,
    },
    
    /// スカラーテンソルのインデックスエラー
    /// Scalar tensor indexing error
    ScalarIndexing,
    
    /// 並列実行エラー
    /// Parallel execution error
    ParallelExecutionError {
        /// Error message
        /// エラーメッセージ
        message: String,
    },
    
    /// SIMD操作エラー
    /// SIMD operation error
    SimdError {
        /// Error message
        /// エラーメッセージ
        message: String,
    },
    
    /// デバイスエラー
    /// Device error
    DeviceError {
        /// Error message
        /// エラーメッセージ
        message: String,
    },
}

impl fmt::Display for ParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParallelError::ShapeMismatch { expected, actual, operation } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    operation, expected, actual
                )
            }
            ParallelError::DimensionError { dimension, max_dimension, operation } => {
                write!(
                    f,
                    "Dimension {} out of bounds for {} (max: {})",
                    dimension, operation, max_dimension
                )
            }
            ParallelError::BatchSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Batch size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ParallelError::MatmulDimensionMismatch { left_shape, right_shape } => {
                write!(
                    f,
                    "Matrix multiplication dimension mismatch: {:?} × {:?}",
                    left_shape, right_shape
                )
            }
            ParallelError::ConvolutionError { input_channels, kernel_channels, message } => {
                write!(
                    f,
                    "Convolution error: input channels {}, kernel channels {}, {}",
                    input_channels, kernel_channels, message
                )
            }
            ParallelError::InsufficientDimensions { required, actual, operation } => {
                write!(
                    f,
                    "{} requires at least {} dimensions, got {}",
                    operation, required, actual
                )
            }
            ParallelError::EmptyTensorList { operation } => {
                write!(f, "Cannot perform {} on empty tensor list", operation)
            }
            ParallelError::ScalarIndexing => {
                write!(f, "Cannot index into scalar tensor")
            }
            ParallelError::ParallelExecutionError { message } => {
                write!(f, "Parallel execution error: {}", message)
            }
            ParallelError::SimdError { message } => {
                write!(f, "SIMD operation error: {}", message)
            }
            ParallelError::DeviceError { message } => {
                write!(f, "Device error: {}", message)
            }
        }
    }
}

impl std::error::Error for ParallelError {}

/// 並列操作の結果型
/// Result type for parallel operations
pub type ParallelResult<T> = Result<T, ParallelError>;

/// エラー生成のヘルパー関数
/// Helper functions for error generation
impl ParallelError {
    /// 形状不一致エラーを作成
    /// Create shape mismatch error
    pub fn shape_mismatch(
        expected: &[usize],
        actual: &[usize],
        operation: &str,
    ) -> Self {
        ParallelError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
            operation: operation.to_string(),
        }
    }
    
    /// 次元エラーを作成
    /// Create dimension error
    pub fn dimension_error(
        dimension: usize,
        max_dimension: usize,
        operation: &str,
    ) -> Self {
        ParallelError::DimensionError {
            dimension,
            max_dimension,
            operation: operation.to_string(),
        }
    }
    
    /// バッチサイズ不一致エラーを作成
    /// Create batch size mismatch error
    pub fn batch_size_mismatch(expected: usize, actual: usize) -> Self {
        ParallelError::BatchSizeMismatch { expected, actual }
    }
    
    /// 行列乗算次元不一致エラーを作成
    /// Create matrix multiplication dimension mismatch error
    pub fn matmul_dimension_mismatch(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Self {
        ParallelError::MatmulDimensionMismatch {
            left_shape: left_shape.to_vec(),
            right_shape: right_shape.to_vec(),
        }
    }
    
    /// 畳み込みエラーを作成
    /// Create convolution error
    pub fn convolution_error(
        input_channels: usize,
        kernel_channels: usize,
        message: &str,
    ) -> Self {
        ParallelError::ConvolutionError {
            input_channels,
            kernel_channels,
            message: message.to_string(),
        }
    }
    
    /// 次元数不足エラーを作成
    /// Create insufficient dimensions error
    pub fn insufficient_dimensions(
        required: usize,
        actual: usize,
        operation: &str,
    ) -> Self {
        ParallelError::InsufficientDimensions {
            required,
            actual,
            operation: operation.to_string(),
        }
    }
    
    /// 空のテンソルリストエラーを作成
    /// Create empty tensor list error
    pub fn empty_tensor_list(operation: &str) -> Self {
        ParallelError::EmptyTensorList {
            operation: operation.to_string(),
        }
    }
    
    /// 並列実行エラーを作成
    /// Create parallel execution error
    pub fn parallel_execution_error(message: &str) -> Self {
        ParallelError::ParallelExecutionError {
            message: message.to_string(),
        }
    }
    
    /// SIMDエラーを作成
    /// Create SIMD error
    pub fn simd_error(message: &str) -> Self {
        ParallelError::SimdError {
            message: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_error() {
        let error = ParallelError::shape_mismatch(&[2, 3], &[3, 2], "addition");
        assert_eq!(
            error.to_string(),
            "Shape mismatch in addition: expected [2, 3], got [3, 2]"
        );
    }

    #[test]
    fn test_dimension_error() {
        let error = ParallelError::dimension_error(5, 3, "reduction");
        assert_eq!(
            error.to_string(),
            "Dimension 5 out of bounds for reduction (max: 3)"
        );
    }

    #[test]
    fn test_batch_size_mismatch() {
        let error = ParallelError::batch_size_mismatch(32, 16);
        assert_eq!(
            error.to_string(),
            "Batch size mismatch: expected 32, got 16"
        );
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let error = ParallelError::matmul_dimension_mismatch(&[2, 3, 4], &[2, 5, 6]);
        assert_eq!(
            error.to_string(),
            "Matrix multiplication dimension mismatch: [2, 3, 4] × [2, 5, 6]"
        );
    }

    #[test]
    fn test_convolution_error() {
        let error = ParallelError::convolution_error(3, 5, "channel mismatch");
        assert_eq!(
            error.to_string(),
            "Convolution error: input channels 3, kernel channels 5, channel mismatch"
        );
    }

    #[test]
    fn test_insufficient_dimensions() {
        let error = ParallelError::insufficient_dimensions(3, 2, "batch matmul");
        assert_eq!(
            error.to_string(),
            "batch matmul requires at least 3 dimensions, got 2"
        );
    }

    #[test]
    fn test_empty_tensor_list() {
        let error = ParallelError::empty_tensor_list("stack");
        assert_eq!(
            error.to_string(),
            "Cannot perform stack on empty tensor list"
        );
    }

    #[test]
    fn test_scalar_indexing() {
        let error = ParallelError::ScalarIndexing;
        assert_eq!(error.to_string(), "Cannot index into scalar tensor");
    }

    #[test]
    fn test_parallel_execution_error() {
        let error = ParallelError::parallel_execution_error("thread pool exhausted");
        assert_eq!(
            error.to_string(),
            "Parallel execution error: thread pool exhausted"
        );
    }

    #[test]
    fn test_simd_error() {
        let error = ParallelError::simd_error("AVX2 not supported");
        assert_eq!(error.to_string(), "SIMD operation error: AVX2 not supported");
    }
}
