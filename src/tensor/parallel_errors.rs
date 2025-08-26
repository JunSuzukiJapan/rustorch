//! 並列テンソル操作の統一エラーハンドリング
//! Unified error handling for parallel tensor operations - now using RusTorchError only

/// 並列操作の結果型 (統一済み)
/// Result type for parallel operations (統一済み)
pub type ParallelResult<T> = crate::error::RusTorchResult<T>;

// All error creation functionality moved to RusTorchError convenience methods
// 全エラー作成機能はRusTorchErrorの便利メソッドに移行

#[cfg(test)]
mod tests {
    use crate::error::RusTorchError;

    #[test]
    fn test_shape_mismatch_error() {
        let error = RusTorchError::shape_mismatch(&[2, 3], &[3, 2]);
        assert_eq!(
            error.to_string(),
            "Shape mismatch: expected [2, 3], got [3, 2]"
        );
    }

    #[test]
    fn test_tensor_operation_error() {
        let error = RusTorchError::tensor_op("parallel reduction failed");
        assert_eq!(
            error.to_string(),
            "Tensor operation failed: parallel reduction failed"
        );
    }

    #[test]
    fn test_empty_tensor_error() {
        let error = RusTorchError::empty_tensor();
        assert_eq!(
            error.to_string(),
            "Tensor operation failed: Operation on empty tensor"
        );
    }

    #[test]
    fn test_gpu_error() {
        let error = RusTorchError::gpu("SIMD operation error: AVX2 not supported");
        assert_eq!(
            error.to_string(),
            "GPU error: SIMD operation error: AVX2 not supported"
        );
    }

    #[test]
    fn test_device_error() {
        let error = RusTorchError::device("GPU:0", "parallel execution failed");
        assert_eq!(
            error.to_string(),
            "Device error on GPU:0: parallel execution failed"
        );
    }
}
