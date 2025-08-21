//! Common utilities for distributed operations
//! 分散操作用共通ユーティリティ

use crate::tensor::Tensor;
use super::{DistributedResult, DistributedError};
use num_traits::Float;

/// Common distributed operation implementations
/// 共通分散操作実装
pub struct CommonOps;

impl CommonOps {
    /// Default all_gather implementation for backends
    /// バックエンド用デフォルトall_gather実装
    pub fn default_all_gather<T: Float + Send + Sync + 'static>(
        tensor: &Tensor<T>,
        world_size: usize,
    ) -> DistributedResult<Vec<Tensor<T>>> {
        // Simplified implementation - in production, this would use actual communication
        // 簡略化実装 - プロダクションでは実際の通信を使用
        Ok(vec![tensor.clone(); world_size])
    }

    /// Default gather implementation for backends
    /// バックエンド用デフォルトgather実装
    pub fn default_gather<T: Float + Send + Sync + 'static>(
        tensor: &Tensor<T>,
        world_size: usize,
        _root: usize,
    ) -> DistributedResult<Vec<Tensor<T>>> {
        // Simplified implementation
        // 簡略化実装
        Ok(vec![tensor.clone(); world_size])
    }

    /// Default broadcast implementation for backends
    /// バックエンド用デフォルトbroadcast実装
    pub fn default_broadcast<T: Float + Send + Sync + 'static>(
        _tensor: &mut Tensor<T>,
        _root: usize,
    ) -> DistributedResult<()> {
        // Simplified implementation
        // 簡略化実装
        Ok(())
    }

    /// Default all_reduce implementation for backends
    /// バックエンド用デフォルトall_reduce実装
    pub fn default_all_reduce<T: Float + Send + Sync + 'static>(
        _tensor: &mut Tensor<T>,
        _op: super::ReduceOp,
    ) -> DistributedResult<()> {
        // Simplified implementation
        // 簡略化実装
        Ok(())
    }

    /// Validate tensor for distributed operations
    /// 分散操作用のテンソル検証
    pub fn validate_tensor<T: Float + 'static>(tensor: &Tensor<T>) -> DistributedResult<()> {
        let shape = tensor.shape();
        if shape.is_empty() {
            return Err(DistributedError::CommunicationError(
                "Empty tensor shape".to_string()
            ));
        }
        
        // Check for zero-sized dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(DistributedError::TensorShapeMismatch {
                expected: vec![1], // Minimum expected shape
                actual: shape.to_vec(),
            });
        }
        
        // Check for reasonable tensor size (prevent memory issues)
        let total_elements: usize = shape.iter().product();
        const MAX_ELEMENTS: usize = 1_000_000_000; // 1B elements max
        if total_elements > MAX_ELEMENTS {
            return Err(DistributedError::CommunicationError(
                format!("Tensor too large: {} elements exceeds maximum {}", 
                       total_elements, MAX_ELEMENTS)
            ));
        }
        
        Ok(())
    }
    
    /// Validate rank for distributed operations
    /// 分散操作用のランク検証
    pub fn validate_rank(rank: usize, world_size: usize) -> DistributedResult<()> {
        if rank >= world_size {
            return Err(DistributedError::InvalidRank { rank, world_size });
        }
        Ok(())
    }
    
    /// Validate tensor shapes match across processes
    /// プロセス間でのテンソル形状一致検証
    pub fn validate_tensor_shapes<T: Float + 'static>(
        tensors: &[Tensor<T>], 
        expected_shape: &[usize]
    ) -> DistributedResult<()> {
        for (_i, tensor) in tensors.iter().enumerate() {
            let actual_shape = tensor.shape();
            if actual_shape != expected_shape {
                return Err(DistributedError::TensorShapeMismatch {
                    expected: expected_shape.to_vec(),
                    actual: actual_shape.to_vec(),
                });
            }
        }
        Ok(())
    }

    /// Create error for unsupported operations
    /// サポートされていない操作のエラー作成
    pub fn unsupported_operation_error(operation: &str, backend: &str) -> DistributedError {
        DistributedError::BackendNotAvailable(
            format!("Operation '{}' not supported by backend '{}'", operation, backend)
        )
    }
}

/// Trait for backend-specific optimizations
/// バックエンド固有の最適化トレイト
pub trait BackendOptimizations<T: Float> {
    /// Enable gradient compression
    /// 勾配圧縮を有効化
    fn enable_gradient_compression(&self) -> bool {
        false
    }
    
    /// Get optimal bucket size for gradient bucketing
    /// 勾配バケット化の最適バケットサイズ取得
    fn optimal_bucket_size(&self) -> usize {
        25 * 1024 * 1024 // 25MB default
    }
    
    /// Check if backend supports async operations
    /// バックエンドが非同期操作をサポートするかチェック
    fn supports_async_ops(&self) -> bool {
        false
    }
    
    /// Get memory pool size for tensor operations
    /// テンソル操作用のメモリプールサイズ取得
    fn memory_pool_size(&self) -> usize {
        512 * 1024 * 1024 // 512MB default
    }
    
    /// Enable zero-copy optimizations
    /// ゼロコピー最適化を有効化
    fn enable_zero_copy(&self) -> bool {
        true
    }
    
    /// Get optimal number of communication streams
    /// 最適な通信ストリーム数取得
    fn optimal_stream_count(&self) -> usize {
        4
    }
    
    /// Enable pipeline parallelism for large tensors
    /// 大きなテンソル用のパイプライン並列化を有効化
    fn enable_pipeline_parallelism(&self) -> bool {
        true
    }
    
    /// Optimize tensor for communication
    /// 通信用テンソル最適化
    fn optimize_for_communication(&self, tensor: &mut Tensor<T>) -> DistributedResult<()> {
        // Default implementation does nothing
        // デフォルト実装は何もしない
        let _ = tensor;
        Ok(())
    }

    /// Get optimal chunk size for communication
    /// 通信用最適チャンクサイズ取得
    fn get_optimal_chunk_size(&self, tensor_size: usize) -> usize {
        // Default chunk size based on tensor size
        // テンソルサイズに基づくデフォルトチャンクサイズ
        (tensor_size / 4).max(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_default_all_gather() {
        let tensor: Tensor<f32> = Tensor::ones(&[2, 2]);
        let result = CommonOps::default_all_gather(&tensor, 4);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_validate_tensor() {
        let valid_tensor: Tensor<f32> = Tensor::ones(&[2, 2]);
        assert!(CommonOps::validate_tensor(&valid_tensor).is_ok());
        let empty_tensor: Tensor<f32> = Tensor::zeros(&[0]);
        assert!(CommonOps::validate_tensor(&empty_tensor).is_err());
    }

    #[test]
    fn test_unsupported_operation_error() {
        let error = CommonOps::unsupported_operation_error("test_op", "test_backend");
        match error {
            DistributedError::BackendNotAvailable(msg) => {
                assert!(msg.contains("test_op"));
                assert!(msg.contains("test_backend"));
            }
            _ => panic!("Expected BackendNotAvailable error"),
        }
    }
}
