//! Performance optimizations for distributed learning
//! 分散学習のパフォーマンス最適化
//! 
//! This module provides performance optimization utilities including:
//! - Gradient compression algorithms
//! - Memory pooling for tensor operations
//! - Communication scheduling and batching
//! - Zero-copy optimizations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::tensor::Tensor;
use super::{DistributedResult, DistributedError};
use num_traits::Float;

/// Gradient compression algorithms
/// 勾配圧縮アルゴリズム
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    /// No compression
    /// 圧縮なし
    None,
    /// Top-K sparsification
    /// Top-K スパース化
    TopK { k: usize },
    /// Random sparsification
    /// ランダムスパース化
    Random { ratio: f32 },
    /// Quantization
    /// 量子化
    Quantization { bits: u8 },
}

/// Gradient compressor for reducing communication overhead
/// 通信オーバーヘッド削減用勾配圧縮器
pub struct GradientCompressor<T: Float> {
    algorithm: CompressionAlgorithm,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> GradientCompressor<T> {
    /// Create new gradient compressor
    /// 新しい勾配圧縮器を作成
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Compress gradient tensor
    /// 勾配テンソルを圧縮
    pub fn compress(&self, gradient: &Tensor<T>) -> DistributedResult<CompressedGradient<T>> {
        match self.algorithm {
            CompressionAlgorithm::None => {
                Ok(CompressedGradient {
                    data: gradient.clone(),
                    algorithm: self.algorithm,
                    original_shape: gradient.shape().to_vec(),
                })
            },
            CompressionAlgorithm::TopK { k } => {
                self.compress_top_k(gradient, k)
            },
            CompressionAlgorithm::Random { ratio } => {
                self.compress_random(gradient, ratio)
            },
            CompressionAlgorithm::Quantization { bits } => {
                self.compress_quantization(gradient, bits)
            },
        }
    }
    
    /// Decompress gradient tensor
    /// 勾配テンソルを展開
    pub fn decompress(&self, compressed: &CompressedGradient<T>) -> DistributedResult<Tensor<T>> {
        match compressed.algorithm {
            CompressionAlgorithm::None => Ok(compressed.data.clone()),
            _ => {
                // Simplified decompression - in practice would restore original format
                // 簡略化された展開 - 実際には元の形式を復元
                Ok(compressed.data.clone())
            }
        }
    }
    
    fn compress_top_k(&self, gradient: &Tensor<T>, k: usize) -> DistributedResult<CompressedGradient<T>> {
        // Simplified top-k compression
        // 簡略化されたtop-k圧縮
        let total_elements = gradient.shape().iter().product::<usize>();
        let actual_k = k.min(total_elements);
        
        // In practice, would select top-k elements by magnitude
        // 実際には、大きさでtop-k要素を選択
        Ok(CompressedGradient {
            data: gradient.clone(),
            algorithm: CompressionAlgorithm::TopK { k: actual_k },
            original_shape: gradient.shape().to_vec(),
        })
    }
    
    fn compress_random(&self, gradient: &Tensor<T>, ratio: f32) -> DistributedResult<CompressedGradient<T>> {
        // Simplified random sparsification
        // 簡略化されたランダムスパース化
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(DistributedError::ConfigurationError(
                format!("Invalid compression ratio: {}", ratio)
            ));
        }
        
        Ok(CompressedGradient {
            data: gradient.clone(),
            algorithm: CompressionAlgorithm::Random { ratio },
            original_shape: gradient.shape().to_vec(),
        })
    }
    
    fn compress_quantization(&self, gradient: &Tensor<T>, bits: u8) -> DistributedResult<CompressedGradient<T>> {
        // Simplified quantization
        // 簡略化された量子化
        if bits == 0 || bits > 32 {
            return Err(DistributedError::ConfigurationError(
                format!("Invalid quantization bits: {}", bits)
            ));
        }
        
        Ok(CompressedGradient {
            data: gradient.clone(),
            algorithm: CompressionAlgorithm::Quantization { bits },
            original_shape: gradient.shape().to_vec(),
        })
    }
}

/// Compressed gradient representation
/// 圧縮勾配表現
#[derive(Debug, Clone)]
pub struct CompressedGradient<T: Float> {
    pub data: Tensor<T>,
    pub algorithm: CompressionAlgorithm,
    pub original_shape: Vec<usize>,
}

/// Memory pool for efficient tensor allocation
/// 効率的なテンソル割り当て用メモリプール
pub struct TensorMemoryPool<T: Float> {
    pools: Arc<Mutex<HashMap<Vec<usize>, Vec<Tensor<T>>>>>,
    max_pool_size: usize,
}

impl<T: Float + 'static> TensorMemoryPool<T> {
    /// Create new memory pool
    /// 新しいメモリプールを作成
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_pool_size,
        }
    }
    
    /// Get tensor from pool or allocate new one
    /// プールからテンソルを取得または新規割り当て
    pub fn get_tensor(&self, shape: &[usize]) -> DistributedResult<Tensor<T>> {
        let mut pools = self.pools.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock memory pool".to_string())
        })?;
        
        let shape_vec = shape.to_vec();
        if let Some(pool) = pools.get_mut(&shape_vec) {
            if let Some(tensor) = pool.pop() {
                return Ok(tensor);
            }
        }
        
        // Allocate new tensor if pool is empty
        // プールが空の場合は新しいテンソルを割り当て
        Ok(Tensor::zeros(shape))
    }
    
    /// Return tensor to pool
    /// テンソルをプールに返却
    pub fn return_tensor(&self, tensor: Tensor<T>) -> DistributedResult<()> {
        let mut pools = self.pools.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock memory pool".to_string())
        })?;
        
        let shape = tensor.shape().to_vec();
        let pool = pools.entry(shape).or_insert_with(Vec::new);
        
        if pool.len() < self.max_pool_size {
            pool.push(tensor);
        }
        // If pool is full, tensor is dropped automatically
        // プールが満杯の場合、テンソルは自動的にドロップ
        
        Ok(())
    }
    
    /// Clear all pools
    /// 全プールをクリア
    pub fn clear(&self) -> DistributedResult<()> {
        let mut pools = self.pools.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock memory pool".to_string())
        })?;
        
        pools.clear();
        Ok(())
    }
    
    /// Get memory pool statistics
    /// メモリプール統計を取得
    pub fn get_stats(&self) -> DistributedResult<MemoryPoolStats> {
        let pools = self.pools.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock memory pool".to_string())
        })?;
        
        let total_tensors = pools.values().map(|pool| pool.len()).sum();
        let unique_shapes = pools.len();
        
        Ok(MemoryPoolStats {
            total_tensors,
            unique_shapes,
            max_pool_size: self.max_pool_size,
        })
    }
}

/// Memory pool statistics
/// メモリプール統計
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_tensors: usize,
    pub unique_shapes: usize,
    pub max_pool_size: usize,
}

/// Communication scheduler for batching operations
/// 操作バッチ化用通信スケジューラ
pub struct CommunicationScheduler<T: Float> {
    pending_operations: Arc<Mutex<Vec<PendingOperation<T>>>>,
    batch_size: usize,
    timeout_ms: u64,
}

impl<T: Float + 'static> CommunicationScheduler<T> {
    /// Create new communication scheduler
    /// 新しい通信スケジューラを作成
    pub fn new(batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            pending_operations: Arc::new(Mutex::new(Vec::new())),
            batch_size,
            timeout_ms,
        }
    }
    
    /// Schedule operation for batched execution
    /// バッチ実行用の操作をスケジュール
    pub fn schedule_operation(&self, operation: PendingOperation<T>) -> DistributedResult<()> {
        let mut ops = self.pending_operations.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock scheduler".to_string())
        })?;
        
        ops.push(operation);
        
        // Execute batch if size threshold reached
        // サイズ閾値に達した場合はバッチを実行
        if ops.len() >= self.batch_size {
            self.execute_batch(&mut ops)?;
        }
        
        Ok(())
    }
    
    /// Force execution of pending operations
    /// 保留中の操作を強制実行
    pub fn flush(&self) -> DistributedResult<()> {
        let mut ops = self.pending_operations.lock().map_err(|_| {
            DistributedError::CommunicationError("Failed to lock scheduler".to_string())
        })?;
        
        if !ops.is_empty() {
            self.execute_batch(&mut ops)?;
        }
        
        Ok(())
    }
    
    fn execute_batch(&self, operations: &mut Vec<PendingOperation<T>>) -> DistributedResult<()> {
        // Simplified batch execution
        // 簡略化されたバッチ実行
        for _op in operations.drain(..) {
            // In practice, would execute batched communication operations
            // 実際には、バッチ化された通信操作を実行
        }
        Ok(())
    }
}

/// Pending communication operation
/// 保留中の通信操作
#[derive(Debug)]
pub struct PendingOperation<T: Float> {
    pub operation_type: OperationType,
    pub tensor: Tensor<T>,
    pub metadata: OperationMetadata,
}

/// Type of communication operation
/// 通信操作の種類
#[derive(Debug, Clone)]
pub enum OperationType {
    AllReduce,
    AllGather,
    Broadcast,
    Reduce,
    Scatter,
    Gather,
}

/// Operation metadata
/// 操作メタデータ
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    pub priority: u8,
    pub timestamp: u64,
    pub root_rank: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_compressor() {
        let compressor = GradientCompressor::<f32>::new(CompressionAlgorithm::None);
        let gradient: Tensor<f32> = Tensor::ones(&[2, 2]);
        
        let compressed = compressor.compress(&gradient).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(gradient.shape(), decompressed.shape());
    }
    
    #[test]
    fn test_memory_pool() {
        let pool = TensorMemoryPool::<f32>::new(10);
        let shape = &[2, 2];
        
        let tensor1 = pool.get_tensor(shape).unwrap();
        pool.return_tensor(tensor1).unwrap();
        
        let tensor2 = pool.get_tensor(shape).unwrap();
        assert_eq!(tensor2.shape(), shape);
        
        let stats = pool.get_stats().unwrap();
        assert_eq!(stats.max_pool_size, 10);
    }
    
    #[test]
    fn test_communication_scheduler() {
        let scheduler = CommunicationScheduler::<f32>::new(5, 1000);
        
        let operation = PendingOperation {
            operation_type: OperationType::AllReduce,
            tensor: Tensor::ones(&[2, 2]),
            metadata: OperationMetadata {
                priority: 1,
                timestamp: 0,
                root_rank: None,
            },
        };
        
        assert!(scheduler.schedule_operation(operation).is_ok());
        assert!(scheduler.flush().is_ok());
    }
}
