//! Performance optimizations for distributed learning
//! 分散学習のパフォーマンス最適化
//!
//! This module provides performance optimization utilities including:
//! - Gradient compression algorithms
//! - Memory pooling for tensor operations
//! - Communication scheduling and batching
//! - Zero-copy optimizations

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Gradient compression algorithms
/// 勾配圧縮アルゴリズム
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    /// No compression
    /// 圧縮なし
    None,
    /// Top-K sparsification
    /// Top-K スパース化
    TopK {
        /// Number of top elements to keep
        /// 保持する上位要素数
        k: usize,
    },
    /// Random sparsification
    /// ランダムスパース化
    Random {
        /// Compression ratio (0.0 to 1.0)
        /// 圧縮率（0.0から1.0）
        ratio: f32,
    },
    /// Quantization
    /// 量子化
    Quantization {
        /// Number of bits for quantization
        /// 量子化のビット数
        bits: u8,
    },
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
    pub fn compress(&self, gradient: &Tensor<T>) -> RusTorchResult<CompressedGradient<T>> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(CompressedGradient {
                data: gradient.clone(),
                algorithm: self.algorithm,
                original_shape: gradient.shape().to_vec(),
            }),
            CompressionAlgorithm::TopK { k } => self.compress_top_k(gradient, k),
            CompressionAlgorithm::Random { ratio } => self.compress_random(gradient, ratio),
            CompressionAlgorithm::Quantization { bits } => {
                self.compress_quantization(gradient, bits)
            }
        }
    }

    /// Decompress gradient tensor
    /// 勾配テンソルを展開
    pub fn decompress(&self, compressed: &CompressedGradient<T>) -> RusTorchResult<Tensor<T>> {
        match compressed.algorithm {
            CompressionAlgorithm::None => Ok(compressed.data.clone()),
            _ => {
                // Simplified decompression - in practice would restore original format
                // 簡略化された展開 - 実際には元の形式を復元
                Ok(compressed.data.clone())
            }
        }
    }

    fn compress_top_k(
        &self,
        gradient: &Tensor<T>,
        k: usize,
    ) -> RusTorchResult<CompressedGradient<T>> {
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

    fn compress_random(
        &self,
        gradient: &Tensor<T>,
        ratio: f32,
    ) -> RusTorchResult<CompressedGradient<T>> {
        // Simplified random sparsification
        // 簡略化されたランダムスパース化
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(RusTorchError::ConfigurationError(format!(
                "Invalid compression ratio: {}",
                ratio
            ))
            .into());
        }

        Ok(CompressedGradient {
            data: gradient.clone(),
            algorithm: CompressionAlgorithm::Random { ratio },
            original_shape: gradient.shape().to_vec(),
        })
    }

    fn compress_quantization(
        &self,
        gradient: &Tensor<T>,
        bits: u8,
    ) -> RusTorchResult<CompressedGradient<T>> {
        // Simplified quantization
        // 簡略化された量子化
        if bits == 0 || bits > 32 {
            return Err(RusTorchError::ConfigurationError(format!(
                "Invalid quantization bits: {}",
                bits
            ))
            .into());
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
    /// Compressed gradient data
    /// 圧縮された勾配データ
    pub data: Tensor<T>,
    /// Compression algorithm used
    /// 使用された圧縮アルゴリズム
    pub algorithm: CompressionAlgorithm,
    /// Original tensor shape before compression
    /// 圧縮前の元のテンソル形状
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
    pub fn get_tensor(&self, shape: &[usize]) -> RusTorchResult<Tensor<T>> {
        let mut pools = self.pools.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock memory pool".to_string())
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
    pub fn return_tensor(&self, tensor: Tensor<T>) -> RusTorchResult<()> {
        let mut pools = self.pools.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock memory pool".to_string())
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
    pub fn clear(&self) -> RusTorchResult<()> {
        let mut pools = self.pools.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock memory pool".to_string())
        })?;

        pools.clear();
        Ok(())
    }

    /// Get memory pool statistics
    /// メモリプール統計を取得
    pub fn get_stats(&self) -> RusTorchResult<MemoryPoolStats> {
        let pools = self.pools.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock memory pool".to_string())
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
    /// Total number of tensors in the pool
    /// プール内のテンソル総数
    pub total_tensors: usize,
    /// Number of unique tensor shapes
    /// 固有のテンソル形状数
    pub unique_shapes: usize,
    /// Maximum pool size allowed
    /// 許可される最大プールサイズ
    pub max_pool_size: usize,
}

/// Communication scheduler for batching operations
/// 操作バッチ化用通信スケジューラ
pub struct CommunicationScheduler<T: Float> {
    pending_operations: Arc<Mutex<Vec<PendingOperation<T>>>>,
    batch_size: usize,
}

impl<T: Float + 'static> CommunicationScheduler<T> {
    /// Create new communication scheduler
    /// 新しい通信スケジューラを作成
    pub fn new(batch_size: usize) -> Self {
        Self {
            pending_operations: Arc::new(Mutex::new(Vec::new())),
            batch_size,
        }
    }

    /// Schedule operation for batched execution
    /// バッチ実行用の操作をスケジュール
    pub fn schedule_operation(&self, operation: PendingOperation<T>) -> RusTorchResult<()> {
        let mut ops = self.pending_operations.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock scheduler".to_string())
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
    pub fn flush(&self) -> RusTorchResult<()> {
        let mut ops = self.pending_operations.lock().map_err(|_| {
            RusTorchError::CommunicationError("Failed to lock scheduler".to_string())
        })?;

        if !ops.is_empty() {
            self.execute_batch(&mut ops)?;
        }

        Ok(())
    }

    fn execute_batch(&self, operations: &mut Vec<PendingOperation<T>>) -> RusTorchResult<()> {
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
    /// Type of the communication operation
    /// 通信操作のタイプ
    pub operation_type: OperationType,
    /// Tensor data for the operation
    /// 操作用のテンソルデータ
    pub tensor: Tensor<T>,
    /// Operation metadata
    /// 操作メタデータ
    pub metadata: OperationMetadata,
}

/// Type of communication operation
/// 通信操作の種類
#[derive(Debug, Clone)]
pub enum OperationType {
    /// All-reduce operation (combine values from all processes)
    /// All-reduce操作（全プロセスからの値を結合）
    AllReduce,
    /// All-gather operation (gather data from all processes to all)
    /// All-gather操作（全プロセスから全プロセスにデータを収集）
    AllGather,
    /// Broadcast operation (send data from one process to all)
    /// ブロードキャスト操作（1つのプロセスから全プロセスにデータ送信）
    Broadcast,
    /// Reduce operation (combine values to one process)
    /// リデュース操作（値を1つのプロセスに結合）
    Reduce,
    /// Scatter operation (distribute data from one process to all)
    /// スキャッター操作（1つのプロセスから全プロセスにデータ分散）
    Scatter,
    /// Gather operation (collect data from all processes to one)
    /// ギャザー操作（全プロセスから1つのプロセスにデータ収集）
    Gather,
}

/// Operation metadata
/// 操作メタデータ
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    /// Priority of the operation (0-255, higher values = higher priority)
    /// 操作の優先度（0-255、高い値ほど高い優先度）
    pub priority: u8,
    /// Timestamp when the operation was created
    /// 操作が作成された時のタイムスタンプ
    pub timestamp: u64,
    /// Root rank for operations that require a root process
    /// ルートプロセスが必要な操作のためのルートランク
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
        let scheduler = CommunicationScheduler::<f32>::new(5);

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
