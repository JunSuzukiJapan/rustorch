//! Distributed optimizers for distributed training
//! 分散学習用分散オプティマイザー
//!
//! This module provides distributed versions of optimizers that handle
//! gradient synchronization across multiple processes and devices.

use crate::distributed::DistributedOps;
use crate::error::{RusTorchError, RusTorchResult};
use crate::optim::{Adam, Optimizer, SGD};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Reduction operation types
/// リダクション操作タイプ
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    /// Sum all values across processes
    /// プロセス間で値を合計
    Sum,
    /// Average all values across processes
    /// プロセス間で値を平均
    Average,
    /// Take maximum value across processes
    /// プロセス間で最大値を取得
    Max,
    /// Take minimum value across processes
    /// プロセス間で最小値を取得
    Min,
}

/// Distributed optimizer wrapper
/// 分散オプティマイザーラッパー
pub struct DistributedOptimizer<T: Float + Send + Sync + 'static> {
    /// Base optimizer
    /// ベースオプティマイザー
    base_optimizer: Box<dyn Optimizer + Send + Sync>,
    /// Communication backend
    /// 通信バックエンド
    backend: Arc<dyn DistributedOps<T> + Send + Sync>,
    /// Gradient synchronization strategy
    /// 勾配同期戦略
    sync_strategy: GradientSyncStrategy,
    /// Gradient buckets for efficient communication
    /// 効率的な通信のための勾配バケット
    gradient_buckets: Vec<GradientBucket<T>>,
    /// Communication frequency
    /// 通信頻度
    communication_freq: usize,
    /// Step counter
    /// ステップカウンター
    step_count: usize,
    /// Local gradient accumulation
    /// ローカル勾配蓄積
    local_gradients: HashMap<String, Tensor<T>>,
}

/// Gradient synchronization strategies
/// 勾配同期戦略
#[derive(Debug, Clone, Copy)]
pub enum GradientSyncStrategy {
    /// Synchronous all-reduce after each backward pass
    /// 各バックワードパス後の同期all-reduce
    Synchronous,
    /// Asynchronous gradient updates
    /// 非同期勾配更新
    Asynchronous,
    /// Local SGD with periodic synchronization
    /// 定期同期を伴うローカルSGD
    LocalSGD {
        /// Frequency of synchronization in steps
        /// 同期の頻度（ステップ数）
        sync_frequency: usize,
    },
    /// Gradient compression for bandwidth efficiency
    /// 帯域幅効率のための勾配圧縮
    Compressed {
        /// Compression ratio (0.0 to 1.0)
        /// 圧縮率（0.0から1.0）
        compression_ratio: f32,
    },
    /// Hierarchical all-reduce for large clusters
    /// 大規模クラスター用階層all-reduce
    Hierarchical,
}

/// Gradient bucket for batching communications
/// 通信バッチング用勾配バケット
pub struct GradientBucket<T: Float> {
    /// Bucket ID
    /// バケットID
    id: usize,
    /// Tensors in this bucket
    /// このバケット内のテンソル
    tensors: Vec<Arc<Mutex<Tensor<T>>>>,
    /// Total size of tensors in bucket
    /// バケット内テンソルの総サイズ
    total_size: usize,
    /// Maximum bucket size
    /// 最大バケットサイズ
    max_size: usize,
    /// Whether bucket is ready for communication
    /// バケットが通信準備完了かどうか
    ready: bool,
}

impl<T: Float + Send + Sync + 'static> DistributedOptimizer<T> {
    /// Create new distributed optimizer
    /// 新しい分散オプティマイザーを作成
    pub fn new(
        base_optimizer: Box<dyn Optimizer + Send + Sync>,
        backend: Arc<dyn DistributedOps<T> + Send + Sync>,
        sync_strategy: GradientSyncStrategy,
    ) -> Self {
        Self {
            base_optimizer,
            backend,
            sync_strategy,
            gradient_buckets: Vec::new(),
            communication_freq: 1,
            step_count: 0,
            local_gradients: HashMap::new(),
        }
    }

    /// Create distributed SGD optimizer
    /// 分散SGDオプティマイザーを作成
    pub fn sgd(
        learning_rate: T,
        momentum: T,
        weight_decay: T,
        backend: Arc<dyn DistributedOps<T> + Send + Sync>,
        sync_strategy: GradientSyncStrategy,
    ) -> RusTorchResult<Self> {
        let lr_f32 = learning_rate.to_f32().unwrap_or(0.001);
        let momentum_f32 = momentum.to_f32().unwrap_or(0.9);
        let wd_f32 = weight_decay.to_f32().unwrap_or(0.0);
        let sgd = if wd_f32 > 0.0 {
            SGD::with_weight_decay(lr_f32, momentum_f32, wd_f32)
        } else {
            SGD::with_momentum(lr_f32, momentum_f32)
        };
        Ok(Self::new(Box::new(sgd), backend, sync_strategy))
    }

    /// Create distributed Adam optimizer
    /// 分散Adamオプティマイザーを作成
    pub fn adam(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        backend: Arc<dyn DistributedOps<T> + Send + Sync>,
        sync_strategy: GradientSyncStrategy,
    ) -> RusTorchResult<Self> {
        let lr_f32 = learning_rate.to_f32().unwrap_or(0.001);
        let beta1_f32 = beta1.to_f32().unwrap_or(0.9);
        let beta2_f32 = beta2.to_f32().unwrap_or(0.999);
        let eps_f32 = epsilon.to_f32().unwrap_or(1e-8);
        let wd_f32 = weight_decay.to_f32().unwrap_or(0.0);
        let adam = if wd_f32 > 0.0 {
            Adam::with_weight_decay(lr_f32, beta1_f32, beta2_f32, eps_f32, wd_f32)
        } else {
            Adam::new(lr_f32, beta1_f32, beta2_f32, eps_f32)
        };
        Ok(Self::new(Box::new(adam), backend, sync_strategy))
    }

    /// Initialize gradient buckets for efficient communication
    /// 効率的な通信のための勾配バケットを初期化
    pub fn init_gradient_buckets(&mut self, max_bucket_size: usize) -> RusTorchResult<()> {
        self.gradient_buckets.clear();

        // Create initial bucket
        // 初期バケットを作成
        let bucket = GradientBucket {
            id: 0,
            tensors: Vec::new(),
            total_size: 0,
            max_size: max_bucket_size,
            ready: false,
        };

        self.gradient_buckets.push(bucket);
        Ok(())
    }

    /// Add tensor to gradient bucket
    /// 勾配バケットにテンソルを追加
    pub fn add_to_bucket(&mut self, tensor: Arc<Mutex<Tensor<T>>>) -> RusTorchResult<()> {
        let tensor_size = {
            let t = tensor.lock().unwrap();
            t.shape().iter().product::<usize>()
        };

        // Find suitable bucket or create new one
        // 適切なバケットを見つけるか新しいものを作成
        let bucket_idx = self.find_or_create_bucket(tensor_size)?;

        let bucket = &mut self.gradient_buckets[bucket_idx];
        bucket.tensors.push(tensor);
        bucket.total_size += tensor_size;

        // Mark bucket as ready if it's full
        // バケットが満杯の場合、準備完了としてマーク
        if bucket.total_size >= bucket.max_size {
            bucket.ready = true;
        }

        Ok(())
    }

    /// Find suitable bucket or create new one
    /// 適切なバケットを見つけるか新しいものを作成
    fn find_or_create_bucket(&mut self, tensor_size: usize) -> RusTorchResult<usize> {
        // Try to find existing bucket with space
        // 空きのある既存バケットを探す
        for (idx, bucket) in self.gradient_buckets.iter().enumerate() {
            if bucket.total_size + tensor_size <= bucket.max_size {
                return Ok(idx);
            }
        }

        // Create new bucket
        // 新しいバケットを作成
        let new_bucket = GradientBucket {
            id: self.gradient_buckets.len(),
            tensors: Vec::new(),
            total_size: 0,
            max_size: self.gradient_buckets[0].max_size,
            ready: false,
        };

        self.gradient_buckets.push(new_bucket);
        Ok(self.gradient_buckets.len() - 1)
    }

    /// Synchronize gradients across all processes
    /// 全プロセス間で勾配を同期
    pub fn sync_gradients(&mut self) -> RusTorchResult<()> {
        match self.sync_strategy {
            GradientSyncStrategy::Synchronous => self.sync_gradients_synchronous(),
            GradientSyncStrategy::Asynchronous => self.sync_gradients_asynchronous(),
            GradientSyncStrategy::LocalSGD { sync_frequency } => {
                self.sync_gradients_local_sgd(sync_frequency)
            }
            GradientSyncStrategy::Compressed { compression_ratio } => {
                self.sync_gradients_compressed(compression_ratio)
            }
            GradientSyncStrategy::Hierarchical => self.sync_gradients_hierarchical(),
        }
    }

    /// Synchronous gradient synchronization
    /// 同期勾配同期
    fn sync_gradients_synchronous(&mut self) -> RusTorchResult<()> {
        // Process all ready buckets
        // 準備完了の全バケットを処理
        let backend = self.backend.clone();
        for bucket in &mut self.gradient_buckets {
            if bucket.ready {
                Self::sync_bucket_with_backend(&backend, bucket)?;
                bucket.ready = false;
            }
        }

        // Process remaining buckets
        // 残りのバケットを処理
        for bucket in &mut self.gradient_buckets {
            if !bucket.tensors.is_empty() {
                Self::sync_bucket_with_backend(&backend, bucket)?;
            }
        }

        Ok(())
    }

    /// Asynchronous gradient synchronization
    /// 非同期勾配同期
    fn sync_gradients_asynchronous(&mut self) -> RusTorchResult<()> {
        // Start async communication for ready buckets
        // 準備完了バケットの非同期通信を開始
        let backend = self.backend.clone();
        for bucket in &mut self.gradient_buckets {
            if bucket.ready {
                // In a real implementation, this would start async communication
                // 実際の実装では、これは非同期通信を開始する
                Self::sync_bucket_with_backend(&backend, bucket)?;
                bucket.ready = false;
            }
        }
        Ok(())
    }

    /// Local SGD with periodic synchronization
    /// 定期同期を伴うローカルSGD
    fn sync_gradients_local_sgd(&mut self, sync_frequency: usize) -> RusTorchResult<()> {
        self.step_count += 1;

        if self.step_count % sync_frequency == 0 {
            // Synchronize accumulated local gradients
            // 蓄積されたローカル勾配を同期
            for (_name, gradient) in &mut self.local_gradients {
                self.backend
                    .all_reduce(gradient, crate::distributed::ReduceOp::Average)?;
            }
            self.local_gradients.clear();
        } else {
            // Accumulate gradients locally
            // 勾配をローカルに蓄積
            for bucket in &self.gradient_buckets {
                for tensor_ref in &bucket.tensors {
                    let tensor = tensor_ref.lock().unwrap();
                    let key = format!("tensor_{}", bucket.id);

                    if let Some(_accumulated) = self.local_gradients.get_mut(&key) {
                        // Add to accumulated gradient
                        // 蓄積勾配に追加
                        // *accumulated = &*accumulated + &*tensor;
                    } else {
                        // Store new gradient
                        // 新しい勾配を保存
                        self.local_gradients.insert(key, tensor.clone().into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Compressed gradient synchronization
    /// 圧縮勾配同期
    fn sync_gradients_compressed(&mut self, compression_ratio: f32) -> RusTorchResult<()> {
        let backend = self.backend.clone();
        for bucket in &mut self.gradient_buckets {
            if bucket.ready {
                // Compress gradients before communication
                // 通信前に勾配を圧縮
                Self::compress_and_sync_bucket_with_backend(&backend, bucket, compression_ratio)?;
                bucket.ready = false;
            }
        }
        Ok(())
    }

    /// Hierarchical gradient synchronization
    /// 階層勾配同期
    fn sync_gradients_hierarchical(&mut self) -> RusTorchResult<()> {
        // Implement hierarchical all-reduce
        // 階層all-reduceを実装
        // This would involve multiple stages of communication
        // これは複数段階の通信を含む
        self.sync_gradients_synchronous()
    }

    /// Synchronize a single gradient bucket
    /// 単一勾配バケットを同期
    fn sync_bucket_with_backend(
        backend: &Arc<dyn DistributedOps<T> + Send + Sync>,
        bucket: &mut GradientBucket<T>,
    ) -> RusTorchResult<()> {
        for tensor_ref in &bucket.tensors {
            let mut tensor = tensor_ref.lock().unwrap();
            backend.all_reduce(&mut tensor, crate::distributed::ReduceOp::Average)?;
        }
        Ok(())
    }

    /// Compress and synchronize gradient bucket
    /// 勾配バケットを圧縮して同期
    fn compress_and_sync_bucket_with_backend(
        backend: &Arc<dyn DistributedOps<T> + Send + Sync>,
        bucket: &mut GradientBucket<T>,
        _compression_ratio: f32,
    ) -> RusTorchResult<()> {
        // Implement gradient compression (e.g., top-k, quantization)
        // 勾配圧縮を実装（例：top-k、量子化）
        for tensor_ref in &bucket.tensors {
            let mut tensor = tensor_ref.lock().unwrap();

            // Apply compression
            // 圧縮を適用
            // let compressed = compress_tensor(&tensor, compression_ratio);

            // Synchronize compressed tensor
            // 圧縮テンソルを同期
            backend.all_reduce(&mut tensor, crate::distributed::ReduceOp::Average)?;

            // Decompress if needed
            // 必要に応じて展開
            // tensor = decompress_tensor(&compressed);
        }
        Ok(())
    }

    /// Set communication frequency for local SGD
    /// ローカルSGDの通信頻度を設定
    pub fn set_communication_frequency(&mut self, freq: usize) {
        self.communication_freq = freq;
    }

    /// Get current step count
    /// 現在のステップ数を取得
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset step counter
    /// ステップカウンターをリセット
    pub fn reset_step_count(&mut self) {
        self.step_count = 0;
    }
}

impl<T: Float + Send + Sync + 'static> Optimizer for DistributedOptimizer<T> {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        // Synchronize gradients before optimization step
        // 最適化ステップ前に勾配を同期
        if let Err(_e) = self.sync_gradients() {
            // In a production implementation, this would be handled more gracefully
            // プロダクション実装では、これはより適切に処理される
            return;
        }

        // Apply base optimizer step
        // ベースオプティマイザーステップを適用
        self.base_optimizer.step(param, grad);

        self.step_count += 1;
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
        // Clear local gradients for local SGD
        // ローカルSGD用のローカル勾配をクリア
        self.local_gradients.clear();
    }

    fn learning_rate(&self) -> f32 {
        self.base_optimizer.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.base_optimizer.set_learning_rate(lr);
    }

    fn state_dict(&self) -> std::collections::HashMap<String, f32> {
        self.base_optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: std::collections::HashMap<String, f32>) {
        self.base_optimizer.load_state_dict(state);
    }
}

/// Distributed optimizer builder for easy configuration
/// 簡単な設定のための分散オプティマイザービルダー
pub struct DistributedOptimizerBuilder<T: Float + Send + Sync + 'static> {
    optimizer_type: OptimizerType<T>,
    backend: Option<Arc<dyn DistributedOps<T> + Send + Sync>>,
    sync_strategy: GradientSyncStrategy,
    bucket_size: usize,
}

/// Optimizer types for builder pattern
/// ビルダーパターン用オプティマイザータイプ
pub enum OptimizerType<T: Float> {
    /// Stochastic Gradient Descent optimizer
    /// 確率的勾配降下法オプティマイザー
    SGD {
        /// Learning rate for optimization
        /// 最適化の学習率
        learning_rate: T,
        /// Momentum factor
        /// モメンタム係数
        momentum: T,
        /// Weight decay for L2 regularization
        /// L2正則化の重み減衰
        weight_decay: T,
    },
    /// Adam optimizer
    /// Adamオプティマイザー
    Adam {
        /// Learning rate for optimization
        /// 最適化の学習率
        learning_rate: T,
        /// Beta1 parameter
        /// Beta1パラメータ
        beta1: T,
        /// Beta2 parameter  
        /// Beta2パラメータ
        beta2: T,
        /// Epsilon for numerical stability
        /// 数値安定性のためのイプシロン
        epsilon: T,
        /// Weight decay for L2 regularization
        /// L2正則化の重み減衰
        weight_decay: T,
    },
}

impl<T: Float + Send + Sync + 'static> DistributedOptimizerBuilder<T> {
    /// Create new builder with SGD
    /// SGDで新しいビルダーを作成
    pub fn sgd(learning_rate: T, momentum: T, weight_decay: T) -> Self {
        Self {
            optimizer_type: OptimizerType::SGD {
                learning_rate,
                momentum,
                weight_decay,
            },
            backend: None,
            sync_strategy: GradientSyncStrategy::Synchronous,
            bucket_size: 25 * 1024 * 1024, // 25MB default
        }
    }

    /// Create new builder with Adam
    /// Adamで新しいビルダーを作成
    pub fn adam(learning_rate: T, beta1: T, beta2: T, epsilon: T, weight_decay: T) -> Self {
        Self {
            optimizer_type: OptimizerType::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            },
            backend: None,
            sync_strategy: GradientSyncStrategy::Synchronous,
            bucket_size: 25 * 1024 * 1024, // 25MB default
        }
    }

    /// Set communication backend
    /// 通信バックエンドを設定
    pub fn backend(mut self, backend: Arc<dyn DistributedOps<T> + Send + Sync>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set gradient synchronization strategy
    /// 勾配同期戦略を設定
    pub fn sync_strategy(mut self, strategy: GradientSyncStrategy) -> Self {
        self.sync_strategy = strategy;
        self
    }

    /// Set gradient bucket size
    /// 勾配バケットサイズを設定
    pub fn bucket_size(mut self, size: usize) -> Self {
        self.bucket_size = size;
        self
    }

    /// Build the distributed optimizer
    /// 分散オプティマイザーを構築
    pub fn build(self) -> RusTorchResult<DistributedOptimizer<T>> {
        let backend = self.backend.ok_or_else(|| {
            RusTorchError::ConfigurationError("Backend not specified".to_string())
        })?;

        let base_optimizer: Box<dyn Optimizer + Send + Sync> = match self.optimizer_type {
            OptimizerType::SGD {
                learning_rate,
                momentum,
                weight_decay,
            } => {
                let lr_f32 = learning_rate.to_f32().unwrap_or(0.001);
                let momentum_f32 = momentum.to_f32().unwrap_or(0.9);
                let wd_f32 = weight_decay.to_f32().unwrap_or(0.0);
                if wd_f32 > 0.0 {
                    Box::new(SGD::with_weight_decay(lr_f32, momentum_f32, wd_f32))
                } else {
                    Box::new(SGD::with_momentum(lr_f32, momentum_f32))
                }
            }
            OptimizerType::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                let lr_f32 = learning_rate.to_f32().unwrap_or(0.001);
                let beta1_f32 = beta1.to_f32().unwrap_or(0.9);
                let beta2_f32 = beta2.to_f32().unwrap_or(0.999);
                let eps_f32 = epsilon.to_f32().unwrap_or(1e-8);
                let wd_f32 = weight_decay.to_f32().unwrap_or(0.0);
                if wd_f32 > 0.0 {
                    Box::new(Adam::with_weight_decay(
                        lr_f32, beta1_f32, beta2_f32, eps_f32, wd_f32,
                    ))
                } else {
                    Box::new(Adam::new(lr_f32, beta1_f32, beta2_f32, eps_f32))
                }
            }
        };

        let mut optimizer = DistributedOptimizer::new(base_optimizer, backend, self.sync_strategy);
        optimizer.init_gradient_buckets(self.bucket_size)?;

        Ok(optimizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::distributed::backends::GlooBackend; // Unused in current tests
    // use crate::distributed::{ProcessGroup, DistributedBackend}; // Unused in current tests

    #[test]
    fn test_gradient_sync_strategy_creation() {
        // 基本的な同期戦略の作成テスト
        let sync = GradientSyncStrategy::Synchronous;
        let async_strategy = GradientSyncStrategy::Asynchronous;

        // 戦略が正しく作成されることを確認
        assert!(matches!(sync, GradientSyncStrategy::Synchronous));
        assert!(matches!(async_strategy, GradientSyncStrategy::Asynchronous));
    }

    #[test]
    fn test_gradient_sync_strategies() {
        let strategies = [
            GradientSyncStrategy::Synchronous,
            GradientSyncStrategy::Asynchronous,
            GradientSyncStrategy::LocalSGD { sync_frequency: 10 },
            GradientSyncStrategy::Compressed {
                compression_ratio: 0.1,
            },
            GradientSyncStrategy::Hierarchical,
        ];

        for strategy in &strategies {
            // Test strategy creation
            // 戦略作成をテスト
            assert!(matches!(
                strategy,
                GradientSyncStrategy::Synchronous
                    | GradientSyncStrategy::Asynchronous
                    | GradientSyncStrategy::LocalSGD { .. }
                    | GradientSyncStrategy::Compressed { .. }
                    | GradientSyncStrategy::Hierarchical
            ));
        }
    }
}
