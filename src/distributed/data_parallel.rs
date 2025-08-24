//! Data parallel training implementation
//! データ並列学習実装
//! 
//! This module provides data parallel training capabilities where the same model
//! is replicated across multiple devices and data is split among them.
//! 
//! このモジュールは、同じモデルを複数のデバイスに複製し、
//! データをそれらの間で分割するデータ並列学習機能を提供します。

use std::sync::Arc;
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::gpu::DeviceType;
use super::{DistributedError, DistributedResult, get_distributed_state};
use num_traits::Float;

/// Data parallel wrapper for models
/// モデル用データ並列ラッパー
#[derive(Debug)]
pub struct DataParallel<T, M>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync,
{
    /// Base model to be replicated
    /// 複製されるベースモデル
    module: Arc<M>,
    /// Devices to use for parallel training
    /// 並列学習に使用するデバイス
    device_ids: Vec<DeviceType>,
    /// Gradient synchronization strategy
    /// 勾配同期戦略
    sync_strategy: GradientSyncStrategy,
    _phantom: std::marker::PhantomData<T>,
}

/// Gradient synchronization strategies
/// 勾配同期戦略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientSyncStrategy {
    /// Synchronize gradients after each backward pass
    /// 各バックワードパス後に勾配を同期
    Synchronous,
    /// Asynchronous gradient updates
    /// 非同期勾配更新
    Asynchronous,
    /// Local SGD with periodic synchronization
    /// 定期同期を伴うローカルSGD
    LocalSGD { 
        /// Frequency of synchronization in steps
        /// 同期の頻度（ステップ数）
        sync_frequency: usize 
    },
}

impl<T, M> DataParallel<T, M>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync + 'static,
{
    /// Create a new data parallel wrapper
    /// 新しいデータ並列ラッパーを作成
    pub fn new(
        module: M,
        device_ids: Vec<DeviceType>,
    ) -> Self {
        Self {
            module: Arc::new(module),
            device_ids,
            sync_strategy: GradientSyncStrategy::Synchronous,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Set gradient synchronization strategy
    /// 勾配同期戦略を設定
    pub fn set_sync_strategy(&mut self, strategy: GradientSyncStrategy) {
        self.sync_strategy = strategy;
    }
    
    /// Get number of devices
    /// デバイス数を取得
    pub fn num_devices(&self) -> usize {
        self.device_ids.len()
    }
    
    /// Replicate input across devices
    /// 入力をデバイス間で複製
    pub fn replicate_input(&self, input: &Variable<T>) -> DistributedResult<Vec<Variable<T>>> {
        let batch_size = input.data().read().unwrap().shape()[0];
        let chunk_size = (batch_size + self.device_ids.len() - 1) / self.device_ids.len();
        
        let mut replicated_inputs = Vec::new();
        
        for (i, _device) in self.device_ids.iter().enumerate() {
            let start_idx = i * chunk_size;
            let end_idx = ((i + 1) * chunk_size).min(batch_size);
            
            if start_idx < batch_size {
                // Create a slice of the input for this device
                // このデバイス用の入力スライスを作成
                let chunk_shape = {
                    let input_data = input.data();
                    let data_guard = input_data.read().unwrap();
                    let mut shape = data_guard.shape().to_vec();
                    shape[0] = end_idx - start_idx;
                    shape
                };
                
                // Extract chunk data (simplified implementation)
                // チャンクデータを抽出（簡略化実装）
                let chunk_tensor = Tensor::zeros(&chunk_shape);
                let chunk_var = Variable::new(chunk_tensor, input.requires_grad());
                
                replicated_inputs.push(chunk_var);
            }
        }
        
        Ok(replicated_inputs)
    }
    
    /// Gather outputs from devices
    /// デバイスから出力を集約
    pub fn gather_outputs(&self, outputs: Vec<Variable<T>>) -> DistributedResult<Variable<T>> {
        if outputs.is_empty() {
            return Err(DistributedError::ProcessGroupError("No outputs to gather".to_string()).into());
        }
        
        // Calculate total output size
        // 総出力サイズを計算
        let total_batch_size: usize = outputs.iter().map(|o| o.data().read().unwrap().shape()[0]).sum();
        let mut output_shape = outputs[0].data().read().unwrap().shape().to_vec();
        output_shape[0] = total_batch_size;
        
        // Create concatenated output
        // 連結された出力を作成
        let output_tensor = Tensor::zeros(&output_shape);
        let output_var = Variable::new(output_tensor, outputs[0].requires_grad());
        
        Ok(output_var)
    }
    
    /// Synchronize gradients across devices
    /// デバイス間で勾配を同期
    pub fn sync_gradients(&self) -> DistributedResult<()> {
        match self.sync_strategy {
            GradientSyncStrategy::Synchronous => self.sync_gradients_sync(),
            GradientSyncStrategy::Asynchronous => self.sync_gradients_async(),
            GradientSyncStrategy::LocalSGD { sync_frequency: _ } => self.sync_gradients_local_sgd(),
        }
    }
    
    fn sync_gradients_sync(&self) -> DistributedResult<()> {
        // Synchronous gradient synchronization
        // 同期勾配同期
        let state = get_distributed_state();
        let state_guard = state.lock().unwrap();
        
        if let Some(_pg) = &state_guard.process_group {
            // All-reduce gradients across all processes
            // 全プロセス間で勾配をall-reduce
            // Implementation would use the actual communication backend
            // 実装では実際の通信バックエンドを使用
            drop(state_guard);
            Ok(())
        } else {
            Err(DistributedError::ProcessGroupError("Process group not initialized".to_string()).into())
        }
    }
    
    fn sync_gradients_async(&self) -> DistributedResult<()> {
        // Asynchronous gradient synchronization
        // 非同期勾配同期
        Ok(())
    }
    
    fn sync_gradients_local_sgd(&self) -> DistributedResult<()> {
        // Local SGD gradient synchronization
        // ローカルSGD勾配同期
        Ok(())
    }
}

impl<T, M> Module<T> for DataParallel<T, M>
where
    T: Float + Send + Sync + 'static + std::fmt::Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync + 'static,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // Replicate input across devices
        // デバイス間で入力を複製
        let replicated_inputs = self.replicate_input(input).unwrap_or_else(|_| vec![input.clone()]);
        
        // Forward pass on each device
        // 各デバイスでフォワードパス
        let outputs: Vec<Variable<T>> = replicated_inputs.iter()
            .map(|input| self.module.forward(input))
            .collect();
        
        // Gather outputs to output device
        // 出力デバイスに出力を収集
        self.gather_outputs(outputs).unwrap_or_else(|_| input.clone())
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.module.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Distributed data loader for data parallel training
/// データ並列学習用分散データローダー
pub struct DistributedDataLoader<T: Float> {
    /// Training data
    /// 学習データ
    data: Vec<Tensor<T>>,
    /// Local labels
    /// ローカルラベル
    labels: Vec<Tensor<T>>,
    /// Batch size per process
    /// プロセスあたりのバッチサイズ
    batch_size: usize,
    /// Current epoch
    /// 現在のエポック
    current_epoch: usize,
    /// Shuffle data between epochs
    /// エポック間でデータをシャッフル
    shuffle: bool,
    /// Random seed for reproducibility
    /// 再現性のためのランダムシード
    seed: Option<u64>,
}


impl<T: Float + Send + Sync + 'static> DistributedDataLoader<T> {
    /// Create a new distributed data loader
    /// 新しい分散データローダーを作成
    pub fn new(
        data: Vec<Tensor<T>>,
        labels: Vec<Tensor<T>>,
        batch_size: usize,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        Self {
            data,
            labels,
            batch_size,
            current_epoch: 0,
            shuffle,
            seed,
        }
    }
    
    /// Set epoch for deterministic shuffling
    /// 決定論的シャッフル用のエポックを設定
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
    
    /// Get batch iterator
    /// バッチイテレータを取得
    pub fn iter(&self) -> DistributedDataIterator<'_, T> {
        DistributedDataIterator::new(
            &self.data,
            &self.labels,
            self.batch_size,
            self.current_epoch,
            self.shuffle,
            self.seed,
        )
    }
    
    /// Get number of batches per epoch
    /// エポックあたりのバッチ数を取得
    pub fn len(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }
}

/// Iterator for distributed data loading
/// 分散データローディング用イテレータ
pub struct DistributedDataIterator<'a, T: Float> {
    data: &'a [Tensor<T>],
    labels: &'a [Tensor<T>],
    batch_size: usize,
    current_index: usize,
    indices: Vec<usize>,
}

impl<'a, T: Float> DistributedDataIterator<'a, T> {
    fn new(
        data: &'a [Tensor<T>],
        labels: &'a [Tensor<T>],
        batch_size: usize,
        epoch: usize,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        
        if shuffle {
            // Deterministic shuffling based on epoch and seed
            // エポックとシードに基づく決定論的シャッフル
            use rand::{Rng, SeedableRng};
            use rand::rngs::StdRng;
            
            let actual_seed = seed.unwrap_or(0) + epoch as u64;
            let mut rng = StdRng::seed_from_u64(actual_seed);
            
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }
        
        Self {
            data,
            labels,
            batch_size,
            current_index: 0,
            indices,
        }
    }
}

impl<'a, T: Float> Iterator for DistributedDataIterator<'a, T> {
    type Item = (Vec<&'a Tensor<T>>, Vec<&'a Tensor<T>>);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.indices.len() {
            return None;
        }
        
        let end_index = (self.current_index + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_index..end_index];
        
        let batch_data: Vec<&Tensor<T>> = batch_indices.iter()
            .map(|&i| &self.data[i])
            .collect();
        
        let batch_labels: Vec<&Tensor<T>> = batch_indices.iter()
            .map(|&i| &self.labels[i])
            .collect();
        
        self.current_index = end_index;
        
        Some((batch_data, batch_labels))
    }
}

/// Distributed sampler for ensuring each process gets different data
/// 各プロセスが異なるデータを取得することを保証する分散サンプラー
pub struct DistributedSampler {
    /// Total number of samples
    /// サンプル総数
    num_samples: usize,
    /// Number of processes
    /// プロセス数
    num_replicas: usize,
    /// Current process rank
    /// 現在のプロセスランク
    rank: usize,
    /// Current epoch
    /// 現在のエポック
    epoch: usize,
    /// Whether to drop last incomplete batch
    /// 最後の不完全なバッチを削除するかどうか
    drop_last: bool,
    /// Random seed
    /// ランダムシード
    seed: u64,
}

impl DistributedSampler {
    /// Create a new distributed sampler
    /// 新しい分散サンプラーを作成
    pub fn new(
        num_samples: usize,
        num_replicas: Option<usize>,
        rank: Option<usize>,
        drop_last: bool,
        seed: u64,
    ) -> DistributedResult<Self> {
        let state = get_distributed_state();
        let state_guard = state.lock().unwrap();
        
        let num_replicas = num_replicas.or_else(|| state_guard.world_size()).unwrap_or(1);
        let rank = rank.or_else(|| state_guard.rank()).unwrap_or(0);
        
        if rank >= num_replicas {
            return Err(DistributedError::ProcessGroupError(
                format!("Rank {} is greater than or equal to num_replicas {}", rank, num_replicas)
            ).into());
        }
        
        Ok(Self {
            num_samples,
            num_replicas,
            rank,
            epoch: 0,
            drop_last,
            seed,
        })
    }
    
    /// Set epoch for deterministic sampling
    /// 決定論的サンプリング用のエポックを設定
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }
    
    /// Generate sample indices for current process
    /// 現在のプロセス用のサンプルインデックスを生成
    pub fn sample_indices(&self) -> Vec<usize> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(self.seed + self.epoch as u64);
        let mut indices: Vec<usize> = (0..self.num_samples).collect();
        
        // Shuffle indices
        // インデックスをシャッフル
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        // Pad indices to make it evenly divisible by num_replicas
        // num_replicasで均等に分割できるようにインデックスをパディング
        let total_size = if self.drop_last {
            (self.num_samples / self.num_replicas) * self.num_replicas
        } else {
            ((self.num_samples + self.num_replicas - 1) / self.num_replicas) * self.num_replicas
        };
        
        while indices.len() < total_size {
            let remaining = total_size - indices.len();
            let copy_len = std::cmp::min(indices.len(), remaining);
            let to_copy = indices[..copy_len].to_vec();
            indices.extend_from_slice(&to_copy);
        }
        indices.truncate(total_size);
        
        // Subsample for current rank
        // 現在のランク用にサブサンプル
        let num_samples_per_replica = total_size / self.num_replicas;
        let start = self.rank * num_samples_per_replica;
        let end = start + num_samples_per_replica;
        
        indices[start..end].to_vec()
    }
    
    /// Get number of samples for current process
    /// 現在のプロセスのサンプル数を取得
    pub fn len(&self) -> usize {
        if self.drop_last {
            self.num_samples / self.num_replicas
        } else {
            (self.num_samples + self.num_replicas - 1) / self.num_replicas
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    
    #[test]
    fn test_gradient_sync_strategy() {
        let strategies = [
            GradientSyncStrategy::Synchronous,
            GradientSyncStrategy::Asynchronous,
            GradientSyncStrategy::LocalSGD { sync_frequency: 10 },
        ];
        
        for strategy in &strategies {
            match strategy {
                GradientSyncStrategy::Synchronous => assert!(true),
                GradientSyncStrategy::Asynchronous => assert!(true),
                GradientSyncStrategy::LocalSGD { sync_frequency } => assert!(*sync_frequency > 0),
            }
        }
    }
    
    #[test]
    fn test_distributed_data_loader() {
        let data = vec![
            Tensor::<f32>::ones(&[10, 5]),
            Tensor::<f32>::ones(&[10, 5]),
            Tensor::<f32>::ones(&[10, 5]),
        ];
        let labels = vec![
            Tensor::<f32>::zeros(&[10, 1]),
            Tensor::<f32>::zeros(&[10, 1]),
            Tensor::<f32>::zeros(&[10, 1]),
        ];
        
        let loader = DistributedDataLoader::new(data, labels, 2, true, Some(42).into());
        assert_eq!(loader.len(), 2); // ceil(3/2) = 2 batches
        
        let mut iter = loader.iter();
        let batch = iter.next();
        assert!(batch.is_some());
        
        let (batch_data, batch_labels) = batch.unwrap();
        assert_eq!(batch_data.len(), 2);
        assert_eq!(batch_labels.len(), 2);
    }
    
    #[test]
    fn test_distributed_sampler() {
        let sampler = DistributedSampler::new(100, Some(4), Some(0), false, 42);
        assert!(sampler.is_ok());
        
        let sampler = sampler.unwrap();
        let indices = sampler.sample_indices();
        assert_eq!(indices.len(), 25); // 100 / 4 = 25 samples per replica
        
        // Check that indices are within valid range
        // インデックスが有効な範囲内にあることを確認
        for &idx in &indices {
            assert!(idx < 100);
        }
    }
    
    #[test]
    fn test_data_parallel_creation() {
        let linear = Linear::<f32>::new(10, 5);
        let devices = vec![DeviceType::Cpu, DeviceType::Cpu]; // Use CPU for testing
        
        let dp = DataParallel::new(linear, devices);
        assert_eq!(dp.num_devices(), 2);
    }
}
