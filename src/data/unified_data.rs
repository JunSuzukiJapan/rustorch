//! Unified data loading and processing interface to reduce code duplication
//! 重複コードを削減するための統一データローディングと処理インターフェース

use crate::common::{RusTorchResult, DataError};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Common dataset interface for all data types
/// 全データタイプの共通データセットインターフェース
pub trait UnifiedDataset<T: Float> {
    /// Get dataset length
    /// データセット長を取得
    fn len(&self) -> usize;
    
    /// Check if dataset is empty
    /// データセットが空かチェック
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item at index
    /// インデックスでアイテムを取得
    fn get_item(&self, index: usize) -> RusTorchResult<(Tensor<T>, Tensor<T>)>;
    
    /// Get batch of items
    /// アイテムのバッチを取得
    fn get_batch(&self, indices: &[usize]) -> RusTorchResult<(Tensor<T>, Tensor<T>)>;
    
    /// Get dataset name
    /// データセット名を取得
    fn name(&self) -> &str;
    
    /// Apply transformation to dataset
    /// データセットに変換を適用
    fn transform(&mut self, transform: Box<dyn DataTransform<T>>) -> RusTorchResult<()>;
}

/// Data transformation interface
/// データ変換インターフェース
pub trait DataTransform<T: Float>: Send + Sync {
    /// Apply transformation to input
    /// 入力に変換を適用
    fn apply(&self, input: &Tensor<T>) -> RusTorchResult<Tensor<T>>;
    
    /// Get transformation name
    /// 変換名を取得
    fn name(&self) -> &str;
}

/// Unified data loader configuration
/// 統一データローダー設定
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size for data loading
    /// データローディングのバッチサイズ
    pub batch_size: usize,
    /// Whether to shuffle the dataset
    /// データセットをシャッフルするかどうか
    pub shuffle: bool,
    /// Number of worker threads for parallel data loading
    /// 並列データローディングのワーカースレッド数
    pub num_workers: usize,
    /// Drop the last incomplete batch
    /// 最後の不完全なバッチを削除する
    pub drop_last: bool,
    /// Pin memory for faster GPU transfers
    /// 高速GPU転送のためのメモリピン留め
    pub pin_memory: bool,
    /// Prefetch factor for data loading
    /// データローディングのプリフェッチ係数
    pub prefetch_factor: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            num_workers: 0,
            drop_last: false,
            pin_memory: false,
            prefetch_factor: 2,
        }
    }
}

/// Unified data loader implementation
/// 統一データローダー実装
pub struct UnifiedDataLoader<T: Float> {
    dataset: Arc<dyn UnifiedDataset<T> + Send + Sync>,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    current_epoch: usize,
    batch_cache: Arc<Mutex<HashMap<usize, (Tensor<T>, Tensor<T>)>>>,
}

impl<T: Float + Send + Sync + 'static> UnifiedDataLoader<T> {
    /// Create new data loader
    /// 新しいデータローダーを作成
    pub fn new(
        dataset: Arc<dyn UnifiedDataset<T> + Send + Sync>,
        config: DataLoaderConfig,
    ) -> Self {
        let dataset_len = dataset.len();
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        
        if config.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            config,
            indices,
            current_epoch: 0,
            batch_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get number of batches
    /// バッチ数を取得
    pub fn num_batches(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            (self.dataset.len() + self.config.batch_size - 1) / self.config.batch_size
        }
    }

    /// Get batch at index
    /// インデックスでバッチを取得
    pub fn get_batch(&self, batch_idx: usize) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if batch_idx >= self.num_batches() {
            return Err(crate::common::RusTorchError::DataError(
                DataError::BatchError(format!("Batch index {} out of range", batch_idx))
            ));
        }

        // Check cache first
        {
            let cache = self.batch_cache.lock().unwrap();
            if let Some(cached_batch) = cache.get(&batch_idx) {
                return Ok(cached_batch.clone());
            }
        }

        let start_idx = batch_idx * self.config.batch_size;
        let end_idx = std::cmp::min(start_idx + self.config.batch_size, self.indices.len());
        
        if self.config.drop_last && end_idx - start_idx < self.config.batch_size {
            return Err(crate::common::RusTorchError::DataError(
                DataError::BatchError("Incomplete batch dropped".to_string())
            ));
        }

        let batch_indices = &self.indices[start_idx..end_idx];
        let batch = self.load_batch_parallel(batch_indices)?;

        // Cache the batch
        {
            let mut cache = self.batch_cache.lock().unwrap();
            cache.insert(batch_idx, batch.clone());
        }

        Ok(batch)
    }

    /// Load batch in parallel
    /// 並列でバッチをロード
    fn load_batch_parallel(&self, indices: &[usize]) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if self.config.num_workers <= 1 {
            // Sequential loading
            self.dataset.get_batch(indices)
        } else {
            // Parallel loading
            let items: Result<Vec<_>, _> = indices
                .par_iter()
                .map(|&idx| self.dataset.get_item(idx))
                .collect();

            let items = items?;
            self.combine_items(items)
        }
    }

    /// Combine individual items into batch
    /// 個別アイテムをバッチに結合
    fn combine_items(&self, items: Vec<(Tensor<T>, Tensor<T>)>) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if items.is_empty() {
            return Err(crate::common::RusTorchError::DataError(
                DataError::BatchError("Empty batch".to_string())
            ));
        }

        // Stack inputs and targets
        let inputs: Vec<Tensor<T>> = items.iter().map(|(input, _)| input.clone()).collect();
        let targets: Vec<Tensor<T>> = items.iter().map(|(_, target)| target.clone()).collect();

        let input_refs: Vec<&Tensor<T>> = inputs.iter().collect();
        let target_refs: Vec<&Tensor<T>> = targets.iter().collect();
        
        let batch_inputs = Tensor::stack(&input_refs).map_err(|e| 
            crate::common::RusTorchError::DataError(DataError::BatchError(format!("Failed to stack inputs: {:?}", e))))?;
        let batch_targets = Tensor::stack(&target_refs).map_err(|e| 
            crate::common::RusTorchError::DataError(DataError::BatchError(format!("Failed to stack targets: {:?}", e))))?;

        Ok((batch_inputs, batch_targets))
    }

    /// Start new epoch
    /// 新しいエポックを開始
    pub fn new_epoch(&mut self) {
        self.current_epoch += 1;
        
        if self.config.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }

        // Clear cache
        {
            let mut cache = self.batch_cache.lock().unwrap();
            cache.clear();
        }
    }

    /// Get current epoch
    /// 現在のエポックを取得
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Create iterator over batches
    /// バッチのイテレータを作成
    pub fn iter(&self) -> DataLoaderIterator<T> {
        DataLoaderIterator {
            loader: self,
            current_batch: 0,
        }
    }
}

/// Data loader iterator
/// データローダーイテレータ
pub struct DataLoaderIterator<'a, T: Float> {
    loader: &'a UnifiedDataLoader<T>,
    current_batch: usize,
}

impl<'a, T: Float + Send + Sync + 'static> Iterator for DataLoaderIterator<'a, T> {
    type Item = RusTorchResult<(Tensor<T>, Tensor<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.loader.num_batches() {
            None
        } else {
            let result = self.loader.get_batch(self.current_batch);
            self.current_batch += 1;
            Some(result)
        }
    }
}

/// Tensor dataset implementation
/// テンソルデータセット実装
pub struct UnifiedTensorDataset<T: Float> {
    features: Vec<Tensor<T>>,
    targets: Vec<Tensor<T>>,
    transforms: Vec<Box<dyn DataTransform<T>>>,
    name: String,
}

impl<T: Float + Send + Sync + 'static> UnifiedTensorDataset<T> {
    /// Create new tensor dataset
    /// 新しいテンソルデータセットを作成
    pub fn new(features: Vec<Tensor<T>>, targets: Vec<Tensor<T>>) -> RusTorchResult<Self> {
        if features.len() != targets.len() {
            return Err(crate::common::RusTorchError::DataError(
                DataError::DatasetError("Features and targets must have same length".to_string())
            ));
        }

        Ok(Self {
            features,
            targets,
            transforms: Vec::new(),
            name: "TensorDataset".to_string(),
        })
    }

    /// Add transform to dataset
    /// データセットに変換を追加
    pub fn add_transform(&mut self, transform: Box<dyn DataTransform<T>>) {
        self.transforms.push(transform);
    }
}

impl<T: Float + Send + Sync + 'static> UnifiedDataset<T> for UnifiedTensorDataset<T> {
    fn len(&self) -> usize {
        self.features.len()
    }

    fn get_item(&self, index: usize) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if index >= self.len() {
            return Err(crate::common::RusTorchError::DataError(
                DataError::DatasetError(format!("Index {} out of range", index))
            ));
        }

        let mut feature = self.features[index].clone();
        let target = self.targets[index].clone();

        // Apply transforms
        for transform in &self.transforms {
            feature = transform.apply(&feature)?;
        }

        Ok((feature, target))
    }

    fn get_batch(&self, indices: &[usize]) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        let items: Result<Vec<_>, _> = indices
            .iter()
            .map(|&idx| self.get_item(idx))
            .collect();

        let items = items?;
        
        if items.is_empty() {
            return Err(crate::common::RusTorchError::DataError(
                DataError::BatchError("Empty batch".to_string())
            ));
        }

        let inputs: Vec<Tensor<T>> = items.iter().map(|(input, _)| input.clone()).collect();
        let targets: Vec<Tensor<T>> = items.iter().map(|(_, target)| target.clone()).collect();

        let input_refs: Vec<&Tensor<T>> = inputs.iter().collect();
        let target_refs: Vec<&Tensor<T>> = targets.iter().collect();
        
        let batch_inputs = Tensor::stack(&input_refs).map_err(|e| 
            crate::common::RusTorchError::DataError(DataError::BatchError(format!("Failed to stack inputs: {:?}", e))))?;
        let batch_targets = Tensor::stack(&target_refs).map_err(|e| 
            crate::common::RusTorchError::DataError(DataError::BatchError(format!("Failed to stack targets: {:?}", e))))?;

        Ok((batch_inputs, batch_targets))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn transform(&mut self, transform: Box<dyn DataTransform<T>>) -> RusTorchResult<()> {
        self.transforms.push(transform);
        Ok(())
    }
}

/// Common data transformations
/// 共通データ変換
pub struct CommonTransforms;

impl CommonTransforms {
    /// Normalize transform
    /// 正規化変換
    pub fn normalize<T: Float + 'static + Send + Sync>(mean: T, std: T) -> Box<dyn DataTransform<T>> {
        Box::new(NormalizeTransform { mean, std })
    }

    /// Random noise transform
    /// ランダムノイズ変換
    pub fn random_noise<T: Float + 'static + Send + Sync>(noise_level: T) -> Box<dyn DataTransform<T>> {
        Box::new(RandomNoiseTransform { noise_level })
    }

    /// Scale transform
    /// スケール変換
    pub fn scale<T: Float + 'static + Send + Sync>(factor: T) -> Box<dyn DataTransform<T>> {
        Box::new(ScaleTransform { factor })
    }
}

/// Normalize transformation
/// 正規化変換
struct NormalizeTransform<T: Float> {
    #[allow(dead_code)]
    mean: T,
    #[allow(dead_code)]
    std: T,
}

impl<T: Float + 'static + Send + Sync> DataTransform<T> for NormalizeTransform<T> {
    fn apply(&self, input: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        // Simple implementation without division for now
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        "Normalize"
    }
}

/// Random noise transformation
/// ランダムノイズ変換
struct RandomNoiseTransform<T: Float> {
    #[allow(dead_code)]
    noise_level: T,
}

impl<T: Float + 'static + Send + Sync> DataTransform<T> for RandomNoiseTransform<T> {
    fn apply(&self, input: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        // Simple implementation without noise addition for now
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        "RandomNoise"
    }
}

/// Scale transformation
/// スケール変換
struct ScaleTransform<T: Float> {
    #[allow(dead_code)]
    factor: T,
}

impl<T: Float + 'static + Send + Sync> DataTransform<T> for ScaleTransform<T> {
    fn apply(&self, input: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        // Simple implementation without scaling for now
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        "Scale"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_tensor_dataset() {
        let features = vec![
            Tensor::from_vec(vec![1.0f32, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0f32, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0f32, 6.0], vec![2]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0f32], vec![1]),
            Tensor::from_vec(vec![1.0f32], vec![1]),
            Tensor::from_vec(vec![0.0f32], vec![1]),
        ];

        let dataset = UnifiedTensorDataset::new(features, targets).unwrap();
        assert_eq!(dataset.len(), 3);

        let (feature, target) = dataset.get_item(0).unwrap();
        assert_eq!(feature.as_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(target.as_slice().unwrap(), &[0.0]);
    }

    #[test]
    fn test_unified_data_loader() {
        let features = vec![
            Tensor::from_vec(vec![1.0f32, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0f32, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0f32, 6.0], vec![2]),
            Tensor::from_vec(vec![7.0f32, 8.0], vec![2]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0f32], vec![1]),
            Tensor::from_vec(vec![1.0f32], vec![1]),
            Tensor::from_vec(vec![0.0f32], vec![1]),
            Tensor::from_vec(vec![1.0f32], vec![1]),
        ];

        let dataset = Arc::new(UnifiedTensorDataset::new(features, targets).unwrap());
        let config = DataLoaderConfig {
            batch_size: 2,
            shuffle: false,
            ..Default::default()
        };

        let loader = UnifiedDataLoader::new(dataset, config);
        assert_eq!(loader.num_batches(), 2);

        let (batch_features, batch_targets) = loader.get_batch(0).unwrap();
        assert_eq!(batch_features.shape(), &[2, 2]);
        assert_eq!(batch_targets.shape(), &[2, 1]);
    }

    #[test]
    fn test_data_transforms() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        
        let normalize = CommonTransforms::normalize(1.0f32, 2.0f32);
        let normalized = normalize.apply(&input).unwrap();
        assert_eq!(normalized.as_slice().unwrap(), &[0.0, 0.5, 1.0]);

        let scale = CommonTransforms::scale(2.0f32);
        let scaled = scale.apply(&input).unwrap();
        assert_eq!(scaled.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_data_loader_iterator() {
        let features = vec![
            Tensor::from_vec(vec![1.0f32], vec![1]),
            Tensor::from_vec(vec![2.0f32], vec![1]),
        ];
        let targets = vec![
            Tensor::from_vec(vec![0.0f32], vec![1]),
            Tensor::from_vec(vec![1.0f32], vec![1]),
        ];

        let dataset = Arc::new(UnifiedTensorDataset::new(features, targets).unwrap());
        let config = DataLoaderConfig {
            batch_size: 1,
            ..Default::default()
        };

        let loader = UnifiedDataLoader::new(dataset, config);
        let mut count = 0;
        
        for batch_result in loader.iter() {
            let (features, targets) = batch_result.unwrap();
            assert_eq!(features.shape(), &[1, 1]);
            assert_eq!(targets.shape(), &[1, 1]);
            count += 1;
        }
        
        assert_eq!(count, 2);
    }
}
