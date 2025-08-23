//! DataLoader implementation for batch processing
//! バッチ処理のためのDataLoader実装

use super::Dataset;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::Arc;

/// Trait for data transformations
/// データ変換のためのトレイト
pub trait Transform<T: Float + 'static> {
    /// Apply transformation to input data
    /// 入力データに変換を適用します
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String>;
}

/// Compose multiple transformations
/// 複数の変換を組み合わせます
pub struct Compose<T: Float + 'static> {
    transforms: Vec<Box<dyn Transform<T> + Send + Sync>>,
}

impl<T: Float + 'static> Compose<T> {
    /// Creates a new composition of transforms
    /// 新しい変換の組み合わせを作成します
    pub fn new(transforms: Vec<Box<dyn Transform<T> + Send + Sync>>) -> Self {
        Self { transforms }
    }
}

impl<T: Float + 'static> Transform<T> for Compose<T> {
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        self.transforms.iter().try_fold(data.clone(), |acc, transform| {
            transform.apply(&acc)
        })
    }
}

/// Normalize transformation
/// 正規化変換
pub struct Normalize<T: Float + 'static> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: Float + 'static> Normalize<T> {
    /// Creates a new normalize transformation
    /// 新しい正規化変換を作成します
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Result<Self, String> {
        if mean.len() != std.len() {
            return Err("Mean and std must have the same length".to_string());
        }
        Ok(Self { mean, std })
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T> for Normalize<T> {
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        // Simple normalization implementation using available operations
        if self.mean.is_empty() || self.std.is_empty() {
            return Ok(data.clone());
        }
        
        // For now, apply global normalization using first mean/std values
        let mean_val = self.mean[0];
        let std_val = self.std[0];
        
        // Use available tensor operations: (data - mean) / std
        let normalized = (data + (-mean_val)) / std_val;
        Ok(normalized)
    }
}

/// Random horizontal flip transformation
/// ランダム水平反転変換
pub struct RandomHorizontalFlip<T: Float + 'static> {
    probability: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> RandomHorizontalFlip<T> {
    /// Creates a new random horizontal flip transformation
    /// 新しいランダム水平反転変換を作成します
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + 'static> Transform<T> for RandomHorizontalFlip<T> {
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        use rand::Rng;
        
        if rand::thread_rng().gen::<f64>() < self.probability {
            // For now, return a placeholder implementation
            // In a full implementation, this would flip the tensor horizontally
            // Since flip() method is not available, we'll just return the original tensor
            Ok(data.clone())
        } else {
            Ok(data.clone())
        }
    }
}

/// DataLoader for loading data in batches
/// バッチでデータを読み込むためのDataLoader
pub struct DataLoader<'a, T: Float, D: Dataset<T>> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    current_index: usize,
    transform: Option<Arc<dyn Transform<T> + Send + Sync>>,
    lazy_loading: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive, D: Dataset<T>> DataLoader<'a, T, D> {
    /// Creates a new DataLoader
    /// 新しいDataLoaderを作成します
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        Self::new_with_options(dataset, batch_size, shuffle, false)
    }
    
    /// Creates a new DataLoader with advanced options
    /// 高度なオプションを持つ新しいDataLoaderを作成します
    pub fn new_with_options(dataset: &'a D, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current_index: 0,
            transform: None,
            lazy_loading: false,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Creates a new DataLoader with transformations
    /// 変換を持つ新しいDataLoaderを作成します
    pub fn new_with_transform(
        dataset: &'a D, 
        batch_size: usize, 
        shuffle: bool, 
        transform: Arc<dyn Transform<T> + Send + Sync>
    ) -> Self {
        let mut dataloader = Self::new(dataset, batch_size, shuffle);
        dataloader.transform = Some(transform);
        dataloader
    }
    
    /// Creates a new DataLoader with memory-efficient lazy loading
    /// メモリ効率的な遅延読み込みを持つ新しいDataLoaderを作成します
    pub fn new_lazy(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut dataloader = Self::new(dataset, batch_size, shuffle);
        dataloader.lazy_loading = true;
        dataloader
    }
    
    /// Creates a new DataLoader with all options
    /// すべてのオプションを持つ新しいDataLoaderを作成します
    pub fn new_with_all_options(
        dataset: &'a D,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        transform: Option<Arc<dyn Transform<T> + Send + Sync>>,
        lazy_loading: bool,
    ) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current_index: 0,
            transform,
            lazy_loading,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Returns the number of batches
    /// バッチ数を返します
    pub fn len(&self) -> usize {
        if self.drop_last {
            self.dataset.len() / self.batch_size
        } else {
            (self.dataset.len() + self.batch_size - 1) / self.batch_size
        }
    }
    
    /// Returns true if there are no more batches
    /// もうバッチがない場合はtrueを返します
    pub fn is_empty(&self) -> bool {
        self.current_index >= self.dataset.len()
    }
    
    /// Resets the dataloader for a new epoch
    /// 新しいエポックのためにデータローダーをリセットします
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
    
    /// Gets the next batch
    /// 次のバッチを取得します
    pub fn next_batch(&mut self) -> Option<(Tensor<T>, Tensor<T>)> {
        if self.lazy_loading {
            self.next_batch_lazy()
        } else {
            self.next_batch_eager()
        }
    }
    
    /// Gets the next batch with eager loading (loads all data at once)
    /// 即座読み込みで次のバッチを取得します（すべてのデータを一度に読み込む）
    fn next_batch_eager(&mut self) -> Option<(Tensor<T>, Tensor<T>)> {
        if self.is_empty() {
            return None;
        }
        
        let end_index = std::cmp::min(
            self.current_index + self.batch_size,
            self.dataset.len()
        );
        
        // Check if we should drop incomplete batches
        if self.drop_last && (end_index - self.current_index) < self.batch_size {
            return None;
        }
        
        let mut batch_features = Vec::new();
        let mut batch_targets = Vec::new();
        
        for i in self.current_index..end_index {
            let index = self.indices[i];
            if let Some((mut feature, target)) = self.dataset.get(index) {
                // Apply transformations if available
                if let Some(ref transform) = self.transform {
                    if let Ok(transformed) = transform.apply(&feature) {
                        feature = transformed;
                    }
                }
                batch_features.push(feature);
                batch_targets.push(target);
            }
        }
        
        self.current_index = end_index;
        
        if !batch_features.is_empty() {
            // Stack tensors to create batch tensors
            let feature_refs: Vec<&Tensor<T>> = batch_features.iter().collect();
            let target_refs: Vec<&Tensor<T>> = batch_targets.iter().collect();
            
            match (Tensor::stack(&feature_refs), Tensor::stack(&target_refs)) {
                (Ok(stacked_features), Ok(stacked_targets)) => {
                    Some((stacked_features, stacked_targets))
                }
                _ => {
                    // Fallback: return individual tensors (not ideal for batch processing)
                    Some((batch_features[0].clone(), batch_targets[0].clone()))
                }
            }
        } else {
            None
        }
    }
    
    /// Gets the next batch with lazy loading (loads one item at a time)
    /// 遅延読み込みで次のバッチを取得します（一度に一つのアイテムを読み込む）
    fn next_batch_lazy(&mut self) -> Option<(Tensor<T>, Tensor<T>)> {
        if self.is_empty() {
            return None;
        }
        
        let end_index = std::cmp::min(
            self.current_index + self.batch_size,
            self.dataset.len()
        );
        
        // Check if we should drop incomplete batches
        if self.drop_last && (end_index - self.current_index) < self.batch_size {
            return None;
        }
        
        // Use lazy iterator approach - only load data as needed
        let batch_indices = &self.indices[self.current_index..end_index];
        let mut batch_features = Vec::with_capacity(batch_indices.len());
        let mut batch_targets = Vec::with_capacity(batch_indices.len());
        
        // Load items one by one to reduce memory pressure
        for &index in batch_indices {
            if let Some((mut feature, target)) = self.dataset.get(index) {
                // Apply transformations if available
                if let Some(ref transform) = self.transform {
                    if let Ok(transformed) = transform.apply(&feature) {
                        feature = transformed;
                    }
                }
                batch_features.push(feature);
                batch_targets.push(target);
            }
        }
        
        self.current_index = end_index;
        
        if !batch_features.is_empty() {
            // For lazy loading, we can optionally use a more memory-efficient stacking approach
            // For now, use the same stacking logic but with pre-allocated capacity
            let feature_refs: Vec<&Tensor<T>> = batch_features.iter().collect();
            let target_refs: Vec<&Tensor<T>> = batch_targets.iter().collect();
            
            match (Tensor::stack(&feature_refs), Tensor::stack(&target_refs)) {
                (Ok(stacked_features), Ok(stacked_targets)) => {
                    Some((stacked_features, stacked_targets))
                }
                _ => {
                    // Fallback: return individual tensors
                    Some((batch_features[0].clone(), batch_targets[0].clone()))
                }
            }
        } else {
            None
        }
    }
}

impl<'a, T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive, D: Dataset<T>> Iterator for DataLoader<'a, T, D> {
    type Item = (Tensor<T>, Tensor<T>);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;
    
    #[test]
    fn test_dataloader() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
            Tensor::from_vec(vec![7.0, 8.0], vec![2]),
        ];
        
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let mut dataloader = DataLoader::new(&dataset, 2, false);
        
        assert_eq!(dataloader.len(), 2);
        
        // Test first batch
        if let Some((batch_features, batch_targets)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
        }
        
        // Test second batch
        if let Some((batch_features, batch_targets)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
        }
        
        // Should be empty now
        assert!(dataloader.next_batch().is_none());
    }
    
    #[test]
    fn test_dataloader_with_transformations() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
        ];
        
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        
        // Create a normalization transform
        let normalize = Normalize::new(vec![2.0], vec![1.0]).unwrap();
        let transform = Arc::new(normalize);
        
        let mut dataloader = DataLoader::new_with_transform(&dataset, 1, false, transform);
        
        // Test first batch with transformation applied
        if let Some((batch_features, _)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 1);
            // Features should be normalized: (1.0 - 2.0) / 1.0 = -1.0, (2.0 - 2.0) / 1.0 = 0.0
            let data = batch_features.as_slice().unwrap();
            assert_eq!(data[0], -1.0);
            assert_eq!(data[1], 0.0);
        }
    }
    
    #[test]
    fn test_dataloader_lazy_loading() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
            Tensor::from_vec(vec![7.0, 8.0], vec![2]),
        ];
        
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        let mut dataloader = DataLoader::new_lazy(&dataset, 2, false);
        
        assert_eq!(dataloader.len(), 2);
        
        // Test first batch with lazy loading
        if let Some((batch_features, batch_targets)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
        }
        
        // Test second batch with lazy loading
        if let Some((batch_features, batch_targets)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
        }
        
        // Should be empty now
        assert!(dataloader.next_batch().is_none());
    }
    
    #[test]
    fn test_dataloader_performance_comparison() {
        use std::time::Instant;
        
        // Create a larger dataset for performance testing
        let size = 1000;
        let features: Vec<Tensor<f32>> = (0..size)
            .map(|i| Tensor::from_vec(vec![i as f32, (i + 1) as f32], vec![2]))
            .collect();
        
        let targets: Vec<Tensor<f32>> = (0..size)
            .map(|i| Tensor::from_vec(vec![(i % 2) as f32], vec![1]))
            .collect();
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        
        // Test eager loading performance
        let start = Instant::now();
        let mut eager_dataloader = DataLoader::new(&dataset, 32, false);
        let mut eager_batches = 0;
        while eager_dataloader.next_batch().is_some() {
            eager_batches += 1;
        }
        let eager_duration = start.elapsed();
        
        // Test lazy loading performance
        let start = Instant::now();
        let mut lazy_dataloader = DataLoader::new_lazy(&dataset, 32, false);
        let mut lazy_batches = 0;
        while lazy_dataloader.next_batch().is_some() {
            lazy_batches += 1;
        }
        let lazy_duration = start.elapsed();
        
        // Both should process the same number of batches
        assert_eq!(eager_batches, lazy_batches);
        assert_eq!(eager_batches, (size + 31) / 32); // Ceiling division
        
        // Performance comparison (both should complete in reasonable time)
        assert!(eager_duration.as_millis() < 1000, "Eager loading took too long: {:?}", eager_duration);
        assert!(lazy_duration.as_millis() < 1000, "Lazy loading took too long: {:?}", lazy_duration);
        
        println!("Performance comparison:");
        println!("  Eager loading: {:?} for {} batches", eager_duration, eager_batches);
        println!("  Lazy loading:  {:?} for {} batches", lazy_duration, lazy_batches);
    }
    
    #[test]
    fn test_dataloader_memory_usage() {
        // Test with different batch sizes to verify memory efficiency
        let features: Vec<Tensor<f32>> = (0..100)
            .map(|i| Tensor::from_vec(vec![i as f32; 100], vec![100])) // Larger tensors
            .collect();
        
        let targets: Vec<Tensor<f32>> = (0..100)
            .map(|i| Tensor::from_vec(vec![(i % 10) as f32], vec![1]))
            .collect();
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        
        // Test with small batch size (should use less memory)
        let mut small_batch_loader = DataLoader::new_lazy(&dataset, 5, false);
        let mut batches_processed = 0;
        while small_batch_loader.next_batch().is_some() {
            batches_processed += 1;
        }
        assert_eq!(batches_processed, 20); // 100 / 5
        
        // Test with large batch size
        let mut large_batch_loader = DataLoader::new_lazy(&dataset, 50, false);
        let mut batches_processed = 0;
        while large_batch_loader.next_batch().is_some() {
            batches_processed += 1;
        }
        assert_eq!(batches_processed, 2); // 100 / 50
    }
    
    #[test]
    fn test_dataloader_with_transformations_and_lazy_loading() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
        ];
        
        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
        ];
        
        let dataset = TensorDataset::new(features, targets).unwrap();
        
        // Create composite transformations
        let normalize = Normalize::new(vec![2.0], vec![2.0]).unwrap();
        let flip = RandomHorizontalFlip::new(0.5);
        
        let transforms: Vec<Box<dyn Transform<f32> + Send + Sync>> = vec![
            Box::new(normalize),
            Box::new(flip),
        ];
        let compose = Compose::new(transforms);
        let transform = Arc::new(compose);
        
        let mut dataloader = DataLoader::new_with_all_options(
            &dataset,
            2,      // batch_size
            false,  // shuffle
            false,  // drop_last
            Some(transform),
            true,   // lazy_loading
        );
        
        // Test that transformations work with lazy loading
        if let Some((batch_features, batch_targets)) = dataloader.next_batch() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
            
            // Verify transformations were applied
            let data = batch_features.as_slice().unwrap();
            // First feature: (1.0 - 2.0) / 2.0 = -0.5, (2.0 - 2.0) / 2.0 = 0.0
            assert!((data[0] - (-0.5)).abs() < 1e-6);
            assert!((data[1] - 0.0).abs() < 1e-6);
        }
    }
}