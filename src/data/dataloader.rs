//! DataLoader implementation for batch processing
//! バッチ処理のためのDataLoader実装

use super::Dataset;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// DataLoader for loading data in batches
/// バッチでデータを読み込むためのDataLoader
pub struct DataLoader<'a, T: Float, D: Dataset<T>> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Float + Send + Sync + 'static, D: Dataset<T>> DataLoader<'a, T, D> {
    /// Creates a new DataLoader
    /// 新しいDataLoaderを作成します
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Returns the number of batches
    /// バッチ数を返します
    pub fn len(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
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
        if self.is_empty() {
            return None;
        }
        
        let end_index = std::cmp::min(
            self.current_index + self.batch_size,
            self.dataset.len()
        );
        
        let mut batch_features = Vec::new();
        let mut batch_targets = Vec::new();
        
        for i in self.current_index..end_index {
            let index = self.indices[i];
            if let Some((feature, target)) = self.dataset.get(index) {
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
}

impl<'a, T: Float + Send + Sync + 'static, D: Dataset<T>> Iterator for DataLoader<'a, T, D> {
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
}