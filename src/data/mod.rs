//! Data loading and processing utilities
//! データ読み込みと処理のユーティリティ

pub mod dataloader;
pub mod parallel_dataloader;

use crate::tensor::Tensor;
use num_traits::Float;

/// Trait for datasets
/// データセットのトレイト
pub trait Dataset<T: Float> {
    /// Returns the number of samples in the dataset
    /// データセット内のサンプル数を返します
    fn len(&self) -> usize;
    
    /// Returns true if the dataset is empty
    /// データセットが空の場合はtrueを返します
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns a sample at the given index
    /// 指定されたインデックスのサンプルを返します
    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)>;
}

/// Simple tensor dataset
/// シンプルなテンソルデータセット
pub struct TensorDataset<T: Float> {
    features: Vec<Tensor<T>>,
    targets: Vec<Tensor<T>>,
}

impl<T: Float> TensorDataset<T> {
    /// Creates a new tensor dataset
    /// 新しいテンソルデータセットを作成します
    pub fn new(features: Vec<Tensor<T>>, targets: Vec<Tensor<T>>) -> Result<Self, String> {
        if features.len() != targets.len() {
            return Err("Features and targets must have the same length".to_string());
        }
        
        Ok(TensorDataset { features, targets })
    }
}

impl<T: Float> Dataset<T> for TensorDataset<T> {
    fn len(&self) -> usize {
        self.features.len()
    }
    
    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        if index < self.len() {
            Some((self.features[index].clone(), self.targets[index].clone()))
        } else {
            None
        }
    }
}

// Re-export
pub use dataloader::DataLoader;
pub use parallel_dataloader::{ParallelDataLoader, ParallelBatchIterator, ParallelBatchProcessor};