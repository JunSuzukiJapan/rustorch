//! Advanced Data Loading and Processing Utilities
//! 高度なデータ読み込みと処理のユーティリティ
//!
//! This module provides comprehensive data loading capabilities including:
//! - Basic and advanced dataset types (CSV, Image, Text, Memory-mapped)
//! - Streaming and dynamic batch loading for large datasets
//! - Comprehensive data transformation pipeline
//! - Parallel and async data loading
//!
//! このモジュールは以下を含む包括的なデータ読み込み機能を提供します：
//! - 基本・高度なデータセットタイプ（CSV、画像、テキスト、メモリマップド）
//! - 大規模データセット用ストリーミング・動的バッチ読み込み
//! - 包括的なデータ変換パイプライン
//! - 並列・非同期データ読み込み

pub mod dataloader;
pub mod datasets;
pub mod parallel_dataloader;
pub mod streaming;
pub mod transforms;

use crate::tensor::Tensor;
use num_traits::Float;

/// Core trait for all datasets
/// すべてのデータセットのコアトレイト
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

    /// Get a batch of samples
    /// サンプルのバッチを取得
    fn get_batch(&self, indices: &[usize]) -> Vec<(Tensor<T>, Tensor<T>)> {
        indices.iter().filter_map(|&idx| self.get(idx)).collect()
    }

    /// Get dataset metadata (optional)
    /// データセットメタデータを取得（オプション）
    fn metadata(&self) -> std::collections::HashMap<String, String> {
        let mut meta = std::collections::HashMap::new();
        meta.insert("type".to_string(), "generic".to_string());
        meta.insert("samples".to_string(), self.len().to_string());
        meta
    }
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

    /// Get features
    /// 特徴量を取得
    pub fn features(&self) -> &[Tensor<T>] {
        &self.features
    }

    /// Get targets
    /// ターゲットを取得
    pub fn targets(&self) -> &[Tensor<T>] {
        &self.targets
    }

    /// Split into train and validation sets
    /// 訓練と検証セットに分割
    pub fn train_val_split(&self, train_ratio: f64) -> (Self, Self) {
        let split_idx = (self.features.len() as f64 * train_ratio) as usize;

        let train_features = self.features[..split_idx].to_vec();
        let train_targets = self.targets[..split_idx].to_vec();
        let val_features = self.features[split_idx..].to_vec();
        let val_targets = self.targets[split_idx..].to_vec();

        let train_dataset = TensorDataset {
            features: train_features,
            targets: train_targets,
        };

        let val_dataset = TensorDataset {
            features: val_features,
            targets: val_targets,
        };

        (train_dataset, val_dataset)
    }
}

impl<T: Float + 'static> Dataset<T> for TensorDataset<T> {
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

    fn metadata(&self) -> std::collections::HashMap<String, String> {
        let mut meta = std::collections::HashMap::new();
        meta.insert("type".to_string(), "tensor".to_string());
        meta.insert("samples".to_string(), self.len().to_string());
        if !self.features.is_empty() {
            meta.insert(
                "feature_shape".to_string(),
                format!("{:?}", self.features[0].shape()),
            );
        }
        if !self.targets.is_empty() {
            meta.insert(
                "target_shape".to_string(),
                format!("{:?}", self.targets[0].shape()),
            );
        }
        meta
    }
}

// Re-export all public components
pub use dataloader::{Compose, DataLoader, Normalize, RandomHorizontalFlip, Transform};
pub use datasets::{CSVDataset, ImageDataset, MemoryMappedDataset, TextDataset};
pub use parallel_dataloader::{ParallelBatchIterator, ParallelBatchProcessor, ParallelDataLoader};
pub use streaming::{AsyncDataLoader, DynamicBatchLoader, StreamingDataset};
pub use transforms::{
    AddGaussianNoise, CenterCrop, Compose as ComposeTransform, MinMaxNormalize,
    Normalize as AdvancedNormalize, RandomBrightness, RandomChoice,
    RandomHorizontalFlip as AdvancedRandomHorizontalFlip, RandomRotation, TokenDropout,
    Transform as TransformTrait,
};
