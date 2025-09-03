//! Phase 5: DataLoader System - PyTorch-compatible data loading
//! フェーズ5: DataLoaderシステム - PyTorch互換データ読み込み
//!
//! This module provides comprehensive data loading capabilities including:
//! - Dataset traits and implementations (TensorDataset, ConcatDataset)
//! - DataLoader with multiprocessing and caching
//! - Sampling strategies (Random, Sequential, Batch, Weighted)
//! - Error handling with unified RusTorchError
//! - Memory-efficient prefetching for GPU optimization
//!
//! このモジュールは以下を含む包括的なデータ読み込み機能を提供します：
//! - データセットトレイトと実装（TensorDataset、ConcatDataset）
//! - マルチプロセシングとキャッシュ機能付きDataLoader
//! - サンプリング戦略（ランダム、順次、バッチ、重み付き）
//! - 統一RusTorchErrorによるエラーハンドリング
//! - GPU最適化のためのメモリ効率的プリフェッチ

pub mod dataloader;
pub mod dataset;
pub mod sampler;

// Legacy modules (Phase 3)
pub mod datasets;
pub mod parallel_dataloader;
pub mod streaming;
pub mod transforms;

use crate::tensor::Tensor;
use num_traits::Float;

/// Legacy Dataset trait for backward compatibility
/// 後方互換性のためのレガシーDatasetトレイト
#[deprecated(since = "0.6.0", note = "Use `Dataset` trait from Phase 5 API instead")]
pub trait LegacyDataset<T: Float> {
    /// Returns the number of samples in the dataset
    /// データセット内のサンプル数を返します
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty
    /// データセットが空の場合はtrueを返します
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a sample at the given index (legacy API)
    /// 指定されたインデックスのサンプルを返します（レガシーAPI）
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

// Re-export Phase 5 components (priority)
pub use dataset::{ConcatDataset, DataError, Dataset, IterableDataset, TensorDataset};
pub use sampler::{
    BatchSampler, RandomSampler, Sampler, SequentialSampler, SubsetRandomSampler,
    WeightedRandomSampler,
};

// Re-export legacy components (maintained for backward compatibility)
#[allow(deprecated)]
pub use dataloader::{
    Compose, DataLoader, LegacyDataLoader, Normalize, RandomHorizontalFlip, Transform,
};
#[allow(deprecated)]
pub use datasets::{CSVDataset, ImageDataset, MemoryMappedDataset, TextDataset};
#[allow(deprecated)]
pub use parallel_dataloader::{ParallelBatchIterator, ParallelBatchProcessor, ParallelDataLoader};
#[allow(deprecated)]
pub use streaming::{AsyncDataLoader, DynamicBatchLoader, StreamingDataset};
#[allow(deprecated)]
pub use transforms::{
    AddGaussianNoise, CenterCrop, Compose as ComposeTransform, MinMaxNormalize,
    Normalize as AdvancedNormalize, RandomBrightness, RandomChoice,
    RandomHorizontalFlip as AdvancedRandomHorizontalFlip, RandomRotation, TokenDropout,
    Transform as TransformTrait,
};
