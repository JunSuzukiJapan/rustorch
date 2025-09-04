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


use crate::tensor::Tensor;
use num_traits::Float;


// Re-export Phase 5 components (priority)
pub use dataset::{ConcatDataset, DataError, Dataset, IterableDataset, TensorDataset};
pub use dataloader::{DataLoader, LegacyDataLoader};
pub use sampler::{
    BatchSampler, RandomSampler, Sampler, SequentialSampler, SubsetRandomSampler,
    WeightedRandomSampler,
};

// Legacy components for backward compatibility (deprecated)
#[deprecated(since = "0.6.0", note = "Use Phase 5 Dataset trait instead")]
pub trait LegacyDataset<T: Float> {
    /// Get dataset length
    fn len(&self) -> usize;
    
    /// Get item at index, returns (input, target) pair
    fn get(&self, index: usize) -> Option<(crate::tensor::Tensor<T>, crate::tensor::Tensor<T>)>;
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
