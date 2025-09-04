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
pub use dataloader::DataLoader;
pub use dataset::{ConcatDataset, DataError, Dataset, IterableDataset, TensorDataset};
pub use sampler::{
    BatchSampler, RandomSampler, Sampler, SequentialSampler, SubsetRandomSampler,
    WeightedRandomSampler,
};
