//! DataLoader implementation for Phase 5 - PyTorch-compatible API
//! フェーズ5用DataLoader実装 - PyTorch互換API

use crate::data::dataset::{DataError, Dataset};
use crate::data::sampler::Sampler;
use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};


/// Main DataLoader implementation using Phase 5 Dataset trait
/// フェーズ5 Datasetトレイトを使用するメインDataLoader実装
pub struct DataLoader<'a, T: Float, D: Dataset<T>> {
    dataset: &'a D,
    sampler: Box<dyn Sampler + Send + Sync>,
    batch_size: usize,
    drop_last: bool,
    num_workers: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Float, D: Dataset<T>> DataLoader<'a, T, D> {
    /// Create new DataLoader
    /// 新しいDataLoaderを作成
    pub fn new(dataset: &'a D, sampler: Box<dyn Sampler + Send + Sync>, batch_size: usize) -> Self {
        Self {
            dataset,
            sampler,
            batch_size,
            drop_last: false,
            num_workers: 1,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create DataLoader with all options
    /// 全オプション付きDataLoaderを作成
    pub fn with_options(
        dataset: &'a D,
        sampler: Box<dyn Sampler + Send + Sync>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
    ) -> Self {
        Self {
            dataset,
            sampler,
            batch_size,
            drop_last,
            num_workers,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get next batch
    /// 次のバッチを取得
    pub fn next_batch(&mut self) -> Option<Vec<T>> {
        let mut indices = Vec::new();

        for _ in 0..self.batch_size {
            if let Some(idx) = self.sampler.sample() {
                indices.push(idx);
            } else {
                break;
            }
        }

        if indices.is_empty() {
            return None;
        }

        if self.drop_last && indices.len() < self.batch_size {
            return None;
        }

        // Collect batch items
        let mut batch = Vec::new();
        for idx in indices {
            if let Ok(item) = self.dataset.get_item(idx) {
                batch.push(item);
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Reset the sampler
    /// サンプラーをリセット
    pub fn reset(&mut self) {
        self.sampler.reset();
    }

    /// Check if exhausted
    /// 枯渇したかチェック
    pub fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }

    /// Get batch size
    /// バッチサイズを取得
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get number of workers
    /// ワーカー数を取得
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}
