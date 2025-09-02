//! Sampling strategies for Phase 5 DataLoader
//! フェーズ5 DataLoader用サンプリング戦略

use crate::data::dataset::DataError;
use crate::error::RusTorchError;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::VecDeque;

/// Core Sampler trait following PyTorch API
/// PyTorch APIに従うコアSamplerトレイト
pub trait Sampler {
    /// Sample next index
    /// 次のインデックスをサンプル
    fn sample(&mut self) -> Option<usize>;
    
    /// Get total number of samples
    /// 総サンプル数を取得
    fn len(&self) -> usize;
    
    /// Reset sampler for new epoch
    /// 新エポック用にサンプラーをリセット
    fn reset(&mut self);
    
    /// Check if sampler is exhausted
    /// サンプラーが枯渇したかチェック
    fn is_empty(&self) -> bool;
}

/// Sequential sampler - returns indices in order
/// 順次サンプラー - インデックスを順番に返す
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    data_source_len: usize,
    current_index: usize,
}

impl SequentialSampler {
    /// Create new sequential sampler
    /// 新しい順次サンプラーを作成
    pub fn new(data_source_len: usize) -> Self {
        Self {
            data_source_len,
            current_index: 0,
        }
    }
}

impl Sampler for SequentialSampler {
    fn sample(&mut self) -> Option<usize> {
        if self.current_index < self.data_source_len {
            let index = self.current_index;
            self.current_index += 1;
            Some(index)
        } else {
            None
        }
    }
    
    fn len(&self) -> usize {
        self.data_source_len
    }
    
    fn reset(&mut self) {
        self.current_index = 0;
    }
    
    fn is_empty(&self) -> bool {
        self.current_index >= self.data_source_len
    }
}

/// Random sampler - returns shuffled indices
/// ランダムサンプラー - シャッフルされたインデックスを返す
#[derive(Debug)]
pub struct RandomSampler {
    indices: VecDeque<usize>,
    original_len: usize,
    replacement: bool,
    generator: Option<u64>, // Seed for reproducible randomness
}

impl RandomSampler {
    /// Create new random sampler
    /// 新しいランダムサンプラーを作成
    pub fn new(data_source_len: usize) -> Self {
        let mut indices: Vec<usize> = (0..data_source_len).collect();
        indices.shuffle(&mut thread_rng());
        
        Self {
            indices: indices.into(),
            original_len: data_source_len,
            replacement: false,
            generator: None,
        }
    }
    
    /// Create random sampler with replacement
    /// 復元ありランダムサンプラーを作成
    pub fn with_replacement(data_source_len: usize, num_samples: usize) -> Self {
        let mut sampler = Self::new(data_source_len);
        sampler.replacement = true;
        
        // Generate random indices with replacement
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..num_samples)
            .map(|_| rng.gen_range(0..data_source_len))
            .collect();
        
        sampler.indices = indices.into();
        sampler
    }
    
    /// Create seeded random sampler for reproducible results
    /// 再現可能な結果のためのシード付きランダムサンプラーを作成
    pub fn with_seed(data_source_len: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..data_source_len).collect();
        indices.shuffle(&mut rng);
        
        Self {
            indices: indices.into(),
            original_len: data_source_len,
            replacement: false,
            generator: Some(seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self) -> Option<usize> {
        if self.replacement && self.indices.is_empty() {
            // Generate new random index for replacement sampling
            let mut rng = thread_rng();
            Some(rng.gen_range(0..self.original_len))
        } else {
            self.indices.pop_front()
        }
    }
    
    fn len(&self) -> usize {
        if self.replacement {
            usize::MAX // Infinite for replacement sampling
        } else {
            self.original_len
        }
    }
    
    fn reset(&mut self) {
        if !self.replacement {
            let mut indices: Vec<usize> = (0..self.original_len).collect();
            
            if let Some(seed) = self.generator {
                use rand::SeedableRng;
                use rand::rngs::StdRng;
                let mut rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                indices.shuffle(&mut thread_rng());
            }
            
            self.indices = indices.into();
        }
    }
    
    fn is_empty(&self) -> bool {
        !self.replacement && self.indices.is_empty()
    }
}

/// Batch sampler - wraps another sampler to yield batches of indices
/// バッチサンプラー - 他のサンプラーをラップしてインデックスのバッチを生成
pub struct BatchSampler {
    sampler: Box<dyn Sampler + Send + Sync>,
    batch_size: usize,
    drop_last: bool,
}

impl BatchSampler {
    /// Create new batch sampler
    /// 新しいバッチサンプラーを作成
    pub fn new(
        sampler: Box<dyn Sampler + Send + Sync>,
        batch_size: usize,
        drop_last: bool,
    ) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }
    
    /// Get next batch of indices
    /// 次のインデックスバッチを取得
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let mut batch = Vec::new();
        
        for _ in 0..self.batch_size {
            if let Some(idx) = self.sampler.sample() {
                batch.push(idx);
            } else {
                break;
            }
        }
        
        if batch.is_empty() {
            None
        } else if self.drop_last && batch.len() < self.batch_size {
            None
        } else {
            Some(batch)
        }
    }
    
    /// Get batch size
    /// バッチサイズを取得
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    
    /// Check if dropping last incomplete batch
    /// 最後の不完全なバッチを破棄するかチェック
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }
}

impl Sampler for BatchSampler {
    fn sample(&mut self) -> Option<usize> {
        // BatchSampler doesn't return individual indices
        // Use next_batch() instead
        None
    }
    
    fn len(&self) -> usize {
        let base_len = self.sampler.len();
        if base_len == usize::MAX {
            return usize::MAX; // Infinite sampling
        }
        
        if self.drop_last {
            base_len / self.batch_size
        } else {
            (base_len + self.batch_size - 1) / self.batch_size
        }
    }
    
    fn reset(&mut self) {
        self.sampler.reset();
    }
    
    fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }
}

/// Subset random sampler - samples from a subset of indices
/// サブセットランダムサンプラー - インデックスのサブセットからサンプル
pub struct SubsetRandomSampler {
    indices: VecDeque<usize>,
    original_indices: Vec<usize>,
}

impl SubsetRandomSampler {
    /// Create sampler for subset of indices
    /// インデックスのサブセット用サンプラーを作成
    pub fn new(indices: Vec<usize>) -> Self {
        let mut shuffled = indices.clone();
        shuffled.shuffle(&mut thread_rng());
        
        Self {
            indices: shuffled.into(),
            original_indices: indices,
        }
    }
}

impl Sampler for SubsetRandomSampler {
    fn sample(&mut self) -> Option<usize> {
        self.indices.pop_front()
    }
    
    fn len(&self) -> usize {
        self.original_indices.len()
    }
    
    fn reset(&mut self) {
        let mut shuffled = self.original_indices.clone();
        shuffled.shuffle(&mut thread_rng());
        self.indices = shuffled.into();
    }
    
    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Weighted random sampler
/// 重み付きランダムサンプラー
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    current_count: usize,
}

impl WeightedRandomSampler {
    /// Create weighted random sampler
    /// 重み付きランダムサンプラーを作成
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Result<Self, DataError> {
        if weights.is_empty() {
            return Err(RusTorchError::InvalidParameters {
                operation: "WeightedRandomSampler::new".to_string(),
                message: "Weights cannot be empty".to_string(),
            });
        }
        
        // Validate weights are non-negative
        for (i, &weight) in weights.iter().enumerate() {
            if weight < 0.0 {
                return Err(RusTorchError::InvalidParameters {
                    operation: "WeightedRandomSampler::new".to_string(),
                    message: format!("Weight at index {} is negative: {}", i, weight),
                });
            }
        }
        
        Ok(Self {
            weights,
            num_samples,
            replacement,
            current_count: 0,
        })
    }
    
    /// Sample index based on weights
    /// 重みに基づいてインデックスをサンプル
    fn sample_weighted(&self) -> Option<usize> {
        use rand::Rng;
        
        let total_weight: f64 = self.weights.iter().sum();
        if total_weight <= 0.0 {
            return None;
        }
        
        let mut rng = thread_rng();
        let target = rng.gen::<f64>() * total_weight;
        let mut cumulative = 0.0;
        
        for (i, &weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                return Some(i);
            }
        }
        
        // Fallback to last index
        Some(self.weights.len() - 1)
    }
}

impl Sampler for WeightedRandomSampler {
    fn sample(&mut self) -> Option<usize> {
        if !self.replacement && self.current_count >= self.num_samples {
            return None;
        }
        
        if let Some(index) = self.sample_weighted() {
            if !self.replacement {
                self.current_count += 1;
            }
            Some(index)
        } else {
            None
        }
    }
    
    fn len(&self) -> usize {
        self.num_samples
    }
    
    fn reset(&mut self) {
        self.current_count = 0;
    }
    
    fn is_empty(&self) -> bool {
        !self.replacement && self.current_count >= self.num_samples
    }
}