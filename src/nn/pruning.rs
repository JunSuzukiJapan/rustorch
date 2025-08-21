//! Neural Network Pruning for model compression and acceleration
//! モデル圧縮と高速化のためのニューラルネットワークプルーニング
//!
//! This module provides comprehensive pruning techniques including:
//! - Magnitude-based pruning (weight magnitude pruning)
//! - Structured pruning (channel/filter pruning)
//! - Unstructured pruning (fine-grained sparsity)
//! - Gradual magnitude pruning (progressive sparsification)
//! - Lottery ticket hypothesis implementation
//! 
//! 包括的なプルーニング技術を提供：
//! - 大きさベースのプルーニング（重み大きさプルーニング）
//! - 構造化プルーニング（チャンネル/フィルタプルーニング）
//! - 非構造化プルーニング（細粒度スパース性）
//! - 段階的大きさプルーニング（漸進的スパース化）
//! - 宝くじ仮説の実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero, One};
use ndarray::{ScalarOperand, Array, IxDyn};
use std::iter::Sum;
use std::collections::{HashMap, HashSet};

/// Pruning method types
/// プルーニング手法の種類
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMethod {
    /// L1 norm based pruning (sum of absolute values)
    /// L1ノルムベースのプルーニング（絶対値の和）
    L1Norm,
    /// L2 norm based pruning (Euclidean norm)
    /// L2ノルムベースのプルーニング（ユークリッドノルム）
    L2Norm,
    /// Random pruning (baseline)
    /// ランダムプルーニング（ベースライン）
    Random,
    /// Gradient-based pruning
    /// 勾配ベースのプルーニング
    Gradient,
    /// Taylor expansion based pruning
    /// テイラー展開ベースのプルーニング
    Taylor,
}

/// Pruning structure types
/// プルーニング構造の種類
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningStructure {
    /// Unstructured pruning (individual weights)
    /// 非構造化プルーニング（個別の重み）
    Unstructured,
    /// Structured pruning by channels
    /// チャンネル単位の構造化プルーニング
    Channel,
    /// Structured pruning by filters
    /// フィルタ単位の構造化プルーニング
    Filter,
    /// Block-wise pruning (n:m sparsity)
    /// ブロック単位のプルーニング（n:mスパース性）
    Block { 
        /// Number of weights to keep in each block
        /// 各ブロックで保持する重みの数
        n: usize, 
        /// Block size
        /// ブロックサイズ
        m: usize 
    },
}

/// Pruning schedule for gradual pruning
/// 段階的プルーニングのスケジュール
#[derive(Debug, Clone)]
pub enum PruningSchedule {
    /// One-shot pruning (prune all at once)
    /// ワンショットプルーニング（一度に全てプルーニング）
    OneShot,
    /// Linear schedule (linearly increase sparsity)
    /// 線形スケジュール（線形にスパース性を増加）
    Linear { 
        /// Epoch to start pruning
        /// プルーニング開始エポック
        start_epoch: usize, 
        /// Epoch to end pruning
        /// プルーニング終了エポック
        end_epoch: usize 
    },
    /// Polynomial schedule (polynomial decay)
    /// 多項式スケジュール（多項式減衰）
    Polynomial { 
        /// Epoch to start pruning
        /// プルーニング開始エポック
        start_epoch: usize, 
        /// Epoch to end pruning
        /// プルーニング終了エポック
        end_epoch: usize, 
        /// Power factor for polynomial decay
        /// 多項式減衰のべき乗係数
        power: f32 
    },
    /// Exponential schedule
    /// 指数スケジュール
    Exponential { 
        /// Epoch to start pruning
        /// プルーニング開始エポック
        start_epoch: usize, 
        /// Epoch to end pruning
        /// プルーニング終了エポック
        end_epoch: usize 
    },
}

/// Pruning mask for a tensor
/// テンソル用プルーニングマスク
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Binary mask (1 = keep, 0 = prune)
    /// バイナリマスク（1 = 保持、0 = プルーニング）
    pub mask: Array<u8, IxDyn>,
    /// Sparsity level (percentage of pruned weights)
    /// スパース性レベル（プルーニングされた重みの割合）
    pub sparsity: f32,
    /// Number of pruned elements
    /// プルーニングされた要素数
    pub pruned_count: usize,
    /// Total number of elements
    /// 総要素数
    pub total_count: usize,
}

impl PruningMask {
    /// Create a new pruning mask
    /// 新しいプルーニングマスクを作成
    pub fn new(mask: Array<u8, IxDyn>) -> Self {
        let total_count = mask.len();
        let pruned_count = mask.iter().filter(|&&x| x == 0).count();
        let sparsity = pruned_count as f32 / total_count as f32;
        
        PruningMask {
            mask,
            sparsity,
            pruned_count,
            total_count,
        }
    }
    
    /// Get the compression ratio achieved
    /// 達成された圧縮比を取得
    pub fn compression_ratio(&self) -> f32 {
        1.0 / (1.0 - self.sparsity)
    }
    
    /// Get the number of remaining parameters
    /// 残りのパラメータ数を取得
    pub fn remaining_params(&self) -> usize {
        self.total_count - self.pruned_count
    }
}

/// Pruner for neural network models
/// ニューラルネットワークモデル用プルーナー
#[derive(Debug)]
pub struct Pruner<T: Float> {
    /// Pruning method
    /// プルーニング手法
    method: PruningMethod,
    /// Pruning structure
    /// プルーニング構造
    structure: PruningStructure,
    /// Target sparsity level
    /// 目標スパース性レベル
    target_sparsity: f32,
    /// Pruning schedule
    /// プルーニングスケジュール
    schedule: PruningSchedule,
    /// Current epoch for gradual pruning
    /// 段階的プルーニング用の現在のエポック
    current_epoch: usize,
    /// Pruning masks for each layer
    /// 各層のプルーニングマスク
    masks: HashMap<String, PruningMask>,
    /// Original weights for lottery ticket hypothesis
    /// 宝くじ仮説用の元の重み
    original_weights: HashMap<String, Tensor<T>>,
}

impl<T> Pruner<T>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Default + Zero + One + Send + Sync + Copy + ScalarOperand + Sum,
{
    /// Create a new pruner
    /// 新しいプルーナーを作成
    pub fn new(
        method: PruningMethod,
        structure: PruningStructure,
        target_sparsity: f32,
        schedule: PruningSchedule,
    ) -> Self {
        assert!(target_sparsity >= 0.0 && target_sparsity < 1.0, 
                "Target sparsity must be in [0, 1)");
        
        Pruner {
            method,
            structure,
            target_sparsity,
            schedule,
            current_epoch: 0,
            masks: HashMap::new(),
            original_weights: HashMap::new(),
        }
    }
    
    /// Compute importance scores for weights
    /// 重みの重要度スコアを計算
    fn compute_importance_scores(&self, weights: &Tensor<T>) -> Array<T, IxDyn> {
        let weights_array = weights.as_array();
        
        match self.method {
            PruningMethod::L1Norm => {
                // L1 norm: absolute values
                weights_array.mapv(|x| x.abs())
            },
            PruningMethod::L2Norm => {
                // L2 norm: squared values
                weights_array.mapv(|x| x * x)
            },
            PruningMethod::Random => {
                // Random scores
                Array::from_shape_vec(
                    weights_array.shape(),
                    (0..weights_array.len())
                        .map(|_| T::from_f32(rand::random::<f32>()).unwrap())
                        .collect()
                ).unwrap()
            },
            PruningMethod::Gradient => {
                // For gradient-based, we'd need gradient information
                // Simplified: use weight magnitude as proxy
                weights_array.mapv(|x| x.abs())
            },
            PruningMethod::Taylor => {
                // Taylor expansion: weight * gradient approximation
                // Simplified: use weight squared
                weights_array.mapv(|x| x * x)
            },
        }
    }
    
    /// Create pruning mask based on importance scores
    /// 重要度スコアに基づいてプルーニングマスクを作成
    fn create_mask(&self, scores: &Array<T, IxDyn>, sparsity: f32) -> PruningMask {
        let total_elements = scores.len();
        let num_to_prune = (total_elements as f32 * sparsity) as usize;
        
        match self.structure {
            PruningStructure::Unstructured => {
                // Flatten scores and find threshold
                let mut flat_scores: Vec<T> = scores.iter().cloned().collect();
                flat_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let threshold = if num_to_prune < total_elements {
                    flat_scores[num_to_prune]
                } else {
                    T::infinity()
                };
                
                // Create binary mask
                let mask = scores.mapv(|x| if x > threshold { 1u8 } else { 0u8 });
                PruningMask::new(mask)
            },
            PruningStructure::Channel => {
                // Channel-wise pruning for Conv2D (shape: [out_channels, in_channels, H, W])
                if scores.ndim() == 4 {
                    let out_channels = scores.shape()[0];
                    let channels_to_prune = (out_channels as f32 * sparsity) as usize;
                    
                    // Compute channel importance (L2 norm across channel)
                    let mut channel_scores = vec![T::zero(); out_channels];
                    for c in 0..out_channels {
                        let channel_slice = scores.index_axis(ndarray::Axis(0), c);
                        channel_scores[c] = channel_slice.iter().map(|&x| x * x).sum::<T>().sqrt();
                    }
                    
                    // Find channels to prune
                    let mut sorted_indices: Vec<usize> = (0..out_channels).collect();
                    sorted_indices.sort_by(|&a, &b| 
                        channel_scores[a].partial_cmp(&channel_scores[b]).unwrap());
                    
                    let pruned_channels: HashSet<usize> = 
                        sorted_indices.iter().take(channels_to_prune).cloned().collect();
                    
                    // Create mask
                    let mut mask = Array::ones(scores.raw_dim());
                    for c in pruned_channels {
                        mask.index_axis_mut(ndarray::Axis(0), c).fill(0);
                    }
                    
                    PruningMask::new(mask)
                } else {
                    // Fallback to unstructured for non-conv layers
                    self.create_unstructured_mask(scores, sparsity)
                }
            },
            PruningStructure::Filter => {
                // Filter-wise pruning (similar to channel but for input dimension)
                self.create_unstructured_mask(scores, sparsity) // Simplified
            },
            PruningStructure::Block { n, m } => {
                // Block-wise n:m sparsity (keep n weights in every m weights)
                self.create_block_mask(scores, n, m)
            },
        }
    }
    
    /// Create unstructured pruning mask
    /// 非構造化プルーニングマスクを作成
    fn create_unstructured_mask(&self, scores: &Array<T, IxDyn>, sparsity: f32) -> PruningMask {
        let total_elements = scores.len();
        let num_to_prune = (total_elements as f32 * sparsity) as usize;
        
        let mut flat_scores: Vec<T> = scores.iter().cloned().collect();
        flat_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let threshold = if num_to_prune < total_elements {
            flat_scores[num_to_prune]
        } else {
            T::infinity()
        };
        
        let mask = scores.mapv(|x| if x > threshold { 1u8 } else { 0u8 });
        PruningMask::new(mask)
    }
    
    /// Create block-wise n:m sparsity mask
    /// ブロック単位のn:mスパース性マスクを作成
    fn create_block_mask(&self, scores: &Array<T, IxDyn>, n: usize, m: usize) -> PruningMask {
        assert!(n <= m, "n must be <= m for n:m sparsity");
        
        let flat_scores: Vec<T> = scores.iter().cloned().collect();
        let mut flat_mask = vec![0u8; flat_scores.len()];
        
        // Process in blocks of size m
        for block_start in (0..flat_scores.len()).step_by(m) {
            let block_end = (block_start + m).min(flat_scores.len());
            let block_size = block_end - block_start;
            
            if block_size <= n {
                // Keep all in incomplete blocks
                for i in block_start..block_end {
                    flat_mask[i] = 1;
                }
            } else {
                // Find top n elements in this block
                let mut block_indices: Vec<usize> = (block_start..block_end).collect();
                block_indices.sort_by(|&a, &b| 
                    flat_scores[b].partial_cmp(&flat_scores[a]).unwrap());
                
                for i in block_indices.iter().take(n) {
                    flat_mask[*i] = 1;
                }
            }
        }
        
        let mask = Array::from_shape_vec(scores.raw_dim(), flat_mask).unwrap();
        PruningMask::new(mask)
    }
    
    /// Get current sparsity based on schedule
    /// スケジュールに基づいて現在のスパース性を取得
    fn get_current_sparsity(&self) -> f32 {
        match &self.schedule {
            PruningSchedule::OneShot => self.target_sparsity,
            PruningSchedule::Linear { start_epoch, end_epoch } => {
                if self.current_epoch < *start_epoch {
                    0.0
                } else if self.current_epoch >= *end_epoch {
                    self.target_sparsity
                } else {
                    let progress = (self.current_epoch - start_epoch) as f32 / 
                                  (*end_epoch - start_epoch) as f32;
                    self.target_sparsity * progress
                }
            },
            PruningSchedule::Polynomial { start_epoch, end_epoch, power } => {
                if self.current_epoch < *start_epoch {
                    0.0
                } else if self.current_epoch >= *end_epoch {
                    self.target_sparsity
                } else {
                    let progress = (self.current_epoch - start_epoch) as f32 / 
                                  (*end_epoch - start_epoch) as f32;
                    self.target_sparsity * progress.powf(*power)
                }
            },
            PruningSchedule::Exponential { start_epoch, end_epoch } => {
                if self.current_epoch < *start_epoch {
                    0.0
                } else if self.current_epoch >= *end_epoch {
                    self.target_sparsity
                } else {
                    let progress = (self.current_epoch - start_epoch) as f32 / 
                                  (*end_epoch - start_epoch) as f32;
                    self.target_sparsity * (1.0 - (-5.0 * progress).exp())
                }
            },
        }
    }
    
    /// Prune a single tensor
    /// 単一のテンソルをプルーニング
    pub fn prune_tensor(&mut self, tensor: &Tensor<T>, layer_name: &str) -> Tensor<T> {
        let current_sparsity = self.get_current_sparsity();
        
        // Store original weights if not already stored (for lottery ticket)
        if !self.original_weights.contains_key(layer_name) {
            self.original_weights.insert(layer_name.to_string(), tensor.clone());
        }
        
        // Compute importance scores
        let scores = self.compute_importance_scores(tensor);
        
        // Create or update mask
        let mask = self.create_mask(&scores, current_sparsity);
        self.masks.insert(layer_name.to_string(), mask.clone());
        
        // Apply mask to tensor
        let tensor_array = tensor.as_array();
        let pruned_array = tensor_array * mask.mask.mapv(|x| T::from_u8(x).unwrap());
        
        Tensor::new(pruned_array)
    }
    
    /// Apply pruning to a module
    /// モジュールにプルーニングを適用
    pub fn prune_module<M: Module<T>>(&mut self, module: &M, layer_prefix: &str) -> Vec<Tensor<T>> {
        let parameters = module.parameters();
        let mut pruned_params = Vec::new();
        
        for (i, param) in parameters.iter().enumerate() {
            let param_name = format!("{}_{}", layer_prefix, i);
            let param_tensor = param.data();
            let param_data = param_tensor.read().unwrap();
            
            let pruned = self.prune_tensor(&*param_data, &param_name);
            pruned_params.push(pruned);
        }
        
        pruned_params
    }
    
    /// Update epoch for gradual pruning
    /// 段階的プルーニング用のエポックを更新
    pub fn step_epoch(&mut self) {
        self.current_epoch += 1;
    }
    
    /// Reset weights to original initialization (lottery ticket hypothesis)
    /// 元の初期化に重みをリセット（宝くじ仮説）
    pub fn reset_to_original_weights(&self, layer_name: &str) -> Option<Tensor<T>> {
        self.original_weights.get(layer_name).cloned()
    }
    
    /// Get pruning statistics
    /// プルーニング統計を取得
    pub fn get_statistics(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        for (name, mask) in &self.masks {
            stats.insert(name.clone(), mask.sparsity);
        }
        
        stats
    }
    
    /// Get global sparsity across all layers
    /// 全層にわたるグローバルスパース性を取得
    pub fn get_global_sparsity(&self) -> f32 {
        if self.masks.is_empty() {
            return 0.0;
        }
        
        let total_pruned: usize = self.masks.values().map(|m| m.pruned_count).sum();
        let total_params: usize = self.masks.values().map(|m| m.total_count).sum();
        
        total_pruned as f32 / total_params as f32
    }
    
    /// Clear all masks
    /// 全てのマスクをクリア
    pub fn clear_masks(&mut self) {
        self.masks.clear();
    }
}

/// Pruning-aware training wrapper
/// プルーニング対応訓練ラッパー
#[derive(Debug)]
pub struct PruningAwareModule<T: Float + Send + Sync + 'static, M: Module<T> + 'static> {
    /// Underlying module
    /// 基底モジュール
    module: M,
    /// Pruner instance
    /// プルーナーインスタンス
    pruner: Pruner<T>,
    /// Whether pruning is enabled
    /// プルーニングが有効かどうか
    pruning_enabled: bool,
}

impl<T, M> PruningAwareModule<T, M>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Default + Zero + One + Send + Sync + Copy + ScalarOperand + Sum + 'static,
    M: Module<T> + 'static,
{
    /// Create a new pruning-aware module
    /// 新しいプルーニング対応モジュールを作成
    pub fn new(module: M, pruner: Pruner<T>) -> Self {
        PruningAwareModule {
            module,
            pruner,
            pruning_enabled: false,
        }
    }
    
    /// Enable pruning
    /// プルーニングを有効化
    pub fn enable_pruning(&mut self) {
        self.pruning_enabled = true;
    }
    
    /// Disable pruning
    /// プルーニングを無効化
    pub fn disable_pruning(&mut self) {
        self.pruning_enabled = false;
    }
    
    /// Step epoch for gradual pruning
    /// 段階的プルーニング用のエポックをステップ
    pub fn step_epoch(&mut self) {
        self.pruner.step_epoch();
    }
    
    /// Get pruning statistics
    /// プルーニング統計を取得
    pub fn get_statistics(&self) -> HashMap<String, f32> {
        self.pruner.get_statistics()
    }
}

impl<T, M> Module<T> for PruningAwareModule<T, M>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Default + Zero + One + Send + Sync + Copy + ScalarOperand + Sum + 'static,
    M: Module<T> + 'static,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // Forward through the module (pruning is applied to weights directly)
        self.module.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.module.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pruning_mask_creation() {
        let mask_array = Array::from_vec(vec![1u8, 0, 1, 0, 1, 0]).into_dyn();
        let mask = PruningMask::new(mask_array);
        
        assert_eq!(mask.total_count, 6);
        assert_eq!(mask.pruned_count, 3);
        assert_eq!(mask.sparsity, 0.5);
        assert_eq!(mask.compression_ratio(), 2.0);
    }
    
    #[test]
    fn test_pruner_creation() {
        let pruner = Pruner::<f32>::new(
            PruningMethod::L1Norm,
            PruningStructure::Unstructured,
            0.5,
            PruningSchedule::OneShot,
        );
        
        assert_eq!(pruner.target_sparsity, 0.5);
        assert_eq!(pruner.method, PruningMethod::L1Norm);
    }
    
    #[test]
    fn test_importance_scores() {
        let pruner = Pruner::<f32>::new(
            PruningMethod::L1Norm,
            PruningStructure::Unstructured,
            0.5,
            PruningSchedule::OneShot,
        );
        
        let weights = Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.5, 3.0, -3.0], vec![2, 3]);
        let scores = pruner.compute_importance_scores(&weights);
        
        // L1 norm should give absolute values
        assert_eq!(scores[[0, 0]], 1.0);
        assert_eq!(scores[[0, 1]], 2.0);
        assert_eq!(scores[[1, 2]], 3.0);
    }
    
    #[test]
    fn test_gradual_pruning_schedule() {
        let mut pruner = Pruner::<f32>::new(
            PruningMethod::L1Norm,
            PruningStructure::Unstructured,
            0.9,
            PruningSchedule::Linear { start_epoch: 0, end_epoch: 10 },
        );
        
        assert_eq!(pruner.get_current_sparsity(), 0.0);
        
        pruner.current_epoch = 5;
        assert!((pruner.get_current_sparsity() - 0.45).abs() < 0.01);
        
        pruner.current_epoch = 10;
        assert_eq!(pruner.get_current_sparsity(), 0.9);
    }
    
    #[test]
    fn test_block_sparsity() {
        let pruner = Pruner::<f32>::new(
            PruningMethod::L1Norm,
            PruningStructure::Block { n: 2, m: 4 },
            0.5,
            PruningSchedule::OneShot,
        );
        
        let scores = Array::from_shape_vec(
            vec![8],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ).unwrap().into_dyn();
        
        let mask = pruner.create_block_mask(&scores, 2, 4);
        
        // Should keep 2 out of every 4 elements (the highest scoring ones)
        assert_eq!(mask.remaining_params(), 4); // 2 per block, 2 blocks
    }
}