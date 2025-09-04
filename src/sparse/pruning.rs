//! Model pruning algorithms for sparsification
//! モデルスパース化用プルーニングアルゴリズム

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use super::SparseTensor;
use ndarray::{ArrayD, Array1, Array2, s};
use num_traits::{Float, Zero, One, FromPrimitive};
use std::cmp::Ordering;

/// Pruning strategy enumeration
/// プルーニング戦略列挙
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningStrategy {
    /// Magnitude-based pruning - remove smallest weights
    /// マグニチュードベースプルーニング - 最小重みを削除
    Magnitude,
    /// Random pruning for baseline comparison
    /// ベースライン比較用ランダムプルーニング
    Random,
    /// Structured pruning - remove entire neurons/channels
    /// 構造化プルーニング - ニューロン/チャンネル全体を削除
    Structured,
    /// Gradient-based pruning using importance scores
    /// 重要度スコアを使用した勾配ベースプルーニング
    GradientBased,
    /// SNIP (Single-shot Network Pruning)
    /// SNIP（シングルショットネットワークプルーニング）
    SNIP,
}

/// Pruning configuration and parameters
/// プルーニング設定とパラメータ
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Target sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    /// 目標スパース率（0.0 = 密, 1.0 = 完全スパース）
    pub target_sparsity: f32,
    /// Pruning strategy to use
    /// 使用するプルーニング戦略
    pub strategy: PruningStrategy,
    /// Whether to prune structured (entire neurons) or unstructured (individual weights)
    /// 構造化（ニューロン全体）または非構造化（個別重み）プルーニング
    pub structured: bool,
    /// Gradual pruning schedule (None for one-shot pruning)
    /// 段階的プルーニングスケジュール（ワンショットプルーニングの場合はNone）
    pub schedule: Option<PruningSchedule>,
}

/// Gradual pruning schedule
/// 段階的プルーニングスケジュール
#[derive(Debug, Clone)]
pub struct PruningSchedule {
    /// Initial sparsity
    /// 初期スパース率
    pub initial_sparsity: f32,
    /// Final sparsity
    /// 最終スパース率
    pub final_sparsity: f32,
    /// Number of pruning steps
    /// プルーニングステップ数
    pub num_steps: usize,
    /// Current step
    /// 現在のステップ
    pub current_step: usize,
}

impl PruningSchedule {
    /// Create a new gradual pruning schedule
    /// 新しい段階的プルーニングスケジュールを作成
    pub fn new(initial_sparsity: f32, final_sparsity: f32, num_steps: usize) -> Self {
        Self {
            initial_sparsity,
            final_sparsity,
            num_steps,
            current_step: 0,
        }
    }

    /// Get current target sparsity based on schedule
    /// スケジュールに基づく現在の目標スパース率を取得
    pub fn current_sparsity(&self) -> f32 {
        if self.current_step >= self.num_steps {
            return self.final_sparsity;
        }
        
        let progress = self.current_step as f32 / self.num_steps as f32;
        self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
    }

    /// Advance to next pruning step
    /// 次のプルーニングステップに進む
    pub fn step(&mut self) {
        self.current_step = (self.current_step + 1).min(self.num_steps);
    }
}

/// Model pruner for applying sparsification algorithms
/// スパース化アルゴリズム適用用モデルプルーナー
pub struct ModelPruner<T: Float> {
    /// Pruning configuration
    /// プルーニング設定
    pub config: PruningConfig,
    /// Importance scores for gradient-based pruning
    /// 勾配ベースプルーニング用重要度スコア
    pub importance_scores: HashMap<String, Array1<T>>,
}

use std::collections::HashMap;

impl<T: Float + PartialOrd + Copy + Send + Sync + ndarray::ScalarOperand + FromPrimitive + std::ops::AddAssign> ModelPruner<T> {
    /// Create a new model pruner
    /// 新しいモデルプルーナーを作成
    pub fn new(config: PruningConfig) -> Self {
        Self {
            config,
            importance_scores: HashMap::new(),
        }
    }

    /// Prune a tensor based on the configured strategy
    /// 設定された戦略に基づいてテンソルをプルーニング
    pub fn prune_tensor(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        match self.config.strategy {
            PruningStrategy::Magnitude => self.magnitude_pruning(tensor),
            PruningStrategy::Random => self.random_pruning(tensor),
            PruningStrategy::Structured => self.structured_pruning(tensor),
            PruningStrategy::GradientBased => self.gradient_based_pruning(tensor),
            PruningStrategy::SNIP => self.snip_pruning(tensor),
        }
    }

    /// Magnitude-based pruning - keep largest magnitude weights
    /// マグニチュードベースプルーニング - 最大マグニチュード重みを保持
    fn magnitude_pruning(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        let target_sparsity = self.get_current_sparsity();
        let total_elements = tensor.len();
        let elements_to_keep = ((1.0 - target_sparsity) * total_elements as f32) as usize;

        // Calculate magnitudes and sort indices
        let mut magnitude_indices: Vec<(usize, T)> = tensor
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();

        magnitude_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Keep top elements_to_keep weights
        let kept_indices: Vec<usize> = magnitude_indices
            .iter()
            .take(elements_to_keep)
            .map(|(idx, _)| *idx)
            .collect();

        self.create_sparse_from_indices(tensor, &kept_indices)
    }

    /// Random pruning for baseline comparison
    /// ベースライン比較用ランダムプルーニング
    fn random_pruning(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        let target_sparsity = self.get_current_sparsity();
        let total_elements = tensor.len();
        let elements_to_keep = ((1.0 - target_sparsity) * total_elements as f32) as usize;

        // Generate random indices to keep
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut all_indices: Vec<usize> = (0..total_elements).collect();
        all_indices.shuffle(&mut rng);
        
        let kept_indices = &all_indices[..elements_to_keep];
        self.create_sparse_from_indices(tensor, kept_indices)
    }

    /// Structured pruning - remove entire neurons or channels
    /// 構造化プルーニング - ニューロンまたはチャンネル全体を削除
    fn structured_pruning(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        if tensor.ndim() < 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "structured_pruning".to_string(),
                message: "Structured pruning requires at least 2D tensors".to_string(),
            });
        }

        let target_sparsity = self.get_current_sparsity();
        
        // For 2D tensors, prune entire rows (neurons)
        if tensor.ndim() == 2 {
            let rows = tensor.shape()[0];
            let rows_to_keep = ((1.0 - target_sparsity) * rows as f32) as usize;
            
            // Calculate L2 norm for each row
            let mut row_norms: Vec<(usize, T)> = (0..rows)
                .map(|i| {
                    let row = tensor.slice(s![i, ..]);
                    let norm_sq = row.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
                    (i, norm_sq.sqrt())
                })
                .collect();
            
            // Sort by norm (descending)
            row_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            
            // Keep top rows
            let kept_rows: Vec<usize> = row_norms
                .iter()
                .take(rows_to_keep)
                .map(|(idx, _)| *idx)
                .collect();

            // Create indices for kept elements
            let mut kept_indices = Vec::new();
            let cols = tensor.shape()[1];
            
            for &row in &kept_rows {
                for col in 0..cols {
                    kept_indices.push(row * cols + col);
                }
            }

            return self.create_sparse_from_indices(tensor, &kept_indices);
        }

        // For higher-dimensional tensors, fall back to magnitude pruning
        self.magnitude_pruning(tensor)
    }

    /// Gradient-based pruning using importance scores
    /// 重要度スコアを使用した勾配ベースプルーニング
    fn gradient_based_pruning(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        // This is a simplified implementation
        // In practice, would use accumulated gradient information
        self.magnitude_pruning(tensor)
    }

    /// SNIP (Single-shot Network Pruning) based on connection sensitivity
    /// 接続感度に基づくSNIP（シングルショットネットワークプルーニング）
    fn snip_pruning(&self, tensor: &ArrayD<T>) -> RusTorchResult<SparseTensor<T>> {
        // SNIP uses gradient information at initialization
        // For now, fall back to magnitude-based pruning
        self.magnitude_pruning(tensor)
    }

    /// Helper function to create sparse tensor from kept indices
    /// 保持インデックスからスパーステンソルを作成するヘルパー関数
    fn create_sparse_from_indices(
        &self, 
        tensor: &ArrayD<T>, 
        kept_indices: &[usize]
    ) -> RusTorchResult<SparseTensor<T>> {
        let shape = tensor.shape().to_vec();
        let mut indices_per_dim = vec![Vec::new(); shape.len()];
        let mut values = Vec::new();

        let flat_tensor = tensor.as_slice().unwrap();

        for &flat_idx in kept_indices {
            if flat_idx >= flat_tensor.len() {
                continue;
            }

            let value = flat_tensor[flat_idx];
            if !value.is_zero() {
                values.push(value);
                
                // Convert flat index to multi-dimensional coordinates
                let mut remaining_idx = flat_idx;
                for (dim, &dim_size) in shape.iter().enumerate().rev() {
                    let coord = remaining_idx % dim_size;
                    indices_per_dim[shape.len() - 1 - dim].push(coord);
                    remaining_idx /= dim_size;
                }
            }
        }

        let indices: Vec<Array1<usize>> = indices_per_dim
            .into_iter()
            .map(|v| Array1::from_vec(v))
            .collect();
        let values_array = Array1::from_vec(values);

        SparseTensor::from_coo(indices, values_array, shape)
    }

    /// Get current sparsity target based on schedule
    /// スケジュールに基づく現在のスパース率目標を取得
    fn get_current_sparsity(&self) -> f32 {
        match &self.config.schedule {
            Some(schedule) => schedule.current_sparsity(),
            None => self.config.target_sparsity,
        }
    }

    /// Update importance scores for gradient-based pruning
    /// 勾配ベースプルーニング用重要度スコアを更新
    pub fn update_importance_scores(&mut self, param_name: &str, gradients: &ArrayD<T>) {
        // Calculate importance as magnitude of gradients
        let importance: Array1<T> = gradients
            .iter()
            .map(|&grad| grad.abs())
            .collect();
        
        self.importance_scores.insert(param_name.to_string(), importance);
    }

    /// Prune an entire model (collection of parameters)
    /// モデル全体（パラメータ集合）をプルーニング
    pub fn prune_model(
        &mut self,
        parameters: &HashMap<String, Variable<T>>
    ) -> RusTorchResult<HashMap<String, SparseTensor<T>>> {
        let mut pruned_params = HashMap::new();

        for (name, param) in parameters.iter() {
            let param_tensor = param.data();
            let param_guard = param_tensor.read().unwrap();
            
            let sparse_param = self.prune_tensor(&param_guard.data)?;
            pruned_params.insert(name.clone(), sparse_param);
        }

        // Update schedule if gradual pruning
        if let Some(ref mut schedule) = self.config.schedule {
            schedule.step();
        }

        Ok(pruned_params)
    }
}

/// Specific pruning algorithms implementation
/// 特定プルーニングアルゴリズム実装
pub struct MagnitudePruner<T: Float> {
    /// Global or layer-wise pruning
    /// グローバルまたはレイヤー単位プルーニング
    pub global: bool,
    /// Target sparsity
    /// 目標スパース率
    pub sparsity: f32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + PartialOrd + Copy> MagnitudePruner<T> {
    /// Create magnitude-based pruner
    /// マグニチュードベースプルーナーを作成
    pub fn new(sparsity: f32, global: bool) -> Self {
        Self {
            global,
            sparsity: sparsity.clamp(0.0, 1.0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply magnitude pruning to tensor
    /// テンソルにマグニチュードプルーニングを適用
    pub fn prune(&self, tensor: &ArrayD<T>) -> RusTorchResult<ArrayD<T>> {
        let flat_tensor = tensor.as_slice().unwrap();
        let total_elements = flat_tensor.len();
        let elements_to_zero = (self.sparsity * total_elements as f32) as usize;

        // Calculate absolute values and sort indices
        let mut magnitude_indices: Vec<(usize, T)> = flat_tensor
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();

        magnitude_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Create pruned tensor
        let mut pruned = tensor.clone();
        let mut pruned_flat = pruned.as_slice_mut().unwrap();
        
        for i in 0..elements_to_zero.min(magnitude_indices.len()) {
            let idx = magnitude_indices[i].0;
            pruned_flat[idx] = T::zero();
        }

        Ok(pruned)
    }
}

/// Structured pruning for neural network layers
/// ニューラルネットワーク層の構造化プルーニング
pub struct StructuredPruner<T: Float> {
    /// Granularity: neuron, channel, or filter
    /// 粒度：ニューロン、チャンネル、フィルター
    pub granularity: StructuredGranularity,
    /// Target pruning ratio
    /// 目標プルーニング率
    pub ratio: f32,
    _phantom: std::marker::PhantomData<T>,
}

/// Structured pruning granularity options
/// 構造化プルーニング粒度オプション
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StructuredGranularity {
    /// Remove entire neurons (rows in weight matrix)
    /// ニューロン全体を削除（重み行列の行）
    Neuron,
    /// Remove entire channels (for convolutional layers)
    /// チャンネル全体を削除（畳み込み層用）
    Channel,
    /// Remove entire filters (for convolutional layers)
    /// フィルター全体を削除（畳み込み層用）
    Filter,
}

impl<T: Float + PartialOrd + Copy> StructuredPruner<T> {
    /// Create structured pruner
    /// 構造化プルーナーを作成
    pub fn new(granularity: StructuredGranularity, ratio: f32) -> Self {
        Self {
            granularity,
            ratio: ratio.clamp(0.0, 1.0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply structured pruning to 2D weight matrix
    /// 2D重み行列に構造化プルーニングを適用
    pub fn prune_linear_weights(&self, weights: &Array2<T>) -> RusTorchResult<Array2<T>> {
        let (rows, cols) = weights.dim();
        
        match self.granularity {
            StructuredGranularity::Neuron => {
                let neurons_to_prune = (self.ratio * rows as f32) as usize;
                
                // Calculate L2 norm for each neuron (row)
                let mut neuron_norms: Vec<(usize, T)> = (0..rows)
                    .map(|i| {
                        let row = weights.row(i);
                        let norm_sq = row.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
                        (i, norm_sq.sqrt())
                    })
                    .collect();

                // Sort by norm (ascending - prune smallest)
                neuron_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                // Zero out smallest neurons
                let mut pruned_weights = weights.clone();
                for i in 0..neurons_to_prune.min(neuron_norms.len()) {
                    let neuron_idx = neuron_norms[i].0;
                    for j in 0..cols {
                        pruned_weights[[neuron_idx, j]] = T::zero();
                    }
                }

                Ok(pruned_weights)
            }
            _ => {
                // For other granularities, fall back to magnitude pruning
                let flattened = weights.clone().into_dyn();
                let magnitude_pruner = MagnitudePruner::new(self.ratio, false);
                let pruned_flat = magnitude_pruner.prune(&flattened)?;
                
                Ok(Array2::from_shape_vec((rows, cols), pruned_flat.into_raw_vec_and_offset().0)?)
            }
        }
    }
}

/// Fisher information-based pruning
/// フィッシャー情報ベースプルーニング
pub struct FisherPruner<T: Float> {
    /// Accumulated Fisher information matrix
    /// 累積フィッシャー情報行列
    pub fisher_info: HashMap<String, ArrayD<T>>,
    /// Number of samples used for Fisher estimation
    /// フィッシャー推定に使用されたサンプル数
    pub n_samples: usize,
}

impl<T: Float + std::ops::AddAssign + Copy + ndarray::ScalarOperand + Send + Sync + FromPrimitive> FisherPruner<T> {
    /// Create Fisher information pruner
    /// フィッシャー情報プルーナーを作成
    pub fn new() -> Self {
        Self {
            fisher_info: HashMap::new(),
            n_samples: 0,
        }
    }

    /// Update Fisher information with gradients
    /// 勾配でフィッシャー情報を更新
    pub fn update_fisher(&mut self, param_name: &str, gradients: &ArrayD<T>) {
        // Fisher information approximation: E[∇log p(x|θ)²]
        let squared_grads = gradients.mapv(|g| g * g);
        
        match self.fisher_info.get_mut(param_name) {
            Some(existing) => {
                // Running average update
                let alpha = T::one() / T::from(self.n_samples + 1).unwrap();
                *existing = &*existing * (T::one() - alpha) + &squared_grads * alpha;
            }
            None => {
                self.fisher_info.insert(param_name.to_string(), squared_grads);
            }
        }
        
        self.n_samples += 1;
    }

    /// Prune based on Fisher information scores
    /// フィッシャー情報スコアに基づくプルーニング
    pub fn prune_with_fisher(
        &self, 
        param_name: &str, 
        tensor: &ArrayD<T>, 
        target_sparsity: f32
    ) -> RusTorchResult<SparseTensor<T>> {
        let fisher_scores = self.fisher_info.get(param_name)
            .ok_or_else(|| RusTorchError::InvalidParameters {
                operation: "fisher_pruning".to_string(),
                message: format!("No Fisher information available for parameter: {}", param_name),
            })?;

        if fisher_scores.shape() != tensor.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: tensor.shape().to_vec(),
                actual: fisher_scores.shape().to_vec(),
            });
        }

        let total_elements = tensor.len();
        let elements_to_keep = ((1.0 - target_sparsity) * total_elements as f32) as usize;

        // Sort by Fisher scores (keep highest importance)
        let mut fisher_indices: Vec<(usize, T)> = fisher_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        fisher_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let kept_indices: Vec<usize> = fisher_indices
            .iter()
            .take(elements_to_keep)
            .map(|(idx, _)| *idx)
            .collect();

        let model_pruner = ModelPruner::new(PruningConfig {
            target_sparsity,
            strategy: PruningStrategy::GradientBased,
            structured: false,
            schedule: None,
        });

        model_pruner.create_sparse_from_indices(tensor, &kept_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_magnitude_pruning() {
        let config = PruningConfig {
            target_sparsity: 0.5,
            strategy: PruningStrategy::Magnitude,
            structured: false,
            schedule: None,
        };
        
        let pruner = ModelPruner::new(config);
        
        let tensor = Array2::from_shape_vec((2, 3), vec![1.0f32, -2.0, 0.5, -4.0, 3.0, 0.1]).unwrap().into_dyn();
        let sparse_result = pruner.prune_tensor(&tensor).unwrap();
        
        // Should keep roughly 50% of elements (3 out of 6)
        assert!(sparse_result.nnz <= 3);
        assert!(sparse_result.sparsity() >= 0.4);
    }

    #[test]
    fn test_structured_pruning() {
        let structured_pruner = StructuredPruner::new(StructuredGranularity::Neuron, 0.5);
        
        let weights = Array2::from_shape_vec((4, 3), vec![
            1.0f32, 2.0, 3.0,  // Strong neuron
            0.1, 0.1, 0.1,     // Weak neuron
            -2.0, 1.5, -1.0,   // Medium neuron
            0.05, 0.02, 0.03,  // Very weak neuron
        ]).unwrap();
        
        let pruned = structured_pruner.prune_linear_weights(&weights).unwrap();
        
        // Should remove 2 weakest neurons (rows)
        let zero_rows = (0..4).filter(|&i| {
            pruned.row(i).iter().all(|&x| x == 0.0)
        }).count();
        
        assert_eq!(zero_rows, 2);
    }

    #[test]
    fn test_pruning_schedule() {
        let mut schedule = PruningSchedule::new(0.0, 0.9, 10);
        
        assert_eq!(schedule.current_sparsity(), 0.0);
        
        schedule.step();
        assert!(schedule.current_sparsity() > 0.0 && schedule.current_sparsity() < 0.9);
        
        // Advance to end
        for _ in 0..10 {
            schedule.step();
        }
        assert_eq!(schedule.current_sparsity(), 0.9);
    }

    #[test]
    fn test_fisher_pruner() {
        let mut fisher_pruner = FisherPruner::new();
        
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1f32, 0.9, 0.3, 0.7]).unwrap().into_dyn();
        fisher_pruner.update_fisher("layer1", &gradients);
        
        let weights = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap().into_dyn();
        let sparse_result = fisher_pruner.prune_with_fisher("layer1", &weights, 0.5).unwrap();
        
        // Should keep elements with higher Fisher scores
        assert!(sparse_result.nnz == 2);
    }
}