//! Common optimizer structures and traits for Phase 2 refactoring
//! フェーズ２リファクタリング用の共通最適化器構造とトレイト

use crate::error::{RusTorchError, RusTorchResult};
use crate::optim::utils::OptimizerUtils;
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Common optimizer state for Adam-based optimizers
/// Adam系最適化器の共通状態
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimate (momentum)
    /// 第一モーメント推定（モーメンタム）
    pub momentum: Tensor<f32>,

    /// Second moment estimate (velocity)  
    /// 第二モーメント推定（ベロシティ）
    pub velocity: Tensor<f32>,

    /// Additional state data for specialized optimizers
    /// 特殊最適化器用の追加状態データ
    pub extra_state: HashMap<String, Tensor<f32>>,
}

impl AdamState {
    /// Create new Adam state with given shape
    /// 指定された形状で新しいAdam状態を作成
    pub fn new(shape: &[usize]) -> Self {
        Self {
            momentum: Tensor::zeros(shape),
            velocity: Tensor::zeros(shape),
            extra_state: HashMap::new(),
        }
    }

    /// Add extra state tensor
    /// 追加状態テンソルを追加
    pub fn add_extra_state(&mut self, key: String, tensor: Tensor<f32>) {
        self.extra_state.insert(key, tensor);
    }

    /// Get extra state tensor
    /// 追加状態テンソルを取得
    pub fn get_extra_state(&self, key: &str) -> Option<&Tensor<f32>> {
        self.extra_state.get(key)
    }

    /// Get mutable extra state tensor  
    /// 可変追加状態テンソルを取得
    pub fn get_extra_state_mut(&mut self, key: &str) -> Option<&mut Tensor<f32>> {
        self.extra_state.get_mut(key)
    }
}

/// Configuration for Adam-based optimizers
/// Adam系最適化器の設定
#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamConfig {
    /// Create standard Adam configuration
    /// 標準Adam設定を作成
    pub fn adam(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Create NAdam configuration with recommended parameters
    /// 推奨パラメータでNAdam設定を作成
    pub fn nadam(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Create RAdam configuration  
    /// RAdam設定を作成
    pub fn radam(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Create Adamax configuration
    /// Adamax設定を作成
    pub fn adamax(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7, // Slightly different default for Adamax
            weight_decay: 0.0,
        }
    }

    /// Add weight decay to configuration
    /// 設定に重み減衰を追加
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Validate configuration parameters
    /// 設定パラメータを検証
    pub fn validate(&self) -> RusTorchResult<()> {
        if self.learning_rate <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adam optimizer config".to_string(),
                message: "Learning rate must be positive".to_string(),
            });
        }
        if self.beta1 < 0.0 || self.beta1 >= 1.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adam optimizer config".to_string(),
                message: "Beta1 must be in [0, 1)".to_string(),
            });
        }
        if self.beta2 < 0.0 || self.beta2 >= 1.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adam optimizer config".to_string(),
                message: "Beta2 must be in [0, 1)".to_string(),
            });
        }
        if self.eps <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adam optimizer config".to_string(),
                message: "Epsilon must be positive".to_string(),
            });
        }
        if self.weight_decay < 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adam optimizer config".to_string(),
                message: "Weight decay must be non-negative".to_string(),
            });
        }
        Ok(())
    }
}

/// Common utilities for Adam-based optimizers
/// Adam系最適化器の共通ユーティリティ
pub struct AdamUtils;

impl AdamUtils {
    /// Apply weight decay to gradient
    /// 勾配に重み減衰を適用
    pub fn apply_weight_decay(
        grad: &Tensor<f32>,
        param: &Tensor<f32>,
        weight_decay: f32,
    ) -> Tensor<f32> {
        if weight_decay > 0.0 {
            let weight_decay_term = param * weight_decay;
            grad + &weight_decay_term
        } else {
            grad.clone()
        }
    }

    /// Update momentum (first moment)
    /// モーメンタム（第一モーメント）を更新
    pub fn update_momentum(momentum: &mut Tensor<f32>, grad: &Tensor<f32>, beta1: f32) {
        let beta1_term = &*momentum * beta1;
        let grad_term = grad * (1.0 - beta1);
        *momentum = &beta1_term + &grad_term;
    }

    /// Update velocity (second moment)
    /// ベロシティ（第二モーメント）を更新  
    pub fn update_velocity(velocity: &mut Tensor<f32>, grad: &Tensor<f32>, beta2: f32) {
        let beta2_term = &*velocity * beta2;
        let grad_squared = grad * grad;
        let grad_term = &grad_squared * (1.0 - beta2);
        *velocity = &beta2_term + &grad_term;
    }

    /// Compute bias correction for first moment
    /// 第一モーメントのバイアス補正を計算
    pub fn bias_correction1(beta1: f32, step: usize) -> f32 {
        1.0 - beta1.powi(step as i32)
    }

    /// Compute bias correction for second moment  
    /// 第二モーメントのバイアス補正を計算
    pub fn bias_correction2(beta2: f32, step: usize) -> f32 {
        1.0 - beta2.powi(step as i32)
    }

    /// Apply bias correction to tensor
    /// テンソルにバイアス補正を適用
    pub fn apply_bias_correction(tensor: &Tensor<f32>, correction: f32) -> Tensor<f32> {
        tensor / correction
    }

    /// Compute standard Adam update
    /// 標準Adam更新を計算
    pub fn compute_adam_update(
        momentum_corrected: &Tensor<f32>,
        velocity_corrected: &Tensor<f32>,
        eps: f32,
    ) -> Tensor<f32> {
        let velocity_sqrt = velocity_corrected.sqrt();
        let denominator = &velocity_sqrt + eps;
        momentum_corrected / &denominator
    }

    /// Apply parameter update in-place
    /// パラメータ更新をインプレースで適用
    pub fn apply_update_inplace(param: &Tensor<f32>, update: &Tensor<f32>, learning_rate: f32) {
        let scaled_update = update * learning_rate;
        let new_param = param - &scaled_update;

        // Update parameter in-place
        unsafe {
            std::ptr::copy_nonoverlapping(
                new_param.as_slice().unwrap().as_ptr(),
                param.as_slice().unwrap().as_ptr() as *mut f32,
                param.as_slice().unwrap().len(),
            );
        }
    }
}

/// Trait for specialized Adam-based optimizer behavior
/// 特殊なAdam系最適化器動作のトレイト
pub trait AdamVariant {
    /// Compute specialized update for this Adam variant
    /// このAdam変種の特殊更新を計算
    fn compute_specialized_update(
        &self,
        state: &mut AdamState,
        grad: &Tensor<f32>,
        config: &AdamConfig,
        step: usize,
    ) -> Tensor<f32>;

    /// Get optimizer name for debugging
    /// デバッグ用の最適化器名を取得
    fn optimizer_name(&self) -> &'static str;

    /// Validate optimizer-specific parameters
    /// 最適化器固有パラメータを検証
    fn validate_specific_config(&self, _config: &AdamConfig) -> RusTorchResult<()> {
        Ok(()) // Default: no additional validation
    }

    /// Get additional configuration fields specific to this variant
    /// この変種固有の追加設定フィールドを取得
    fn additional_config_fields(&self) -> HashMap<String, f32> {
        HashMap::new() // Default: no additional fields
    }

    /// Load additional configuration fields specific to this variant
    /// この変種固有の追加設定フィールドを読み込み
    fn load_additional_config(&mut self, _config: &HashMap<String, f32>) {
        // Default: no additional configuration to load
    }
}

/// Generic Adam-based optimizer implementation
/// 汎用Adam系最適化器実装
pub struct GenericAdamOptimizer<V: AdamVariant> {
    config: AdamConfig,
    variant: V,
    step: usize,
    state: HashMap<usize, AdamState>,
}

impl<V: AdamVariant> GenericAdamOptimizer<V> {
    /// Create new generic Adam optimizer
    /// 新しい汎用Adam最適化器を作成
    pub fn from_config_variant(config: AdamConfig, variant: V) -> RusTorchResult<Self> {
        config.validate()?;
        variant.validate_specific_config(&config)?;

        Ok(Self {
            config,
            variant,
            step: 0,
            state: HashMap::new(),
        })
    }

    /// Get or create state for parameter
    /// パラメータの状態を取得または作成
    fn get_or_create_state(&mut self, param_id: usize, shape: &[usize]) -> &mut AdamState {
        self.state
            .entry(param_id)
            .or_insert_with(|| AdamState::new(shape))
    }

    /// Perform optimization step
    /// 最適化ステップを実行
    pub fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        self.step += 1;
        let param_id = param as *const _ as usize;

        // Copy values to avoid borrowing issues
        let weight_decay = self.config.weight_decay;
        let learning_rate = self.config.learning_rate;
        let step = self.step;

        // Apply weight decay to gradient
        let grad_with_decay = AdamUtils::apply_weight_decay(grad, param, weight_decay);

        // Split the borrowing by creating the state first, then calling the variant
        let state_exists = self.state.contains_key(&param_id);
        if !state_exists {
            self.state.insert(param_id, AdamState::new(param.shape()));
        }

        // Now we can safely borrow variant and state separately
        let config = &self.config;
        let update = {
            let state = self.state.get_mut(&param_id).unwrap();
            self.variant
                .compute_specialized_update(state, &grad_with_decay, config, step)
        };

        // Apply update to parameter
        AdamUtils::apply_update_inplace(param, &update, learning_rate);
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set learning rate
    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Get current step count
    /// 現在のステップ数を取得
    pub fn get_step(&self) -> usize {
        self.step
    }

    /// Get optimizer configuration
    /// 最適化器設定を取得
    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    /// Get parameter state for debugging
    /// デバッグ用パラメータ状態を取得
    pub fn get_state(&self, param_id: usize) -> Option<&AdamState> {
        self.state.get(&param_id)
    }

    /// Get optimizer variant reference
    /// 最適化器変種の参照を取得
    pub fn variant(&self) -> &V {
        &self.variant
    }

    /// Get mutable optimizer variant reference
    /// 可変最適化器変種の参照を取得
    pub fn variant_mut(&mut self) -> &mut V {
        &mut self.variant
    }
}

impl<V: AdamVariant> Optimizer for GenericAdamOptimizer<V> {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        self.step(param, grad);
    }

    fn zero_grad(&mut self) {
        // Generic Adam optimizers don't store gradients, so nothing to do
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.set_learning_rate(lr);
    }

    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.config.learning_rate);
        state.insert("beta1".to_string(), self.config.beta1);
        state.insert("beta2".to_string(), self.config.beta2);
        state.insert("eps".to_string(), self.config.eps);
        state.insert("weight_decay".to_string(), self.config.weight_decay);
        state.insert("step".to_string(), self.step as f32);

        // Add variant-specific fields
        let additional_fields = self.variant.additional_config_fields();
        state.extend(additional_fields);

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.config.learning_rate = lr;
        }
        if let Some(&beta1) = state.get("beta1") {
            self.config.beta1 = beta1;
        }
        if let Some(&beta2) = state.get("beta2") {
            self.config.beta2 = beta2;
        }
        if let Some(&eps) = state.get("eps") {
            self.config.eps = eps;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.config.weight_decay = weight_decay;
        }
        if let Some(&step) = state.get("step") {
            self.step = step as usize;
        }

        // Load variant-specific fields (needs mutable access to variant)
        let additional_config = state.clone();
        self.variant.load_additional_config(&additional_config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_state_creation() {
        let state = AdamState::new(&[2, 3]);
        assert_eq!(state.momentum.shape(), &[2, 3]);
        assert_eq!(state.velocity.shape(), &[2, 3]);
        assert!(state.extra_state.is_empty());
    }

    #[test]
    fn test_adam_state_extra_state() {
        let mut state = AdamState::new(&[2, 3]);
        let extra_tensor = Tensor::ones(&[2, 3]);
        state.add_extra_state("test".to_string(), extra_tensor);

        assert!(state.get_extra_state("test").is_some());
        assert!(state.get_extra_state("nonexistent").is_none());
    }

    #[test]
    fn test_adam_config_validation() {
        let valid_config = AdamConfig::adam(0.001);
        assert!(valid_config.validate().is_ok());

        let invalid_config = AdamConfig {
            learning_rate: -0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_adam_config_presets() {
        let nadam_config = AdamConfig::nadam(0.002);
        assert_eq!(nadam_config.learning_rate, 0.002);
        assert_eq!(nadam_config.beta1, 0.9);

        let adamax_config = AdamConfig::adamax(0.002);
        assert_eq!(adamax_config.eps, 1e-7); // Different default
    }

    #[test]
    fn test_adam_utils_weight_decay() {
        let grad = Tensor::ones(&[2]);
        let param = Tensor::from_vec(vec![2.0, 3.0], vec![2]);

        let result = AdamUtils::apply_weight_decay(&grad, &param, 0.1);
        let expected_data = &[1.2, 1.3]; // 1.0 + 0.1 * [2.0, 3.0]
        let result_data = result.as_slice().unwrap();

        for (r, e) in result_data.iter().zip(expected_data.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bias_correction() {
        let correction1 = AdamUtils::bias_correction1(0.9, 1);
        let expected1 = 1.0 - 0.9;
        assert!((correction1 - expected1).abs() < 1e-6);

        let correction2 = AdamUtils::bias_correction2(0.999, 1);
        let expected2 = 1.0 - 0.999;
        assert!((correction2 - expected2).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_update() {
        let mut momentum = Tensor::zeros(&[2]);
        let grad = Tensor::ones(&[2]);

        AdamUtils::update_momentum(&mut momentum, &grad, 0.9);

        let expected = vec![0.1, 0.1]; // (1 - 0.9) * 1.0
        let result_data = momentum.as_slice().unwrap();

        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
}
