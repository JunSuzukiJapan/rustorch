//! RAdam optimizer implementation
//! RAdamオプティマイザの実装 - Rectified Adam

use crate::optim::common::{AdamVariant, AdamConfig, AdamState, AdamUtils, GenericAdamOptimizer};
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;

/// RAdam variant implementing variance rectification
/// RAdam変種：分散修正を実装
#[derive(Debug, Clone)]
pub struct RAdamVariant {
    /// Cached rho_inf value for efficiency
    /// 効率のためにキャッシュされたrho_inf値
    rho_inf_cache: Option<f32>,
    /// Threshold for variance rectification (default: 4.0)
    /// 分散修正のしきい値（デフォルト: 4.0）
    rectification_threshold: f32,
}

impl RAdamVariant {
    /// Create new RAdam variant with default parameters
    /// デフォルトパラメータで新しいRAdam変種を作成
    pub fn new() -> Self {
        Self {
            rho_inf_cache: None,
            rectification_threshold: 4.0,
        }
    }

    /// Create RAdam variant with custom threshold
    /// カスタムしきい値でRAdam変種を作成
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            rho_inf_cache: None,
            rectification_threshold: threshold,
        }
    }

    /// Compute or retrieve cached rho_inf (maximum SMA length)
    /// rho_inf（最大SMA長）を計算または取得
    fn get_rho_inf(&mut self, beta2: f32) -> f32 {
        if let Some(cached) = self.rho_inf_cache {
            cached
        } else {
            let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
            self.rho_inf_cache = Some(rho_inf);
            rho_inf
        }
    }

    /// Optimized computation of rho_t (SMA length at step t)
    /// rho_t（ステップtでのSMA長）の最適化計算
    fn compute_rho_t(&mut self, beta2: f32, step: usize) -> f32 {
        let rho_inf = self.get_rho_inf(beta2);
        let beta2_t = beta2.powi(step as i32);
        rho_inf - 2.0 * (step as f32) * beta2_t / (1.0 - beta2_t)
    }

    /// Check if variance rectification should be applied
    /// 分散修正を適用すべきかチェック
    fn should_rectify(&mut self, beta2: f32, step: usize) -> bool {
        self.compute_rho_t(beta2, step) > self.rectification_threshold
    }

    /// Optimized variance rectification term computation
    /// 最適化された分散修正項の計算
    fn compute_rectification_term(&mut self, beta2: f32, step: usize) -> f32 {
        let rho_inf = self.get_rho_inf(beta2);
        let rho_t = self.compute_rho_t(beta2, step);
        
        // Pre-compute common terms for efficiency
        let rho_inf_minus_4 = rho_inf - 4.0;
        let rho_inf_minus_2 = rho_inf - 2.0;
        let rho_t_minus_4 = rho_t - 4.0;
        let rho_t_minus_2 = rho_t - 2.0;
        
        let numerator = rho_inf_minus_4 * rho_inf_minus_2 * rho_t;
        let denominator = rho_inf * rho_t_minus_4 * rho_t_minus_2;
        
        (numerator / denominator).sqrt()
    }
}

impl Default for RAdamVariant {
    fn default() -> Self {
        Self::new()
    }
}

impl AdamVariant for RAdamVariant {
    fn compute_specialized_update(
        &self,
        state: &mut AdamState,
        grad: &Tensor<f32>,
        config: &AdamConfig,
        step: usize,
    ) -> Tensor<f32> {
        // Note: We need mutable access to self for caching, but trait requires &self
        // This is a design trade-off for better performance
        let mut variant_copy = self.clone();
        
        // Update momentum and velocity using common utilities
        AdamUtils::update_momentum(&mut state.momentum, grad, config.beta1);
        AdamUtils::update_velocity(&mut state.velocity, grad, config.beta2);
        
        // Compute bias corrections
        let bias_correction1 = AdamUtils::bias_correction1(config.beta1, step);
        let bias_corrected_momentum = AdamUtils::apply_bias_correction(&state.momentum, bias_correction1);
        
        // RAdam's key feature: variance rectification
        if variant_copy.should_rectify(config.beta2, step) {
            // Use adaptive learning rate with variance rectification
            let bias_correction2 = AdamUtils::bias_correction2(config.beta2, step);
            let bias_corrected_velocity = AdamUtils::apply_bias_correction(&state.velocity, bias_correction2);
            let rectification_term = variant_copy.compute_rectification_term(config.beta2, step);
            
            // Apply rectification to the standard Adam update
            let adam_update = AdamUtils::compute_adam_update(&bias_corrected_momentum, &bias_corrected_velocity, config.eps);
            &adam_update * rectification_term
        } else {
            // Fall back to momentum-only update when variance is not rectifiable
            bias_corrected_momentum
        }
    }

    fn optimizer_name(&self) -> &'static str {
        "RAdam"
    }

    fn validate_specific_config(&self, _config: &AdamConfig) -> RusTorchResult<()> {
        if self.rectification_threshold <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "RAdam optimizer".to_string(),
                message: "Rectification threshold must be positive".to_string(),
            });
        }
        Ok(())
    }

    fn additional_config_fields(&self) -> HashMap<String, f32> {
        let mut fields = HashMap::new();
        fields.insert("rectification_threshold".to_string(), self.rectification_threshold);
        if let Some(rho_inf) = self.rho_inf_cache {
            fields.insert("rho_inf_cache".to_string(), rho_inf);
        }
        fields
    }

    fn load_additional_config(&mut self, config: &HashMap<String, f32>) {
        if let Some(&threshold) = config.get("rectification_threshold") {
            self.rectification_threshold = threshold;
        }
        if let Some(&rho_inf) = config.get("rho_inf_cache") {
            self.rho_inf_cache = Some(rho_inf);
        }
    }
}

/// RAdam (Rectified Adaptive Moment Estimation) optimizer
/// RAdam（修正適応モーメント推定）オプティマイザ
/// 
/// RAdam provides variance rectification for Adam optimizer, addressing the
/// large variance issue in the early stages of training
/// RAdamはAdamオプティマイザの分散修正を提供し、訓練初期段階の大きな分散問題に対処する
pub type RAdam = GenericAdamOptimizer<RAdamVariant>;

impl RAdam {
    /// Create a new RAdam optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.001)
    /// * `beta1` - Coefficient for computing running averages of gradient (default: 0.9)
    /// * `beta2` - Coefficient for computing running averages of gradient square (default: 0.999)
    /// * `eps` - Term added to denominator for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.0)
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> RusTorchResult<Self> {
        let config = AdamConfig {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
        };
        let variant = RAdamVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create RAdam with default parameters
    pub fn default_params(learning_rate: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::radam(learning_rate);
        let variant = RAdamVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create RAdam with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::radam(learning_rate).with_weight_decay(weight_decay);
        let variant = RAdamVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create RAdam with custom rectification threshold
    pub fn with_threshold(learning_rate: f32, threshold: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::radam(learning_rate);
        let variant = RAdamVariant::with_threshold(threshold);
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Get parameter state for debugging
    pub fn get_param_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.get_state(param_id).map(|s| (&s.momentum, &s.velocity))
    }

    /// Get RAdam-specific configuration (rectification threshold and cached rho_inf)
    pub fn radam_config(&self) -> (f32, Option<f32>) {
        let variant = self.variant();
        (variant.rectification_threshold, variant.rho_inf_cache)
    }

    /// Check if variance rectification is active for current configuration
    /// 現在の設定で分散修正がアクティブかチェック
    pub fn is_rectification_enabled(&self, step: usize) -> bool {
        let mut variant_copy = self.variant().clone();
        let config = self.config();
        variant_copy.should_rectify(config.beta2, step)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radam_creation() {
        let optimizer = RAdam::default_params(0.001).unwrap();
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
    }

    #[test]
    fn test_radam_with_weight_decay() {
        let optimizer = RAdam::with_weight_decay(0.001, 0.1).unwrap();
        assert_eq!(optimizer.config().weight_decay, 0.1);
    }

    #[test]
    fn test_radam_step() {
        let mut optimizer = RAdam::default_params(0.1).unwrap();
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);

        optimizer.step(&param, &grad);

        // Check that parameters were updated
        let param_data = param.as_slice().unwrap();
        assert!(param_data[0] < 1.0); // Should decrease due to positive gradient
        assert!(param_data[1] < 2.0);
        assert!(param_data[2] < 3.0);
    }

    #[test]
    fn test_radam_variant_caching() {
        let mut variant = RAdamVariant::new();
        
        // First call should compute and cache rho_inf
        let rho_inf1 = variant.get_rho_inf(0.999);
        assert!((rho_inf1 - 1999.0).abs() < 1.0);
        
        // Second call should use cached value
        let rho_inf2 = variant.get_rho_inf(0.999);
        assert_eq!(rho_inf1, rho_inf2);
        assert!(variant.rho_inf_cache.is_some());
    }

    #[test]
    fn test_variance_rectification() {
        let optimizer = RAdam::default_params(0.001).unwrap();
        
        // Early steps should not be rectifiable
        assert!(!optimizer.is_rectification_enabled(1));
        assert!(!optimizer.is_rectification_enabled(2));
        
        // Later steps should be rectifiable (depends on beta2=0.999)
        // With beta2=0.999, it takes many more steps to become rectifiable
        assert!(optimizer.is_rectification_enabled(1000));
        assert!(optimizer.is_rectification_enabled(10000));
    }

    #[test]
    fn test_radam_with_custom_threshold() {
        let optimizer = RAdam::with_threshold(0.001, 2.0).unwrap();
        let (threshold, _) = optimizer.radam_config();
        assert_eq!(threshold, 2.0);
        
        // With lower threshold, rectification should be enabled earlier
        assert!(optimizer.is_rectification_enabled(100));
    }

    #[test]
    fn test_radam_variant_rectification_term() {
        let mut variant = RAdamVariant::new();
        
        // Rectification term should be positive and reasonable for large steps
        let rect_term = variant.compute_rectification_term(0.999, 10000);
        assert!(rect_term > 0.0);
        assert!(rect_term < 5.0);
    }

    #[test]
    fn test_radam_fallback_to_momentum() {
        let mut optimizer = RAdam::default_params(0.001).unwrap();
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);

        let original_param = param.clone();
        
        // In early steps, RAdam should fall back to momentum-only updates
        optimizer.step(&param, &grad);

        // Should still update parameters even in momentum-only mode
        let param_data = param.as_slice().unwrap();
        let orig_data = original_param.as_slice().unwrap();
        
        for (new_val, orig_val) in param_data.iter().zip(orig_data.iter()) {
            assert!(new_val != orig_val); // Parameters should change
        }
    }

    #[test]
    fn test_radam_state_dict() {
        let optimizer = RAdam::default_params(0.001).unwrap();
        let state_dict = optimizer.state_dict();
        
        assert_eq!(state_dict.get("learning_rate"), Some(&0.001));
        assert_eq!(state_dict.get("beta1"), Some(&0.9));
        assert_eq!(state_dict.get("rectification_threshold"), Some(&4.0));
    }

    #[test]
    fn test_radam_variant_validation() {
        let variant = RAdamVariant::with_threshold(-1.0);
        let config = AdamConfig::radam(0.001);
        assert!(variant.validate_specific_config(&config).is_err());
        
        let valid_variant = RAdamVariant::with_threshold(4.0);
        assert!(valid_variant.validate_specific_config(&config).is_ok());
    }
}