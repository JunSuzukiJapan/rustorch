//! NAdam optimizer implementation
//! NAdamオプティマイザの実装 - Nesterov加速Adam

use crate::error::{RusTorchError, RusTorchResult};
use crate::optim::common::{AdamConfig, AdamState, AdamUtils, AdamVariant, GenericAdamOptimizer};
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// NAdam variant implementing specialized Nesterov acceleration
/// NAdam変種：特殊なNesterov加速を実装
#[derive(Debug, Clone)]
pub struct NAdamVariant {
    momentum_decay: f32,
    schedule_decay: f32,
}

impl NAdamVariant {
    /// Create new NAdam variant with default parameters
    /// デフォルトパラメータで新しいNAdam変種を作成
    pub fn new(momentum_decay: f32, schedule_decay: f32) -> Self {
        Self {
            momentum_decay,
            schedule_decay,
        }
    }

    /// Create NAdam variant with default decay parameters
    /// デフォルト減衰パラメータでNAdam変種を作成
    pub fn default_decay() -> Self {
        Self::new(0.004, 0.004)
    }

    /// Compute time-dependent beta1 with decay
    /// 減衰を伴う時間依存beta1を計算
    fn beta1_t(&self, beta1: f32, t: usize) -> f32 {
        let momentum_cache_t =
            beta1 * (1.0 - 0.5 * 0.96_f32.powi(t as i32 * self.schedule_decay as i32));
        let momentum_cache_t_1 =
            beta1 * (1.0 - 0.5 * 0.96_f32.powi((t + 1) as i32 * self.schedule_decay as i32));

        momentum_cache_t_1 / (1.0 - momentum_cache_t)
    }
}

impl AdamVariant for NAdamVariant {
    fn compute_specialized_update(
        &self,
        state: &mut AdamState,
        grad: &Tensor<f32>,
        config: &AdamConfig,
        step: usize,
    ) -> Tensor<f32> {
        // Update momentum
        AdamUtils::update_momentum(&mut state.momentum, grad, config.beta1);

        // Update velocity
        AdamUtils::update_velocity(&mut state.velocity, grad, config.beta2);

        // Compute bias corrections
        let bias_correction1 = AdamUtils::bias_correction1(config.beta1, step);
        let bias_correction2 = AdamUtils::bias_correction2(config.beta2, step);

        // Bias-corrected estimates
        let momentum_corrected =
            AdamUtils::apply_bias_correction(&state.momentum, bias_correction1);
        let velocity_corrected =
            AdamUtils::apply_bias_correction(&state.velocity, bias_correction2);

        // NAdam's key feature: Nesterov acceleration
        let beta1_t = self.beta1_t(config.beta1, step);
        let momentum_term = &momentum_corrected * beta1_t;
        let gradient_term = grad * ((1.0 - config.beta1) / bias_correction1);
        let nesterov_momentum = &momentum_term + &gradient_term;

        // Compute update
        AdamUtils::compute_adam_update(&nesterov_momentum, &velocity_corrected, config.eps)
    }

    fn optimizer_name(&self) -> &'static str {
        "NAdam"
    }

    fn validate_specific_config(&self, _config: &AdamConfig) -> RusTorchResult<()> {
        if self.momentum_decay < 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "NAdam optimizer".to_string(),
                message: "Momentum decay must be non-negative".to_string(),
            });
        }
        if self.schedule_decay < 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "NAdam optimizer".to_string(),
                message: "Schedule decay must be non-negative".to_string(),
            });
        }
        Ok(())
    }

    fn additional_config_fields(&self) -> HashMap<String, f32> {
        let mut fields = HashMap::new();
        fields.insert("momentum_decay".to_string(), self.momentum_decay);
        fields.insert("schedule_decay".to_string(), self.schedule_decay);
        fields
    }

    fn load_additional_config(&mut self, config: &HashMap<String, f32>) {
        if let Some(&momentum_decay) = config.get("momentum_decay") {
            self.momentum_decay = momentum_decay;
        }
        if let Some(&schedule_decay) = config.get("schedule_decay") {
            self.schedule_decay = schedule_decay;
        }
    }
}

/// NAdam (Nesterov-accelerated Adaptive Moment Estimation) optimizer
/// NAdam（Nesterov加速適応モーメント推定）オプティマイザ
///
/// NAdam combines Adam with Nesterov momentum for better convergence
/// NAdamはAdamとNesterovモーメンタムを組み合わせ、より良い収束を実現
pub type NAdam = GenericAdamOptimizer<NAdamVariant>;

impl NAdam {
    /// Create a new NAdam optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.002)
    /// * `beta1` - Coefficient for computing running averages of gradient (default: 0.9)
    /// * `beta2` - Coefficient for computing running averages of gradient square (default: 0.999)
    /// * `eps` - Term added to denominator for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.0)
    /// * `momentum_decay` - Momentum decay parameter (default: 0.004)
    /// * `schedule_decay` - Schedule decay parameter (default: 0.004)
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        momentum_decay: f32,
        schedule_decay: f32,
    ) -> RusTorchResult<Self> {
        let config = AdamConfig {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
        };
        let variant = NAdamVariant::new(momentum_decay, schedule_decay);
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create NAdam with default parameters
    pub fn default_params(learning_rate: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::nadam(learning_rate);
        let variant = NAdamVariant::default_decay();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create NAdam with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::nadam(learning_rate).with_weight_decay(weight_decay);
        let variant = NAdamVariant::default_decay();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create NAdam with custom momentum decay
    pub fn with_momentum_decay(learning_rate: f32, momentum_decay: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::nadam(learning_rate);
        let variant = NAdamVariant::new(momentum_decay, 0.004);
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Get parameter state for debugging  
    pub fn get_param_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.get_state(param_id).map(|s| (&s.momentum, &s.velocity))
    }

    /// Get NAdam variant configuration
    pub fn nadam_config(&self) -> (f32, f32) {
        let variant = self.variant();
        (variant.momentum_decay, variant.schedule_decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nadam_creation() {
        let optimizer = NAdam::default_params(0.002).unwrap();
        assert_eq!(optimizer.learning_rate(), 0.002);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
    }

    #[test]
    fn test_nadam_with_weight_decay() {
        let optimizer = NAdam::with_weight_decay(0.001, 0.1).unwrap();
        assert_eq!(optimizer.config().weight_decay, 0.1);
    }

    #[test]
    fn test_nadam_step() {
        let mut optimizer = NAdam::default_params(0.1).unwrap();
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
    fn test_nadam_momentum_decay() {
        let mut optimizer = NAdam::with_momentum_decay(0.001, 0.01).unwrap();
        let (momentum_decay, _) = optimizer.nadam_config();
        assert_eq!(momentum_decay, 0.01);

        let param = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let grad = Tensor::from_vec(vec![0.1, 0.2], vec![2]);

        optimizer.step(&param, &grad);

        // State should be created for the parameter
        let param_id = &param as *const _ as usize;
        assert!(optimizer.get_state(param_id).is_some());
    }

    #[test]
    fn test_nadam_variant_beta1_t() {
        let variant = NAdamVariant::default_decay();

        // Beta1 should change over time with NAdam scheduling
        let beta1_t1 = variant.beta1_t(0.9, 1);
        let beta1_t10 = variant.beta1_t(0.9, 10);
        let beta1_t100 = variant.beta1_t(0.9, 100);

        // Beta1_t values should be reasonable
        assert!(beta1_t1 > 0.0 && beta1_t1 <= 1.5);
        assert!(beta1_t10 > 0.0 && beta1_t10 <= 1.5);
        assert!(beta1_t100 > 0.0 && beta1_t100 <= 1.5);

        // Just verify the computation is stable
        assert!(beta1_t1.is_finite());
        assert!(beta1_t10.is_finite());
        assert!(beta1_t100.is_finite());
    }

    #[test]
    fn test_nadam_state_dict() {
        let optimizer = NAdam::default_params(0.001).unwrap();
        let state_dict = optimizer.state_dict();

        assert_eq!(state_dict.get("learning_rate"), Some(&0.001));
        assert_eq!(state_dict.get("beta1"), Some(&0.9));
        assert_eq!(state_dict.get("momentum_decay"), Some(&0.004));
        assert_eq!(state_dict.get("schedule_decay"), Some(&0.004));
    }

    #[test]
    fn test_nadam_variant_validation() {
        let variant = NAdamVariant::new(-0.1, 0.004);
        let config = AdamConfig::nadam(0.001);
        assert!(variant.validate_specific_config(&config).is_err());

        let valid_variant = NAdamVariant::new(0.004, 0.004);
        assert!(valid_variant.validate_specific_config(&config).is_ok());
    }
}
