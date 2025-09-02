//! Adamax optimizer implementation
//! Adamaxオプティマイザの実装 - Adam with infinity norm

use crate::optim::common::{AdamVariant, AdamConfig, AdamState, AdamUtils, GenericAdamOptimizer};
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;

/// Adamax variant implementing optimized infinity norm computation
/// Adamax変種：最適化された無限大ノルム計算を実装
#[derive(Debug, Clone)]
pub struct AdamaxVariant {
    /// Custom epsilon specifically for infinity norm stability
    /// 無限大ノルムの安定性のための専用イプシロン
    infinity_eps: f32,
}

impl AdamaxVariant {
    /// Create new Adamax variant with default parameters
    /// デフォルトパラメータで新しいAdamax変種を作成
    pub fn new() -> Self {
        Self {
            infinity_eps: 1e-8,
        }
    }

    /// Create Adamax variant with custom infinity epsilon
    /// カスタム無限大イプシロンでAdamax変種を作成
    pub fn with_infinity_eps(infinity_eps: f32) -> Self {
        Self { infinity_eps }
    }

    /// Optimized element-wise maximum computation for infinity norm
    /// 無限大ノルムのための最適化された要素ごと最大値計算
    fn tensor_max_optimized(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        let a_data = a.as_slice().unwrap();
        let b_data = b.as_slice().unwrap();
        
        // Use iterator chaining for better performance
        let max_data: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&a_val, &b_val)| a_val.max(b_val))
            .collect();
            
        Tensor::from_vec(max_data, a.shape().to_vec())
    }

    /// Fast absolute value computation with SIMD-friendly operations
    /// SIMD対応操作による高速絶対値計算
    fn tensor_abs_fast(tensor: &Tensor<f32>) -> Tensor<f32> {
        let data = tensor.as_slice().unwrap();
        let abs_data: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        Tensor::from_vec(abs_data, tensor.shape().to_vec())
    }
}

impl Default for AdamaxVariant {
    fn default() -> Self {
        Self::new()
    }
}

impl AdamVariant for AdamaxVariant {
    fn compute_specialized_update(
        &self,
        state: &mut AdamState,
        grad: &Tensor<f32>,
        config: &AdamConfig,
        step: usize,
    ) -> Tensor<f32> {
        // Update momentum using common utility
        AdamUtils::update_momentum(&mut state.momentum, grad, config.beta1);
        
        // Adamax key feature: Update infinity norm instead of velocity
        let grad_abs = Self::tensor_abs_fast(grad);
        let beta2_scaled = &state.velocity * config.beta2;
        state.velocity = Self::tensor_max_optimized(&beta2_scaled, &grad_abs);
        
        // Apply bias correction only to first moment (momentum)
        let bias_correction1 = AdamUtils::bias_correction1(config.beta1, step);
        let momentum_corrected = AdamUtils::apply_bias_correction(&state.momentum, bias_correction1);
        
        // Compute Adamax update: No bias correction for infinity norm
        // The infinity norm is inherently "corrected" by its max operation
        let denominator = &state.velocity + self.infinity_eps.max(config.eps);
        &momentum_corrected / &denominator
    }

    fn optimizer_name(&self) -> &'static str {
        "Adamax"
    }

    fn validate_specific_config(&self, _config: &AdamConfig) -> RusTorchResult<()> {
        if self.infinity_eps <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "Adamax optimizer".to_string(),
                message: "Infinity epsilon must be positive".to_string(),
            });
        }
        Ok(())
    }

    fn additional_config_fields(&self) -> HashMap<String, f32> {
        let mut fields = HashMap::new();
        fields.insert("infinity_eps".to_string(), self.infinity_eps);
        fields
    }

    fn load_additional_config(&mut self, config: &HashMap<String, f32>) {
        if let Some(&infinity_eps) = config.get("infinity_eps") {
            self.infinity_eps = infinity_eps;
        }
    }
}

/// Adamax (Adam with infinity norm) optimizer
/// Adamax（無限大ノルムを使用するAdam）オプティマイザ
/// 
/// Adamax is a variant of Adam where the second moment is replaced by the infinity norm
/// Adamaxは第2モーメントを無限大ノルムに置き換えたAdamの変種
pub type Adamax = GenericAdamOptimizer<AdamaxVariant>;

impl Adamax {
    /// Create a new Adamax optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.002)
    /// * `beta1` - Coefficient for computing running averages of gradient (default: 0.9)
    /// * `beta2` - Coefficient for computing running averages of infinity norm (default: 0.999)
    /// * `eps` - Term added to denominator for numerical stability (default: 1e-7)
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
        let variant = AdamaxVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create Adamax with default parameters
    pub fn default_params(learning_rate: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::adamax(learning_rate);
        let variant = AdamaxVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create Adamax with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::adamax(learning_rate).with_weight_decay(weight_decay);
        let variant = AdamaxVariant::new();
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Create Adamax with custom infinity epsilon
    pub fn with_infinity_eps(learning_rate: f32, infinity_eps: f32) -> RusTorchResult<Self> {
        let config = AdamConfig::adamax(learning_rate);
        let variant = AdamaxVariant::with_infinity_eps(infinity_eps);
        GenericAdamOptimizer::from_config_variant(config, variant)
    }

    /// Get parameter state for debugging
    pub fn get_param_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.get_state(param_id).map(|s| (&s.momentum, &s.velocity))
    }

    /// Get Adamax-specific configuration (infinity epsilon)
    pub fn adamax_config(&self) -> f32 {
        self.variant().infinity_eps
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamax_creation() {
        let optimizer = Adamax::default_params(0.002).unwrap();
        assert_eq!(optimizer.learning_rate(), 0.002);
        assert_eq!(optimizer.config().beta1, 0.9);
        assert_eq!(optimizer.config().beta2, 0.999);
        assert_eq!(optimizer.config().eps, 1e-7);
    }

    #[test]
    fn test_adamax_with_weight_decay() {
        let optimizer = Adamax::with_weight_decay(0.001, 0.1).unwrap();
        assert_eq!(optimizer.config().weight_decay, 0.1);
    }

    #[test]
    fn test_adamax_step() {
        let mut optimizer = Adamax::default_params(0.1).unwrap();
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
    fn test_tensor_max() {
        let a = Tensor::from_vec(vec![1.0, 5.0, 2.0], vec![3]);
        let b = Tensor::from_vec(vec![3.0, 1.0, 4.0], vec![3]);
        
        let max_tensor = AdamaxVariant::tensor_max_optimized(&a, &b);
        let max_data = max_tensor.as_slice().unwrap();
        
        assert_eq!(max_data, &[3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_tensor_abs() {
        let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        let abs_tensor = AdamaxVariant::tensor_abs_fast(&tensor);
        let abs_data = abs_tensor.as_slice().unwrap();
        
        assert_eq!(abs_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_adamax_infinity_norm_update() {
        let mut optimizer = Adamax::default_params(0.001).unwrap();
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let grad1 = Tensor::from_vec(vec![0.1, 0.5, 0.2], vec![3]);
        let grad2 = Tensor::from_vec(vec![0.3, 0.1, 0.8], vec![3]);

        let param_id = &param as *const _ as usize;
        
        // First step
        optimizer.step(&param, &grad1);
        let inf_norm1 = {
            let state = optimizer.get_state(param_id).unwrap();
            state.velocity.clone()
        };
        let inf_norm1_data = inf_norm1.as_slice().unwrap();
        
        // Second step with different gradient
        optimizer.step(&param, &grad2);
        let inf_norm2 = {
            let state = optimizer.get_state(param_id).unwrap();
            state.velocity.clone()
        };
        let inf_norm2_data = inf_norm2.as_slice().unwrap();
        
        // Infinity norm should track the maximum absolute values
        // After step 1: approximately [0.1, 0.5, 0.2]
        // After step 2: should be max of (beta2 * prev, |new_grad|)
        assert!(inf_norm2_data[2] > inf_norm1_data[2]); // 0.8 > previous values
    }

    #[test]
    fn test_adamax_no_bias_correction_for_infinity_norm() {
        let mut optimizer = Adamax::default_params(0.1).unwrap();
        let param = Tensor::from_vec(vec![1.0], vec![1]);
        let grad = Tensor::from_vec(vec![1.0], vec![1]);

        let original_param_val = param.as_slice().unwrap()[0];
        
        // Take a step
        optimizer.step(&param, &grad);
        
        let new_param_val = param.as_slice().unwrap()[0];
        let param_id = &param as *const _ as usize;
        let state = optimizer.get_state(param_id).unwrap();
        
        // The infinity norm should be used directly (no bias correction)
        // Unlike Adam, Adamax doesn't need bias correction for the second moment
        assert!(state.velocity.as_slice().unwrap()[0] > 0.0);
        assert!(new_param_val != original_param_val);
    }

    #[test]
    fn test_adamax_with_infinity_eps() {
        let optimizer = Adamax::with_infinity_eps(0.001, 1e-6).unwrap();
        assert_eq!(optimizer.adamax_config(), 1e-6);
    }

    #[test]
    fn test_adamax_variant_validation() {
        let variant = AdamaxVariant::with_infinity_eps(-1e-8);
        let config = AdamConfig::adamax(0.001);
        assert!(variant.validate_specific_config(&config).is_err());
        
        let valid_variant = AdamaxVariant::with_infinity_eps(1e-8);
        assert!(valid_variant.validate_specific_config(&config).is_ok());
    }

    #[test]
    fn test_adamax_state_dict() {
        let optimizer = Adamax::default_params(0.002).unwrap();
        let state_dict = optimizer.state_dict();
        
        assert_eq!(state_dict.get("learning_rate"), Some(&0.002));
        assert_eq!(state_dict.get("beta1"), Some(&0.9));
        assert_eq!(state_dict.get("beta2"), Some(&0.999));
        assert_eq!(state_dict.get("infinity_eps"), Some(&1e-8));
    }
}