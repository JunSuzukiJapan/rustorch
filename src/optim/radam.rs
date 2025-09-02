//! RAdam optimizer implementation
//! RAdamオプティマイザの実装 - Rectified Adam

use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// RAdam (Rectified Adaptive Moment Estimation) optimizer
/// RAdam（修正適応モーメント推定）オプティマイザ
/// 
/// RAdam provides variance rectification for Adam optimizer, addressing the
/// large variance issue in the early stages of training
/// RAdamはAdamオプティマイザの分散修正を提供し、訓練初期段階の大きな分散問題に対処する
pub struct RAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    state: HashMap<usize, RAdamState>,
}

/// State for each parameter in RAdam
struct RAdamState {
    momentum: Tensor<f32>,             // First moment estimate
    velocity: Tensor<f32>,             // Second moment estimate
}

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
    ) -> Self {
        assert!(learning_rate > 0.0, "Learning rate must be positive");
        assert!(beta1 >= 0.0 && beta1 < 1.0, "Beta1 must be in [0, 1)");
        assert!(beta2 >= 0.0 && beta2 < 1.0, "Beta2 must be in [0, 1)");
        assert!(eps > 0.0, "Epsilon must be positive");
        assert!(weight_decay >= 0.0, "Weight decay must be non-negative");

        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
            state: HashMap::new(),
        }
    }

    /// Create RAdam with default parameters
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Create RAdam with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, weight_decay)
    }

    /// Get parameter state for debugging
    pub fn get_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.state
            .get(&param_id)
            .map(|s| (&s.momentum, &s.velocity))
    }

    fn get_or_create_state(&mut self, param_id: usize, shape: &[usize]) -> &mut RAdamState {
        self.state.entry(param_id).or_insert_with(|| RAdamState {
            momentum: Tensor::zeros(shape),
            velocity: Tensor::zeros(shape),
        })
    }

    /// Compute the maximum length of the approximated SMA (Simple Moving Average)
    fn rho_inf(&self) -> f32 {
        2.0 / (1.0 - self.beta2) - 1.0
    }

    /// Compute the length of approximated SMA at step t
    fn rho_t(&self, t: usize) -> f32 {
        let rho_inf = self.rho_inf();
        let beta2_t = self.beta2.powi(t as i32);
        rho_inf - 2.0 * (t as f32) * beta2_t / (1.0 - beta2_t)
    }

    /// Check if variance is rectifiable (rho_t > threshold)
    fn is_variance_rectifiable(&self, t: usize) -> bool {
        self.rho_t(t) > 4.0
    }

    /// Compute variance rectification term
    fn variance_rectification_term(&self, t: usize) -> f32 {
        let rho_inf = self.rho_inf();
        let rho_t = self.rho_t(t);
        
        let numerator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
        let denominator = rho_inf * (rho_t - 4.0) * (rho_t - 2.0);
        
        (numerator / denominator).sqrt()
    }
}

impl Optimizer for RAdam {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        self.step += 1;
        let param_id = param as *const _ as usize;

        // Copy parameters to avoid borrow issues
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let learning_rate = self.learning_rate;
        let step = self.step;

        // Apply weight decay to gradient if specified
        let grad_with_decay = if weight_decay > 0.0 {
            let weight_decay_term = param * weight_decay;
            grad + &weight_decay_term
        } else {
            grad.clone()
        };

        // Get or create state for this parameter
        let state = self.get_or_create_state(param_id, param.shape());

        // Update biased first moment estimate
        state.momentum = (&(&state.momentum * beta1)) + (&(&grad_with_decay * (1.0 - beta1)));

        // Update biased second raw moment estimate
        let grad_squared = &grad_with_decay * &grad_with_decay;
        state.velocity = (&(&state.velocity * beta2)) + (&(&grad_squared * (1.0 - beta2)));

        // Compute bias correction for first moment
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let momentum_corrected = &state.momentum / bias_correction1;

        // RAdam's key feature: variance rectification
        let is_rectifiable = {
            let rho_t = {
                let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
                let beta2_t = beta2.powi(step as i32);
                rho_inf - 2.0 * (step as f32) * beta2_t / (1.0 - beta2_t)
            };
            rho_t > 4.0
        };
        
        let rect_term = if is_rectifiable {
            let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
            let rho_t = {
                let beta2_t = beta2.powi(step as i32);
                rho_inf - 2.0 * (step as f32) * beta2_t / (1.0 - beta2_t)
            };
            let numerator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
            let denominator = rho_inf * (rho_t - 4.0) * (rho_t - 2.0);
            (numerator / denominator).sqrt()
        } else {
            1.0
        };
        
        let update = if is_rectifiable {
            // Use adaptive learning rate with variance rectification
            let bias_correction2 = 1.0 - beta2.powi(step as i32);
            let velocity_corrected = &state.velocity / bias_correction2;
            let velocity_sqrt = velocity_corrected.sqrt();
            
            let adaptive_step_size = learning_rate * rect_term;
            
            &momentum_corrected / &(&velocity_sqrt + eps) * adaptive_step_size
        } else {
            // Fall back to momentum-only update (like SGD with momentum)
            // when variance is not rectifiable
            &momentum_corrected * learning_rate
        };

        // Apply update
        let new_param = param - &update;

        // Update parameter in-place
        unsafe {
            std::ptr::copy_nonoverlapping(
                new_param.as_slice().unwrap().as_ptr(),
                param.as_slice().unwrap().as_ptr() as *mut f32,
                param.as_slice().unwrap().len(),
            );
        }
    }

    fn zero_grad(&mut self) {
        // RAdam doesn't store gradients, so nothing to do here
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        assert!(lr > 0.0, "Learning rate must be positive");
        self.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("eps".to_string(), self.eps);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("step".to_string(), self.step as f32);
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&beta1) = state.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&eps) = state.get("eps") {
            self.eps = eps;
        }
        if let Some(&wd) = state.get("weight_decay") {
            self.weight_decay = wd;
        }
        if let Some(&step) = state.get("step") {
            self.step = step as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radam_creation() {
        let optimizer = RAdam::default_params(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
    }

    #[test]
    fn test_radam_with_weight_decay() {
        let optimizer = RAdam::with_weight_decay(0.001, 0.1);
        assert_eq!(optimizer.weight_decay, 0.1);
    }

    #[test]
    fn test_radam_step() {
        let mut optimizer = RAdam::default_params(0.1);
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
    fn test_rho_inf_calculation() {
        let optimizer = RAdam::default_params(0.001);
        let rho_inf = optimizer.rho_inf();
        
        // With default beta2 = 0.999, rho_inf should be approximately 1999
        assert!((rho_inf - 1999.0).abs() < 1.0);
    }

    #[test]
    fn test_variance_rectification() {
        let optimizer = RAdam::default_params(0.001);
        
        // Early steps should not be rectifiable
        assert!(!optimizer.is_variance_rectifiable(1));
        assert!(!optimizer.is_variance_rectifiable(2));
        
        // Later steps should be rectifiable (depends on beta2=0.999)
        // With beta2=0.999, it takes many more steps to become rectifiable
        assert!(optimizer.is_variance_rectifiable(1000));
        assert!(optimizer.is_variance_rectifiable(10000));
    }

    #[test]
    fn test_variance_rectification_term() {
        let optimizer = RAdam::default_params(0.001);
        
        // Rectification term should be positive and reasonable for large steps
        let rect_term = optimizer.variance_rectification_term(10000);
        assert!(rect_term > 0.0);
        assert!(rect_term < 5.0); // Should be reasonable magnitude, but can be larger than 2.0
    }

    #[test]
    fn test_radam_fallback_to_momentum() {
        let mut optimizer = RAdam::default_params(0.001);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);

        let original_param = param.clone();
        
        // In early steps, RAdam should fall back to momentum-only updates
        optimizer.step = 1; // Force early step
        optimizer.step(&param, &grad);

        // Should still update parameters even in momentum-only mode
        let param_data = param.as_slice().unwrap();
        let orig_data = original_param.as_slice().unwrap();
        
        for (new_val, orig_val) in param_data.iter().zip(orig_data.iter()) {
            assert!(new_val != orig_val); // Parameters should change
        }
    }
}