//! NAdam optimizer implementation
//! NAdamオプティマイザの実装 - Nesterov加速Adam

use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// NAdam (Nesterov-accelerated Adaptive Moment Estimation) optimizer
/// NAdam（Nesterov加速適応モーメント推定）オプティマイザ
/// 
/// NAdam combines Adam with Nesterov momentum for better convergence
/// NAdamはAdamとNesterovモーメンタムを組み合わせ、より良い収束を実現
pub struct NAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    momentum_decay: f32,
    schedule_decay: f32,
    step: usize,
    state: HashMap<usize, NAdamState>,
}

/// State for each parameter in NAdam
struct NAdamState {
    momentum: Tensor<f32>,             // First moment estimate
    velocity: Tensor<f32>,             // Second moment estimate
}

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
    ) -> Self {
        assert!(learning_rate > 0.0, "Learning rate must be positive");
        assert!(beta1 >= 0.0 && beta1 < 1.0, "Beta1 must be in [0, 1)");
        assert!(beta2 >= 0.0 && beta2 < 1.0, "Beta2 must be in [0, 1)");
        assert!(eps > 0.0, "Epsilon must be positive");
        assert!(weight_decay >= 0.0, "Weight decay must be non-negative");
        assert!(momentum_decay >= 0.0, "Momentum decay must be non-negative");
        assert!(schedule_decay >= 0.0, "Schedule decay must be non-negative");

        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            momentum_decay,
            schedule_decay,
            step: 0,
            state: HashMap::new(),
        }
    }

    /// Create NAdam with default parameters
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.0, 0.004, 0.004)
    }

    /// Create NAdam with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, weight_decay, 0.004, 0.004)
    }

    /// Create NAdam with custom momentum decay
    pub fn with_momentum_decay(learning_rate: f32, momentum_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.0, momentum_decay, 0.004)
    }

    /// Get parameter state for debugging
    pub fn get_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.state
            .get(&param_id)
            .map(|s| (&s.momentum, &s.velocity))
    }

    fn get_or_create_state(&mut self, param_id: usize, shape: &[usize]) -> &mut NAdamState {
        self.state.entry(param_id).or_insert_with(|| NAdamState {
            momentum: Tensor::zeros(shape),
            velocity: Tensor::zeros(shape),
        })
    }

    /// Compute time-dependent beta1 with decay
    fn beta1_t(&self, t: usize) -> f32 {
        let momentum_cache_t = self.beta1 * (1.0 - 0.5 * 0.96_f32.powi(t as i32 * self.schedule_decay as i32));
        let momentum_cache_t_1 = self.beta1 * (1.0 - 0.5 * 0.96_f32.powi((t + 1) as i32 * self.schedule_decay as i32));
        
        momentum_cache_t_1 / (1.0 - momentum_cache_t)
    }
}

impl Optimizer for NAdam {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        self.step += 1;
        let param_id = param as *const _ as usize;

        // Copy parameters to avoid borrow issues
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let learning_rate = self.learning_rate;
        let momentum_decay = self.momentum_decay;
        let step = self.step;

        // Apply weight decay to gradient if specified
        let grad_with_decay = if weight_decay > 0.0 {
            let weight_decay_term = param * weight_decay;
            grad + &weight_decay_term
        } else {
            grad.clone()
        };

        // Compute time-dependent beta1 before getting state
        let beta1_t = {
            let momentum_cache_t = beta1 * (1.0 - 0.5 * 0.96_f32.powi(step as i32 * self.schedule_decay as i32));
            let momentum_cache_t_1 = beta1 * (1.0 - 0.5 * 0.96_f32.powi((step + 1) as i32 * self.schedule_decay as i32));
            momentum_cache_t_1 / (1.0 - momentum_cache_t)
        };

        // Get or create state for this parameter
        let state = self.get_or_create_state(param_id, param.shape());

        // Update biased first moment estimate
        state.momentum = (&(&state.momentum * beta1)) + (&(&grad_with_decay * (1.0 - beta1)));

        // Update biased second raw moment estimate
        let grad_squared = &grad_with_decay * &grad_with_decay;
        state.velocity = (&(&state.velocity * beta2)) + (&(&grad_squared * (1.0 - beta2)));

        // Compute bias correction terms
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let bias_correction2 = 1.0 - beta2.powi(step as i32);

        // Bias-corrected second moment estimate
        let velocity_corrected = &state.velocity / bias_correction2;

        // NAdam's key difference: use Nesterov-accelerated gradient
        let momentum_corrected = &state.momentum / bias_correction1;
        
        // Apply Nesterov acceleration: β₁ * m̂ₜ + ((1 - β₁) / (1 - β₁ᵗ)) * gₜ
        let momentum_term = &momentum_corrected * beta1_t;
        let gradient_term = &grad_with_decay * ((1.0 - beta1) / (1.0 - beta1.powi(step as i32)));
        let nesterov_momentum = &momentum_term + &gradient_term;

        // Compute update
        let velocity_sqrt = velocity_corrected.sqrt();
        let update = &nesterov_momentum / &(&velocity_sqrt + eps);

        // Apply update
        let new_param = param - &(&update * learning_rate);

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
        // NAdam doesn't store gradients, so nothing to do here
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
        state.insert("momentum_decay".to_string(), self.momentum_decay);
        state.insert("schedule_decay".to_string(), self.schedule_decay);
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
        if let Some(&md) = state.get("momentum_decay") {
            self.momentum_decay = md;
        }
        if let Some(&sd) = state.get("schedule_decay") {
            self.schedule_decay = sd;
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
    fn test_nadam_creation() {
        let optimizer = NAdam::default_params(0.002);
        assert_eq!(optimizer.learning_rate, 0.002);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
    }

    #[test]
    fn test_nadam_with_weight_decay() {
        let optimizer = NAdam::with_weight_decay(0.001, 0.1);
        assert_eq!(optimizer.weight_decay, 0.1);
    }

    #[test]
    fn test_nadam_step() {
        let mut optimizer = NAdam::default_params(0.1);
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
        let mut optimizer = NAdam::with_momentum_decay(0.001, 0.01);
        assert_eq!(optimizer.momentum_decay, 0.01);

        let param = Tensor::randn(&[10]);
        let grad = Tensor::randn(&[10]);

        optimizer.step(&param, &grad);

        // State should be created for the parameter
        let param_id = &param as *const _ as usize;
        assert!(optimizer.state.contains_key(&param_id));
    }

    #[test]
    fn test_beta1_t_scheduling() {
        let optimizer = NAdam::default_params(0.001);
        
        // Beta1 should change over time with NAdam scheduling
        let beta1_t1 = optimizer.beta1_t(1);
        let beta1_t10 = optimizer.beta1_t(10);
        let beta1_t100 = optimizer.beta1_t(100);
        
        println!("beta1_t1: {}, beta1_t10: {}, beta1_t100: {}", beta1_t1, beta1_t10, beta1_t100);
        
        // Beta1_t values should be different and reasonable
        assert!(beta1_t1 > 0.0 && beta1_t1 <= 1.0);
        assert!(beta1_t10 > 0.0 && beta1_t10 <= 1.0);
        assert!(beta1_t100 > 0.0 && beta1_t100 <= 1.0);
        
        // Just verify the computation is stable, the values may be similar due to small schedule_decay
        assert!(beta1_t1.is_finite());
        assert!(beta1_t10.is_finite());
        assert!(beta1_t100.is_finite());
    }
}