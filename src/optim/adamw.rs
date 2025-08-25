//! AdamW optimizer implementation
//! AdamWオプティマイザの実装 - 独立した重み減衰を持つAdam

use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// AdamW optimizer with decoupled weight decay
/// 分離された重み減衰を持つAdamWオプティマイザ
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    step: usize,
    state: HashMap<usize, AdamWState>,
}

/// State for each parameter in AdamW
struct AdamWState {
    momentum: Tensor<f32>,             // First moment estimate
    velocity: Tensor<f32>,             // Second moment estimate
    max_velocity: Option<Tensor<f32>>, // Max second moment (for AMSGrad)
}

impl AdamW {
    /// Create a new AdamW optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.001)
    /// * `beta1` - Coefficient for computing running averages of gradient (default: 0.9)
    /// * `beta2` - Coefficient for computing running averages of gradient square (default: 0.999)
    /// * `eps` - Term added to denominator for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    /// * `amsgrad` - Whether to use AMSGrad variant (default: false)
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
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
            amsgrad,
            step: 0,
            state: HashMap::new(),
        }
    }

    /// Create AdamW with default parameters
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.01, false)
    }

    /// Create AdamW with custom weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, weight_decay, false)
    }

    /// Create AdamW with AMSGrad
    pub fn with_amsgrad(learning_rate: f32, weight_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, weight_decay, true)
    }

    /// Get parameter state for debugging
    pub fn get_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.state
            .get(&param_id)
            .map(|s| (&s.momentum, &s.velocity))
    }

    fn get_or_create_state(&mut self, param_id: usize, shape: &[usize]) -> &mut AdamWState {
        self.state.entry(param_id).or_insert_with(|| AdamWState {
            momentum: Tensor::zeros(shape),
            velocity: Tensor::zeros(shape),
            max_velocity: if self.amsgrad {
                Some(Tensor::zeros(shape))
            } else {
                None
            },
        })
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        self.step += 1;
        let param_id = param as *const _ as usize;

        // Copy parameters to avoid borrow issues
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let learning_rate = self.learning_rate;
        let amsgrad = self.amsgrad;
        let step = self.step;

        // Get or create state for this parameter
        let state = self.get_or_create_state(param_id, param.shape());

        // Update biased first moment estimate
        state.momentum = &state.momentum * beta1 + grad * (1.0 - beta1);

        // Update biased second raw moment estimate
        let grad_squared = grad * grad;
        state.velocity = &state.velocity * beta2 + &grad_squared * (1.0 - beta2);

        // Compute bias-corrected first moment estimate
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let bias_correction2 = 1.0 - beta2.powi(step as i32);

        let momentum_corrected = &state.momentum / bias_correction1;

        // Compute bias-corrected second moment estimate and update
        let velocity_corrected = if amsgrad {
            // Use the max of past and current second moment
            if let Some(ref mut max_vel) = state.max_velocity {
                let current_velocity = &state.velocity / bias_correction2;

                // Element-wise maximum
                let max_data = max_vel.as_slice().unwrap();
                let curr_data = current_velocity.as_slice().unwrap();
                let mut new_max = Vec::with_capacity(max_data.len());

                for (m, c) in max_data.iter().zip(curr_data.iter()) {
                    new_max.push(m.max(*c));
                }

                *max_vel = Tensor::from_vec(new_max, max_vel.shape().to_vec());
                max_vel.clone()
            } else {
                &state.velocity / bias_correction2
            }
        } else {
            &state.velocity / bias_correction2
        };

        // Compute update with AdamW weight decay
        let velocity_sqrt = velocity_corrected.sqrt();
        let update = &momentum_corrected / &(&velocity_sqrt + eps);

        // Apply weight decay directly to parameters (decoupled from gradient)
        let param_with_decay = if weight_decay > 0.0 {
            param * (1.0 - learning_rate * weight_decay)
        } else {
            param.clone()
        };

        // Apply update
        let new_param = &param_with_decay - &(&update * learning_rate);

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
        // AdamW doesn't store gradients, so nothing to do here
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
        state.insert("amsgrad".to_string(), if self.amsgrad { 1.0 } else { 0.0 });
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
        if let Some(&amsgrad) = state.get("amsgrad") {
            self.amsgrad = amsgrad > 0.5;
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
    fn test_adamw_creation() {
        let optimizer = AdamW::default_params(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_with_weight_decay() {
        let optimizer = AdamW::with_weight_decay(0.001, 0.1);
        assert_eq!(optimizer.weight_decay, 0.1);
    }

    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamW::default_params(0.1);
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
    fn test_adamw_amsgrad() {
        let mut optimizer = AdamW::with_amsgrad(0.001, 0.01);
        assert!(optimizer.amsgrad);

        let param = Tensor::randn(&[10]);
        let grad = Tensor::randn(&[10]);

        optimizer.step(&param, &grad);

        // State should include max_velocity for AMSGrad
        let param_id = &param as *const _ as usize;
        assert!(optimizer
            .state
            .get(&param_id)
            .unwrap()
            .max_velocity
            .is_some());
    }
}
