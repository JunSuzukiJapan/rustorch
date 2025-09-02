//! Adamax optimizer implementation
//! Adamaxオプティマイザの実装 - Adam with infinity norm

use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Adamax optimizer (Adam with infinity norm)
/// Adamaxオプティマイザ（無限大ノルムを使用するAdam）
/// 
/// Adamax is a variant of Adam where the second moment is replaced by the infinity norm
/// Adamaxは第2モーメントを無限大ノルムに置き換えたAdamの変種
pub struct Adamax {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    state: HashMap<usize, AdamaxState>,
}

/// State for each parameter in Adamax
struct AdamaxState {
    momentum: Tensor<f32>,             // First moment estimate
    infinity_norm: Tensor<f32>,        // Infinity norm (max) estimate
}

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

    /// Create Adamax with default parameters
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-7, 0.0)
    }

    /// Create Adamax with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-7, weight_decay)
    }

    /// Get parameter state for debugging
    pub fn get_state(&self, param_id: usize) -> Option<(&Tensor<f32>, &Tensor<f32>)> {
        self.state
            .get(&param_id)
            .map(|s| (&s.momentum, &s.infinity_norm))
    }

    fn get_or_create_state(&mut self, param_id: usize, shape: &[usize]) -> &mut AdamaxState {
        self.state.entry(param_id).or_insert_with(|| AdamaxState {
            momentum: Tensor::zeros(shape),
            infinity_norm: Tensor::zeros(shape),
        })
    }

    /// Compute element-wise maximum between two tensors
    fn tensor_max(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        let a_data = a.as_slice().unwrap();
        let b_data = b.as_slice().unwrap();
        
        let max_data: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(a_val, b_val)| a_val.max(*b_val))
            .collect();
            
        Tensor::from_vec(max_data, a.shape().to_vec())
    }

    /// Compute element-wise absolute value of tensor
    fn tensor_abs(tensor: &Tensor<f32>) -> Tensor<f32> {
        let data = tensor.as_slice().unwrap();
        let abs_data: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        Tensor::from_vec(abs_data, tensor.shape().to_vec())
    }
}

impl Optimizer for Adamax {
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

        // Update the exponentially weighted infinity norm (key difference from Adam)
        let grad_abs = Self::tensor_abs(&grad_with_decay);
        let beta2_scaled = &state.infinity_norm * beta2;
        state.infinity_norm = Self::tensor_max(&beta2_scaled, &grad_abs);

        // Compute bias correction for first moment
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let momentum_corrected = &state.momentum / bias_correction1;

        // Note: No bias correction for infinity norm in Adamax
        // The infinity norm is already "corrected" by its max operation

        // Compute update using infinity norm instead of sqrt of second moment
        let update = &momentum_corrected / &(&state.infinity_norm + eps);

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
        // Adamax doesn't store gradients, so nothing to do here
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
    fn test_adamax_creation() {
        let optimizer = Adamax::default_params(0.002);
        assert_eq!(optimizer.learning_rate, 0.002);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.eps, 1e-7);
    }

    #[test]
    fn test_adamax_with_weight_decay() {
        let optimizer = Adamax::with_weight_decay(0.001, 0.1);
        assert_eq!(optimizer.weight_decay, 0.1);
    }

    #[test]
    fn test_adamax_step() {
        let mut optimizer = Adamax::default_params(0.1);
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
        
        let max_tensor = Adamax::tensor_max(&a, &b);
        let max_data = max_tensor.as_slice().unwrap();
        
        assert_eq!(max_data, &[3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_tensor_abs() {
        let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        let abs_tensor = Adamax::tensor_abs(&tensor);
        let abs_data = abs_tensor.as_slice().unwrap();
        
        assert_eq!(abs_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_adamax_infinity_norm_update() {
        let mut optimizer = Adamax::default_params(0.001);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let grad1 = Tensor::from_vec(vec![0.1, 0.5, 0.2], vec![3]);
        let grad2 = Tensor::from_vec(vec![0.3, 0.1, 0.8], vec![3]);

        let param_id = &param as *const _ as usize;
        
        // First step
        optimizer.step(&param, &grad1);
        let inf_norm1 = {
            let state = optimizer.state.get(&param_id).unwrap();
            state.infinity_norm.clone()
        };
        let inf_norm1_data = inf_norm1.as_slice().unwrap();
        
        // Second step with different gradient
        optimizer.step(&param, &grad2);
        let inf_norm2 = {
            let state = optimizer.state.get(&param_id).unwrap();
            state.infinity_norm.clone()
        };
        let inf_norm2_data = inf_norm2.as_slice().unwrap();
        
        // Infinity norm should track the maximum absolute values
        // After step 1: approximately [0.1, 0.5, 0.2]
        // After step 2: should be max of (beta2 * prev, |new_grad|)
        assert!(inf_norm2_data[2] > inf_norm1_data[2]); // 0.8 > previous values
    }

    #[test]
    fn test_adamax_no_bias_correction_for_infinity_norm() {
        let mut optimizer = Adamax::default_params(0.1);
        let param = Tensor::from_vec(vec![1.0], vec![1]);
        let grad = Tensor::from_vec(vec![1.0], vec![1]);

        let original_param_val = param.as_slice().unwrap()[0];
        
        // Take a step
        optimizer.step(&param, &grad);
        
        let new_param_val = param.as_slice().unwrap()[0];
        let param_id = &param as *const _ as usize;
        let state = optimizer.state.get(&param_id).unwrap();
        
        // The infinity norm should be used directly (no bias correction)
        // Unlike Adam, Adamax doesn't need bias correction for the second moment
        assert!(state.infinity_norm.as_slice().unwrap()[0] > 0.0);
        assert!(new_param_val != original_param_val);
    }
}