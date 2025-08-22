/// Optimization algorithms for neural networks
/// ニューラルネットワーク用最適化アルゴリズム

use crate::tensor::Tensor;
use std::collections::HashMap;

/// Optimizer trait for parameter updates
pub trait Optimizer {
    /// Update parameters with gradients
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>);
    
    /// Zero gradients (if needed)
    fn zero_grad(&mut self) {}
    
    /// Get learning rate
    fn learning_rate(&self) -> f32;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
    
    /// Get optimizer state
    fn state_dict(&self) -> HashMap<String, f32>;
    
    /// Load optimizer state
    fn load_state_dict(&mut self, state: HashMap<String, f32>);
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    momentum_buffers: HashMap<usize, Tensor<f32>>,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }
    
    /// Create SGD with weight decay
    pub fn with_weight_decay(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }
    
    /// Create SGD with Nesterov momentum
    pub fn with_nesterov(learning_rate: f32, momentum: f32, nesterov: bool) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov,
            momentum_buffers: HashMap::new(),
        }
    }
    
    /// Set dampening factor
    pub fn set_dampening(&mut self, dampening: f32) {
        self.dampening = dampening;
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        let mut d_p = grad.clone();
        
        // Apply weight decay
        if self.weight_decay != 0.0 {
            let weight_decay_term = param * self.weight_decay;
            d_p = &d_p + &weight_decay_term;
        }
        
        // Apply momentum
        if self.momentum != 0.0 {
            let buf = if let Some(momentum_buffer) = self.momentum_buffers.get(&param_id) {
                let momentum_term = momentum_buffer * self.momentum;
                let dampening_term = &d_p * (1.0 - self.dampening);
                momentum_term + dampening_term
            } else {
                d_p.clone()
            };
            
            self.momentum_buffers.insert(param_id, buf.clone());
            
            if self.nesterov {
                let momentum_term = &buf * self.momentum;
                d_p = &d_p + &momentum_term;
            } else {
                d_p = buf;
            }
        }
        
        // Update parameters
        let lr_term = &d_p * self.learning_rate;
        let update = param - &lr_term;
        param.copy_from(&update);
    }
    
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
    
    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("momentum".to_string(), self.momentum);
        state.insert("dampening".to_string(), self.dampening);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("nesterov".to_string(), if self.nesterov { 1.0 } else { 0.0 });
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&momentum) = state.get("momentum") {
            self.momentum = momentum;
        }
        if let Some(&dampening) = state.get("dampening") {
            self.dampening = dampening;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&nesterov) = state.get("nesterov") {
            self.nesterov = nesterov > 0.5;
        }
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    amsgrad: bool,
    step_count: usize,
    exp_avg: HashMap<usize, Tensor<f32>>,
    exp_avg_sq: HashMap<usize, Tensor<f32>>,
    max_exp_avg_sq: HashMap<usize, Tensor<f32>>,
}

impl Adam {
    /// Create new Adam optimizer
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }
    
    /// Create Adam with default parameters
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8)
    }
    
    /// Create Adam with weight decay (AdamW)
    pub fn with_weight_decay(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            amsgrad: false,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }
    
    /// Create Adam with AMSGrad
    pub fn with_amsgrad(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, amsgrad: bool) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        self.step_count += 1;
        
        let mut d_p = grad.clone();
        
        // Apply weight decay
        if self.weight_decay != 0.0 {
            let weight_decay_term = param * self.weight_decay;
            d_p = &d_p + &weight_decay_term;
        }
        
        // Get or initialize momentum buffers
        let exp_avg = if let Some(avg) = self.exp_avg.get(&param_id) {
            let beta1_term = avg * self.beta1;
            let one_minus_beta1_term = &d_p * (1.0 - self.beta1);
            beta1_term + one_minus_beta1_term
        } else {
            d_p.clone() * (1.0 - self.beta1)
        };
        
        let exp_avg_sq = if let Some(avg_sq) = self.exp_avg_sq.get(&param_id) {
            let beta2_term = avg_sq * self.beta2;
            let d_p_squared = &d_p * &d_p;
            let one_minus_beta2_term = &d_p_squared * (1.0 - self.beta2);
            beta2_term + one_minus_beta2_term
        } else {
            let d_p_squared = &d_p * &d_p;
            d_p_squared * (1.0 - self.beta2)
        };
        
        self.exp_avg.insert(param_id, exp_avg.clone());
        self.exp_avg_sq.insert(param_id, exp_avg_sq.clone());
        
        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
        
        let corrected_exp_avg = &exp_avg / bias_correction1;
        let corrected_exp_avg_sq = &exp_avg_sq / bias_correction2;
        
        // Compute update
        let denom = if self.amsgrad {
            let max_exp_avg_sq = if let Some(max_avg_sq) = self.max_exp_avg_sq.get(&param_id) {
                max_avg_sq.maximum(&corrected_exp_avg_sq)
            } else {
                corrected_exp_avg_sq.clone()
            };
            self.max_exp_avg_sq.insert(param_id, max_exp_avg_sq.clone());
            max_exp_avg_sq.sqrt() + self.epsilon
        } else {
            corrected_exp_avg_sq.sqrt() + self.epsilon
        };
        
        let step_size = self.learning_rate;
        let update_term = (&corrected_exp_avg / &denom) * step_size;
        let update = param - &update_term;
        param.copy_from(&update);
    }
    
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
    
    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("amsgrad".to_string(), if self.amsgrad { 1.0 } else { 0.0 });
        state.insert("step_count".to_string(), self.step_count as f32);
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
        if let Some(&epsilon) = state.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&amsgrad) = state.get("amsgrad") {
            self.amsgrad = amsgrad > 0.5;
        }
        if let Some(&step_count) = state.get("step_count") {
            self.step_count = step_count as usize;
        }
    }
}