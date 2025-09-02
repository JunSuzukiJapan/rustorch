/// Optimization algorithms for neural networks
/// ニューラルネットワーク用最適化アルゴリズム
pub mod adamw;
pub mod lr_scheduler;

// Phase 2: Advanced Adam-based optimizers
/// NAdam optimizer - Nesterov-accelerated Adam
pub mod nadam;
/// RAdam optimizer - Rectified Adam with variance rectification
pub mod radam;
/// Adamax optimizer - Adam with infinity norm
pub mod adamax;

// Advanced optimizers
/// AdaBound optimizer bridging Adam and SGD
pub mod adabound;
/// LAMB optimizer for large batch training
pub mod lamb;
/// L-BFGS second-order optimizer with enhanced line search
pub mod lbfgs;

/// Optimization utilities for memory efficiency and numerical stability
pub mod utils;

// Standard optimizer exports
pub use adamw::AdamW;
pub use lr_scheduler::{
    AnnealStrategy, CosineAnnealingLR, ExponentialLR, LRScheduler, MultiStepLR, OneCycleLR,
    PlateauMode, PolynomialLR, ReduceLROnPlateau, StepLR, ThresholdMode, WarmupScheduler,
};

// Phase 2 optimizer exports
pub use nadam::NAdam;
pub use radam::RAdam;
pub use adamax::Adamax;

// Re-export advanced optimizers
pub use adabound::AdaBound;
pub use lamb::LAMB;
pub use lbfgs::{LineSearchMethod, LBFGS};

/// SGD (Stochastic Gradient Descent) optimizer module
/// SGD（確率的勾配降下法）オプティマイザモジュール
pub mod sgd {
    pub use super::SGD;
}

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::ops::Add;

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
    pub fn new(learning_rate: f32) -> Self {
        Self::with_momentum(learning_rate, 0.0)
    }

    /// Create new SGD optimizer with momentum
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
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
                (&momentum_term) + (&dampening_term)
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
        state.insert(
            "nesterov".to_string(),
            if self.nesterov { 1.0 } else { 0.0 },
        );
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
    pub fn with_weight_decay(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
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
    pub fn with_amsgrad(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        amsgrad: bool,
    ) -> Self {
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
            let one_minus_beta1_term = (&d_p) * (1.0 - self.beta1);
            (&beta1_term) + (&one_minus_beta1_term)
        } else {
            d_p.clone() * (1.0 - self.beta1)
        };

        let exp_avg_sq = if let Some(avg_sq) = self.exp_avg_sq.get(&param_id) {
            let beta2_term = avg_sq * self.beta2;
            let d_p_squared = &d_p * &d_p;
            let one_minus_beta2_term = (&d_p_squared) * (1.0 - self.beta2);
            (&beta2_term) + (&one_minus_beta2_term)
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
                max_avg_sq
                    .maximum(&corrected_exp_avg_sq)
                    .unwrap_or_else(|_| corrected_exp_avg_sq.clone())
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

/// RMSprop optimizer
#[derive(Debug, Clone)]
pub struct RMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    step_count: usize,
    square_avg: HashMap<usize, Tensor<f32>>,
    momentum_buffer: HashMap<usize, Tensor<f32>>,
    grad_avg: HashMap<usize, Tensor<f32>>, // for centered variant
}

impl RMSprop {
    /// Create new RMSprop optimizer
    /// 新しいRMSpropオプティマイザーを作成
    pub fn new(learning_rate: f32, alpha: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            step_count: 0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }

    /// Create RMSprop with default parameters
    /// デフォルトパラメータでRMSpropを作成
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.99, 1e-8)
    }

    /// Create RMSprop with momentum
    /// モーメンタム付きRMSpropを作成
    pub fn with_momentum(learning_rate: f32, alpha: f32, epsilon: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum,
            centered: false,
            step_count: 0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }

    /// Create centered RMSprop
    /// センタード RMSpropを作成
    pub fn centered(learning_rate: f32, alpha: f32, epsilon: f32, centered: bool) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum: 0.0,
            centered,
            step_count: 0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }

    /// Create RMSprop with weight decay
    /// 重み減衰付きRMSpropを作成
    pub fn with_weight_decay(
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay,
            momentum: 0.0,
            centered: false,
            step_count: 0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        self.step_count += 1;

        let mut d_p = grad.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            let weight_decay_term = param * self.weight_decay;
            d_p = &d_p + &weight_decay_term;
        }

        // Update biased second raw moment estimate
        let square_avg = if let Some(sq_avg) = self.square_avg.get(&param_id) {
            let alpha_term = sq_avg * self.alpha;
            let grad_squared = &d_p * &d_p;
            let one_minus_alpha_term = (&grad_squared) * (1.0 - self.alpha);
            (&alpha_term) + (&one_minus_alpha_term)
        } else {
            let grad_squared = &d_p * &d_p;
            grad_squared * (1.0 - self.alpha)
        };

        self.square_avg.insert(param_id, square_avg.clone());

        let avg = if self.centered {
            // Centered variant: subtract the squared mean of gradients
            let grad_avg = if let Some(g_avg) = self.grad_avg.get(&param_id) {
                let alpha_term = g_avg * self.alpha;
                let one_minus_alpha_term = (&d_p) * (1.0 - self.alpha);
                (&alpha_term) + (&one_minus_alpha_term)
            } else {
                d_p.clone() * (1.0 - self.alpha)
            };

            self.grad_avg.insert(param_id, grad_avg.clone());

            // avg = square_avg - grad_avg^2
            let grad_avg_squared = &grad_avg * &grad_avg;
            &square_avg - &grad_avg_squared
        } else {
            square_avg.clone()
        };

        // Compute update
        let denom = avg.sqrt() + self.epsilon;

        if self.momentum > 0.0 {
            // Apply momentum
            let buf = if let Some(momentum_buf) = self.momentum_buffer.get(&param_id) {
                let momentum_term = momentum_buf * self.momentum;
                let grad_term = (&d_p / &denom) * self.learning_rate;
                (&momentum_term) + (&grad_term)
            } else {
                (&d_p / &denom) * self.learning_rate
            };

            self.momentum_buffer.insert(param_id, buf.clone());
            let update = param - &buf;
            param.copy_from(&update);
        } else {
            // No momentum
            let update_term = (&d_p / &denom) * self.learning_rate;
            let update = param - &update_term;
            param.copy_from(&update);
        }
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
        state.insert("alpha".to_string(), self.alpha);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("momentum".to_string(), self.momentum);
        state.insert(
            "centered".to_string(),
            if self.centered { 1.0 } else { 0.0 },
        );
        state.insert("step_count".to_string(), self.step_count as f32);
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&alpha) = state.get("alpha") {
            self.alpha = alpha;
        }
        if let Some(&epsilon) = state.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&momentum) = state.get("momentum") {
            self.momentum = momentum;
        }
        if let Some(&centered) = state.get("centered") {
            self.centered = centered > 0.5;
        }
        if let Some(&step_count) = state.get("step_count") {
            self.step_count = step_count as usize;
        }
    }
}

/// AdaGrad optimizer
#[derive(Debug, Clone)]
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: f32,
    initial_accumulator_value: f32,
    step_count: usize,
    sum_of_squares: HashMap<usize, Tensor<f32>>,
}

impl AdaGrad {
    /// Create new AdaGrad optimizer
    /// 新しいAdaGradオプティマイザーを作成
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            step_count: 0,
            sum_of_squares: HashMap::new(),
        }
    }

    /// Create AdaGrad with default parameters
    /// デフォルトパラメータでAdaGradを作成
    pub fn default_params(learning_rate: f32) -> Self {
        Self::new(learning_rate, 1e-10)
    }

    /// Create AdaGrad with weight decay
    /// 重み減衰付きAdaGradを作成
    pub fn with_weight_decay(learning_rate: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay,
            initial_accumulator_value: 0.0,
            step_count: 0,
            sum_of_squares: HashMap::new(),
        }
    }

    /// Create AdaGrad with initial accumulator value
    /// 初期アキュムレータ値付きAdaGradを作成
    pub fn with_initial_accumulator(
        learning_rate: f32,
        epsilon: f32,
        initial_accumulator_value: f32,
    ) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: 0.0,
            initial_accumulator_value,
            step_count: 0,
            sum_of_squares: HashMap::new(),
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        self.step_count += 1;

        let mut d_p = grad.clone();

        // Apply weight decay
        if self.weight_decay != 0.0 {
            let weight_decay_term = param * self.weight_decay;
            d_p = &d_p + &weight_decay_term;
        }

        // Update sum of squares of gradients
        let sum_of_squares = if let Some(sos) = self.sum_of_squares.get(&param_id) {
            let grad_squared = &d_p * &d_p;
            sos + &grad_squared
        } else {
            // Initialize with initial accumulator value if first step
            let grad_squared = &d_p * &d_p;
            if self.initial_accumulator_value > 0.0 {
                grad_squared + self.initial_accumulator_value
            } else {
                grad_squared
            }
        };

        self.sum_of_squares.insert(param_id, sum_of_squares.clone());

        // Compute adaptive learning rate
        let std = sum_of_squares.sqrt() + self.epsilon;
        let update_term = (&d_p / &std) * self.learning_rate;
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
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert(
            "initial_accumulator_value".to_string(),
            self.initial_accumulator_value,
        );
        state.insert("step_count".to_string(), self.step_count as f32);
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&epsilon) = state.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&initial_accumulator_value) = state.get("initial_accumulator_value") {
            self.initial_accumulator_value = initial_accumulator_value;
        }
        if let Some(&step_count) = state.get("step_count") {
            self.step_count = step_count as usize;
        }
    }
}
