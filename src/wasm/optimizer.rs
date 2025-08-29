//! Optimization algorithms for WASM neural networks
//! WASMニューラルネットワーク用最適化アルゴリズム

#[cfg(feature = "wasm")]
use std::collections::HashMap;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// SGD (Stochastic Gradient Descent) optimizer for WASM
/// WASM用SGD（確率的勾配降下法）オプティマイザ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSGD {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    momentum_buffers: HashMap<String, Vec<f32>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSGD {
    /// Create new SGD optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD with momentum
    #[wasm_bindgen]
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> WasmSGD {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD with weight decay
    #[wasm_bindgen]
    pub fn with_weight_decay(learning_rate: f32, momentum: f32, weight_decay: f32) -> WasmSGD {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            nesterov: false,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Update parameters with gradients
    #[wasm_bindgen]
    pub fn step(&mut self, param_id: &str, parameters: Vec<f32>, gradients: Vec<f32>) -> Vec<f32> {
        if parameters.len() != gradients.len() {
            panic!("Parameters and gradients must have the same length");
        }

        let mut d_p = gradients;

        // Apply weight decay
        if self.weight_decay != 0.0 {
            for (param, grad) in parameters.iter().zip(d_p.iter_mut()) {
                *grad += self.weight_decay * param;
            }
        }

        // Apply momentum
        if self.momentum != 0.0 {
            let momentum_buffer = if let Some(buf) = self.momentum_buffers.get(param_id) {
                buf.iter()
                    .zip(d_p.iter())
                    .map(|(&buf_val, &grad_val)| self.momentum * buf_val + grad_val)
                    .collect()
            } else {
                d_p.clone()
            };

            self.momentum_buffers
                .insert(param_id.to_string(), momentum_buffer.clone());

            if self.nesterov {
                for (i, grad) in d_p.iter_mut().enumerate() {
                    *grad += self.momentum * momentum_buffer[i];
                }
            } else {
                d_p = momentum_buffer;
            }
        }

        // Update parameters
        parameters
            .iter()
            .zip(d_p.iter())
            .map(|(param, grad)| param - self.learning_rate * grad)
            .collect()
    }

    /// Get learning rate
    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate
    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Clear momentum buffers
    #[wasm_bindgen]
    pub fn zero_grad(&mut self) {
        // In simple implementations, we don't need to zero gradients
        // as they are passed fresh each time
    }
}

/// Adam optimizer for WASM
/// WASM用Adamオプティマイザ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step_count: u32,
    exp_avg: HashMap<String, Vec<f32>>, // First moment estimates
    exp_avg_sq: HashMap<String, Vec<f32>>, // Second moment estimates
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAdam {
    /// Create new Adam optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f32) -> Self {
        Self::with_params(learning_rate, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Create Adam with custom parameters
    #[wasm_bindgen]
    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> WasmAdam {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Update parameters with gradients using Adam algorithm
    #[wasm_bindgen]
    pub fn step(&mut self, param_id: &str, parameters: Vec<f32>, gradients: Vec<f32>) -> Vec<f32> {
        if parameters.len() != gradients.len() {
            panic!("Parameters and gradients must have the same length");
        }

        self.step_count += 1;
        let mut d_p = gradients;

        // Apply weight decay
        if self.weight_decay != 0.0 {
            for (param, grad) in parameters.iter().zip(d_p.iter_mut()) {
                *grad += self.weight_decay * param;
            }
        }

        // Get or initialize first moment (momentum)
        let exp_avg: Vec<f32> = if let Some(avg) = self.exp_avg.get(param_id) {
            avg.iter()
                .zip(d_p.iter())
                .map(|(&avg_val, &grad_val)| self.beta1 * avg_val + (1.0 - self.beta1) * grad_val)
                .collect()
        } else {
            d_p.iter().map(|&grad| (1.0 - self.beta1) * grad).collect()
        };

        // Get or initialize second moment (uncentered variance)
        let exp_avg_sq: Vec<f32> = if let Some(avg_sq) = self.exp_avg_sq.get(param_id) {
            avg_sq
                .iter()
                .zip(d_p.iter())
                .map(|(&avg_sq_val, &grad_val)| {
                    self.beta2 * avg_sq_val + (1.0 - self.beta2) * grad_val * grad_val
                })
                .collect()
        } else {
            d_p.iter()
                .map(|&grad| (1.0 - self.beta2) * grad * grad)
                .collect()
        };

        // Store updated moments
        self.exp_avg.insert(param_id.to_string(), exp_avg.clone());
        self.exp_avg_sq
            .insert(param_id.to_string(), exp_avg_sq.clone());

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        let step_size = self.learning_rate / bias_correction1;

        // Update parameters
        parameters
            .iter()
            .zip(exp_avg.iter())
            .zip(exp_avg_sq.iter())
            .map(|((&param, &avg), &avg_sq)| {
                let corrected_avg_sq = avg_sq / bias_correction2;
                let denom = corrected_avg_sq.sqrt() + self.epsilon;
                param - step_size * avg / denom
            })
            .collect()
    }

    /// Get learning rate
    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate
    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Get step count
    #[wasm_bindgen]
    pub fn get_step_count(&self) -> u32 {
        self.step_count
    }

    /// Reset optimizer state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
    }
}

/// AdaGrad optimizer for WASM (simpler than Adam)
/// WASM用AdaGradオプティマイザ（Adamより簡単）
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAdaGrad {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: f32,
    sum_of_squares: HashMap<String, Vec<f32>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAdaGrad {
    /// Create new AdaGrad optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: 0.0,
            sum_of_squares: HashMap::new(),
        }
    }

    /// Update parameters with AdaGrad algorithm
    #[wasm_bindgen]
    pub fn step(&mut self, param_id: &str, parameters: Vec<f32>, gradients: Vec<f32>) -> Vec<f32> {
        if parameters.len() != gradients.len() {
            panic!("Parameters and gradients must have the same length");
        }

        let mut d_p = gradients;

        // Apply weight decay
        if self.weight_decay != 0.0 {
            for (param, grad) in parameters.iter().zip(d_p.iter_mut()) {
                *grad += self.weight_decay * param;
            }
        }

        // Update sum of squares
        let sum_of_squares: Vec<f32> = if let Some(sos) = self.sum_of_squares.get(param_id) {
            sos.iter()
                .zip(d_p.iter())
                .map(|(&sos_val, &grad_val)| sos_val + grad_val * grad_val)
                .collect()
        } else {
            d_p.iter().map(|&grad| grad * grad).collect()
        };

        self.sum_of_squares
            .insert(param_id.to_string(), sum_of_squares.clone());

        // Update parameters
        parameters
            .iter()
            .zip(d_p.iter())
            .zip(sum_of_squares.iter())
            .map(|((&param, &grad), &sos)| {
                let adaptive_lr = self.learning_rate / (sos.sqrt() + self.epsilon);
                param - adaptive_lr * grad
            })
            .collect()
    }
}

/// RMSprop optimizer for WASM
/// WASM用RMSpropオプティマイザ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmRMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    square_avg: HashMap<String, Vec<f32>>,
    momentum_buffer: HashMap<String, Vec<f32>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmRMSprop {
    /// Create new RMSprop optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f32, alpha: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum: 0.0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
        }
    }

    /// Create RMSprop with momentum
    #[wasm_bindgen]
    pub fn with_momentum(
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        momentum: f32,
    ) -> WasmRMSprop {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
        }
    }

    /// Update parameters with RMSprop algorithm
    #[wasm_bindgen]
    pub fn step(&mut self, param_id: &str, parameters: Vec<f32>, gradients: Vec<f32>) -> Vec<f32> {
        if parameters.len() != gradients.len() {
            panic!("Parameters and gradients must have the same length");
        }

        let mut d_p = gradients;

        // Apply weight decay
        if self.weight_decay != 0.0 {
            for (param, grad) in parameters.iter().zip(d_p.iter_mut()) {
                *grad += self.weight_decay * param;
            }
        }

        // Update exponential moving average of squared gradients
        let square_avg: Vec<f32> = if let Some(sq_avg) = self.square_avg.get(param_id) {
            sq_avg
                .iter()
                .zip(d_p.iter())
                .map(|(&avg_val, &grad_val)| {
                    self.alpha * avg_val + (1.0 - self.alpha) * grad_val * grad_val
                })
                .collect()
        } else {
            d_p.iter()
                .map(|&grad| (1.0 - self.alpha) * grad * grad)
                .collect()
        };

        self.square_avg
            .insert(param_id.to_string(), square_avg.clone());

        // Apply momentum if specified
        let update_values = if self.momentum > 0.0 {
            let momentum_buffer: Vec<f32> = if let Some(buf) = self.momentum_buffer.get(param_id) {
                buf.iter()
                    .zip(d_p.iter())
                    .zip(square_avg.iter())
                    .map(|((&buf_val, &grad_val), &sq_avg)| {
                        let adaptive_grad = grad_val / (sq_avg.sqrt() + self.epsilon);
                        self.momentum * buf_val + self.learning_rate * adaptive_grad
                    })
                    .collect()
            } else {
                d_p.iter()
                    .zip(square_avg.iter())
                    .map(|(&grad_val, &sq_avg)| {
                        let adaptive_grad = grad_val / (sq_avg.sqrt() + self.epsilon);
                        self.learning_rate * adaptive_grad
                    })
                    .collect()
            };

            self.momentum_buffer
                .insert(param_id.to_string(), momentum_buffer.clone());
            momentum_buffer
        } else {
            // No momentum
            d_p.iter()
                .zip(square_avg.iter())
                .map(|(&grad_val, &sq_avg)| {
                    let adaptive_grad = grad_val / (sq_avg.sqrt() + self.epsilon);
                    self.learning_rate * adaptive_grad
                })
                .collect()
        };

        // Update parameters
        parameters
            .iter()
            .zip(update_values.iter())
            .map(|(&param, &update)| param - update)
            .collect()
    }
}

/// Learning rate scheduler for WASM optimizers
/// WASMオプティマイザ用学習率スケジューラ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLRScheduler {
    initial_lr: f32,
    current_lr: f32,
    step_count: u32,
    scheduler_type: String,
    // Parameters for different scheduler types
    step_size: u32, // For StepLR
    gamma: f32,     // For StepLR and ExponentialLR
    t_max: u32,     // For CosineAnnealingLR
    eta_min: f32,   // For CosineAnnealingLR
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLRScheduler {
    /// Create StepLR scheduler
    #[wasm_bindgen]
    pub fn step_lr(initial_lr: f32, step_size: u32, gamma: f32) -> WasmLRScheduler {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_count: 0,
            scheduler_type: "step".to_string(),
            step_size,
            gamma,
            t_max: 0,
            eta_min: 0.0,
        }
    }

    /// Create ExponentialLR scheduler
    #[wasm_bindgen]
    pub fn exponential_lr(initial_lr: f32, gamma: f32) -> WasmLRScheduler {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_count: 0,
            scheduler_type: "exponential".to_string(),
            step_size: 0,
            gamma,
            t_max: 0,
            eta_min: 0.0,
        }
    }

    /// Create CosineAnnealingLR scheduler
    #[wasm_bindgen]
    pub fn cosine_annealing_lr(initial_lr: f32, t_max: u32, eta_min: f32) -> WasmLRScheduler {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_count: 0,
            scheduler_type: "cosine".to_string(),
            step_size: 0,
            gamma: 0.0,
            t_max,
            eta_min,
        }
    }

    /// Step the scheduler and get updated learning rate
    #[wasm_bindgen]
    pub fn step(&mut self) -> f32 {
        self.step_count += 1;

        self.current_lr = match self.scheduler_type.as_str() {
            "step" => {
                if self.step_count % self.step_size == 0 {
                    self.current_lr * self.gamma
                } else {
                    self.current_lr
                }
            }
            "exponential" => self.initial_lr * self.gamma.powi(self.step_count as i32),
            "cosine" => {
                let t = (self.step_count % self.t_max) as f32;
                self.eta_min
                    + (self.initial_lr - self.eta_min)
                        * (1.0 + (std::f32::consts::PI * t / self.t_max as f32).cos())
                        / 2.0
            }
            _ => self.current_lr,
        };

        self.current_lr
    }

    /// Get current learning rate
    #[wasm_bindgen]
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Reset scheduler
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.current_lr = self.initial_lr;
    }
}

/// Optimizer factory for creating different optimizers
/// 異なるオプティマイザを作成するファクトリ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmOptimizerFactory;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmOptimizerFactory {
    /// Create optimizer by name
    #[wasm_bindgen]
    pub fn create_sgd(learning_rate: f32, momentum: f32, weight_decay: f32) -> WasmSGD {
        WasmSGD::with_weight_decay(learning_rate, momentum, weight_decay)
    }

    /// Create Adam optimizer
    #[wasm_bindgen]
    pub fn create_adam(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> WasmAdam {
        WasmAdam::with_params(learning_rate, beta1, beta2, epsilon, weight_decay)
    }

    /// Create AdaGrad optimizer
    #[wasm_bindgen]
    pub fn create_adagrad(learning_rate: f32, epsilon: f32) -> WasmAdaGrad {
        WasmAdaGrad::new(learning_rate, epsilon)
    }

    /// Create RMSprop optimizer
    #[wasm_bindgen]
    pub fn create_rmsprop(
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        momentum: f32,
    ) -> WasmRMSprop {
        WasmRMSprop::with_momentum(learning_rate, alpha, epsilon, momentum)
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_optimizer() {
        let mut sgd = WasmSGD::new(0.01);

        let params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        let updated_params = sgd.step("param1", params.clone(), grads);

        // Check that parameters moved in the opposite direction of gradients
        for (i, (&original, &updated)) in params.iter().zip(updated_params.iter()).enumerate() {
            assert!(updated < original, "Parameter {} should decrease", i);
        }
    }

    #[test]
    fn test_adam_optimizer() {
        let mut adam = WasmAdam::new(0.001);

        let params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        let updated_params = adam.step("param1", params.clone(), grads);

        assert_eq!(updated_params.len(), params.len());
        assert_eq!(adam.get_step_count(), 1);

        // Parameters should have moved
        for (&original, &updated) in params.iter().zip(updated_params.iter()) {
            assert_ne!(original, updated);
        }
    }

    #[test]
    fn test_adagrad_optimizer() {
        let mut adagrad = WasmAdaGrad::new(0.01, 1e-8);

        let params = vec![1.0, 2.0];
        let grads = vec![0.1, 0.2];

        let updated_params = adagrad.step("param1", params.clone(), grads);

        assert_eq!(updated_params.len(), 2);

        // Check that parameters moved in the opposite direction of gradients
        for (&original, &updated) in params.iter().zip(updated_params.iter()) {
            assert!(updated < original);
        }
    }

    #[test]
    fn test_lr_scheduler() {
        let mut scheduler = WasmLRScheduler::step_lr(0.1, 10, 0.5);

        assert_eq!(scheduler.get_lr(), 0.1);

        // Step 9 times (no change yet)
        for _ in 0..9 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.1);

        // Step once more (should trigger decay)
        scheduler.step();
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let mut scheduler = WasmLRScheduler::cosine_annealing_lr(1.0, 100, 0.0);

        assert_eq!(scheduler.get_lr(), 1.0);

        // At t_max/2, lr should be at minimum
        for _ in 0..50 {
            scheduler.step();
        }

        let mid_lr = scheduler.get_lr();
        assert!(mid_lr < 0.1); // Should be close to eta_min (0.0)
    }

    #[test]
    fn test_optimizer_factory() {
        let sgd = WasmOptimizerFactory::create_sgd(0.01, 0.9, 1e-4);
        let adam = WasmOptimizerFactory::create_adam(0.001, 0.9, 0.999, 1e-8, 1e-2);
        let adagrad = WasmOptimizerFactory::create_adagrad(0.01, 1e-8);
        let rmsprop = WasmOptimizerFactory::create_rmsprop(0.01, 0.99, 1e-8, 0.0);

        assert_eq!(sgd.get_learning_rate(), 0.01);
        assert_eq!(adam.get_learning_rate(), 0.001);
        assert_eq!(adagrad.learning_rate, 0.01);
        assert_eq!(rmsprop.learning_rate, 0.01);
    }
}
