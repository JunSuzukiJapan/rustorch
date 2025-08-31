//! Enhanced optimizers for WebAssembly  
//! WebAssembly向け強化最適化アルゴリズム

use crate::optim::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// WASM-compatible optimizer implementations / WASM互換最適化実装

#[wasm_bindgen]
pub struct SGDWasm {
    learning_rate: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
    velocity: HashMap<String, Vec<f64>>,
}

#[wasm_bindgen]
impl SGDWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        learning_rate: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> SGDWasm {
        SGDWasm {
            learning_rate,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            velocity: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn step(&mut self, param_name: &str, params: &mut [f64], gradients: &[f64]) {
        if params.len() != gradients.len() {
            return;
        }

        // Get or initialize velocity
        let velocity = self
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradients[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Apply momentum
            if self.momentum != 0.0 {
                velocity[i] = self.momentum * velocity[i] + (1.0 - self.dampening) * grad;

                if self.nesterov {
                    grad = grad + self.momentum * velocity[i];
                } else {
                    grad = velocity[i];
                }
            }

            // Update parameter
            params[i] -= self.learning_rate * grad;
        }
    }

    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    #[wasm_bindgen]
    pub fn reset_state(&mut self) {
        self.velocity.clear();
    }
}

#[wasm_bindgen]
pub struct AdamWasm {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    step_count: u64,
    m: HashMap<String, Vec<f64>>, // First moment
    v: HashMap<String, Vec<f64>>, // Second moment
}

#[wasm_bindgen]
impl AdamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> AdamWasm {
        AdamWasm {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn step(&mut self, param_name: &str, params: &mut [f64], gradients: &[f64]) {
        if params.len() != gradients.len() {
            return;
        }

        self.step_count += 1;

        // Get or initialize moments
        let m = self
            .m
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);
        let v = self
            .v
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradients[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = m[i] / (1.0 - self.beta1.powi(self.step_count as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = v[i] / (1.0 - self.beta2.powi(self.step_count as i32));

            // Update parameter
            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    #[wasm_bindgen]
    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    #[wasm_bindgen]
    pub fn reset_state(&mut self) {
        self.step_count = 0;
        self.m.clear();
        self.v.clear();
    }
}

#[wasm_bindgen]
pub struct RMSpropWasm {
    learning_rate: f64,
    alpha: f64,
    epsilon: f64,
    weight_decay: f64,
    momentum: f64,
    v: HashMap<String, Vec<f64>>, // Moving average of squared gradients
    momentum_buffer: HashMap<String, Vec<f64>>,
}

#[wasm_bindgen]
impl RMSpropWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> RMSpropWasm {
        RMSpropWasm {
            learning_rate,
            alpha,
            epsilon,
            weight_decay,
            momentum,
            v: HashMap::new(),
            momentum_buffer: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn step(&mut self, param_name: &str, params: &mut [f64], gradients: &[f64]) {
        if params.len() != gradients.len() {
            return;
        }

        // Get or initialize state
        let v = self
            .v
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);
        let momentum_buffer = self
            .momentum_buffer
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradients[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Update exponential moving average of squared gradients
            v[i] = self.alpha * v[i] + (1.0 - self.alpha) * grad * grad;

            let update = if self.momentum > 0.0 {
                momentum_buffer[i] =
                    self.momentum * momentum_buffer[i] + grad / (v[i].sqrt() + self.epsilon);
                momentum_buffer[i]
            } else {
                grad / (v[i].sqrt() + self.epsilon)
            };

            // Update parameter
            params[i] -= self.learning_rate * update;
        }
    }

    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    #[wasm_bindgen]
    pub fn reset_state(&mut self) {
        self.v.clear();
        self.momentum_buffer.clear();
    }
}

#[wasm_bindgen]
pub struct AdaGradWasm {
    learning_rate: f64,
    epsilon: f64,
    weight_decay: f64,
    sum_sq_gradients: HashMap<String, Vec<f64>>,
}

#[wasm_bindgen]
impl AdaGradWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f64, epsilon: f64, weight_decay: f64) -> AdaGradWasm {
        AdaGradWasm {
            learning_rate,
            epsilon,
            weight_decay,
            sum_sq_gradients: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn step(&mut self, param_name: &str, params: &mut [f64], gradients: &[f64]) {
        if params.len() != gradients.len() {
            return;
        }

        // Get or initialize accumulated squared gradients
        let sum_sq = self
            .sum_sq_gradients
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradients[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }

            // Accumulate squared gradients
            sum_sq[i] += grad * grad;

            // Update parameter
            params[i] -= self.learning_rate * grad / (sum_sq[i].sqrt() + self.epsilon);
        }
    }

    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    #[wasm_bindgen]
    pub fn reset_state(&mut self) {
        self.sum_sq_gradients.clear();
    }
}

// Optimizer utility functions / 最適化ユーティリティ関数
#[wasm_bindgen]
pub fn learning_rate_schedule_wasm(
    initial_lr: f64,
    step: u64,
    decay_rate: f64,
    decay_steps: u64,
) -> f64 {
    initial_lr * decay_rate.powi((step / decay_steps) as i32)
}

#[wasm_bindgen]
pub fn cosine_annealing_wasm(initial_lr: f64, current_step: u64, total_steps: u64) -> f64 {
    let min_lr = initial_lr * 0.01;
    min_lr
        + (initial_lr - min_lr)
            * 0.5
            * (1.0 + ((current_step as f64 * std::f64::consts::PI) / total_steps as f64).cos())
}
