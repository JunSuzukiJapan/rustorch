//! Optimization utilities for memory efficiency and numerical stability
//! メモリ効率と数値安定性のための最適化ユーティリティ

use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;

/// Utility functions for optimizer memory management and numerical stability
/// オプティマイザのメモリ管理と数値安定性のためのユーティリティ関数
pub struct OptimizerUtils;

impl OptimizerUtils {
    /// Ensure numerical stability by clamping values to a safe range
    /// 値を安全な範囲にクランプして数値安定性を確保
    pub fn clamp_for_stability(tensor: &Tensor<f32>, min_val: f32, max_val: f32) -> Tensor<f32> {
        let data = tensor.as_slice().unwrap();
        let clamped_data: Vec<f32> = data.iter()
            .map(|&x| x.max(min_val).min(max_val))
            .collect();
        Tensor::from_vec(clamped_data, tensor.shape().to_vec())
    }

    /// Apply gradient clipping by norm to prevent gradient explosion
    /// 勾配爆発を防ぐためにノルムによる勾配クリッピングを適用
    pub fn clip_gradient_norm(grad: &Tensor<f32>, max_norm: f32) -> Tensor<f32> {
        let grad_squared = grad * grad;
        let grad_norm = grad_squared.sum().sqrt();
        
        if grad_norm > max_norm {
            let scale_factor = max_norm / grad_norm;
            grad * scale_factor
        } else {
            grad.clone()
        }
    }

    /// Apply gradient clipping by value to prevent numerical instability
    /// 数値不安定性を防ぐために値による勾配クリッピングを適用
    pub fn clip_gradient_value(grad: &Tensor<f32>, min_val: f32, max_val: f32) -> Tensor<f32> {
        Self::clamp_for_stability(grad, min_val, max_val)
    }

    /// Compute numerically stable square root with epsilon
    /// イプシロンによる数値安定な平方根を計算
    pub fn stable_sqrt(tensor: &Tensor<f32>, eps: f32) -> Tensor<f32> {
        let data = tensor.as_slice().unwrap();
        let sqrt_data: Vec<f32> = data.iter()
            .map(|&x| (x + eps).sqrt())
            .collect();
        Tensor::from_vec(sqrt_data, tensor.shape().to_vec())
    }

    /// Apply exponential moving average efficiently
    /// 指数移動平均を効率的に適用
    pub fn ema_update(
        current: &mut Tensor<f32>,
        new_value: &Tensor<f32>,
        decay: f32,
    ) {
        // current = decay * current + (1 - decay) * new_value
        let decay_term = &*current * decay;
        let new_term = new_value * (1.0 - decay);
        let updated = &decay_term + &new_term;
        current.copy_from(&updated);
    }

    /// Check if tensor contains NaN or infinity values
    /// テンソルにNaNや無限大値が含まれているかチェック
    pub fn has_invalid_values(tensor: &Tensor<f32>) -> bool {
        let data = tensor.as_slice().unwrap();
        data.iter().any(|&x| x.is_nan() || x.is_infinite())
    }

    /// Replace NaN and infinity values with specified defaults
    /// NaNと無限大値を指定されたデフォルト値で置換
    pub fn sanitize_tensor(tensor: &Tensor<f32>, nan_replacement: f32, inf_replacement: f32) -> Tensor<f32> {
        let data = tensor.as_slice().unwrap();
        let sanitized_data: Vec<f32> = data.iter()
            .map(|&x| {
                if x.is_nan() {
                    nan_replacement
                } else if x.is_infinite() {
                    if x > 0.0 { inf_replacement } else { -inf_replacement }
                } else {
                    x
                }
            })
            .collect();
        Tensor::from_vec(sanitized_data, tensor.shape().to_vec())
    }

    /// Compute tensor L2 norm efficiently
    /// テンソルのL2ノルムを効率的に計算
    pub fn l2_norm(tensor: &Tensor<f32>) -> f32 {
        let squared = tensor * tensor;
        squared.sum().sqrt()
    }

    /// Compute tensor L1 norm efficiently
    /// テンソルのL1ノルムを効率的に計算
    pub fn l1_norm(tensor: &Tensor<f32>) -> f32 {
        let data = tensor.as_slice().unwrap();
        data.iter().map(|&x| x.abs()).sum()
    }

    /// Apply weight decay efficiently in-place
    /// 重み減衰をインプレースで効率的に適用
    pub fn apply_weight_decay(param: &Tensor<f32>, weight_decay: f32) -> Tensor<f32> {
        param * (1.0 - weight_decay)
    }

    /// Apply weight decay to gradient (AdamW style)
    /// 勾配に重み減衰を適用（AdamW形式）
    pub fn apply_weight_decay_to_grad(grad: &Tensor<f32>, param: &Tensor<f32>, weight_decay: f32) -> Tensor<f32> {
        if weight_decay > 0.0 {
            let weight_decay_term = param * weight_decay;
            grad + &weight_decay_term
        } else {
            grad.clone()
        }
    }

    /// Update momentum (first moment) with numerical stability
    /// 数値安定性を考慮したモーメンタム（第一モーメント）更新
    pub fn update_momentum(momentum: &mut Tensor<f32>, grad: &Tensor<f32>, beta1: f32) -> RusTorchResult<()> {
        if !(0.0..1.0).contains(&beta1) {
            return Err(RusTorchError::InvalidParameters {
                operation: "momentum update".to_string(),
                message: format!("Beta1 must be in [0, 1), got {}", beta1),
            });
        }

        let beta1_term = &*momentum * beta1;
        let grad_term = grad * (1.0 - beta1);
        let updated = &beta1_term + &grad_term;
        
        // Check for numerical stability
        if Self::has_invalid_values(&updated) {
            return Err(RusTorchError::InvalidParameters {
                operation: "momentum update".to_string(),
                message: "Numerical instability detected in momentum update".to_string(),
            });
        }
        
        momentum.copy_from(&updated);
        Ok(())
    }

    /// Update velocity (second moment) with numerical stability
    /// 数値安定性を考慮したベロシティ（第二モーメント）更新  
    pub fn update_velocity(velocity: &mut Tensor<f32>, grad: &Tensor<f32>, beta2: f32) -> RusTorchResult<()> {
        if !(0.0..1.0).contains(&beta2) {
            return Err(RusTorchError::InvalidParameters {
                operation: "velocity update".to_string(),
                message: format!("Beta2 must be in [0, 1), got {}", beta2),
            });
        }

        let beta2_term = &*velocity * beta2;
        let grad_squared = grad * grad;
        let grad_term = &grad_squared * (1.0 - beta2);
        let updated = &beta2_term + &grad_term;
        
        // Check for numerical stability
        if Self::has_invalid_values(&updated) {
            return Err(RusTorchError::InvalidParameters {
                operation: "velocity update".to_string(),
                message: "Numerical instability detected in velocity update".to_string(),
            });
        }
        
        velocity.copy_from(&updated);
        Ok(())
    }

    /// Compute bias correction factor
    /// バイアス補正係数を計算
    pub fn bias_correction(beta: f32, step: usize) -> f32 {
        if step == 0 || beta == 0.0 {
            1.0
        } else {
            1.0 - beta.powi(step as i32)
        }
    }

    /// Apply bias correction to tensor
    /// テンソルにバイアス補正を適用
    pub fn apply_bias_correction(tensor: &Tensor<f32>, correction: f32) -> Tensor<f32> {
        if correction.abs() < 1e-12 {
            // Avoid division by very small numbers
            tensor.clone()
        } else {
            tensor / correction
        }
    }

    /// Compute Adam-style parameter update
    /// Adam形式のパラメータ更新を計算
    pub fn compute_adam_update(momentum: &Tensor<f32>, velocity: &Tensor<f32>, eps: f32) -> Tensor<f32> {
        let denominator = &Self::stable_sqrt(velocity, eps);
        momentum / denominator
    }

    /// Compute element-wise maximum between two tensors (for Adamax)
    /// 二つのテンソル間の要素毎最大値を計算（Adamax用）
    pub fn tensor_max(a: &Tensor<f32>, b: &Tensor<f32>) -> RusTorchResult<Tensor<f32>> {
        let a_data = a.as_slice().ok_or_else(|| RusTorchError::InvalidParameters {
            operation: "tensor_max".to_string(),
            message: "Failed to access tensor A data".to_string(),
        })?;
        let b_data = b.as_slice().ok_or_else(|| RusTorchError::InvalidParameters {
            operation: "tensor_max".to_string(),
            message: "Failed to access tensor B data".to_string(),
        })?;
        
        if a_data.len() != b_data.len() || a.shape() != b.shape() {
            return Err(RusTorchError::InvalidParameters {
                operation: "tensor_max".to_string(),
                message: "Tensor shapes must match".to_string(),
            });
        }
        
        let max_data: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&a_val, &b_val)| a_val.max(b_val))
            .collect();
            
        Ok(Tensor::from_vec(max_data, a.shape().to_vec()))
    }

    /// Compute element-wise absolute value of tensor
    /// テンソルの要素毎絶対値を計算
    pub fn tensor_abs(tensor: &Tensor<f32>) -> RusTorchResult<Tensor<f32>> {
        let data = tensor.as_slice().ok_or_else(|| RusTorchError::InvalidParameters {
            operation: "tensor_abs".to_string(),
            message: "Failed to access tensor data".to_string(),
        })?;
        
        let abs_data: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        Ok(Tensor::from_vec(abs_data, tensor.shape().to_vec()))
    }

    /// Advanced exponential moving average with momentum scheduling
    /// モーメンタムスケジューリング付き高度な指数移動平均
    pub fn advanced_ema_update(
        current: &mut Tensor<f32>,
        new_value: &Tensor<f32>,
        base_decay: f32,
        step: usize,
        warmup_steps: usize,
    ) -> RusTorchResult<()> {
        // Apply warmup scheduling
        let effective_decay = if step < warmup_steps {
            let warmup_factor = (step as f32) / (warmup_steps as f32);
            base_decay * warmup_factor
        } else {
            base_decay
        };

        let decay_term = &*current * effective_decay;
        let new_term = new_value * (1.0 - effective_decay);
        let updated = &decay_term + &new_term;
        
        if Self::has_invalid_values(&updated) {
            return Err(RusTorchError::InvalidParameters {
                operation: "advanced_ema_update".to_string(),
                message: "Numerical instability detected".to_string(),
            });
        }
        
        current.copy_from(&updated);
        Ok(())
    }

    /// Compute learning rate with cosine annealing
    /// コサインアニーリングによる学習率計算
    pub fn cosine_annealing_lr(
        base_lr: f32,
        current_step: usize,
        total_steps: usize,
        min_lr: f32,
    ) -> f32 {
        if total_steps == 0 {
            return base_lr;
        }
        
        let progress = (current_step.min(total_steps) as f32) / (total_steps as f32);
        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        min_lr + (base_lr - min_lr) * cosine_factor
    }

    /// Compute adaptive learning rate scaling based on gradient statistics
    /// 勾配統計に基づく適応学習率スケーリング計算
    pub fn adaptive_lr_scaling(
        grad_norm: f32,
        velocity_norm: f32,
        trust_ratio: f32,
    ) -> f32 {
        if velocity_norm < 1e-12 {
            trust_ratio
        } else {
            trust_ratio * (grad_norm / velocity_norm).min(1.0)
        }
    }
}

/// Memory-efficient state management for optimizers
/// オプティマイザのメモリ効率的な状態管理
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Parameter states indexed by parameter ID
    states: HashMap<usize, ParameterState>,
    /// Global step counter
    global_step: usize,
    /// Memory usage threshold for cleanup (in MB)
    memory_threshold_mb: usize,
}

/// Individual parameter state for memory-efficient storage
/// メモリ効率的な保存のための個別パラメータ状態
#[derive(Debug, Clone)]
pub struct ParameterState {
    /// First moment (momentum)
    pub momentum: Option<Tensor<f32>>,
    /// Second moment (velocity)
    pub velocity: Option<Tensor<f32>>,
    /// Additional state for specialized optimizers
    pub extra_state: HashMap<String, Tensor<f32>>,
    /// Last update step
    pub last_step: usize,
}

impl OptimizerState {
    /// Create new optimizer state with memory management
    pub fn new(memory_threshold_mb: usize) -> Self {
        Self {
            states: HashMap::new(),
            global_step: 0,
            memory_threshold_mb,
        }
    }

    /// Get or create parameter state
    pub fn get_or_create_state(&mut self, param_id: usize, param_shape: &[usize]) -> &mut ParameterState {
        self.states.entry(param_id).or_insert_with(|| ParameterState {
            momentum: None,
            velocity: None,
            extra_state: HashMap::new(),
            last_step: 0,
        })
    }

    /// Initialize momentum for parameter
    pub fn init_momentum(&mut self, param_id: usize, param_shape: &[usize]) {
        if let Some(state) = self.states.get_mut(&param_id) {
            if state.momentum.is_none() {
                state.momentum = Some(Tensor::zeros(param_shape));
            }
        }
    }

    /// Initialize velocity for parameter
    pub fn init_velocity(&mut self, param_id: usize, param_shape: &[usize]) {
        if let Some(state) = self.states.get_mut(&param_id) {
            if state.velocity.is_none() {
                state.velocity = Some(Tensor::zeros(param_shape));
            }
        }
    }

    /// Clean up states for parameters not used recently
    pub fn cleanup_stale_states(&mut self, steps_threshold: usize) {
        let current_step = self.global_step;
        self.states.retain(|_, state| {
            current_step - state.last_step < steps_threshold
        });
    }

    /// Increment global step counter
    pub fn step(&mut self) {
        self.global_step += 1;
        
        // Periodic cleanup every 1000 steps
        if self.global_step % 1000 == 0 {
            self.cleanup_stale_states(5000);
        }
    }

    /// Get current global step
    pub fn get_step(&self) -> usize {
        self.global_step
    }

    /// Estimate memory usage (rough approximation)
    pub fn estimate_memory_mb(&self) -> usize {
        let mut total_elements = 0;
        for state in self.states.values() {
            if let Some(ref momentum) = state.momentum {
                total_elements += momentum.as_slice().unwrap().len();
            }
            if let Some(ref velocity) = state.velocity {
                total_elements += velocity.as_slice().unwrap().len();
            }
            for tensor in state.extra_state.values() {
                total_elements += tensor.as_slice().unwrap().len();
            }
        }
        // Rough estimate: 4 bytes per f32 element
        (total_elements * 4) / (1024 * 1024)
    }
}

/// Numerical stability configuration for optimizers
/// オプティマイザの数値安定性設定
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Minimum epsilon for numerical stability
    pub min_eps: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Maximum parameter value to prevent overflow
    pub max_param_value: f32,
    /// Enable automatic NaN detection and correction
    pub auto_nan_correction: bool,
    /// Enable gradient clipping
    pub gradient_clipping: bool,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            min_eps: 1e-8,
            max_grad_norm: 10.0,
            max_param_value: 1e6,
            auto_nan_correction: true,
            gradient_clipping: true,
        }
    }
}

impl StabilityConfig {
    /// Apply stability measures to gradient
    pub fn stabilize_gradient(&self, grad: &Tensor<f32>) -> Tensor<f32> {
        let mut stabilized_grad = grad.clone();
        
        // Check for invalid values
        if self.auto_nan_correction && OptimizerUtils::has_invalid_values(&stabilized_grad) {
            stabilized_grad = OptimizerUtils::sanitize_tensor(&stabilized_grad, 0.0, self.max_grad_norm);
        }
        
        // Apply gradient clipping
        if self.gradient_clipping {
            stabilized_grad = OptimizerUtils::clip_gradient_norm(&stabilized_grad, self.max_grad_norm);
        }
        
        stabilized_grad
    }

    /// Apply stability measures to parameter update
    pub fn stabilize_parameter(&self, param: &Tensor<f32>) -> Tensor<f32> {
        let mut stabilized_param = param.clone();
        
        // Check for invalid values
        if self.auto_nan_correction && OptimizerUtils::has_invalid_values(&stabilized_param) {
            stabilized_param = OptimizerUtils::sanitize_tensor(&stabilized_param, 0.0, self.max_param_value);
        }
        
        // Clamp parameter values
        stabilized_param = OptimizerUtils::clamp_for_stability(
            &stabilized_param, 
            -self.max_param_value, 
            self.max_param_value
        );
        
        stabilized_param
    }
}

/// Advanced optimizer metadata and statistics tracking
/// 高度なオプティマイザメタデータと統計追跡
#[derive(Debug, Clone)]
pub struct OptimizerMetrics {
    /// Gradient norm history for convergence analysis
    gradient_norms: Vec<f32>,
    /// Parameter change norms for step size analysis
    param_change_norms: Vec<f32>,
    /// Learning rate history
    learning_rates: Vec<f32>,
    /// Step timings for performance analysis
    step_times: Vec<f32>,
    /// Maximum history length
    max_history: usize,
    /// Current step count
    step_count: usize,
}

impl OptimizerMetrics {
    /// Create new optimizer metrics tracker
    pub fn new(max_history: usize) -> Self {
        Self {
            gradient_norms: Vec::with_capacity(max_history),
            param_change_norms: Vec::with_capacity(max_history),
            learning_rates: Vec::with_capacity(max_history),
            step_times: Vec::with_capacity(max_history),
            max_history,
            step_count: 0,
        }
    }

    /// Record step metrics
    pub fn record_step(
        &mut self,
        grad_norm: f32,
        param_change_norm: f32,
        learning_rate: f32,
        step_time: f32,
    ) {
        self.gradient_norms.push(grad_norm);
        self.param_change_norms.push(param_change_norm);
        self.learning_rates.push(learning_rate);
        self.step_times.push(step_time);
        
        // Maintain maximum history size
        if self.gradient_norms.len() > self.max_history {
            self.gradient_norms.remove(0);
            self.param_change_norms.remove(0);
            self.learning_rates.remove(0);
            self.step_times.remove(0);
        }
        
        self.step_count += 1;
    }

    /// Get gradient norm statistics
    pub fn gradient_stats(&self) -> (f32, f32, f32) {
        if self.gradient_norms.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let mean = self.gradient_norms.iter().sum::<f32>() / self.gradient_norms.len() as f32;
        let min = *self.gradient_norms.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = *self.gradient_norms.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        (mean, min, max)
    }

    /// Check for convergence based on recent gradient norms
    pub fn check_convergence(&self, threshold: f32, window_size: usize) -> bool {
        if self.gradient_norms.len() < window_size {
            return false;
        }
        
        let recent_norms = &self.gradient_norms[self.gradient_norms.len().saturating_sub(window_size)..];
        let avg_norm = recent_norms.iter().sum::<f32>() / recent_norms.len() as f32;
        
        avg_norm < threshold
    }

    /// Detect optimization issues (gradient explosion, vanishing, etc.)
    pub fn detect_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        if let Some(&latest_grad_norm) = self.gradient_norms.last() {
            if latest_grad_norm > 100.0 {
                issues.push("Gradient explosion detected".to_string());
            }
            if latest_grad_norm < 1e-8 {
                issues.push("Vanishing gradients detected".to_string());
            }
        }
        
        if self.gradient_norms.len() > 10 {
            let recent = &self.gradient_norms[self.gradient_norms.len()-10..];
            let variance = {
                let mean = recent.iter().sum::<f32>() / recent.len() as f32;
                recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32
            };
            
            if variance < 1e-12 {
                issues.push("Optimization stagnation detected".to_string());
            }
        }
        
        issues
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        self.gradient_norms.clear();
        self.param_change_norms.clear();
        self.learning_rates.clear();
        self.step_times.clear();
        self.step_count = 0;
    }
}

/// Unified optimizer factory for creating optimizers with consistent configuration
/// 一貫した設定でオプティマイザを作成するための統合オプティマイザファクトリ
pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create optimizer configuration with validation
    pub fn validate_config(
        learning_rate: f32,
        weight_decay: f32,
        eps: f32,
    ) -> RusTorchResult<()> {
        if learning_rate <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "optimizer creation".to_string(),
                message: "Learning rate must be positive".to_string(),
            });
        }
        
        if weight_decay < 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "optimizer creation".to_string(),
                message: "Weight decay must be non-negative".to_string(),
            });
        }
        
        if eps <= 0.0 {
            return Err(RusTorchError::InvalidParameters {
                operation: "optimizer creation".to_string(),
                message: "Epsilon must be positive".to_string(),
            });
        }
        
        Ok(())
    }

    /// Suggest optimal parameters based on problem characteristics
    pub fn suggest_parameters(
        problem_type: &str,
        model_size: usize,
    ) -> (f32, f32, f32) { // (learning_rate, weight_decay, eps)
        match problem_type {
            "vision" => {
                if model_size > 50_000_000 { // Large model
                    (1e-4, 1e-4, 1e-8)
                } else {
                    (1e-3, 1e-4, 1e-8)
                }
            }
            "nlp" => {
                if model_size > 100_000_000 { // Large language model
                    (5e-5, 1e-2, 1e-6)
                } else {
                    (2e-4, 1e-3, 1e-8)
                }
            }
            "reinforcement_learning" => (3e-4, 0.0, 1e-5),
            "fine_tuning" => (5e-5, 1e-5, 1e-8),
            _ => (1e-3, 1e-4, 1e-8), // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_for_stability() {
        let tensor = Tensor::from_vec(vec![-10.0, 0.0, 5.0, 15.0], vec![4]);
        let clamped = OptimizerUtils::clamp_for_stability(&tensor, -5.0, 10.0);
        let data = clamped.as_slice().unwrap();
        assert_eq!(data, &[-5.0, 0.0, 5.0, 10.0]);
    }

    #[test]
    fn test_clip_gradient_norm() {
        let grad = Tensor::from_vec(vec![3.0, 4.0], vec![2]); // Norm = 5.0
        let clipped = OptimizerUtils::clip_gradient_norm(&grad, 2.0);
        let clipped_data = clipped.as_slice().unwrap();
        
        // Should be scaled down by factor of 2.0/5.0 = 0.4
        assert!((clipped_data[0] - 1.2).abs() < 1e-5);
        assert!((clipped_data[1] - 1.6).abs() < 1e-5);
    }

    #[test]
    fn test_stable_sqrt() {
        let tensor = Tensor::from_vec(vec![0.0, 4.0, 9.0], vec![3]);
        let sqrt_tensor = OptimizerUtils::stable_sqrt(&tensor, 1e-8);
        let data = sqrt_tensor.as_slice().unwrap();
        
        assert!((data[0] - (1e-8_f32).sqrt()).abs() < 1e-10);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let norm = OptimizerUtils::l2_norm(&tensor);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l1_norm() {
        let tensor = Tensor::from_vec(vec![-3.0, 4.0], vec![2]);
        let norm = OptimizerUtils::l1_norm(&tensor);
        assert!((norm - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_sanitize_tensor() {
        let tensor = Tensor::from_vec(vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY], vec![4]);
        let sanitized = OptimizerUtils::sanitize_tensor(&tensor, 0.0, 100.0);
        let data = sanitized.as_slice().unwrap();
        
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);   // NaN -> 0.0
        assert_eq!(data[2], 100.0);  // +inf -> 100.0
        assert_eq!(data[3], -100.0); // -inf -> -100.0
    }

    #[test]
    fn test_optimizer_state_management() {
        let mut state = OptimizerState::new(100);
        let param_id = 12345;
        let param_shape = vec![2, 3];

        // First create the state
        state.get_or_create_state(param_id, &param_shape);
        
        // Initialize state
        state.init_momentum(param_id, &param_shape);
        state.init_velocity(param_id, &param_shape);

        // Check state exists
        let param_state = state.get_or_create_state(param_id, &param_shape);
        assert!(param_state.momentum.is_some());
        assert!(param_state.velocity.is_some());

        // Test step increment
        assert_eq!(state.get_step(), 0);
        state.step();
        assert_eq!(state.get_step(), 1);
    }

    #[test]
    fn test_stability_config() {
        let config = StabilityConfig::default();
        let grad = Tensor::from_vec(vec![100.0, -100.0], vec![2]); // Large gradient
        
        let stabilized = config.stabilize_gradient(&grad);
        let norm = OptimizerUtils::l2_norm(&stabilized);
        
        // Should be clipped to max_grad_norm
        assert!(norm <= config.max_grad_norm + 1e-5);
    }

    #[test]
    fn test_enhanced_momentum_update() {
        let mut momentum = Tensor::zeros(&[2]);
        let grad = Tensor::from_vec(vec![0.1, 0.2], vec![2]);
        
        let result = OptimizerUtils::update_momentum(&mut momentum, &grad, 0.9);
        assert!(result.is_ok());
        
        let momentum_data = momentum.as_slice().unwrap();
        assert!((momentum_data[0] - 0.01).abs() < 1e-6);
        assert!((momentum_data[1] - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_enhanced_velocity_update() {
        let mut velocity = Tensor::zeros(&[2]);
        let grad = Tensor::from_vec(vec![0.1, 0.2], vec![2]);
        
        let result = OptimizerUtils::update_velocity(&mut velocity, &grad, 0.999);
        assert!(result.is_ok());
        
        let velocity_data = velocity.as_slice().unwrap();
        // 0.1^2 * (1-0.999) = 0.01 * 0.001 = 0.00001
        // 0.2^2 * (1-0.999) = 0.04 * 0.001 = 0.00004  
        assert!((velocity_data[0] - 0.00001).abs() < 1e-6);
        assert!((velocity_data[1] - 0.00004).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_max() {
        let a = Tensor::from_vec(vec![1.0, 5.0, 2.0], vec![3]);
        let b = Tensor::from_vec(vec![3.0, 1.0, 4.0], vec![3]);
        
        let max_tensor = OptimizerUtils::tensor_max(&a, &b).unwrap();
        let max_data = max_tensor.as_slice().unwrap();
        
        assert_eq!(max_data, &[3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_tensor_abs() {
        let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        let abs_tensor = OptimizerUtils::tensor_abs(&tensor).unwrap();
        let abs_data = abs_tensor.as_slice().unwrap();
        
        assert_eq!(abs_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_bias_correction() {
        assert!((OptimizerUtils::bias_correction(0.9, 1) - 0.1).abs() < 1e-6);
        assert_eq!(OptimizerUtils::bias_correction(0.9, 0), 1.0);
        
        let correction = OptimizerUtils::bias_correction(0.99, 100);
        assert!(correction > 0.6);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let base_lr = 1e-3;
        let min_lr = 1e-5;
        
        // At start
        let lr_start = OptimizerUtils::cosine_annealing_lr(base_lr, 0, 1000, min_lr);
        assert!((lr_start - base_lr).abs() < 1e-8);
        
        // At middle
        let lr_middle = OptimizerUtils::cosine_annealing_lr(base_lr, 500, 1000, min_lr);
        assert!(lr_middle < base_lr && lr_middle > min_lr);
        
        // At end
        let lr_end = OptimizerUtils::cosine_annealing_lr(base_lr, 1000, 1000, min_lr);
        assert!((lr_end - min_lr).abs() < 1e-8);
    }

    #[test]
    fn test_optimizer_metrics() {
        let mut metrics = OptimizerMetrics::new(100);
        
        // Record some steps
        metrics.record_step(1.0, 0.1, 1e-3, 0.01);
        metrics.record_step(0.5, 0.05, 1e-3, 0.012);
        metrics.record_step(0.1, 0.01, 1e-3, 0.009);
        
        assert_eq!(metrics.step_count(), 3);
        
        let (mean, min, max) = metrics.gradient_stats();
        assert!((mean - (1.0 + 0.5 + 0.1) / 3.0).abs() < 1e-6);
        assert_eq!(min, 0.1);
        assert_eq!(max, 1.0);
        
        // Test convergence detection
        assert!(metrics.check_convergence(2.0, 3));
        assert!(!metrics.check_convergence(0.05, 3));
    }

    #[test]
    fn test_optimizer_factory_validation() {
        assert!(OptimizerFactory::validate_config(1e-3, 1e-4, 1e-8).is_ok());
        assert!(OptimizerFactory::validate_config(-1e-3, 1e-4, 1e-8).is_err());
        assert!(OptimizerFactory::validate_config(1e-3, -1e-4, 1e-8).is_err());
        assert!(OptimizerFactory::validate_config(1e-3, 1e-4, -1e-8).is_err());
    }

    #[test]
    fn test_parameter_suggestions() {
        let (lr, wd, eps) = OptimizerFactory::suggest_parameters("vision", 10_000_000);
        assert!(lr > 0.0 && wd >= 0.0 && eps > 0.0);
        
        let (lr_large, _, _) = OptimizerFactory::suggest_parameters("vision", 100_000_000);
        assert!(lr_large < lr); // Large models should use smaller learning rates
        
        let (lr_nlp, _, _) = OptimizerFactory::suggest_parameters("nlp", 500_000_000);
        assert!(lr_nlp > 0.0);
    }
}