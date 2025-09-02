//! Optimization utilities for memory efficiency and numerical stability
//! メモリ効率と数値安定性のための最適化ユーティリティ

use crate::tensor::Tensor;
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
}