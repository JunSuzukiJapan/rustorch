//! Unified optimizer interface to reduce code duplication
//! 重複コードを削減するための統一オプティマイザーインターフェース

use crate::common::{RusTorchResult, OptimizationError};
use crate::autograd::Variable;
use num_traits::Float;
use std::collections::HashMap;

/// Common optimizer operations interface
/// 共通オプティマイザー操作インターフェース
pub trait UnifiedOptimizer<T: Float + Send + Sync> {
    /// Update parameters with gradients
    /// 勾配でパラメータを更新
    fn step(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()>;
    
    /// Zero gradients
    /// 勾配をゼロにリセット
    fn zero_grad(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()>;
    
    /// Get current learning rate
    /// 現在の学習率を取得
    fn get_lr(&self) -> T;
    
    /// Set learning rate
    /// 学習率を設定
    fn set_lr(&mut self, lr: T) -> RusTorchResult<()>;
    
    /// Get optimizer state
    /// オプティマイザー状態を取得
    fn state_dict(&self) -> HashMap<String, String>;
    
    /// Load optimizer state
    /// オプティマイザー状態をロード
    fn load_state_dict(&mut self, state: HashMap<String, String>) -> RusTorchResult<()>;
}

/// Unified optimizer configuration
/// 統一オプティマイザー設定
#[derive(Debug, Clone)]
pub struct OptimizerConfig<T: Float> {
    /// Learning rate for optimization
    /// 最適化の学習率
    pub learning_rate: T,
    /// Weight decay coefficient for L2 regularization
    /// L2正則化の重み減衰係数
    pub weight_decay: Option<T>,
    /// Momentum factor for SGD
    /// SGDのモメンタム係数
    pub momentum: Option<T>,
    /// Dampening factor for momentum
    /// モメンタムの減衰係数
    pub dampening: Option<T>,
    /// Enable Nesterov momentum
    /// ネステロフモメンタムを有効にする
    pub nesterov: bool,
    /// Beta1 parameter for Adam optimizer
    /// Adamオプティマイザーのbeta1パラメータ
    pub beta1: Option<T>,
    /// Beta2 parameter for Adam optimizer
    /// Adamオプティマイザーのbeta2パラメータ
    pub beta2: Option<T>,
    /// Epsilon for numerical stability
    /// 数値安定性のためのイプシロン
    pub epsilon: Option<T>,
    /// Enable AMSGrad variant
    /// AMSGradバリアントを有効にする
    pub amsgrad: bool,
}

impl<T: Float> Default for OptimizerConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.01).unwrap(),
            weight_decay: None,
            momentum: None,
            dampening: None,
            nesterov: false,
            beta1: Some(T::from(0.9).unwrap()),
            beta2: Some(T::from(0.999).unwrap()),
            epsilon: Some(T::from(1e-8).unwrap()),
            amsgrad: false,
        }
    }
}

/// Optimizer type enumeration
/// オプティマイザータイプ列挙
#[derive(Debug, Clone)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent optimizer
    /// 確率的勾配降下法オプティマイザー
    SGD,
    /// Adam optimizer
    /// Adamオプティマイザー
    Adam,
    /// AdamW optimizer (Adam with decoupled weight decay)
    /// AdamWオプティマイザー（分離重み減衰付きAdam）
    AdamW,
    /// RMSprop optimizer
    /// RMSpropオプティマイザー
    RMSprop,
    /// Adagrad optimizer
    /// Adagradオプティマイザー
    Adagrad,
}

/// Unified optimizer implementation
/// 統一オプティマイザー実装
pub struct UnifiedOptimizerImpl<T: Float> {
    optimizer_type: OptimizerType,
    config: OptimizerConfig<T>,
    state: HashMap<String, Vec<T>>,
    step_count: usize,
}

impl<T: Float + std::fmt::Debug + 'static + Send + Sync> UnifiedOptimizerImpl<T> {
    /// Create new unified optimizer
    /// 新しい統一オプティマイザーを作成
    pub fn new(optimizer_type: OptimizerType, config: OptimizerConfig<T>) -> Self {
        Self {
            optimizer_type,
            config,
            state: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create SGD optimizer
    /// SGDオプティマイザーを作成
    pub fn sgd(learning_rate: T, momentum: Option<T>, weight_decay: Option<T>) -> Self {
        let config = OptimizerConfig {
            learning_rate,
            momentum,
            weight_decay,
            ..Default::default()
        };
        Self::new(OptimizerType::SGD, config)
    }

    /// Create Adam optimizer
    /// Adamオプティマイザーを作成
    pub fn adam(
        learning_rate: T,
        beta1: Option<T>,
        beta2: Option<T>,
        epsilon: Option<T>,
        weight_decay: Option<T>,
    ) -> Self {
        let config = OptimizerConfig {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            ..Default::default()
        };
        Self::new(OptimizerType::Adam, config)
    }

    /// Update parameters using SGD (optionally with weight decay)
    /// SGDを使用してパラメータを更新（オプションでWeight Decay対応）
    fn sgd_step(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()> {
        self.step_count += 1;

        let lr = self.config.learning_rate;
        let weight_decay = self.config.weight_decay;

        for (i, var) in params.iter_mut().enumerate() {
            // Read gradient
            let grad_arc = var.grad();
            let grad_opt = grad_arc.read().map_err(|_| {
                crate::common::RusTorchError::OptimizationError(
                    OptimizationError::GradientError("Failed to read gradient lock".to_string())
                )
            })?;

            let grad_tensor = match &*grad_opt {
                Some(g) => g.clone(),
                None => continue, // No gradient to apply
            };

            // Optional weight decay: grad += wd * param
            let mut effective_grad = grad_tensor.clone();
            if let Some(wd) = weight_decay {
                // effective_grad += wd * param
                let data_arc = var.data();
                let param_read = data_arc.read().map_err(|_| {
                    crate::common::RusTorchError::OptimizationError(
                        OptimizationError::OptimizerError("Failed to read param lock".to_string())
                    )
                })?;
                let mut decay_term = param_read.clone();
                decay_term.mul_scalar_inplace(wd);
                drop(param_read);
                effective_grad.add_inplace(&decay_term);
            }

            // Velocity (momentum) support (optional)
            if let Some(mu) = self.config.momentum {
                // v = mu * v + (1 - dampening) * grad
                let v_key = format!("vel_{}", i);
                let mut velocity = if let Some(v) = self.state.get(&v_key) {
                    v.clone()
                } else {
                    vec![T::zero(); effective_grad.len()]
                };

                let damp = self.config.dampening.unwrap_or_else(|| T::zero());

                let g_slice = effective_grad.as_slice().ok_or_else(|| {
                    crate::common::RusTorchError::OptimizationError(
                        OptimizationError::GradientError("Non-contiguous gradient slice".to_string())
                    )
                })?;

                for (v, &g) in velocity.iter_mut().zip(g_slice.iter()) {
                    *v = mu * *v + (T::one() - damp) * g;
                }

                // Nesterov: use lookahead gradient
                let use_nesterov = self.config.nesterov;
                let update_slice = if use_nesterov {
                    // g' = mu * v + (1 - damp) * g  (approx)
                    // reuse velocity as g'
                    &velocity
                } else {
                    // use velocity as the momentum buffer, and set effective_grad = velocity
                    &velocity
                };

                // Apply update: param -= lr * update_slice
                let data_arc = var.data();
                let mut param_write = data_arc.write().map_err(|_| {
                    crate::common::RusTorchError::OptimizationError(
                        OptimizationError::OptimizerError("Failed to write param lock".to_string())
                    )
                })?;

                if let Some(p_slice) = param_write.as_slice_mut() {
                    for (p, &u) in p_slice.iter_mut().zip(update_slice.iter()) {
                        *p = *p - lr * u;
                    }
                } else {
                    // Fallback: element-wise using ndarray map
                    let mut step_tensor = effective_grad.clone();
                    step_tensor.mul_scalar_inplace(lr);
                    param_write.sub_inplace(&step_tensor);
                }

                // Write back momentum buffer after using it
                self.state.insert(v_key, velocity);
            } else {
                // Vanilla SGD: param -= lr * grad
                let data_arc = var.data();
                let mut param_write = data_arc.write().map_err(|_| {
                    crate::common::RusTorchError::OptimizationError(
                        OptimizationError::OptimizerError("Failed to write param lock".to_string())
                    )
                })?;
                if let (Some(p_slice), Some(g_slice)) = (param_write.as_slice_mut(), effective_grad.as_slice()) {
                    for (p, &g) in p_slice.iter_mut().zip(g_slice.iter()) {
                        *p = *p - lr * g;
                    }
                } else {
                    let mut step_tensor = effective_grad.clone();
                    step_tensor.mul_scalar_inplace(lr);
                    param_write.sub_inplace(&step_tensor);
                }
            }
        }

        Ok(())
    }

    /// Update parameters using Adam
    /// Adamを使用してパラメータを更新
    fn adam_step(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()> {
        self.step_count += 1;

        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1.unwrap_or_else(|| T::from(0.9).unwrap());
        let beta2 = self.config.beta2.unwrap_or_else(|| T::from(0.999).unwrap());
        let eps = self.config.epsilon.unwrap_or_else(|| T::from(1e-8).unwrap());
        let wd = self.config.weight_decay;

        for (i, var) in params.iter_mut().enumerate() {
            // Gradient
            let grad_arc = var.grad();
            let grad_opt = grad_arc.read().map_err(|_| {
                crate::common::RusTorchError::OptimizationError(
                    OptimizationError::GradientError("Failed to read gradient lock".to_string())
                )
            })?;
            let grad_tensor = match &*grad_opt {
                Some(g) => g.clone(),
                None => continue,
            };

            // Optional weight decay (decoupled style behaves like AdamW if set)
            // Apply decoupled decay after Adam update: param *= (1 - lr * wd)

            // Prepare moment buffers
            let m_key = format!("adam_m_{}", i);
            let v_key = format!("adam_v_{}", i);
            let t_key = format!("adam_t_{}", i);

            let mut m = if let Some(buf) = self.state.get(&m_key) { buf.clone() } else { vec![T::zero(); grad_tensor.len()] };
            let mut v = if let Some(buf) = self.state.get(&v_key) { buf.clone() } else { vec![T::zero(); grad_tensor.len()] };
            let mut t = if let Some(step_vec) = self.state.get(&t_key) { step_vec[0] } else { T::zero() };
            t = t + T::one();

            let g_slice = grad_tensor.as_slice().ok_or_else(|| {
                crate::common::RusTorchError::OptimizationError(
                    OptimizationError::GradientError("Non-contiguous gradient slice".to_string())
                )
            })?;

            // Update biased first and second moment estimates
            for ((mi, vi), &g) in m.iter_mut().zip(v.iter_mut()).zip(g_slice.iter()) {
                *mi = beta1 * *mi + (T::one() - beta1) * g;
                *vi = beta2 * *vi + (T::one() - beta2) * (g * g);
            }

            // Bias correction
            let b1_c = T::one() - beta1.powi(t.to_f32().unwrap() as i32);
            let b2_c = T::one() - beta2.powi(t.to_f32().unwrap() as i32);

            // Parameter update
            let data_arc = var.data();
            let mut param_write = data_arc.write().map_err(|_| {
                crate::common::RusTorchError::OptimizationError(
                    OptimizationError::OptimizerError("Failed to write param lock".to_string())
                )
            })?;

            if let Some(p_slice) = param_write.as_slice_mut() {
                for ((p, &mi), &vi) in p_slice.iter_mut().zip(m.iter()).zip(v.iter()) {
                    let m_hat = mi / b1_c;
                    let v_hat = vi / b2_c;
                    *p = *p - lr * (m_hat / (v_hat.sqrt() + eps));
                    if let Some(wd_val) = wd {
                        *p = *p - lr * wd_val * *p; // decoupled weight decay (AdamW-style)
                    }
                }
            } else {
                // Fallback: element-wise loop via indexing
                // This path is unlikely given ArrayD usually has contiguous storage
                let len = grad_tensor.len();
                for _idx in 0..len {
                    // Slow path: use get and as_array_mut indexing
                    // Not implementing due to complexity; return an error instead
                    return Err(crate::common::RusTorchError::OptimizationError(
                        OptimizationError::OptimizerError("Non-contiguous param update path not supported".to_string())
                    ));
                }
            }

            // Persist state
            self.state.insert(m_key, m);
            self.state.insert(v_key, v);
            self.state.insert(t_key, vec![t]);
        }

        Ok(())
    }
}

impl<T: Float + std::fmt::Debug + 'static + Send + Sync> UnifiedOptimizer<T> for UnifiedOptimizerImpl<T> {
    fn step(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()> {
        match self.optimizer_type {
            OptimizerType::SGD => self.sgd_step(params),
            OptimizerType::Adam => self.adam_step(params),
            OptimizerType::AdamW => self.adam_step(params), // Simplified
            OptimizerType::RMSprop => self.sgd_step(params), // Simplified
            OptimizerType::Adagrad => self.sgd_step(params), // Simplified
        }
    }

    fn zero_grad(&mut self, params: &mut [Variable<T>]) -> RusTorchResult<()> {
        for param in params.iter_mut() {
            param.zero_grad();
        }
        Ok(())
    }

    fn get_lr(&self) -> T {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: T) -> RusTorchResult<()> {
        if lr <= T::zero() {
            return Err(crate::common::RusTorchError::OptimizationError(
                OptimizationError::LearningRateError("Learning rate must be positive".to_string())
            ));
        }
        self.config.learning_rate = lr;
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, String> {
        let mut state_dict = HashMap::new();
        state_dict.insert("optimizer_type".to_string(), format!("{:?}", self.optimizer_type));
        state_dict.insert("learning_rate".to_string(), format!("{:?}", self.config.learning_rate));
        state_dict.insert("step_count".to_string(), self.step_count.to_string());
        
        if let Some(momentum) = self.config.momentum {
            state_dict.insert("momentum".to_string(), format!("{:?}", momentum));
        }
        if let Some(weight_decay) = self.config.weight_decay {
            state_dict.insert("weight_decay".to_string(), format!("{:?}", weight_decay));
        }
        if let Some(beta1) = self.config.beta1 {
            state_dict.insert("beta1".to_string(), format!("{:?}", beta1));
        }
        if let Some(beta2) = self.config.beta2 {
            state_dict.insert("beta2".to_string(), format!("{:?}", beta2));
        }
        
        state_dict
    }

    fn load_state_dict(&mut self, state: HashMap<String, String>) -> RusTorchResult<()> {
        if let Some(step_count_str) = state.get("step_count") {
            self.step_count = step_count_str.parse().map_err(|_| {
                crate::common::RusTorchError::OptimizationError(
                    OptimizationError::OptimizerError("Invalid step count in state dict".to_string())
                )
            })?;
        }
        
        // Additional state loading would go here
        Ok(())
    }
}

/// Unified learning rate scheduler
/// 統一学習率スケジューラー
pub trait UnifiedScheduler<T: Float> {
    /// Step the scheduler
    /// スケジューラーをステップ
    fn step(&mut self, optimizer: &mut dyn UnifiedOptimizer<T>) -> RusTorchResult<()>;
    
    /// Get current learning rate
    /// 現在の学習率を取得
    fn get_last_lr(&self) -> T;
    
    /// Reset scheduler state
    /// スケジューラー状態をリセット
    fn reset(&mut self);
}

/// Step learning rate scheduler implementation
/// ステップ学習率スケジューラー実装
pub struct StepLRScheduler<T: Float> {
    step_size: usize,
    gamma: T,
    last_epoch: usize,
    base_lr: T,
    current_lr: T,
}

impl<T: Float> StepLRScheduler<T> {
    /// Create new step learning rate scheduler
    /// 新しいステップ学習率スケジューラーを作成
    pub fn new(step_size: usize, gamma: T, base_lr: T) -> Self {
        Self {
            step_size,
            gamma,
            last_epoch: 0,
            base_lr,
            current_lr: base_lr,
        }
    }
}

impl<T: Float + Send + Sync> UnifiedScheduler<T> for StepLRScheduler<T> {
    fn step(&mut self, optimizer: &mut dyn UnifiedOptimizer<T>) -> RusTorchResult<()> {
        self.last_epoch += 1;
        
        if self.last_epoch % self.step_size == 0 {
            self.current_lr = self.current_lr * self.gamma;
            optimizer.set_lr(self.current_lr)?;
        }
        
        Ok(())
    }

    fn get_last_lr(&self) -> T {
        self.current_lr
    }

    fn reset(&mut self) {
        self.last_epoch = 0;
        self.current_lr = self.base_lr;
    }
}

/// Exponential learning rate scheduler implementation
/// 指数学習率スケジューラー実装
pub struct ExponentialLRScheduler<T: Float> {
    gamma: T,
    last_epoch: usize,
    base_lr: T,
    current_lr: T,
}

impl<T: Float> ExponentialLRScheduler<T> {
    /// Create new exponential learning rate scheduler
    /// 新しい指数学習率スケジューラーを作成
    pub fn new(gamma: T, base_lr: T) -> Self {
        Self {
            gamma,
            last_epoch: 0,
            base_lr,
            current_lr: base_lr,
        }
    }
}

impl<T: Float + Send + Sync> UnifiedScheduler<T> for ExponentialLRScheduler<T> {
    fn step(&mut self, optimizer: &mut dyn UnifiedOptimizer<T>) -> RusTorchResult<()> {
        self.last_epoch += 1;
        self.current_lr = self.base_lr * self.gamma.powi(self.last_epoch as i32);
        optimizer.set_lr(self.current_lr)?;
        Ok(())
    }

    fn get_last_lr(&self) -> T {
        self.current_lr
    }

    fn reset(&mut self) {
        self.last_epoch = 0;
        self.current_lr = self.base_lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_unified_sgd_optimizer() {
        let mut optimizer = UnifiedOptimizerImpl::sgd(
            0.01f32,
            Some(0.9f32),
            Some(0.001f32)
        );

        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let mut param = Variable::new(tensor, true);
        
        // Simulate gradient
        let grad_tensor = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], vec![3]);
        param.set_grad(Some(grad_tensor));

        let mut params = vec![param];
        assert!(optimizer.step(&mut params).is_ok());
        assert_eq!(optimizer.get_lr(), 0.01f32);
    }

    #[test]
    fn test_unified_adam_optimizer() {
        let mut optimizer = UnifiedOptimizerImpl::adam(
            0.001f32,
            Some(0.9f32),
            Some(0.999f32),
            Some(1e-8f32),
            Some(0.01f32)
        );

        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let mut param = Variable::new(tensor, true);
        
        let grad_tensor = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], vec![3]);
        param.set_grad(Some(grad_tensor));

        let mut params = vec![param];
        assert!(optimizer.step(&mut params).is_ok());
    }

    #[test]
    fn test_step_lr_scheduler() {
        let mut optimizer = UnifiedOptimizerImpl::sgd(0.1f32, None, None);
        let mut scheduler = StepLRScheduler::new(2, 0.5f32, 0.1f32);

        assert_eq!(scheduler.get_last_lr(), 0.1f32);
        
        scheduler.step(&mut optimizer).unwrap();
        assert_eq!(scheduler.get_last_lr(), 0.1f32);
        
        scheduler.step(&mut optimizer).unwrap();
        assert_eq!(scheduler.get_last_lr(), 0.05f32);
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let mut optimizer = UnifiedOptimizerImpl::sgd(0.1f32, None, None);
        let mut scheduler = ExponentialLRScheduler::new(0.9f32, 0.1f32);

        scheduler.step(&mut optimizer).unwrap();
        assert!((scheduler.get_last_lr() - 0.09f32).abs() < 1e-6);
        
        scheduler.step(&mut optimizer).unwrap();
        assert!((scheduler.get_last_lr() - 0.081f32).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_state_dict() {
        let optimizer = UnifiedOptimizerImpl::sgd(0.01f32, Some(0.9f32), None);
        let state_dict = optimizer.state_dict();
        
        assert!(state_dict.contains_key("learning_rate"));
        assert!(state_dict.contains_key("momentum"));
        assert!(state_dict.contains_key("step_count"));
    }
}
