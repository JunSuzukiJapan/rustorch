//! Gradient scaler for mixed precision training
//! 混合精度学習のための勾配スケーラー

use crate::tensor::Tensor;
use crate::autograd::Variable;
use crate::optim::Optimizer;
use std::collections::HashMap;

/// Result of a gradient scaling step
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Step completed successfully
    Success {
        /// Scale factor used
        scale: f32,
        /// Gradient norm before clipping (if clipping was applied)
        grad_norm: Option<f32>,
    },
    /// Step skipped due to gradient overflow
    Overflow {
        /// Current scale factor
        scale: f32,
        /// New scale factor after backoff
        new_scale: f32,
    },
    /// Step skipped due to inf/nan in gradients
    InfNan {
        /// Current scale factor
        scale: f32,
    },
}

/// State of the gradient scaler
#[derive(Clone, Debug)]
pub struct ScalerState {
    /// Current scale factor
    pub scale: f32,
    /// Growth factor for scale
    pub growth_factor: f32,
    /// Backoff factor for scale
    pub backoff_factor: f32,
    /// Number of iterations between scale updates
    pub growth_interval: usize,
    /// Number of iterations since last scale update
    pub growth_tracker: usize,
    /// Number of consecutive non-overflowed iterations
    pub consecutive_non_overflow: usize,
    /// Whether dynamic scaling is enabled
    pub enabled: bool,
}

impl Default for ScalerState {
    fn default() -> Self {
        Self {
            scale: 65536.0,  // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
            consecutive_non_overflow: 0,
            enabled: true,
        }
    }
}

/// Gradient scaler for automatic mixed precision training
pub struct GradScaler {
    state: ScalerState,
    /// Track if gradients overflowed in current iteration
    found_overflow: bool,
    /// Cache for scaled gradients
    scaled_grads: HashMap<usize, Tensor<f32>>,
}

impl GradScaler {
    /// Create a new gradient scaler
    pub fn new(
        init_scale: Option<f32>,
        growth_factor: Option<f32>,
        backoff_factor: Option<f32>,
        growth_interval: Option<usize>,
        enabled: Option<bool>,
    ) -> Self {
        let mut state = ScalerState::default();
        
        if let Some(scale) = init_scale {
            state.scale = scale;
        }
        if let Some(factor) = growth_factor {
            state.growth_factor = factor;
        }
        if let Some(factor) = backoff_factor {
            state.backoff_factor = factor;
        }
        if let Some(interval) = growth_interval {
            state.growth_interval = interval;
        }
        if let Some(enabled) = enabled {
            state.enabled = enabled;
        }
        
        Self {
            state,
            found_overflow: false,
            scaled_grads: HashMap::new(),
        }
    }
    
    /// Create a default scaler
    pub fn default() -> Self {
        Self::new(None, None, None, None, None)
    }
    
    /// Scale the loss
    pub fn scale(&self, loss: &Variable<f32>) -> Variable<f32> {
        if !self.state.enabled {
            return loss.clone();
        }
        
        // Scale the loss by multiplying with scale factor
        let scale_tensor = Tensor::from_vec(vec![self.state.scale], vec![1]);
        let scale_var = Variable::new(scale_tensor, false);
        
        // loss * scale
        loss * &scale_var
    }
    
    /// Scale tensor directly
    pub fn scale_tensor(&self, tensor: &Tensor<f32>) -> Tensor<f32> {
        if !self.state.enabled {
            return tensor.clone();
        }
        
        tensor * self.state.scale
    }
    
    /// Unscale gradients
    pub fn unscale_grads(&mut self, _optimizer: &mut dyn Optimizer) {
        if !self.state.enabled {
            return;
        }
        
        // In a real implementation, we would iterate through optimizer's parameters
        // and unscale their gradients by dividing by the scale factor
        
        // For now, mark that unscaling has been done
        // This would be called before optimizer.step()
    }
    
    /// Check for gradient overflow/underflow
    pub fn check_overflow(&mut self, gradients: &[Tensor<f32>]) -> bool {
        if !self.state.enabled {
            return false;
        }
        
        for grad in gradients {
            if let Some(slice) = grad.as_slice() {
                for &value in slice {
                    if !value.is_finite() || value.abs() > 65504.0 {  // FP16 max
                        self.found_overflow = true;
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Perform optimizer step with scaling
    pub fn step<O: Optimizer>(&mut self, optimizer: &mut O, params: &[Tensor<f32>], grads: &[Tensor<f32>]) {
        if !self.state.enabled {
            // Normal step without scaling
            for (param, grad) in params.iter().zip(grads.iter()) {
                optimizer.step(param, grad);
            }
            return;
        }
        
        // Check for overflow
        if self.check_overflow(grads) {
            // Skip this step due to overflow
            self.found_overflow = true;
            return;
        }
        
        // Unscale gradients
        let unscaled_grads: Vec<Tensor<f32>> = grads.iter()
            .map(|g| g / self.state.scale)
            .collect();
        
        // Perform optimizer step with unscaled gradients
        for (param, grad) in params.iter().zip(unscaled_grads.iter()) {
            optimizer.step(param, grad);
        }
        
        // Update scale
        self.update_scale();
    }
    
    /// Advanced step with gradient clipping and inf/nan detection
    pub fn step_with_clipping<O: Optimizer>(
        &mut self, 
        optimizer: &mut O, 
        params: &[Tensor<f32>], 
        grads: &mut [Tensor<f32>],
        max_grad_norm: Option<f32>
    ) -> StepResult {
        if !self.state.enabled {
            // Normal step without scaling
            if let Some(max_norm) = max_grad_norm {
                crate::amp::dtype_utils::utils::clip_grad_norm(grads, max_norm);
            }
            for (param, grad) in params.iter().zip(grads.iter()) {
                optimizer.step(param, grad);
            }
            return StepResult::Success { scale: 1.0, grad_norm: None };
        }
        
        // Check for overflow before unscaling
        if self.check_overflow(grads) {
            self.found_overflow = true;
            return StepResult::Overflow { 
                scale: self.state.scale,
                new_scale: self.state.scale * self.state.backoff_factor 
            };
        }
        
        // Unscale gradients
        for grad in grads.iter_mut() {
            *grad = grad.clone() / self.state.scale;
        }
        
        // Clip gradients if requested
        let grad_norm = if let Some(max_norm) = max_grad_norm {
            Some(crate::amp::dtype_utils::utils::clip_grad_norm(grads, max_norm))
        } else {
            None
        };
        
        // Final inf/nan check after unscaling and clipping
        if self.check_overflow(grads) {
            self.found_overflow = true;
            return StepResult::InfNan { scale: self.state.scale };
        }
        
        // Perform optimizer step
        for (param, grad) in params.iter().zip(grads.iter()) {
            optimizer.step(param, grad);
        }
        
        // Update scale
        let old_scale = self.state.scale;
        self.update_scale();
        
        StepResult::Success { 
            scale: old_scale,
            grad_norm 
        }
    }
    
    /// Update the scale factor
    pub fn update_scale(&mut self) {
        if !self.state.enabled {
            return;
        }
        
        self.state.growth_tracker += 1;
        
        if self.found_overflow {
            // Backoff the scale
            self.state.scale *= self.state.backoff_factor;
            self.state.consecutive_non_overflow = 0;
            self.found_overflow = false;
        } else {
            self.state.consecutive_non_overflow += 1;
            
            // Check if we should grow the scale
            if self.state.growth_tracker >= self.state.growth_interval {
                self.state.scale *= self.state.growth_factor;
                self.state.growth_tracker = 0;
            }
        }
        
        // Clamp scale to reasonable bounds
        self.state.scale = self.state.scale.max(1.0).min(65536.0 * 65536.0);
    }
    
    /// Get current scale
    pub fn get_scale(&self) -> f32 {
        self.state.scale
    }
    
    /// Set scale manually
    pub fn set_scale(&mut self, scale: f32) {
        self.state.scale = scale;
    }
    
    /// Load scaler state
    pub fn load_state_dict(&mut self, state: ScalerState) {
        self.state = state;
    }
    
    /// Get scaler state
    pub fn state_dict(&self) -> ScalerState {
        self.state.clone()
    }
    
    /// Check if scaling is enabled
    pub fn is_enabled(&self) -> bool {
        self.state.enabled
    }
    
    /// Enable or disable scaling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.state.enabled = enabled;
    }
    
    /// Reset scaler state (useful after loading checkpoints)
    pub fn reset(&mut self) {
        self.found_overflow = false;
        self.scaled_grads.clear();
        self.state.growth_tracker = 0;
        self.state.consecutive_non_overflow = 0;
    }
    
    /// Get detailed statistics about scaling
    pub fn get_stats(&self) -> ScalerStats {
        ScalerStats {
            current_scale: self.state.scale,
            growth_factor: self.state.growth_factor,
            backoff_factor: self.state.backoff_factor,
            growth_interval: self.state.growth_interval,
            growth_tracker: self.state.growth_tracker,
            consecutive_non_overflow: self.state.consecutive_non_overflow,
            enabled: self.state.enabled,
            has_overflow: self.found_overflow,
        }
    }
    
    /// Set scale bounds
    pub fn set_scale_bounds(&mut self, min_scale: f32, max_scale: f32) {
        self.state.scale = self.state.scale.max(min_scale).min(max_scale);
    }
    
    /// Gradually adjust growth interval based on overflow frequency
    pub fn adaptive_growth_interval(&mut self, overflow_rate: f32) {
        if overflow_rate > 0.1 {
            // Increase interval if overflow rate is high
            self.state.growth_interval = (self.state.growth_interval * 2).min(10000);
        } else if overflow_rate < 0.01 {
            // Decrease interval if overflow rate is very low
            self.state.growth_interval = (self.state.growth_interval / 2).max(100);
        }
    }
}

/// Statistics about gradient scaling
/// 勾配スケーリングの統計情報
#[derive(Debug, Clone)]
pub struct ScalerStats {
    /// Current scale factor
    /// 現在のスケール係数
    pub current_scale: f32,
    /// Growth factor for scale updates
    /// スケール更新の成長係数
    pub growth_factor: f32,
    /// Backoff factor when overflow occurs
    /// オーバーフロー時の後退係数
    pub backoff_factor: f32,
    /// Interval between growth updates
    /// 成長更新の間隔
    pub growth_interval: usize,
    /// Current growth tracking counter
    /// 現在の成長追跡カウンタ
    pub growth_tracker: usize,
    /// Number of consecutive non-overflow steps
    /// 連続非オーバーフローステップ数
    pub consecutive_non_overflow: usize,
    /// Whether scaling is enabled
    /// スケーリングが有効かどうか
    pub enabled: bool,
    /// Whether overflow was detected
    /// オーバーフローが検出されたかどうか
    pub has_overflow: bool,
}

/// Utility functions for gradient scaling
pub mod utils {
    
    
    
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grad_scaler_creation() {
        let scaler = GradScaler::default();
        assert_eq!(scaler.get_scale(), 65536.0);
        assert!(scaler.is_enabled());
    }
    
    #[test]
    fn test_grad_scaler_custom() {
        let scaler = GradScaler::new(
            Some(1024.0),
            Some(3.0),
            Some(0.3),
            Some(1000),
            Some(true),
        );
        assert_eq!(scaler.get_scale(), 1024.0);
        assert_eq!(scaler.state.growth_factor, 3.0);
        assert_eq!(scaler.state.backoff_factor, 0.3);
    }
    
    #[test]
    fn test_scale_tensor() {
        let scaler = GradScaler::default();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let scaled = scaler.scale_tensor(&tensor);
        
        let expected = tensor * scaler.get_scale();
        assert_eq!(scaled.as_slice(), expected.as_slice());
    }
    
    #[test]
    fn test_overflow_detection() {
        let mut scaler = GradScaler::default();
        
        // Normal gradients
        let normal_grads = vec![
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]),
        ];
        assert!(!scaler.check_overflow(&normal_grads));
        
        // Overflow gradients
        let overflow_grads = vec![
            Tensor::from_vec(vec![1.0, 2.0, 100000.0], vec![3]),
        ];
        assert!(scaler.check_overflow(&overflow_grads));
        
        // NaN gradients
        let nan_grads = vec![
            Tensor::from_vec(vec![1.0, f32::NAN, 3.0], vec![3]),
        ];
        assert!(scaler.check_overflow(&nan_grads));
    }
    
    #[test]
    fn test_scale_update() {
        let mut scaler = GradScaler::new(
            Some(1024.0),
            Some(2.0),
            Some(0.5),
            Some(2),
            Some(true),
        );
        
        let initial_scale = scaler.get_scale();
        
        // Simulate overflow
        scaler.found_overflow = true;
        scaler.update_scale();
        assert_eq!(scaler.get_scale(), initial_scale * 0.5);
        
        // Simulate successful iterations
        scaler.found_overflow = false;
        scaler.update_scale();
        scaler.update_scale();
        // Should grow after growth_interval
        assert_eq!(scaler.get_scale(), initial_scale * 0.5 * 2.0);
    }
    
}