//! AMP-aware optimizer wrapper
//! AMP対応オプティマイザラッパー

use crate::tensor::Tensor;
use crate::optim::Optimizer;
use crate::amp::{GradScaler, StepResult, ScalerStats};

/// AMP-aware optimizer wrapper that handles mixed precision training
pub struct AMPOptimizer<O: Optimizer> {
    /// Base optimizer
    optimizer: O,
    /// Gradient scaler for loss scaling
    scaler: GradScaler,
    /// Parameter groups for different scaling strategies
    param_groups: Vec<ParamGroup>,
    /// Statistics tracking
    step_count: usize,
    overflow_count: usize,
    successful_steps: usize,
}

/// Parameter group configuration for AMP
#[derive(Debug, Clone)]
pub struct ParamGroup {
    /// Parameter IDs in this group
    pub param_ids: Vec<usize>,
    /// Whether to use gradient clipping for this group
    pub clip_gradients: bool,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f32>,
    /// Whether to use mixed precision for this group
    pub use_amp: bool,
}

impl<O: Optimizer> AMPOptimizer<O> {
    /// Create a new AMP optimizer wrapper
    pub fn new(
        optimizer: O,
        scaler_config: Option<GradScaler>,
    ) -> Self {
        let scaler = scaler_config.unwrap_or_else(GradScaler::default);
        
        Self {
            optimizer,
            scaler,
            param_groups: Vec::new(),
            step_count: 0,
            overflow_count: 0,
            successful_steps: 0,
        }
    }
    
    /// Add a parameter group with specific AMP settings
    pub fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }
    
    /// Perform an optimization step with AMP
    pub fn step(&mut self, params: &[Tensor<f32>], grads: &mut [Tensor<f32>]) -> StepResult {
        self.step_count += 1;
        
        if self.param_groups.is_empty() {
            // Default behavior for all parameters
            let result = self.scaler.step_with_clipping(
                &mut self.optimizer,
                params,
                grads,
                None
            );
            
            self.update_stats(&result);
            result
        } else {
            // Process each parameter group separately
            self.step_with_groups(params, grads)
        }
    }
    
    /// Step with parameter groups
    fn step_with_groups(&mut self, params: &[Tensor<f32>], grads: &mut [Tensor<f32>]) -> StepResult {
        let mut overall_result = StepResult::Success { scale: 1.0, grad_norm: None };
        
        for group in &self.param_groups {
            if !group.use_amp {
                // Process this group without AMP
                for &param_id in &group.param_ids {
                    if param_id < params.len() && param_id < grads.len() {
                        // Apply gradient clipping if requested
                        if group.clip_gradients {
                            if let Some(max_norm) = group.max_grad_norm {
                                let mut single_grad = vec![grads[param_id].clone()];
                                crate::amp::dtype_utils::utils::clip_grad_norm(&mut single_grad, max_norm);
                                grads[param_id] = single_grad.into_iter().next().unwrap();
                            }
                        }
                        
                        self.optimizer.step(&params[param_id], &grads[param_id]);
                    }
                }
            } else {
                // Process this group with AMP
                let group_params: Vec<_> = group.param_ids.iter()
                    .filter_map(|&id| if id < params.len() { Some(params[id].clone()) } else { None })
                    .collect();
                let mut group_grads: Vec<_> = group.param_ids.iter()
                    .filter_map(|&id| if id < grads.len() { Some(grads[id].clone()) } else { None })
                    .collect();
                
                if !group_params.is_empty() && !group_grads.is_empty() {
                    let result = self.scaler.step_with_clipping(
                        &mut self.optimizer,
                        &group_params,
                        &mut group_grads,
                        group.max_grad_norm
                    );
                    
                    // Update original grads with processed ones
                    for (i, &param_id) in group.param_ids.iter().enumerate() {
                        if param_id < grads.len() && i < group_grads.len() {
                            grads[param_id] = group_grads[i].clone();
                        }
                    }
                    
                    // Aggregate results (use worst case)
                    overall_result = match (&overall_result, &result) {
                        (_, StepResult::Overflow { .. }) => result,
                        (_, StepResult::InfNan { .. }) => result,
                        (StepResult::Success { .. }, _) => result,
                        _ => overall_result,
                    };
                }
            }
        }
        
        self.update_stats(&overall_result);
        overall_result
    }
    
    /// Update internal statistics
    fn update_stats(&mut self, result: &StepResult) {
        match result {
            StepResult::Success { .. } => {
                self.successful_steps += 1;
            },
            StepResult::Overflow { .. } | StepResult::InfNan { .. } => {
                self.overflow_count += 1;
            },
        }
    }
    
    /// Get training statistics
    pub fn get_training_stats(&self) -> TrainingStats {
        let overflow_rate = if self.step_count > 0 {
            self.overflow_count as f32 / self.step_count as f32
        } else {
            0.0
        };
        
        TrainingStats {
            total_steps: self.step_count,
            successful_steps: self.successful_steps,
            overflow_count: self.overflow_count,
            overflow_rate,
            scaler_stats: self.scaler.get_stats(),
        }
    }
    
    /// Get the underlying optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }
    
    /// Get mutable reference to the underlying optimizer
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
    
    /// Get the gradient scaler
    pub fn scaler(&self) -> &GradScaler {
        &self.scaler
    }
    
    /// Get mutable reference to the gradient scaler
    pub fn scaler_mut(&mut self) -> &mut GradScaler {
        &mut self.scaler
    }
    
    /// Zero gradients (delegate to underlying optimizer)
    pub fn zero_grad(&mut self) {
        // Note: This would need to be implemented based on the actual Optimizer trait
        // For now, it's a placeholder
    }
    
    /// Update learning rate schedule and adaptive scaling
    pub fn update_schedule(&mut self) {
        let stats = self.get_training_stats();
        
        // Adapt growth interval based on overflow rate
        self.scaler.adaptive_growth_interval(stats.overflow_rate);
        
        // Could also update learning rate here if the optimizer supports it
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.scaler.reset();
        self.step_count = 0;
        self.overflow_count = 0;
        self.successful_steps = 0;
    }
}

/// Training statistics for AMP optimization
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub overflow_count: usize,
    pub overflow_rate: f32,
    pub scaler_stats: ScalerStats,
}

impl TrainingStats {
    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_steps > 0 {
            self.successful_steps as f32 / self.total_steps as f32
        } else {
            0.0
        }
    }
    
    /// Check if training is stable (low overflow rate)
    pub fn is_stable(&self) -> bool {
        self.overflow_rate < 0.05 // Less than 5% overflow rate
    }
    
    /// Get recommended actions based on statistics
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.overflow_rate > 0.1 {
            recommendations.push("Consider reducing initial loss scale".to_string());
            recommendations.push("Consider increasing growth interval".to_string());
        }
        
        if self.overflow_rate > 0.2 {
            recommendations.push("Consider using gradient clipping".to_string());
        }
        
        if self.overflow_rate < 0.01 && self.scaler_stats.current_scale < 1000.0 {
            recommendations.push("Consider increasing initial loss scale".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Training appears stable".to_string());
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::sgd::SGD;
    
    #[test]
    fn test_amp_optimizer_creation() {
        let sgd = SGD::new(0.01);
        let amp_optimizer = AMPOptimizer::new(sgd, None);
        
        assert_eq!(amp_optimizer.step_count, 0);
        assert_eq!(amp_optimizer.overflow_count, 0);
    }
    
    #[test]
    fn test_param_group() {
        let group = ParamGroup {
            param_ids: vec![0, 1, 2],
            clip_gradients: true,
            max_grad_norm: Some(1.0),
            use_amp: true,
        };
        
        assert_eq!(group.param_ids.len(), 3);
        assert!(group.clip_gradients);
        assert!(group.use_amp);
    }
    
    #[test]
    fn test_training_stats() {
        let stats = TrainingStats {
            total_steps: 100,
            successful_steps: 98,
            overflow_count: 2,
            overflow_rate: 0.02,
            scaler_stats: ScalerStats {
                current_scale: 65536.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
                growth_tracker: 500,
                consecutive_non_overflow: 10,
                enabled: true,
                has_overflow: false,
            },
        };
        
        assert_eq!(stats.success_rate(), 0.98);
        assert!(stats.is_stable());
        
        let recommendations = stats.get_recommendations();
        assert!(!recommendations.is_empty());
    }
}