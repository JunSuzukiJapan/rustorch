//! Automatic Mixed Precision (AMP) training support
//! 自動混合精度学習のサポート

mod autocast;
mod grad_scaler;
mod dtype_utils;
mod optimizer_wrapper;

pub use autocast::{autocast, AutocastContext, AutocastMode, maybe_autocast_f32};
pub use grad_scaler::{GradScaler, ScalerState, StepResult, ScalerStats};
pub use dtype_utils::{
    cast_to_fp16, cast_to_fp32, cast_to_bf16, cast_bf16_to_fp32,
    MixedPrecisionTensor, cast_tensor
};
pub use optimizer_wrapper::{AMPOptimizer, ParamGroup, TrainingStats};

use crate::dtype::DType;
use std::sync::{Arc, RwLock};

/// Global AMP configuration
/// グローバルAMP設定
pub struct AMPConfig {
    /// Whether to enable autocast
    pub enabled: bool,
    /// Preferred reduced precision dtype (FP16 or BF16)
    pub dtype: DType,
    /// Initial loss scale
    pub init_scale: f32,
    /// Growth factor for loss scale
    pub growth_factor: f32,
    /// Backoff factor for loss scale
    pub backoff_factor: f32,
    /// Growth interval
    pub growth_interval: usize,
    /// Whether to use dynamic loss scaling
    pub dynamic_loss_scaling: bool,
}

impl Default for AMPConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dtype: DType::Float16,
            init_scale: 65536.0,  // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            dynamic_loss_scaling: true,
        }
    }
}

impl AMPConfig {
    /// Create config for BF16
    pub fn bf16() -> Self {
        Self {
            dtype: DType::BFloat16,
            // BF16 doesn't need loss scaling due to wider range
            init_scale: 1.0,
            dynamic_loss_scaling: false,
            ..Default::default()
        }
    }
    
    /// Create config for FP16 with static scaling
    pub fn fp16_static(scale: f32) -> Self {
        Self {
            dtype: DType::Float16,
            init_scale: scale,
            dynamic_loss_scaling: false,
            ..Default::default()
        }
    }
}

// Global AMP state
lazy_static::lazy_static! {
    static ref AMP_STATE: Arc<RwLock<AMPState>> = Arc::new(RwLock::new(AMPState::default()));
}

struct AMPState {
    enabled: bool,
    _autocast_mode: AutocastMode,
    config: AMPConfig,
}

impl Default for AMPState {
    fn default() -> Self {
        Self {
            enabled: false,
            _autocast_mode: AutocastMode::None,
            config: AMPConfig::default(),
        }
    }
}

/// Enable AMP globally
pub fn enable_amp(config: AMPConfig) {
    let mut state = AMP_STATE.write().unwrap();
    state.enabled = true;
    state.config = config;
}

/// Disable AMP globally
pub fn disable_amp() {
    let mut state = AMP_STATE.write().unwrap();
    state.enabled = false;
}

/// Check if AMP is enabled
pub fn is_amp_enabled() -> bool {
    AMP_STATE.read().unwrap().enabled
}

/// Get current AMP configuration
pub fn get_amp_config() -> AMPConfig {
    let state = AMP_STATE.read().unwrap();
    AMPConfig {
        enabled: state.config.enabled,
        dtype: state.config.dtype,
        init_scale: state.config.init_scale,
        growth_factor: state.config.growth_factor,
        backoff_factor: state.config.backoff_factor,
        growth_interval: state.config.growth_interval,
        dynamic_loss_scaling: state.config.dynamic_loss_scaling,
    }
}

/// Mixed precision training utilities
pub mod utils {
    use super::*;
    
    /// Check if operation should use reduced precision
    pub fn should_use_reduced_precision(op_name: &str) -> bool {
        // List of operations that should stay in FP32
        const FP32_OPS: &[&str] = &[
            "softmax",
            "log_softmax",
            "cross_entropy",
            "nll_loss",
            "batch_norm",
            "layer_norm",
        ];
        
        !FP32_OPS.contains(&op_name)
    }
    
    /// Get optimal dtype for current hardware
    pub fn get_optimal_dtype() -> DType {
        // Check hardware capabilities
        if has_bf16_support() {
            DType::BFloat16
        } else if has_fp16_support() {
            DType::Float16
        } else {
            DType::Float32
        }
    }
    
    /// Check if hardware supports FP16
    pub fn has_fp16_support() -> bool {
        // Check for CUDA compute capability >= 7.0 or similar
        // For now, assume support exists
        true
    }
    
    /// Check if hardware supports BF16
    pub fn has_bf16_support() -> bool {
        // Check for CUDA compute capability >= 8.0, or
        // Intel CPUs with AVX512_BF16, or
        // Apple Silicon with BF16 support
        #[cfg(target_arch = "aarch64")]
        {
            // Apple Silicon supports BF16
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Check for AVX512_BF16 on x86_64
            false  // Conservative default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_amp_config_default() {
        let config = AMPConfig::default();
        assert_eq!(config.dtype, DType::Float16);
        assert_eq!(config.init_scale, 65536.0);
        assert!(config.dynamic_loss_scaling);
    }
    
    #[test]
    fn test_amp_config_bf16() {
        let config = AMPConfig::bf16();
        assert_eq!(config.dtype, DType::BFloat16);
        assert_eq!(config.init_scale, 1.0);
        assert!(!config.dynamic_loss_scaling);
    }
    
    #[test]
    fn test_amp_state() {
        disable_amp();
        assert!(!is_amp_enabled());
        
        enable_amp(AMPConfig::default());
        assert!(is_amp_enabled());
        
        disable_amp();
        assert!(!is_amp_enabled());
    }
    
    #[test]
    fn test_should_use_reduced_precision() {
        assert!(utils::should_use_reduced_precision("matmul"));
        assert!(utils::should_use_reduced_precision("conv2d"));
        assert!(!utils::should_use_reduced_precision("softmax"));
        assert!(!utils::should_use_reduced_precision("batch_norm"));
    }
}