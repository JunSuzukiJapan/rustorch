//! Autocast context manager for automatic mixed precision
//! 自動混合精度のためのAutocastコンテキストマネージャー

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::autograd::Variable;
use std::cell::RefCell;

thread_local! {
    static AUTOCAST_STATE: RefCell<AutocastState> = RefCell::new(AutocastState::default());
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutocastMode {
    /// No autocasting
    None,
    /// Use FP16 for reduced precision
    FP16,
    /// Use BF16 for reduced precision
    BF16,
}

#[derive(Clone, Debug)]
struct AutocastState {
    mode: AutocastMode,
    enabled: bool,
    cache_enabled: bool,
    /// Nested level for autocast contexts
    level: usize,
}

impl Default for AutocastState {
    fn default() -> Self {
        Self {
            mode: AutocastMode::None,
            enabled: false,
            cache_enabled: true,
            level: 0,
        }
    }
}

/// Autocast context manager
pub struct AutocastContext {
    prev_state: AutocastState,
    device_type: String,
}

impl AutocastContext {
    /// Create a new autocast context
    pub fn new(device_type: &str, enabled: bool, dtype: Option<DType>) -> Self {
        let mode = if enabled {
            match dtype {
                Some(DType::Float16) => AutocastMode::FP16,
                Some(DType::BFloat16) => AutocastMode::BF16,
                _ => AutocastMode::FP16,  // Default to FP16
            }
        } else {
            AutocastMode::None
        };
        
        let prev_state = AUTOCAST_STATE.with(|state| {
            let mut s = state.borrow_mut();
            let prev = s.clone();
            s.mode = mode;
            s.enabled = enabled;
            s.level += 1;
            prev
        });
        
        Self {
            prev_state,
            device_type: device_type.to_string(),
        }
    }
    
    /// Enter the autocast context
    pub fn enter(&self) {
        // Context is already entered in new()
    }
    
    /// Exit the autocast context
    pub fn exit(&self) {
        AUTOCAST_STATE.with(|state| {
            let mut s = state.borrow_mut();
            s.mode = self.prev_state.mode;
            s.enabled = self.prev_state.enabled;
            s.level = self.prev_state.level;
        });
    }
}

impl Drop for AutocastContext {
    fn drop(&mut self) {
        self.exit();
    }
}

/// Create an autocast context
pub fn autocast(device_type: &str, enabled: bool, dtype: Option<DType>) -> AutocastContext {
    AutocastContext::new(device_type, enabled, dtype)
}

/// Check if autocast is currently enabled
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_STATE.with(|state| state.borrow().enabled)
}

/// Get current autocast mode
pub fn get_autocast_mode() -> AutocastMode {
    AUTOCAST_STATE.with(|state| state.borrow().mode)
}

/// Get current autocast dtype
pub fn get_autocast_dtype() -> DType {
    match get_autocast_mode() {
        AutocastMode::FP16 => DType::Float16,
        AutocastMode::BF16 => DType::BFloat16,
        AutocastMode::None => DType::Float32,
    }
}

/// Cast tensor to autocast dtype if enabled
pub fn maybe_autocast_f32(tensor: &Tensor<f32>) -> Tensor<f32> {
    use crate::amp::dtype_utils::{cast_to_fp16, cast_to_bf16};
    
    if !is_autocast_enabled() {
        return tensor.clone();
    }
    
    match get_autocast_mode() {
        AutocastMode::FP16 => {
            // Cast to FP16 for reduced precision (simulated)
            cast_to_fp16(tensor)
        },
        AutocastMode::BF16 => {
            // Cast to BF16 for reduced precision (simulated)
            cast_to_bf16(tensor)
        },
        AutocastMode::None => tensor.clone(),
    }
}

/// Cast tensor to autocast dtype if enabled (generic version - limited support)
pub fn maybe_autocast<T: num_traits::Float>(tensor: &Tensor<T>) -> Tensor<T> {
    if !is_autocast_enabled() {
        return tensor.clone();
    }
    
    // For non-f32 types, we can't do actual casting yet
    // This would require more complex type system
    tensor.clone()
}

/// Cast variable to autocast dtype if enabled
pub fn maybe_autocast_variable<T: num_traits::Float + Send + Sync + 'static>(var: Variable<T>) -> Variable<T> {
    if !is_autocast_enabled() {
        return var;
    }
    
    // TODO: Implement actual dtype conversion for variables
    var
}

/// Operations that should be performed in float32
pub fn is_fp32_op(op_name: &str) -> bool {
    matches!(
        op_name,
        "softmax" | "log_softmax" | "cross_entropy" | 
        "nll_loss" | "mse_loss" | "l1_loss" |
        "batch_norm" | "layer_norm" | "group_norm"
    )
}

/// Decorator for autocast-aware operations
pub struct AutocastOp;

impl AutocastOp {
    /// Execute operation with autocast
    pub fn apply<F, T>(op_name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !is_autocast_enabled() || is_fp32_op(op_name) {
            // Run in original precision
            f()
        } else {
            // Would cast inputs to reduced precision here
            // For now, just run the operation
            f()
        }
    }
}

/// Macro for autocast-aware operations
#[macro_export]
macro_rules! autocast_op {
    ($op_name:expr, $body:expr) => {
        $crate::amp::autocast::AutocastOp::apply($op_name, || $body)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_autocast_context() {
        assert!(!is_autocast_enabled());
        
        {
            let _ctx = autocast("cuda", true, Some(DType::Float16));
            assert!(is_autocast_enabled());
            assert_eq!(get_autocast_mode(), AutocastMode::FP16);
            assert_eq!(get_autocast_dtype(), DType::Float16);
        }
        
        assert!(!is_autocast_enabled());
    }
    
    #[test]
    fn test_nested_autocast() {
        assert!(!is_autocast_enabled());
        
        {
            let _ctx1 = autocast("cuda", true, Some(DType::Float16));
            assert_eq!(get_autocast_mode(), AutocastMode::FP16);
            
            {
                let _ctx2 = autocast("cuda", true, Some(DType::BFloat16));
                assert_eq!(get_autocast_mode(), AutocastMode::BF16);
            }
            
            assert_eq!(get_autocast_mode(), AutocastMode::FP16);
        }
        
        assert!(!is_autocast_enabled());
    }
    
    #[test]
    fn test_fp32_ops() {
        assert!(is_fp32_op("softmax"));
        assert!(is_fp32_op("batch_norm"));
        assert!(is_fp32_op("cross_entropy"));
        
        assert!(!is_fp32_op("matmul"));
        assert!(!is_fp32_op("conv2d"));
        assert!(!is_fp32_op("linear"));
    }
}