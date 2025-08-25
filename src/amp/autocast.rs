//! Autocast context manager for automatic mixed precision
//! 自動混合精度のためのAutocastコンテキストマネージャー

use crate::dtype::DType;
use crate::tensor::Tensor;
use std::cell::RefCell;

thread_local! {
    static AUTOCAST_STATE: RefCell<AutocastState> = RefCell::new(AutocastState::default());
}

/// Autocast mode for mixed precision training
/// 混合精度学習のためのAutocastモード
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
    _cache_enabled: bool,
    /// Nested level for autocast contexts
    level: usize,
}

impl Default for AutocastState {
    fn default() -> Self {
        Self {
            mode: AutocastMode::None,
            enabled: false,
            _cache_enabled: true,
            level: 0,
        }
    }
}

/// Autocast context manager
pub struct AutocastContext {
    prev_state: AutocastState,
    _device_type: String,
}

impl AutocastContext {
    /// Create a new autocast context
    pub fn new(device_type: &str, enabled: bool, dtype: Option<DType>) -> Self {
        let mode = if enabled {
            match dtype {
                Some(DType::Float16) => AutocastMode::FP16,
                Some(DType::BFloat16) => AutocastMode::BF16,
                _ => AutocastMode::FP16, // Default to FP16
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
            _device_type: device_type.to_string(),
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

/// Cast tensor to autocast dtype if enabled
pub fn maybe_autocast_f32(tensor: &Tensor<f32>) -> Tensor<f32> {
    use crate::amp::dtype_utils::{cast_to_bf16, cast_to_fp16};

    if !is_autocast_enabled() {
        return tensor.clone();
    }

    match get_autocast_mode() {
        AutocastMode::FP16 => {
            // Cast to FP16 for reduced precision (simulated)
            cast_to_fp16(tensor)
        }
        AutocastMode::BF16 => {
            // Cast to BF16 for reduced precision (simulated)
            cast_to_bf16(tensor)
        }
        AutocastMode::None => tensor.clone(),
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
}
