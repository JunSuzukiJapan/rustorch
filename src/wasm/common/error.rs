//! Centralized error handling for WASM operations
//! WASM操作用集中エラーハンドリング

use wasm_bindgen::prelude::*;

/// Standardized error builder for WASM operations
pub struct WasmError;

impl WasmError {
    /// Shape mismatch between tensors
    pub fn shape_mismatch() -> JsValue {
        JsValue::from_str("Shape mismatch between tensors")
    }
    
    /// Tensor dimension requirement not met
    pub fn dimension_error(required: usize, got: usize) -> JsValue {
        JsValue::from_str(&format!("Requires {}D tensor, got {}D", required, got))
    }
    
    /// Size mismatch error
    pub fn size_mismatch(expected: usize, got: usize) -> JsValue {
        JsValue::from_str(&format!("Size mismatch: expected {}, got {}", expected, got))
    }
    
    /// Empty tensor error
    pub fn empty_tensor() -> JsValue {
        JsValue::from_str("Operation requires non-empty tensor")
    }
    
    /// Invalid parameter range
    pub fn invalid_range<T: std::fmt::Display>(param: &str, value: T, min: T, max: T) -> JsValue {
        JsValue::from_str(&format!("{} must be between {} and {}, got {}", param, min, max, value))
    }
    
    /// Invalid parameter value
    pub fn invalid_param<T: std::fmt::Display>(param: &str, value: T, reason: &str) -> JsValue {
        JsValue::from_str(&format!("Invalid {}: {} ({})", param, value, reason))
    }
    
    /// Insufficient data for operation
    pub fn insufficient_data(operation: &str, required: usize, got: usize) -> JsValue {
        JsValue::from_str(&format!("{} requires at least {} elements, got {}", operation, required, got))
    }
    
    /// Index out of bounds
    pub fn index_out_of_bounds(index: usize, len: usize) -> JsValue {
        JsValue::from_str(&format!("Index {} out of bounds for length {}", index, len))
    }
    
    /// Mathematical domain error
    pub fn math_domain_error(function: &str, input: f32) -> JsValue {
        JsValue::from_str(&format!("{} domain error: input {} is invalid", function, input))
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, JsValue>;

/// Macro for quick error generation
#[macro_export]
macro_rules! wasm_error {
    (shape_mismatch) => {
        WasmError::shape_mismatch()
    };
    (empty_tensor) => {
        WasmError::empty_tensor()
    };
    (dimension $required:expr, $got:expr) => {
        WasmError::dimension_error($required, $got)
    };
    (insufficient_data $op:expr, $required:expr, $got:expr) => {
        WasmError::insufficient_data($op, $required, $got)
    };
}