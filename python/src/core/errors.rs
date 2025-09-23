//! Error handling for RusTorch Python bindings
//! RusTorch Pythonバインディング用エラーハンドリング

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyTypeError};

/// Common error types for RusTorch Python bindings
/// RusTorch Pythonバインディング用共通エラー型
#[derive(Debug, thiserror::Error)]
pub enum RusTorchError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid dimension: {dimension} for tensor with {ndim} dimensions")]
    InvalidDimension {
        dimension: isize,
        ndim: usize,
    },

    #[error("Invalid parameter: {parameter} = {value} (reason: {reason})")]
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },

    #[error("Tensor operation failed: {operation} - {message}")]
    TensorOperation {
        operation: String,
        message: String,
    },

    #[error("Neural network error: {layer} - {message}")]
    NeuralNetwork {
        layer: String,
        message: String,
    },

    #[error("Optimization error: {optimizer} - {message}")]
    Optimization {
        optimizer: String,
        message: String,
    },
}

impl From<RusTorchError> for PyErr {
    fn from(err: RusTorchError) -> PyErr {
        match err {
            RusTorchError::ShapeMismatch { .. } => PyValueError::new_err(err.to_string()),
            RusTorchError::InvalidDimension { .. } => PyValueError::new_err(err.to_string()),
            RusTorchError::InvalidParameter { .. } => PyValueError::new_err(err.to_string()),
            RusTorchError::TensorOperation { .. } => PyRuntimeError::new_err(err.to_string()),
            RusTorchError::NeuralNetwork { .. } => PyRuntimeError::new_err(err.to_string()),
            RusTorchError::Optimization { .. } => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Result type alias for RusTorch operations
/// RusTorch操作用Result型エイリアス
pub type RusTorchResult<T> = Result<T, RusTorchError>;

/// Helper macros for common error patterns
/// 共通エラーパターン用ヘルパーマクロ

#[macro_export]
macro_rules! shape_mismatch {
    ($expected:expr, $actual:expr) => {
        $crate::core::errors::RusTorchError::ShapeMismatch {
            expected: $expected,
            actual: $actual,
        }
    };
}

#[macro_export]
macro_rules! invalid_dim {
    ($dim:expr, $ndim:expr) => {
        $crate::core::errors::RusTorchError::InvalidDimension {
            dimension: $dim,
            ndim: $ndim,
        }
    };
}

#[macro_export]
macro_rules! invalid_param {
    ($param:expr, $value:expr, $reason:expr) => {
        $crate::core::errors::RusTorchError::InvalidParameter {
            parameter: $param.to_string(),
            value: $value.to_string(),
            reason: $reason.to_string(),
        }
    };
}

#[macro_export]
macro_rules! tensor_op_error {
    ($op:expr, $msg:expr) => {
        $crate::core::errors::RusTorchError::TensorOperation {
            operation: $op.to_string(),
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! nn_error {
    ($layer:expr, $msg:expr) => {
        $crate::core::errors::RusTorchError::NeuralNetwork {
            layer: $layer.to_string(),
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! optim_error {
    ($optimizer:expr, $msg:expr) => {
        $crate::core::errors::RusTorchError::Optimization {
            optimizer: $optimizer.to_string(),
            message: $msg.to_string(),
        }
    };
}

/// Validation helper functions
/// バリデーションヘルパー関数

pub fn validate_positive(value: f32, name: &str) -> RusTorchResult<()> {
    if value <= 0.0 {
        Err(invalid_param!(name, value, "must be positive"))
    } else {
        Ok(())
    }
}

pub fn validate_range(value: f32, min: f32, max: f32, name: &str) -> RusTorchResult<()> {
    if value < min || value > max {
        Err(invalid_param!(name, value, &format!("must be in range [{}, {}]", min, max)))
    } else {
        Ok(())
    }
}

pub fn validate_non_zero(value: usize, name: &str) -> RusTorchResult<()> {
    if value == 0 {
        Err(invalid_param!(name, value, "must be non-zero"))
    } else {
        Ok(())
    }
}

pub fn validate_dimension(dim: isize, ndim: usize) -> RusTorchResult<usize> {
    let positive_dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if positive_dim >= ndim {
        Err(invalid_dim!(dim, ndim))
    } else {
        Ok(positive_dim)
    }
}