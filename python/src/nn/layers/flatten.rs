//! Flatten layer implementation
//! Flatten層実装

use pyo3::prelude::*;
use rustorch::tensor::core::Tensor as RustTensor;
use rustorch::autograd::Variable as RustVariable;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_dim, tensor_op_error};

/// Python Flatten layer wrapper
/// Python Flatten層ラッパー
#[pyclass(name = "Flatten")]
pub struct PyFlatten {
    pub start_dim: usize,
    pub end_dim: isize,
}

impl PyFlatten {
    /// Create new PyFlatten
    /// PyFlattenを作成
    pub fn new(start_dim: usize, end_dim: isize) -> Self {
        Self { start_dim, end_dim }
    }
}

#[pymethods]
impl PyFlatten {
    /// Create a new Flatten layer
    /// 新しいFlatten層を作成
    #[new]
    fn py_new(start_dim: Option<usize>, end_dim: Option<isize>) -> PyResult<Self> {
        let start_dim = start_dim.unwrap_or(1); // Default: keep batch dimension
        let end_dim = end_dim.unwrap_or(-1); // Default: flatten to end

        Ok(PyFlatten::new(start_dim, end_dim))
    }

    /// Forward pass through Flatten layer
    /// Flatten層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        // Get input shape
        let input_shape = input.inner.data().read().unwrap().shape().to_vec();

        if self.start_dim >= input_shape.len() {
            return Err(invalid_dim!(self.start_dim as isize, input_shape.len()).into());
        }

        // Calculate the new shape
        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..self.start_dim {
            new_shape.push(input_shape[i]);
        }

        // Calculate flattened dimension size
        let end_dim = if self.end_dim < 0 {
            input_shape.len() as isize + self.end_dim + 1
        } else {
            self.end_dim + 1
        } as usize;

        let mut flattened_size = 1;
        for i in self.start_dim..end_dim.min(input_shape.len()) {
            flattened_size *= input_shape[i];
        }
        new_shape.push(flattened_size);

        // Add dimensions after end_dim
        for i in end_dim.min(input_shape.len())..input_shape.len() {
            new_shape.push(input_shape[i]);
        }

        // Create flattened tensor
        let input_data = input.inner.data().read().unwrap().clone();
        let flattened_data = input_data.data.into_shape_with_order(new_shape).map_err(|e| {
            tensor_op_error!("flatten", &format!("Failed to flatten tensor: {}", e))
        })?;

        let flattened_tensor = RustTensor::new(flattened_data);
        let result = RustVariable::new(flattened_tensor, input.inner.requires_grad());

        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get start dimension
    /// 開始次元を取得
    #[getter]
    fn start_dim(&self) -> usize {
        self.start_dim
    }

    /// Get end dimension
    /// 終了次元を取得
    #[getter]
    fn end_dim(&self) -> isize {
        self.end_dim
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("Flatten(start_dim={}, end_dim={})", self.start_dim, self.end_dim)
    }
}