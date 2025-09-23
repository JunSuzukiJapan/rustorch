//! Activation functions implementation
//! 活性化関数実装

use pyo3::prelude::*;
use rustorch::nn::activation::{ReLU as RustReLU, sigmoid, tanh};
use rustorch::autograd::Variable as RustVariable;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{nn_error};

/// Python ReLU activation wrapper
/// Python ReLU活性化ラッパー
#[pyclass(name = "ReLU")]
pub struct PyReLU {
    pub inner: RustReLU<f32>,
}

impl PyReLU {
    /// Create new PyReLU from RustReLU
    /// RustReLUからPyReLUを作成
    pub fn new(relu: RustReLU<f32>) -> Self {
        Self { inner: relu }
    }
}

#[pymethods]
impl PyReLU {
    /// Create a new ReLU activation
    /// 新しいReLU活性化を作成
    #[new]
    fn py_new() -> PyResult<Self> {
        let relu = RustReLU::new();
        Ok(PyReLU::new(relu))
    }

    /// Forward pass through ReLU
    /// ReLUのフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes activation callable)
    /// 呼び出しメソッド（活性化を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        "ReLU()".to_string()
    }
}

/// Python Sigmoid activation wrapper
/// Python Sigmoid活性化ラッパー
#[pyclass(name = "Sigmoid")]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    /// Create a new Sigmoid activation
    /// 新しいSigmoid活性化を作成
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(PySigmoid)
    }

    /// Forward pass through Sigmoid
    /// Sigmoidのフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = sigmoid(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes activation callable)
    /// 呼び出しメソッド（活性化を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        "Sigmoid()".to_string()
    }
}

/// Python Tanh activation wrapper
/// Python Tanh活性化ラッパー
#[pyclass(name = "Tanh")]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    /// Create a new Tanh activation
    /// 新しいTanh活性化を作成
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(PyTanh)
    }

    /// Forward pass through Tanh
    /// Tanhのフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = tanh(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes activation callable)
    /// 呼び出しメソッド（活性化を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        "Tanh()".to_string()
    }
}