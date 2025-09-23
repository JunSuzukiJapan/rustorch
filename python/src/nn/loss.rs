//! Loss functions implementation
//! 損失関数実装

use pyo3::prelude::*;
use rustorch::nn::loss::{MSELoss as RustMSELoss, CrossEntropyLoss as RustCrossEntropyLoss, Loss};
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{nn_error};

/// Python MSELoss wrapper
/// Python MSELoss ラッパー
#[pyclass(name = "MSELoss")]
pub struct PyMSELoss {
    pub inner: RustMSELoss<f32>,
}

impl PyMSELoss {
    /// Create new PyMSELoss from RustMSELoss
    /// RustMSELossからPyMSELossを作成
    pub fn new(mse_loss: RustMSELoss<f32>) -> Self {
        Self { inner: mse_loss }
    }
}

#[pymethods]
impl PyMSELoss {
    /// Create a new MSELoss
    /// 新しいMSELossを作成
    #[new]
    fn py_new() -> PyResult<Self> {
        let mse_loss = RustMSELoss::new();
        Ok(PyMSELoss::new(mse_loss))
    }

    /// Forward pass of MSELoss
    /// MSELossのフォワードパス
    fn forward(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&predictions.inner, &targets.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes loss function callable)
    /// 呼び出しメソッド（損失関数を呼び出し可能にする）
    fn __call__(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        self.forward(predictions, targets)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        "MSELoss()".to_string()
    }
}

/// Python CrossEntropyLoss wrapper
/// Python CrossEntropyLoss ラッパー
#[pyclass(name = "CrossEntropyLoss")]
pub struct PyCrossEntropyLoss {
    pub inner: RustCrossEntropyLoss<f32>,
}

impl PyCrossEntropyLoss {
    /// Create new PyCrossEntropyLoss from RustCrossEntropyLoss
    /// RustCrossEntropyLossからPyCrossEntropyLossを作成
    pub fn new(cross_entropy_loss: RustCrossEntropyLoss<f32>) -> Self {
        Self { inner: cross_entropy_loss }
    }
}

#[pymethods]
impl PyCrossEntropyLoss {
    /// Create a new CrossEntropyLoss
    /// 新しいCrossEntropyLossを作成
    #[new]
    fn py_new() -> PyResult<Self> {
        let cross_entropy_loss = RustCrossEntropyLoss::new();
        Ok(PyCrossEntropyLoss::new(cross_entropy_loss))
    }

    /// Forward pass of CrossEntropyLoss
    /// CrossEntropyLossのフォワードパス
    fn forward(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&predictions.inner, &targets.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes loss function callable)
    /// 呼び出しメソッド（損失関数を呼び出し可能にする）
    fn __call__(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        self.forward(predictions, targets)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        "CrossEntropyLoss()".to_string()
    }
}