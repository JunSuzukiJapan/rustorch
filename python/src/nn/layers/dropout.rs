//! Dropout regularization layer implementation
//! Dropout正則化層実装

use pyo3::prelude::*;
use rustorch::nn::Dropout as RustDropout;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, nn_error};

/// Python Dropout layer wrapper
/// Python Dropout層ラッパー
#[pyclass(name = "Dropout")]
pub struct PyDropout {
    pub inner: RustDropout<f32>,
}

impl PyDropout {
    /// Create new PyDropout from RustDropout
    /// RustDropoutからPyDropoutを作成
    pub fn new(dropout: RustDropout<f32>) -> Self {
        Self { inner: dropout }
    }
}

#[pymethods]
impl PyDropout {
    /// Create a new Dropout layer
    /// 新しいDropout層を作成
    #[new]
    fn py_new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        // Validate parameters
        if !(0.0..=1.0).contains(&p) {
            return Err(invalid_param!("p", p, "must be in [0, 1]").into());
        }

        let dropout = RustDropout::new(p, inplace);
        Ok(PyDropout::new(dropout))
    }

    /// Forward pass through Dropout layer
    /// Dropout層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set to training mode
    /// 訓練モードに設定
    fn train(&mut self) -> PyResult<()> {
        self.inner.train();
        Ok(())
    }

    /// Set to evaluation mode
    /// 評価モードに設定
    fn eval(&mut self) -> PyResult<()> {
        self.inner.eval();
        Ok(())
    }

    /// Get dropout probability
    /// Dropout確率を取得
    #[getter]
    fn p(&self) -> f32 {
        self.inner.p()
    }

    /// Get inplace flag
    /// inplaceフラグを取得
    #[getter]
    fn inplace(&self) -> bool {
        self.inner.inplace()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("Dropout(p={}, inplace={})", self.p(), self.inplace())
    }
}