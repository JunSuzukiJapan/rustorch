//! Linear layer implementation
//! Linear層実装

use pyo3::prelude::*;
use rustorch::nn::Linear as RustLinear;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, nn_error};

/// Python Linear layer wrapper
/// Python Linear層ラッパー
#[pyclass(name = "Linear")]
pub struct PyLinear {
    pub inner: RustLinear<f32>,
}

impl PyLinear {
    /// Create new PyLinear from RustLinear
    /// RustLinearからPyLinearを作成
    pub fn new(linear: RustLinear<f32>) -> Self {
        Self { inner: linear }
    }
}

#[pymethods]
impl PyLinear {
    /// Create a new Linear layer
    /// 新しいLinear層を作成
    #[new]
    fn py_new(input_size: usize, output_size: usize, bias: Option<bool>) -> PyResult<Self> {
        // Validate parameters
        if input_size == 0 {
            return Err(invalid_param!("input_size", input_size, "must be positive").into());
        }
        if output_size == 0 {
            return Err(invalid_param!("output_size", output_size, "must be positive").into());
        }

        let bias = bias.unwrap_or(true);
        let linear = RustLinear::new(input_size, output_size, bias);
        Ok(PyLinear::new(linear))
    }

    /// Forward pass through Linear layer
    /// Linear層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        // Validate input dimensions
        let input_shape = input.inner.data().read().unwrap().shape().to_vec();
        let expected_features = self.inner.in_features();

        if input_shape.is_empty() {
            return Err(nn_error!("Linear", "Input tensor cannot be empty").into());
        }

        let actual_features = input_shape[input_shape.len() - 1];
        if actual_features != expected_features {
            return Err(nn_error!(
                "Linear",
                &format!("Input features mismatch: expected {}, got {}",
                        expected_features, actual_features)
            ).into());
        }

        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get weight parameter
    /// 重みパラメータを取得
    #[getter]
    fn weight(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable::new(params[0].clone())
    }

    /// Get bias parameter
    /// バイアスパラメータを取得
    #[getter]
    fn bias(&self) -> Option<PyVariable> {
        let params = self.inner.parameters();
        if params.len() > 1 {
            Some(PyVariable::new(params[1].clone()))
        } else {
            None
        }
    }

    /// Get input features
    /// 入力特徴数を取得
    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    /// Get output features
    /// 出力特徴数を取得
    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "Linear(input_size={}, output_size={}, bias={})",
            self.inner.in_features(),
            self.inner.out_features(),
            self.bias().is_some()
        )
    }
}