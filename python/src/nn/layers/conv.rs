//! Convolutional layers implementation
//! 畳み込み層実装

use pyo3::prelude::*;
use rustorch::nn::Conv2d as RustConv2d;
use rustorch::nn::MaxPool2d as RustMaxPool2d;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, nn_error};

/// Python Conv2d layer wrapper
/// Python Conv2d層ラッパー
#[pyclass(name = "Conv2d")]
pub struct PyConv2d {
    pub inner: RustConv2d<f32>,
}

impl PyConv2d {
    /// Create new PyConv2d from RustConv2d
    /// RustConv2dからPyConv2dを作成
    pub fn new(conv2d: RustConv2d<f32>) -> Self {
        Self { inner: conv2d }
    }
}

#[pymethods]
impl PyConv2d {
    /// Create a new Conv2d layer
    /// 新しいConv2d層を作成
    #[new]
    fn py_new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        // Validate parameters
        if in_channels == 0 {
            return Err(invalid_param!("in_channels", in_channels, "must be positive").into());
        }
        if out_channels == 0 {
            return Err(invalid_param!("out_channels", out_channels, "must be positive").into());
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(invalid_param!("kernel_size", format!("{:?}", kernel_size), "must have positive dimensions").into());
        }

        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let bias = bias.unwrap_or(true);

        let conv2d = RustConv2d::new(in_channels, out_channels, kernel_size, stride, padding, bias);
        Ok(PyConv2d::new(conv2d))
    }

    /// Forward pass through Conv2d layer
    /// Conv2d層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
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

    /// Get input channels
    /// 入力チャンネル数を取得
    #[getter]
    fn in_channels(&self) -> usize {
        self.inner.in_channels()
    }

    /// Get output channels
    /// 出力チャンネル数を取得
    #[getter]
    fn out_channels(&self) -> usize {
        self.inner.out_channels()
    }

    /// Get kernel size
    /// カーネルサイズを取得
    #[getter]
    fn kernel_size(&self) -> (usize, usize) {
        self.inner.kernel_size()
    }

    /// Get stride
    /// ストライドを取得
    #[getter]
    fn stride(&self) -> (usize, usize) {
        self.inner.stride()
    }

    /// Get padding
    /// パディングを取得
    #[getter]
    fn padding(&self) -> (usize, usize) {
        self.inner.padding()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "Conv2d({}, {}, kernel_size={:?}, stride={:?}, padding={:?})",
            self.in_channels(),
            self.out_channels(),
            self.kernel_size(),
            self.stride(),
            self.padding()
        )
    }
}

/// Python MaxPool2d layer wrapper
/// Python MaxPool2d層ラッパー
#[pyclass(name = "MaxPool2d")]
pub struct PyMaxPool2d {
    pub inner: RustMaxPool2d<f32>,
}

impl PyMaxPool2d {
    /// Create new PyMaxPool2d from RustMaxPool2d
    /// RustMaxPool2dからPyMaxPool2dを作成
    pub fn new(maxpool2d: RustMaxPool2d<f32>) -> Self {
        Self { inner: maxpool2d }
    }
}

#[pymethods]
impl PyMaxPool2d {
    /// Create a new MaxPool2d layer
    /// 新しいMaxPool2d層を作成
    #[new]
    fn py_new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        // Validate parameters
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(invalid_param!("kernel_size", format!("{:?}", kernel_size), "must have positive dimensions").into());
        }

        let stride = stride.unwrap_or(kernel_size); // Default stride = kernel_size
        let padding = padding.unwrap_or((0, 0));

        let maxpool2d = RustMaxPool2d::new(kernel_size, stride, padding);
        Ok(PyMaxPool2d::new(maxpool2d))
    }

    /// Forward pass through MaxPool2d layer
    /// MaxPool2d層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get kernel size
    /// カーネルサイズを取得
    #[getter]
    fn kernel_size(&self) -> (usize, usize) {
        self.inner.kernel_size()
    }

    /// Get stride
    /// ストライドを取得
    #[getter]
    fn stride(&self) -> (usize, usize) {
        self.inner.stride()
    }

    /// Get padding
    /// パディングを取得
    #[getter]
    fn padding(&self) -> (usize, usize) {
        self.inner.padding()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "MaxPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
            self.kernel_size(),
            self.stride(),
            self.padding()
        )
    }
}