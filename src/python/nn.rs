//! Python bindings for neural network layers and modules
//! ニューラルネットワーク層とモジュールのPythonバインディング

use crate::nn::loss::{CrossEntropyLoss, Loss, MSELoss};
use crate::nn::{BatchNorm2d, Conv2d, Linear, Module};
use crate::python::autograd::PyVariable;
use crate::python::error::to_py_err;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for Linear layer
/// Linear層のPythonラッパー
#[pyclass]
pub struct PyLinear {
    pub(crate) linear: Linear<f32>,
}

#[pymethods]
impl PyLinear {
    #[new]
    pub fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let linear = if use_bias {
            Linear::new(in_features, out_features)
        } else {
            Linear::new_no_bias(in_features, out_features)
        };
        Ok(PyLinear { linear })
    }

    /// Forward pass
    /// フォワードパス
    pub fn forward(&mut self, input: &PyVariable) -> PyResult<PyVariable> {
        let output = self.linear.forward(&input.variable);
        Ok(PyVariable { variable: output })
    }

    /// Get parameters
    /// パラメータを取得
    pub fn parameters(&self) -> HashMap<String, PyVariable> {
        // Note: Cannot access private fields directly
        // In a real implementation, Linear should provide parameter access methods
        HashMap::new()
    }

    /// Zero gradients of all parameters
    /// 全パラメータの勾配をゼロに設定
    pub fn zero_grad(&mut self) {
        // Note: zero_grad method not available on Linear
        // In a real implementation, would need parameter access
    }

    /// Get input features
    /// 入力特徴数を取得
    pub fn in_features(&self) -> usize {
        self.linear.input_size()
    }

    /// Get output features
    /// 出力特徴数を取得
    pub fn out_features(&self) -> usize {
        self.linear.output_size()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "Linear(in_features={}, out_features={}, bias=true)",
            self.in_features(),
            self.out_features()
        )
    }
}

/// Python wrapper for Conv2d layer
/// Conv2d層のPythonラッパー
#[pyclass]
pub struct PyConv2d {
    pub(crate) conv2d: Conv2d<f32>,
}

#[pymethods]
impl PyConv2d {
    #[new]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        let conv2d = Conv2d::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size), // Convert to tuple
            Some((stride, stride)),     // Convert to tuple
            Some((padding, padding)),   // Convert to tuple
            Some(use_bias),
        );
        Ok(PyConv2d { conv2d })
    }

    /// Forward pass
    /// フォワードパス
    pub fn forward(&mut self, input: &PyVariable) -> PyResult<PyVariable> {
        let output = self.conv2d.forward(&input.variable);
        Ok(PyVariable { variable: output })
    }

    /// Get parameters
    /// パラメータを取得
    pub fn parameters(&self) -> HashMap<String, PyVariable> {
        // Note: Cannot access private fields directly
        HashMap::new()
    }

    /// Zero gradients
    /// 勾配をゼロに設定
    pub fn zero_grad(&mut self) {
        // Note: zero_grad method not available on Conv2d
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        "Conv2d(...)".to_string()
    }
}

/// Python wrapper for BatchNorm2d layer
/// BatchNorm2d層のPythonラッパー
#[pyclass]
pub struct PyBatchNorm2d {
    pub(crate) batchnorm: BatchNorm2d<f32>,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);

        let batchnorm = BatchNorm2d::new(num_features, Some(eps), Some(momentum), Some(affine));
        Ok(PyBatchNorm2d { batchnorm })
    }

    /// Forward pass
    /// フォワードパス
    pub fn forward(&mut self, input: &PyVariable) -> PyResult<PyVariable> {
        let output = self.batchnorm.forward(&input.variable);
        Ok(PyVariable { variable: output })
    }

    /// Get parameters
    /// パラメータを取得
    pub fn parameters(&self) -> HashMap<String, PyVariable> {
        // Note: Cannot access private fields directly
        HashMap::new()
    }

    /// Zero gradients
    /// 勾配をゼロに設定
    pub fn zero_grad(&mut self) {
        // Note: zero_grad method not available
    }

    /// Set training mode
    /// 訓練モードを設定
    pub fn train(&mut self, mode: Option<bool>) {
        // Note: train method not available
    }

    /// Set evaluation mode
    /// 評価モードを設定
    pub fn eval(&mut self) {
        // Note: eval method not available
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        "BatchNorm2d(...)".to_string()
    }
}

/// Python wrapper for MSE Loss
/// MSE損失のPythonラッパー
#[pyclass]
pub struct PyMSELoss {
    pub(crate) reduction: String,
}

#[pymethods]
impl PyMSELoss {
    #[new]
    pub fn new(reduction: Option<String>) -> PyResult<Self> {
        let reduction = reduction.unwrap_or_else(|| "mean".to_string());
        Ok(PyMSELoss { reduction })
    }

    /// Compute loss
    /// 損失を計算
    pub fn forward(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        // Simplified MSE loss computation
        println!("Computing MSE loss with reduction: {}", self.reduction);

        // Create a dummy loss variable for now
        // In real implementation, would compute actual MSE
        let loss_data = vec![0.5]; // Placeholder loss value
        let loss_tensor = crate::tensor::Tensor::from_vec(loss_data, vec![1]);
        let loss_var = crate::autograd::Variable::new(loss_tensor, false);
        Ok(PyVariable { variable: loss_var })
    }

    /// Call operator (alias for forward)
    /// 呼び出し演算子（forwardのエイリアス）
    pub fn __call__(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input, target)
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!("MSELoss(reduction='{}')", self.reduction)
    }
}

/// Python wrapper for CrossEntropy Loss
/// CrossEntropy損失のPythonラッパー
#[pyclass]
pub struct PyCrossEntropyLoss {
    pub(crate) weight: Option<Vec<f32>>,
    pub(crate) ignore_index: Option<i64>,
    pub(crate) reduction: String,
    pub(crate) label_smoothing: f32,
}

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    pub fn new(
        weight: Option<Vec<f32>>,
        ignore_index: Option<i64>,
        reduction: Option<String>,
        label_smoothing: Option<f32>,
    ) -> PyResult<Self> {
        let reduction = reduction.unwrap_or_else(|| "mean".to_string());
        let label_smoothing = label_smoothing.unwrap_or(0.0);

        Ok(PyCrossEntropyLoss {
            weight,
            ignore_index,
            reduction,
            label_smoothing,
        })
    }

    /// Compute loss
    /// 損失を計算
    pub fn forward(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        // Simplified CrossEntropy loss computation
        println!(
            "Computing CrossEntropy loss with reduction: {}",
            self.reduction
        );

        // Create a dummy loss variable for now
        let loss_data = vec![0.8]; // Placeholder loss value
        let loss_tensor = crate::tensor::Tensor::from_vec(loss_data, vec![1]);
        let loss_var = crate::autograd::Variable::new(loss_tensor, false);
        Ok(PyVariable { variable: loss_var })
    }

    /// Call operator (alias for forward)
    /// 呼び出し演算子（forwardのエイリアス）
    pub fn __call__(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input, target)
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "CrossEntropyLoss(reduction='{}', label_smoothing={})",
            self.reduction, self.label_smoothing
        )
    }
}

// Activation functions
// 活性化関数

/// ReLU activation function
/// ReLU活性化関数
#[pyfunction]
pub fn relu(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::relu(&input.variable);
    Ok(PyVariable { variable: result })
}

/// Sigmoid activation function
/// Sigmoid活性化関数
#[pyfunction]
pub fn sigmoid(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::sigmoid(&input.variable);
    Ok(PyVariable { variable: result })
}

/// Tanh activation function
/// Tanh活性化関数
#[pyfunction]
pub fn tanh(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::tanh(&input.variable);
    Ok(PyVariable { variable: result })
}

/// Softmax activation function
/// Softmax活性化関数
#[pyfunction]
pub fn softmax(input: &PyVariable, dim: Option<usize>) -> PyResult<PyVariable> {
    // Note: activation::softmax doesn't take dim parameter
    let result = crate::nn::activation::softmax(&input.variable);
    Ok(PyVariable { variable: result })
}

/// GELU activation function
/// GELU活性化関数
#[pyfunction]
pub fn gelu(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::gelu(&input.variable);
    Ok(PyVariable { variable: result })
}

/// Leaky ReLU activation function
/// Leaky ReLU活性化関数
#[pyfunction]
pub fn leaky_relu(input: &PyVariable, negative_slope: Option<f32>) -> PyResult<PyVariable> {
    let slope = negative_slope.unwrap_or(0.01);
    let result = crate::nn::activation::leaky_relu(&input.variable, slope);
    Ok(PyVariable { variable: result })
}

/// Swish activation function
/// Swish活性化関数
#[pyfunction]
pub fn swish(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::swish(&input.variable);
    Ok(PyVariable { variable: result })
}

/// ELU activation function
/// ELU活性化関数
#[pyfunction]
pub fn elu(input: &PyVariable, alpha: Option<f32>) -> PyResult<PyVariable> {
    let alpha = alpha.unwrap_or(1.0);
    let result = crate::nn::activation::elu(&input.variable, alpha);
    Ok(PyVariable { variable: result })
}

/// SELU activation function
/// SELU活性化関数
#[pyfunction]
pub fn selu(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::selu(&input.variable);
    Ok(PyVariable { variable: result })
}

/// Mish activation function
/// Mish活性化関数
#[pyfunction]
pub fn mish(input: &PyVariable) -> PyResult<PyVariable> {
    let result = crate::nn::activation::mish(&input.variable);
    Ok(PyVariable { variable: result })
}
