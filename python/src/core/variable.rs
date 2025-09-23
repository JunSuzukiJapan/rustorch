//! Variable implementation for RusTorch Python bindings
//! RusTorch Pythonバインディング用Variable実装

use pyo3::prelude::*;
use rustorch::autograd::Variable as RustVariable;
use rustorch::tensor::core::Tensor as RustTensor;
use crate::core::tensor::PyTensor;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{tensor_op_error};

/// Python Variable wrapper for RusTorch Variable
/// RusTorch VariableのPythonラッパー
#[pyclass(name = "Variable")]
#[derive(Clone)]
pub struct PyVariable {
    pub inner: RustVariable<f32>,
}

impl PyVariable {
    /// Create a new PyVariable from RustVariable
    /// RustVariableからPyVariableを作成
    pub fn new(variable: RustVariable<f32>) -> Self {
        Self { inner: variable }
    }

    /// Get reference to inner variable
    /// 内部Variableへの参照を取得
    pub fn as_rust_variable(&self) -> &RustVariable<f32> {
        &self.inner
    }

    /// Create PyVariable from tensor and requires_grad
    /// テンソルとrequires_gradからPyVariableを作成
    pub fn from_tensor(tensor: RustTensor<f32>, requires_grad: bool) -> Self {
        let variable = RustVariable::new(tensor, requires_grad);
        Self::new(variable)
    }
}

#[pymethods]
impl PyVariable {
    /// Create a new Variable
    /// 新しいVariableを作成
    #[new]
    fn py_new(data: PyTensor, requires_grad: Option<bool>) -> PyResult<Self> {
        let requires_grad = requires_grad.unwrap_or(false);
        let variable = RustVariable::new(data.inner, requires_grad);
        Ok(PyVariable::new(variable))
    }

    /// Get the tensor data
    /// テンソルデータを取得
    #[getter]
    fn data(&self) -> PyTensor {
        PyTensor::new(self.inner.data().read().unwrap().clone())
    }

    /// Check if variable requires gradients
    /// 勾配計算が必要かチェック
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Get the gradient if available
    /// 勾配を取得（利用可能な場合）
    #[getter]
    fn grad(&self) -> Option<PyVariable> {
        self.inner.grad().map(|g| PyVariable::new(g.clone()))
    }

    /// Perform backward pass
    /// 逆伝播を実行
    fn backward(&self) -> PyResult<()> {
        self.inner.backward().map_err(|e| {
            tensor_op_error!("backward", &format!("Backward pass failed: {:?}", e))
        })?;
        Ok(())
    }

    /// Zero the gradients
    /// 勾配をゼロにする
    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad().map_err(|e| {
            tensor_op_error!("zero_grad", &format!("Zero grad failed: {:?}", e))
        })?;
        Ok(())
    }

    /// Detach variable from computation graph
    /// 計算グラフから変数を切り離し
    fn detach(&self) -> PyVariable {
        let detached = self.inner.detach();
        PyVariable::new(detached)
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        let data_shape = self.inner.data().read().unwrap().shape().to_vec();
        format!(
            "Variable(shape={:?}, requires_grad={}, data=...)",
            data_shape,
            self.requires_grad()
        )
    }

    /// Addition operation
    /// 加算操作
    fn __add__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner + &other.inner;
        Ok(PyVariable::new(result))
    }

    /// Subtraction operation
    /// 減算操作
    fn __sub__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner - &other.inner;
        Ok(PyVariable::new(result))
    }

    /// Multiplication operation
    /// 乗算操作
    fn __mul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner * &other.inner;
        Ok(PyVariable::new(result))
    }

    /// Matrix multiplication
    /// 行列乗算
    fn matmul(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Validate shapes for matrix multiplication
        let self_shape = self.inner.data().read().unwrap().shape().to_vec();
        let other_shape = other.inner.data().read().unwrap().shape().to_vec();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(tensor_op_error!(
                "matmul",
                "Matrix multiplication requires 2D tensors"
            ).into());
        }

        if self_shape[1] != other_shape[0] {
            return Err(tensor_op_error!(
                "matmul",
                &format!("Incompatible shapes for matmul: {:?} and {:?}",
                        self_shape, other_shape)
            ).into());
        }

        let result = self.inner.matmul(&other.inner).map_err(|e| {
            tensor_op_error!("matmul", &format!("Matrix multiplication failed: {:?}", e))
        })?;

        Ok(PyVariable::new(result))
    }

    /// Clone variable
    /// 変数をクローン
    fn clone(&self) -> PyVariable {
        PyVariable::new(self.inner.clone())
    }

    /// Reshape variable
    /// 変数を再形状化
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyVariable> {
        let data = self.inner.data().read().unwrap();
        let current_numel = data.numel();
        let new_numel: usize = new_shape.iter().product();

        if current_numel != new_numel {
            return Err(tensor_op_error!(
                "reshape",
                &format!("Cannot reshape variable with {} elements to shape {:?} ({} elements)",
                        current_numel, new_shape, new_numel)
            ).into());
        }

        let reshaped_data = data.clone().into_shape_with_order(new_shape).map_err(|e| {
            tensor_op_error!("reshape", &format!("Reshape failed: {}", e))
        })?;

        let new_tensor = RustTensor::new(reshaped_data);
        let new_variable = RustVariable::new(new_tensor, self.requires_grad());

        Ok(PyVariable::new(new_variable))
    }

    /// Transpose variable (2D only)
    /// 変数の転置（2Dのみ）
    fn t(&self) -> PyResult<PyVariable> {
        let data = self.inner.data().read().unwrap();
        let shape = data.shape().to_vec();

        if shape.len() != 2 {
            return Err(tensor_op_error!(
                "transpose",
                &format!("Transpose only supports 2D variables, got {}D", shape.len())
            ).into());
        }

        // Create transposed data
        let flat_data: Vec<f32> = data.as_slice().unwrap().to_vec();
        let mut transposed_data = vec![0.0; flat_data.len()];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                transposed_data[j * shape[0] + i] = flat_data[i * shape[1] + j];
            }
        }

        let new_shape = vec![shape[1], shape[0]];
        let new_tensor = RustTensor::from_vec(transposed_data, new_shape);
        let new_variable = RustVariable::new(new_tensor, self.requires_grad());

        Ok(PyVariable::new(new_variable))
    }
}