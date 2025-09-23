//! Tensor implementation for RusTorch Python bindings
//! RusTorch Pythonバインディング用Tensor実装

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use rustorch::tensor::core::Tensor as RustTensor;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{shape_mismatch, tensor_op_error};

/// Python Tensor wrapper for RusTorch Tensor<f32>
/// RusTorch Tensor<f32>のPythonラッパー
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: RustTensor<f32>,
}

impl PyTensor {
    /// Create a new PyTensor from RustTensor
    /// RustTensorからPyTensorを作成
    pub fn new(tensor: RustTensor<f32>) -> Self {
        Self { inner: tensor }
    }

    /// Get reference to inner tensor
    /// 内部テンソルへの参照を取得
    pub fn as_rust_tensor(&self) -> &RustTensor<f32> {
        &self.inner
    }

    /// Validate tensor shapes match
    /// テンソル形状の一致を検証
    fn validate_shape_match(&self, other: &PyTensor, op_name: &str) -> PyResult<()> {
        if self.inner.shape() != other.inner.shape() {
            return Err(shape_mismatch!(
                self.inner.shape().to_vec(),
                other.inner.shape().to_vec()
            ).into());
        }
        Ok(())
    }

    /// Execute binary operation with shape validation
    /// 形状検証付きバイナリ操作実行
    fn binary_op<F>(&self, other: &PyTensor, op_name: &str, op: F) -> PyResult<PyTensor>
    where
        F: FnOnce(&RustTensor<f32>, &RustTensor<f32>) -> RustTensor<f32>,
    {
        self.validate_shape_match(other, op_name)?;
        let result = op(&self.inner, &other.inner);
        Ok(PyTensor::new(result))
    }
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from data and shape
    /// データと形状から新しいテンソルを作成
    #[new]
    fn py_new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        // Validate data length matches shape
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(tensor_op_error!(
                "tensor_creation",
                &format!("Data length {} does not match shape {:?} (expected {})",
                        data.len(), shape, expected_len)
            ).into());
        }

        let tensor = RustTensor::from_vec(data, shape);
        Ok(PyTensor::new(tensor))
    }

    /// Get tensor shape
    /// テンソル形状を取得
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get number of elements
    /// 要素数を取得
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Get number of dimensions
    /// 次元数を取得
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get tensor data as Python list
    /// テンソルデータをPythonリストとして取得
    fn to_list(&self, py: Python) -> PyResult<PyObject> {
        let data = self.inner.data().read().unwrap();
        let vec: Vec<f32> = data.as_slice().unwrap().to_vec();
        Ok(vec.to_object(py))
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, numel={})", self.shape(), self.numel())
    }

    /// Addition operation
    /// 加算操作
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, "add", |a, b| a + b)
    }

    /// Subtraction operation
    /// 減算操作
    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, "sub", |a, b| a - b)
    }

    /// Multiplication operation
    /// 乗算操作
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, "mul", |a, b| a * b)
    }

    /// Division operation
    /// 除算操作
    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, "div", |a, b| a / b)
    }

    /// Clone tensor
    /// テンソルをクローン
    fn clone(&self) -> PyTensor {
        PyTensor::new(self.inner.clone())
    }

    /// Reshape tensor
    /// テンソルを再形状化
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyTensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(tensor_op_error!(
                "reshape",
                &format!("Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                        self.numel(), new_shape, new_numel)
            ).into());
        }

        let data = self.inner.data().read().unwrap().clone();
        let reshaped = data.into_shape_with_order(new_shape).map_err(|e| {
            tensor_op_error!("reshape", &format!("Reshape failed: {}", e))
        })?;

        let new_tensor = RustTensor::new(reshaped);
        Ok(PyTensor::new(new_tensor))
    }

    /// Transpose tensor (2D only for now)
    /// テンソルの転置（現在は2Dのみ）
    fn t(&self) -> PyResult<PyTensor> {
        if self.ndim() != 2 {
            return Err(tensor_op_error!(
                "transpose",
                &format!("Transpose only supports 2D tensors, got {}D", self.ndim())
            ).into());
        }

        let shape = self.shape();
        let new_shape = vec![shape[1], shape[0]];

        // For simplicity, implement as reshape for now
        // 簡単のため、現在はreshapeとして実装
        let data = self.inner.data().read().unwrap();
        let flat_data: Vec<f32> = data.as_slice().unwrap().to_vec();

        // Transpose data
        let mut transposed_data = vec![0.0; flat_data.len()];
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                transposed_data[j * shape[0] + i] = flat_data[i * shape[1] + j];
            }
        }

        let tensor = RustTensor::from_vec(transposed_data, new_shape);
        Ok(PyTensor::new(tensor))
    }
}

/// Create a tensor filled with zeros
/// ゼロで埋められたテンソルを作成
#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    if shape.is_empty() {
        return Err(tensor_op_error!("zeros", "Shape cannot be empty").into());
    }

    let tensor = RustTensor::<f32>::zeros(&shape);
    Ok(PyTensor::new(tensor))
}

/// Create a tensor filled with ones
/// 1で埋められたテンソルを作成
#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    if shape.is_empty() {
        return Err(tensor_op_error!("ones", "Shape cannot be empty").into());
    }

    let tensor = RustTensor::<f32>::ones(&shape);
    Ok(PyTensor::new(tensor))
}

/// Create a tensor from a Python list
/// Pythonリストからテンソルを作成
#[pyfunction]
pub fn tensor(data: &Bound<'_, PyList>) -> PyResult<PyTensor> {
    // Simple 1D case for now
    // 現在は簡単な1Dケースのみ
    let mut vec_data = Vec::new();
    for item in data.iter() {
        let value: f32 = item.extract().map_err(|e| {
            tensor_op_error!("tensor_creation", &format!("Failed to extract float: {}", e))
        })?;
        vec_data.push(value);
    }

    if vec_data.is_empty() {
        return Err(tensor_op_error!("tensor_creation", "Cannot create tensor from empty list").into());
    }

    let shape = vec![vec_data.len()];
    let tensor = RustTensor::from_vec(vec_data, shape);
    Ok(PyTensor::new(tensor))
}