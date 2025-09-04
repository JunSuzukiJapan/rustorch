//! Python bindings for RusTorch Tensor
//! RusTorch TensorのPythonバインディング

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{IntoPyArray, PyArray1};
use rustorch::tensor::Tensor;
use std::fmt;

/// Python wrapper for RusTorch Tensor
/// RusTorch TensorのPythonラッパー
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    tensor: Tensor<f32>,
}

impl PyTensor {
    pub fn new(tensor: Tensor<f32>) -> Self {
        Self { tensor }
    }
    
    pub fn inner(&self) -> &Tensor<f32> {
        &self.tensor
    }
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from data and shape
    /// データとシェイプから新しいテンソルを作成
    #[new]
    fn py_new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new(Tensor::from_vec(data, shape))
    }
    
    /// Get tensor shape
    /// テンソルのシェイプを取得
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }
    
    /// Get number of dimensions
    /// 次元数を取得
    #[getter]
    fn ndim(&self) -> usize {
        self.tensor.shape().len()
    }
    
    /// Get total number of elements
    /// 要素の総数を取得
    #[getter]
    fn size(&self) -> usize {
        self.tensor.shape().iter().product()
    }
    
    /// Convert to NumPy array
    /// NumPy配列に変換
    fn numpy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        let data = self.tensor.as_array();
        data.into_pyarray(py)
    }
    
    /// Reshape tensor
    /// テンソルをリシェイプ
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        let reshaped = self.tensor.reshape(&shape);
        Ok(PyTensor::new(reshaped))
    }
    
    /// Sum all elements
    /// すべての要素の合計
    fn sum(&self) -> PyTensor {
        PyTensor::new(self.tensor.sum())
    }
    
    /// Mean of all elements
    /// すべての要素の平均
    fn mean(&self) -> PyTensor {
        PyTensor::new(self.tensor.mean())
    }
    
    /// Matrix multiplication
    /// 行列乗算
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.tensor.matmul(&other.tensor);
        Ok(PyTensor::new(result))
    }
    
    /// Add scalar
    /// スカラー加算
    fn add_scalar(&self, scalar: f32) -> PyTensor {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let result = &self.tensor + &scalar_tensor;
        PyTensor::new(result)
    }
    
    /// Multiply by scalar
    /// スカラー乗算
    fn mul_scalar(&self, scalar: f32) -> PyTensor {
        let scalar_tensor = Tensor::from_scalar(scalar);
        let result = &self.tensor * &scalar_tensor;
        PyTensor::new(result)
    }
    
    /// Clone tensor
    /// テンソルをクローン
    fn clone(&self) -> PyTensor {
        PyTensor::new(self.tensor.clone())
    }
    
    /// Element-wise addition
    /// 要素ごとの加算
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.tensor + &other.tensor;
        Ok(PyTensor::new(result))
    }
    
    /// Element-wise subtraction
    /// 要素ごとの減算
    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.tensor - &other.tensor;
        Ok(PyTensor::new(result))
    }
    
    /// Element-wise multiplication
    /// 要素ごとの乗算
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.tensor * &other.tensor;
        Ok(PyTensor::new(result))
    }
    
    /// Matrix multiplication (@ operator)
    /// 行列乗算（@演算子）
    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }
    
    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype=float32)", self.tensor.shape())
    }
    
    /// String representation for print
    /// print用の文字列表現
    fn __str__(&self) -> String {
        // For now, just show shape and type
        // Note: data display for small tensors not implemented
        format!("tensor(shape={:?}, dtype=float32)", self.tensor.shape())
    }
    
    /// Get item by index (for 1D tensors)
    /// インデックスで要素を取得（1次元テンソル用）
    fn __getitem__(&self, index: usize) -> PyResult<f32> {
        if self.tensor.shape().len() != 1 {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Indexing only supported for 1D tensors"
            ));
        }
        
        if index >= self.tensor.shape()[0] {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }
        
        let data = self.tensor.as_array();
        Ok(data[index])
    }
    
    /// Get length (for 1D tensors)
    /// 長さを取得（1次元テンソル用）
    fn __len__(&self) -> PyResult<usize> {
        if self.tensor.shape().len() != 1 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "len() only supported for 1D tensors"
            ));
        }
        Ok(self.tensor.shape()[0])
    }
    
    /// Support for numpy array interface
    /// numpy配列インターフェースのサポート
    #[getter]
    fn __array_interface__(&self, py: Python) -> PyResult<PyObject> {
        let shape = PyTuple::new(py, self.tensor.shape());
        let interface = pyo3::types::PyDict::new(py);
        
        interface.set_item("shape", shape)?;
        interface.set_item("typestr", "<f4")?;  // little-endian float32
        interface.set_item("version", 3)?;
        
        Ok(interface.into())
    }
}