//! Tensor implementation for RusTorch Python bindings - Working Version
//! RusTorch Pythonバインディング用Tensor実装 - 動作版

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use rustorch::tensor::core::Tensor as RustTensor;

/// Python Tensor wrapper for RusTorch Tensor<f32>
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: RustTensor<f32>,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from data and shape
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        // Validate data length matches shape
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Data length {} does not match shape {:?} (expected {})",
                        data.len(), shape, expected_len)
            ));
        }

        if shape.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape cannot be empty"));
        }

        let tensor = RustTensor::from_vec(data, shape);
        Ok(PyTensor { inner: tensor })
    }

    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get number of elements
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Get number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, data=...)", self.shape())
    }

    /// Addition operation
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner + &other.inner;
        Ok(PyTensor { inner: result })
    }

    /// Subtraction operation
    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner - &other.inner;
        Ok(PyTensor { inner: result })
    }

    /// Multiplication operation
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner * &other.inner;
        Ok(PyTensor { inner: result })
    }

    /// Division operation
    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner / &other.inner;
        Ok(PyTensor { inner: result })
    }
}

/// Create a tensor filled with zeros
#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = RustTensor::<f32>::zeros(&shape);
    Ok(PyTensor { inner: tensor })
}

/// Create a tensor filled with ones
#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = RustTensor::<f32>::ones(&shape);
    Ok(PyTensor { inner: tensor })
}

/// Create a tensor from a Python list
#[pyfunction]
pub fn tensor(data: &Bound<'_, PyList>) -> PyResult<PyTensor> {
    // Simple 1D case for now
    let mut vec_data = Vec::new();
    for item in data.iter() {
        let value: f32 = item.extract()?;
        vec_data.push(value);
    }

    let shape = vec![vec_data.len()];
    let tensor = RustTensor::from_vec(vec_data, shape);
    Ok(PyTensor { inner: tensor })
}