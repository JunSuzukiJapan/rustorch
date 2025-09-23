use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use rustorch::tensor::core::Tensor as RustTensor;
use num_traits::Float;

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
fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = RustTensor::<f32>::zeros(&shape);
    Ok(PyTensor { inner: tensor })
}

/// Create a tensor filled with ones
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = RustTensor::<f32>::ones(&shape);
    Ok(PyTensor { inner: tensor })
}

/// Create a tensor from a Python list
#[pyfunction]
fn tensor(data: &Bound<'_, PyList>) -> PyResult<PyTensor> {
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

/// RusTorch Python bindings module
#[pymodule]
fn _rustorch_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    Ok(())
}