//! Python bindings for RusTorch
//! RusTorchのPythonバインディング

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray, IntoPyArray};

#[cfg(feature = "python")]
use crate::tensor::operations::zero_copy::TensorIterOps;
#[cfg(feature = "python")]
use crate::tensor::Tensor;

#[cfg(feature = "python")]
/// Python wrapper for RusTorch Tensor
#[pyclass]
pub struct PyTensor {
    tensor: Tensor<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let tensor = Tensor::from_vec(data, shape);
        Ok(PyTensor { tensor })
    }

    /// Create PyTensor from NumPy array
    #[staticmethod]
    pub fn from_numpy(array: PyReadonlyArray1<f32>) -> PyResult<Self> {
        let array = array.as_array();
        let data = array.to_vec();
        let shape = vec![array.len()];
        let tensor = Tensor::from_vec(data, shape);
        Ok(PyTensor { tensor })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }

    pub fn data(&self) -> Vec<f32> {
        self.tensor.iter().cloned().collect()
    }

    /// Convert PyTensor to NumPy array
    pub fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let data = self.tensor.iter().cloned().collect::<Vec<f32>>();
        data.into_pyarray(py)
    }

    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.tensor + &other.tensor;
        Ok(PyTensor { tensor: result })
    }

    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match self.tensor.matmul(&other.tensor) {
            Ok(result) => Ok(PyTensor { tensor: result }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("{:?}", e))),
        }
    }

    pub fn relu(&self) -> PyTensor {
        let data: Vec<f32> = self.tensor.iter().map(|&x| x.max(0.0)).collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    pub fn sigmoid(&self) -> PyTensor {
        let data: Vec<f32> = self
            .tensor
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    /// Element-wise exponential function
    pub fn exp(&self) -> PyTensor {
        let data: Vec<f32> = self.tensor.iter().map(|&x| x.exp()).collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> PyTensor {
        let data: Vec<f32> = self.tensor.iter().map(|&x| x.ln()).collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    /// Element-wise sine function
    pub fn sin(&self) -> PyTensor {
        let data: Vec<f32> = self.tensor.iter().map(|&x| x.sin()).collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    /// Element-wise cosine function  
    pub fn cos(&self) -> PyTensor {
        let data: Vec<f32> = self.tensor.iter().map(|&x| x.cos()).collect();
        let tensor = Tensor::from_vec(data, self.tensor.shape().to_vec());
        PyTensor { tensor }
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyTensor> {
        let total_elements: usize = self.tensor.shape().iter().product();
        let new_elements: usize = new_shape.iter().product();
        
        if total_elements != new_elements {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Cannot reshape tensor of size {} to shape {:?}", total_elements, new_shape)
            ));
        }
        
        let data = self.tensor.iter().cloned().collect();
        let tensor = Tensor::from_vec(data, new_shape);
        Ok(PyTensor { tensor })
    }

    /// Transpose tensor (swap dimensions 0 and 1)
    pub fn transpose(&self) -> PyResult<PyTensor> {
        let shape = self.tensor.shape();
        if shape.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Transpose requires at least 2 dimensions"
            ));
        }

        // Simple 2D transpose for now
        if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            let mut result = vec![0.0f32; rows * cols];
            
            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = self.tensor.iter().nth(i * cols + j).cloned().unwrap();
                }
            }
            
            let tensor = Tensor::from_vec(result, vec![cols, rows]);
            Ok(PyTensor { tensor })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Transpose currently only supports 2D tensors"
            ))
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.tensor.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        let sum: f32 = self.tensor.iter().sum();
        let count = self.tensor.iter().count() as f32;
        sum / count
    }

    /// Standard deviation of all elements
    pub fn std(&self) -> f32 {
        let mean = self.mean();
        let variance = self.tensor.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / (self.tensor.iter().count() as f32);
        variance.sqrt()
    }

    /// Variance of all elements  
    pub fn var(&self) -> f32 {
        let mean = self.mean();
        let variance = self.tensor.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / (self.tensor.iter().count() as f32);
        variance
    }

    pub fn __repr__(&self) -> String {
        format!("PyTensor(shape={:?})", self.tensor.shape())
    }
}

#[cfg(feature = "python")]
/// A Python module implemented in Rust
#[pymodule]
fn rustorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;

    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];
    PyTensor::new(data, shape)
}

#[cfg(feature = "python")]
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![1.0f32; size];
    PyTensor::new(data, shape)
}

#[cfg(feature = "python")]
#[pyfunction]
fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    PyTensor::new(data, shape)
}

#[cfg(feature = "python")]
#[pyfunction]
fn from_numpy(array: PyReadonlyArray1<f32>) -> PyResult<PyTensor> {
    PyTensor::from_numpy(array)
}
