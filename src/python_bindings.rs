//! Python bindings for RusTorch
//! RusTorchのPythonバインディング

#[cfg(feature = "python")]
use pyo3::prelude::*;

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

    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }

    pub fn data(&self) -> Vec<f32> {
        self.tensor.iter().cloned().collect()
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
