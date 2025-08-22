//! Simplified Python bindings for RusTorch

use pyo3::prelude::*;
use pyo3::types::PyList;
use rustorch::tensor::Tensor;

/// Simple PyTensor wrapper
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    tensor: Tensor<f32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self {
            tensor: Tensor::from_vec(data, shape),
        }
    }
    
    fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }
    
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?})", self.shape())
    }
}

impl PyTensor {
    pub fn from_tensor(tensor: Tensor<f32>) -> Self {
        Self { tensor }
    }
    
    pub fn inner(&self) -> &Tensor<f32> {
        &self.tensor
    }
}

/// Create a tensor from a Python list
#[pyfunction]
fn tensor(data: &PyList) -> PyResult<PyTensor> {
    let mut flat_data = Vec::new();
    for item in data.iter() {
        flat_data.push(item.extract::<f32>()?);
    }
    Ok(PyTensor::new(flat_data))
}

/// Create a tensor filled with zeros
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyTensor {
    PyTensor::from_tensor(Tensor::<f32>::zeros(&shape))
}

/// Create a tensor filled with ones
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyTensor {
    PyTensor::from_tensor(Tensor::<f32>::ones(&shape))
}

/// Simple Variable wrapper for autograd
#[pyclass(name = "Variable")]
pub struct PyVariable {
    data: PyTensor,
    requires_grad: bool,
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new(data: PyTensor, requires_grad: Option<bool>) -> Self {
        Self {
            data,
            requires_grad: requires_grad.unwrap_or(false),
        }
    }
    
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    fn data(&self) -> PyTensor {
        self.data.clone()
    }
    
    fn zero_grad(&self) {
        // Placeholder for gradient zeroing
    }
    
    fn __repr__(&self) -> String {
        format!("Variable(shape={:?}, requires_grad={})", 
                self.data.shape(), self.requires_grad)
    }
}

/// Simple Linear layer
#[pyclass(name = "Linear")]
pub struct PyLinear {
    weight: PyTensor,
    bias: Option<PyTensor>,
    in_features: usize,
    out_features: usize,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize) -> Self {
        // Create random weights
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rand::random::<f32>() * 0.1)
            .collect();
        let weight_shape = vec![out_features, in_features];
        let weight = PyTensor::from_tensor(Tensor::from_vec(weight_data, weight_shape));
        
        // Create bias
        let bias_data: Vec<f32> = (0..out_features)
            .map(|_| 0.0)
            .collect();
        let bias = Some(PyTensor::new(bias_data));
        
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
    
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        // Simplified forward pass - just return input for now
        let output_data = input.data().inner().clone();
        let output_tensor = PyTensor::from_tensor(output_data);
        Ok(PyVariable::new(output_tensor, Some(input.requires_grad())))
    }
    
    fn parameters(&self) -> Vec<PyVariable> {
        let mut params = vec![PyVariable::new(
            PyTensor::from_tensor(self.weight.tensor.clone()), 
            Some(true)
        )];
        
        if let Some(ref bias) = self.bias {
            params.push(PyVariable::new(
                PyTensor::from_tensor(bias.tensor.clone()), 
                Some(true)
            ));
        }
        
        params
    }
    
    fn __repr__(&self) -> String {
        format!("Linear(in_features={}, out_features={})", 
                self.in_features, self.out_features)
    }
}

/// Simple SGD optimizer
#[pyclass(name = "SGD")]
pub struct PySGD {
    lr: f32,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(params: &PyList, lr: f32) -> Self {
        // For now, just store lr - we'll need to handle params differently
        Self { lr }
    }
    
    fn step(&self) {
        // Placeholder for optimization step
    }
    
    fn zero_grad(&self) {
        // Placeholder for zeroing gradients
    }
    
    fn __repr__(&self) -> String {
        format!("SGD(lr={})", self.lr)
    }
}

/// Python module
#[pymodule]
fn _rustorch_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySGD>()?;
    
    m.add("__version__", "0.3.3")?;
    
    Ok(())
}