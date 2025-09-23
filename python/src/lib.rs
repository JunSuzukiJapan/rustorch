use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use rustorch::tensor::core::Tensor as RustTensor;
use rustorch::autograd::Variable as RustVariable;
use rustorch::nn::Linear as RustLinear;
use rustorch::nn::Module;
use rustorch::nn::loss::{MSELoss as RustMSELoss, Loss};
use rustorch::nn::activation::{ReLU as RustReLU, sigmoid, tanh};

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

/// Python Variable wrapper for RusTorch Variable<f32>
#[pyclass(name = "Variable")]
#[derive(Clone)]
pub struct PyVariable {
    pub inner: RustVariable<f32>,
}

#[pymethods]
impl PyVariable {
    /// Create a new variable from tensor data
    #[new]
    fn new(data: &PyTensor, requires_grad: Option<bool>) -> PyResult<Self> {
        let requires_grad = requires_grad.unwrap_or(false);
        let variable = RustVariable::new(data.inner.clone(), requires_grad);
        Ok(PyVariable { inner: variable })
    }

    /// Get the underlying tensor data
    #[getter]
    fn data(&self) -> PyResult<PyTensor> {
        let data = self.inner.data().read().unwrap().clone();
        Ok(PyTensor { inner: data })
    }

    /// Get the gradient tensor if available
    #[getter]
    fn grad(&self) -> PyResult<Option<PyTensor>> {
        let grad_binding = self.inner.grad();
        let grad_lock = grad_binding.read().unwrap();
        match grad_lock.as_ref() {
            Some(grad) => Ok(Some(PyTensor { inner: grad.clone() })),
            None => Ok(None),
        }
    }

    /// Check if this variable requires gradients
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Zero out the gradient
    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad();
        Ok(())
    }

    /// Perform backward pass
    fn backward(&self) -> PyResult<()> {
        self.inner.backward();
        Ok(())
    }

    /// Sum all elements
    fn sum(&self) -> PyResult<PyVariable> {
        let result = self.inner.sum();
        Ok(PyVariable { inner: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        let data_binding = self.inner.data();
        let data = data_binding.read().unwrap();
        format!(
            "Variable(shape={:?}, requires_grad={}, data=...)",
            data.shape().to_vec(),
            self.requires_grad()
        )
    }

    /// Addition operation for Variables
    fn __add__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner + &other.inner;
        Ok(PyVariable { inner: result })
    }

    /// Subtraction operation for Variables
    fn __sub__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner - &other.inner;
        Ok(PyVariable { inner: result })
    }

    /// Multiplication operation for Variables
    fn __mul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.inner * &other.inner;
        Ok(PyVariable { inner: result })
    }

    /// Matrix multiplication
    fn matmul(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.matmul(&other.inner);
        Ok(PyVariable { inner: result })
    }
}

/// Python Linear layer wrapper for RusTorch Linear<f32>
#[pyclass(name = "Linear")]
pub struct PyLinear {
    pub inner: RustLinear<f32>,
}

#[pymethods]
impl PyLinear {
    /// Create a new linear layer
    #[new]
    fn new(input_size: usize, output_size: usize, bias: Option<bool>) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);
        let linear = if bias {
            RustLinear::new(input_size, output_size)
        } else {
            RustLinear::new_no_bias(input_size, output_size)
        };
        Ok(PyLinear { inner: linear })
    }

    /// Forward pass through the linear layer
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get the weight variable
    #[getter]
    fn weight(&self) -> PyResult<PyVariable> {
        let params = self.inner.parameters();
        // Weight is always the first parameter
        Ok(PyVariable { inner: params[0].clone() })
    }

    /// Get the bias variable if it exists
    #[getter]
    fn bias(&self) -> PyResult<Option<PyVariable>> {
        let params = self.inner.parameters();
        // If there are 2 parameters, bias is the second one
        if params.len() > 1 {
            Ok(Some(PyVariable { inner: params[1].clone() }))
        } else {
            Ok(None)
        }
    }

    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        self.inner.input_size()
    }

    /// Get output size
    #[getter]
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }

    /// String representation
    fn __repr__(&self) -> String {
        let params = self.inner.parameters();
        let has_bias = params.len() > 1;
        format!(
            "Linear(input_size={}, output_size={}, bias={})",
            self.input_size(),
            self.output_size(),
            has_bias
        )
    }
}

/// Python SGD optimizer wrapper (simplified implementation)
#[pyclass(name = "SGD")]
pub struct PySGD {
    pub parameters: Vec<PyVariable>,
    pub lr: f32,
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    #[new]
    fn new(
        parameters: Vec<PyVariable>,
        lr: f32,
        momentum: Option<f32>
    ) -> PyResult<Self> {
        // For now, ignore momentum - can be implemented later
        Ok(PySGD {
            parameters,
            lr
        })
    }

    /// Zero all parameter gradients
    fn zero_grad(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            param.inner.zero_grad();
        }
        Ok(())
    }

    /// Perform a single optimization step (simplified implementation)
    fn step(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            if !param.inner.requires_grad() {
                continue;
            }

            let grad_arc = param.inner.grad();
            let grad_lock = grad_arc.read().unwrap();

            if let Some(grad) = grad_lock.as_ref() {
                let param_data = param.inner.data();
                let mut param_lock = param_data.write().unwrap();
                let lr_tensor = RustTensor::from_vec(vec![self.lr], vec![]);
                let update = &*grad * &lr_tensor;
                *param_lock = &*param_lock - &update;
            }
        }
        Ok(())
    }

    /// Get current learning rate
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    fn set_lr(&mut self, lr: f32) -> PyResult<()> {
        self.lr = lr;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("SGD(lr={})", self.lr)
    }
}

/// Python MSELoss wrapper for RusTorch MSELoss
#[pyclass(name = "MSELoss")]
pub struct PyMSELoss {
    pub inner: RustMSELoss,
}

#[pymethods]
impl PyMSELoss {
    /// Create a new MSE loss function
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyMSELoss { inner: RustMSELoss })
    }

    /// Forward pass through MSE loss
    fn forward(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner, &target.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input, target)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "MSELoss()".to_string()
    }
}

/// Python ReLU activation wrapper for RusTorch ReLU<f32>
#[pyclass(name = "ReLU")]
pub struct PyReLU {
    pub inner: RustReLU<f32>,
}

#[pymethods]
impl PyReLU {
    /// Create a new ReLU activation function
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyReLU { inner: RustReLU::new() })
    }

    /// Forward pass through ReLU
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "ReLU()".to_string()
    }
}

/// Python Sigmoid activation wrapper using RusTorch sigmoid function
#[pyclass(name = "Sigmoid")]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    /// Create a new Sigmoid activation function
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PySigmoid)
    }

    /// Forward pass through Sigmoid
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = sigmoid(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Sigmoid()".to_string()
    }
}

/// Python Tanh activation wrapper using RusTorch tanh function
#[pyclass(name = "Tanh")]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    /// Create a new Tanh activation function
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyTanh)
    }

    /// Forward pass through Tanh
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = tanh(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Tanh()".to_string()
    }
}

/// RusTorch Python bindings module
#[pymodule]
fn _rustorch_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    Ok(())
}