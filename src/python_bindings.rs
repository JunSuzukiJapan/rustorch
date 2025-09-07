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
use crate::nn::{Linear, Module};
#[cfg(feature = "python")]
use crate::autograd::Variable;
#[cfg(feature = "python")]
use crate::nn::activation::{relu, sigmoid, tanh, softmax, gelu, leaky_relu, swish, elu, selu, mish};
#[cfg(feature = "python")]
use crate::nn::loss::{mse_loss, cross_entropy_loss, MSELoss, CrossEntropyLoss};
#[cfg(feature = "python")]
use crate::tensor::device::Device;
#[cfg(feature = "python")]
use crate::optim::{SGD, Adam, Optimizer};

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
/// Python wrapper for RusTorch Variable
#[pyclass]
pub struct PyVariable {
    variable: Variable<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVariable {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: Option<bool>) -> PyResult<Self> {
        let tensor = Tensor::from_vec(data, shape);
        let variable = Variable::new(tensor, requires_grad.unwrap_or(false));
        Ok(PyVariable { variable })
    }

    pub fn data(&self) -> PyTensor {
        let data_binding = self.variable.data();
        let data = data_binding.read().unwrap();
        let shape = data.shape().to_vec();
        let values: Vec<f32> = data.iter().cloned().collect();
        let tensor = Tensor::from_vec(values, shape);
        PyTensor { tensor }
    }

    pub fn requires_grad(&self) -> bool {
        self.variable.requires_grad()
    }

    pub fn shape(&self) -> Vec<usize> {
        let data_binding = self.variable.data();
        let data = data_binding.read().unwrap();
        data.shape().to_vec()
    }

    /// Apply ReLU activation
    pub fn relu(&self) -> PyVariable {
        let result = relu(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply Sigmoid activation
    pub fn sigmoid(&self) -> PyVariable {
        let result = sigmoid(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply Tanh activation
    pub fn tanh(&self) -> PyVariable {
        let result = tanh(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply Softmax activation
    pub fn softmax(&self) -> PyVariable {
        let result = softmax(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply GELU activation
    pub fn gelu(&self) -> PyVariable {
        let result = gelu(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply Leaky ReLU activation
    pub fn leaky_relu(&self, alpha: f32) -> PyVariable {
        let result = leaky_relu(&self.variable, alpha);
        PyVariable { variable: result }
    }

    /// Apply Swish activation
    pub fn swish(&self) -> PyVariable {
        let result = swish(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply ELU activation
    pub fn elu(&self, alpha: f32) -> PyVariable {
        let result = elu(&self.variable, alpha);
        PyVariable { variable: result }
    }

    /// Apply SELU activation
    pub fn selu(&self) -> PyVariable {
        let result = selu(&self.variable);
        PyVariable { variable: result }
    }

    /// Apply Mish activation
    pub fn mish(&self) -> PyVariable {
        let result = mish(&self.variable);
        PyVariable { variable: result }
    }

    pub fn __repr__(&self) -> String {
        let data_binding = self.variable.data();
        let data = data_binding.read().unwrap();
        format!("PyVariable(shape={:?}, requires_grad={})", 
                data.shape(), self.variable.requires_grad())
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch Linear layer
#[pyclass]
pub struct PyLinear {
    linear: Linear<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLinear {
    #[new]
    pub fn new(input_size: usize, output_size: usize, bias: Option<bool>) -> PyResult<Self> {
        let linear = if bias.unwrap_or(true) {
            Linear::new(input_size, output_size)
        } else {
            Linear::new_no_bias(input_size, output_size)
        };
        Ok(PyLinear { linear })
    }

    pub fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let output = self.linear.forward(&input.variable);
        Ok(PyVariable { variable: output })
    }

    pub fn input_size(&self) -> usize {
        self.linear.input_size()
    }

    pub fn output_size(&self) -> usize {
        self.linear.output_size()
    }

    pub fn parameters(&self) -> Vec<PyVariable> {
        self.linear.parameters()
            .into_iter()
            .map(|param| PyVariable { variable: param })
            .collect()
    }

    pub fn __repr__(&self) -> String {
        format!("PyLinear(in_features={}, out_features={})",
                self.linear.input_size(), self.linear.output_size())
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch MSE Loss
#[pyclass]
pub struct PyMSELoss {
    loss: MSELoss,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMSELoss {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyMSELoss { loss: MSELoss })
    }

    pub fn forward(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        let result = mse_loss(&predictions.variable, &targets.variable);
        Ok(PyVariable { variable: result })
    }

    pub fn __call__(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        self.forward(predictions, targets)
    }

    pub fn __repr__(&self) -> String {
        "PyMSELoss()".to_string()
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch CrossEntropy Loss  
#[pyclass]
pub struct PyCrossEntropyLoss {
    loss: CrossEntropyLoss<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyCrossEntropyLoss { 
            loss: CrossEntropyLoss::new() 
        })
    }

    pub fn forward(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        let result = cross_entropy_loss(&predictions.variable, &targets.variable);
        Ok(PyVariable { variable: result })
    }

    pub fn __call__(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        self.forward(predictions, targets)
    }

    pub fn __repr__(&self) -> String {
        "PyCrossEntropyLoss()".to_string()
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch Device
#[pyclass]
pub struct PyDevice {
    device: Device,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDevice {
    /// Create CPU device
    #[staticmethod]
    pub fn cpu() -> PyResult<Self> {
        Ok(PyDevice { device: Device::Cpu })
    }

    /// Create CUDA device with optional device index
    #[staticmethod]
    pub fn cuda(device_id: Option<usize>) -> PyResult<Self> {
        let device_id = device_id.unwrap_or(0);
        Ok(PyDevice { device: Device::Cuda(device_id) })
    }

    /// Create MPS device (Apple Metal Performance Shaders)
    #[staticmethod]
    pub fn mps() -> PyResult<Self> {
        Ok(PyDevice { device: Device::Mps })
    }

    /// Create WASM device
    #[staticmethod]
    pub fn wasm() -> PyResult<Self> {
        Ok(PyDevice { device: Device::Wasm })
    }

    /// Check if device is CPU
    pub fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    /// Check if device is CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// Check if device is MPS
    pub fn is_mps(&self) -> bool {
        self.device.is_mps()
    }

    /// Check if device is WASM
    pub fn is_wasm(&self) -> bool {
        self.device.is_wasm()
    }

    /// Get CUDA device index if applicable
    pub fn cuda_index(&self) -> Option<usize> {
        self.device.cuda_index()
    }

    /// Get device type as string
    pub fn type_str(&self) -> String {
        match self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(idx) => format!("cuda:{}", idx),
            Device::Mps => "mps".to_string(),
            Device::Wasm => "wasm".to_string(),
        }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.device)
    }

    pub fn __repr__(&self) -> String {
        format!("PyDevice({})", self.device)
    }

    pub fn __eq__(&self, other: &PyDevice) -> bool {
        self.device == other.device
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch SGD optimizer
#[pyclass]
pub struct PySGD {
    optimizer: SGD,
    parameters: Vec<pyo3::Py<PyVariable>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySGD {
    #[new]
    pub fn new(
        params: &pyo3::Bound<'_, pyo3::types::PyList>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: Option<bool>,
    ) -> PyResult<Self> {
        let mut parameters = Vec::new();
        for item in params.iter() {
            let py_var: pyo3::Py<PyVariable> = item.extract()?;
            parameters.push(py_var);
        }
        
        let mut optimizer = if let Some(momentum_val) = momentum {
            if let Some(weight_decay_val) = weight_decay {
                SGD::with_weight_decay(lr, momentum_val, weight_decay_val)
            } else {
                SGD::with_momentum(lr, momentum_val)
            }
        } else {
            SGD::new(lr)
        };
        
        if let Some(nesterov_val) = nesterov {
            optimizer = SGD::with_nesterov(lr, momentum.unwrap_or(0.0), nesterov_val);
        }

        Ok(PySGD { optimizer, parameters })
    }

    /// Perform a single optimization step
    pub fn step(&mut self, py: pyo3::Python<'_>) -> PyResult<()> {
        for param in &self.parameters {
            let param_borrowed = param.borrow(py);
            let param_data = param_borrowed.variable.data();
            let param_tensor = param_data.read().unwrap();
            
            let grad_arc = param_borrowed.variable.grad();
            let grad_lock = grad_arc.read().unwrap();
            
            if let Some(grad_tensor) = grad_lock.as_ref() {
                self.optimizer.step(&param_tensor, grad_tensor);
            }
        }
        Ok(())
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Zero gradients of all parameters
    pub fn zero_grad(&mut self, py: pyo3::Python<'_>) -> PyResult<()> {
        for param in &self.parameters {
            let param_borrowed = param.borrow(py);
            let grad_arc = param_borrowed.variable.grad();
            let mut grad_lock = grad_arc.write().unwrap();
            *grad_lock = None;
        }
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!("PySGD(lr={})", self.optimizer.learning_rate())
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch Adam optimizer
#[pyclass]
pub struct PyAdam {
    optimizer: Adam,
    parameters: Vec<pyo3::Py<PyVariable>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdam {
    #[new]
    pub fn new(
        params: &pyo3::Bound<'_, pyo3::types::PyList>,
        lr: Option<f32>,
        beta1: Option<f32>,
        beta2: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
    ) -> PyResult<Self> {
        let mut parameters = Vec::new();
        for item in params.iter() {
            let py_var: pyo3::Py<PyVariable> = item.extract()?;
            parameters.push(py_var);
        }
        
        let lr = lr.unwrap_or(0.001);
        let beta1 = beta1.unwrap_or(0.9);
        let beta2 = beta2.unwrap_or(0.999);
        let eps = eps.unwrap_or(1e-8);

        let optimizer = if let Some(weight_decay_val) = weight_decay {
            Adam::with_weight_decay(lr, beta1, beta2, eps, weight_decay_val)
        } else {
            Adam::new(lr, beta1, beta2, eps)
        };

        Ok(PyAdam { optimizer, parameters })
    }

    /// Perform a single optimization step
    pub fn step(&mut self, py: pyo3::Python<'_>) -> PyResult<()> {
        for param in &self.parameters {
            let param_borrowed = param.borrow(py);
            let param_data = param_borrowed.variable.data();
            let param_tensor = param_data.read().unwrap();
            
            let grad_arc = param_borrowed.variable.grad();
            let grad_lock = grad_arc.read().unwrap();
            
            if let Some(grad_tensor) = grad_lock.as_ref() {
                self.optimizer.step(&param_tensor, grad_tensor);
            }
        }
        Ok(())
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Zero gradients of all parameters
    pub fn zero_grad(&mut self, py: pyo3::Python<'_>) -> PyResult<()> {
        for param in &self.parameters {
            let param_borrowed = param.borrow(py);
            let grad_arc = param_borrowed.variable.grad();
            let mut grad_lock = grad_arc.write().unwrap();
            *grad_lock = None;
        }
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!("PyAdam(lr={})", self.optimizer.learning_rate())
    }
}

#[cfg(feature = "python")]
/// A Python module implemented in Rust
#[pymodule]
fn rustorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;

    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    
    // Activation functions
    m.add_function(wrap_pyfunction!(py_relu, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(py_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(py_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(py_gelu, m)?)?;
    m.add_function(wrap_pyfunction!(py_leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(py_swish, m)?)?;
    m.add_function(wrap_pyfunction!(py_elu, m)?)?;
    m.add_function(wrap_pyfunction!(py_selu, m)?)?;
    m.add_function(wrap_pyfunction!(py_mish, m)?)?;
    
    // Loss functions
    m.add_function(wrap_pyfunction!(py_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_entropy_loss, m)?)?;
    
    // Device management
    m.add_function(wrap_pyfunction!(py_device_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_mps, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_wasm, m)?)?;

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

// Activation functions
#[cfg(feature = "python")]
#[pyfunction]
fn py_relu(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.relu())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_sigmoid(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.sigmoid())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_tanh(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.tanh())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_softmax(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.softmax())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_gelu(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.gelu())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_leaky_relu(x: &PyVariable, alpha: f32) -> PyResult<PyVariable> {
    Ok(x.leaky_relu(alpha))
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_swish(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.swish())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_elu(x: &PyVariable, alpha: f32) -> PyResult<PyVariable> {
    Ok(x.elu(alpha))
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_selu(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.selu())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_mish(x: &PyVariable) -> PyResult<PyVariable> {
    Ok(x.mish())
}

// Loss functions
#[cfg(feature = "python")]
#[pyfunction]
fn py_mse_loss(predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
    let result = mse_loss(&predictions.variable, &targets.variable);
    Ok(PyVariable { variable: result })
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_cross_entropy_loss(predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
    let result = cross_entropy_loss(&predictions.variable, &targets.variable);
    Ok(PyVariable { variable: result })
}

// Device management functions
#[cfg(feature = "python")]
#[pyfunction]
fn py_device_cpu() -> PyResult<PyDevice> {
    PyDevice::cpu()
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_device_cuda(device_id: Option<usize>) -> PyResult<PyDevice> {
    PyDevice::cuda(device_id)
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_device_mps() -> PyResult<PyDevice> {
    PyDevice::mps()
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_device_wasm() -> PyResult<PyDevice> {
    PyDevice::wasm()
}
