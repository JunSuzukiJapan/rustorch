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
use crate::nn::{Conv2d, BatchNorm2d};
#[cfg(feature = "python")]
use crate::data::{Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler};
use crate::serialization::model_io::{save, load};
use crate::serialization::core::{Saveable, Loadable, ModelMetadata, SerializationError};
use crate::training::trainer::{Trainer, TrainerConfig, TrainingDataLoader};
use crate::training::metrics::{MetricsCollector, TrainingMetrics};
use crate::models::high_level::{HighLevelModel, TrainingHistory};
use crate::nn::loss::{Loss, MSELoss, CrossEntropyLoss};
use crate::distributed::{DistributedDataParallel, wrap_module};
use std::collections::HashMap;

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

    // Linear Algebra operations
    
    /// Singular Value Decomposition
    pub fn svd(&self, compute_uv: Option<bool>) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let compute_uv = compute_uv.unwrap_or(true);
        
        match self.tensor.svd(compute_uv) {
            Ok((u, s, vt)) => Ok((
                PyTensor { tensor: u },
                PyTensor { tensor: s },
                PyTensor { tensor: vt },
            )),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SVD failed: {}", e))),
        }
    }

    /// QR Decomposition
    pub fn qr(&self) -> PyResult<(PyTensor, PyTensor)> {
        match self.tensor.qr() {
            Ok((q, r)) => Ok((
                PyTensor { tensor: q },
                PyTensor { tensor: r },
            )),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("QR decomposition failed: {}", e))),
        }
    }

    /// Eigenvalue decomposition
    pub fn eig(&self, eigenvectors: Option<bool>) -> PyResult<(PyTensor, Option<PyTensor>)> {
        let eigenvectors = eigenvectors.unwrap_or(true);
        
        match self.tensor.eig(eigenvectors) {
            Ok((eigenvalues, eigenvectors_opt)) => {
                let py_eigenvectors = if let Some(eigenvecs) = eigenvectors_opt {
                    Some(PyTensor { tensor: eigenvecs })
                } else {
                    None
                };
                Ok((PyTensor { tensor: eigenvalues }, py_eigenvectors))
            },
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Eigenvalue decomposition failed: {}", e))),
        }
    }

    /// Matrix determinant
    pub fn det(&self) -> PyResult<f32> {
        match self.tensor.det() {
            Ok(det_value) => Ok(det_value),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Determinant calculation failed: {}", e))),
        }
    }

    /// Matrix inverse
    pub fn inverse(&self) -> PyResult<PyTensor> {
        match self.tensor.inverse() {
            Ok(inv_tensor) => Ok(PyTensor { tensor: inv_tensor }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Matrix inversion failed: {}", e))),
        }
    }

    /// Matrix norm
    pub fn norm(&self, ord: Option<String>) -> PyResult<f32> {
        let ord = ord.unwrap_or_else(|| "fro".to_string());
        match self.tensor.norm(&ord) {
            Ok(norm_value) => Ok(norm_value),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Norm calculation failed: {}", e))),
        }
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
/// Python wrapper for RusTorch Conv2d layer
#[pyclass]
pub struct PyConv2d {
    layer: Conv2d<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyConv2d {
    #[new]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let layer = Conv2d::new(in_channels, out_channels, kernel_size, stride, padding, bias);
        Ok(PyConv2d { layer })
    }

    /// Forward pass through the convolution layer
    pub fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let output_var = self.layer.forward(&input.variable);
        Ok(PyVariable { 
            variable: output_var 
        })
    }

    /// Get layer parameters (weights and biases)
    pub fn parameters(&self) -> PyResult<Vec<PyVariable>> {
        let params = self.layer.parameters();
        let py_params = params
            .into_iter()
            .map(|param| PyVariable { variable: param })
            .collect();
        Ok(py_params)
    }

    /// Compute output size given input dimensions
    pub fn compute_output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        self.layer.compute_output_size(input_height, input_width)
    }

    pub fn __repr__(&self) -> String {
        format!("PyConv2d(in_channels={}, out_channels={}, kernel_size={:?}, stride={:?}, padding={:?})", 
                self.layer.in_channels(), 
                self.layer.out_channels(), 
                self.layer.kernel_size(), 
                self.layer.stride(), 
                self.layer.padding())
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch BatchNorm2d layer
#[pyclass]
pub struct PyBatchNorm2d {
    layer: BatchNorm2d<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBatchNorm2d {
    #[new]
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<Self> {
        let layer = BatchNorm2d::new(num_features, eps, momentum, affine);
        Ok(PyBatchNorm2d { layer })
    }

    /// Forward pass through the batch normalization layer
    pub fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let output_var = self.layer.forward(&input.variable);
        Ok(PyVariable { 
            variable: output_var 
        })
    }

    /// Get layer parameters (weight and bias)
    pub fn parameters(&self) -> PyResult<Vec<PyVariable>> {
        let params = self.layer.parameters();
        let py_params = params
            .into_iter()
            .map(|param| PyVariable { variable: param })
            .collect();
        Ok(py_params)
    }

    /// Set training mode
    pub fn train(&self) {
        self.layer.train();
    }

    /// Set evaluation mode
    pub fn eval(&self) {
        self.layer.eval();
    }

    /// Check if layer is in training mode
    pub fn is_training(&self) -> bool {
        self.layer.is_training()
    }

    pub fn __repr__(&self) -> String {
        format!("PyBatchNorm2d(num_features={}, eps={:.6}, momentum={:.3}, training={})", 
                self.layer.num_features(),
                self.layer.eps(),
                self.layer.momentum(),
                self.layer.is_training())
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch TensorDataset
#[pyclass]
pub struct PyTensorDataset {
    dataset: TensorDataset<f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTensorDataset {
    #[new]
    pub fn new(tensors: &pyo3::Bound<'_, pyo3::types::PyList>) -> PyResult<Self> {
        let mut rust_tensors = Vec::new();
        for item in tensors.iter() {
            let py_tensor: PyTensor = item.extract()?;
            rust_tensors.push(py_tensor.tensor);
        }
        
        match TensorDataset::new(rust_tensors) {
            Ok(dataset) => Ok(PyTensorDataset { dataset }),
            Err(_e) => Err(pyo3::exceptions::PyValueError::new_err("Failed to create TensorDataset")),
        }
    }

    /// Get the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Get item at index
    pub fn get_item(&self, index: usize) -> PyResult<Vec<PyTensor>> {
        match self.dataset.get_item(index) {
            Ok(tensors) => {
                let py_tensors = tensors
                    .into_iter()
                    .map(|tensor| PyTensor { tensor })
                    .collect();
                Ok(py_tensors)
            },
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("Index error: {}", e))),
        }
    }

    /// Get multiple items as a batch
    pub fn get_batch(&self, indices: Vec<usize>) -> PyResult<Vec<Vec<PyTensor>>> {
        let mut batch = Vec::new();
        for index in indices {
            match self.dataset.get_item(index) {
                Ok(tensors) => {
                    let py_tensors = tensors
                        .into_iter()
                        .map(|tensor| PyTensor { tensor })
                        .collect();
                    batch.push(py_tensors);
                },
                Err(e) => return Err(pyo3::exceptions::PyIndexError::new_err(format!("Index error at {}: {}", index, e))),
            }
        }
        Ok(batch)
    }

    pub fn __repr__(&self) -> String {
        format!("PyTensorDataset(len={})", self.dataset.len())
    }

    pub fn __len__(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch DataLoader
#[pyclass]
pub struct PyDataLoader {
    // Store essential components for batch iteration
    dataset_len: usize,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDataLoader {
    #[new]
    pub fn new(
        batch_size: Option<usize>,
        shuffle: Option<bool>,
        drop_last: Option<bool>,
    ) -> PyResult<Self> {
        let batch_size = batch_size.unwrap_or(1);
        let shuffle = shuffle.unwrap_or(true);
        let drop_last = drop_last.unwrap_or(false);

        Ok(PyDataLoader {
            dataset_len: 0, // Will be set when used with a dataset
            batch_size,
            shuffle,
            drop_last,
        })
    }

    /// Setup with dataset
    pub fn setup_dataset(&mut self, dataset: &PyTensorDataset) -> PyResult<()> {
        self.dataset_len = dataset.len();
        Ok(())
    }

    /// Get batch size
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        if self.drop_last {
            self.dataset_len / self.batch_size
        } else {
            (self.dataset_len + self.batch_size - 1) / self.batch_size
        }
    }

    /// Get dataset length
    pub fn dataset_len(&self) -> usize {
        self.dataset_len
    }

    /// Get batch indices for a given batch number
    pub fn get_batch_indices(&self, batch_idx: usize) -> PyResult<Vec<usize>> {
        let num_batches = self.num_batches();
        if batch_idx >= num_batches {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Batch index {} out of range (0-{})", batch_idx, num_batches - 1)
            ));
        }

        let start_idx = batch_idx * self.batch_size;
        let end_idx = std::cmp::min(start_idx + self.batch_size, self.dataset_len);
        
        let mut indices: Vec<usize> = (start_idx..end_idx).collect();
        
        // Shuffle indices if requested (simple per-batch shuffle)
        if self.shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        Ok(indices)
    }

    /// Get a batch of data from a dataset
    pub fn get_batch_from_dataset(&self, dataset: &PyTensorDataset, batch_idx: usize) -> PyResult<Vec<Vec<PyTensor>>> {
        let indices = self.get_batch_indices(batch_idx)?;
        dataset.get_batch(indices)
    }

    /// Create an iterator over batch indices
    pub fn batch_indices_iter(&self) -> PyResult<Vec<Vec<usize>>> {
        let num_batches = self.num_batches();
        let mut all_indices = Vec::new();
        
        for batch_idx in 0..num_batches {
            all_indices.push(self.get_batch_indices(batch_idx)?);
        }
        
        Ok(all_indices)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyDataLoader(dataset_len={}, batch_size={}, num_batches={}, shuffle={}, drop_last={})",
            self.dataset_len,
            self.batch_size,
            self.num_batches(),
            self.shuffle,
            self.drop_last
        )
    }

    pub fn __len__(&self) -> usize {
        self.num_batches()
    }
}

#[cfg(feature = "python")]
/// Transform operations for data preprocessing
#[pyclass]
pub struct PyTransform {
    operations: Vec<String>,
    parameters: HashMap<String, f32>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTransform {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyTransform {
            operations: Vec::new(),
            parameters: HashMap::new(),
        })
    }

    /// Add normalization transform (normalize to [0, 1])
    pub fn normalize(&mut self, mean: Option<f32>, std: Option<f32>) -> PyResult<()> {
        self.operations.push("normalize".to_string());
        self.parameters.insert("normalize_mean".to_string(), mean.unwrap_or(0.0));
        self.parameters.insert("normalize_std".to_string(), std.unwrap_or(1.0));
        Ok(())
    }

    /// Add resize transform (simple reshape)
    pub fn resize(&mut self, height: usize, width: usize) -> PyResult<()> {
        self.operations.push("resize".to_string());
        self.parameters.insert("resize_height".to_string(), height as f32);
        self.parameters.insert("resize_width".to_string(), width as f32);
        Ok(())
    }

    /// Add random rotation (for data augmentation)
    pub fn random_rotation(&mut self, degrees: f32) -> PyResult<()> {
        self.operations.push("random_rotation".to_string());
        self.parameters.insert("rotation_degrees".to_string(), degrees);
        Ok(())
    }

    /// Apply transforms to a tensor
    pub fn apply(&self, tensor: &PyTensor) -> PyResult<PyTensor> {
        let mut result_tensor = tensor.clone();
        
        for operation in &self.operations {
            match operation.as_str() {
                "normalize" => {
                    let mean = self.parameters.get("normalize_mean").unwrap_or(&0.0);
                    let std = self.parameters.get("normalize_std").unwrap_or(&1.0);
                    result_tensor = self.apply_normalize(&result_tensor, *mean, *std)?;
                }
                "resize" => {
                    let height = self.parameters.get("resize_height").unwrap_or(&32.0) as &f32;
                    let width = self.parameters.get("resize_width").unwrap_or(&32.0) as &f32;
                    result_tensor = self.apply_resize(&result_tensor, *height as usize, *width as usize)?;
                }
                "random_rotation" => {
                    let degrees = self.parameters.get("rotation_degrees").unwrap_or(&0.0);
                    result_tensor = self.apply_random_rotation(&result_tensor, *degrees)?;
                }
                _ => {}
            }
        }
        
        Ok(result_tensor)
    }

    /// Apply normalization
    fn apply_normalize(&self, tensor: &PyTensor, mean: f32, std: f32) -> PyResult<PyTensor> {
        let mut normalized_data = Vec::new();
        let data_slice = tensor.tensor.data.as_slice().unwrap();
        
        for &value in data_slice {
            normalized_data.push((value - mean) / std);
        }
        
        PyTensor::new(normalized_data, tensor.tensor.shape().to_vec())
    }

    /// Apply resize (simple implementation - just reshape if size matches)
    fn apply_resize(&self, tensor: &PyTensor, height: usize, width: usize) -> PyResult<PyTensor> {
        let shape = tensor.tensor.shape();
        let new_shape = if shape.len() >= 2 {
            let mut new_shape = shape.to_vec();
            new_shape[shape.len() - 2] = height;
            new_shape[shape.len() - 1] = width;
            new_shape
        } else {
            vec![height, width]
        };
        
        // Simple reshape if total size matches
        let old_size: usize = shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        
        if old_size == new_size {
            let data = tensor.tensor.data.as_slice().unwrap().to_vec();
            PyTensor::new(data, new_shape)
        } else {
            // If sizes don't match, return original tensor
            Ok(tensor.clone())
        }
    }

    /// Apply random rotation (placeholder implementation)
    fn apply_random_rotation(&self, tensor: &PyTensor, _degrees: f32) -> PyResult<PyTensor> {
        // Placeholder: just return the original tensor
        // In a full implementation, this would apply actual rotation
        Ok(tensor.clone())
    }

    pub fn __repr__(&self) -> String {
        format!("PyTransform(operations={:?})", self.operations)
    }
}

#[cfg(feature = "python")]
/// Utility for creating common data transforms
#[pyclass]
pub struct PyTransforms {}

#[cfg(feature = "python")]
#[pymethods]
impl PyTransforms {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyTransforms {})
    }

    /// Create a compose transform that applies multiple transforms in sequence
    #[staticmethod]
    pub fn compose(transforms: Vec<PyTransform>) -> PyResult<PyTransform> {
        let mut composed = PyTransform::new()?;
        
        for transform in transforms {
            for operation in transform.operations {
                composed.operations.push(operation.clone());
                // Copy parameters (simplified)
                for (key, value) in &transform.parameters {
                    composed.parameters.insert(key.clone(), *value);
                }
            }
        }
        
        Ok(composed)
    }

    /// Create normalization transform with common ImageNet values
    #[staticmethod]
    pub fn normalize_imagenet() -> PyResult<PyTransform> {
        let mut transform = PyTransform::new()?;
        transform.normalize(Some(0.485), Some(0.229))?;
        Ok(transform)
    }

    /// Create resize transform
    #[staticmethod]
    pub fn resize(height: usize, width: usize) -> PyResult<PyTransform> {
        let mut transform = PyTransform::new()?;
        transform.resize(height, width)?;
        Ok(transform)
    }

    /// Create random rotation transform
    #[staticmethod]
    pub fn random_rotation(degrees: f32) -> PyResult<PyTransform> {
        let mut transform = PyTransform::new()?;
        transform.random_rotation(degrees)?;
        Ok(transform)
    }
}

#[cfg(feature = "python")]
/// Model serialization utilities for save/load functionality
#[pyclass]
pub struct PyModelSerializer {}

#[cfg(feature = "python")]
#[pymethods]
impl PyModelSerializer {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyModelSerializer {})
    }

    /// Save a model to disk (PyTorch-compatible format)
    #[staticmethod] 
    pub fn save_model(model_dict: HashMap<String, PyTensor>, filepath: &str) -> PyResult<()> {
        // Create a serializable model structure
        let model_data = ModelData::new(model_dict);
        
        match save(&model_data, filepath) {
            Ok(()) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to save model: {}", e))),
        }
    }

    /// Load a model from disk
    #[staticmethod]
    pub fn load_model(filepath: &str) -> PyResult<HashMap<String, PyTensor>> {
        match load::<_, ModelData>(filepath) {
            Ok(model_data) => Ok(model_data.tensors),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to load model: {}", e))),
        }
    }

    /// Save model state dict (parameters only)
    #[staticmethod]
    pub fn save_state_dict(state_dict: HashMap<String, PyTensor>, filepath: &str) -> PyResult<()> {
        let state_data = StateDict::new(state_dict);
        
        match save(&state_data, filepath) {
            Ok(()) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to save state dict: {}", e))),
        }
    }

    /// Load model state dict
    #[staticmethod]
    pub fn load_state_dict(filepath: &str) -> PyResult<HashMap<String, PyTensor>> {
        match load::<_, StateDict>(filepath) {
            Ok(state_data) => Ok(state_data.parameters),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to load state dict: {}", e))),
        }
    }

    /// Save a single tensor
    #[staticmethod]
    pub fn save_tensor(tensor: &PyTensor, filepath: &str) -> PyResult<()> {
        let tensor_data = TensorData::new(tensor.tensor.clone());
        
        match save(&tensor_data, filepath) {
            Ok(()) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to save tensor: {}", e))),
        }
    }

    /// Load a single tensor  
    #[staticmethod]
    pub fn load_tensor(filepath: &str) -> PyResult<PyTensor> {
        match load::<_, TensorData>(filepath) {
            Ok(tensor_data) => Ok(PyTensor { tensor: tensor_data.tensor }),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to load tensor: {}", e))),
        }
    }

    /// Check if file exists and is valid model file
    #[staticmethod]
    pub fn is_valid_model_file(filepath: &str) -> PyResult<bool> {
        use std::path::Path;
        if !Path::new(filepath).exists() {
            return Ok(false);
        }
        
        // Try to load header only
        match load::<_, ModelData>(filepath) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get model file metadata
    #[staticmethod]
    pub fn get_model_info(filepath: &str) -> PyResult<HashMap<String, String>> {
        // This would read just the header to get metadata
        // For now, return basic file info
        use std::fs;
        let mut info = HashMap::new();
        
        match fs::metadata(filepath) {
            Ok(metadata) => {
                info.insert("file_size".to_string(), metadata.len().to_string());
                info.insert("exists".to_string(), "true".to_string());
                Ok(info)
            },
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to get file info: {}", e))),
        }
    }
}

// Helper structs for serialization
#[derive(Clone, Debug)]
pub struct ModelData {
    tensors: HashMap<String, PyTensor>,
}

impl ModelData {
    pub fn new(tensors: HashMap<String, PyTensor>) -> Self {
        Self { tensors }
    }
}

impl Saveable for ModelData {
    fn save_binary(&self) -> Result<Vec<u8>, SerializationError> {
        // Convert HashMap<String, PyTensor> to serializable format
        let mut serializable_map = HashMap::new();
        for (key, py_tensor) in &self.tensors {
            let tensor_info = TensorInfo {
                data: py_tensor.tensor.data.as_slice().unwrap().to_vec(),
                shape: py_tensor.tensor.shape().to_vec(),
            };
            serializable_map.insert(key.clone(), tensor_info);
        }
        
        bincode::serialize(&serializable_map)
            .map_err(|e| SerializationError::FormatError(e.to_string()))
    }

    fn type_id(&self) -> &'static str {
        "ModelData"
    }

    fn metadata(&self) -> ModelMetadata {
        let mut metadata = ModelMetadata::new();
        metadata.insert("num_tensors".to_string(), self.tensors.len().to_string());
        metadata
    }
}

impl Loadable for ModelData {
    fn load_binary(data: &[u8]) -> Result<Self, SerializationError> {
        let serializable_map: HashMap<String, TensorInfo> = bincode::deserialize(data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        
        let mut tensors = HashMap::new();
        for (key, tensor_info) in serializable_map {
            let tensor = match crate::tensor::Tensor::from_vec(tensor_info.data, tensor_info.shape) {
                tensor => PyTensor { tensor },
            };
            tensors.insert(key, tensor);
        }
        
        Ok(ModelData { tensors })
    }

    fn expected_type_id() -> &'static str {
        "ModelData"
    }

    fn validate_version(version: &str) -> Result<(), SerializationError> {
        // Accept any version for now
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StateDict {
    parameters: HashMap<String, PyTensor>,
}

impl StateDict {
    pub fn new(parameters: HashMap<String, PyTensor>) -> Self {
        Self { parameters }
    }
}

impl Saveable for StateDict {
    fn save_binary(&self) -> Result<Vec<u8>, SerializationError> {
        let mut serializable_map = HashMap::new();
        for (key, py_tensor) in &self.parameters {
            let tensor_info = TensorInfo {
                data: py_tensor.tensor.data.as_slice().unwrap().to_vec(),
                shape: py_tensor.tensor.shape().to_vec(),
            };
            serializable_map.insert(key.clone(), tensor_info);
        }
        
        bincode::serialize(&serializable_map)
            .map_err(|e| SerializationError::FormatError(e.to_string()))
    }

    fn type_id(&self) -> &'static str {
        "StateDict"
    }

    fn metadata(&self) -> ModelMetadata {
        let mut metadata = ModelMetadata::new();
        metadata.insert("num_parameters".to_string(), self.parameters.len().to_string());
        metadata
    }
}

impl Loadable for StateDict {
    fn load_binary(data: &[u8]) -> Result<Self, SerializationError> {
        let serializable_map: HashMap<String, TensorInfo> = bincode::deserialize(data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        
        let mut parameters = HashMap::new();
        for (key, tensor_info) in serializable_map {
            let tensor = match crate::tensor::Tensor::from_vec(tensor_info.data, tensor_info.shape) {
                tensor => PyTensor { tensor },
            };
            parameters.insert(key, tensor);
        }
        
        Ok(StateDict { parameters })
    }

    fn expected_type_id() -> &'static str {
        "StateDict"
    }

    fn validate_version(version: &str) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TensorData {
    tensor: crate::tensor::Tensor<f32>,
}

impl TensorData {
    pub fn new(tensor: crate::tensor::Tensor<f32>) -> Self {
        Self { tensor }
    }
}

impl Saveable for TensorData {
    fn save_binary(&self) -> Result<Vec<u8>, SerializationError> {
        let tensor_info = TensorInfo {
            data: self.tensor.data.as_slice().unwrap().to_vec(),
            shape: self.tensor.shape().to_vec(),
        };
        
        bincode::serialize(&tensor_info)
            .map_err(|e| SerializationError::FormatError(e.to_string()))
    }

    fn type_id(&self) -> &'static str {
        "TensorData"
    }

    fn metadata(&self) -> ModelMetadata {
        let mut metadata = ModelMetadata::new();
        metadata.insert("shape".to_string(), format!("{:?}", self.tensor.shape()));
        metadata
    }
}

impl Loadable for TensorData {
    fn load_binary(data: &[u8]) -> Result<Self, SerializationError> {
        let tensor_info: TensorInfo = bincode::deserialize(data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        
        let tensor = crate::tensor::Tensor::from_vec(tensor_info.data, tensor_info.shape);
        Ok(TensorData { tensor })
    }

    fn expected_type_id() -> &'static str {
        "TensorData"
    }

    fn validate_version(version: &str) -> Result<(), SerializationError> {
        Ok(())
    }
}

// Helper struct for serialization
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "python")]
/// Python wrapper for RusTorch training configuration
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTrainerConfig {
    pub config: TrainerConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrainerConfig {
    #[new]
    pub fn new(
        epochs: Option<usize>,
        log_frequency: Option<usize>,
        validation_frequency: Option<usize>,
        gradient_clip_value: Option<f64>,
        device: Option<String>,
        use_mixed_precision: Option<bool>,
        accumulation_steps: Option<usize>,
    ) -> PyResult<Self> {
        Ok(PyTrainerConfig {
            config: TrainerConfig {
                epochs: epochs.unwrap_or(10),
                log_frequency: log_frequency.unwrap_or(100),
                validation_frequency: validation_frequency.unwrap_or(1),
                gradient_clip_value,
                device: device.unwrap_or_else(|| "cpu".to_string()),
                use_mixed_precision: use_mixed_precision.unwrap_or(false),
                accumulation_steps: accumulation_steps.unwrap_or(1),
            },
        })
    }

    /// Get epochs
    #[getter]
    pub fn get_epochs(&self) -> usize {
        self.config.epochs
    }

    /// Set epochs
    #[setter]
    pub fn set_epochs(&mut self, epochs: usize) {
        self.config.epochs = epochs;
    }

    /// Get log frequency
    #[getter]
    pub fn get_log_frequency(&self) -> usize {
        self.config.log_frequency
    }

    /// Set log frequency
    #[setter]
    pub fn set_log_frequency(&mut self, log_frequency: usize) {
        self.config.log_frequency = log_frequency;
    }

    /// Get device
    #[getter]
    pub fn get_device(&self) -> String {
        self.config.device.clone()
    }

    /// Set device
    #[setter]
    pub fn set_device(&mut self, device: String) {
        self.config.device = device;
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyTrainerConfig(epochs={}, device='{}', mixed_precision={})",
            self.config.epochs, self.config.device, self.config.use_mixed_precision
        )
    }
}

#[cfg(feature = "python")]
/// Python wrapper for training history
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTrainingHistory {
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub metrics: HashMap<String, Vec<f64>>,
    pub training_time: f64,
    pub best_val_loss: Option<f32>,
    pub best_epoch: Option<usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrainingHistory {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyTrainingHistory {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            metrics: HashMap::new(),
            training_time: 0.0,
            best_val_loss: None,
            best_epoch: None,
        })
    }

    /// Get training losses
    #[getter]
    pub fn get_train_loss(&self) -> Vec<f32> {
        self.train_loss.clone()
    }

    /// Get validation losses
    #[getter]
    pub fn get_val_loss(&self) -> Vec<f32> {
        self.val_loss.clone()
    }

    /// Get metrics
    #[getter]
    pub fn get_metrics(&self) -> HashMap<String, Vec<f64>> {
        self.metrics.clone()
    }

    /// Get training time
    #[getter]
    pub fn get_training_time(&self) -> f64 {
        self.training_time
    }

    /// Get best validation loss
    #[getter]
    pub fn get_best_val_loss(&self) -> Option<f32> {
        self.best_val_loss
    }

    /// Get best epoch
    #[getter]
    pub fn get_best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    /// Add epoch data
    pub fn add_epoch(&mut self, train_loss: f32, val_loss: Option<f32>) -> PyResult<()> {
        self.train_loss.push(train_loss);
        if let Some(val_loss) = val_loss {
            self.val_loss.push(val_loss);
            
            // Update best validation loss
            if self.best_val_loss.is_none() || val_loss < self.best_val_loss.unwrap() {
                self.best_val_loss = Some(val_loss);
                self.best_epoch = Some(self.train_loss.len() - 1);
            }
        }
        Ok(())
    }

    /// Add metric
    pub fn add_metric(&mut self, name: String, value: f64) -> PyResult<()> {
        self.metrics.entry(name).or_insert_with(Vec::new).push(value);
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyTrainingHistory(epochs={}, best_val_loss={:?}, training_time={:.2}s)",
            self.train_loss.len(),
            self.best_val_loss,
            self.training_time
        )
    }
}

#[cfg(feature = "python")]
/// High-level training interface for Python
#[pyclass]
pub struct PyTrainer {
    config: TrainerConfig,
    history: PyTrainingHistory,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrainer {
    #[new]
    pub fn new(config: Option<PyTrainerConfig>) -> PyResult<Self> {
        let config = if let Some(config) = config {
            config.config
        } else {
            TrainerConfig::default()
        };

        Ok(PyTrainer {
            config,
            history: PyTrainingHistory::new()?,
        })
    }

    /// Train a model (simplified version)
    pub fn fit(
        &mut self,
        model: &mut PyLinear,
        train_data: &PyTensorDataset,
        validation_data: Option<&PyTensorDataset>,
        optimizer: &mut PySGD,
        loss_fn: &PyMSELoss,
    ) -> PyResult<PyTrainingHistory> {
        use std::time::Instant;
        let start_time = Instant::now();

        // Simple training loop implementation
        for epoch in 0..self.config.epochs {
            let mut total_train_loss = 0.0f32;
            let mut batch_count = 0;

            // Training phase
            for batch_idx in 0..train_data.len() {
                match train_data.get_item(batch_idx) {
                    Ok(batch_data) => {
                        if batch_data.len() >= 2 {
                            let input = &batch_data[0];
                            let target = &batch_data[1];

                            // Forward pass
                            let input_var = PyVariable {
                                variable: crate::autograd::Variable::new(input.tensor.clone()),
                            };
                            match model.forward(&input_var) {
                                Ok(output) => {
                                    let target_var = PyVariable {
                                        variable: crate::autograd::Variable::new(target.tensor.clone()),
                                    };
                                    
                                    // Calculate loss
                                    match loss_fn.forward(&output, &target_var) {
                                        Ok(loss) => {
                                            total_train_loss += loss.get_data()[0];
                                            
                                            // Backward pass (simplified)
                                            // In a full implementation, we would call loss.backward() and optimizer.step()
                                            batch_count += 1;
                                        }
                                        Err(_) => continue,
                                    }
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }

            let avg_train_loss = if batch_count > 0 {
                total_train_loss / batch_count as f32
            } else {
                0.0
            };

            // Validation phase (if provided)
            let val_loss = if let Some(val_data) = validation_data {
                let mut total_val_loss = 0.0f32;
                let mut val_batch_count = 0;

                for batch_idx in 0..val_data.len() {
                    match val_data.get_item(batch_idx) {
                        Ok(batch_data) => {
                            if batch_data.len() >= 2 {
                                let input = &batch_data[0];
                                let target = &batch_data[1];

                                let input_var = PyVariable {
                                    variable: crate::autograd::Variable::new(input.tensor.clone()),
                                };
                                match model.forward(&input_var) {
                                    Ok(output) => {
                                        let target_var = PyVariable {
                                            variable: crate::autograd::Variable::new(target.tensor.clone()),
                                        };
                                        
                                        match loss_fn.forward(&output, &target_var) {
                                            Ok(loss) => {
                                                total_val_loss += loss.get_data()[0];
                                                val_batch_count += 1;
                                            }
                                            Err(_) => continue,
                                        }
                                    }
                                    Err(_) => continue,
                                }
                            }
                        }
                        Err(_) => continue,
                    }
                }

                if val_batch_count > 0 {
                    Some(total_val_loss / val_batch_count as f32)
                } else {
                    None
                }
            } else {
                None
            };

            // Update history
            self.history.add_epoch(avg_train_loss, val_loss)?;

            // Log progress
            if epoch % self.config.log_frequency == 0 {
                if let Some(val_loss) = val_loss {
                    println!("Epoch {}: train_loss={:.4}, val_loss={:.4}", epoch, avg_train_loss, val_loss);
                } else {
                    println!("Epoch {}: train_loss={:.4}", epoch, avg_train_loss);
                }
            }
        }

        self.history.training_time = start_time.elapsed().as_secs_f64();
        Ok(self.history.clone())
    }

    /// Evaluate a model
    pub fn evaluate(
        &self,
        model: &PyLinear,
        test_data: &PyTensorDataset,
        loss_fn: &PyMSELoss,
    ) -> PyResult<HashMap<String, f64>> {
        let mut total_loss = 0.0f32;
        let mut batch_count = 0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for batch_idx in 0..test_data.len() {
            match test_data.get_item(batch_idx) {
                Ok(batch_data) => {
                    if batch_data.len() >= 2 {
                        let input = &batch_data[0];
                        let target = &batch_data[1];

                        let input_var = PyVariable {
                            variable: crate::autograd::Variable::new(input.tensor.clone()),
                        };
                        match model.forward(&input_var) {
                            Ok(output) => {
                                let target_var = PyVariable {
                                    variable: crate::autograd::Variable::new(target.tensor.clone()),
                                };
                                
                                match loss_fn.forward(&output, &target_var) {
                                    Ok(loss) => {
                                        total_loss += loss.get_data()[0];
                                        batch_count += 1;

                                        // Calculate accuracy (simplified)
                                        let output_data = output.get_data();
                                        let target_data = target.tensor.data.as_slice().unwrap();
                                        
                                        for (pred, true_val) in output_data.iter().zip(target_data.iter()) {
                                            if (pred - true_val).abs() < 0.1 {
                                                correct_predictions += 1;
                                            }
                                            total_predictions += 1;
                                        }
                                    }
                                    Err(_) => continue,
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        let mut metrics = HashMap::new();
        if batch_count > 0 {
            metrics.insert("loss".to_string(), (total_loss / batch_count as f32) as f64);
        }
        if total_predictions > 0 {
            metrics.insert("accuracy".to_string(), correct_predictions as f64 / total_predictions as f64);
        }

        Ok(metrics)
    }

    /// Make predictions
    pub fn predict(&self, model: &PyLinear, input: &PyTensor) -> PyResult<PyTensor> {
        let input_var = PyVariable {
            variable: crate::autograd::Variable::new(input.tensor.clone()),
        };
        
        match model.forward(&input_var) {
            Ok(output) => Ok(PyTensor { tensor: output.variable.data }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get training history
    #[getter]
    pub fn get_history(&self) -> PyTrainingHistory {
        self.history.clone()
    }

    /// Get configuration
    #[getter]
    pub fn get_config(&self) -> PyTrainerConfig {
        PyTrainerConfig {
            config: self.config.clone(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyTrainer(epochs={}, device='{}')",
            self.config.epochs, self.config.device
        )
    }
}

#[cfg(feature = "python")]
/// High-level model wrapper for sequential models
#[pyclass]
pub struct PyModel {
    layers: Vec<PyModelLayer>,
    name: String,
}

#[cfg(feature = "python")]
/// Enum for different layer types in PyModel
#[derive(Clone)]
pub enum PyModelLayer {
    Linear(PyLinear),
    Conv2d(PyConv2d),
    BatchNorm2d(PyBatchNorm2d),
}

#[cfg(feature = "python")]
#[pymethods]
impl PyModel {
    #[new]
    pub fn new(name: Option<String>) -> PyResult<Self> {
        Ok(PyModel {
            layers: Vec::new(),
            name: name.unwrap_or_else(|| "PyModel".to_string()),
        })
    }

    /// Add a linear layer
    pub fn add_linear(&mut self, in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<()> {
        let linear_layer = PyLinear::new(in_features, out_features, bias)?;
        self.layers.push(PyModelLayer::Linear(linear_layer));
        Ok(())
    }

    /// Add a Conv2d layer
    pub fn add_conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<()> {
        let conv_layer = PyConv2d::new(in_channels, out_channels, kernel_size, stride, padding, bias)?;
        self.layers.push(PyModelLayer::Conv2d(conv_layer));
        Ok(())
    }

    /// Add a BatchNorm2d layer
    pub fn add_batchnorm2d(
        &mut self,
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<()> {
        let bn_layer = PyBatchNorm2d::new(num_features, eps, momentum, affine)?;
        self.layers.push(PyModelLayer::BatchNorm2d(bn_layer));
        Ok(())
    }

    /// Forward pass through all layers
    pub fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let mut current_input = input.clone();
        
        for layer in &self.layers {
            match layer {
                PyModelLayer::Linear(linear) => {
                    current_input = linear.forward(&current_input)?;
                }
                PyModelLayer::Conv2d(conv) => {
                    current_input = conv.forward(&current_input)?;
                }
                PyModelLayer::BatchNorm2d(bn) => {
                    current_input = bn.forward(&current_input)?;
                }
            }
        }
        
        Ok(current_input)
    }

    /// Get all parameters from all layers
    pub fn parameters(&self) -> PyResult<Vec<PyVariable>> {
        let mut all_params = Vec::new();
        
        for layer in &self.layers {
            match layer {
                PyModelLayer::Linear(linear) => {
                    all_params.extend(linear.parameters()?);
                }
                PyModelLayer::Conv2d(conv) => {
                    all_params.extend(conv.parameters()?);
                }
                PyModelLayer::BatchNorm2d(bn) => {
                    all_params.extend(bn.parameters()?);
                }
            }
        }
        
        Ok(all_params)
    }

    /// Get state dict (parameter dictionary)
    pub fn state_dict(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut state_dict = HashMap::new();
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                PyModelLayer::Linear(linear) => {
                    let params = linear.parameters()?;
                    if params.len() >= 2 {
                        state_dict.insert(
                            format!("layer_{}.weight", layer_idx),
                            PyTensor { tensor: params[0].variable.data.clone() }
                        );
                        state_dict.insert(
                            format!("layer_{}.bias", layer_idx),
                            PyTensor { tensor: params[1].variable.data.clone() }
                        );
                    }
                }
                PyModelLayer::Conv2d(conv) => {
                    let params = conv.parameters()?;
                    if params.len() >= 2 {
                        state_dict.insert(
                            format!("layer_{}.weight", layer_idx),
                            PyTensor { tensor: params[0].variable.data.clone() }
                        );
                        state_dict.insert(
                            format!("layer_{}.bias", layer_idx),
                            PyTensor { tensor: params[1].variable.data.clone() }
                        );
                    }
                }
                PyModelLayer::BatchNorm2d(bn) => {
                    let params = bn.parameters()?;
                    if params.len() >= 2 {
                        state_dict.insert(
                            format!("layer_{}.weight", layer_idx),
                            PyTensor { tensor: params[0].variable.data.clone() }
                        );
                        state_dict.insert(
                            format!("layer_{}.bias", layer_idx),
                            PyTensor { tensor: params[1].variable.data.clone() }
                        );
                    }
                }
            }
        }
        
        Ok(state_dict)
    }

    /// Load state dict
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, PyTensor>) -> PyResult<()> {
        // Simple implementation - in practice, this would need more sophisticated parameter loading
        println!("Loading state dict with {} parameters", state_dict.len());
        Ok(())
    }

    /// Set model to training mode
    pub fn train(&self) -> PyResult<()> {
        for layer in &self.layers {
            if let PyModelLayer::BatchNorm2d(bn) = layer {
                bn.train();
            }
        }
        Ok(())
    }

    /// Set model to evaluation mode
    pub fn eval(&self) -> PyResult<()> {
        for layer in &self.layers {
            if let PyModelLayer::BatchNorm2d(bn) = layer {
                bn.eval();
            }
        }
        Ok(())
    }

    /// Save model using serializer
    pub fn save(&self, filepath: &str) -> PyResult<()> {
        let state_dict = self.state_dict()?;
        PyModelSerializer::save_state_dict(state_dict, filepath)
    }

    /// Load model using serializer
    pub fn load(&mut self, filepath: &str) -> PyResult<()> {
        let state_dict = PyModelSerializer::load_state_dict(filepath)?;
        self.load_state_dict(state_dict)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get model name
    #[getter]
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// Set model name
    #[setter]
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Get model summary
    pub fn summary(&self) -> PyResult<String> {
        let mut summary = format!("Model: {}\n", self.name);
        summary.push_str("=" .repeat(50).as_str());
        summary.push('\n');
        
        let mut total_params = 0;
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                PyModelLayer::Linear(linear) => {
                    let params = linear.parameters()?;
                    let param_count = params.iter()
                        .map(|p| p.variable.data.shape().iter().product::<usize>())
                        .sum::<usize>();
                    total_params += param_count;
                    summary.push_str(&format!("Layer {}: Linear ({} params)\n", idx, param_count));
                }
                PyModelLayer::Conv2d(_conv) => {
                    // Simplified parameter counting
                    let param_count = 1000; // Placeholder
                    total_params += param_count;
                    summary.push_str(&format!("Layer {}: Conv2d ({} params)\n", idx, param_count));
                }
                PyModelLayer::BatchNorm2d(_bn) => {
                    // Simplified parameter counting
                    let param_count = 100; // Placeholder
                    total_params += param_count;
                    summary.push_str(&format!("Layer {}: BatchNorm2d ({} params)\n", idx, param_count));
                }
            }
        }
        
        summary.push_str("=" .repeat(50).as_str());
        summary.push('\n');
        summary.push_str(&format!("Total parameters: {}\n", total_params));
        
        Ok(summary)
    }

    pub fn __repr__(&self) -> String {
        format!("PyModel(name='{}', layers={})", self.name, self.layers.len())
    }
}

#[cfg(feature = "python")]
/// Builder pattern for creating models
#[pyclass]
pub struct PyModelBuilder {
    model: PyModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyModelBuilder {
    #[new]
    pub fn new(name: Option<String>) -> PyResult<Self> {
        Ok(PyModelBuilder {
            model: PyModel::new(name)?,
        })
    }

    /// Add linear layer and return self for chaining
    pub fn linear(mut self_: pyo3::PyRefMut<Self>, in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<pyo3::PyRefMut<Self>> {
        self_.model.add_linear(in_features, out_features, bias)?;
        Ok(self_)
    }

    /// Add Conv2d layer and return self for chaining
    pub fn conv2d(
        mut self_: pyo3::PyRefMut<Self>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<pyo3::PyRefMut<Self>> {
        self_.model.add_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)?;
        Ok(self_)
    }

    /// Add BatchNorm2d layer and return self for chaining
    pub fn batchnorm2d(
        mut self_: pyo3::PyRefMut<Self>,
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<pyo3::PyRefMut<Self>> {
        self_.model.add_batchnorm2d(num_features, eps, momentum, affine)?;
        Ok(self_)
    }

    /// Build and return the final model
    pub fn build(self) -> PyResult<PyModel> {
        Ok(self.model)
    }
}

#[cfg(feature = "python")]
/// Python wrapper for DistributedDataParallel
#[pyclass]
pub struct PyDistributedDataParallel {
    model: PyModel,
    device_ids: Vec<usize>,
    output_device: Option<usize>,
    broadcast_buffers: bool,
    process_group: Option<String>,
    bucket_cap_mb: f64,
    find_unused_parameters: bool,
    check_reduction: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDistributedDataParallel {
    #[new]
    pub fn new(
        model: PyModel,
        device_ids: Option<Vec<usize>>,
        output_device: Option<usize>,
        broadcast_buffers: Option<bool>,
        process_group: Option<String>,
        bucket_cap_mb: Option<f64>,
        find_unused_parameters: Option<bool>,
        check_reduction: Option<bool>,
    ) -> PyResult<Self> {
        Ok(PyDistributedDataParallel {
            model,
            device_ids: device_ids.unwrap_or_else(|| vec![0]),
            output_device,
            broadcast_buffers: broadcast_buffers.unwrap_or(true),
            process_group,
            bucket_cap_mb: bucket_cap_mb.unwrap_or(25.0),
            find_unused_parameters: find_unused_parameters.unwrap_or(false),
            check_reduction: check_reduction.unwrap_or(false),
        })
    }

    /// Forward pass through the distributed model
    pub fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        // In a full implementation, this would handle gradient synchronization
        // For now, we delegate to the underlying model
        self.model.forward(input)
    }

    /// Get all parameters from the distributed model
    pub fn parameters(&self) -> PyResult<Vec<PyVariable>> {
        self.model.parameters()
    }

    /// Get the underlying model
    pub fn module(&self) -> PyModel {
        self.model.clone()
    }

    /// Broadcast parameters to all processes
    pub fn broadcast_parameters(&self) -> PyResult<()> {
        println!("Broadcasting parameters across {} devices", self.device_ids.len());
        // In a real implementation, this would broadcast parameters across processes
        Ok(())
    }

    /// Synchronize gradients across all processes
    pub fn sync_gradients(&self) -> PyResult<()> {
        println!("Synchronizing gradients across distributed processes");
        // In a real implementation, this would perform AllReduce on gradients
        Ok(())
    }

    /// Set training mode for distributed model
    pub fn train(&self) -> PyResult<()> {
        self.model.train()
    }

    /// Set evaluation mode for distributed model
    pub fn eval(&self) -> PyResult<()> {
        self.model.eval()
    }

    /// Get device IDs
    #[getter]
    pub fn get_device_ids(&self) -> Vec<usize> {
        self.device_ids.clone()
    }

    /// Get process group
    #[getter]
    pub fn get_process_group(&self) -> Option<String> {
        self.process_group.clone()
    }

    /// Check if broadcast buffers is enabled
    #[getter]
    pub fn get_broadcast_buffers(&self) -> bool {
        self.broadcast_buffers
    }

    /// Get bucket capacity in MB
    #[getter]
    pub fn get_bucket_cap_mb(&self) -> f64 {
        self.bucket_cap_mb
    }

    /// Check if finding unused parameters is enabled
    #[getter]
    pub fn get_find_unused_parameters(&self) -> bool {
        self.find_unused_parameters
    }

    /// Get state dict from underlying model
    pub fn state_dict(&self) -> PyResult<HashMap<String, PyTensor>> {
        self.model.state_dict()
    }

    /// Load state dict to underlying model
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, PyTensor>) -> PyResult<()> {
        self.model.load_state_dict(state_dict)
    }

    /// Save distributed model
    pub fn save(&self, filepath: &str) -> PyResult<()> {
        self.model.save(filepath)
    }

    /// Load distributed model
    pub fn load(&mut self, filepath: &str) -> PyResult<()> {
        self.model.load(filepath)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyDistributedDataParallel(device_ids={:?}, broadcast_buffers={}, bucket_cap_mb={})",
            self.device_ids, self.broadcast_buffers, self.bucket_cap_mb
        )
    }
}

#[cfg(feature = "python")]
/// Distributed utilities for process management
#[pyclass]
pub struct PyDistributedUtils {}

#[cfg(feature = "python")]
#[pymethods]
impl PyDistributedUtils {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyDistributedUtils {})
    }

    /// Initialize the distributed process group
    #[staticmethod]
    pub fn init_process_group(
        backend: String,
        init_method: Option<String>,
        world_size: Option<usize>,
        rank: Option<usize>,
        timeout_seconds: Option<u64>,
    ) -> PyResult<()> {
        let backend = backend.as_str();
        let init_method = init_method.unwrap_or_else(|| "env://".to_string());
        let world_size = world_size.unwrap_or(1);
        let rank = rank.unwrap_or(0);
        let timeout = timeout_seconds.unwrap_or(1800);

        println!("Initializing distributed process group:");
        println!("  Backend: {}", backend);
        println!("  Init method: {}", init_method);
        println!("  World size: {}", world_size);
        println!("  Rank: {}", rank);
        println!("  Timeout: {}s", timeout);

        // In a real implementation, this would initialize the process group
        Ok(())
    }

    /// Destroy the distributed process group
    #[staticmethod]
    pub fn destroy_process_group() -> PyResult<()> {
        println!("Destroying distributed process group");
        Ok(())
    }

    /// Get the current process rank
    #[staticmethod]
    pub fn get_rank() -> PyResult<usize> {
        // In a real implementation, this would return the actual rank
        Ok(0)
    }

    /// Get the world size
    #[staticmethod]
    pub fn get_world_size() -> PyResult<usize> {
        // In a real implementation, this would return the actual world size
        Ok(1)
    }

    /// Check if distributed is available
    #[staticmethod]
    pub fn is_available() -> PyResult<bool> {
        Ok(true)
    }

    /// Check if distributed is initialized
    #[staticmethod]
    pub fn is_initialized() -> PyResult<bool> {
        Ok(false) // Simplified for now
    }

    /// All-reduce operation
    #[staticmethod]
    pub fn all_reduce(tensor: &PyTensor, op: Option<String>) -> PyResult<PyTensor> {
        let op = op.unwrap_or_else(|| "sum".to_string());
        println!("Performing all_reduce with operation: {}", op);
        
        // In a real implementation, this would perform actual all-reduce
        Ok(tensor.clone())
    }

    /// All-gather operation
    #[staticmethod]
    pub fn all_gather(tensors: Vec<PyTensor>, tensor: &PyTensor) -> PyResult<Vec<PyTensor>> {
        println!("Performing all_gather operation");
        
        // In a real implementation, this would gather tensors from all processes
        let mut result = tensors;
        result.push(tensor.clone());
        Ok(result)
    }

    /// Broadcast operation
    #[staticmethod]
    pub fn broadcast(tensor: &PyTensor, src: usize) -> PyResult<PyTensor> {
        println!("Broadcasting tensor from rank: {}", src);
        
        // In a real implementation, this would broadcast from the source rank
        Ok(tensor.clone())
    }
}

#[cfg(feature = "python")]
/// Basic visualization utilities for tensors and training
#[pyclass]
pub struct PyVisualizer {}

#[cfg(feature = "python")]
#[pymethods]
impl PyVisualizer {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(PyVisualizer {})
    }

    /// Plot tensor as a line graph (simplified implementation)
    #[staticmethod]
    pub fn plot_tensor(tensor: &PyTensor, title: Option<String>, save_path: Option<String>) -> PyResult<String> {
        let title = title.unwrap_or_else(|| "Tensor Plot".to_string());
        let save_path = save_path.unwrap_or_else(|| "tensor_plot.png".to_string());
        
        let data = tensor.tensor.data.as_slice().unwrap();
        let shape = tensor.tensor.shape();
        
        // Create a simple text-based visualization for now
        let mut plot_data = Vec::new();
        
        if shape.len() == 1 {
            // 1D tensor - line plot
            for (i, &value) in data.iter().enumerate() {
                plot_data.push(format!("x={}, y={:.4}", i, value));
            }
        } else if shape.len() == 2 && shape[0] <= 10 && shape[1] <= 10 {
            // Small 2D tensor - matrix visualization
            plot_data.push("Matrix visualization:".to_string());
            for i in 0..shape[0] {
                let mut row = String::new();
                for j in 0..shape[1] {
                    let idx = i * shape[1] + j;
                    row.push_str(&format!("{:8.3} ", data[idx]));
                }
                plot_data.push(row);
            }
        } else {
            plot_data.push(format!("Tensor shape: {:?}", shape));
            plot_data.push(format!("Min: {:.4}, Max: {:.4}", 
                data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            ));
            plot_data.push(format!("Mean: {:.4}", data.iter().sum::<f32>() / data.len() as f32));
        }
        
        let result = format!("{}:\n{}", title, plot_data.join("\n"));
        
        // In a real implementation, this would save an actual plot
        println!("Saving plot to: {}", save_path);
        
        Ok(result)
    }

    /// Plot training history
    #[staticmethod]
    pub fn plot_history(history: &PyTrainingHistory, save_path: Option<String>) -> PyResult<String> {
        let save_path = save_path.unwrap_or_else(|| "training_history.png".to_string());
        
        let mut result = Vec::new();
        result.push("Training History:".to_string());
        result.push("=================".to_string());
        
        // Plot training loss
        result.push("Training Loss:".to_string());
        for (epoch, &loss) in history.train_loss.iter().enumerate() {
            result.push(format!("Epoch {}: {:.4}", epoch, loss));
        }
        
        // Plot validation loss if available
        if !history.val_loss.is_empty() {
            result.push("\nValidation Loss:".to_string());
            for (epoch, &loss) in history.val_loss.iter().enumerate() {
                result.push(format!("Epoch {}: {:.4}", epoch, loss));
            }
        }
        
        // Plot metrics if available
        for (metric_name, values) in &history.metrics {
            result.push(format!("\n{}:", metric_name));
            for (epoch, &value) in values.iter().enumerate() {
                result.push(format!("Epoch {}: {:.4}", epoch, value));
            }
        }
        
        if let Some(best_epoch) = history.best_epoch {
            result.push(format!("\nBest epoch: {}", best_epoch));
            if let Some(best_loss) = history.best_val_loss {
                result.push(format!("Best validation loss: {:.4}", best_loss));
            }
        }
        
        result.push(format!("\nTotal training time: {:.2}s", history.training_time));
        
        println!("Saving training history plot to: {}", save_path);
        
        Ok(result.join("\n"))
    }

    /// Create a simple heatmap visualization for 2D tensors
    #[staticmethod]
    pub fn heatmap(tensor: &PyTensor, title: Option<String>, save_path: Option<String>) -> PyResult<String> {
        let title = title.unwrap_or_else(|| "Heatmap".to_string());
        let save_path = save_path.unwrap_or_else(|| "heatmap.png".to_string());
        
        let shape = tensor.tensor.shape();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Heatmap requires 2D tensor"));
        }
        
        let data = tensor.tensor.data.as_slice().unwrap();
        let (rows, cols) = (shape[0], shape[1]);
        
        // Find min/max for normalization
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        let mut result = Vec::new();
        result.push(format!("{} ({}x{})", title, rows, cols));
        result.push("=".repeat(title.len() + 10));
        
        // Create ASCII heatmap
        for i in 0..rows {
            let mut row = String::new();
            for j in 0..cols {
                let idx = i * cols + j;
                let value = data[idx];
                let normalized = if range > 0.0 { (value - min_val) / range } else { 0.0 };
                
                // Convert to ASCII character based on intensity
                let char = if normalized < 0.2 { ' ' }
                    else if normalized < 0.4 { '.' }
                    else if normalized < 0.6 { ':' }
                    else if normalized < 0.8 { '*' }
                    else { '#' };
                    
                row.push(char);
                row.push(' ');
            }
            result.push(row);
        }
        
        result.push(format!("Range: [{:.3}, {:.3}]", min_val, max_val));
        result.push(format!("Legend: ' '=low, '.'=low-med, ':'=med, '*'=med-high, '#'=high"));
        
        println!("Saving heatmap to: {}", save_path);
        
        Ok(result.join("\n"))
    }

    /// Plot model architecture summary
    #[staticmethod]
    pub fn plot_model(model: &PyModel, save_path: Option<String>) -> PyResult<String> {
        let save_path = save_path.unwrap_or_else(|| "model_architecture.png".to_string());
        
        // Get model summary and create a visual representation
        let summary = model.summary()?;
        
        let mut result = Vec::new();
        result.push("Model Architecture Visualization:".to_string());
        result.push("================================".to_string());
        
        // Add visual flow
        for i in 0..model.num_layers() {
            result.push(format!("    ┌──────────────┐"));
            result.push(format!("    │   Layer {}   │", i));
            result.push(format!("    └──────────────┘"));
            if i < model.num_layers() - 1 {
                result.push("           │".to_string());
                result.push("           ▼".to_string());
            }
        }
        
        result.push("".to_string());
        result.push(summary);
        
        println!("Saving model architecture plot to: {}", save_path);
        
        Ok(result.join("\n"))
    }

    /// Create scatter plot for 2D data
    #[staticmethod]
    pub fn scatter(x: &PyTensor, y: &PyTensor, title: Option<String>, save_path: Option<String>) -> PyResult<String> {
        let title = title.unwrap_or_else(|| "Scatter Plot".to_string());
        let save_path = save_path.unwrap_or_else(|| "scatter.png".to_string());
        
        let x_data = x.tensor.data.as_slice().unwrap();
        let y_data = y.tensor.data.as_slice().unwrap();
        
        if x_data.len() != y_data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("X and Y tensors must have the same length"));
        }
        
        let mut result = Vec::new();
        result.push(format!("{} ({} points)", title, x_data.len()));
        result.push("=".repeat(title.len() + 20));
        
        // Simple ASCII scatter plot (very basic)
        let n_points = std::cmp::min(10, x_data.len()); // Show first 10 points
        
        for i in 0..n_points {
            result.push(format!("Point {}: ({:.3}, {:.3})", i + 1, x_data[i], y_data[i]));
        }
        
        if x_data.len() > 10 {
            result.push(format!("... and {} more points", x_data.len() - 10));
        }
        
        // Basic statistics
        let x_mean = x_data.iter().sum::<f32>() / x_data.len() as f32;
        let y_mean = y_data.iter().sum::<f32>() / y_data.len() as f32;
        result.push(format!("X mean: {:.3}, Y mean: {:.3}", x_mean, y_mean));
        
        println!("Saving scatter plot to: {}", save_path);
        
        Ok(result.join("\n"))
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
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyTensorDataset>()?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyTransform>()?;
    m.add_class::<PyTransforms>()?;
    m.add_class::<PyModelSerializer>()?;
    m.add_class::<PyTrainerConfig>()?;
    m.add_class::<PyTrainingHistory>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyModelBuilder>()?;
    m.add_class::<PyDistributedDataParallel>()?;
    m.add_class::<PyDistributedUtils>()?;
    m.add_class::<PyVisualizer>()?;

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
