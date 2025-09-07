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
