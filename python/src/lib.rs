use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyAny};
use rustorch::tensor::core::Tensor as RustTensor;
use rustorch::autograd::Variable as RustVariable;
use rustorch::nn::Linear as RustLinear;
use rustorch::nn::Module;
use rustorch::nn::loss::{MSELoss as RustMSELoss, CrossEntropyLoss as RustCrossEntropyLoss, Loss};
use rustorch::nn::activation::{ReLU as RustReLU, sigmoid, tanh};
use rustorch::nn::BatchNorm1d as RustBatchNorm1d;
use rustorch::nn::BatchNorm2d as RustBatchNorm2d;
use rustorch::nn::Dropout as RustDropout;
use rustorch::nn::Conv2d as RustConv2d;
use rustorch::nn::MaxPool2d as RustMaxPool2d;
use rustorch::error::RusTorchError;

/// Convert RusTorchError to PyErr for Python bindings
fn rustorch_error_to_pyerr(err: RusTorchError) -> PyErr {
    match err {
        RusTorchError::ShapeMismatch { expected, actual } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: expected {:?}, got {:?}", expected, actual
            ))
        },
        RusTorchError::TensorOp { message, .. } => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor operation failed: {}", message))
        },
        RusTorchError::Device { device, message } => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Device error on {}: {}", device, message))
        },
        _ => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("RusTorch error: {}", err))
        }
    }
}

/// Macro to convert RusTorchError results to PyResult
macro_rules! map_rustorch_err {
    ($result:expr) => {
        $result.map_err(rustorch_error_to_pyerr)
    };
}

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
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: format!("Data length {} does not match shape {:?} (expected {})",
                               data.len(), shape, expected_len),
                source: None,
            }));
        }

        if shape.is_empty() {
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: "Shape cannot be empty".to_string(),
                source: None,
            }));
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

    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        // Check dimensions for matrix multiplication
        let self_shape = self.inner.shape();
        let other_shape = other.inner.shape();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: "Matrix multiplication requires at least 2D tensors".to_string(),
                source: None,
            }));
        }

        let self_rows = self_shape[self_shape.len() - 2];
        let self_cols = self_shape[self_shape.len() - 1];
        let other_rows = other_shape[other_shape.len() - 2];
        let other_cols = other_shape[other_shape.len() - 1];

        if self_cols != other_rows {
            return Err(rustorch_error_to_pyerr(RusTorchError::ShapeMismatch {
                expected: vec![self_rows, other_rows],
                actual: vec![self_rows, self_cols, other_rows, other_cols],
            }));
        }

        let result = map_rustorch_err!(self.inner.matmul(&other.inner))?;
        Ok(PyTensor { inner: result })
    }

    /// Transpose tensor (swap last two dimensions)
    fn transpose(&self) -> PyResult<PyTensor> {
        let shape = self.inner.shape();
        if shape.len() < 2 {
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: "Transpose requires at least 2D tensor".to_string(),
                source: None,
            }));
        }

        let result = map_rustorch_err!(self.inner.transpose())?;
        Ok(PyTensor { inner: result })
    }

    /// Reshape tensor to new shape
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyTensor> {
        // Validate new shape
        let current_numel = self.inner.numel();
        let new_numel: usize = new_shape.iter().product();

        if current_numel != new_numel {
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: format!("Cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                               current_numel, new_shape, new_numel),
                source: None,
            }));
        }

        if new_shape.is_empty() || new_shape.iter().any(|&dim| dim == 0) {
            return Err(rustorch_error_to_pyerr(RusTorchError::TensorOp {
                message: "Invalid shape: dimensions must be positive".to_string(),
                source: None,
            }));
        }

        let result = map_rustorch_err!(self.inner.reshape(&new_shape))?;
        Ok(PyTensor { inner: result })
    }

    /// Get sum of all elements in tensor
    fn sum(&self) -> PyResult<f32> {
        Ok(self.inner.sum())
    }

    /// Get mean of all elements in tensor
    fn mean(&self) -> PyResult<f32> {
        Ok(self.inner.mean())
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

    /// Detach variable from computation graph
    /// 計算グラフから変数を切り離し
    fn detach(&self) -> PyResult<PyVariable> {
        let data_binding = self.inner.data();
        let data = data_binding.read().unwrap().clone();
        let detached = RustVariable::new(data, false); // No gradient tracking
        Ok(PyVariable { inner: detached })
    }

    /// Mark variable to retain gradients (for intermediate variables)
    /// 中間変数の勾配を保持するようマーク
    fn retain_grad(&self) -> PyResult<()> {
        // Note: This is a placeholder implementation
        // In practice, this would set a flag in RusTorch's Variable
        // to prevent gradient from being cleared after backward pass
        Ok(())
    }

    /// Register a hook function for gradient computation
    /// 勾配計算用のフック関数を登録
    fn register_hook(&self, _hook: PyObject) -> PyResult<()> {
        // Note: This is a placeholder implementation
        // In practice, this would register a callback in RusTorch's Variable
        // that gets called during backward pass
        Ok(())
    }

    /// Clone the variable (creates a new variable with same data)
    /// 変数をクローン（同じデータで新しい変数を作成）
    fn clone(&self) -> PyVariable {
        PyVariable { inner: self.inner.clone() }
    }

    /// Create a variable from existing tensor with gradient tracking
    /// 既存のテンソルから勾配トラッキング付きの変数を作成
    #[staticmethod]
    fn from_tensor(tensor: &PyTensor, requires_grad: Option<bool>) -> PyResult<PyVariable> {
        let requires_grad = requires_grad.unwrap_or(false);
        let variable = RustVariable::new(tensor.inner.clone(), requires_grad);
        Ok(PyVariable { inner: variable })
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
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: Option<bool>
    ) -> PyResult<Self> {
        // Store parameters for future implementation
        let _momentum = momentum.unwrap_or(0.0);
        let _weight_decay = weight_decay.unwrap_or(0.0);
        let _nesterov = nesterov.unwrap_or(false);

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

/// Python Adam optimizer wrapper (Phase 4 - Advanced optimizer)
#[pyclass(name = "Adam")]
pub struct PyAdam {
    pub parameters: Vec<PyVariable>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub amsgrad: bool,
    pub step_count: usize,
}

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer
    #[new]
    fn new(
        parameters: Vec<PyVariable>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>
    ) -> PyResult<Self> {
        let lr = lr.unwrap_or(0.001);
        let (beta1, beta2) = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let amsgrad = amsgrad.unwrap_or(false);

        // Validation
        if lr <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Learning rate must be positive"));
        }
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(pyo3::exceptions::PyValueError::new_err("Betas must be in [0, 1)"));
        }
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Epsilon must be positive"));
        }
        if weight_decay < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Weight decay must be non-negative"));
        }

        Ok(PyAdam {
            parameters,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            step_count: 0,
        })
    }

    /// Zero all parameter gradients
    fn zero_grad(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            param.inner.zero_grad();
        }
        Ok(())
    }

    /// Perform a single optimization step using Adam algorithm (simplified implementation)
    fn step(&mut self) -> PyResult<()> {
        self.step_count += 1;

        for param in &self.parameters {
            if !param.inner.requires_grad() {
                continue;
            }

            let grad_arc = param.inner.grad();
            let grad_lock = grad_arc.read().unwrap();

            if let Some(grad) = grad_lock.as_ref() {
                let param_data = param.inner.data();
                let mut param_lock = param_data.write().unwrap();

                // Simplified Adam update (for now, using SGD-like approach)
                // TODO: Implement proper Adam momentum and variance tracking
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
        if lr <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Learning rate must be positive"));
        }
        self.lr = lr;
        Ok(())
    }

    /// Get beta1 parameter
    #[getter]
    fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get beta2 parameter
    #[getter]
    fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get epsilon parameter
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get step count
    #[getter]
    fn step_count(&self) -> usize {
        self.step_count
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas=({}, {}), eps={}, weight_decay={})",
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay
        )
    }
}

/// Python BatchNorm1d wrapper for RusTorch BatchNorm1d (Phase 4 - Normalization layer)
#[pyclass(name = "BatchNorm1d")]
pub struct PyBatchNorm1d {
    pub inner: RustBatchNorm1d<f32>,
    pub num_features: usize,
}

#[pymethods]
impl PyBatchNorm1d {
    /// Create a new BatchNorm1d layer
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let _track_running_stats = track_running_stats.unwrap_or(true); // Store for future use

        // Validation
        if num_features == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Number of features must be positive"));
        }
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Epsilon must be positive"));
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(pyo3::exceptions::PyValueError::new_err("Momentum must be in [0, 1]"));
        }

        let batchnorm = RustBatchNorm1d::new(num_features, Some(eps), Some(momentum), Some(affine));
        Ok(PyBatchNorm1d {
            inner: batchnorm,
            num_features,
        })
    }

    /// Forward pass through BatchNorm1d
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set training mode
    fn train(&mut self) -> PyResult<()> {
        self.inner.train();
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.inner.eval();
        Ok(())
    }

    /// Get weight parameter (gamma)
    #[getter]
    fn weight(&self) -> PyResult<PyVariable> {
        let params = self.inner.parameters();
        // Weight is typically the first parameter
        if !params.is_empty() {
            Ok(PyVariable { inner: params[0].clone() })
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("No weight parameter available"))
        }
    }

    /// Get bias parameter (beta)
    #[getter]
    fn bias(&self) -> PyResult<PyVariable> {
        let params = self.inner.parameters();
        // Bias is typically the second parameter
        if params.len() > 1 {
            Ok(PyVariable { inner: params[1].clone() })
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("No bias parameter available"))
        }
    }

    /// Get number of features
    #[getter]
    fn num_features(&self) -> usize {
        self.num_features
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("BatchNorm1d(num_features={}, eps=1e-5, momentum=0.1)", self.num_features())
    }
}

/// Python Dropout wrapper for RusTorch Dropout (Phase 4 - Regularization layer)
#[pyclass(name = "Dropout")]
pub struct PyDropout {
    pub inner: RustDropout<f32>,
}

#[pymethods]
impl PyDropout {
    /// Create a new Dropout layer
    #[new]
    fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        // Validation
        if !(0.0..=1.0).contains(&p) {
            return Err(pyo3::exceptions::PyValueError::new_err("Dropout probability must be in [0, 1]"));
        }

        let dropout = RustDropout::new(p, inplace);
        Ok(PyDropout { inner: dropout })
    }

    /// Forward pass through Dropout
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Callable interface (Python convention)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set training mode
    fn train(&mut self) -> PyResult<()> {
        self.inner.train();
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.inner.eval();
        Ok(())
    }

    /// Get dropout probability
    #[getter]
    fn p(&self) -> f32 {
        // For now, we can't directly access p from the struct
        // This would need to be added to RusTorch's Dropout API
        0.5 // TODO: Access actual p from inner
    }

    /// Get inplace flag
    #[getter]
    fn inplace(&self) -> bool {
        // For now, we can't directly access inplace from the struct
        // This would need to be added to RusTorch's Dropout API
        false // TODO: Access actual inplace from inner
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Dropout(p={}, inplace={})", self.p(), self.inplace())
    }
}

/// Python Conv2d wrapper for RusTorch Conv2d (Phase 4 - Convolutional layer)
#[pyclass(name = "Conv2d")]
pub struct PyConv2d {
    pub inner: RustConv2d<f32>,
}

#[pymethods]
impl PyConv2d {
    /// Create a new Conv2d layer
    #[new]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        // Validate parameters
        if in_channels == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("in_channels must be > 0"));
        }
        if out_channels == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("out_channels must be > 0"));
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be > 0"));
        }

        let conv2d = RustConv2d::new(in_channels, out_channels, kernel_size, stride, padding, bias);
        Ok(PyConv2d { inner: conv2d })
    }

    /// Forward pass through Conv2d
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Call method (makes layer callable)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get weight parameter
    #[getter]
    fn weight(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable { inner: params[0].clone() }
    }

    /// Get bias parameter
    #[getter]
    fn bias(&self) -> Option<PyVariable> {
        let params = self.inner.parameters();
        if params.len() > 1 {
            Some(PyVariable { inner: params[1].clone() })
        } else {
            None
        }
    }

    /// Get input channels
    #[getter]
    fn in_channels(&self) -> usize {
        self.inner.in_channels()
    }

    /// Get output channels
    #[getter]
    fn out_channels(&self) -> usize {
        self.inner.out_channels()
    }

    /// Get kernel size
    #[getter]
    fn kernel_size(&self) -> (usize, usize) {
        self.inner.kernel_size()
    }

    /// Get stride
    #[getter]
    fn stride(&self) -> (usize, usize) {
        self.inner.stride()
    }

    /// Get padding
    #[getter]
    fn padding(&self) -> (usize, usize) {
        self.inner.padding()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Conv2d({}, {}, kernel_size={:?}, stride={:?}, padding={:?})",
            self.inner.in_channels(),
            self.inner.out_channels(),
            self.inner.kernel_size(),
            self.inner.stride(),
            self.inner.padding()
        )
    }
}

/// Python MaxPool2d wrapper for RusTorch MaxPool2d (Phase 4 - Pooling layer)
#[pyclass(name = "MaxPool2d")]
pub struct PyMaxPool2d {
    pub inner: RustMaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    /// Create a new MaxPool2d layer
    #[new]
    fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        // Validate parameters
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be > 0"));
        }

        let maxpool2d = RustMaxPool2d::new(kernel_size, stride, padding);
        Ok(PyMaxPool2d { inner: maxpool2d })
    }

    /// Forward pass through MaxPool2d
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Call method (makes layer callable)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get kernel size
    #[getter]
    fn kernel_size(&self) -> (usize, usize) {
        self.inner.kernel_size()
    }

    /// Get stride
    #[getter]
    fn stride(&self) -> (usize, usize) {
        self.inner.stride()
    }

    /// Get padding
    #[getter]
    fn padding(&self) -> (usize, usize) {
        self.inner.padding()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "MaxPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
            self.inner.kernel_size(), self.inner.stride(), self.inner.padding()
        )
    }
}

/// Python BatchNorm2d wrapper for RusTorch BatchNorm2d (Phase 4 - 2D Batch Normalization)
#[pyclass(name = "BatchNorm2d")]
pub struct PyBatchNorm2d {
    pub inner: RustBatchNorm2d<f32>,
}

#[pymethods]
impl PyBatchNorm2d {
    /// Create a new BatchNorm2d layer
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<Self> {
        // Validate parameters
        if num_features == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("num_features must be > 0"));
        }

        let batchnorm2d = RustBatchNorm2d::new(num_features, eps, momentum, affine);
        Ok(PyBatchNorm2d { inner: batchnorm2d })
    }

    /// Forward pass through BatchNorm2d
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    /// Call method (makes layer callable)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set to training mode
    fn train(&self) {
        self.inner.train();
    }

    /// Set to evaluation mode
    fn eval(&self) {
        self.inner.eval();
    }

    /// Get weight parameter
    #[getter]
    fn weight(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable { inner: params[0].clone() }
    }

    /// Get bias parameter
    #[getter]
    fn bias(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable { inner: params[1].clone() }
    }

    /// Get number of features
    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Get epsilon value
    #[getter]
    fn eps(&self) -> f32 {
        self.inner.eps()
    }

    /// Get momentum value
    #[getter]
    fn momentum(&self) -> f32 {
        self.inner.momentum()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "BatchNorm2d(num_features={}, eps={:.1e}, momentum={:.1})",
            self.inner.num_features(),
            self.inner.eps(),
            self.inner.momentum()
        )
    }
}

/// Python CrossEntropyLoss wrapper for RusTorch CrossEntropyLoss (Phase 4 - Classification Loss)
#[pyclass(name = "CrossEntropyLoss")]
pub struct PyCrossEntropyLoss {
    pub inner: RustCrossEntropyLoss<f32>,
}

#[pymethods]
impl PyCrossEntropyLoss {
    /// Create a new CrossEntropyLoss
    #[new]
    fn new() -> PyResult<Self> {
        let loss = RustCrossEntropyLoss::new();
        Ok(PyCrossEntropyLoss { inner: loss })
    }

    /// Forward pass of CrossEntropyLoss
    fn forward(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&predictions.inner, &targets.inner);
        Ok(PyVariable { inner: result })
    }

    /// Call method (makes loss function callable)
    fn __call__(&self, predictions: &PyVariable, targets: &PyVariable) -> PyResult<PyVariable> {
        self.forward(predictions, targets)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "CrossEntropyLoss()".to_string()
    }
}

/// Python Flatten wrapper (Phase 4 - Tensor Flattening for CNN->FC transition)
#[pyclass(name = "Flatten")]
pub struct PyFlatten {
    pub start_dim: usize,
    pub end_dim: isize,
}

#[pymethods]
impl PyFlatten {
    /// Create a new Flatten layer
    #[new]
    fn new(start_dim: Option<usize>, end_dim: Option<isize>) -> PyResult<Self> {
        let start_dim = start_dim.unwrap_or(1); // Default: flatten from dimension 1 (keep batch dimension)
        let end_dim = end_dim.unwrap_or(-1); // Default: flatten to end

        Ok(PyFlatten { start_dim, end_dim })
    }

    /// Forward pass through Flatten
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        // Get input shape
        let input_shape = input.inner.data().read().unwrap().shape().to_vec();

        if self.start_dim >= input_shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("start_dim {} is out of range for tensor with {} dimensions",
                        self.start_dim, input_shape.len())
            ));
        }

        // Calculate the new shape
        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..self.start_dim {
            new_shape.push(input_shape[i]);
        }

        // Calculate flattened dimension size
        let end_dim = if self.end_dim < 0 {
            input_shape.len() as isize + self.end_dim + 1
        } else {
            self.end_dim + 1
        } as usize;

        let mut flattened_size = 1;
        for i in self.start_dim..end_dim.min(input_shape.len()) {
            flattened_size *= input_shape[i];
        }
        new_shape.push(flattened_size);

        // Add dimensions after end_dim
        for i in end_dim.min(input_shape.len())..input_shape.len() {
            new_shape.push(input_shape[i]);
        }

        // Create flattened tensor
        let input_data = input.inner.data().read().unwrap().clone();
        let flattened_data = input_data.data.into_shape_with_order(new_shape).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to flatten tensor: {}", e))
        })?;

        let flattened_tensor = RustTensor::new(flattened_data);
        let result = RustVariable::new(flattened_tensor, input.inner.requires_grad());

        Ok(PyVariable { inner: result })
    }

    /// Call method (makes layer callable)
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Get start dimension
    #[getter]
    fn start_dim(&self) -> usize {
        self.start_dim
    }

    /// Get end dimension
    #[getter]
    fn end_dim(&self) -> isize {
        self.end_dim
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Flatten(start_dim={}, end_dim={})", self.start_dim, self.end_dim)
    }
}

/// Phase 5: Advanced Autograd API Implementation

/// No gradient context manager for disabling gradient computation
#[pyclass]
#[derive(Clone)]
pub struct NoGradContext {
    prev_state: bool,
}

#[pymethods]
impl NoGradContext {
    #[new]
    fn new() -> Self {
        NoGradContext { prev_state: true }
    }

    fn __enter__(&mut self) -> PyResult<()> {
        // Store current gradient state and disable gradients
        // Note: This is a simplified implementation
        // In practice, you'd interface with RusTorch's gradient state
        self.prev_state = true; // Assume gradients were enabled
        Ok(())
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // Restore previous gradient state
        Ok(false) // Don't suppress exceptions
    }
}

/// Create a no_grad context manager
#[pyfunction]
fn no_grad() -> NoGradContext {
    NoGradContext::new()
}

/// Enable gradient context manager for forcing gradient computation
#[pyclass]
#[derive(Clone)]
pub struct EnableGradContext {
    prev_state: bool,
}

#[pymethods]
impl EnableGradContext {
    #[new]
    fn new() -> Self {
        EnableGradContext { prev_state: false }
    }

    fn __enter__(&mut self) -> PyResult<()> {
        self.prev_state = false; // Assume gradients were disabled
        Ok(())
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(false)
    }
}

/// Create an enable_grad context manager
#[pyfunction]
fn enable_grad() -> EnableGradContext {
    EnableGradContext::new()
}

/// Functional gradient computation
#[pyfunction]
fn grad(
    outputs: Vec<PyVariable>,
    inputs: Vec<PyVariable>,
    retain_graph: Option<bool>,
    create_graph: Option<bool>,
) -> PyResult<Vec<Option<PyVariable>>> {
    let _retain_graph = retain_graph.unwrap_or(false);
    let _create_graph = create_graph.unwrap_or(false);
    
    // Simplified implementation - compute gradients for each output w.r.t each input
    let mut gradients = Vec::new();
    
    for _output in outputs.iter() {
        for input in inputs.iter() {
            // In practice, this would use RusTorch's autograd engine
            // For now, return the gradient if available
            let grad_binding = input.inner.grad();
            let grad_lock = grad_binding.read().unwrap();
            match grad_lock.as_ref() {
                Some(grad) => {
                    // Create a Variable from the gradient tensor
                    let grad_variable = RustVariable::new(grad.clone(), _create_graph);
                    gradients.push(Some(PyVariable { inner: grad_variable }));
                },
                None => gradients.push(None),
            }
        }
    }
    
    Ok(gradients)
}

/// RusTorch Python bindings module
#[pymodule]
fn _rustorch_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;
    
    // Neural network layers
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyFlatten>()?;
    
    // Activation functions
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    
    // Loss functions
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;
    
    // Optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    
    // Phase 5: Advanced Autograd classes
    m.add_class::<NoGradContext>()?;
    m.add_class::<EnableGradContext>()?;
    
    // Tensor creation functions
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    
    // Phase 5: Advanced Autograd functions
    m.add_function(wrap_pyfunction!(no_grad, m)?)?;
    m.add_function(wrap_pyfunction!(enable_grad, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    
    Ok(())
}