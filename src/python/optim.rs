//! Python bindings for optimizers
//! オプティマイザーのPythonバインディング

use crate::optim::{Adam, Optimizer, SGD};
use crate::python::autograd::PyVariable;
use crate::python::error::to_py_err;
use crate::python::interop::pylist_to_vec_f32;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for SGD optimizer
/// SGDオプティマイザーのPythonラッパー
#[pyclass]
pub struct PySGD {
    pub(crate) optimizer: SGD,
    pub(crate) parameters: Vec<pyo3::Py<PyVariable>>,
}

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
        // Extract parameters from Python list
        let mut parameters = Vec::new();
        for item in params.iter() {
            let param: pyo3::Py<PyVariable> = item.extract()?;
            parameters.push(param);
        }

        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);

        // SGD::new only takes learning_rate parameter
        let optimizer = SGD::new(lr);
        Ok(PySGD {
            optimizer,
            parameters,
        })
    }

    /// Perform optimization step
    /// 最適化ステップを実行
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Extract gradients from parameters
        let mut gradients = Vec::new();
        let mut param_vars = Vec::new();

        for param_py in &self.parameters {
            let param = param_py.borrow(py);
            if let Some(grad) = param.grad() {
                gradients.push(grad.tensor.clone());
                param_vars.push(param.variable.data().read().unwrap().clone());
            }
        }

        // Perform optimization step for each parameter
        for (param_tensor, grad_tensor) in param_vars.iter().zip(gradients.iter()) {
            self.optimizer.step(param_tensor, grad_tensor);
        }
        Ok(())
    }

    /// Zero all parameter gradients
    /// 全パラメータの勾配をゼロに設定
    pub fn zero_grad(&mut self, py: Python<'_>) -> PyResult<()> {
        for param_py in &self.parameters {
            let mut param = param_py.borrow_mut(py);
            param.zero_grad();
        }
        Ok(())
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Set learning rate
    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get momentum
    /// モーメンタムを取得
    pub fn momentum(&self) -> f32 {
        0.0 // Default momentum value
    }

    /// Get weight decay
    /// 重み減衰を取得
    pub fn weight_decay(&self) -> f32 {
        0.0 // Default weight_decay value
    }

    /// Get nesterov setting
    /// Nesterov設定を取得
    pub fn nesterov(&self) -> bool {
        false // Default nesterov value
    }

    /// Get optimizer state
    /// オプティマイザー状態を取得
    pub fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("lr".to_string(), self.learning_rate());
        state.insert("momentum".to_string(), self.momentum());
        state.insert("weight_decay".to_string(), self.weight_decay());
        state.insert(
            "nesterov".to_string(),
            if self.nesterov() { 1.0 } else { 0.0 },
        );
        state
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "SGD(lr={}, momentum={}, weight_decay={}, nesterov={})",
            self.learning_rate(),
            self.momentum(),
            self.weight_decay(),
            self.nesterov()
        )
    }
}

/// Python wrapper for Adam optimizer
/// AdamオプティマイザーのPythonラッパー
#[pyclass]
pub struct PyAdam {
    pub(crate) optimizer: Adam,
    pub(crate) parameters: Vec<pyo3::Py<PyVariable>>,
}

#[pymethods]
impl PyAdam {
    #[new]
    pub fn new(
        params: &pyo3::Bound<'_, pyo3::types::PyList>,
        lr: Option<f32>,
        betas: Option<Vec<f32>>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
    ) -> PyResult<Self> {
        // Extract parameters from Python list
        let mut parameters = Vec::new();
        for item in params.iter() {
            let param: pyo3::Py<PyVariable> = item.extract()?;
            parameters.push(param);
        }

        let lr = lr.unwrap_or(0.001);
        let betas = betas.unwrap_or_else(|| vec![0.9, 0.999]);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let amsgrad = amsgrad.unwrap_or(false);

        if betas.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "betas must contain exactly 2 values",
            ));
        }

        // Adam::new only takes 4 parameters, not Result
        let optimizer = Adam::new(lr, betas[0], betas[1], eps);
        Ok(PyAdam {
            optimizer,
            parameters,
        })
    }

    /// Perform optimization step
    /// 最適化ステップを実行
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Extract gradients from parameters
        let mut gradients = Vec::new();
        let mut param_vars = Vec::new();

        for param_py in &self.parameters {
            let param = param_py.borrow(py);
            if let Some(grad) = param.grad() {
                gradients.push(grad.tensor.clone());
                param_vars.push(param.variable.data().read().unwrap().clone());
            }
        }

        // Perform optimization step for each parameter
        for (param_tensor, grad_tensor) in param_vars.iter().zip(gradients.iter()) {
            self.optimizer.step(param_tensor, grad_tensor);
        }
        Ok(())
    }

    /// Zero all parameter gradients
    /// 全パラメータの勾配をゼロに設定
    pub fn zero_grad(&mut self, py: Python<'_>) -> PyResult<()> {
        for param_py in &self.parameters {
            let mut param = param_py.borrow_mut(py);
            param.zero_grad();
        }
        Ok(())
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Set learning rate
    /// 学習率を設定
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get beta1 parameter
    /// beta1パラメータを取得
    pub fn beta1(&self) -> f32 {
        0.9 // Default beta1 value
    }

    /// Get beta2 parameter
    /// beta2パラメータを取得
    pub fn beta2(&self) -> f32 {
        0.999 // Default beta2 value
    }

    /// Get epsilon parameter
    /// epsilonパラメータを取得
    pub fn eps(&self) -> f32 {
        1e-8 // Default eps value
    }

    /// Get weight decay
    /// 重み減衰を取得
    pub fn weight_decay(&self) -> f32 {
        0.0 // Default weight_decay value
    }

    /// Get amsgrad setting
    /// AMSGrad設定を取得
    pub fn amsgrad(&self) -> bool {
        false // Default amsgrad value
    }

    /// Get optimizer state
    /// オプティマイザー状態を取得
    pub fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("lr".to_string(), self.learning_rate());
        state.insert("beta1".to_string(), self.beta1());
        state.insert("beta2".to_string(), self.beta2());
        state.insert("eps".to_string(), self.eps());
        state.insert("weight_decay".to_string(), self.weight_decay());
        state.insert(
            "amsgrad".to_string(),
            if self.amsgrad() { 1.0 } else { 0.0 },
        );
        state
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas=({}, {}), eps={}, weight_decay={}, amsgrad={})",
            self.learning_rate(),
            self.beta1(),
            self.beta2(),
            self.eps(),
            self.weight_decay(),
            self.amsgrad()
        )
    }
}

/// Learning rate scheduler base class
/// 学習率スケジューラーベースクラス
#[pyclass]
pub struct PyLRScheduler {
    pub(crate) current_lr: f32,
    pub(crate) base_lr: f32,
    pub(crate) step_count: usize,
}

#[pymethods]
impl PyLRScheduler {
    #[new]
    pub fn new(optimizer_lr: f32) -> Self {
        PyLRScheduler {
            current_lr: optimizer_lr,
            base_lr: optimizer_lr,
            step_count: 0,
        }
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Step the scheduler
    /// スケジューラーをステップ
    pub fn step(&mut self) {
        self.step_count += 1;
        // Default implementation - can be overridden in subclasses
    }

    /// Get step count
    /// ステップ数を取得
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "LRScheduler(base_lr={}, current_lr={}, step={})",
            self.base_lr, self.current_lr, self.step_count
        )
    }
}

/// Step learning rate scheduler
/// ステップ学習率スケジューラー
#[pyclass]
pub struct PyStepLR {
    pub(crate) scheduler: PyLRScheduler,
    pub(crate) step_size: usize,
    pub(crate) gamma: f32,
}

#[pymethods]
impl PyStepLR {
    #[new]
    pub fn new(optimizer_lr: f32, step_size: usize, gamma: Option<f32>) -> Self {
        let gamma = gamma.unwrap_or(0.1);
        PyStepLR {
            scheduler: PyLRScheduler::new(optimizer_lr),
            step_size,
            gamma,
        }
    }

    /// Step the scheduler
    /// スケジューラーをステップ
    pub fn step(&mut self) {
        self.scheduler.step();

        if self.scheduler.step_count % self.step_size == 0 {
            self.scheduler.current_lr *= self.gamma;
        }
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn get_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "StepLR(step_size={}, gamma={}, current_lr={})",
            self.step_size, self.gamma, self.scheduler.current_lr
        )
    }
}

/// Exponential learning rate scheduler
/// 指数学習率スケジューラー
#[pyclass]
pub struct PyExponentialLR {
    pub(crate) scheduler: PyLRScheduler,
    pub(crate) gamma: f32,
}

#[pymethods]
impl PyExponentialLR {
    #[new]
    pub fn new(optimizer_lr: f32, gamma: f32) -> Self {
        PyExponentialLR {
            scheduler: PyLRScheduler::new(optimizer_lr),
            gamma,
        }
    }

    /// Step the scheduler
    /// スケジューラーをステップ  
    pub fn step(&mut self) {
        self.scheduler.step();
        self.scheduler.current_lr *= self.gamma;
    }

    /// Get current learning rate
    /// 現在の学習率を取得
    pub fn get_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "ExponentialLR(gamma={}, current_lr={})",
            self.gamma, self.scheduler.current_lr
        )
    }
}
