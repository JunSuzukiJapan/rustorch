//! Python bindings for RusTorch optimizers
//! RusTorchオプティマイザーのPythonバインディング

use pyo3::prelude::*;
use rustorch::optim::{SGD, Adam, Optimizer};
use crate::PyVariable;

/// Python wrapper for RusTorch SGD optimizer
/// RusTorch SGDオプティマイザーのPythonラッパー
#[pyclass(name = "SGD")]
pub struct PySGD {
    sgd: SGD,
    parameters: Vec<PyVariable>,
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    /// 新しいSGDオプティマイザーを作成
    #[new]
    #[pyo3(signature = (params, lr, momentum = 0.0))]
    fn py_new(params: Vec<PyVariable>, lr: f32, momentum: f32) -> Self {
        Self {
            sgd: SGD::new(lr, momentum),
            parameters: params,
        }
    }
    
    /// Perform optimization step
    /// 最適化ステップを実行
    fn step(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            let param_data = param.data();
            let grad_opt = param.grad();
            
            if let Some(grad) = grad_opt {
                self.sgd.step(param_data.inner(), grad.inner());
            }
        }
        Ok(())
    }
    
    /// Zero gradients for all parameters
    /// すべてのパラメータの勾配をゼロに設定
    fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
    
    /// Get learning rate
    /// 学習率を取得
    #[getter]
    fn lr(&self) -> f32 {
        // This would need to be exposed from the Rust SGD implementation
        0.01 // Placeholder
    }
    
    /// Set learning rate
    /// 学習率を設定
    #[setter]
    fn set_lr(&mut self, lr: f32) {
        // This would need to be implemented in the Rust SGD
        // For now, this is a placeholder
    }
    
    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("SGD(lr={}, momentum={})", self.lr(), 0.0) // Placeholder values
    }
}

/// Python wrapper for RusTorch Adam optimizer
/// RusTorch AdamオプティマイザーのPythonラッパー
#[pyclass(name = "Adam")]
pub struct PyAdam {
    adam: Adam,
    parameters: Vec<PyVariable>,
}

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer
    /// 新しいAdamオプティマイザーを作成
    #[new]
    #[pyo3(signature = (params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8))]
    fn py_new(
        params: Vec<PyVariable>, 
        lr: f32, 
        beta1: f32, 
        beta2: f32, 
        eps: f32
    ) -> Self {
        Self {
            adam: Adam::new(lr, beta1, beta2, eps),
            parameters: params,
        }
    }
    
    /// Perform optimization step
    /// 最適化ステップを実行
    fn step(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            let param_data = param.data();
            let grad_opt = param.grad();
            
            if let Some(grad) = grad_opt {
                self.adam.step(param_data.inner(), grad.inner());
            }
        }
        Ok(())
    }
    
    /// Zero gradients for all parameters
    /// すべてのパラメータの勾配をゼロに設定
    fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
    
    /// Get learning rate
    /// 学習率を取得
    #[getter]
    fn lr(&self) -> f32 {
        // This would need to be exposed from the Rust Adam implementation
        0.001 // Placeholder
    }
    
    /// Set learning rate
    /// 学習率を設定
    #[setter]
    fn set_lr(&mut self, lr: f32) {
        // This would need to be implemented in the Rust Adam
        // For now, this is a placeholder
    }
    
    /// Get beta1 parameter
    /// beta1パラメータを取得
    #[getter]
    fn beta1(&self) -> f32 {
        0.9 // Placeholder
    }
    
    /// Get beta2 parameter
    /// beta2パラメータを取得
    #[getter]
    fn beta2(&self) -> f32 {
        0.999 // Placeholder
    }
    
    /// Get epsilon parameter
    /// epsilonパラメータを取得
    #[getter]
    fn eps(&self) -> f32 {
        1e-8 // Placeholder
    }
    
    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("Adam(lr={}, beta1={}, beta2={}, eps={})", 
               self.lr(), self.beta1(), self.beta2(), self.eps())
    }
}

/// Learning rate scheduler
/// 学習率スケジューラー
#[pyclass(name = "StepLR")]
pub struct PyStepLR {
    optimizer: PyObject,
    step_size: usize,
    gamma: f32,
    last_epoch: i32,
}

#[pymethods]
impl PyStepLR {
    /// Create a new StepLR scheduler
    /// 新しいStepLRスケジューラーを作成
    #[new]
    #[pyo3(signature = (optimizer, step_size, gamma = 0.1, last_epoch = -1))]
    fn py_new(
        optimizer: PyObject, 
        step_size: usize, 
        gamma: f32, 
        last_epoch: i32
    ) -> Self {
        Self {
            optimizer,
            step_size,
            gamma,
            last_epoch,
        }
    }
    
    /// Step the scheduler
    /// スケジューラーをステップ
    fn step(&mut self, py: Python) -> PyResult<()> {
        self.last_epoch += 1;
        
        if (self.last_epoch as usize) % self.step_size == 0 && self.last_epoch > 0 {
            // Decrease learning rate
            if let Ok(lr_attr) = self.optimizer.getattr(py, "lr") {
                let current_lr: f32 = lr_attr.extract(py)?;
                let new_lr = current_lr * self.gamma;
                self.optimizer.setattr(py, "lr", new_lr)?;
            }
        }
        
        Ok(())
    }
    
    /// Get current learning rate
    /// 現在の学習率を取得
    fn get_last_lr(&self, py: Python) -> PyResult<f32> {
        if let Ok(lr_attr) = self.optimizer.getattr(py, "lr") {
            lr_attr.extract(py)
        } else {
            Ok(0.0)
        }
    }
    
    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("StepLR(step_size={}, gamma={})", self.step_size, self.gamma)
    }
}