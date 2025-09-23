//! SGD optimizer implementation
//! SGDオプティマイザー実装

use pyo3::prelude::*;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, optim_error};

/// Python SGD optimizer wrapper
/// Python SGDオプティマイザーラッパー
#[pyclass(name = "SGD")]
pub struct PySGD {
    pub parameters: Vec<PyVariable>,
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl PySGD {
    /// Create new PySGD
    /// PySGDを作成
    pub fn new(
        parameters: Vec<PyVariable>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        Self {
            parameters,
            lr,
            momentum,
            weight_decay,
            nesterov,
        }
    }
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    /// 新しいSGDオプティマイザーを作成
    #[new]
    fn py_new(
        parameters: Vec<PyVariable>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: Option<bool>,
    ) -> PyResult<Self> {
        // Parameter validation
        if lr <= 0.0 {
            return Err(invalid_param!("lr", lr, "must be positive").into());
        }

        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);

        if momentum < 0.0 {
            return Err(invalid_param!("momentum", momentum, "must be non-negative").into());
        }

        if weight_decay < 0.0 {
            return Err(invalid_param!("weight_decay", weight_decay, "must be non-negative").into());
        }

        Ok(PySGD::new(parameters, lr, momentum, weight_decay, nesterov))
    }

    /// Zero all parameter gradients
    /// 全パラメータの勾配をゼロに
    fn zero_grad(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            param.inner.zero_grad().map_err(|e| {
                optim_error!("SGD", &format!("Zero grad failed: {:?}", e))
            })?;
        }
        Ok(())
    }

    /// Perform a single optimization step
    /// 単一の最適化ステップを実行
    fn step(&mut self) -> PyResult<()> {
        // Simplified SGD implementation
        // Note: This is a basic implementation - full SGD with momentum
        // would require velocity buffers
        for param in &mut self.parameters {
            if let Some(grad) = param.inner.grad() {
                // Simple gradient descent: param = param - lr * grad
                let scaled_grad = &grad * self.lr;
                let updated_param = &param.inner - &scaled_grad;
                param.inner = updated_param;
            }
        }
        Ok(())
    }

    /// Get learning rate
    /// 学習率を取得
    #[getter]
    fn lr(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    /// 学習率を設定
    #[setter]
    fn set_lr(&mut self, lr: f32) -> PyResult<()> {
        if lr <= 0.0 {
            return Err(invalid_param!("lr", lr, "must be positive").into());
        }
        self.lr = lr;
        Ok(())
    }

    /// Get momentum
    /// モーメンタムを取得
    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get weight decay
    /// 重み減衰を取得
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get nesterov flag
    /// nesterovフラグを取得
    #[getter]
    fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!("SGD(lr={})", self.lr)
    }
}