//! Adam optimizer implementation
//! Adamオプティマイザー実装

use pyo3::prelude::*;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, optim_error};

/// Python Adam optimizer wrapper
/// Python Adamオプティマイザーラッパー
#[pyclass(name = "Adam")]
pub struct PyAdam {
    pub parameters: Vec<PyVariable>,
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
    pub amsgrad: bool,
}

impl PyAdam {
    /// Create new PyAdam
    /// PyAdamを作成
    pub fn new(
        parameters: Vec<PyVariable>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            parameters,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
        }
    }
}

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer
    /// 新しいAdamオプティマイザーを作成
    #[new]
    fn py_new(
        parameters: Vec<PyVariable>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
    ) -> PyResult<Self> {
        let lr = lr.unwrap_or(0.001);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let amsgrad = amsgrad.unwrap_or(false);

        // Parameter validation
        if lr <= 0.0 {
            return Err(invalid_param!("lr", lr, "must be positive").into());
        }

        if !(0.0..1.0).contains(&betas.0) {
            return Err(invalid_param!("betas[0]", betas.0, "must be in [0, 1)").into());
        }

        if !(0.0..1.0).contains(&betas.1) {
            return Err(invalid_param!("betas[1]", betas.1, "must be in [0, 1)").into());
        }

        if eps <= 0.0 {
            return Err(invalid_param!("eps", eps, "must be positive").into());
        }

        if weight_decay < 0.0 {
            return Err(invalid_param!("weight_decay", weight_decay, "must be non-negative").into());
        }

        Ok(PyAdam::new(parameters, lr, betas, eps, weight_decay, amsgrad))
    }

    /// Zero all parameter gradients
    /// 全パラメータの勾配をゼロに
    fn zero_grad(&mut self) -> PyResult<()> {
        for param in &self.parameters {
            param.inner.zero_grad().map_err(|e| {
                optim_error!("Adam", &format!("Zero grad failed: {:?}", e))
            })?;
        }
        Ok(())
    }

    /// Perform a single optimization step
    /// 単一の最適化ステップを実行
    fn step(&mut self) -> PyResult<()> {
        // Simplified Adam implementation
        // Note: This is a basic implementation - full Adam would require
        // momentum buffers and bias correction
        for param in &mut self.parameters {
            if let Some(grad) = param.inner.grad() {
                // Simple gradient descent with learning rate
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

    /// Get betas
    /// ベータ値を取得
    #[getter]
    fn betas(&self) -> (f32, f32) {
        self.betas
    }

    /// Get epsilon
    /// イプシロンを取得
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay
    /// 重み減衰を取得
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get amsgrad flag
    /// amsgradフラグを取得
    #[getter]
    fn amsgrad(&self) -> bool {
        self.amsgrad
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas={:?}, eps={:.0e}, weight_decay={})",
            self.lr, self.betas, self.eps, self.weight_decay
        )
    }
}