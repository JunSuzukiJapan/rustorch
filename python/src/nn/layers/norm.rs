//! Normalization layers implementation
//! 正規化層実装

use pyo3::prelude::*;
use rustorch::nn::BatchNorm1d as RustBatchNorm1d;
use rustorch::nn::BatchNorm2d as RustBatchNorm2d;
use crate::core::variable::PyVariable;
use crate::core::errors::{RusTorchResult, RusTorchError};
use crate::{invalid_param, nn_error};

/// Python BatchNorm1d layer wrapper
/// Python BatchNorm1d層ラッパー
#[pyclass(name = "BatchNorm1d")]
pub struct PyBatchNorm1d {
    pub inner: RustBatchNorm1d<f32>,
    pub num_features: usize,
}

impl PyBatchNorm1d {
    /// Create new PyBatchNorm1d from RustBatchNorm1d
    /// RustBatchNorm1dからPyBatchNorm1dを作成
    pub fn new(batchnorm1d: RustBatchNorm1d<f32>, num_features: usize) -> Self {
        Self {
            inner: batchnorm1d,
            num_features,
        }
    }
}

#[pymethods]
impl PyBatchNorm1d {
    /// Create a new BatchNorm1d layer
    /// 新しいBatchNorm1d層を作成
    #[new]
    fn py_new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let _track_running_stats = track_running_stats.unwrap_or(true);

        // Validation
        if num_features == 0 {
            return Err(invalid_param!("num_features", num_features, "must be positive").into());
        }
        if eps <= 0.0 {
            return Err(invalid_param!("eps", eps, "must be positive").into());
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(invalid_param!("momentum", momentum, "must be in [0, 1]").into());
        }

        let batchnorm = RustBatchNorm1d::new(num_features, Some(eps), Some(momentum), Some(affine));
        Ok(PyBatchNorm1d::new(batchnorm, num_features))
    }

    /// Forward pass through BatchNorm1d
    /// BatchNorm1d層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set to training mode
    /// 訓練モードに設定
    fn train(&self) {
        self.inner.train();
    }

    /// Set to evaluation mode
    /// 評価モードに設定
    fn eval(&self) {
        self.inner.eval();
    }

    /// Get weight parameter
    /// 重みパラメータを取得
    #[getter]
    fn weight(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable::new(params[0].clone())
    }

    /// Get bias parameter
    /// バイアスパラメータを取得
    #[getter]
    fn bias(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable::new(params[1].clone())
    }

    /// Get number of features
    /// 特徴数を取得
    #[getter]
    fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get epsilon value
    /// イプシロン値を取得
    #[getter]
    fn eps(&self) -> f32 {
        self.inner.eps()
    }

    /// Get momentum value
    /// モーメンタム値を取得
    #[getter]
    fn momentum(&self) -> f32 {
        self.inner.momentum()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "BatchNorm1d(num_features={}, eps={:.1e}, momentum={:.1})",
            self.num_features(),
            self.eps(),
            self.momentum()
        )
    }
}

/// Python BatchNorm2d layer wrapper
/// Python BatchNorm2d層ラッパー
#[pyclass(name = "BatchNorm2d")]
pub struct PyBatchNorm2d {
    pub inner: RustBatchNorm2d<f32>,
    pub num_features: usize,
}

impl PyBatchNorm2d {
    /// Create new PyBatchNorm2d from RustBatchNorm2d
    /// RustBatchNorm2dからPyBatchNorm2dを作成
    pub fn new(batchnorm2d: RustBatchNorm2d<f32>, num_features: usize) -> Self {
        Self {
            inner: batchnorm2d,
            num_features,
        }
    }
}

#[pymethods]
impl PyBatchNorm2d {
    /// Create a new BatchNorm2d layer
    /// 新しいBatchNorm2d層を作成
    #[new]
    fn py_new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<Self> {
        // Validate parameters
        if num_features == 0 {
            return Err(invalid_param!("num_features", num_features, "must be positive").into());
        }

        let batchnorm2d = RustBatchNorm2d::new(num_features, eps, momentum, affine);
        Ok(PyBatchNorm2d::new(batchnorm2d, num_features))
    }

    /// Forward pass through BatchNorm2d
    /// BatchNorm2d層のフォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable::new(result))
    }

    /// Call method (makes layer callable)
    /// 呼び出しメソッド（層を呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    /// Set to training mode
    /// 訓練モードに設定
    fn train(&self) {
        self.inner.train();
    }

    /// Set to evaluation mode
    /// 評価モードに設定
    fn eval(&self) {
        self.inner.eval();
    }

    /// Get weight parameter
    /// 重みパラメータを取得
    #[getter]
    fn weight(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable::new(params[0].clone())
    }

    /// Get bias parameter
    /// バイアスパラメータを取得
    #[getter]
    fn bias(&self) -> PyVariable {
        let params = self.inner.parameters();
        PyVariable::new(params[1].clone())
    }

    /// Get number of features
    /// 特徴数を取得
    #[getter]
    fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get epsilon value
    /// イプシロン値を取得
    #[getter]
    fn eps(&self) -> f32 {
        self.inner.eps()
    }

    /// Get momentum value
    /// モーメンタム値を取得
    #[getter]
    fn momentum(&self) -> f32 {
        self.inner.momentum()
    }

    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        format!(
            "BatchNorm2d(num_features={}, eps={:.1e}, momentum={:.1})",
            self.num_features(),
            self.eps(),
            self.momentum()
        )
    }
}