//! Python bindings module for RusTorch
//! RusTorchのPythonバインディングモジュール
//!
//! このモジュールは以下のサブモジュールに分割されています:
//! - tensor: テンソル操作とNumPy相互運用
//! - nn: ニューラルネットワーク層とモジュール
//! - autograd: 自動微分とVariable
//! - optim: 最適化アルゴリズム
//! - data: データローディングとデータセット
//! - training: 高レベル訓練API
//! - utils: ユーティリティ関数
//! - distributed: 分散訓練サポート
//! - visualization: 可視化機能

#[cfg(feature = "python")]
pub mod tensor;

#[cfg(feature = "python")]
pub mod autograd;

#[cfg(feature = "python")]
pub mod nn;

#[cfg(feature = "python")]
pub mod optim;

#[cfg(feature = "python")]
pub mod data;

#[cfg(feature = "python")]
pub mod training;

#[cfg(feature = "python")]
pub mod utils;

#[cfg(feature = "python")]
pub mod distributed;

#[cfg(feature = "python")]
pub mod visualization;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module registration for RusTorch
/// RusTorchのPythonモジュール登録
#[cfg(feature = "python")]
#[pymodule]
fn rustorch(_py: Python, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    // Core tensor operations
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tensor::PyDevice>()?;

    // Automatic differentiation
    m.add_class::<autograd::PyVariable>()?;

    // Neural network layers and modules
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyBatchNorm2d>()?;
    m.add_class::<nn::PyMSELoss>()?;
    m.add_class::<nn::PyCrossEntropyLoss>()?;

    // Optimizers
    m.add_class::<optim::PySGD>()?;
    m.add_class::<optim::PyAdam>()?;

    // Data loading and processing
    m.add_class::<data::PyTensorDataset>()?;
    m.add_class::<data::PyDataLoader>()?;
    m.add_class::<data::PyTransform>()?;
    m.add_class::<data::PyTransforms>()?;

    // High-level training API
    m.add_class::<training::PyTrainer>()?;
    m.add_class::<training::PyTrainerConfig>()?;
    m.add_class::<training::PyTrainingHistory>()?;
    m.add_class::<training::PyModel>()?;
    m.add_class::<training::PyModelBuilder>()?;

    // Utilities
    m.add_class::<utils::PyModelSerializer>()?;
    m.add_class::<utils::PyModelComparator>()?;
    m.add_class::<utils::PyConfig>()?;
    m.add_class::<utils::PyProfiler>()?;

    // Distributed training
    m.add_class::<distributed::PyDistributedDataParallel>()?;
    m.add_class::<distributed::PyDistributedConfig>()?;
    m.add_class::<distributed::PyDistributedBackend>()?;
    m.add_class::<distributed::PyDistributedSampler>()?;

    // Visualization
    m.add_class::<visualization::PyPlotter>()?;
    m.add_class::<visualization::PyModelVisualizer>()?;
    m.add_class::<visualization::PyTensorStats>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "RusTorch Team")?;
    m.add(
        "__description__",
        "High-performance tensor library and deep learning framework implemented in Rust",
    )?;

    Ok(())
}

#[cfg(feature = "python")]
pub use {
    autograd::PyVariable,
    data::{PyDataLoader, PyTensorDataset, PyTransform, PyTransforms},
    distributed::{
        PyDistributedBackend, PyDistributedConfig, PyDistributedDataParallel, PyDistributedSampler,
    },
    nn::{PyBatchNorm2d, PyConv2d, PyCrossEntropyLoss, PyLinear, PyMSELoss},
    optim::{PyAdam, PySGD},
    tensor::{PyDevice, PyTensor},
    training::{PyModel, PyModelBuilder, PyTrainer, PyTrainerConfig, PyTrainingHistory},
    utils::{PyConfig, PyModelComparator, PyModelSerializer, PyProfiler},
    visualization::{PyModelVisualizer, PyPlotter, PyTensorStats},
};

/// Common error handling utilities for Python bindings
#[cfg(feature = "python")]
pub mod error {
    use crate::error::RusTorchError;
    use pyo3::exceptions::*;
    use pyo3::PyErr;

    /// Convert RusTorchError to PyErr
    pub fn to_py_err(error: RusTorchError) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }
}

/// Common utilities for Python-Rust interop
#[cfg(feature = "python")]
pub mod interop {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
    use pyo3::prelude::*;

    /// Convert Vec<f32> to Python array
    pub fn vec_to_pyarray<'py>(vec: Vec<f32>, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vec.into_pyarray(py)
    }

    /// Convert Python array to Vec<f32>
    pub fn pyarray_to_vec(array: PyReadonlyArray1<f32>) -> Vec<f32> {
        array.as_array().to_vec()
    }

    /// Convert Python list to Vec<usize>
    pub fn pylist_to_vec_usize(
        list: &pyo3::Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<usize>> {
        let mut result = Vec::new();
        for item in list.iter() {
            let value: usize = item.extract()?;
            result.push(value);
        }
        Ok(result)
    }

    /// Convert Python list to Vec<f32>
    pub fn pylist_to_vec_f32(list: &pyo3::Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<f32>> {
        let mut result = Vec::new();
        for item in list.iter() {
            let value: f32 = item.extract()?;
            result.push(value);
        }
        Ok(result)
    }
}
