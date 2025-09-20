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
pub mod common;
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

    // CoreML functionality (when features are enabled)
    #[cfg(all(
        feature = "python",
        any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )
    ))]
    {
        m.add_class::<coreml::PyCoreMLDevice>()?;
        m.add_class::<coreml::PyCoreMLBackend>()?;
        m.add_class::<coreml::PyCoreMLBackendConfig>()?;
        m.add_class::<coreml::PyCoreMLStats>()?;
        m.add_function(wrap_pyfunction!(coreml::is_coreml_available, m)?)?;
        m.add_function(wrap_pyfunction!(coreml::get_coreml_device_info, m)?)?;
    }

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "RusTorch Team")?;
    m.add(
        "__description__",
        "High-performance tensor library and deep learning framework implemented in Rust",
    )?;

    Ok(())
}

/// CoreML Python bindings module
/// CoreMLのPythonバインディングモジュール
#[cfg(all(
    feature = "python",
    any(
        feature = "coreml",
        feature = "coreml-hybrid",
        feature = "coreml-fallback"
    )
))]
pub mod coreml {
    use crate::error::RusTorchError;
    use crate::gpu::coreml::{CoreMLBackend, CoreMLBackendConfig, CoreMLDevice};
    use crate::python::error::to_py_err;
    use pyo3::prelude::*;

    /// Python wrapper for CoreML device
    /// CoreMLデバイスのPythonラッパー
    #[pyclass(name = "CoreMLDevice")]
    pub struct PyCoreMLDevice {
        device: CoreMLDevice,
    }

    #[pymethods]
    impl PyCoreMLDevice {
        #[new]
        pub fn new(device_id: Option<usize>) -> PyResult<Self> {
            let device_id = device_id.unwrap_or(0);
            // 簡易的な実装 - 実際のCoreMLデバイス作成は後で実装
            #[cfg(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            ))]
            {
                let device = CoreMLDevice::new(device_id).map_err(to_py_err)?;
                Ok(Self { device })
            }

            #[cfg(not(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            )))]
            {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "CoreML features not enabled",
                ))
            }
        }

        pub fn device_id(&self) -> usize {
            self.device.device_id()
        }

        pub fn is_available(&self) -> bool {
            // TODO: 実際の可用性チェック
            true
        }

        pub fn memory_limit(&self) -> Option<usize> {
            // TODO: 実際のメモリ制限取得
            Some(1024 * 1024 * 1024) // 1GB
        }

        pub fn compute_units_limit(&self) -> Option<u32> {
            // TODO: 実際のコンピュートユニット制限取得
            Some(8)
        }

        pub fn model_cache_size(&self) -> usize {
            // TODO: 実際のモデルキャッシュサイズ取得
            100
        }

        pub fn cleanup_cache(&mut self) -> PyResult<()> {
            // TODO: 実際のキャッシュクリーンアップ
            Ok(())
        }

        pub fn __repr__(&self) -> String {
            format!("CoreMLDevice(device_id={})", self.device.device_id())
        }
    }

    /// Python wrapper for CoreML backend
    /// CoreMLバックエンドのPythonラッパー
    #[pyclass(name = "CoreMLBackend")]
    pub struct PyCoreMLBackend {
        backend: CoreMLBackend,
    }

    #[pymethods]
    impl PyCoreMLBackend {
        #[new]
        pub fn new(config: Option<PyCoreMLBackendConfig>) -> PyResult<Self> {
            let config = config.map(|c| c.config).unwrap_or_default();
            CoreMLBackend::new(config)
                .map(|backend| Self { backend })
                .map_err(to_py_err)
        }

        pub fn is_available(&self) -> bool {
            self.backend.is_available()
        }

        pub fn get_stats(&self) -> PyCoreMLStats {
            PyCoreMLStats {
                stats: self.backend.get_stats(),
            }
        }

        pub fn cleanup_cache(&mut self) -> PyResult<()> {
            self.backend.cleanup_cache().map_err(to_py_err)
        }

        pub fn __repr__(&self) -> String {
            "CoreMLBackend".to_string()
        }
    }

    /// Python wrapper for CoreML backend configuration
    /// CoreMLバックエンド設定のPythonラッパー
    #[pyclass(name = "CoreMLBackendConfig")]
    #[derive(Clone)]
    pub struct PyCoreMLBackendConfig {
        config: CoreMLBackendConfig,
    }

    #[pymethods]
    impl PyCoreMLBackendConfig {
        #[new]
        pub fn new(
            enable_caching: Option<bool>,
            max_cache_size: Option<usize>,
            enable_profiling: Option<bool>,
            auto_fallback: Option<bool>,
        ) -> Self {
            Self {
                config: CoreMLBackendConfig {
                    enable_caching: enable_caching.unwrap_or(true),
                    max_cache_size: max_cache_size.unwrap_or(1000),
                    enable_profiling: enable_profiling.unwrap_or(false),
                    auto_fallback: auto_fallback.unwrap_or(true),
                },
            }
        }

        #[getter]
        pub fn enable_caching(&self) -> bool {
            self.config.enable_caching
        }

        #[setter]
        pub fn set_enable_caching(&mut self, value: bool) {
            self.config.enable_caching = value;
        }

        #[getter]
        pub fn max_cache_size(&self) -> usize {
            self.config.max_cache_size
        }

        #[setter]
        pub fn set_max_cache_size(&mut self, value: usize) {
            self.config.max_cache_size = value;
        }

        #[getter]
        pub fn enable_profiling(&self) -> bool {
            self.config.enable_profiling
        }

        #[setter]
        pub fn set_enable_profiling(&mut self, value: bool) {
            self.config.enable_profiling = value;
        }

        #[getter]
        pub fn auto_fallback(&self) -> bool {
            self.config.auto_fallback
        }

        #[setter]
        pub fn set_auto_fallback(&mut self, value: bool) {
            self.config.auto_fallback = value;
        }

        pub fn __repr__(&self) -> String {
            format!(
                "CoreMLBackendConfig(enable_caching={}, max_cache_size={}, enable_profiling={}, auto_fallback={})",
                self.config.enable_caching,
                self.config.max_cache_size,
                self.config.enable_profiling,
                self.config.auto_fallback
            )
        }
    }

    /// Python wrapper for CoreML statistics
    /// CoreML統計のPythonラッパー
    #[pyclass(name = "CoreMLStats")]
    pub struct PyCoreMLStats {
        stats: crate::gpu::coreml::CoreMLBackendStats,
    }

    #[pymethods]
    impl PyCoreMLStats {
        #[getter]
        pub fn total_operations(&self) -> u64 {
            self.stats.total_operations
        }

        #[getter]
        pub fn cache_hits(&self) -> u64 {
            self.stats.cache_hits
        }

        #[getter]
        pub fn cache_misses(&self) -> u64 {
            self.stats.cache_misses
        }

        #[getter]
        pub fn fallback_operations(&self) -> u64 {
            self.stats.fallback_operations
        }

        #[getter]
        pub fn average_execution_time_ms(&self) -> f64 {
            self.stats.average_execution_time.as_secs_f64() * 1000.0
        }

        pub fn cache_hit_rate(&self) -> f64 {
            if self.stats.cache_hits + self.stats.cache_misses == 0 {
                0.0
            } else {
                self.stats.cache_hits as f64
                    / (self.stats.cache_hits + self.stats.cache_misses) as f64
            }
        }

        pub fn fallback_rate(&self) -> f64 {
            if self.stats.total_operations == 0 {
                0.0
            } else {
                self.stats.fallback_operations as f64 / self.stats.total_operations as f64
            }
        }

        pub fn __repr__(&self) -> String {
            format!(
                "CoreMLStats(operations={}, cache_hit_rate={:.2}%, fallback_rate={:.2}%)",
                self.stats.total_operations,
                self.cache_hit_rate() * 100.0,
                self.fallback_rate() * 100.0
            )
        }
    }

    /// Check if CoreML is available on the current system
    /// 現在のシステムでCoreMLが利用可能かチェック
    #[pyfunction]
    pub fn is_coreml_available() -> bool {
        crate::backends::DeviceManager::is_coreml_available()
    }

    /// Get CoreML device information
    /// CoreMLデバイス情報を取得
    #[pyfunction]
    pub fn get_coreml_device_info() -> PyResult<String> {
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            if !is_coreml_available() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "CoreML is not available on this system",
                ));
            }

            let info = format!(
                "CoreML Device Information:\n\
                - Platform: {}\n\
                - Available: {}\n\
                - Neural Engine: {}\n\
                - GPU Support: {}",
                if cfg!(target_os = "macos") {
                    "macOS"
                } else {
                    "Other"
                },
                is_coreml_available(),
                "Available", // Neural Engine availability would require platform-specific checks
                cfg!(feature = "metal")
            );
            Ok(info)
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CoreML features are not enabled in this build",
            ))
        }
    }
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
