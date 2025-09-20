//! Common utilities and traits for Python bindings
//! Pythonバインディング用の共通ユーティリティと特性

use crate::error::RusTorchError;
use pyo3::exceptions::*;
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

/// Common error conversion for all Python bindings
/// すべてのPythonバインディング用の共通エラー変換
pub fn to_py_err(error: RusTorchError) -> PyErr {
    match error {
        RusTorchError::ShapeMismatch { expected, actual } => PyValueError::new_err(format!(
            "Shape mismatch: expected {:?}, got {:?}", expected, actual
        )),
        RusTorchError::Device { device, message } => PyRuntimeError::new_err(format!(
            "Device error on {}: {}", device, message
        )),
        RusTorchError::TensorOp { message, .. } => PyRuntimeError::new_err(message),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Trait for Python wrapper types that can be converted to/from Rust types
/// RustタイプとPythonラッパータイプ間の変換が可能な型用の特性
pub trait PyWrapper<T> {
    /// Convert from Rust type to Python wrapper
    /// RustタイプからPythonラッパーに変換
    fn from_rust(value: T) -> Self;

    /// Convert from Python wrapper to Rust type
    /// PythonラッパーからRustタイプに変換
    fn to_rust(&self) -> &T;

    /// Convert from Python wrapper to owned Rust type
    /// Pythonラッパーから所有権付きRustタイプに変換
    fn into_rust(self) -> T;
}

/// Trait for thread-safe Python wrappers
/// スレッドセーフなPythonラッパー用の特性
pub trait ThreadSafePyWrapper<T> {
    /// Create from thread-safe Rust type
    /// スレッドセーフなRustタイプから作成
    fn from_arc_rwlock(value: Arc<RwLock<T>>) -> Self;

    /// Get reference to thread-safe Rust type
    /// スレッドセーフなRustタイプへの参照を取得
    fn as_arc_rwlock(&self) -> &Arc<RwLock<T>>;

    /// Get cloned thread-safe Rust type
    /// スレッドセーフなRustタイプのクローンを取得
    fn clone_arc_rwlock(&self) -> Arc<RwLock<T>>;
}

/// Common validation utilities
/// 共通検証ユーティリティ
pub mod validation {
    use super::*;

    /// Validate tensor dimensions
    /// テンソル次元の検証
    pub fn validate_dimensions(dims: &[usize]) -> PyResult<()> {
        if dims.is_empty() {
            return Err(PyValueError::new_err("Tensor dimensions cannot be empty"));
        }

        if dims.iter().any(|&d| d == 0) {
            return Err(PyValueError::new_err("Tensor dimensions cannot contain zero"));
        }

        let total_elements: usize = dims.iter().product();
        if total_elements > 1_000_000_000 {
            return Err(PyValueError::new_err("Tensor too large (>1B elements)"));
        }

        Ok(())
    }

    /// Validate learning rate
    /// 学習率の検証
    pub fn validate_learning_rate(lr: f64) -> PyResult<()> {
        if lr <= 0.0 || lr > 1.0 {
            return Err(PyValueError::new_err("Learning rate must be in (0, 1]"));
        }
        Ok(())
    }

    /// Validate beta parameters for optimizers
    /// オプティマイザーのベータパラメータ検証
    pub fn validate_beta(beta: f64, name: &str) -> PyResult<()> {
        if !(0.0..1.0).contains(&beta) {
            return Err(PyValueError::new_err(format!("{} must be in [0, 1)", name)));
        }
        Ok(())
    }

    /// Validate epsilon parameter
    /// イプシロンパラメータの検証
    pub fn validate_epsilon(eps: f64) -> PyResult<()> {
        if eps <= 0.0 {
            return Err(PyValueError::new_err("Epsilon must be positive"));
        }
        Ok(())
    }
}

/// Common conversion utilities for NumPy interop
/// NumPy相互運用のための共通変換ユーティリティ
pub mod conversions {
    use super::*;
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};

    /// Convert Vec<f32> to Python array
    /// Vec<f32>をPython配列に変換
    pub fn vec_to_pyarray<'py>(vec: Vec<f32>, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vec.into_pyarray(py)
    }

    /// Convert Python array to Vec<f32>
    /// Python配列をVec<f32>に変換
    pub fn pyarray_to_vec(array: PyReadonlyArray1<f32>) -> Vec<f32> {
        array.as_array().to_vec()
    }

    /// Convert Python list to Vec<usize> with validation
    /// Python listをVec<usize>に検証付きで変換
    pub fn pylist_to_vec_usize(
        list: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<usize>> {
        let mut result = Vec::with_capacity(list.len());

        for (i, item) in list.iter().enumerate() {
            let value: usize = item.extract()
                .map_err(|_| PyTypeError::new_err(format!("Item {} is not a valid integer", i)))?;
            result.push(value);
        }

        Ok(result)
    }

    /// Convert Python list to Vec<f32> with validation
    /// Python listをVec<f32>に検証付きで変換
    pub fn pylist_to_vec_f32(
        list: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<f32>> {
        let mut result = Vec::with_capacity(list.len());

        for (i, item) in list.iter().enumerate() {
            let value: f32 = item.extract()
                .map_err(|_| PyTypeError::new_err(format!("Item {} is not a valid float", i)))?;
            result.push(value);
        }

        Ok(result)
    }

    /// Safe tensor shape conversion
    /// 安全なテンソル形状変換
    pub fn pylist_to_shape(
        list: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<usize>> {
        let shape = pylist_to_vec_usize(list)?;
        crate::python::common::validation::validate_dimensions(&shape)?;
        Ok(shape)
    }
}

/// Memory management utilities
/// メモリ管理ユーティリティ
pub mod memory {
    use super::*;

    /// Safe access to Arc<RwLock<T>> with timeout
    /// タイムアウト付きArc<RwLock<T>>への安全アクセス
    pub fn safe_read<T, F, R>(arc_lock: &Arc<RwLock<T>>, f: F) -> PyResult<R>
    where
        F: FnOnce(&T) -> R,
    {
        match arc_lock.try_read() {
            Ok(guard) => Ok(f(&*guard)),
            Err(_) => Err(PyRuntimeError::new_err("Failed to acquire read lock")),
        }
    }

    /// Safe mutable access to Arc<RwLock<T>> with timeout
    /// タイムアウト付きArc<RwLock<T>>への安全な可変アクセス
    pub fn safe_write<T, F, R>(arc_lock: &Arc<RwLock<T>>, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut T) -> R,
    {
        match arc_lock.try_write() {
            Ok(mut guard) => Ok(f(&mut *guard)),
            Err(_) => Err(PyRuntimeError::new_err("Failed to acquire write lock")),
        }
    }
}

/// Macro for implementing common PyO3 methods
/// 共通PyO3メソッド実装用マクロ
#[macro_export]
macro_rules! impl_py_common_methods {
    ($type:ty, $rust_type:ty) => {
        #[pymethods]
        impl $type {
            /// String representation
            /// 文字列表現
            fn __repr__(&self) -> String {
                format!("{}(...)", stringify!($type))
            }

            /// Clone the object
            /// オブジェクトのクローン
            fn __copy__(&self) -> Self {
                self.clone()
            }

            /// Deep copy the object
            /// オブジェクトの深いコピー
            fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyDict>) -> Self {
                self.clone()
            }
        }
    };
}

/// Macro for implementing thread-safe wrapper methods
/// スレッドセーフラッパーメソッド実装用マクロ
#[macro_export]
macro_rules! impl_thread_safe_wrapper {
    ($type:ty, $rust_type:ty) => {
        impl ThreadSafePyWrapper<$rust_type> for $type {
            fn from_arc_rwlock(value: Arc<RwLock<$rust_type>>) -> Self {
                Self { inner: value }
            }

            fn as_arc_rwlock(&self) -> &Arc<RwLock<$rust_type>> {
                &self.inner
            }

            fn clone_arc_rwlock(&self) -> Arc<RwLock<$rust_type>> {
                Arc::clone(&self.inner)
            }
        }
    };
}