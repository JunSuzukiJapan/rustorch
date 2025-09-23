//! Neural network modules for RusTorch Python bindings
//! RusTorch Pythonバインディング用ニューラルネットワークモジュール

pub mod layers;
pub mod activation;
pub mod loss;

// Re-exports for convenience
// 便利な再エクスポート
pub use layers::{
    PyLinear,
    PyConv2d,
    PyMaxPool2d,
    PyBatchNorm1d,
    PyBatchNorm2d,
    PyDropout,
    PyFlatten,
};
pub use activation::{PyReLU, PySigmoid, PyTanh};
pub use loss::{PyMSELoss, PyCrossEntropyLoss};

use pyo3::prelude::*;
use crate::core::variable::PyVariable;

/// Common trait for neural network modules
/// ニューラルネットワークモジュール用共通トレイト
pub trait PyModule {
    /// Forward pass
    /// フォワードパス
    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable>;

    /// Call interface (makes module callable)
    /// 呼び出しインターフェース（モジュールを呼び出し可能にする）
    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }
}

/// Macro to implement common module patterns
/// 共通モジュールパターン実装用マクロ
#[macro_export]
macro_rules! impl_module {
    ($struct_name:ident, $rust_type:ty, $display_name:expr) => {
        #[pymethods]
        impl $struct_name {
            /// Forward pass through the module
            fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
                let result = self.inner.forward(&input.inner);
                Ok(PyVariable::new(result))
            }

            /// Call method (makes module callable)
            fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
                self.forward(input)
            }

            /// String representation
            fn __repr__(&self) -> String {
                format!("{}()", $display_name)
            }
        }
    };
}

/// Macro to implement modules with parameters
/// パラメータ付きモジュール実装用マクロ
#[macro_export]
macro_rules! impl_module_with_params {
    ($struct_name:ident, $rust_type:ty, $display_name:expr) => {
        #[pymethods]
        impl $struct_name {
            /// Forward pass through the module
            fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
                let result = self.inner.forward(&input.inner);
                Ok(PyVariable::new(result))
            }

            /// Call method (makes module callable)
            fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
                self.forward(input)
            }

            /// Get weight parameter
            #[getter]
            fn weight(&self) -> PyVariable {
                let params = self.inner.parameters();
                PyVariable::new(params[0].clone())
            }

            /// Get bias parameter
            #[getter]
            fn bias(&self) -> Option<PyVariable> {
                let params = self.inner.parameters();
                if params.len() > 1 {
                    Some(PyVariable::new(params[1].clone()))
                } else {
                    None
                }
            }

            /// String representation
            fn __repr__(&self) -> String {
                format!("{}(...)", $display_name)
            }
        }
    };
}

/// Macro to implement training/evaluation mode switching
/// 訓練/評価モード切り替え実装用マクロ
#[macro_export]
macro_rules! impl_train_eval {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            /// Set to training mode
            fn train(&self) {
                self.inner.train();
            }

            /// Set to evaluation mode
            fn eval(&self) {
                self.inner.eval();
            }
        }
    };
}