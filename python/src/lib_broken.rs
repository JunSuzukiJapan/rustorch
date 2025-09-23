//! RusTorch Python bindings - Refactored modular version
//! RusTorch Pythonバインディング - リファクタリング済みモジュラー版

use pyo3::prelude::*;

// Import core modules
// コアモジュールをインポート
pub mod core;
pub mod nn;
pub mod optim;

// Re-export for external use
// 外部使用のための再エクスポート
pub use core::tensor::PyTensor;
pub use core::variable::PyVariable;

// NN layers
pub use nn::layers::linear::PyLinear;
pub use nn::layers::conv::{PyConv2d, PyMaxPool2d};
pub use nn::layers::norm::{PyBatchNorm1d, PyBatchNorm2d};
pub use nn::layers::dropout::PyDropout;
pub use nn::layers::flatten::PyFlatten;

// Activation functions
pub use nn::activation::{PyReLU, PySigmoid, PyTanh};

// Loss functions
pub use nn::loss::{PyMSELoss, PyCrossEntropyLoss};

// Optimizers
pub use optim::sgd::PySGD;
pub use optim::adam::PyAdam;

// Import tensor creation functions
// テンソル作成関数をインポート
use core::tensor::{zeros, ones, tensor};

/// RusTorch Python bindings module - Refactored version
/// RusTorch Pythonバインディングモジュール - リファクタリング版
#[pymodule]
fn _rustorch_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    // コアクラス
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;

    // Neural network layers
    // ニューラルネットワーク層
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyFlatten>()?;

    // Activation functions
    // 活性化関数
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;

    // Loss functions
    // 損失関数
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;

    // Optimizers
    // オプティマイザー
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;

    // Tensor creation functions
    // テンソル作成関数
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Test basic module loading
        // 基本的なモジュール読み込みテスト
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "test_rustorch").unwrap();
            _rustorch_py(&module).unwrap();

            // Verify classes are available
            // クラスが利用可能であることを確認
            assert!(module.getattr("Tensor").is_ok());
            assert!(module.getattr("Variable").is_ok());
            assert!(module.getattr("Linear").is_ok());
        });
    }
}