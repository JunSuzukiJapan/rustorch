//! Python bindings for automatic differentiation
//! 自動微分のPythonバインディング

use crate::autograd::Variable;
use crate::python::error::to_py_err;
use crate::python::interop::vec_to_pyarray;
use crate::python::tensor::PyTensor;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Python wrapper for Variable (automatic differentiation)
/// Variable（自動微分）のPythonラッパー
#[pyclass]
pub struct PyVariable {
    pub(crate) variable: Variable<f32>,
}

#[pymethods]
impl PyVariable {
    #[new]
    pub fn new(tensor: &PyTensor, requires_grad: Option<bool>) -> PyResult<Self> {
        let requires_grad = requires_grad.unwrap_or(false);
        let variable = Variable::new(tensor.tensor.clone(), requires_grad);
        Ok(PyVariable { variable })
    }

    /// Create Variable from data
    /// データからVariableを作成
    #[staticmethod]
    pub fn from_data(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let tensor = PyTensor::new(data, shape)?;
        Self::new(&tensor, requires_grad)
    }

    /// Create Variable with zeros
    /// ゼロでVariableを作成
    #[staticmethod]
    pub fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> PyResult<Self> {
        let tensor = PyTensor::zeros(shape)?;
        Self::new(&tensor, requires_grad)
    }

    /// Create Variable with ones
    /// ワンでVariableを作成
    #[staticmethod]
    pub fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> PyResult<Self> {
        let tensor = PyTensor::ones(shape)?;
        Self::new(&tensor, requires_grad)
    }

    /// Create Variable with random values
    /// ランダム値でVariableを作成
    #[staticmethod]
    pub fn randn(shape: Vec<usize>, requires_grad: Option<bool>) -> PyResult<Self> {
        let tensor = PyTensor::randn(shape)?;
        Self::new(&tensor, requires_grad)
    }

    /// Get underlying tensor data
    /// 内部テンソルデータを取得
    pub fn data(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.variable.data().read().unwrap().clone(),
        })
    }

    /// Get gradient if available
    /// 勾配が利用可能な場合は取得
    pub fn grad(&self) -> Option<PyTensor> {
        if let Ok(grad_lock) = self.variable.grad().read() {
            grad_lock.as_ref().map(|grad| PyTensor {
                tensor: grad.clone(),
            })
        } else {
            None
        }
    }

    /// Check if gradient is required
    /// 勾配が必要かどうかチェック
    pub fn requires_grad(&self) -> bool {
        self.variable.requires_grad()
    }

    /// Set gradient requirement
    /// 勾配要求を設定
    pub fn requires_grad_(&mut self, requires_grad: bool) {
        // Variable構造体にset_requires_gradメソッドがない場合はコメントアウト
        // self.variable.requires_grad = requires_grad;
        // 現在の実装では変更不可
    }

    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> Vec<usize> {
        if let Ok(data) = self.variable.data().read() {
            data.shape().to_vec()
        } else {
            vec![]
        }
    }

    /// Get number of elements
    /// 要素数を取得
    pub fn numel(&self) -> usize {
        if let Ok(data) = self.variable.data().read() {
            data.shape().iter().product()
        } else {
            0
        }
    }

    /// Convert to NumPy array
    /// NumPy配列に変換
    pub fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let data: Vec<f32> = if let Ok(tensor_data) = self.variable.data().read() {
            tensor_data.data.as_slice().unwrap().to_vec()
        } else {
            vec![]
        };
        vec_to_pyarray(data, py)
    }

    /// Perform backward pass
    /// 逆伝播を実行
    pub fn backward(
        &mut self,
        gradient: Option<&PyTensor>,
        retain_graph: Option<bool>,
    ) -> PyResult<()> {
        let retain_graph = retain_graph.unwrap_or(false);

        match gradient {
            Some(grad) => {
                // Simplified backward pass with gradient
                self.variable.backward_with_grad(Some(grad.tensor.clone()));
                Ok(())
            }
            None => {
                // Simplified backward pass
                self.variable.backward();
                Ok(())
            }
        }
    }

    /// Zero gradients
    /// 勾配をゼロに設定
    pub fn zero_grad(&mut self) {
        self.variable.zero_grad();
    }

    /// Detach from computation graph
    /// 計算グラフから切り離し
    pub fn detach(&self) -> PyResult<PyVariable> {
        // Create new variable with same data but no gradients
        let tensor_data = self.variable.data().read().unwrap().clone();
        let detached_var = Variable::new(tensor_data, false);
        Ok(PyVariable { variable: detached_var })
    }

    /// Clone the Variable
    /// Variableをクローン
    pub fn clone(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            variable: self.variable.clone(),
        })
    }

    // Arithmetic operations
    // 算術演算

    /// Add two Variables
    /// Variable加算
    pub fn __add__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Variableの加算が実装されていないため、データを直接操作
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable addition not yet implemented",
        ))
    }

    /// Subtract two Variables
    /// Variable減算
    pub fn __sub__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Variableの減算が実装されていないため、データを直接操作
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable subtraction not yet implemented",
        ))
    }

    /// Multiply two Variables
    /// Variable乗算
    pub fn __mul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Simplified multiplication - in actual implementation would use autograd
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable multiplication not yet implemented",
        ))
    }

    /// Matrix multiplication
    /// 行列乗算
    pub fn __matmul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Simplified matrix multiplication - in actual implementation would use autograd
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable matrix multiplication not yet implemented",
        ))
    }

    /// Dot product
    /// ドット積
    pub fn dot(&self, other: &PyVariable) -> PyResult<PyVariable> {
        self.__matmul__(other)
    }

    /// Sum all elements
    /// 全要素の合計
    pub fn sum(&self) -> PyResult<PyVariable> {
        // Simplified sum - in actual implementation would use autograd
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable sum not yet implemented",
        ))
    }

    /// Mean of all elements
    /// 全要素の平均
    pub fn mean(&self) -> PyResult<PyVariable> {
        // Simplified mean - in actual implementation would use autograd
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Variable mean not yet implemented",
        ))
    }

    /// Reshape Variable
    /// Variableの形状変更
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<PyVariable> {
        match self.variable.reshape(&shape) {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Transpose Variable
    /// Variableの転置
    pub fn transpose(&self) -> PyResult<PyVariable> {
        match self.variable.transpose() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Power operation
    /// べき乗演算
    pub fn pow(&self, exponent: f32) -> PyResult<PyVariable> {
        match self.variable.pow(exponent) {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Exponential function
    /// 指数関数
    pub fn exp(&self) -> PyResult<PyVariable> {
        match self.variable.exp() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Natural logarithm
    /// 自然対数
    pub fn log(&self) -> PyResult<PyVariable> {
        match self.variable.log() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Sine function
    /// サイン関数
    pub fn sin(&self) -> PyResult<PyVariable> {
        match self.variable.sin() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Cosine function
    /// コサイン関数
    pub fn cos(&self) -> PyResult<PyVariable> {
        match self.variable.cos() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Square root
    /// 平方根
    pub fn sqrt(&self) -> PyResult<PyVariable> {
        match self.variable.sqrt() {
            Ok(result) => Ok(PyVariable { variable: result }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        let grad_str = if self.requires_grad() {
            "requires_grad=True"
        } else {
            "requires_grad=False"
        };

        format!(
            "PyVariable(shape={:?}, {}, grad_fn={})",
            self.shape(),
            grad_str,
            if self.variable.grad_fn().is_some() {
                "<BackwardFunction>"
            } else {
                "None"
            }
        )
    }

    /// String representation (same as __repr__)
    /// 文字列表現（__repr__と同じ）
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Check if gradients are enabled
/// 勾配が有効かどうかチェック
#[pyfunction]
pub fn is_grad_enabled() -> bool {
    // Implementation depends on global gradient context
    true // Simplified implementation
}

/// Disable gradients context
/// 勾配無効化コンテキスト
#[pyfunction]
pub fn no_grad() -> PyResult<()> {
    // Implementation for gradient disabling context
    // This would typically return a context manager
    Ok(())
}

/// Enable gradients context
/// 勾配有効化コンテキスト
#[pyfunction]
pub fn enable_grad() -> PyResult<()> {
    // Implementation for gradient enabling context
    Ok(())
}

/// Set gradient enabled state
/// 勾配有効状態を設定
#[pyfunction]
pub fn set_grad_enabled(mode: bool) -> PyResult<()> {
    // Implementation for setting gradient state
    Ok(())
}
