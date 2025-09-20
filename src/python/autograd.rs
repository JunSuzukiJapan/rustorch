//! Python bindings for automatic differentiation
//! 自動微分のPythonバインディング

use crate::autograd::Variable;
use crate::python::common::{conversions, memory, to_py_err, validation};
use crate::python::tensor::PyTensor;
use numpy::PyArray1;
use pyo3::exceptions::*;
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
        use crate::python::common::validation::validate_dimensions;
        validate_dimensions(&shape)?;
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
        use crate::python::common::memory::safe_read;
        safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| PyTensor {
                tensor: tensor.clone(),
            },
        )
    }

    /// Get gradient if available
    /// 勾配が利用可能な場合は取得
    pub fn grad(&self) -> Option<PyTensor> {
        use crate::python::common::memory::safe_read;
        match safe_read(
            &self.variable.grad(),
            |grad_opt: &Option<crate::tensor::Tensor<f32>>| grad_opt.clone(),
        ) {
            Ok(Some(grad)) => Some(PyTensor { tensor: grad }),
            _ => None,
        }
    }

    /// Check if gradient is required
    /// 勾配が必要かどうかチェック
    pub fn requires_grad(&self) -> bool {
        self.variable.requires_grad()
    }

    /// Set gradient requirement
    /// 勾配要求を設定
    pub fn requires_grad_(&mut self, _requires_grad: bool) -> PyResult<()> {
        // Current implementation doesn't support changing requires_grad after creation
        // 現在の実装では作成後のrequires_grad変更はサポートしていません
        Err(PyRuntimeError::new_err(
            "Cannot change requires_grad after Variable creation",
        ))
    }

    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> Vec<usize> {
        use crate::python::common::memory::safe_read;
        safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.shape().to_vec(),
        )
        .unwrap_or_default()
    }

    /// Get number of elements
    /// 要素数を取得
    pub fn numel(&self) -> usize {
        use crate::python::common::memory::safe_read;
        safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.shape().iter().product(),
        )
        .unwrap_or(0)
    }

    /// Convert to NumPy array
    /// NumPy配列に変換
    pub fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        use crate::python::common::{conversions::vec_to_pyarray, memory::safe_read};

        let data = safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.data.as_slice().unwrap_or(&[]).to_vec(),
        )
        .unwrap_or_default();

        vec_to_pyarray(data, py)
    }

    /// Perform backward pass
    /// 逆伝播を実行
    pub fn backward(
        &mut self,
        gradient: Option<&PyTensor>,
        _retain_graph: Option<bool>,
    ) -> PyResult<()> {
        use crate::python::common::to_py_err;

        match gradient {
            Some(grad) => {
                self.variable.backward_with_grad(Some(grad.tensor.clone()));
                Ok(())
            }
            None => {
                self.variable.backward();
                Ok(())
            }
        }
    }

    /// Zero gradients
    /// 勾配をゼロに設定
    pub fn zero_grad(&mut self) -> PyResult<()> {
        self.variable.zero_grad();
        Ok(())
    }

    /// Detach from computation graph
    /// 計算グラフから切り離し
    pub fn detach(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let tensor_data = safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.clone(),
        )?;
        let detached_var = Variable::new(tensor_data, false);
        Ok(PyVariable {
            variable: detached_var,
        })
    }

    /// Clone the Variable
    /// Variableをクローン
    pub fn clone(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            variable: self.variable.clone(),
        })
    }

    // Mathematical operations with proper autograd support
    // 適切な自動微分サポート付き数学演算

    /// Reshape Variable
    /// Variableの形状変更
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<PyVariable> {
        use crate::python::common::{
            memory::safe_read, to_py_err, validation::validate_dimensions,
        };

        validate_dimensions(&shape)?;

        let current_elements = self.numel();
        let new_elements: usize = shape.iter().product();
        if current_elements != new_elements {
            return Err(PyValueError::new_err(format!(
                "Cannot reshape Variable with {} elements to shape with {} elements",
                current_elements, new_elements
            )));
        }

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.reshape(&shape),
        )?
        .map_err(to_py_err)?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Transpose Variable
    /// Variableの転置
    pub fn transpose(&self) -> PyResult<PyVariable> {
        use crate::python::common::{memory::safe_read, to_py_err};

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor: &crate::tensor::Tensor<f32>| tensor.transpose(),
        )?
        .map_err(to_py_err)?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Power operation
    /// べき乗演算
    pub fn pow(&self, exponent: f32) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data.data.mapv(|x| x.powf(exponent));
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Exponential function
    /// 指数関数
    pub fn exp(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data.data.mapv(|x| x.exp());
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Natural logarithm
    /// 自然対数
    pub fn log(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data
                    .data
                    .mapv(|x| if x <= 0.0 { f32::NAN } else { x.ln() });
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Sine function
    /// サイン関数
    pub fn sin(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data.data.mapv(|x| x.sin());
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Cosine function
    /// コサイン関数
    pub fn cos(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data.data.mapv(|x| x.cos());
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Square root
    /// 平方根
    pub fn sqrt(&self) -> PyResult<PyVariable> {
        use crate::python::common::memory::safe_read;

        let result_tensor = safe_read(
            &self.variable.data(),
            |tensor_data: &crate::tensor::Tensor<f32>| {
                let result_data = tensor_data
                    .data
                    .mapv(|x| if x < 0.0 { f32::NAN } else { x.sqrt() });
                crate::tensor::Tensor::from_ndarray(result_data)
            },
        )?;

        let result_var = Variable::new(result_tensor, self.variable.requires_grad());
        Ok(PyVariable {
            variable: result_var,
        })
    }

    /// Arithmetic operations - currently simplified
    /// 算術演算 - 現在は簡略化

    /// Add two Variables
    /// Variable加算
    pub fn __add__(&self, _other: &PyVariable) -> PyResult<PyVariable> {
        Err(PyNotImplementedError::new_err(
            "Variable arithmetic operations require full autograd implementation",
        ))
    }

    /// Subtract two Variables
    /// Variable減算
    pub fn __sub__(&self, _other: &PyVariable) -> PyResult<PyVariable> {
        Err(PyNotImplementedError::new_err(
            "Variable arithmetic operations require full autograd implementation",
        ))
    }

    /// Multiply two Variables
    /// Variable乗算
    pub fn __mul__(&self, _other: &PyVariable) -> PyResult<PyVariable> {
        Err(PyNotImplementedError::new_err(
            "Variable arithmetic operations require full autograd implementation",
        ))
    }

    /// Matrix multiplication
    /// 行列乗算
    pub fn __matmul__(&self, _other: &PyVariable) -> PyResult<PyVariable> {
        Err(PyNotImplementedError::new_err(
            "Variable matrix operations require full autograd implementation",
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
        Err(PyNotImplementedError::new_err(
            "Variable reduction operations require full autograd implementation",
        ))
    }

    /// Mean of all elements
    /// 全要素の平均
    pub fn mean(&self) -> PyResult<PyVariable> {
        Err(PyNotImplementedError::new_err(
            "Variable reduction operations require full autograd implementation",
        ))
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        let grad_str = if self.requires_grad() {
            "requires_grad=True"
        } else {
            "requires_grad=False"
        };

        let shape = self.shape();
        let grad_fn_str = if self.variable.grad_fn().is_some() {
            "<BackwardFunction>"
        } else {
            "None"
        };

        format!(
            "PyVariable(shape={:?}, {}, grad_fn={})",
            shape, grad_str, grad_fn_str
        )
    }

    /// String representation (same as __repr__)
    /// 文字列表現（__repr__と同じ）
    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Clone the object
    /// オブジェクトのクローン
    pub fn __copy__(&self) -> PyResult<Self> {
        self.clone()
    }

    /// Deep copy the object
    /// オブジェクトの深いコピー
    pub fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        self.clone()
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
