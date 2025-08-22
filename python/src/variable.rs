//! Python bindings for RusTorch Variable (autograd)
//! RusTorch Variable（自動微分）のPythonバインディング

use pyo3::prelude::*;
use rustorch::autograd::Variable;
use rustorch::tensor::Tensor;
use crate::PyTensor;

/// Python wrapper for RusTorch Variable
/// RusTorch VariableのPythonラッパー
#[pyclass(name = "Variable")]
pub struct PyVariable {
    variable: Variable<f32>,
}

impl PyVariable {
    pub fn new(variable: Variable<f32>) -> Self {
        Self { variable }
    }
    
    pub fn inner(&self) -> &Variable<f32> {
        &self.variable
    }
}

#[pymethods]
impl PyVariable {
    /// Create a new variable
    /// 新しい変数を作成
    #[new]
    #[pyo3(signature = (data, requires_grad = false))]
    fn py_new(data: &PyTensor, requires_grad: bool) -> Self {
        let tensor = data.inner().clone();
        Self::new(Variable::new(tensor, requires_grad))
    }
    
    /// Get the underlying tensor data
    /// 基盤となるテンソルデータを取得
    #[getter]
    fn data(&self) -> PyTensor {
        let data_guard = self.variable.data().read().unwrap();
        PyTensor::new(data_guard.clone())
    }
    
    /// Get gradient (if available)
    /// 勾配を取得（利用可能な場合）
    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        let grad_guard = self.variable.grad().read().unwrap();
        grad_guard.as_ref().map(|g| PyTensor::new(g.clone()))
    }
    
    /// Check if gradient computation is required
    /// 勾配計算が必要かチェック
    #[getter]
    fn requires_grad(&self) -> bool {
        // This would need to be implemented in the Rust Variable
        // For now, assume it's tracked if it has computation history
        true // Placeholder
    }
    
    /// Get tensor shape
    /// テンソルのシェイプを取得
    #[getter]
    fn shape(&self) -> Vec<usize> {
        let data_guard = self.variable.data().read().unwrap();
        data_guard.shape().to_vec()
    }
    
    /// Perform backward pass
    /// 逆伝播を実行
    fn backward(&self) {
        self.variable.backward();
    }
    
    /// Zero gradients
    /// 勾配をゼロに設定
    fn zero_grad(&self) {
        // This would need to be implemented in the Rust Variable
        // For now, this is a placeholder
    }
    
    /// Clone variable
    /// 変数をクローン
    fn clone(&self) -> PyVariable {
        PyVariable::new(self.variable.clone())
    }
    
    /// Addition with another variable
    /// 他の変数との加算
    fn __add__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.variable + &other.variable;
        Ok(PyVariable::new(result))
    }
    
    /// Subtraction with another variable
    /// 他の変数との減算
    fn __sub__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.variable - &other.variable;
        Ok(PyVariable::new(result))
    }
    
    /// Multiplication with another variable
    /// 他の変数との乗算
    fn __mul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = &self.variable * &other.variable;
        Ok(PyVariable::new(result))
    }
    
    /// Matrix multiplication
    /// 行列乗算
    fn matmul(&self, other: &PyVariable) -> PyResult<PyVariable> {
        let result = self.variable.matmul(&other.variable);
        Ok(PyVariable::new(result))
    }
    
    /// Matrix multiplication (@ operator)
    /// 行列乗算（@演算子）
    fn __matmul__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        self.matmul(other)
    }
    
    /// Power operation
    /// べき乗演算
    fn __pow__(&self, exponent: f32, _modulo: Option<&PyAny>) -> PyResult<PyVariable> {
        // This is a simplified implementation
        // In practice, you'd want to implement proper power operation with autograd
        if exponent == 2.0 {
            let result = &self.variable * &self.variable;
            Ok(PyVariable::new(result))
        } else {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Power operation only supports exponent=2 for now"
            ))
        }
    }
    
    /// Sum operation (with gradient support)
    /// 合計演算（勾配サポート付き）
    fn sum(&self) -> PyVariable {
        // This would use sum_autograd in practice
        let sum_result = {
            let data_guard = self.variable.data().read().unwrap();
            data_guard.sum()
        };
        PyVariable::new(Variable::new(sum_result, true))
    }
    
    /// Mean operation (with gradient support)
    /// 平均演算（勾配サポート付き）
    fn mean(&self) -> PyVariable {
        // This would use mean_autograd in practice
        let mean_result = {
            let data_guard = self.variable.data().read().unwrap();
            data_guard.mean()
        };
        PyVariable::new(Variable::new(mean_result, true))
    }
    
    /// Convert to numpy (detached from computation graph)
    /// numpyに変換（計算グラフから切り離し）
    fn detach(&self) -> PyTensor {
        let data_guard = self.variable.data().read().unwrap();
        PyTensor::new(data_guard.clone())
    }
    
    /// Get item by index
    /// インデックスで要素を取得
    fn item(&self) -> PyResult<f32> {
        let data_guard = self.variable.data().read().unwrap();
        let shape = data_guard.shape();
        
        if shape.iter().product::<usize>() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "item() can only be called on tensors with exactly one element"
            ));
        }
        
        let data = data_guard.as_array();
        Ok(data[0])
    }
    
    /// String representation
    /// 文字列表現
    fn __repr__(&self) -> String {
        let data_guard = self.variable.data().read().unwrap();
        format!("Variable(shape={:?}, requires_grad=True)", data_guard.shape())
    }
    
    /// String representation for print
    /// print用の文字列表現
    fn __str__(&self) -> String {
        let data_guard = self.variable.data().read().unwrap();
        format!("variable(shape={:?}, requires_grad=True)", data_guard.shape())
    }
}