use crate::autograd::grad_fn::{
    AddBackward, MatMulBackward, MulBackward, SubBackward, SumBackward,
};
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops;
use std::sync::{Arc, RwLock};

pub mod function;
pub mod grad_fn;
pub mod graph;
pub mod linear_grad_fn;
pub mod visualization;

#[cfg(test)]
mod tests;

/// Gradient function trait for backward computation
/// 逆伝播計算のための勾配関数トレイト
pub trait GradFn<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Apply the gradient function to compute input gradients
    /// 勾配関数を適用して入力勾配を計算
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>>;
}

/// A variable that supports automatic differentiation.
/// 自動微分をサポートする変数
pub struct Variable<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    _marker: PhantomData<T>,
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> std::fmt::Debug
    for Variable<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Clone
    for Variable<T>
{
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Variable<T>
{
    /// Creates a new variable with the given tensor.
    /// 与えられたテンソルで新しい変数を作成します。
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            grad_fn: None,
            _marker: PhantomData,
        }
    }

    /// Creates a new variable with gradient function
    /// 勾配関数付きの新しい変数を作成します
    pub fn new_with_grad_fn(
        data: Tensor<T>,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradFn<T>>>,
    ) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            grad_fn,
            _marker: PhantomData,
        }
    }

    /// Returns the data tensor.
    /// データテンソルを返します。
    pub fn data(&self) -> Arc<RwLock<Tensor<T>>> {
        self.data.clone()
    }

    /// Returns whether this variable requires gradients.
    /// この変数が勾配を必要とするかどうかを返します。
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the gradient tensor.
    /// 勾配テンソルを返します。
    pub fn grad(&self) -> Arc<RwLock<Option<Tensor<T>>>> {
        self.grad.clone()
    }

    /// Returns the gradient function if any.
    /// 勾配関数があれば返します。
    pub fn grad_fn(&self) -> &Option<Arc<dyn GradFn<T>>> {
        &self.grad_fn
    }

    /// Zeros out the gradient.
    /// 勾配をゼロクリアします。
    pub fn zero_grad(&self) {
        if let Ok(mut grad) = self.grad.write() {
            *grad = None;
        }
    }

    /// Performs backward pass to compute gradients.
    /// 逆伝播を実行して勾配を計算します。
    pub fn backward(&self) {
        self.backward_with_grad(None);
    }

    /// Performs backward pass with a specific gradient.
    /// 特定の勾配で逆伝播を実行します。
    pub fn backward_with_grad(&self, grad_output: Option<Tensor<T>>) {
        if !self.requires_grad {
            return;
        }

        // Initialize gradient if not provided
        let initial_grad = grad_output.unwrap_or_else(|| {
            let data = self.data.read().unwrap();
            if data.numel() == 1 && data.shape().is_empty() {
                // Scalar case - gradient is 1 with scalar shape
                Tensor::ones(&[])
            } else {
                // Vector/matrix case - gradient is ones with same shape
                Tensor::ones(data.shape())
            }
        });

        // Set the gradient for this variable
        if let Ok(mut grad) = self.grad.write() {
            match grad.as_mut() {
                Some(existing_grad) => {
                    // Accumulate gradients
                    *existing_grad = &*existing_grad + &initial_grad;
                }
                None => {
                    *grad = Some(initial_grad.clone());
                }
            }
        }

        // Call the gradient function if it exists (for non-leaf nodes)
        if let Some(grad_fn) = &self.grad_fn {
            let _grad_inputs = grad_fn.apply(&[initial_grad]);
        }
    }

    /// Matrix multiplication with automatic differentiation support
    /// 自動微分をサポートする行列乗算
    pub fn matmul(&self, other: &Variable<T>) -> Variable<T> {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = other.data.read().unwrap().clone();
        let result_data = lhs_data.matmul(&rhs_data).expect("MatMul failed");

        if self.requires_grad || other.requires_grad {
            let grad_fn = Arc::new(MatMulBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: Some(self.clone()),
                input1_var: Some(other.clone()),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Transpose the last two dimensions
    /// 最後の2次元を転置
    pub fn transpose_last_two(&self) -> Variable<T> {
        let input_data = self.data.read().unwrap();
        let result_data = input_data.transpose().expect("Transpose failed");

        // For now, no gradient support for transpose
        // 現在のところ、転置の勾配サポートはなし
        Variable::new(result_data, false)
    }

    /// Sum all elements with automatic differentiation support
    /// 自動微分をサポートする全要素の和
    pub fn sum(&self) -> Variable<T> {
        let input_data = self.data.read().unwrap();
        let input_shape = input_data.shape().to_vec();
        let sum_value = input_data.sum();
        let result_data = Tensor::from_vec(vec![sum_value], vec![1]);

        if self.requires_grad {
            let grad_fn = Arc::new(SumBackward {
                input_shape,
                input_var: self.clone(),
                _phantom: PhantomData,
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Power function with automatic differentiation support
    /// 自動微分をサポートするべき乗関数
    pub fn pow(&self, exponent: T) -> Variable<T> {
        let input_data = self.data.read().unwrap().clone();
        let mut result_data = input_data.clone();
        result_data
            .as_array_mut()
            .mapv_inplace(|x| x.powf(exponent));

        if self.requires_grad {
            // For now, return without proper gradient function
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Mean of all elements with automatic differentiation support
    /// 自動微分をサポートする全要素の平均
    pub fn mean_autograd(&self) -> Variable<T> {
        let sum_var = self.sum();
        let input_data = self.data.read().unwrap();
        let numel = T::from(input_data.numel()).unwrap();

        let sum_data = sum_var.data.read().unwrap().clone();
        let mut mean_data = sum_data;
        mean_data.as_array_mut().mapv_inplace(|x| x / numel);

        if self.requires_grad {
            let grad_fn = std::sync::Arc::new(crate::autograd::grad_fn::MeanBackward {
                input_var: Some(self.clone()),
                numel,
            });
            Variable::new_with_grad_fn(mean_data, true, Some(grad_fn))
        } else {
            Variable::new(mean_data, false)
        }
    }
}

// Implement arithmetic operators for Variables
impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add
    for &Variable<T>
{
    type Output = Variable<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data + &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(AddBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul
    for &Variable<T>
{
    type Output = Variable<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data * &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(MulBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub
    for &Variable<T>
{
    type Output = Variable<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data - &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(SubBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    ops::Sub<&Variable<T>> for Variable<T>
{
    type Output = Variable<T>;

    fn sub(self, rhs: &Variable<T>) -> Self::Output {
        &self - rhs
    }
}
