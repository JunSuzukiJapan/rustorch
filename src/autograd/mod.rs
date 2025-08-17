use crate::tensor::Tensor;
use std::sync::{Arc, RwLock};
use num_traits::Float;
use std::marker::PhantomData;
use std::ops;

#[cfg(test)]
mod tests;

/// A variable that supports automatic differentiation.
/// 自動微分をサポートする変数
pub struct Variable<T: Float + Send + Sync> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    _marker: PhantomData<T>,
}

impl<T: Float + Send + Sync> std::fmt::Debug for Variable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

impl<T: Float + Send + Sync + 'static> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            _marker: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static> Variable<T> {
    /// Creates a new variable with the given tensor.
    /// 与えられたテンソルで新しい変数を作成します。
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
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
            if data.len() == 1 {
                // Scalar case - gradient is 1
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


    }

    /// Matrix multiplication with automatic differentiation support
    /// 自動微分をサポートする行列乗算
    pub fn matmul(&self, other: &Variable<T>) -> Variable<T> {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = other.data.read().unwrap().clone();
        let result_data = lhs_data.matmul(&rhs_data);

        if self.requires_grad || other.requires_grad {
            // For now, return without grad_fn - we'll implement this properly later
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Sum all elements with automatic differentiation support
    /// 自動微分をサポートする全要素の和
    pub fn sum(&self) -> Variable<T> {
        let input_data = self.data.read().unwrap();
        let result_data = input_data.sum(); // Use the new sum method
        
        if self.requires_grad {
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }
}

// Implement arithmetic operators for Variables
impl<T: Float + Send + Sync + 'static> ops::Add for &Variable<T> {
    type Output = Variable<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data + &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static> ops::Mul for &Variable<T> {
    type Output = Variable<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data * &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }
}
