use crate::tensor::Tensor;
use std::sync::{Arc, RwLock};
use num_traits::Float;
use std::marker::PhantomData;

/// A variable that supports automatic differentiation.
/// 自動微分をサポートする変数
pub struct Variable<T: Float> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    grad_fn: Option<Arc<dyn Fn(&Tensor<T>) -> (Option<Tensor<T>>, Option<Tensor<T>>, Option<Tensor<T>>) + Send + Sync>>,
    input: Option<Arc<RwLock<Variable<T>>>>,
    weight: Option<Arc<RwLock<Variable<T>>>>,
    bias: Option<Arc<RwLock<Variable<T>>>>,
    _marker: PhantomData<T>,
}

impl<T: Float> std::fmt::Debug for Variable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

impl<T: Float + 'static> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            input: self.input.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Float + 'static> Variable<T> {
    /// Creates a new variable with the given tensor.
    /// 与えられたテンソルで新しい変数を作成します。
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        let shape = data.shape().to_vec();
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(if requires_grad {
                Some(Tensor::zeros(shape.as_slice()))
            } else {
                None
            })),
            requires_grad,
            grad_fn: None,
            input: None,
            weight: None,
            bias: None,
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
}
