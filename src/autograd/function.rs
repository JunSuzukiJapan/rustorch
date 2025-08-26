//! Automatic differentiation functions
//! 自動微分関数

use crate::tensor::Tensor;
use num_traits::Float;

/// Function trait for automatic differentiation
/// 自動微分のための関数トレイト
pub trait Function<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Apply the forward pass
    /// 順伝播を適用
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T>;

    /// Apply the backward pass
    /// 逆伝播を適用
    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>>;
}

/// Addition function
/// 加算関数
#[derive(Debug)]
pub struct AddFunction;

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Function<T> for AddFunction
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        inputs[0] + inputs[1]
    }

    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// Subtraction function
/// 減算関数
#[derive(Debug)]
pub struct SubFunction;

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Function<T> for SubFunction
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        inputs[0] - inputs[1]
    }

    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        vec![Some(grad_output.clone()), Some(-grad_output)]
    }
}

/// Multiplication function
/// 乗算関数
#[derive(Debug)]
pub struct MulFunction;

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Function<T> for MulFunction
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        inputs[0] * inputs[1]
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_input0 = grad_output * inputs[1];
        let grad_input1 = grad_output * inputs[0];
        vec![Some(grad_input0), Some(grad_input1)]
    }
}

/// Matrix multiplication function
/// 行列乗算関数
#[derive(Debug)]
pub struct MatMulFunction;

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Function<T> for MatMulFunction
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        inputs[0]
            .matmul(inputs[1])
            .expect("Matrix multiplication failed")
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_input0 = grad_output
            .matmul(&inputs[1].transpose().expect("Transpose failed"))
            .expect("MatMul failed");
        let grad_input1 = inputs[0]
            .transpose()
            .expect("Transpose failed")
            .matmul(grad_output)
            .expect("MatMul failed");
        vec![Some(grad_input0), Some(grad_input1)]
    }
}

/// Sum function
/// 総和関数
#[derive(Debug)]
pub struct SumFunction;

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Function<T> for SumFunction
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        let sum_value = inputs[0].sum();
        Tensor::from_vec(vec![sum_value], vec![1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        // For sum, gradient is broadcast to match input shape
        let grad_expanded = Tensor::ones(inputs[0].shape());
        // Multiply by the gradient output value
        let grad_value = grad_output
            .as_array()
            .iter()
            .next()
            .copied()
            .unwrap_or(T::zero());
        let mut result = grad_expanded;
        result.as_array_mut().mapv_inplace(|_| grad_value);
        vec![Some(result)]
    }
}
