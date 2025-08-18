//! Gradient functions for automatic differentiation
//! 自動微分のための勾配関数

use super::GradFn;
use crate::tensor::Tensor;
use num_traits::Float;
use std::sync::{Weak, RwLock};

/// Addition backward function
/// 加算の逆伝播関数
pub struct AddBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor reference
    /// 最初の入力テンソル参照
    pub input0: Weak<RwLock<Option<Tensor<T>>>>,
    /// Second input tensor reference
    /// 2番目の入力テンソル参照
    pub input1: Weak<RwLock<Option<Tensor<T>>>>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for AddBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// Subtraction backward function
/// 減算の逆伝播関数
pub struct SubBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor reference
    /// 最初の入力テンソル参照
    pub input0: Weak<RwLock<Option<Tensor<T>>>>,
    /// Second input tensor reference
    /// 2番目の入力テンソル参照
    pub input1: Weak<RwLock<Option<Tensor<T>>>>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for SubBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        vec![Some(grad_output.clone()), Some(-grad_output)]
    }
}

/// Multiplication backward function
/// 乗算の逆伝播関数
pub struct MulBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor data for gradient computation
    /// 勾配計算用の最初の入力テンソルデータ
    pub input0_data: Tensor<T>,
    /// Second input tensor data for gradient computation
    /// 勾配計算用の2番目の入力テンソルデータ
    pub input1_data: Tensor<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for MulBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        let grad_input0 = grad_output * &self.input1_data;
        let grad_input1 = grad_output * &self.input0_data;
        vec![Some(grad_input0), Some(grad_input1)]
    }
}

/// Matrix multiplication backward function
/// 行列乗算の逆伝播関数
pub struct MatMulBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor data for gradient computation
    /// 勾配計算用の最初の入力テンソルデータ
    pub input0_data: Tensor<T>,
    /// Second input tensor data for gradient computation
    /// 勾配計算用の2番目の入力テンソルデータ
    pub input1_data: Tensor<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for MatMulBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        let grad_input0 = grad_output.matmul(&self.input1_data.transpose());
        let grad_input1 = self.input0_data.transpose().matmul(grad_output);
        vec![Some(grad_input0), Some(grad_input1)]
    }
}

/// Sum backward function
/// 総和の逆伝播関数
pub struct SumBackward<T: Float + Send + Sync + 'static> {
    /// Original input tensor shape for gradient broadcasting
    /// 勾配ブロードキャスト用の元の入力テンソル形状
    pub input_shape: Vec<usize>,
    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    pub _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for SumBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        // For sum, gradient is broadcast to match input shape
        let grad_expanded = Tensor::ones(&self.input_shape);
        // Multiply by the gradient output value
        let grad_value = grad_output.as_array().iter().next().copied().unwrap_or(T::zero());
        let mut result = grad_expanded;
        result.as_array_mut().mapv_inplace(|_| grad_value);
        vec![Some(result)]
    }
}