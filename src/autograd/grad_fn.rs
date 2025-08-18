//! Gradient functions for automatic differentiation
//! 自動微分のための勾配関数

use super::GradFn;
use crate::tensor::Tensor;
use num_traits::Float;

/// Addition backward function
/// 加算の逆伝播関数
pub struct AddBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor data
    /// 最初の入力テンソルデータ
    pub input0_data: crate::tensor::Tensor<T>,
    /// Second input tensor data
    /// 2番目の入力テンソルデータ
    pub input1_data: crate::tensor::Tensor<T>,
    /// First input variable
    /// 最初の入力変数
    pub input0_var: crate::autograd::Variable<T>,
    /// Second input variable
    /// 2番目の入力変数
    pub input1_var: crate::autograd::Variable<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for AddBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        
        // Propagate gradients to input variables
        if self.input0_var.requires_grad() {
            self.input0_var.backward_with_grad(Some(grad_output.clone()));
        }
        
        if self.input1_var.requires_grad() {
            self.input1_var.backward_with_grad(Some(grad_output.clone()));
        }
        
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// Subtraction backward function
/// 減算の逆伝播関数
pub struct SubBackward<T: Float + Send + Sync + 'static> {
    /// First input tensor data
    /// 最初の入力テンソルデータ
    pub input0_data: crate::tensor::Tensor<T>,
    /// Second input tensor data
    /// 2番目の入力テンソルデータ
    pub input1_data: crate::tensor::Tensor<T>,
    /// First input variable
    /// 最初の入力変数
    pub input0_var: crate::autograd::Variable<T>,
    /// Second input variable
    /// 2番目の入力変数
    pub input1_var: crate::autograd::Variable<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for SubBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        
        // For subtraction: d(a-b)/da = 1, d(a-b)/db = -1
        let grad_input0 = grad_output.clone();
        let grad_input1 = grad_output * &Tensor::from_vec(vec![T::from(-1).unwrap()], vec![1]);
        
        // Propagate gradients to input variables
        if self.input0_var.requires_grad() {
            self.input0_var.backward_with_grad(Some(grad_input0.clone()));
        }
        
        if self.input1_var.requires_grad() {
            self.input1_var.backward_with_grad(Some(grad_input1.clone()));
        }
        
        vec![Some(grad_input0), Some(grad_input1)]
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
    /// First input variable
    /// 最初の入力変数
    pub input0_var: crate::autograd::Variable<T>,
    /// Second input variable
    /// 2番目の入力変数
    pub input1_var: crate::autograd::Variable<T>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for MulBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];
        
        // For multiplication: d(a*b)/da = b, d(a*b)/db = a
        let grad_input0 = grad_output * &self.input1_data;
        let grad_input1 = grad_output * &self.input0_data;
        
        // Propagate gradients to input variables
        if self.input0_var.requires_grad() {
            self.input0_var.backward_with_grad(Some(grad_input0.clone()));
        }
        
        if self.input1_var.requires_grad() {
            self.input1_var.backward_with_grad(Some(grad_input1.clone()));
        }
        
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
    /// First input variable
    /// 最初の入力変数
    pub input0_var: Option<crate::autograd::Variable<T>>,
    /// Second input variable
    /// 2番目の入力変数
    pub input1_var: Option<crate::autograd::Variable<T>>,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for MatMulBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        if grad_outputs.is_empty() {
            return vec![];
        }
        
        let grad_output = &grad_outputs[0];
        
        // Gradient w.r.t. first input: grad_output @ input1^T
        let grad_input0 = grad_output.matmul(&self.input1_data.transpose());
        
        // Gradient w.r.t. second input: input0^T @ grad_output
        let grad_input1 = self.input0_data.transpose().matmul(grad_output);
        
        // Propagate gradients to input variables
        if let Some(ref input0_var) = self.input0_var {
            if input0_var.requires_grad() {
                input0_var.backward_with_grad(Some(grad_input0.clone()));
            }
        }
        
        if let Some(ref input1_var) = self.input1_var {
            if input1_var.requires_grad() {
                input1_var.backward_with_grad(Some(grad_input1.clone()));
            }
        }
        
        vec![Some(grad_input0), Some(grad_input1)]
    }
}

/// Sum backward function
/// 総和の逆伝播関数
pub struct SumBackward<T: Float + Send + Sync + 'static> {
    /// Original input tensor shape for gradient broadcasting
    /// 勾配ブロードキャスト用の元の入力テンソル形状
    pub input_shape: Vec<usize>,
    /// Input variable for gradient propagation
    /// 勾配伝播用の入力変数
    pub input_var: crate::autograd::Variable<T>,
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
        
        // Propagate gradients to input variable
        if self.input_var.requires_grad() {
            let grad = self.input_var.grad();
            let mut grad_lock = grad.write().unwrap();
            match grad_lock.as_mut() {
                Some(existing_grad) => {
                    *existing_grad = &*existing_grad + &result;
                }
                None => {
                    *grad_lock = Some(result.clone());
                }
            }
        }
        
        vec![Some(result)]
    }
}

/// Mean backward function
/// 平均の逆伝播関数
pub struct MeanBackward<T: Float + Send + Sync + 'static> {
    /// Input variable
    /// 入力変数
    pub input_var: Option<crate::autograd::Variable<T>>,
    /// Number of elements
    /// 要素数
    pub numel: T,
}

impl<T: Float + Send + Sync + 'static> GradFn<T> for MeanBackward<T> {
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        if grad_outputs.is_empty() {
            return vec![];
        }
        
        let grad_output = &grad_outputs[0];
        
        // For mean operation, gradient is broadcasted and divided by number of elements
        // grad_input = grad_output / numel (broadcasted to input shape)
        let grad_input_scalar = *grad_output.as_array().iter().next().unwrap_or(&T::zero()) / self.numel;
        
        // Create gradient tensor with same shape as input, filled with the scaled gradient
        if let Some(ref input_var) = self.input_var {
            let input_data = input_var.data();
            let input_shape = {
                let input_lock = input_data.read().unwrap();
                input_lock.shape().to_vec()
            };
            
            let grad_input_vec = vec![grad_input_scalar; input_shape.iter().product::<usize>()];
            let grad_input = Tensor::from_vec(grad_input_vec, input_shape);
            
            // Propagate gradient to input variable
            if input_var.requires_grad() {
                input_var.backward_with_grad(Some(grad_input.clone()));
            }
            
            vec![Some(grad_input)]
        } else {
            vec![None]
        }
    }
}