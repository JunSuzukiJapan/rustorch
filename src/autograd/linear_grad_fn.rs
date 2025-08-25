//! Linear layer gradient functions
//! 線形レイヤーの勾配関数

use super::GradFn;
use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;
use std::sync::{Arc, RwLock};

/// Linear layer backward function
/// 線形レイヤーの逆伝播関数
pub struct LinearBackward<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    /// Input tensor for gradient computation
    /// 勾配計算用の入力テンソル
    pub input: Arc<RwLock<Tensor<T>>>,
    /// Weight tensor for gradient computation
    /// 勾配計算用の重みテンソル
    pub weight: Arc<RwLock<Tensor<T>>>,
    /// Input variable for gradient propagation
    /// 勾配伝播用の入力変数
    pub input_var: Variable<T>,
    /// Weight variable for gradient propagation
    /// 勾配伝播用の重み変数
    pub weight_var: Variable<T>,
    /// Bias variable for gradient propagation (optional)
    /// 勾配伝播用のバイアス変数（オプション）
    pub bias_var: Option<Variable<T>>,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    GradFn<T> for LinearBackward<T>
{
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        let grad_output = &grad_outputs[0];

        // Get input and weight data
        let input_data = self.input.read().unwrap();
        let weight_data = self.weight.read().unwrap();

        // Compute gradients for input: grad_output @ weight
        let grad_input = grad_output.matmul(&weight_data).expect("MatMul failed");

        // Compute gradients for weight: grad_output.T @ input
        let grad_weight = grad_output
            .transpose()
            .expect("Transpose failed")
            .matmul(&input_data)
            .expect("MatMul failed");

        // Compute gradients for bias: sum(grad_output, axis=0)
        let grad_bias = if self.bias_var.is_some() {
            Some(grad_output.sum_axis(0).expect("Sum failed"))
        } else {
            None
        };

        // Propagate gradients to input variables
        if self.input_var.requires_grad() {
            self.input_var.backward_with_grad(Some(grad_input.clone()));
        }

        // Propagate gradients to weight variables
        if self.weight_var.requires_grad() {
            self.weight_var
                .backward_with_grad(Some(grad_weight.clone()));
        }

        // Propagate gradients to bias variables
        if let Some(ref bias_var) = self.bias_var {
            if bias_var.requires_grad() {
                bias_var.backward_with_grad(grad_bias.clone());
            }
        }

        // Return gradients (not used in this simplified implementation)
        vec![Some(grad_input), Some(grad_weight)]
    }
}
