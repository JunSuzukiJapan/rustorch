//! GRU Cell implementation
//! GRUセル実装

use crate::autograd::Variable;
use crate::nn::{
    recurrent_common::{collect_recurrent_parameters, RecurrentOps},
    Module,
};
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

/// GRU cell implementation
/// GRUセルの実装
///
/// A GRU cell computes:
/// GRUセルは以下を計算します：
///
/// r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  (reset gate)
/// z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  (update gate)
/// n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))  (new gate)
/// h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}  (hidden state)
#[derive(Debug)]
pub struct GRUCell<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// Input-to-hidden weights for all gates [reset, update, new]
    /// 全ゲート用の入力から隠れ状態への重み [reset, update, new]
    weight_ih: Variable<T>,

    /// Hidden-to-hidden weights for all gates [reset, update, new]
    /// 全ゲート用の隠れ状態から隠れ状態への重み [reset, update, new]
    weight_hh: Variable<T>,

    /// Input-to-hidden bias for all gates
    /// 全ゲート用の入力から隠れ状態へのバイアス
    bias_ih: Option<Variable<T>>,

    /// Hidden-to-hidden bias for all gates
    /// 全ゲート用の隠れ状態から隠れ状態へのバイアス
    bias_hh: Option<Variable<T>>,

    /// Input size
    /// 入力サイズ
    input_size: usize,

    /// Hidden size
    /// 隠れ状態サイズ
    hidden_size: usize,

    /// Training mode flag
    /// 学習モードフラグ
    training: bool,
}

impl<T> GRUCell<T>
where
    T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    /// Create a new GRU cell
    /// 新しいGRUセルを作成
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        // Use common recurrent operations for weight initialization
        let (weight_ih, weight_hh) = RecurrentOps::init_weights(input_size, hidden_size, 3);
        let (bias_ih, bias_hh) = if bias {
            RecurrentOps::init_bias(hidden_size, 3)
        } else {
            (None, None)
        };

        GRUCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            training: true,
        }
    }

    /// Forward pass through the GRU cell
    /// GRUセルの順伝播
    pub fn forward(&self, input: &Variable<T>, hidden: Option<&Variable<T>>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let batch_size = input_data.shape()[0];

        let h_prev = match hidden {
            Some(h) => h.clone(),
            None => Variable::new(Tensor::zeros(&[batch_size, self.hidden_size]), false),
        };

        // Compute input gates
        let gi = RecurrentOps::linear_transform(&input, &self.weight_ih, self.bias_ih.as_ref());
        let gh = RecurrentOps::linear_transform(&h_prev, &self.weight_hh, self.bias_hh.as_ref());

        // Split gates: [reset, update, new]
        let i_reset = RecurrentOps::sigmoid(&RecurrentOps::slice_gates(&gi, 0, self.hidden_size));
        let i_update = RecurrentOps::sigmoid(&RecurrentOps::slice_gates(&gi, 1, self.hidden_size));
        let i_new = RecurrentOps::slice_gates(&gi, 2, self.hidden_size);

        let h_reset = RecurrentOps::slice_gates(&gh, 0, self.hidden_size);
        let h_update = RecurrentOps::slice_gates(&gh, 1, self.hidden_size);
        let h_new = RecurrentOps::slice_gates(&gh, 2, self.hidden_size);

        // Reset gate
        let reset_gate = RecurrentOps::sigmoid(&RecurrentOps::add_variables(&i_reset, &h_reset));

        // Update gate
        let update_gate = RecurrentOps::sigmoid(&RecurrentOps::add_variables(&i_update, &h_update));

        // New gate with reset applied
        let reset_h_new = RecurrentOps::multiply_variables(&reset_gate, &h_new);
        let new_gate = RecurrentOps::tanh(&RecurrentOps::add_variables(&i_new, &reset_h_new));

        // Compute new hidden state
        // h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
        let one_minus_update = RecurrentOps::subtract_from_scalar(&update_gate, T::one());
        let new_hidden_part = RecurrentOps::multiply_variables(&one_minus_update, &new_gate);
        let old_hidden_part = RecurrentOps::multiply_variables(&update_gate, &h_prev);
        let new_hidden = RecurrentOps::add_variables(&new_hidden_part, &old_hidden_part);

        new_hidden
    }

    /// Get input size
    /// 入力サイズを取得
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get hidden size
    /// 隠れ状態サイズを取得
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Set training mode
    /// 学習モードを設定
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Check if in training mode
    /// 学習モードかどうかをチェック
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl<T> Module<T> for GRUCell<T>
where
    T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input, None)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        collect_recurrent_parameters(
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}
