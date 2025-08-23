//! LSTM Cell implementation
//! LSTMセル実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::{Module, recurrent_common::{RecurrentOps, collect_recurrent_parameters}};
use num_traits::Float;
use std::fmt::Debug;

/// LSTM cell implementation
/// LSTMセルの実装
/// 
/// An LSTM cell computes:
/// LSTMセルは以下を計算します：
/// 
/// i_t = σ(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  (input gate)
/// f_t = σ(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  (forget gate) 
/// g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)  (cell gate)
/// o_t = σ(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  (output gate)
/// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  (cell state)
/// h_t = o_t ⊙ tanh(c_t)  (hidden state)
#[derive(Debug)]
pub struct LSTMCell<T: Float + Send + Sync> {
    /// Input-to-hidden weights for all gates [input, forget, cell, output]
    /// 全ゲート用の入力から隠れ状態への重み [input, forget, cell, output]
    weight_ih: Variable<T>,
    
    /// Hidden-to-hidden weights for all gates [input, forget, cell, output]
    /// 全ゲート用の隠れ状態から隠れ状態への重み [input, forget, cell, output]
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

impl<T> LSTMCell<T>
where
    T: Float + Send + Sync + Debug + 'static,
{
    /// Create a new LSTM cell
    /// 新しいLSTMセルを作成
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        // Use common recurrent operations for weight initialization
        let (weight_ih, weight_hh) = RecurrentOps::init_weights(input_size, hidden_size, 4);
        let (bias_ih, bias_hh) = if bias {
            RecurrentOps::init_bias(hidden_size, 4)
        } else {
            (None, None)
        };
        
        LSTMCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            training: true,
        }
    }
    
    /// Forward pass through the LSTM cell
    /// LSTMセルの順伝播
    pub fn forward(&self, input: &Variable<T>, hidden: Option<(&Variable<T>, &Variable<T>)>) -> (Variable<T>, Variable<T>) {
        let batch_size = input.data().shape()[0];
        
        let (h_prev, c_prev) = match hidden {
            Some((h, c)) => (h.clone(), c.clone()),
            None => {
                let h = Variable::new(
                    Tensor::zeros(&[batch_size, self.hidden_size]),
                    false,
                );
                let c = Variable::new(
                    Tensor::zeros(&[batch_size, self.hidden_size]),
                    false,
                );
                (h, c)
            }
        };
        
        // Compute gate values
        let gi = RecurrentOps::linear_transform(&input, &self.weight_ih, self.bias_ih.as_ref());
        let gh = RecurrentOps::linear_transform(&h_prev, &self.weight_hh, self.bias_hh.as_ref());
        
        let gates = RecurrentOps::add_variables(&gi, &gh);
        
        // Split gates into [input, forget, cell, output]
        let input_gate = RecurrentOps::sigmoid(&RecurrentOps::slice_gates(&gates, 0, self.hidden_size));
        let forget_gate = RecurrentOps::sigmoid(&RecurrentOps::slice_gates(&gates, 1, self.hidden_size));
        let cell_gate = RecurrentOps::tanh(&RecurrentOps::slice_gates(&gates, 2, self.hidden_size));
        let output_gate = RecurrentOps::sigmoid(&RecurrentOps::slice_gates(&gates, 3, self.hidden_size));
        
        // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let forget_term = RecurrentOps::multiply_variables(&forget_gate, &c_prev);
        let input_term = RecurrentOps::multiply_variables(&input_gate, &cell_gate);
        let new_cell = RecurrentOps::add_variables(&forget_term, &input_term);
        
        // Update hidden state: h_t = o_t * tanh(c_t)
        let cell_tanh = RecurrentOps::tanh(&new_cell);
        let new_hidden = RecurrentOps::multiply_variables(&output_gate, &cell_tanh);
        
        (new_hidden, new_cell)
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

impl<T> Module<T> for LSTMCell<T>
where
    T: Float + Send + Sync + Debug + 'static,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For Module trait compatibility, return only hidden state
        let (hidden, _cell) = self.forward(input, None);
        hidden
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        collect_recurrent_parameters(&self.weight_ih, &self.weight_hh, &self.bias_ih, &self.bias_hh)
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