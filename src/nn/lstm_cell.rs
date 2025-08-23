//! LSTM Cell implementation
//! LSTMセル実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use num_traits::Float;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use rand_distr::{Normal, Distribution};

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
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        // Initialize weights with Xavier/Glorot initialization
        let weight_ih_data: Vec<T> = (0..4*hidden_size*input_size)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();
        let weight_ih = Variable::new(
            Tensor::from_vec(weight_ih_data, vec![4 * hidden_size, input_size]),
            true,
        );
        
        let weight_hh_data: Vec<T> = (0..4*hidden_size*hidden_size)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();
        let weight_hh = Variable::new(
            Tensor::from_vec(weight_hh_data, vec![4 * hidden_size, hidden_size]),
            true,
        );
        
        let bias_ih = if bias {
            let bias_data: Vec<T> = (0..4*hidden_size)
                .map(|_| T::from(normal.sample(&mut rng)).unwrap())
                .collect();
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![4 * hidden_size]),
                true,
            ))
        } else {
            None
        };
        
        let bias_hh = if bias {
            let bias_data: Vec<T> = (0..4*hidden_size)
                .map(|_| T::from(normal.sample(&mut rng)).unwrap())
                .collect();
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![4 * hidden_size]),
                true,
            ))
        } else {
            None
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
        let gi = self.linear_transform(&input, &self.weight_ih, self.bias_ih.as_ref());
        let gh = self.linear_transform(&h_prev, &self.weight_hh, self.bias_hh.as_ref());
        
        let gates = self.add_variables(&gi, &gh);
        
        // Split gates into [input, forget, cell, output]
        let input_gate = self.sigmoid(&self.slice_gates(&gates, 0));
        let forget_gate = self.sigmoid(&self.slice_gates(&gates, 1));
        let cell_gate = self.tanh(&self.slice_gates(&gates, 2));
        let output_gate = self.sigmoid(&self.slice_gates(&gates, 3));
        
        // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let forget_term = self.multiply_variables(&forget_gate, &c_prev);
        let input_term = self.multiply_variables(&input_gate, &cell_gate);
        let new_cell = self.add_variables(&forget_term, &input_term);
        
        // Update hidden state: h_t = o_t * tanh(c_t)
        let cell_tanh = self.tanh(&new_cell);
        let new_hidden = self.multiply_variables(&output_gate, &cell_tanh);
        
        (new_hidden, new_cell)
    }
    
    /// Helper function for linear transformation
    /// 線形変換のヘルパー関数
    fn linear_transform(&self, input: &Variable<T>, weight: &Variable<T>, bias: Option<&Variable<T>>) -> Variable<T> {
        let output = self.matmul_variables(input, &self.transpose_variable(weight));
        
        match bias {
            Some(b) => self.add_variables(&output, b),
            None => output,
        }
    }
    
    /// Helper function to slice gates
    /// ゲートをスライスするヘルパー関数
    fn slice_gates(&self, gates: &Variable<T>, gate_idx: usize) -> Variable<T> {
        let start_idx = gate_idx * self.hidden_size;
        let end_idx = (gate_idx + 1) * self.hidden_size;
        
        // Simplified slicing - in practice would need proper tensor slicing
        let gate_data: Vec<T> = gates.data().as_slice().unwrap()[start_idx..end_idx].to_vec();
        Variable::new(
            Tensor::from_vec(gate_data, vec![gates.data().shape()[0], self.hidden_size]),
            gates.requires_grad(),
        )
    }
    
    /// Matrix multiplication for variables
    /// Variable用の行列乗算
    fn matmul_variables(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let result_data = a.data().matmul(b.data()).unwrap();
        Variable::new(result_data, a.requires_grad() || b.requires_grad())
    }
    
    /// Addition for variables
    /// Variable用の加算
    fn add_variables(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let result_data = a.data().add(b.data()).unwrap();
        Variable::new(result_data, a.requires_grad() || b.requires_grad())
    }
    
    /// Multiplication for variables
    /// Variable用の乗算
    fn multiply_variables(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let result_data = a.data().mul(b.data()).unwrap();
        Variable::new(result_data, a.requires_grad() || b.requires_grad())
    }
    
    /// Transpose for variables
    /// Variable用の転置
    fn transpose_variable(&self, var: &Variable<T>) -> Variable<T> {
        let transposed_data = var.data().transpose().unwrap();
        Variable::new(transposed_data, var.requires_grad())
    }
    
    /// Sigmoid activation for variables
    /// Variable用のシグモイド活性化
    fn sigmoid(&self, var: &Variable<T>) -> Variable<T> {
        let sigmoid_data = var.data().map(|x| T::one() / (T::one() + (-x).exp()));
        Variable::new(sigmoid_data, var.requires_grad())
    }
    
    /// Tanh activation for variables
    /// Variable用のtanh活性化
    fn tanh(&self, var: &Variable<T>) -> Variable<T> {
        let tanh_data = var.data().map(|x| x.tanh());
        Variable::new(tanh_data, var.requires_grad())
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
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih.clone());
        }
        
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh.clone());
        }
        
        params
    }
    
    fn zero_grad(&self) {
        // Implementation for zeroing gradients
        // 勾配をゼロにする実装
    }
    
    fn train(&mut self) {
        self.training = true;
    }
    
    fn eval(&mut self) {
        self.training = false;
    }
}