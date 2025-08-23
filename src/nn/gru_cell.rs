//! GRU Cell implementation
//! GRUセル実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use num_traits::Float;
use std::fmt::Debug;
use rand_distr::{Normal, Distribution};

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
pub struct GRUCell<T: Float + Send + Sync> {
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
    T: Float + Send + Sync + Debug + 'static,
{
    /// Create a new GRU cell
    /// 新しいGRUセルを作成
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        // Initialize weights with Xavier/Glorot initialization
        // GRU has 3 gates (reset, update, new) so we need 3 * hidden_size
        let weight_ih_data: Vec<T> = (0..3*hidden_size*input_size)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();
        let weight_ih = Variable::new(
            Tensor::from_vec(weight_ih_data, vec![3 * hidden_size, input_size]),
            true,
        );
        
        let weight_hh_data: Vec<T> = (0..3*hidden_size*hidden_size)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();
        let weight_hh = Variable::new(
            Tensor::from_vec(weight_hh_data, vec![3 * hidden_size, hidden_size]),
            true,
        );
        
        let bias_ih = if bias {
            let bias_data: Vec<T> = (0..3*hidden_size)
                .map(|_| T::from(normal.sample(&mut rng)).unwrap())
                .collect();
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![3 * hidden_size]),
                true,
            ))
        } else {
            None
        };
        
        let bias_hh = if bias {
            let bias_data: Vec<T> = (0..3*hidden_size)
                .map(|_| T::from(normal.sample(&mut rng)).unwrap())
                .collect();
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![3 * hidden_size]),
                true,
            ))
        } else {
            None
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
        let batch_size = input.data().shape()[0];
        
        let h_prev = match hidden {
            Some(h) => h.clone(),
            None => Variable::new(
                Tensor::zeros(&[batch_size, self.hidden_size]),
                false,
            ),
        };
        
        // Compute input gates
        let gi = self.linear_transform(&input, &self.weight_ih, self.bias_ih.as_ref());
        let gh = self.linear_transform(&h_prev, &self.weight_hh, self.bias_hh.as_ref());
        
        // Split gates: [reset, update, new]
        let i_reset = self.sigmoid(&self.slice_gates(&gi, 0));
        let i_update = self.sigmoid(&self.slice_gates(&gi, 1));
        let i_new = self.slice_gates(&gi, 2);
        
        let h_reset = self.slice_gates(&gh, 0);
        let h_update = self.slice_gates(&gh, 1);
        let h_new = self.slice_gates(&gh, 2);
        
        // Reset gate
        let reset_gate = self.sigmoid(&self.add_variables(&i_reset, &h_reset));
        
        // Update gate  
        let update_gate = self.sigmoid(&self.add_variables(&i_update, &h_update));
        
        // New gate with reset applied
        let reset_h_new = self.multiply_variables(&reset_gate, &h_new);
        let new_gate = self.tanh(&self.add_variables(&i_new, &reset_h_new));
        
        // Compute new hidden state
        // h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
        let one_minus_update = self.subtract_from_scalar(&update_gate, T::one());
        let new_hidden_part = self.multiply_variables(&one_minus_update, &new_gate);
        let old_hidden_part = self.multiply_variables(&update_gate, &h_prev);
        let new_hidden = self.add_variables(&new_hidden_part, &old_hidden_part);
        
        new_hidden
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
    
    /// Subtract variable from scalar
    /// スカラーから変数を減算
    fn subtract_from_scalar(&self, var: &Variable<T>, scalar: T) -> Variable<T> {
        let result_data = var.data().map(|x| scalar - x);
        Variable::new(result_data, var.requires_grad())
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

impl<T> Module<T> for GRUCell<T>
where
    T: Float + Send + Sync + Debug + 'static,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input, None)
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