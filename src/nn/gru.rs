//! Gated Recurrent Unit (GRU) layers implementation
//! Gated Recurrent Unit（GRU）レイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use num_traits::Float;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use rand::Rng;
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
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> GRUCell<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Creates a new GRU cell
    /// 新しいGRUセルを作成
    /// 
    /// # Arguments
    /// * `input_size` - Size of the input features
    /// * `hidden_size` - Size of the hidden state
    /// * `bias` - If true, add bias parameters
    /// 
    /// # 引数
    /// * `input_size` - 入力特徴量のサイズ
    /// * `hidden_size` - 隠れ状態のサイズ
    /// * `bias` - trueの場合、バイアスパラメータを追加
    pub fn new(input_size: usize, hidden_size: usize, bias: Option<bool>) -> Self {
        let use_bias = bias.unwrap_or(true);
        
        // Initialize weights with Xavier/Glorot initialization
        // Each gate has its own set of weights, so we need 3x the parameters
        let std_ih = (6.0 / (input_size + hidden_size * 3) as f32).sqrt();
        let std_hh = (6.0 / (hidden_size + hidden_size * 3) as f32).sqrt();
        
        // Weight matrices: [3 * hidden_size, input_size/hidden_size]
        // For input-to-hidden: [3 * hidden_size, input_size]
        // For hidden-to-hidden: [3 * hidden_size, hidden_size]
        let weight_ih = Self::init_weight([3 * hidden_size, input_size], std_ih);
        let weight_hh = Self::init_weight([3 * hidden_size, hidden_size], std_hh);
        
        let bias_ih = if use_bias {
            Some(Self::init_bias(3 * hidden_size))
        } else {
            None
        };
        
        let bias_hh = if use_bias {
            Some(Self::init_bias(3 * hidden_size))
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
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Initialize weight matrix with Xavier initialization
    /// Xavier初期化で重み行列を初期化
    fn init_weight(shape: [usize; 2], std: f32) -> Variable<T> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, std as f64).unwrap();
        
        let data: Vec<T> = (0..shape[0] * shape[1])
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rng) as f32))
            .collect();
        
        Variable::new(Tensor::from_vec(data, vec![shape[0], shape[1]]), true)
    }
    
    /// Initialize bias vector
    /// バイアスベクトルを初期化
    fn init_bias(size: usize) -> Variable<T> {
        let data = vec![<T as From<f32>>::from(0.0f32); size];
        Variable::new(Tensor::from_vec(data, vec![size]), true)
    }
    
    /// Forward pass through the GRU cell
    /// GRUセルの順伝播
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, input_size]
    /// * `hidden` - Previous hidden state of shape [batch_size, hidden_size]
    /// 
    /// # Returns
    /// New hidden state of shape [batch_size, hidden_size]
    /// 
    /// # 引数
    /// * `input` - 形状[batch_size, input_size]の入力テンソル
    /// * `hidden` - 形状[batch_size, hidden_size]の前の隠れ状態
    /// 
    /// # 戻り値
    /// 形状[batch_size, hidden_size]の新しい隠れ状態
    pub fn forward(&self, input: &Variable<T>, hidden: &Variable<T>) -> Variable<T> {
        // Compute input-to-hidden and hidden-to-hidden transformations
        let ih_out = self.linear_transform(input, &self.weight_ih, &self.bias_ih);
        let hh_out = self.linear_transform(hidden, &self.weight_hh, &self.bias_hh);
        
        // Split into 3 gates: reset, update, new
        let ih_gates = self.split_gates(&ih_out);
        let hh_gates = self.split_gates(&hh_out);
        
        // Compute reset and update gates
        let reset_gate = self.sigmoid_activation(&self.add_tensors(&ih_gates[0], &hh_gates[0]));
        let update_gate = self.sigmoid_activation(&self.add_tensors(&ih_gates[1], &hh_gates[1]));
        
        // Compute new gate with reset applied to hidden state
        let reset_hidden = self.element_multiply(&reset_gate, &hh_gates[2]);
        let new_gate = self.tanh_activation(&self.add_tensors(&ih_gates[2], &reset_hidden));
        
        // Compute new hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
        let one_minus_update = self.subtract_from_one(&update_gate);
        let new_part = self.element_multiply(&one_minus_update, &new_gate);
        let old_part = self.element_multiply(&update_gate, hidden);
        
        self.add_tensors(&new_part, &old_part)
    }
    
    /// Linear transformation: x @ W^T + b
    /// 線形変換: x @ W^T + b
    fn linear_transform(
        &self,
        input: &Variable<T>,
        weight: &Variable<T>,
        bias: &Option<Variable<T>>,
    ) -> Variable<T> {
        // Matrix multiplication: input @ weight^T
        let output = self.matmul(input, weight, true);
        
        // Add bias if present
        if let Some(b) = bias {
            self.add_bias(&output, b)
        } else {
            output
        }
    }
    
    /// Matrix multiplication with optional transpose
    /// 転置オプション付き行列乗算
    fn matmul(&self, a: &Variable<T>, b: &Variable<T>, transpose_b: bool) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_shape = a_data.shape();
        let b_shape = b_data.shape();
        
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = if transpose_b {
            (b_shape[1], b_shape[0])
        } else {
            (b_shape[0], b_shape[1])
        };
        
        assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        let mut result = vec![<T as From<f32>>::from(0.0f32); m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as From<f32>>::from(0.0f32);
                for l in 0..k {
                    let a_val = a_array[[i, l]];
                    let b_val = if transpose_b {
                        b_array[[j, l]]
                    } else {
                        b_array[[l, j]]
                    };
                    sum = sum + a_val * b_val;
                }
                result[i * n + j] = sum;
            }
        }
        
        Variable::new(
            Tensor::from_vec(result, vec![m, n]),
            a.requires_grad() || b.requires_grad(),
        )
    }
    
    /// Add bias to tensor
    /// テンソルにバイアスを追加
    fn add_bias(&self, input: &Variable<T>, bias: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let bias_binding = bias.data();
        let bias_data = bias_binding.read().unwrap();
        
        let input_shape = input_data.shape();
        let bias_array = bias_data.as_array();
        let input_array = input_data.as_array();
        
        let batch_size = input_shape[0];
        let feature_size = input_shape[1];
        
        let mut result = vec![<T as From<f32>>::from(0.0f32); batch_size * feature_size];
        
        for i in 0..batch_size {
            for j in 0..feature_size {
                result[i * feature_size + j] = input_array[[i, j]] + bias_array[[j]];
            }
        }
        
        Variable::new(
            Tensor::from_vec(result, input_shape.to_vec()),
            input.requires_grad() || bias.requires_grad(),
        )
    }
    
    /// Add two tensors element-wise
    /// 2つのテンソルを要素ごとに加算
    fn add_tensors(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        assert_eq!(a_data.shape(), b_data.shape(), "Tensor shapes must match for addition");
        
        let result: Vec<T> = a_array
            .iter()
            .zip(b_array.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        
        Variable::new(
            Tensor::from_vec(result, a_data.shape().to_vec()),
            a.requires_grad() || b.requires_grad(),
        )
    }
    
    /// Element-wise multiplication
    /// 要素ごとの乗算
    fn element_multiply(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        assert_eq!(a_data.shape(), b_data.shape(), "Tensor shapes must match for multiplication");
        
        let result: Vec<T> = a_array
            .iter()
            .zip(b_array.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        
        Variable::new(
            Tensor::from_vec(result, a_data.shape().to_vec()),
            a.requires_grad() || b.requires_grad(),
        )
    }
    
    /// Subtract tensor from one: 1 - x
    /// テンソルを1から減算: 1 - x
    fn subtract_from_one(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        let result: Vec<T> = input_array
            .iter()
            .map(|&x| <T as From<f32>>::from(1.0f32) - x)
            .collect();
        
        Variable::new(
            Tensor::from_vec(result, input_data.shape().to_vec()),
            input.requires_grad(),
        )
    }
    
    /// Split concatenated gates into 3 separate tensors
    /// 連結されたゲートを3つの別々のテンソルに分割
    fn split_gates(&self, combined: &Variable<T>) -> Vec<Variable<T>> {
        let combined_binding = combined.data();
        let combined_data = combined_binding.read().unwrap();
        let combined_shape = combined_data.shape();
        let combined_array = combined_data.as_array();
        
        let batch_size = combined_shape[0];
        let total_features = combined_shape[1];
        let gate_size = total_features / 3;
        
        assert_eq!(total_features % 3, 0, "Total features must be divisible by 3");
        
        let mut gates = Vec::with_capacity(3);
        
        for gate_idx in 0..3 {
            let mut gate_data = Vec::with_capacity(batch_size * gate_size);
            
            for i in 0..batch_size {
                let start_idx = i * total_features + gate_idx * gate_size;
                for j in 0..gate_size {
                    gate_data.push(combined_array[start_idx + j]);
                }
            }
            
            gates.push(Variable::new(
                Tensor::from_vec(gate_data, vec![batch_size, gate_size]),
                combined.requires_grad(),
            ));
        }
        
        gates
    }
    
    /// Apply sigmoid activation
    /// sigmoid活性化関数を適用
    fn sigmoid_activation(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        let result: Vec<T> = input_array
            .iter()
            .map(|&x| {
                let x_f32 = x.to_f32().unwrap_or(0.0);
                let sigmoid = 1.0 / (1.0 + (-x_f32).exp());
                <T as From<f32>>::from(sigmoid)
            })
            .collect();
        
        Variable::new(
            Tensor::from_vec(result, input_data.shape().to_vec()),
            input.requires_grad(),
        )
    }
    
    /// Apply tanh activation
    /// tanh活性化関数を適用
    fn tanh_activation(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        let result: Vec<T> = input_array
            .iter()
            .map(|&x| {
                let x_f32 = x.to_f32().unwrap_or(0.0);
                <T as From<f32>>::from(x_f32.tanh())
            })
            .collect();
        
        Variable::new(
            Tensor::from_vec(result, input_data.shape().to_vec()),
            input.requires_grad(),
        )
    }
    
    /// Initialize hidden state with zeros
    /// ゼロで隠れ状態を初期化
    pub fn init_hidden(&self, batch_size: usize) -> Variable<T> {
        let data = vec![<T as From<f32>>::from(0.0f32); batch_size * self.hidden_size];
        Variable::new(Tensor::from_vec(data, vec![batch_size, self.hidden_size]), false)
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
    /// 訓練モードを設定
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
    }
    
    /// Set evaluation mode
    /// 評価モードを設定
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
    }
    
    /// Check if in training mode
    /// 訓練モードかどうかをチェック
    pub fn is_training(&self) -> bool {
        self.training.read().unwrap_or_else(|_| panic!("Failed to read training mode")).clone()
    }
}

impl<T> Module<T> for GRUCell<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For single-step forward, we need a hidden state
        // This is a simplified implementation - normally you'd pass hidden state
        let batch_size = input.data().read().unwrap().shape()[0];
        let hidden = self.init_hidden(batch_size);
        self.forward(input, &hidden)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        
        if let Some(ref bias) = self.bias_ih {
            params.push(bias.clone());
        }
        
        if let Some(ref bias) = self.bias_hh {
            params.push(bias.clone());
        }
        
        params
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Multi-layer GRU implementation
/// 多層GRUの実装
/// 
/// A multi-layer GRU that processes sequences of inputs
/// 入力シーケンスを処理する多層GRU
#[derive(Debug)]
pub struct GRU<T: Float + Send + Sync> {
    /// GRU cells for each layer
    /// 各層のGRUセル
    layers: Vec<GRUCell<T>>,
    
    /// Number of layers
    /// 層数
    num_layers: usize,
    
    /// Bidirectional flag
    /// 双方向フラグ
    bidirectional: bool,
    
    /// Dropout probability between layers
    /// 層間のドロップアウト確率
    dropout: T,
    
    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> GRU<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Creates a new multi-layer GRU
    /// 新しい多層GRUを作成
    /// 
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of GRU layers
    /// * `bias` - If true, add bias parameters
    /// * `batch_first` - If true, input shape is [batch, seq, feature]
    /// * `dropout` - Dropout probability between layers
    /// * `bidirectional` - If true, use bidirectional GRU
    /// 
    /// # 引数
    /// * `input_size` - 入力特徴量のサイズ
    /// * `hidden_size` - 隠れ状態のサイズ
    /// * `num_layers` - GRU層の数
    /// * `bias` - trueの場合、バイアスパラメータを追加
    /// * `batch_first` - trueの場合、入力形状は[batch, seq, feature]
    /// * `dropout` - 層間のドロップアウト確率
    /// * `bidirectional` - trueの場合、双方向GRUを使用
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        _batch_first: Option<bool>,
        dropout: Option<T>,
        bidirectional: Option<bool>,
    ) -> Self {
        let num_layers = num_layers.unwrap_or(1);
        let dropout = dropout.unwrap_or_else(|| <T as From<f32>>::from(0.0f32));
        let bidirectional = bidirectional.unwrap_or(false);
        
        let mut layers = Vec::new();
        
        for i in 0..num_layers {
            let layer_input_size = if i == 0 {
                input_size
            } else {
                if bidirectional { hidden_size * 2 } else { hidden_size }
            };
            
            layers.push(GRUCell::new(layer_input_size, hidden_size, bias));
            
            // Add backward layer for bidirectional GRU
            if bidirectional {
                layers.push(GRUCell::new(layer_input_size, hidden_size, bias));
            }
        }
        
        GRU {
            layers,
            num_layers,
            bidirectional,
            dropout,
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Forward pass through the GRU
    /// GRUの順伝播
    /// 
    /// # Arguments
    /// * `input` - Input sequence tensor of shape [seq_len, batch_size, input_size]
    /// * `hidden` - Initial hidden state (optional)
    /// 
    /// # Returns
    /// (output, final_hidden) where:
    /// - output: [seq_len, batch_size, hidden_size * num_directions]
    /// - final_hidden: [num_layers * num_directions, batch_size, hidden_size]
    /// 
    /// # 引数
    /// * `input` - 形状[seq_len, batch_size, input_size]の入力シーケンステンソル
    /// * `hidden` - 初期隠れ状態（オプション）
    /// 
    /// # 戻り値
    /// (output, final_hidden)：
    /// - output: [seq_len, batch_size, hidden_size * num_directions]
    /// - final_hidden: [num_layers * num_directions, batch_size, hidden_size]
    pub fn forward_with_hidden(
        &self,
        input: &Variable<T>,
        hidden: Option<&Variable<T>>,
    ) -> (Variable<T>, Variable<T>) {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let _input_size = input_shape[2];
        
        // Initialize hidden states if not provided
        let mut current_hidden = if let Some(h) = hidden {
            h.clone()
        } else {
            self.init_hidden(batch_size)
        };
        
        let mut layer_outputs = Vec::new();
        let mut final_hiddens = Vec::new();
        
        // Process each layer
        for layer_idx in 0..self.num_layers {
            let cell_idx = if self.bidirectional { layer_idx * 2 } else { layer_idx };
            let cell = &self.layers[cell_idx];
            
            let mut layer_output = Vec::new();
            let mut step_hidden = self.extract_layer_hidden(&current_hidden, layer_idx);
            
            // Forward direction
            for t in 0..seq_len {
                let step_input = self.extract_timestep(input, t);
                step_hidden = cell.forward(&step_input, &step_hidden);
                layer_output.push(step_hidden.clone());
            }
            
            final_hiddens.push(step_hidden);
            
            // Backward direction (if bidirectional)
            if self.bidirectional {
                let backward_cell = &self.layers[cell_idx + 1];
                let mut backward_output = Vec::new();
                let mut backward_hidden = self.extract_layer_hidden(&current_hidden, layer_idx);
                
                for t in (0..seq_len).rev() {
                    let step_input = self.extract_timestep(input, t);
                    backward_hidden = backward_cell.forward(&step_input, &backward_hidden);
                    backward_output.push(backward_hidden.clone());
                }
                
                backward_output.reverse();
                final_hiddens.push(backward_hidden);
                
                // Concatenate forward and backward outputs
                for t in 0..seq_len {
                    layer_output[t] = self.concatenate(&layer_output[t], &backward_output[t]);
                }
            }
            
            layer_outputs = layer_output;
            current_hidden = self.stack_hidden_states(&final_hiddens);
        }
        
        let output = self.stack_sequence_outputs(&layer_outputs);
        (output, current_hidden)
    }
    
    /// Initialize hidden state
    /// 隠れ状態を初期化
    fn init_hidden(&self, batch_size: usize) -> Variable<T> {
        let hidden_size = self.layers[0].hidden_size();
        let num_directions = if self.bidirectional { 2 } else { 1 };
        let total_hidden_size = self.num_layers * num_directions * batch_size * hidden_size;
        
        let data = vec![<T as From<f32>>::from(0.0f32); total_hidden_size];
        Variable::new(
            Tensor::from_vec(data, vec![self.num_layers * num_directions, batch_size, hidden_size]),
            false,
        )
    }
    
    /// Extract hidden state for a specific layer
    /// 特定の層の隠れ状態を抽出
    fn extract_layer_hidden(&self, hidden: &Variable<T>, layer_idx: usize) -> Variable<T> {
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        let hidden_shape = hidden_data.shape();
        let batch_size = hidden_shape[1];
        let hidden_size = hidden_shape[2];
        
        let start_idx = layer_idx * batch_size * hidden_size;
        let end_idx = start_idx + batch_size * hidden_size;
        
        let hidden_array = hidden_data.as_array();
        let mut layer_data = Vec::with_capacity(batch_size * hidden_size);
        for i in start_idx..end_idx {
            layer_data.push(hidden_array[i]);
        }
        
        Variable::new(
            Tensor::from_vec(layer_data, vec![batch_size, hidden_size]),
            hidden.requires_grad(),
        )
    }
    
    /// Extract input at a specific timestep
    /// 特定のタイムステップの入力を抽出
    fn extract_timestep(&self, input: &Variable<T>, timestep: usize) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        let batch_size = input_shape[1];
        let input_size = input_shape[2];
        
        let start_idx = timestep * batch_size * input_size;
        let end_idx = start_idx + batch_size * input_size;
        
        let input_array = input_data.as_array();
        let mut step_data = Vec::with_capacity(batch_size * input_size);
        for i in start_idx..end_idx {
            step_data.push(input_array[i]);
        }
        
        Variable::new(
            Tensor::from_vec(step_data, vec![batch_size, input_size]),
            input.requires_grad(),
        )
    }
    
    /// Concatenate two tensors along the feature dimension
    /// 特徴次元に沿って2つのテンソルを連結
    fn concatenate(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_shape = a_data.shape();
        let b_shape = b_data.shape();
        
        assert_eq!(a_shape[0], b_shape[0], "Batch sizes must match");
        
        let batch_size = a_shape[0];
        let a_features = a_shape[1];
        let b_features = b_shape[1];
        let total_features = a_features + b_features;
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        let mut result = Vec::with_capacity(batch_size * total_features);
        
        for i in 0..batch_size {
            // Add features from tensor a
            for j in 0..a_features {
                result.push(a_array[[i, j]]);
            }
            // Add features from tensor b
            for j in 0..b_features {
                result.push(b_array[[i, j]]);
            }
        }
        
        Variable::new(
            Tensor::from_vec(result, vec![batch_size, total_features]),
            a.requires_grad() || b.requires_grad(),
        )
    }
    
    /// Stack hidden states from multiple layers
    /// 複数層の隠れ状態をスタック
    fn stack_hidden_states(&self, hiddens: &[Variable<T>]) -> Variable<T> {
        if hiddens.is_empty() {
            panic!("Cannot stack empty hidden states");
        }
        
        let first_hidden = &hiddens[0];
        let first_binding = first_hidden.data();
        let hidden_data = first_binding.read().unwrap();
        let hidden_shape = hidden_data.shape();
        let batch_size = hidden_shape[0];
        let hidden_size = hidden_shape[1];
        
        let num_states = hiddens.len();
        let mut result_data = Vec::with_capacity(num_states * batch_size * hidden_size);
        
        for hidden in hiddens {
            let data_binding = hidden.data();
            let data = data_binding.read().unwrap();
            let array = data.as_array();
            for &val in array {
                result_data.push(val);
            }
        }
        
        Variable::new(
            Tensor::from_vec(result_data, vec![num_states, batch_size, hidden_size]),
            hiddens.iter().any(|h| h.requires_grad()),
        )
    }
    
    /// Stack sequence outputs
    /// シーケンス出力をスタック
    fn stack_sequence_outputs(&self, outputs: &[Variable<T>]) -> Variable<T> {
        if outputs.is_empty() {
            panic!("Cannot stack empty sequence outputs");
        }
        
        let first_output = &outputs[0];
        let output_binding = first_output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        let batch_size = output_shape[0];
        let hidden_size = output_shape[1];
        
        let seq_len = outputs.len();
        let mut result_data = Vec::with_capacity(seq_len * batch_size * hidden_size);
        
        for output in outputs {
            let data_binding = output.data();
            let data = data_binding.read().unwrap();
            let array = data.as_array();
            for &val in array {
                result_data.push(val);
            }
        }
        
        Variable::new(
            Tensor::from_vec(result_data, vec![seq_len, batch_size, hidden_size]),
            outputs.iter().any(|o| o.requires_grad()),
        )
    }
    
    /// Set training mode
    /// 訓練モードを設定
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
        for layer in &self.layers {
            layer.train();
        }
    }
    
    /// Set evaluation mode
    /// 評価モードを設定
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
        for layer in &self.layers {
            layer.eval();
        }
    }
}

impl<T> Module<T> for GRU<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let (output, _) = self.forward_with_hidden(input, None);
        output
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gru_cell_creation() {
        let cell = GRUCell::<f32>::new(10, 20, Some(true));
        assert_eq!(cell.input_size(), 10);
        assert_eq!(cell.hidden_size(), 20);
        assert!(cell.is_training());
        
        let params = cell.parameters();
        assert_eq!(params.len(), 4); // weight_ih, weight_hh, bias_ih, bias_hh
    }
    
    #[test]
    fn test_gru_cell_forward() {
        let cell = GRUCell::<f32>::new(3, 5, Some(true));
        
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            false
        );
        
        let hidden = cell.init_hidden(2);
        let output = cell.forward(&input, &hidden);
        
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        assert_eq!(output_data.shape(), &[2, 5]);
        
        // Output should be different from zero (since we have non-zero input)
        let output_array = output_data.as_array();
        assert!(output_array.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_gru_cell_no_bias() {
        let cell = GRUCell::<f32>::new(5, 10, Some(false));
        let params = cell.parameters();
        assert_eq!(params.len(), 2); // Only weight_ih and weight_hh
    }
    
    #[test]
    fn test_gru_multi_layer() {
        let gru = GRU::<f32>::new(
            10,    // input_size
            20,    // hidden_size
            Some(2), // num_layers
            Some(true), // bias
            Some(true), // batch_first
            None,   // dropout
            Some(false), // bidirectional
        );
        
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 5 * 3 * 10], vec![5, 3, 10]), // [seq_len, batch, input_size]
            false
        );
        
        let (output, hidden) = gru.forward_with_hidden(&input, None);
        
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        
        assert_eq!(output_data.shape(), &[5, 3, 20]); // [seq_len, batch, hidden_size]
        assert_eq!(hidden_data.shape(), &[2, 3, 20]); // [num_layers, batch, hidden_size]
    }
    
    #[test]
    fn test_gru_bidirectional() {
        let gru = GRU::<f32>::new(
            5,     // input_size
            10,    // hidden_size
            Some(1), // num_layers
            Some(true), // bias
            Some(true), // batch_first
            None,   // dropout
            Some(true), // bidirectional
        );
        
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 3 * 2 * 5], vec![3, 2, 5]), // [seq_len, batch, input_size]
            false
        );
        
        let (output, hidden) = gru.forward_with_hidden(&input, None);
        
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        
        assert_eq!(output_data.shape(), &[3, 2, 20]); // [seq_len, batch, hidden_size * 2]
        assert_eq!(hidden_data.shape(), &[2, 2, 10]); // [num_layers * 2, batch, hidden_size]
    }
    
    #[test]
    fn test_init_hidden() {
        let cell = GRUCell::<f32>::new(5, 10, Some(true));
        let hidden = cell.init_hidden(3);
        
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        assert_eq!(hidden_data.shape(), &[3, 10]);
        
        // All values should be zero
        let hidden_array = hidden_data.as_array();
        for &val in hidden_array {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-6);
        }
    }
}