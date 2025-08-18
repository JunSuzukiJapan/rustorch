//! Recurrent Neural Network (RNN) layers implementation
//! リカレントニューラルネットワーク（RNN）レイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use num_traits::Float;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use rand_distr::{Normal, Distribution};

/// Basic RNN cell implementation
/// 基本的なRNNセルの実装
/// 
/// A simple recurrent neural network cell that computes:
/// h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
/// 
/// シンプルなリカレントニューラルネットワークセル：
/// h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
#[derive(Debug)]
pub struct RNNCell<T: Float + Send + Sync> {
    /// Input-to-hidden weight matrix
    /// 入力から隠れ状態への重み行列
    weight_ih: Variable<T>,
    
    /// Hidden-to-hidden weight matrix  
    /// 隠れ状態から隠れ状態への重み行列
    weight_hh: Variable<T>,
    
    /// Input-to-hidden bias
    /// 入力から隠れ状態へのバイアス
    bias_ih: Option<Variable<T>>,
    
    /// Hidden-to-hidden bias
    /// 隠れ状態から隠れ状態へのバイアス
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

impl<T> RNNCell<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Creates a new RNN cell
    /// 新しいRNNセルを作成
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
        let std_ih = (6.0 / (input_size + hidden_size) as f32).sqrt();
        let std_hh = (6.0 / (hidden_size + hidden_size) as f32).sqrt();
        
        let weight_ih = Self::init_weight([hidden_size, input_size], std_ih);
        let weight_hh = Self::init_weight([hidden_size, hidden_size], std_hh);
        
        let bias_ih = if use_bias {
            Some(Self::init_bias(hidden_size))
        } else {
            None
        };
        
        let bias_hh = if use_bias {
            Some(Self::init_bias(hidden_size))
        } else {
            None
        };
        
        RNNCell {
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
    
    /// Forward pass through the RNN cell
    /// RNNセルの順伝播
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
        // Compute input-to-hidden transformation: W_ih @ x_t + b_ih
        let ih_out = self.linear_transform(input, &self.weight_ih, &self.bias_ih);
        
        // Compute hidden-to-hidden transformation: W_hh @ h_{t-1} + b_hh
        let hh_out = self.linear_transform(hidden, &self.weight_hh, &self.bias_hh);
        
        // Add transformations and apply tanh activation
        let combined = self.add_tensors(&ih_out, &hh_out);
        self.tanh_activation(&combined)
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

impl<T> Module<T> for RNNCell<T>
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

/// Multi-layer RNN implementation
/// 多層RNNの実装
/// 
/// A multi-layer RNN that processes sequences of inputs
/// 入力シーケンスを処理する多層RNN
#[derive(Debug)]
pub struct RNN<T: Float + Send + Sync> {
    /// RNN cells for each layer
    /// 各層のRNNセル
    layers: Vec<RNNCell<T>>,
    
    /// Number of layers
    /// 層数
    num_layers: usize,
    
    /// Bidirectional flag
    /// 双方向フラグ
    bidirectional: bool,
    
    /// Dropout probability between layers
    /// 層間のドロップアウト確率
    #[allow(dead_code)]
    dropout: T,
    
    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> RNN<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Creates a new multi-layer RNN
    /// 新しい多層RNNを作成
    /// 
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of RNN layers
    /// * `bias` - If true, add bias parameters
    /// * `batch_first` - If true, input shape is [batch, seq, feature]
    /// * `dropout` - Dropout probability between layers
    /// * `bidirectional` - If true, use bidirectional RNN
    /// 
    /// # 引数
    /// * `input_size` - 入力特徴量のサイズ
    /// * `hidden_size` - 隠れ状態のサイズ
    /// * `num_layers` - RNN層の数
    /// * `bias` - trueの場合、バイアスパラメータを追加
    /// * `batch_first` - trueの場合、入力形状は[batch, seq, feature]
    /// * `dropout` - 層間のドロップアウト確率
    /// * `bidirectional` - trueの場合、双方向RNNを使用
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
            
            layers.push(RNNCell::new(layer_input_size, hidden_size, bias));
            
            // Add backward layer for bidirectional RNN
            if bidirectional {
                layers.push(RNNCell::new(layer_input_size, hidden_size, bias));
            }
        }
        
        RNN {
            layers,
            num_layers,
            bidirectional,
            dropout,
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Forward pass through the RNN
    /// RNNの順伝播
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
        let mut current_layer_input = input.clone();
        
        // Process each layer
        for layer_idx in 0..self.num_layers {
            let cell_idx = if self.bidirectional { layer_idx * 2 } else { layer_idx };
            let cell = &self.layers[cell_idx];
            
            let mut layer_output = Vec::new();
            let mut step_hidden = self.extract_layer_hidden(&current_hidden, layer_idx);
            
            // Forward direction
            for t in 0..seq_len {
                let step_input = self.extract_timestep(&current_layer_input, t);
                step_hidden = cell.forward(&step_input, &step_hidden);
                layer_output.push(step_hidden.clone());
            }
            
            let mut layer_final_hiddens = vec![step_hidden];
            
            // Backward direction (if bidirectional)
            if self.bidirectional {
                let backward_cell = &self.layers[cell_idx + 1];
                let mut backward_output = Vec::new();
                let mut backward_hidden = self.extract_layer_hidden(&current_hidden, layer_idx);
                
                for t in (0..seq_len).rev() {
                    let step_input = self.extract_timestep(&current_layer_input, t);
                    backward_hidden = backward_cell.forward(&step_input, &backward_hidden);
                    backward_output.push(backward_hidden.clone());
                }
                
                backward_output.reverse();
                layer_final_hiddens.push(backward_hidden);
                
                // Concatenate forward and backward outputs
                for t in 0..seq_len {
                    layer_output[t] = self.concatenate(&layer_output[t], &backward_output[t]);
                }
            }
            
            layer_outputs = layer_output;
            
            // Update input for next layer and hidden state for next layer
            if layer_idx < self.num_layers - 1 {
                current_layer_input = self.stack_sequence_outputs(&layer_outputs);
            }
            
            // Update current_hidden for this layer only
            current_hidden = self.update_layer_hidden(&current_hidden, layer_idx, &layer_final_hiddens);
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
    
    /// Update hidden state for a specific layer
    /// 特定の層の隠れ状態を更新
    fn update_layer_hidden(&self, current_hidden: &Variable<T>, layer_idx: usize, new_layer_hiddens: &[Variable<T>]) -> Variable<T> {
        let current_binding = current_hidden.data();
        let current_data = current_binding.read().unwrap();
        let current_shape = current_data.shape();
        let current_array = current_data.as_array();
        
        let num_layers_dirs = current_shape[0];
        let batch_size = current_shape[1];
        let hidden_size = current_shape[2];
        
        let mut new_data = vec![<T as From<f32>>::from(0.0f32); num_layers_dirs * batch_size * hidden_size];
        
        // Copy all existing data
        for layer_dir in 0..num_layers_dirs {
            for i in 0..batch_size {
                for j in 0..hidden_size {
                    new_data[layer_dir * batch_size * hidden_size + i * hidden_size + j] = current_array[[layer_dir, i, j]];
                }
            }
        }
        
        // Update the specific layer
        let num_directions = if self.bidirectional { 2 } else { 1 };
        for (dir_idx, layer_hidden) in new_layer_hiddens.iter().enumerate() {
            let layer_dir_idx = layer_idx * num_directions + dir_idx;
            let layer_binding = layer_hidden.data();
            let layer_data = layer_binding.read().unwrap();
            let layer_array = layer_data.as_array();
            
            for i in 0..batch_size {
                for j in 0..hidden_size {
                    new_data[layer_dir_idx * batch_size * hidden_size + i * hidden_size + j] = layer_array[[i, j]];
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(new_data, current_shape.to_vec()),
            current_hidden.requires_grad(),
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
        
        let hidden_array = hidden_data.as_array();
        let mut layer_data = Vec::with_capacity(batch_size * hidden_size);
        
        for i in 0..batch_size {
            for j in 0..hidden_size {
                layer_data.push(hidden_array[[layer_idx, i, j]]);
            }
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
        
        let input_array = input_data.as_array();
        let mut step_data = Vec::with_capacity(batch_size * input_size);
        
        for i in 0..batch_size {
            for j in 0..input_size {
                step_data.push(input_array[[timestep, i, j]]);
            }
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
    #[allow(dead_code)]
    fn stack_hidden_states(&self, hiddens: &[Variable<T>]) -> Variable<T> {
        if hiddens.is_empty() {
            panic!("Cannot stack empty hidden states");
        }
        
        let first_hidden = &hiddens[0];
        let first_binding = first_hidden.data();
        let first_data = first_binding.read().unwrap();
        let hidden_shape = first_data.shape();
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
        let first_binding = first_output.data();
        let first_data = first_binding.read().unwrap();
        let output_shape = first_data.shape();
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

impl<T> Module<T> for RNN<T>
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
    fn test_rnn_cell_creation() {
        let cell = RNNCell::<f32>::new(10, 20, Some(true));
        assert_eq!(cell.input_size(), 10);
        assert_eq!(cell.hidden_size(), 20);
        assert!(cell.is_training());
        
        let params = cell.parameters();
        assert_eq!(params.len(), 4); // weight_ih, weight_hh, bias_ih, bias_hh
    }
    
    #[test]
    fn test_rnn_cell_forward() {
        let cell = RNNCell::<f32>::new(3, 5, Some(true));
        
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
    fn test_rnn_cell_no_bias() {
        let cell = RNNCell::<f32>::new(5, 10, Some(false));
        let params = cell.parameters();
        assert_eq!(params.len(), 2); // Only weight_ih and weight_hh
    }
    
    #[test]
    fn test_rnn_multi_layer() {
        let rnn = RNN::<f32>::new(
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
        
        let (output, hidden) = rnn.forward_with_hidden(&input, None);
        
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        
        assert_eq!(output_data.shape(), &[5, 3, 20]); // [seq_len, batch, hidden_size]
        assert_eq!(hidden_data.shape(), &[2, 3, 20]); // [num_layers, batch, hidden_size]
    }
    
    #[test]
    fn test_rnn_bidirectional() {
        let rnn = RNN::<f32>::new(
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
        
        let (output, hidden) = rnn.forward_with_hidden(&input, None);
        
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let hidden_binding = hidden.data();
        let hidden_data = hidden_binding.read().unwrap();
        
        assert_eq!(output_data.shape(), &[3, 2, 20]); // [seq_len, batch, hidden_size * 2]
        assert_eq!(hidden_data.shape(), &[2, 2, 10]); // [num_layers * 2, batch, hidden_size]
    }
    
    #[test]
    fn test_init_hidden() {
        let cell = RNNCell::<f32>::new(5, 10, Some(true));
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