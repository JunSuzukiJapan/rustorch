//! Multi-layer GRU implementation
//! 多層GRU実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::{Module, gru_cell::GRUCell};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Multi-layer GRU implementation
/// 多層GRU実装
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
    /// * `bias` - Whether to use bias parameters
    /// * `batch_first` - Whether input is batch-first
    /// * `dropout` - Dropout probability between layers
    /// * `bidirectional` - Whether to use bidirectional GRU
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        _batch_first: bool,
        _dropout: T,
        bidirectional: bool,
    ) -> Self {
        let mut layers = Vec::new();
        
        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 {
                input_size
            } else {
                if bidirectional { hidden_size * 2 } else { hidden_size }
            };
            
            let cell = GRUCell::new(layer_input_size, hidden_size, bias);
            layers.push(cell);
        }
        
        GRU {
            layers,
            num_layers,
            bidirectional,
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Forward pass through the GRU
    /// GRUの順伝播
    pub fn forward(
        &self,
        input: &Variable<T>,
        hidden: Option<&Variable<T>>,
    ) -> (Variable<T>, Variable<T>) {
        let batch_size = input.data().shape()[0];
        let seq_len = input.data().shape()[1];
        
        // Initialize hidden states if not provided
        let mut h_states = match hidden {
            Some(h) => self.split_hidden_states(h),
            None => {
                (0..self.num_layers)
                    .map(|_| Variable::new(
                        Tensor::zeros(&[batch_size, self.layers[0].hidden_size()]),
                        false,
                    ))
                    .collect()
            }
        };
        
        let mut layer_input = input.clone();
        let mut outputs = Vec::new();
        
        // Process each time step
        for t in 0..seq_len {
            let time_input = self.get_timestep_input(&layer_input, t);
            
            // Process through each layer
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_hidden = if t == 0 {
                    Some(&h_states[layer_idx])
                } else {
                    Some(&h_states[layer_idx])
                };
                
                let input_for_layer = if layer_idx == 0 {
                    time_input.clone()
                } else {
                    h_states[layer_idx - 1].clone()
                };
                
                let new_h = layer.forward(&input_for_layer, layer_hidden);
                h_states[layer_idx] = new_h;
            }
            
            outputs.push(h_states[self.num_layers - 1].clone());
        }
        
        // Stack outputs along sequence dimension
        let output = self.stack_outputs(&outputs);
        let final_h = self.stack_hidden_states(&h_states);
        
        (output, final_h)
    }
    
    /// Get input for a specific timestep
    /// 特定のタイムステップの入力を取得
    fn get_timestep_input(&self, input: &Variable<T>, timestep: usize) -> Variable<T> {
        // Simplified implementation - would need proper tensor slicing
        let input_data = input.data();
        let batch_size = input_data.shape()[0];
        let feature_size = input_data.shape()[2];
        
        // Extract data for this timestep
        let timestep_data: Vec<T> = (0..batch_size * feature_size)
            .map(|i| {
                let batch_idx = i / feature_size;
                let feat_idx = i % feature_size;
                input_data.as_slice().unwrap()[batch_idx * input_data.shape()[1] * feature_size + timestep * feature_size + feat_idx]
            })
            .collect();
            
        Variable::new(
            Tensor::from_vec(timestep_data, vec![batch_size, feature_size]),
            input.requires_grad(),
        )
    }
    
    /// Stack outputs along sequence dimension
    /// シーケンス次元に沿って出力をスタック
    fn stack_outputs(&self, outputs: &[Variable<T>]) -> Variable<T> {
        let batch_size = outputs[0].data().shape()[0];
        let hidden_size = outputs[0].data().shape()[1];
        let seq_len = outputs.len();
        
        let mut stacked_data = Vec::new();
        
        for batch_idx in 0..batch_size {
            for t in 0..seq_len {
                let output_slice = outputs[t].data().as_slice().unwrap();
                let start_idx = batch_idx * hidden_size;
                let end_idx = start_idx + hidden_size;
                stacked_data.extend_from_slice(&output_slice[start_idx..end_idx]);
            }
        }
        
        Variable::new(
            Tensor::from_vec(stacked_data, vec![batch_size, seq_len, hidden_size]),
            outputs[0].requires_grad(),
        )
    }
    
    /// Split hidden states by layer
    /// レイヤーごとに隠れ状態を分割
    fn split_hidden_states(&self, hidden: &Variable<T>) -> Vec<Variable<T>> {
        let data = hidden.data();
        let hidden_size = data.shape()[1];
        let batch_size = data.shape()[0];
        
        (0..self.num_layers)
            .map(|layer_idx| {
                let start_idx = layer_idx * batch_size * hidden_size;
                let end_idx = start_idx + batch_size * hidden_size;
                let layer_data = data.as_slice().unwrap()[start_idx..end_idx].to_vec();
                Variable::new(
                    Tensor::from_vec(layer_data, vec![batch_size, hidden_size]),
                    hidden.requires_grad(),
                )
            })
            .collect()
    }
    
    /// Stack hidden states by layer
    /// レイヤーごとに隠れ状態をスタック
    fn stack_hidden_states(&self, states: &[Variable<T>]) -> Variable<T> {
        let batch_size = states[0].data().shape()[0];
        let hidden_size = states[0].data().shape()[1];
        
        let mut stacked_data = Vec::new();
        
        for state in states {
            stacked_data.extend_from_slice(state.data().as_slice().unwrap());
        }
        
        Variable::new(
            Tensor::from_vec(stacked_data, vec![self.num_layers, batch_size, hidden_size]),
            states[0].requires_grad(),
        )
    }
    
    /// Get number of layers
    /// レイヤー数を取得
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    
    /// Check if bidirectional
    /// 双方向かどうかをチェック
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
    
    /// Get hidden size
    /// 隠れ状態サイズを取得
    pub fn hidden_size(&self) -> usize {
        self.layers[0].hidden_size()
    }
    
    /// Set training mode
    /// 学習モードを設定
    pub fn set_training(&self, training: bool) {
        *self.training.write().unwrap() = training;
    }
}

impl<T> Module<T> for GRU<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let (output, _hidden) = self.forward(input, None);
        output
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
    
    fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }
    
    fn train(&mut self) {
        self.set_training(true);
        for layer in &mut self.layers {
            layer.train();
        }
    }
    
    fn eval(&mut self) {
        self.set_training(false);
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}