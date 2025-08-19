//! Attention mechanisms implementation
//! アテンション機構の実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::{Module, Linear};
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero, One};
use ndarray::ScalarOperand;
use std::iter::Sum;

/// Multi-Head Attention layer
/// マルチヘッドアテンション層
/// 
/// Implements the multi-head attention mechanism from "Attention Is All You Need".
/// "Attention Is All You Need"のマルチヘッドアテンション機構を実装します。
#[derive(Debug)]
pub struct MultiHeadAttention<T: Float + Send + Sync> {
    /// Number of attention heads
    /// アテンションヘッド数
    num_heads: usize,
    
    /// Model dimension (d_model)
    /// モデル次元（d_model）
    d_model: usize,
    
    /// Head dimension (d_k = d_v = d_model / num_heads)
    /// ヘッド次元（d_k = d_v = d_model / num_heads）
    d_k: usize,
    
    /// Query projection layer
    /// クエリ射影層
    w_q: Linear<T>,
    
    /// Key projection layer
    /// キー射影層
    w_k: Linear<T>,
    
    /// Value projection layer
    /// バリュー射影層
    w_v: Linear<T>,
    
    /// Output projection layer
    /// 出力射影層
    w_o: Linear<T>,
    
    /// Dropout probability for attention weights
    /// アテンション重みのドロップアウト確率
    dropout_p: T,
    
    /// Temperature scaling factor
    /// 温度スケーリング因子
    temperature: T,
}

impl<T> MultiHeadAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new MultiHeadAttention layer
    /// 新しいMultiHeadAttention層を作成します
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout: Option<T>,
        bias: Option<bool>,
    ) -> Self {
        assert!(d_model > 0, "d_model must be greater than 0");
        assert!(num_heads > 0, "num_heads must be greater than 0");
        assert!(d_model % num_heads == 0, 
                "d_model ({}) must be divisible by num_heads ({})", d_model, num_heads);
        
        let d_k = d_model / num_heads;
        let dropout_p = dropout.unwrap_or_else(|| T::from(0.1).unwrap());
        let bias = bias.unwrap_or(true);
        
        // Create projection layers
        let w_q = if bias { Linear::new(d_model, d_model) } else { Linear::new_no_bias(d_model, d_model) };
        let w_k = if bias { Linear::new(d_model, d_model) } else { Linear::new_no_bias(d_model, d_model) };
        let w_v = if bias { Linear::new(d_model, d_model) } else { Linear::new_no_bias(d_model, d_model) };
        let w_o = if bias { Linear::new(d_model, d_model) } else { Linear::new_no_bias(d_model, d_model) };
        
        // Temperature scaling (1/sqrt(d_k))
        let temperature = T::from(1.0 / (d_k as f32).sqrt()).unwrap();
        
        MultiHeadAttention {
            num_heads,
            d_model,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            dropout_p,
            temperature,
        }
    }
    
    /// Forward pass of MultiHeadAttention
    /// MultiHeadAttentionの順伝播
    pub fn forward(
        &self,
        query: &Variable<T>,
        key: &Variable<T>,
        value: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        let q_binding = query.data();
        let q_data = q_binding.read().unwrap();
        let q_shape = q_data.shape();
        
        // Input shape: (batch_size, seq_length, d_model)
        if q_shape.len() != 3 {
            panic!("MultiHeadAttention expects 3D input (batch_size, seq_length, d_model), got {:?}", q_shape);
        }
        
        let batch_size = q_shape[0];
        let seq_length = q_shape[1];
        let d_model = q_shape[2];
        
        if d_model != self.d_model {
            panic!("Input d_model {} doesn't match layer d_model {}", d_model, self.d_model);
        }
        
        // Project to Q, K, V
        let q = self.w_q.forward(query);
        let k = self.w_k.forward(key);
        let v = self.w_v.forward(value);
        
        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_length);
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_length);
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_length);
        
        // Compute scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask);
        
        // Reshape back and apply output projection
        let concat_output = self.reshape_from_heads(&attention_output, batch_size, seq_length);
        
        self.w_o.forward(&concat_output)
    }
    
    /// Reshape tensor for multi-head attention
    /// マルチヘッドアテンション用にテンソルを再形成
    fn reshape_for_heads(&self, input: &Variable<T>, batch_size: usize, seq_length: usize) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        // Reshape from (batch_size, seq_length, d_model) to (batch_size, num_heads, seq_length, d_k)
        let mut output_data = Vec::with_capacity(batch_size * self.num_heads * seq_length * self.d_k);
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_length {
                    for d in 0..self.d_k {
                        let input_d = h * self.d_k + d;
                        output_data.push(input_array[[b, s, input_d]]);
                    }
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, vec![batch_size, self.num_heads, seq_length, self.d_k]),
            input.requires_grad()
        )
    }
    
    /// Reshape tensor back from multi-head format
    /// マルチヘッド形式からテンソルを元に戻す
    fn reshape_from_heads(&self, input: &Variable<T>, batch_size: usize, seq_length: usize) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        // Reshape from (batch_size, num_heads, seq_length, d_k) to (batch_size, seq_length, d_model)
        let mut output_data = Vec::with_capacity(batch_size * seq_length * self.d_model);
        
        for b in 0..batch_size {
            for s in 0..seq_length {
                for h in 0..self.num_heads {
                    for d in 0..self.d_k {
                        output_data.push(input_array[[b, h, s, d]]);
                    }
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, vec![batch_size, seq_length, self.d_model]),
            input.requires_grad()
        )
    }
    
    /// Scaled dot-product attention
    /// スケール付きドット積アテンション
    fn scaled_dot_product_attention(
        &self,
        q: &Variable<T>,
        k: &Variable<T>,
        v: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        // Q * K^T / sqrt(d_k)
        let scores = self.compute_attention_scores(q, k);
        
        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            self.apply_mask(&scores, mask)
        } else {
            scores
        };
        
        // Softmax
        let attention_weights = self.softmax(&masked_scores);
        
        // Apply dropout (simplified - would need proper dropout implementation)
        let dropped_weights = attention_weights; // TODO: Apply dropout
        
        // Attention * V
        self.apply_attention(&dropped_weights, v)
    }
    
    /// Compute attention scores (Q * K^T / sqrt(d_k))
    /// アテンションスコアを計算（Q * K^T / sqrt(d_k)）
    fn compute_attention_scores(&self, q: &Variable<T>, k: &Variable<T>) -> Variable<T> {
        let q_binding = q.data();
        let q_data = q_binding.read().unwrap();
        let k_binding = k.data();
        let k_data = k_binding.read().unwrap();
        
        let q_array = q_data.as_array();
        let k_array = k_data.as_array();
        let q_shape = q_data.shape();
        
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_length_q = q_shape[2];
        let d_k = q_shape[3];
        let seq_length_k = k_data.shape()[2];
        
        // Compute Q * K^T
        let mut scores_data = Vec::with_capacity(batch_size * num_heads * seq_length_q * seq_length_k);
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_length_q {
                    for j in 0..seq_length_k {
                        let mut dot_product = T::zero();
                        
                        for d in 0..d_k {
                            let q_val = q_array[[b, h, i, d]];
                            let k_val = k_array[[b, h, j, d]];
                            dot_product = dot_product + q_val * k_val;
                        }
                        
                        // Scale by temperature (1/sqrt(d_k))
                        scores_data.push(dot_product * self.temperature);
                    }
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(scores_data, vec![batch_size, num_heads, seq_length_q, seq_length_k]),
            q.requires_grad() || k.requires_grad()
        )
    }
    
    /// Apply attention mask
    /// アテンションマスクを適用
    fn apply_mask(&self, scores: &Variable<T>, mask: &Variable<T>) -> Variable<T> {
        let scores_binding = scores.data();
        let scores_data = scores_binding.read().unwrap();
        let mask_binding = mask.data();
        let mask_data = mask_binding.read().unwrap();
        
        let scores_array = scores_data.as_array();
        let mask_array = mask_data.as_array();
        
        let mut masked_data = Vec::with_capacity(scores_array.len());
        let neg_inf = T::from_f32(-1e9).unwrap(); // Large negative value
        
        if let (Some(scores_slice), Some(mask_slice)) = (scores_array.as_slice(), mask_array.as_slice()) {
            for (_i, (&score, &mask_val)) in scores_slice.iter().zip(mask_slice.iter()).enumerate() {
                if mask_val == T::zero() {
                    masked_data.push(neg_inf);
                } else {
                    masked_data.push(score);
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            for _i in 0..scores_array.len() {
                masked_data.push(neg_inf); // Simplified
            }
        }
        
        Variable::new(
            Tensor::from_vec(masked_data, scores_data.shape().to_vec()),
            scores.requires_grad()
        )
    }
    
    /// Apply softmax to attention scores
    /// アテンションスコアにソフトマックスを適用
    fn softmax(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        let input_shape = input_data.shape();
        
        // Apply softmax along the last dimension (seq_length_k)
        let batch_size = input_shape[0];
        let num_heads = input_shape[1];
        let seq_length_q = input_shape[2];
        let seq_length_k = input_shape[3];
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_length_q {
                    // Find max for numerical stability
                    let mut max_val = T::neg_infinity();
                    for j in 0..seq_length_k {
                        let val = input_array[[b, h, i, j]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    
                    // Compute exp(x - max) and sum
                    let mut exp_values = Vec::with_capacity(seq_length_k);
                    let mut sum_exp = T::zero();
                    
                    for j in 0..seq_length_k {
                        let val = input_array[[b, h, i, j]];
                        let exp_val = (val - max_val).exp();
                        exp_values.push(exp_val);
                        sum_exp = sum_exp + exp_val;
                    }
                    
                    // Normalize
                    for exp_val in exp_values {
                        output_data.push(exp_val / sum_exp);
                    }
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, input_shape.to_vec()),
            input.requires_grad()
        )
    }
    
    /// Apply attention weights to values
    /// バリューにアテンション重みを適用
    fn apply_attention(&self, attention_weights: &Variable<T>, values: &Variable<T>) -> Variable<T> {
        let weights_binding = attention_weights.data();
        let weights_data = weights_binding.read().unwrap();
        let values_binding = values.data();
        let values_data = values_binding.read().unwrap();
        
        let weights_array = weights_data.as_array();
        let values_array = values_data.as_array();
        let weights_shape = weights_data.shape();
        let values_shape = values_data.shape();
        
        let batch_size = weights_shape[0];
        let num_heads = weights_shape[1];
        let seq_length_q = weights_shape[2];
        let seq_length_k = weights_shape[3];
        let d_k = values_shape[3];
        
        let mut output_data = Vec::with_capacity(batch_size * num_heads * seq_length_q * d_k);
        
        // Compute attention_weights * values
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_length_q {
                    for d in 0..d_k {
                        let mut weighted_sum = T::zero();
                        
                        for j in 0..seq_length_k {
                            let weight = weights_array[[b, h, i, j]];
                            let value = values_array[[b, h, j, d]];
                            weighted_sum = weighted_sum + weight * value;
                        }
                        
                        output_data.push(weighted_sum);
                    }
                }
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, vec![batch_size, num_heads, seq_length_q, d_k]),
            attention_weights.requires_grad() || values.requires_grad()
        )
    }
    
    /// Returns the number of attention heads
    /// アテンションヘッド数を返します
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    
    /// Returns the model dimension
    /// モデル次元を返します
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Returns the head dimension
    /// ヘッド次元を返します
    pub fn d_k(&self) -> usize {
        self.d_k
    }
    
    /// Returns the dropout probability
    /// ドロップアウト確率を返します
    pub fn dropout_p(&self) -> T {
        self.dropout_p
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

impl<T> Module<T> for MultiHeadAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For self-attention, use input as query, key, and value
        self.forward(input, input, input, None)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Self-Attention layer (simplified multi-head attention)
/// セルフアテンション層（簡略化されたマルチヘッドアテンション）
#[derive(Debug)]
pub struct SelfAttention<T: Float + Send + Sync> {
    /// Multi-head attention implementation
    /// マルチヘッドアテンション実装
    mha: MultiHeadAttention<T>,
}

impl<T> SelfAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new SelfAttention layer
    /// 新しいSelfAttention層を作成します
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout: Option<T>,
    ) -> Self {
        let mha = MultiHeadAttention::new(d_model, num_heads, dropout, Some(true));
        
        SelfAttention { mha }
    }
    
    /// Forward pass of SelfAttention
    /// SelfAttentionの順伝播
    pub fn forward(&self, input: &Variable<T>, mask: Option<&Variable<T>>) -> Variable<T> {
        self.mha.forward(input, input, input, mask)
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        self.mha.parameters()
    }
}

impl<T> Module<T> for SelfAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input, None)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Cross-Attention layer
/// クロスアテンション層
/// 
/// Attention mechanism where queries come from one sequence and keys/values from another.
/// クエリが一つのシーケンスから、キー/バリューが別のシーケンスから来るアテンション機構。
#[derive(Debug)]
pub struct CrossAttention<T: Float + Send + Sync> {
    /// Multi-head attention implementation
    /// マルチヘッドアテンション実装
    mha: MultiHeadAttention<T>,
}

impl<T> CrossAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new CrossAttention layer
    /// 新しいCrossAttention層を作成します
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout: Option<T>,
    ) -> Self {
        let mha = MultiHeadAttention::new(d_model, num_heads, dropout, Some(true));
        
        CrossAttention { mha }
    }
    
    /// Forward pass of CrossAttention
    /// CrossAttentionの順伝播
    pub fn forward(
        &self,
        query: &Variable<T>,
        key: &Variable<T>,
        value: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        self.mha.forward(query, key, value, mask)
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        self.mha.parameters()
    }
}

impl<T> Module<T> for CrossAttention<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For Module trait, use input as query, key, and value
        self.forward(input, input, input, None)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::<f32>::new(512, 8, None, None);
        
        assert_eq!(mha.num_heads(), 8);
        assert_eq!(mha.d_model(), 512);
        assert_eq!(mha.d_k(), 64); // 512 / 8
        
        let params = mha.parameters();
        assert_eq!(params.len(), 8); // 4 layers * 2 params each (weight + bias)
    }
    
    #[test]
    fn test_self_attention_creation() {
        let self_attn = SelfAttention::<f32>::new(256, 4, None);
        
        let params = self_attn.parameters();
        assert_eq!(params.len(), 8); // Same as MultiHeadAttention
    }
    
    #[test]
    fn test_cross_attention_creation() {
        let cross_attn = CrossAttention::<f32>::new(128, 2, None);
        
        let params = cross_attn.parameters();
        assert_eq!(params.len(), 8); // Same as MultiHeadAttention
    }
    
    #[test]
    fn test_attention_forward_shape() {
        let mha = MultiHeadAttention::<f32>::new(64, 4, None, None);
        
        // Create input: batch_size=2, seq_length=10, d_model=64
        let input_data: Vec<f32> = (0..2*10*64).map(|i| i as f32 * 0.01).collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![2, 10, 64]),
            false
        );
        
        let output = mha.forward(&input, &input, &input, None);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        // Output shape should be same as input
        assert_eq!(output_data.shape(), &[2, 10, 64]);
    }
}
