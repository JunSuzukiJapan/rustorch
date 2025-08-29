//! Attention mechanisms implementation
//! アテンション機構の実装

use crate::autograd::Variable;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Multi-Head Attention layer
/// マルチヘッドアテンション層
///
/// Implements the multi-head attention mechanism from "Attention Is All You Need".
/// "Attention Is All You Need"のマルチヘッドアテンション機構を実装します。
#[derive(Debug)]
pub struct MultiHeadAttention<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
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

    /// Temperature scaling factor
    /// 温度スケーリング因子
    temperature: T,
}

impl<T> MultiHeadAttention<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Creates a new MultiHeadAttention layer
    /// 新しいMultiHeadAttention層を作成します
    pub fn new(d_model: usize, num_heads: usize, _dropout: Option<T>, bias: Option<bool>) -> Self {
        assert!(d_model > 0, "d_model must be greater than 0");
        assert!(num_heads > 0, "num_heads must be greater than 0");
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        let d_k = d_model / num_heads;
        let bias = bias.unwrap_or(true);

        // Create projection layers
        let w_q = if bias {
            Linear::new(d_model, d_model)
        } else {
            Linear::new_no_bias(d_model, d_model)
        };
        let w_k = if bias {
            Linear::new(d_model, d_model)
        } else {
            Linear::new_no_bias(d_model, d_model)
        };
        let w_v = if bias {
            Linear::new(d_model, d_model)
        } else {
            Linear::new_no_bias(d_model, d_model)
        };
        let w_o = if bias {
            Linear::new(d_model, d_model)
        } else {
            Linear::new_no_bias(d_model, d_model)
        };

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
            temperature,
        }
    }

    /// Get the model dimension
    /// モデル次元を取得
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get the number of attention heads
    /// アテンションヘッド数を取得
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the head dimension
    /// ヘッド次元を取得
    pub fn d_k(&self) -> usize {
        self.d_k
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
            panic!(
                "MultiHeadAttention expects 3D input (batch_size, seq_length, d_model), got {:?}",
                q_shape
            );
        }

        let batch_size = q_shape[0];
        let seq_length = q_shape[1];
        let d_model = q_shape[2];

        if d_model != self.d_model {
            panic!(
                "Input d_model {} doesn't match layer d_model {}",
                d_model, self.d_model
            );
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
        let attention_output =
            self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask);

        // Reshape back and apply output projection
        let concat_output = self.reshape_from_heads(&attention_output, batch_size, seq_length);

        self.w_o.forward(&concat_output)
    }

    /// Reshape tensor for multi-head attention
    /// マルチヘッドアテンション用にテンソルを再形成
    fn reshape_for_heads(
        &self,
        input: &Variable<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();

        // Shape: (batch_size, seq_length, d_model) -> (batch_size, seq_length, num_heads, d_k)
        // Then permute to: (batch_size, num_heads, seq_length, d_k)
        let data_vec = input_data.as_array().iter().cloned().collect::<Vec<_>>();
        let mut reshaped_data = Vec::with_capacity(data_vec.len());

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_length {
                    for d in 0..self.d_k {
                        let original_idx =
                            b * seq_length * self.d_model + s * self.d_model + h * self.d_k + d;
                        reshaped_data.push(data_vec[original_idx]);
                    }
                }
            }
        }

        let reshaped_tensor = Tensor::from_vec(
            reshaped_data,
            vec![batch_size, self.num_heads, seq_length, self.d_k],
        );

        Variable::new(reshaped_tensor, input.requires_grad())
    }

    /// Scaled Dot-Product Attention
    /// スケール付きドット積アテンション
    fn scaled_dot_product_attention(
        &self,
        query: &Variable<T>,
        key: &Variable<T>,
        value: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        // QK^T / sqrt(d_k)
        let key_transposed = key.transpose_last_two();
        let scores = query.attention_matmul(&key_transposed);
        let scaled_scores = self.apply_temperature(&scores);

        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            self.apply_mask(&scaled_scores, mask)
        } else {
            scaled_scores
        };

        // Softmax
        let attention_weights = self.softmax(&masked_scores);

        // Apply to values
        attention_weights.attention_matmul(value)
    }

    /// Apply temperature scaling
    /// 温度スケーリングを適用
    fn apply_temperature(&self, scores: &Variable<T>) -> Variable<T> {
        let scores_binding = scores.data();
        let scores_data = scores_binding.read().unwrap();
        let scaled_data: Vec<T> = scores_data
            .as_array()
            .iter()
            .map(|&x| x * self.temperature)
            .collect();

        let scaled_tensor = Tensor::from_vec(scaled_data, scores_data.shape().to_vec());
        Variable::new(scaled_tensor, scores.requires_grad())
    }

    /// Apply attention mask (set masked positions to large negative value)
    /// アテンションマスクを適用（マスクされた位置を大きな負の値に設定）
    fn apply_mask(&self, scores: &Variable<T>, mask: &Variable<T>) -> Variable<T> {
        let scores_binding = scores.data();
        let scores_data = scores_binding.read().unwrap();
        let mask_binding = mask.data();
        let mask_data = mask_binding.read().unwrap();

        let large_neg = T::from(-1e9).unwrap();
        let masked_data: Vec<T> = scores_data
            .as_array()
            .iter()
            .zip(mask_data.as_array().iter())
            .map(|(&score, &mask_val)| {
                if mask_val == T::zero() {
                    large_neg
                } else {
                    score
                }
            })
            .collect();

        let masked_tensor = Tensor::from_vec(masked_data, scores_data.shape().to_vec());
        Variable::new(masked_tensor, scores.requires_grad())
    }

    /// Softmax implementation for attention weights
    /// アテンション重みのSoftmax実装
    fn softmax(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        let data = input_data.as_array();

        // Find max for numerical stability
        let max_val = data
            .iter()
            .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });

        // Compute exp(x - max)
        let exp_data: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum for normalization
        let sum: T = exp_data.iter().fold(T::zero(), |acc, &x| acc + x);

        // Normalize
        let softmax_data: Vec<T> = exp_data.iter().map(|&x| x / sum).collect();

        let softmax_tensor = Tensor::from_vec(softmax_data, input_shape.to_vec());
        Variable::new(softmax_tensor, input.requires_grad())
    }

    /// Reshape from multi-head back to concatenated form
    /// マルチヘッドから連結形式に再形成
    fn reshape_from_heads(
        &self,
        input: &Variable<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let data_vec = input_data.as_array().iter().cloned().collect::<Vec<_>>();

        // From (batch_size, num_heads, seq_length, d_k) to (batch_size, seq_length, d_model)
        let mut output_data = Vec::with_capacity(batch_size * seq_length * self.d_model);

        for b in 0..batch_size {
            for s in 0..seq_length {
                for h in 0..self.num_heads {
                    for d in 0..self.d_k {
                        let input_idx = b * self.num_heads * seq_length * self.d_k
                            + h * seq_length * self.d_k
                            + s * self.d_k
                            + d;
                        output_data.push(data_vec[input_idx]);
                    }
                }
            }
        }

        let output_tensor =
            Tensor::from_vec(output_data, vec![batch_size, seq_length, self.d_model]);
        Variable::new(output_tensor, input.requires_grad())
    }
}

impl<T> Module<T> for MultiHeadAttention<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Forward pass for MultiHeadAttention (self-attention version)
    /// MultiHeadAttentionの順伝播（セルフアテンション版）
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For self-attention, query, key, and value are all the same input
        self.forward(input, input, input, None)
    }

    /// Get all parameters of the multi-head attention layer
    /// マルチヘッドアテンション層の全パラメータを取得
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    /// Downcast reference for the module
    /// モジュールのダウンキャスト参照
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Self-Attention layer (alias for MultiHeadAttention with self-attention usage)
/// セルフアテンション層（セルフアテンション使用のMultiHeadAttentionのエイリアス）
pub type SelfAttention<T> = MultiHeadAttention<T>;

impl<T> SelfAttention<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Forward pass for self-attention with optional mask
    /// マスク付きセルフアテンションの順伝播
    pub fn forward_self_attention(
        &self,
        input: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        self.forward(input, input, input, mask)
    }
}

/// Cross-Attention layer for encoder-decoder architectures
/// エンコーダー・デコーダーアーキテクチャ用クロスアテンション層
#[derive(Debug)]
pub struct CrossAttention<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    /// Underlying multi-head attention mechanism
    /// 基底のマルチヘッドアテンション機構
    attention: MultiHeadAttention<T>,
}

impl<T> CrossAttention<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Creates a new CrossAttention layer
    /// 新しいCrossAttention層を作成
    pub fn new(d_model: usize, num_heads: usize, dropout: Option<T>) -> Self {
        CrossAttention {
            attention: MultiHeadAttention::new(d_model, num_heads, dropout, Some(true)),
        }
    }

    /// Forward pass with separate query, key, and value inputs
    /// 個別のクエリ、キー、バリュー入力による順伝播
    pub fn forward_cross_attention(
        &self,
        query: &Variable<T>,
        key: &Variable<T>,
        value: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        self.attention.forward(query, key, value, mask)
    }

    /// Forward pass with separate query and key-value inputs (encoder-decoder style)
    /// 個別のクエリとキー・バリュー入力による順伝播（エンコーダー・デコーダースタイル）
    pub fn forward_encoder_decoder(
        &self,
        query: &Variable<T>,
        key_value: &Variable<T>,
        mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        self.attention.forward(query, key_value, key_value, mask)
    }
}

impl<T> Module<T> for CrossAttention<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For Module trait, assume self-attention behavior
        self.attention.forward(input, input, input, None)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        self.attention.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let self_attn = SelfAttention::<f32>::new(256, 4, None, None);

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
    #[ignore] // TODO: Fix 3D tensor matrix multiplication in linear layer
    fn test_attention_forward_shape() {
        let mha = MultiHeadAttention::<f32>::new(64, 4, None, None);

        // Create input: batch_size=2, seq_length=10, d_model=64
        let input_data: Vec<f32> = (0..2 * 10 * 64).map(|i| i as f32 * 0.01).collect();
        let input = Variable::new(Tensor::from_vec(input_data, vec![2, 10, 64]), false);

        let output = mha.forward(&input, &input, &input, None);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // Output shape should be same as input
        assert_eq!(output_data.shape(), &[2, 10, 64]);
    }
}
