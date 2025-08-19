//! Transformer architecture implementation
//! Transformerアーキテクチャの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::{Module, Linear, LayerNorm, MultiHeadAttention, Dropout};
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero, One};
use ndarray::ScalarOperand;
use std::iter::Sum;

/// Transformer Encoder Layer
/// Transformerエンコーダー層
/// 
/// A single layer of the Transformer encoder consisting of multi-head attention and feed-forward network.
/// マルチヘッドアテンションとフィードフォワードネットワークからなるTransformerエンコーダーの単一層。
#[derive(Debug)]
pub struct TransformerEncoderLayer<T: Float + Send + Sync> {
    /// Multi-head self-attention
    /// マルチヘッドセルフアテンション
    self_attention: MultiHeadAttention<T>,
    
    /// Feed-forward network (first linear layer)
    /// フィードフォワードネットワーク（第一線形層）
    ff_linear1: Linear<T>,
    
    /// Feed-forward network (second linear layer)
    /// フィードフォワードネットワーク（第二線形層）
    ff_linear2: Linear<T>,
    
    /// Layer normalization for attention
    /// アテンション用レイヤー正規化
    norm1: LayerNorm<T>,
    
    /// Layer normalization for feed-forward
    /// フィードフォワード用レイヤー正規化
    norm2: LayerNorm<T>,
    
    /// Dropout for attention
    /// アテンション用ドロップアウト
    dropout1: Dropout<T>,
    
    /// Dropout for feed-forward
    /// フィードフォワード用ドロップアウト
    dropout2: Dropout<T>,
    
    /// Model dimension
    /// モデル次元
    d_model: usize,
    
    /// Feed-forward dimension
    /// フィードフォワード次元
    d_ff: usize,
}

impl<T> TransformerEncoderLayer<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new TransformerEncoderLayer
    /// 新しいTransformerEncoderLayerを作成します
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: Option<T>,
    ) -> Self {
        assert!(d_model > 0, "d_model must be greater than 0");
        assert!(num_heads > 0, "num_heads must be greater than 0");
        assert!(d_ff > 0, "d_ff must be greater than 0");
        assert!(d_model % num_heads == 0, 
                "d_model ({}) must be divisible by num_heads ({})", d_model, num_heads);
        
        let dropout_p = dropout.unwrap_or_else(|| T::from(0.1).unwrap());
        
        // Create components
        let self_attention = MultiHeadAttention::new(d_model, num_heads, Some(dropout_p), Some(true));
        let ff_linear1 = Linear::new(d_model, d_ff);
        let ff_linear2 = Linear::new(d_ff, d_model);
        let norm1 = LayerNorm::new(vec![d_model], None, None);
        let norm2 = LayerNorm::new(vec![d_model], None, None);
        let dropout1 = Dropout::new(dropout_p, false);
        let dropout2 = Dropout::new(dropout_p, false);
        
        TransformerEncoderLayer {
            self_attention,
            ff_linear1,
            ff_linear2,
            norm1,
            norm2,
            dropout1,
            dropout2,
            d_model,
            d_ff,
        }
    }
    
    /// Forward pass of TransformerEncoderLayer
    /// TransformerEncoderLayerの順伝播
    pub fn forward(&self, input: &Variable<T>, mask: Option<&Variable<T>>) -> Variable<T> {
        // Multi-head self-attention with residual connection and layer norm
        let attn_output = self.self_attention.forward(input, input, input, mask);
        let attn_output = self.dropout1.forward(&attn_output);
        let attn_residual = self.add_tensors(input, &attn_output);
        let norm1_output = self.norm1.forward(&attn_residual);
        
        // Feed-forward network with residual connection and layer norm
        let ff_output = self.ff_linear1.forward(&norm1_output);
        let ff_output = self.apply_relu(&ff_output); // ReLU activation
        let ff_output = self.ff_linear2.forward(&ff_output);
        let ff_output = self.dropout2.forward(&ff_output);
        let ff_residual = self.add_tensors(&norm1_output, &ff_output);
        let norm2_output = self.norm2.forward(&ff_residual);
        
        norm2_output
    }
    
    /// Add two tensors (residual connection)
    /// 二つのテンソルを加算（残差接続）
    fn add_tensors(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        if a_data.shape() != b_data.shape() {
            panic!("Cannot add tensors with different shapes: {:?} vs {:?}", 
                   a_data.shape(), b_data.shape());
        }
        
        let mut result_data = Vec::with_capacity(a_array.len());
        
        if let (Some(a_slice), Some(b_slice)) = (a_array.as_slice(), b_array.as_slice()) {
            for (&a_val, &b_val) in a_slice.iter().zip(b_slice.iter()) {
                result_data.push(a_val + b_val);
            }
        } else {
            // Fallback for non-contiguous arrays
            for _i in 0..a_array.len() {
                result_data.push(T::zero()); // Simplified
            }
        }
        
        Variable::new(
            Tensor::from_vec(result_data, a_data.shape().to_vec()),
            a.requires_grad() || b.requires_grad()
        )
    }
    
    /// Apply ReLU activation
    /// ReLU活性化を適用
    fn apply_relu(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        if let Some(input_slice) = input_array.as_slice() {
            for &val in input_slice {
                output_data.push(if val > T::zero() { val } else { T::zero() });
            }
        } else {
            // Fallback
            for _ in 0..input_array.len() {
                output_data.push(T::zero());
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, input_data.shape().to_vec()),
            input.requires_grad()
        )
    }
    
    /// Returns the model dimension
    /// モデル次元を返します
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Returns the feed-forward dimension
    /// フィードフォワード次元を返します
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attention.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

impl<T> Module<T> for TransformerEncoderLayer<T>
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

/// Transformer Encoder
/// Transformerエンコーダー
/// 
/// Stack of Transformer encoder layers.
/// Transformerエンコーダー層のスタック。
#[derive(Debug)]
pub struct TransformerEncoder<T: Float + Send + Sync> {
    /// Stack of encoder layers
    /// エンコーダー層のスタック
    layers: Vec<TransformerEncoderLayer<T>>,
    
    /// Number of layers
    /// 層数
    num_layers: usize,
    
    /// Model dimension
    /// モデル次元
    d_model: usize,
}

impl<T> TransformerEncoder<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new TransformerEncoder
    /// 新しいTransformerEncoderを作成します
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: Option<T>,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be greater than 0");
        
        let mut layers = Vec::with_capacity(num_layers);
        
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(d_model, num_heads, d_ff, dropout));
        }
        
        TransformerEncoder {
            layers,
            num_layers,
            d_model,
        }
    }
    
    /// Forward pass of TransformerEncoder
    /// TransformerEncoderの順伝播
    pub fn forward(&self, input: &Variable<T>, mask: Option<&Variable<T>>) -> Variable<T> {
        let mut x = input.clone();
        
        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }
        
        x
    }
    
    /// Returns the number of layers
    /// 層数を返します
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    
    /// Returns the model dimension
    /// モデル次元を返します
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Returns the parameters of all layers
    /// すべての層のパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl<T> Module<T> for TransformerEncoder<T>
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

/// Transformer Decoder Layer
/// Transformerデコーダー層
/// 
/// A single layer of the Transformer decoder with masked self-attention and cross-attention.
/// マスク付きセルフアテンションとクロスアテンションを持つTransformerデコーダーの単一層。
#[derive(Debug)]
pub struct TransformerDecoderLayer<T: Float + Send + Sync> {
    /// Masked multi-head self-attention
    /// マスク付きマルチヘッドセルフアテンション
    self_attention: MultiHeadAttention<T>,
    
    /// Multi-head cross-attention
    /// マルチヘッドクロスアテンション
    cross_attention: MultiHeadAttention<T>,
    
    /// Feed-forward network (first linear layer)
    /// フィードフォワードネットワーク（第一線形層）
    ff_linear1: Linear<T>,
    
    /// Feed-forward network (second linear layer)
    /// フィードフォワードネットワーク（第二線形層）
    ff_linear2: Linear<T>,
    
    /// Layer normalization for self-attention
    /// セルフアテンション用レイヤー正規化
    norm1: LayerNorm<T>,
    
    /// Layer normalization for cross-attention
    /// クロスアテンション用レイヤー正規化
    norm2: LayerNorm<T>,
    
    /// Layer normalization for feed-forward
    /// フィードフォワード用レイヤー正規化
    norm3: LayerNorm<T>,
    
    /// Dropout for self-attention
    /// セルフアテンション用ドロップアウト
    dropout1: Dropout<T>,
    
    /// Dropout for cross-attention
    /// クロスアテンション用ドロップアウト
    dropout2: Dropout<T>,
    
    /// Dropout for feed-forward
    /// フィードフォワード用ドロップアウト
    dropout3: Dropout<T>,
    
    /// Model dimension
    /// モデル次元
    d_model: usize,
    
    /// Feed-forward dimension
    /// フィードフォワード次元
    d_ff: usize,
}

impl<T> TransformerDecoderLayer<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new TransformerDecoderLayer
    /// 新しいTransformerDecoderLayerを作成します
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: Option<T>,
    ) -> Self {
        assert!(d_model > 0, "d_model must be greater than 0");
        assert!(num_heads > 0, "num_heads must be greater than 0");
        assert!(d_ff > 0, "d_ff must be greater than 0");
        assert!(d_model % num_heads == 0, 
                "d_model ({}) must be divisible by num_heads ({})", d_model, num_heads);
        
        let dropout_p = dropout.unwrap_or_else(|| T::from(0.1).unwrap());
        
        // Create components
        let self_attention = MultiHeadAttention::new(d_model, num_heads, Some(dropout_p), Some(true));
        let cross_attention = MultiHeadAttention::new(d_model, num_heads, Some(dropout_p), Some(true));
        let ff_linear1 = Linear::new(d_model, d_ff);
        let ff_linear2 = Linear::new(d_ff, d_model);
        let norm1 = LayerNorm::new(vec![d_model], None, None);
        let norm2 = LayerNorm::new(vec![d_model], None, None);
        let norm3 = LayerNorm::new(vec![d_model], None, None);
        let dropout1 = Dropout::new(dropout_p, false);
        let dropout2 = Dropout::new(dropout_p, false);
        let dropout3 = Dropout::new(dropout_p, false);
        
        TransformerDecoderLayer {
            self_attention,
            cross_attention,
            ff_linear1,
            ff_linear2,
            norm1,
            norm2,
            norm3,
            dropout1,
            dropout2,
            dropout3,
            d_model,
            d_ff,
        }
    }
    
    /// Forward pass of TransformerDecoderLayer
    /// TransformerDecoderLayerの順伝播
    pub fn forward(
        &self,
        target: &Variable<T>,
        memory: &Variable<T>,
        target_mask: Option<&Variable<T>>,
        memory_mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        // Masked self-attention with residual connection and layer norm
        let self_attn_output = self.self_attention.forward(target, target, target, target_mask);
        let self_attn_output = self.dropout1.forward(&self_attn_output);
        let self_attn_residual = self.add_tensors(target, &self_attn_output);
        let norm1_output = self.norm1.forward(&self_attn_residual);
        
        // Cross-attention with residual connection and layer norm
        let cross_attn_output = self.cross_attention.forward(&norm1_output, memory, memory, memory_mask);
        let cross_attn_output = self.dropout2.forward(&cross_attn_output);
        let cross_attn_residual = self.add_tensors(&norm1_output, &cross_attn_output);
        let norm2_output = self.norm2.forward(&cross_attn_residual);
        
        // Feed-forward network with residual connection and layer norm
        let ff_output = self.ff_linear1.forward(&norm2_output);
        let ff_output = self.apply_relu(&ff_output);
        let ff_output = self.ff_linear2.forward(&ff_output);
        let ff_output = self.dropout3.forward(&ff_output);
        let ff_residual = self.add_tensors(&norm2_output, &ff_output);
        let norm3_output = self.norm3.forward(&ff_residual);
        
        norm3_output
    }
    
    /// Add two tensors (residual connection)
    /// 二つのテンソルを加算（残差接続）
    fn add_tensors(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        
        let a_array = a_data.as_array();
        let b_array = b_data.as_array();
        
        if a_data.shape() != b_data.shape() {
            panic!("Cannot add tensors with different shapes: {:?} vs {:?}", 
                   a_data.shape(), b_data.shape());
        }
        
        let mut result_data = Vec::with_capacity(a_array.len());
        
        if let (Some(a_slice), Some(b_slice)) = (a_array.as_slice(), b_array.as_slice()) {
            for (&a_val, &b_val) in a_slice.iter().zip(b_slice.iter()) {
                result_data.push(a_val + b_val);
            }
        } else {
            // Fallback for non-contiguous arrays
            for _i in 0..a_array.len() {
                result_data.push(T::zero()); // Simplified
            }
        }
        
        Variable::new(
            Tensor::from_vec(result_data, a_data.shape().to_vec()),
            a.requires_grad() || b.requires_grad()
        )
    }
    
    /// Apply ReLU activation
    /// ReLU活性化を適用
    fn apply_relu(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_array = input_data.as_array();
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        if let Some(input_slice) = input_array.as_slice() {
            for &val in input_slice {
                output_data.push(if val > T::zero() { val } else { T::zero() });
            }
        } else {
            // Fallback
            for _ in 0..input_array.len() {
                output_data.push(T::zero());
            }
        }
        
        Variable::new(
            Tensor::from_vec(output_data, input_data.shape().to_vec()),
            input.requires_grad()
        )
    }
    
    /// Returns the model dimension
    /// モデル次元を返します
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Returns the feed-forward dimension
    /// フィードフォワード次元を返します
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attention.parameters());
        params.extend(self.cross_attention.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }
}

impl<T> Module<T> for TransformerDecoderLayer<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For Module trait, use input as both target and memory
        self.forward(input, input, None, None)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Complete Transformer model
/// 完全なTransformerモデル
/// 
/// Full Transformer architecture with encoder and decoder stacks.
/// エンコーダーとデコーダースタックを持つ完全なTransformerアーキテクチャ。
#[derive(Debug)]
pub struct Transformer<T: Float + Send + Sync> {
    /// Transformer encoder
    /// Transformerエンコーダー
    encoder: TransformerEncoder<T>,
    
    /// Transformer decoder layers
    /// Transformerデコーダー層
    decoder_layers: Vec<TransformerDecoderLayer<T>>,
    
    /// Number of decoder layers
    /// デコーダー層数
    num_decoder_layers: usize,
    
    /// Model dimension
    /// モデル次元
    d_model: usize,
}

impl<T> Transformer<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new Transformer
    /// 新しいTransformerを作成します
    pub fn new(
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: Option<T>,
    ) -> Self {
        let encoder = TransformerEncoder::new(num_encoder_layers, d_model, num_heads, d_ff, dropout);
        
        let mut decoder_layers = Vec::with_capacity(num_decoder_layers);
        for _ in 0..num_decoder_layers {
            decoder_layers.push(TransformerDecoderLayer::new(d_model, num_heads, d_ff, dropout));
        }
        
        Transformer {
            encoder,
            decoder_layers,
            num_decoder_layers,
            d_model,
        }
    }
    
    /// Forward pass of Transformer
    /// Transformerの順伝播
    pub fn forward(
        &self,
        src: &Variable<T>,
        tgt: &Variable<T>,
        src_mask: Option<&Variable<T>>,
        tgt_mask: Option<&Variable<T>>,
        memory_mask: Option<&Variable<T>>,
    ) -> Variable<T> {
        // Encode source sequence
        let memory = self.encoder.forward(src, src_mask);
        
        // Decode target sequence
        let mut x = tgt.clone();
        
        for layer in &self.decoder_layers {
            x = layer.forward(&x, &memory, tgt_mask, memory_mask);
        }
        
        x
    }
    
    /// Encode only (for encoder-only models like BERT)
    /// エンコードのみ（BERTのようなエンコーダーのみのモデル用）
    pub fn encode(&self, src: &Variable<T>, src_mask: Option<&Variable<T>>) -> Variable<T> {
        self.encoder.forward(src, src_mask)
    }
    
    /// Returns the model dimension
    /// モデル次元を返します
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Returns the number of encoder layers
    /// エンコーダー層数を返します
    pub fn num_encoder_layers(&self) -> usize {
        self.encoder.num_layers()
    }
    
    /// Returns the number of decoder layers
    /// デコーダー層数を返します
    pub fn num_decoder_layers(&self) -> usize {
        self.num_decoder_layers
    }
    
    /// Returns all parameters
    /// すべてのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        for layer in &self.decoder_layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl<T> Module<T> for Transformer<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For Module trait, use input as both source and target
        self.forward(input, input, None, None, None)
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
    
    #[test]
    fn test_transformer_encoder_layer_creation() {
        let layer = TransformerEncoderLayer::<f32>::new(512, 8, 2048, None);
        
        assert_eq!(layer.d_model(), 512);
        assert_eq!(layer.d_ff(), 2048);
        
        let params = layer.parameters();
        assert!(params.len() > 0); // Should have multiple parameters
    }
    
    #[test]
    fn test_transformer_encoder_creation() {
        let encoder = TransformerEncoder::<f32>::new(6, 512, 8, 2048, None);
        
        assert_eq!(encoder.num_layers(), 6);
        assert_eq!(encoder.d_model(), 512);
        
        let params = encoder.parameters();
        assert!(params.len() > 0);
    }
    
    #[test]
    fn test_transformer_creation() {
        let transformer = Transformer::<f32>::new(6, 6, 512, 8, 2048, None);
        
        assert_eq!(transformer.num_encoder_layers(), 6);
        assert_eq!(transformer.num_decoder_layers(), 6);
        assert_eq!(transformer.d_model(), 512);
        
        let params = transformer.parameters();
        assert!(params.len() > 0);
    }
}
