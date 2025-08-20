//! Embedding layers implementation
//! 埋め込み層の実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use rand::distributions::Distribution;
use rand_distr::Normal;

/// Word Embedding layer
/// 単語埋め込み層
/// 
/// Maps discrete tokens to dense vectors of fixed size.
/// 離散的なトークンを固定サイズの密ベクトルにマッピングします。
#[derive(Debug)]
pub struct Embedding<T: Float + Send + Sync> {
    /// Embedding weight matrix of shape (vocab_size, embedding_dim)
    /// 埋め込み重み行列 (語彙サイズ, 埋め込み次元)
    weight: Variable<T>,
    
    /// Vocabulary size
    /// 語彙サイズ
    vocab_size: usize,
    
    /// Embedding dimension
    /// 埋め込み次元
    embedding_dim: usize,
    
    /// Padding index (tokens at this index will have zero gradients)
    /// パディングインデックス（このインデックスのトークンは勾配がゼロになる）
    padding_idx: Option<usize>,
    
    /// Maximum norm for embedding vectors
    /// 埋め込みベクトルの最大ノルム
    max_norm: Option<T>,
    
    /// Whether to freeze the embedding weights
    /// 埋め込み重みを凍結するかどうか
    freeze: bool,
}

impl<T> Embedding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    /// Creates a new Embedding layer
    /// 新しいEmbedding層を作成します
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<T>,
        freeze: Option<bool>,
    ) -> Self {
        assert!(vocab_size > 0, "vocab_size must be greater than 0");
        assert!(embedding_dim > 0, "embedding_dim must be greater than 0");
        
        let freeze = freeze.unwrap_or(false);
        
        // Initialize embedding weights with normal distribution
        let std = T::from_f32(1.0 / (embedding_dim as f32).sqrt()).unwrap();
        let normal = Normal::new(0.0, std.to_f64().unwrap()).unwrap();
        let mut rng = rand::thread_rng();
        
        let weight_data: Vec<T> = (0..vocab_size * embedding_dim)
            .map(|_| T::from_f32(normal.sample(&mut rng) as f32).unwrap())
            .collect();
        
        let weight = Variable::new(
            Tensor::from_vec(weight_data, vec![vocab_size, embedding_dim]),
            !freeze, // requires_grad is opposite of freeze
        );
        
        // Zero out padding index if specified
        if let Some(pad_idx) = padding_idx {
            if pad_idx < vocab_size {
                let weight_binding = weight.data();
                let mut weight_data = weight_binding.write().unwrap();
                let weight_array = weight_data.as_array_mut();
                
                // Zero out the padding row
                for j in 0..embedding_dim {
                    weight_array[[pad_idx, j]] = T::zero();
                }
            }
        }
        
        Embedding {
            weight,
            vocab_size,
            embedding_dim,
            padding_idx,
            max_norm,
            freeze,
        }
    }
    
    /// Forward pass of the Embedding layer
    /// Embedding層の順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Input should contain integer indices
        // Convert to usize indices for lookup
        let indices = self.extract_indices(&input_data);
        
        // Perform embedding lookup
        let output_data = self.embedding_lookup(&indices, input_shape);
        
        let requires_grad = !self.freeze && self.weight.requires_grad();
        Variable::new(output_data, requires_grad)
    }
    
    /// Extract indices from input tensor
    /// 入力テンソルからインデックスを抽出
    fn extract_indices(&self, input: &Tensor<T>) -> Vec<usize> {
        input.as_array().iter()
            .map(|&x| {
                let idx = x.to_usize().unwrap_or(0);
                if idx >= self.vocab_size {
                    panic!("Index {} is out of bounds for vocab_size {}", idx, self.vocab_size);
                }
                idx
            })
            .collect()
    }
    
    /// Perform embedding lookup
    /// 埋め込み検索を実行
    fn embedding_lookup(&self, indices: &[usize], input_shape: &[usize]) -> Tensor<T> {
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        let weight_array = weight_data.as_array();
        
        // Create output shape: input_shape + [embedding_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);
        
        // Lookup embeddings
        let mut output_data = Vec::with_capacity(indices.len() * self.embedding_dim);
        
        for &idx in indices {
            for j in 0..self.embedding_dim {
                let embedding_val = weight_array[[idx, j]];
                
                // Apply max_norm constraint if specified
                let final_val = if let Some(max_norm) = self.max_norm {
                    // Simplified norm constraint (would need proper vector norm in full implementation)
                    if embedding_val.abs() > max_norm {
                        if embedding_val > T::zero() { max_norm } else { -max_norm }
                    } else {
                        embedding_val
                    }
                } else {
                    embedding_val
                };
                
                output_data.push(final_val);
            }
        }
        
        Tensor::from_vec(output_data, output_shape)
    }
    
    /// Returns the vocabulary size
    /// 語彙サイズを返します
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Returns the embedding dimension
    /// 埋め込み次元を返します
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    /// Returns the padding index
    /// パディングインデックスを返します
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }
    
    /// Freezes or unfreezes the embedding weights
    /// 埋め込み重みを凍結または解凍します
    pub fn set_freeze(&mut self, freeze: bool) {
        self.freeze = freeze;
        // Update requires_grad of weight
        let weight_data = self.weight.data();
        let _weight_lock = weight_data.write().unwrap();
        // Note: In a full implementation, we'd update the requires_grad flag
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        if self.freeze {
            vec![]
        } else {
            vec![self.weight.clone()]
        }
    }
}

impl<T> Module<T> for Embedding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Positional Embedding layer
/// 位置埋め込み層
/// 
/// Adds positional information to input embeddings using learnable parameters.
/// 学習可能なパラメータを使用して入力埋め込みに位置情報を追加します。
#[derive(Debug)]
pub struct PositionalEmbedding<T: Float + Send + Sync> {
    /// Positional embedding weight matrix of shape (max_length, embedding_dim)
    /// 位置埋め込み重み行列 (最大長, 埋め込み次元)
    weight: Variable<T>,
    
    /// Maximum sequence length
    /// 最大シーケンス長
    max_length: usize,
    
    /// Embedding dimension
    /// 埋め込み次元
    embedding_dim: usize,
}

impl<T> PositionalEmbedding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    /// Creates a new PositionalEmbedding layer
    /// 新しいPositionalEmbedding層を作成します
    pub fn new(max_length: usize, embedding_dim: usize) -> Self {
        assert!(max_length > 0, "max_length must be greater than 0");
        assert!(embedding_dim > 0, "embedding_dim must be greater than 0");
        
        // Initialize with small random values
        let std = T::from_f32(0.02).unwrap();
        let normal = Normal::new(0.0, std.to_f64().unwrap()).unwrap();
        let mut rng = rand::thread_rng();
        
        let weight_data: Vec<T> = (0..max_length * embedding_dim)
            .map(|_| T::from_f32(normal.sample(&mut rng) as f32).unwrap())
            .collect();
        
        let weight = Variable::new(
            Tensor::from_vec(weight_data, vec![max_length, embedding_dim]),
            true,
        );
        
        PositionalEmbedding {
            weight,
            max_length,
            embedding_dim,
        }
    }
    
    /// Forward pass of the PositionalEmbedding layer
    /// PositionalEmbedding層の順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Input shape should be (batch_size, seq_length, embedding_dim)
        if input_shape.len() != 3 {
            panic!("PositionalEmbedding expects 3D input (batch_size, seq_length, embedding_dim), got {:?}", input_shape);
        }
        
        let batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let embed_dim = input_shape[2];
        
        if embed_dim != self.embedding_dim {
            panic!("Input embedding_dim {} doesn't match layer embedding_dim {}", embed_dim, self.embedding_dim);
        }
        
        if seq_length > self.max_length {
            panic!("Sequence length {} exceeds max_length {}", seq_length, self.max_length);
        }
        
        // Get positional embeddings for the sequence length
        let pos_embeddings = self.get_positional_embeddings(seq_length);
        
        // Add positional embeddings to input
        let output_data = self.add_positional_embeddings(&input_data, &pos_embeddings, batch_size, seq_length);
        
        let requires_grad = input.requires_grad() || self.weight.requires_grad();
        Variable::new(output_data, requires_grad)
    }
    
    /// Get positional embeddings for given sequence length
    /// 指定されたシーケンス長の位置埋め込みを取得
    fn get_positional_embeddings(&self, seq_length: usize) -> Tensor<T> {
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        let weight_array = weight_data.as_array();
        
        // Extract first seq_length rows
        let mut pos_data = Vec::with_capacity(seq_length * self.embedding_dim);
        
        for i in 0..seq_length {
            for j in 0..self.embedding_dim {
                pos_data.push(weight_array[[i, j]]);
            }
        }
        
        Tensor::from_vec(pos_data, vec![seq_length, self.embedding_dim])
    }
    
    /// Add positional embeddings to input
    /// 入力に位置埋め込みを追加
    fn add_positional_embeddings(
        &self,
        input: &Tensor<T>,
        pos_embeddings: &Tensor<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Tensor<T> {
        let input_array = input.as_array();
        let pos_array = pos_embeddings.as_array();
        
        let mut output_data = Vec::with_capacity(batch_size * seq_length * self.embedding_dim);
        
        for b in 0..batch_size {
            for s in 0..seq_length {
                for d in 0..self.embedding_dim {
                    let input_val = input_array[[b, s, d]];
                    let pos_val = pos_array[[s, d]];
                    output_data.push(input_val + pos_val);
                }
            }
        }
        
        Tensor::from_vec(output_data, vec![batch_size, seq_length, self.embedding_dim])
    }
    
    /// Returns the maximum sequence length
    /// 最大シーケンス長を返します
    pub fn max_length(&self) -> usize {
        self.max_length
    }
    
    /// Returns the embedding dimension
    /// 埋め込み次元を返します
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone()]
    }
}

impl<T> Module<T> for PositionalEmbedding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Sinusoidal Positional Encoding (fixed, non-learnable)
/// 正弦波位置エンコーディング（固定、学習不可）
/// 
/// Uses sinusoidal functions to encode positional information.
/// 正弦波関数を使用して位置情報をエンコードします。
#[derive(Debug)]
pub struct SinusoidalPositionalEncoding<T: Float + Send + Sync> {
    /// Maximum sequence length
    /// 最大シーケンス長
    max_length: usize,
    
    /// Embedding dimension
    /// 埋め込み次元
    embedding_dim: usize,
    
    /// Precomputed positional encodings
    /// 事前計算された位置エンコーディング
    encodings: Tensor<T>,
}

impl<T> SinusoidalPositionalEncoding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    /// Creates a new SinusoidalPositionalEncoding layer
    /// 新しいSinusoidalPositionalEncoding層を作成します
    pub fn new(max_length: usize, embedding_dim: usize) -> Self {
        assert!(max_length > 0, "max_length must be greater than 0");
        assert!(embedding_dim > 0, "embedding_dim must be greater than 0");
        
        // Precompute sinusoidal encodings
        let encodings = Self::create_sinusoidal_encodings(max_length, embedding_dim);
        
        SinusoidalPositionalEncoding {
            max_length,
            embedding_dim,
            encodings,
        }
    }
    
    /// Create sinusoidal positional encodings
    /// 正弦波位置エンコーディングを作成
    fn create_sinusoidal_encodings(max_length: usize, embedding_dim: usize) -> Tensor<T> {
        let mut encodings = Vec::with_capacity(max_length * embedding_dim);
        
        for pos in 0..max_length {
            for i in 0..embedding_dim {
                let pos_f = pos as f32;
                let i_f = i as f32;
                let dim_f = embedding_dim as f32;
                
                let angle = pos_f / (10000.0_f32).powf(2.0 * (i_f / 2.0).floor() / dim_f);
                
                let encoding = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                
                encodings.push(T::from_f32(encoding).unwrap());
            }
        }
        
        Tensor::from_vec(encodings, vec![max_length, embedding_dim])
    }
    
    /// Forward pass of the SinusoidalPositionalEncoding layer
    /// SinusoidalPositionalEncoding層の順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Input shape should be (batch_size, seq_length, embedding_dim)
        if input_shape.len() != 3 {
            panic!("SinusoidalPositionalEncoding expects 3D input (batch_size, seq_length, embedding_dim), got {:?}", input_shape);
        }
        
        let batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let embed_dim = input_shape[2];
        
        if embed_dim != self.embedding_dim {
            panic!("Input embedding_dim {} doesn't match layer embedding_dim {}", embed_dim, self.embedding_dim);
        }
        
        if seq_length > self.max_length {
            panic!("Sequence length {} exceeds max_length {}", seq_length, self.max_length);
        }
        
        // Add positional encodings to input
        let output_data = self.add_sinusoidal_encodings(&input_data, batch_size, seq_length);
        
        Variable::new(output_data, input.requires_grad())
    }
    
    /// Add sinusoidal encodings to input
    /// 入力に正弦波エンコーディングを追加
    fn add_sinusoidal_encodings(
        &self,
        input: &Tensor<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Tensor<T> {
        let input_array = input.as_array();
        let encodings_array = self.encodings.as_array();
        
        let mut output_data = Vec::with_capacity(batch_size * seq_length * self.embedding_dim);
        
        for b in 0..batch_size {
            for s in 0..seq_length {
                for d in 0..self.embedding_dim {
                    let input_val = input_array[[b, s, d]];
                    let encoding_val = encodings_array[[s, d]];
                    output_data.push(input_val + encoding_val);
                }
            }
        }
        
        Tensor::from_vec(output_data, vec![batch_size, seq_length, self.embedding_dim])
    }
    
    /// Returns the maximum sequence length
    /// 最大シーケンス長を返します
    pub fn max_length(&self) -> usize {
        self.max_length
    }
    
    /// Returns the embedding dimension
    /// 埋め込み次元を返します
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl<T> Module<T> for SinusoidalPositionalEncoding<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        // Sinusoidal encoding has no learnable parameters
        vec![]
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
    fn test_embedding_creation() {
        let embedding = Embedding::<f32>::new(1000, 128, Some(0), None, None);
        
        assert_eq!(embedding.vocab_size(), 1000);
        assert_eq!(embedding.embedding_dim(), 128);
        assert_eq!(embedding.padding_idx(), Some(0));
        
        let params = embedding.parameters();
        assert_eq!(params.len(), 1);
        
        let weight_binding = params[0].data();
        let weight_data = weight_binding.read().unwrap();
        assert_eq!(weight_data.shape(), &[1000, 128]);
    }
    
    #[test]
    fn test_embedding_forward() {
        let embedding = Embedding::<f32>::new(10, 4, None, None, None);
        
        // Create input with token indices [1, 2, 3]
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            false
        );
        
        let output = embedding.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        // Output shape should be [3, 4] (3 tokens, 4 embedding dims)
        assert_eq!(output_data.shape(), &[3, 4]);
    }
    
    #[test]
    fn test_positional_embedding() {
        let pos_emb = PositionalEmbedding::<f32>::new(100, 64);
        
        assert_eq!(pos_emb.max_length(), 100);
        assert_eq!(pos_emb.embedding_dim(), 64);
        
        // Create input: batch_size=2, seq_length=5, embedding_dim=64
        let input_data: Vec<f32> = (0..2*5*64).map(|i| i as f32 * 0.01).collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![2, 5, 64]),
            false
        );
        
        let output = pos_emb.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        // Output shape should be same as input
        assert_eq!(output_data.shape(), &[2, 5, 64]);
    }
    
    #[test]
    fn test_sinusoidal_positional_encoding() {
        let sin_pos = SinusoidalPositionalEncoding::<f32>::new(50, 32);
        
        assert_eq!(sin_pos.max_length(), 50);
        assert_eq!(sin_pos.embedding_dim(), 32);
        
        // Create input: batch_size=1, seq_length=10, embedding_dim=32
        let input_data: Vec<f32> = (0..1*10*32).map(|i| i as f32 * 0.01).collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![1, 10, 32]),
            false
        );
        
        let output = sin_pos.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        // Output shape should be same as input
        assert_eq!(output_data.shape(), &[1, 10, 32]);
        
        // Sinusoidal encoding has no parameters
        assert_eq!(sin_pos.parameters().len(), 0);
    }
    
    #[test]
    fn test_embedding_with_padding() {
        let embedding = Embedding::<f32>::new(5, 3, Some(0), None, None);
        
        // Check that padding index (0) has zero embeddings
        let weight_binding = embedding.weight.data();
        let weight_data = weight_binding.read().unwrap();
        let weight_array = weight_data.as_array();
        
        for j in 0..3 {
            assert_abs_diff_eq!(weight_array[[0, j]], 0.0, epsilon = 1e-6);
        }
    }
}
