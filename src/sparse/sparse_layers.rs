//! Sparse neural network layers
//! スパースニューラルネットワーク層

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use crate::nn::Module;
use super::{SparseTensor, SparseFormat, SparseOps};
use super::pruning::{ModelPruner, PruningConfig, PruningStrategy};
use ndarray::{ArrayD, Array1, Array2};
use num_traits::{Float, Zero, One, FromPrimitive};
use std::collections::HashMap;

/// Sparse linear layer with efficient sparse matrix operations
/// 効率的スパース行列演算付きスパース線形層
pub struct SparseLinear<T: Float> {
    /// Sparse weight matrix in CSR format for efficient computation
    /// 効率的計算のためのCSR形式スパース重み行列
    pub weight: SparseTensor<T>,
    /// Dense bias vector (kept dense for efficiency)
    /// 密バイアスベクトル（効率のため密で保持）
    pub bias: Option<Array1<T>>,
    /// Input and output dimensions
    /// 入力と出力次元
    pub in_features: usize,
    pub out_features: usize,
    /// Pruning configuration for dynamic sparsification
    /// 動的スパース化用プルーニング設定
    pub pruning_config: Option<PruningConfig>,
}

impl<T: Float + Zero + One + std::ops::AddAssign + Copy + PartialOrd + Send + Sync + ndarray::ScalarOperand + FromPrimitive> SparseLinear<T> {
    /// Create sparse linear layer from dense weights
    /// 密重みからスパース線形層を作成
    pub fn from_dense(
        weight: Array2<T>,
        bias: Option<Array1<T>>,
        threshold: T,
    ) -> RusTorchResult<Self> {
        let (out_features, in_features) = weight.dim();
        
        // Convert to sparse CSR format for efficient SpMV
        let weight_dense = weight.into_dyn();
        let sparse_weight = SparseTensor::from_dense(&weight_dense, threshold)?;
        let csr_weight = sparse_weight.to_csr()?;

        Ok(Self {
            weight: csr_weight,
            bias,
            in_features,
            out_features,
            pruning_config: None,
        })
    }

    /// Create sparse linear layer with pruning
    /// プルーニング付きスパース線形層を作成
    pub fn with_pruning(
        weight: Array2<T>,
        bias: Option<Array1<T>>,
        pruning_config: PruningConfig,
    ) -> RusTorchResult<Self> {
        let (out_features, in_features) = weight.dim();
        
        // Apply initial pruning
        let pruner = ModelPruner::new(pruning_config.clone());
        let weight_dense = weight.into_dyn();
        let sparse_weight = pruner.prune_tensor(&weight_dense)?;
        let csr_weight = sparse_weight.to_csr()?;

        Ok(Self {
            weight: csr_weight,
            bias,
            in_features,
            out_features,
            pruning_config: Some(pruning_config),
        })
    }

    /// Forward pass using sparse matrix operations
    /// スパース行列演算を使用したフォワードパス
    pub fn forward(&self, input: &Variable<T>) -> RusTorchResult<Variable<T>> {
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();

        // Handle batch dimension
        let (batch_size, features) = if input_shape.len() == 1 {
            (1, input_shape[0])
        } else if input_shape.len() == 2 {
            (input_shape[0], input_shape[1])
        } else {
            return Err(RusTorchError::InvalidOperation {
                operation: "sparse_linear_forward".to_string(),
                message: "Input must be 1D or 2D tensor".to_string(),
            });
        };

        if features != self.in_features {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self.in_features],
                actual: vec![features],
            });
        }

        let mut output_data = Array2::zeros((batch_size, self.out_features));

        // Perform sparse matrix-vector multiplication for each batch element
        for b in 0..batch_size {
            let input_vector = if batch_size == 1 {
                Array1::from_shape_vec(features, input_guard.data.as_slice().unwrap().to_vec())?
            } else {
                input_guard.data.slice(ndarray::s![b, ..]).to_owned()
            };

            let result = self.weight.spmv(&input_vector)?;
            
            // Add bias if present
            let final_result = if let Some(ref bias) = self.bias {
                &result + bias
            } else {
                result
            };

            for (i, &val) in final_result.iter().enumerate() {
                output_data[[b, i]] = val;
            }
        }

        let output_shape = if batch_size == 1 {
            vec![self.out_features]
        } else {
            vec![batch_size, self.out_features]
        };

        let reshaped_output = output_data.to_shape(output_shape)?.into_owned().into_dyn();
        
        Ok(Variable::new(
            Tensor::from_ndarray(reshaped_output),
            input.requires_grad(),
        ))
    }

    /// Apply pruning to existing weights
    /// 既存重みにプルーニングを適用
    pub fn prune_weights(&mut self, new_sparsity: f32) -> RusTorchResult<()> {
        if let Some(ref mut config) = self.pruning_config {
            config.target_sparsity = new_sparsity;
            let pruner = ModelPruner::new(config.clone());
            
            // Convert to dense, prune, convert back to sparse
            let dense_weight = self.weight.to_dense()?;
            let new_sparse_weight = pruner.prune_tensor(&dense_weight)?;
            self.weight = new_sparse_weight.to_csr()?;
        }
        Ok(())
    }

    /// Get current sparsity level
    /// 現在のスパース率レベルを取得
    pub fn sparsity(&self) -> f64 {
        self.weight.sparsity()
    }

    /// Get memory usage compared to dense equivalent
    /// 密等価品と比較したメモリ使用量を取得
    pub fn memory_efficiency(&self) -> f64 {
        let sparse_memory = self.weight.memory_usage();
        let dense_memory = self.in_features * self.out_features * std::mem::size_of::<T>();
        
        1.0 - (sparse_memory as f64 / dense_memory as f64)
    }
}

/// Sparse convolutional layer (placeholder for future implementation)
/// スパース畳み込み層（将来実装用プレースホルダー）
pub struct SparseConv2d<T: Float> {
    /// Sparse convolution kernels
    /// スパース畳み込みカーネル
    pub weight: SparseTensor<T>,
    /// Dense bias (optional)
    /// 密バイアス（オプション）
    pub bias: Option<Array1<T>>,
    /// Convolution parameters
    /// 畳み込みパラメータ
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
}

impl<T: Float + Zero + One + Copy + PartialOrd + Send + Sync + ndarray::ScalarOperand + FromPrimitive + std::ops::AddAssign> SparseConv2d<T> {
    /// Create sparse convolutional layer (simplified implementation)
    /// スパース畳み込み層を作成（簡略化実装）
    pub fn from_dense(
        weight: ArrayD<T>,
        bias: Option<Array1<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
        threshold: T,
    ) -> RusTorchResult<Self> {
        if weight.ndim() != 4 {
            return Err(RusTorchError::InvalidParameters {
                operation: "sparse_conv2d_creation".to_string(),
                message: "Conv2D weights must be 4D (out_channels, in_channels, height, width)".to_string(),
            });
        }

        let shape = weight.shape();
        let (out_channels, in_channels, kernel_h, kernel_w) = (shape[0], shape[1], shape[2], shape[3]);
        
        let sparse_weight = SparseTensor::from_dense(&weight, threshold)?;

        Ok(Self {
            weight: sparse_weight,
            bias,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
        })
    }

    /// Forward pass (placeholder - would implement optimized sparse convolution)
    /// フォワードパス（プレースホルダー - 最適化スパース畳み込みを実装予定）
    pub fn forward(&self, input: &Variable<T>) -> RusTorchResult<Variable<T>> {
        // For now, return input unchanged as placeholder
        // This would be replaced with efficient sparse convolution implementation
        Ok(input.clone())
    }

    /// Get filter sparsity information
    /// フィルタースパース情報を取得
    pub fn filter_sparsity(&self) -> Vec<f64> {
        // Calculate sparsity per output channel
        // 出力チャンネル毎のスパース率を計算
        let mut sparsities = Vec::with_capacity(self.out_channels);
        
        // This is a simplified calculation - would need proper filter extraction
        for _ in 0..self.out_channels {
            sparsities.push(self.weight.sparsity());
        }
        
        sparsities
    }
}

/// Sparse embedding layer for efficient sparse lookups
/// 効率的スパース検索のためのスパース埋め込み層
pub struct SparseEmbedding<T: Float> {
    /// Sparse embedding table
    /// スパース埋め込みテーブル
    pub weight: SparseTensor<T>,
    /// Vocabulary size
    /// 語彙サイズ
    pub num_embeddings: usize,
    /// Embedding dimension
    /// 埋め込み次元
    pub embedding_dim: usize,
    /// Padding index (if any)
    /// パディングインデックス（もしあれば）
    pub padding_idx: Option<usize>,
}

impl<T: Float + Zero + One + Copy + Send + Sync + ndarray::ScalarOperand + FromPrimitive + std::ops::AddAssign> SparseEmbedding<T> {
    /// Create sparse embedding layer
    /// スパース埋め込み層を作成
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        sparsity: f32,
        padding_idx: Option<usize>,
    ) -> RusTorchResult<Self> {
        // Initialize with random sparse weights
        // ランダムスパース重みで初期化
        let total_elements = num_embeddings * embedding_dim;
        let nnz = ((1.0 - sparsity) * total_elements as f32) as usize;
        
        // Generate random indices and values for initialization
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        
        for _ in 0..nnz {
            row_indices.push(rng.gen_range(0..num_embeddings));
            col_indices.push(rng.gen_range(0..embedding_dim));
            values.push(T::from(rng.gen_range(-0.1..0.1)).unwrap());
        }
        
        let indices = vec![
            Array1::from_vec(row_indices),
            Array1::from_vec(col_indices),
        ];
        let values_array = Array1::from_vec(values);
        let shape = vec![num_embeddings, embedding_dim];
        
        let weight = SparseTensor::from_coo(indices, values_array, shape)?.to_csr()?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
        })
    }

    /// Lookup embeddings for given indices
    /// 指定されたインデックスの埋め込みを検索
    pub fn forward(&self, input: &Array1<usize>) -> RusTorchResult<Array2<T>> {
        let batch_size = input.len();
        let mut output = Array2::zeros((batch_size, self.embedding_dim));

        for (b, &idx) in input.iter().enumerate() {
            if idx >= self.num_embeddings {
                return Err(RusTorchError::InvalidParameters {
                    operation: "sparse_embedding_lookup".to_string(),
                    message: format!("Index {} out of bounds for vocabulary size {}", idx, self.num_embeddings),
                });
            }

            // Handle padding index
            if Some(idx) == self.padding_idx {
                continue; // Leave as zeros
            }

            // Extract sparse row for the embedding
            if self.weight.format == SparseFormat::CSR {
                let row_ptr = &self.weight.indices[0];
                let col_indices = &self.weight.indices[1];
                
                let start = row_ptr[idx];
                let end = row_ptr[idx + 1];
                
                for i in start..end {
                    let col = col_indices[i];
                    output[[b, col]] = self.weight.values[i];
                }
            }
        }

        Ok(output)
    }
}

/// Adaptive sparse attention for transformers
/// Transformer用適応スパースアテンション
pub struct SparseAttention<T: Float> {
    /// Attention matrix sparsity pattern
    /// アテンション行列スパースパターン
    pub attention_mask: Option<SparseTensor<T>>,
    /// Embedding dimension
    /// 埋め込み次元
    pub embed_dim: usize,
    /// Number of attention heads
    /// アテンションヘッド数
    pub num_heads: usize,
    /// Head dimension
    /// ヘッド次元
    pub head_dim: usize,
    /// Dropout probability
    /// ドロップアウト確率
    pub dropout: f32,
}

impl<T: Float + Zero + One + Copy + std::ops::AddAssign + PartialOrd + Send + Sync + ndarray::ScalarOperand + FromPrimitive> SparseAttention<T> {
    /// Create sparse attention module
    /// スパースアテンションモジュールを作成
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        let head_dim = embed_dim / num_heads;
        
        Self {
            attention_mask: None,
            embed_dim,
            num_heads,
            head_dim,
            dropout,
        }
    }

    /// Set sparse attention pattern
    /// スパースアテンションパターンを設定
    pub fn set_attention_mask(&mut self, mask: SparseTensor<T>) -> RusTorchResult<()> {
        if mask.shape.len() != 2 {
            return Err(RusTorchError::InvalidParameters {
                operation: "sparse_attention_mask".to_string(),
                message: "Attention mask must be 2D".to_string(),
            });
        }
        
        self.attention_mask = Some(mask);
        Ok(())
    }

    /// Compute sparse attention (simplified implementation)
    /// スパースアテンションを計算（簡略化実装）
    pub fn forward(
        &self,
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
    ) -> RusTorchResult<Array2<T>> {
        let (seq_len, embed_dim) = query.dim();
        
        if embed_dim != self.embed_dim {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self.embed_dim],
                actual: vec![embed_dim],
            });
        }

        // Simplified attention computation
        // QK^T attention scores
        let scores = query.dot(&key.t());
        
        // Apply sparse mask if available
        let masked_scores = if let Some(ref mask) = self.attention_mask {
            let dense_mask = mask.to_dense()?;
            if dense_mask.shape() != scores.shape() {
                return Err(RusTorchError::ShapeMismatch {
                    expected: scores.shape().to_vec(),
                    actual: dense_mask.shape().to_vec(),
                });
            }
            
            // Apply mask (multiply by mask, add large negative for masked positions)
            scores.clone() // Placeholder - would implement proper masking
        } else {
            scores
        };

        // Apply softmax (simplified - would implement proper softmax)
        let attention_weights = masked_scores; // Placeholder
        
        // Apply attention to values
        let output = attention_weights.dot(value);

        Ok(output)
    }

    /// Calculate attention sparsity statistics
    /// アテンションスパース統計を計算
    pub fn attention_stats(&self) -> Option<(f64, usize)> {
        self.attention_mask.as_ref().map(|mask| {
            (mask.sparsity(), mask.nnz)
        })
    }
}

/// Sparse transformer block with sparse attention and sparse FFN
/// スパースアテンションとスパースFFN付きスパースTransformerブロック
pub struct SparseTransformerBlock<T: Float> {
    /// Sparse self-attention layer
    /// スパースセルフアテンション層
    pub attention: SparseAttention<T>,
    /// Sparse feedforward network
    /// スパースフィードフォワードネットワーク
    pub ffn: Vec<SparseLinear<T>>,
    /// Layer normalization (kept dense)
    /// レイヤー正規化（密で保持）
    pub norm1: Option<Array1<T>>, // Simplified layer norm weights
    pub norm2: Option<Array1<T>>,
}

impl<T: Float + Zero + One + Copy + std::ops::AddAssign + PartialOrd + Send + Sync + ndarray::ScalarOperand + FromPrimitive> SparseTransformerBlock<T> {
    /// Create sparse transformer block
    /// スパースTransformerブロックを作成
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        attention_sparsity: f32,
        ffn_sparsity: f32,
    ) -> RusTorchResult<Self> {
        let attention = SparseAttention::new(embed_dim, num_heads, 0.1);
        
        // Create sparse FFN layers
        let ffn_config = PruningConfig {
            target_sparsity: ffn_sparsity,
            strategy: PruningStrategy::Magnitude,
            structured: false,
            schedule: None,
        };

        // Initialize random weights for FFN
        let ffn1_weight = Array2::from_shape_fn((ffn_dim, embed_dim), |_| {
            T::from(rand::random::<f32>() * 0.02 - 0.01).unwrap()
        });
        let ffn2_weight = Array2::from_shape_fn((embed_dim, ffn_dim), |_| {
            T::from(rand::random::<f32>() * 0.02 - 0.01).unwrap()
        });

        let ffn1 = SparseLinear::with_pruning(ffn1_weight, None, ffn_config.clone())?;
        let ffn2 = SparseLinear::with_pruning(ffn2_weight, None, ffn_config)?;

        Ok(Self {
            attention,
            ffn: vec![ffn1, ffn2],
            norm1: Some(Array1::ones(embed_dim)),
            norm2: Some(Array1::ones(embed_dim)),
        })
    }

    /// Forward pass through sparse transformer block
    /// スパースTransformerブロックのフォワードパス
    pub fn forward(&self, input: &Array2<T>) -> RusTorchResult<Array2<T>> {
        // Self-attention with residual connection
        let attn_output = self.attention.forward(input, input, input)?;
        let attn_residual = input + &attn_output;

        // FFN with residual connection (placeholder implementation)
        // This would implement proper FFN forward pass with sparse layers
        Ok(attn_residual)
    }

    /// Get comprehensive sparsity statistics
    /// 包括的スパース統計を取得
    pub fn sparsity_report(&self) -> HashMap<String, f64> {
        let mut report = HashMap::new();
        
        if let Some((attn_sparsity, _)) = self.attention.attention_stats() {
            report.insert("attention".to_string(), attn_sparsity);
        }
        
        for (i, layer) in self.ffn.iter().enumerate() {
            report.insert(format!("ffn_{}", i), layer.sparsity());
        }
        
        // Calculate overall block sparsity
        let total_sparsity = report.values().sum::<f64>() / report.len() as f64;
        report.insert("overall".to_string(), total_sparsity);
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sparse_linear_creation() {
        let weight = Array2::from_shape_vec((3, 4), vec![
            1.0f32, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 4.0,
            5.0, 0.0, 0.0, 6.0,
        ]).unwrap();
        let bias = Some(Array1::zeros(3));
        
        let sparse_linear = SparseLinear::from_dense(weight, bias, 0.5).unwrap();
        assert_eq!(sparse_linear.in_features, 4);
        assert_eq!(sparse_linear.out_features, 3);
        assert!(sparse_linear.sparsity() > 0.4);
    }

    #[test]
    fn test_sparse_linear_forward() {
        let weight = Array2::from_shape_vec((2, 3), vec![
            1.0f32, 0.0, 2.0,
            0.0, 3.0, 0.0,
        ]).unwrap();
        
        let sparse_linear = SparseLinear::from_dense(weight, None, 0.1).unwrap();
        
        let input_data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        let input_var = Variable::new(Tensor::from_ndarray(input_data), false);
        
        let output = sparse_linear.forward(&input_var).unwrap();
        let output_tensor = output.data();
        let output_guard = output_tensor.read().unwrap();
        
        assert_eq!(output_guard.shape(), &[2]);
    }

    #[test]
    fn test_sparse_embedding() {
        let sparse_embedding = SparseEmbedding::<f32>::new(10, 4, 0.7, Some(0)).unwrap();
        assert_eq!(sparse_embedding.num_embeddings, 10);
        assert_eq!(sparse_embedding.embedding_dim, 4);
        assert!(sparse_embedding.weight.sparsity() > 0.6);
        
        let indices = Array1::from_vec(vec![1, 3, 5]);
        let embeddings = sparse_embedding.forward(&indices).unwrap();
        assert_eq!(embeddings.shape(), [3, 4]);
    }

    #[test]
    fn test_sparse_conv2d_creation() {
        let weight = ArrayD::from_shape_vec(
            vec![2, 3, 3, 3], 
            (0..54).map(|x| if x % 3 == 0 { x as f32 } else { 0.0 }).collect()
        ).unwrap();
        
        let sparse_conv = SparseConv2d::from_dense(
            weight, 
            None, 
            (1, 1), 
            (0, 0), 
            0.5
        ).unwrap();
        
        assert_eq!(sparse_conv.in_channels, 3);
        assert_eq!(sparse_conv.out_channels, 2);
        assert!(sparse_conv.weight.sparsity() > 0.6);
    }

    #[test]
    fn test_sparse_transformer_block() {
        let block = SparseTransformerBlock::<f32>::new(16, 4, 32, 0.8, 0.7).unwrap(); // Much smaller dimensions
        let sparsity_report = block.sparsity_report();
        
        assert!(sparsity_report.contains_key("overall"));
        assert!(sparsity_report["overall"] > 0.5); // Lower threshold for smaller dimensions
    }
}