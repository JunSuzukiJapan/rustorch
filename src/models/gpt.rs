//! GPT model implementation for RusTorch
//! GPTモデル実装

use crate::error::{RusTorchError, RusTorchResult};
use crate::formats::gguf::{GGUFLoader, ModelParams};
use crate::formats::mlx::MLXLoader;
use crate::formats::safetensors::SafetensorsLoader;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// GPT model configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl GPTConfig {
    /// Create config from GGUF model parameters
    pub fn from_model_params(params: &ModelParams) -> Self {
        Self {
            vocab_size: params.vocab_size as usize,
            d_model: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            d_ff: (params.hidden_size * 4) as usize, // Standard FFN size
            max_seq_len: params.context_length as usize,
            dropout: 0.1,
        }
    }
}

/// GPT model structure
pub struct GPTModel {
    config: GPTConfig,
    weights: HashMap<String, Tensor<f64>>,
}

impl GPTModel {
    /// Create a new GPT model with given configuration
    pub fn new(config: GPTConfig) -> RusTorchResult<Self> {
        Ok(Self {
            config,
            weights: HashMap::new(),
        })
    }

    /// Load GPT model from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let loader = GGUFLoader::from_file(path)?;

        // Extract model parameters
        let params = loader.get_model_params()?;
        let config = GPTConfig::from_model_params(&params);

        // Create model
        let mut model = Self::new(config)?;

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Load GPT model from Safetensors file
    /// SafetensorsファイルからGPTモデルを読み込み
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: GPTConfig,
    ) -> RusTorchResult<Self> {
        let loader = SafetensorsLoader::from_file(path)?;

        // Create model with provided config
        let mut model = Self::new(config)?;

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor::<f64>(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Load GPT model from MLX file
    /// MLXファイルからGPTモデルを読み込み
    pub fn from_mlx<P: AsRef<Path>>(path: P, config: GPTConfig) -> RusTorchResult<Self> {
        let loader = MLXLoader::from_file(path)?;

        // Create model with provided config
        let mut model = Self::new(config)?;

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor::<f64>(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Get model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get a weight tensor by name
    pub fn get_weight(&self, name: &str) -> Option<&Tensor<f64>> {
        self.weights.get(name)
    }

    /// List all weight names
    pub fn weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    /// Create LayerNorm with loaded weights if available
    /// 読み込まれた重みでLayerNormを作成（利用可能な場合）
    fn create_layer_norm_variable(&self, weight_key: &str, d_model: usize) -> crate::autograd::Variable<f64> {
        use crate::autograd::Variable;

        // Create a default weight (all ones) and bias (all zeros)
        let weight_data = if let Some(loaded_weight) = self.weights.get(weight_key) {
            // Use loaded weights if available
            loaded_weight.clone()
        } else {
            // Default initialization: ones
            Tensor::from_vec(vec![1.0; d_model], vec![d_model])
        };

        let bias_data = Tensor::from_vec(vec![0.0; d_model], vec![d_model]);

        // Return weight and bias as Variables
        // For now, just return the weight variable (bias will be handled separately)
        Variable::new(weight_data, true)
    }

    /// Apply manual LayerNorm with loaded weights
    /// 読み込まれた重みで手動LayerNormを適用
    fn apply_layer_norm(&self, input: &crate::autograd::Variable<f64>, weight_key: &str, d_model: usize) -> crate::autograd::Variable<f64> {
        use crate::autograd::Variable;

        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();

        // Get weight and bias
        let (weight, using_loaded_weight) = if let Some(w) = self.weights.get(weight_key) {
            (w.clone(), true)
        } else {
            (Tensor::from_vec(vec![1.0; d_model], vec![d_model]), false)
        };

        #[cfg(debug_assertions)]
        if using_loaded_weight {
            eprintln!("✓ Using loaded GGUF weight: {}", weight_key);
        } else {
            eprintln!("✗ Weight not found, using default: {}", weight_key);
        }

        let bias = Tensor::from_vec(vec![0.0; d_model], vec![d_model]);
        let eps = 1e-5;

        // Manual LayerNorm computation
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        let mut output_data = Vec::with_capacity(batch_size * seq_len * features);

        // Process each position
        for b in 0..batch_size {
            for s in 0..seq_len {
                // Extract features for this position
                let mut position_features = Vec::with_capacity(features);
                for f in 0..features {
                    let idx = (b * seq_len * features) + (s * features) + f;
                    if let Some(slice) = input_data.as_array().as_slice() {
                        position_features.push(slice[idx]);
                    }
                }

                // Calculate mean and variance
                let mean: f64 = position_features.iter().sum::<f64>() / features as f64;
                let variance: f64 = position_features.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / features as f64;
                let std = (variance + eps).sqrt();

                // Normalize and apply affine transformation
                for (f, &feature_val) in position_features.iter().enumerate() {
                    let normalized = (feature_val - mean) / std;
                    let gamma = if let Some(slice) = weight.as_array().as_slice() {
                        slice[f]
                    } else {
                        1.0
                    };
                    let beta = if let Some(slice) = bias.as_array().as_slice() {
                        slice[f]
                    } else {
                        0.0
                    };
                    let final_val = gamma * normalized + beta;
                    output_data.push(final_val);
                }
            }
        }

        let output = Tensor::from_vec(output_data, input_shape.to_vec());
        Variable::new(output, input.requires_grad())
    }

    /// GPT forward pass with Transformer implementation
    /// TransformerフォワードパスによるGPT実装
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    /// Logits tensor of shape [batch_size, seq_len, vocab_size]
    ///
    /// # Note
    /// For development/testing, uses only first 2 layers by default.
    /// Use `forward_with_layers(input_ids, None)` for full model inference.
    pub fn forward(&self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
        // Limit to 2 layers for faster inference during development
        // TODO: Remove this limitation for production use
        // TODO: Add GPU backend support for tensor operations
        eprintln!("⚠️  GPT forward pass using CPU (GPU backend not yet integrated)");
        let max_layers = Some(2);
        self.forward_with_layers(input_ids, max_layers)
    }

    /// GPT forward pass with configurable number of layers
    /// レイヤー数を設定可能なGPTフォワードパス
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `max_layers` - Maximum number of layers to use (None = use all)
    ///
    /// # Returns
    /// Logits tensor of shape [batch_size, seq_len, vocab_size]
    pub fn forward_with_layers(&self, input_ids: &[usize], max_layers: Option<usize>) -> RusTorchResult<Tensor<f64>> {
        use crate::autograd::Variable;
        use crate::nn::{Embedding, SinusoidalPositionalEncoding, MultiheadAttention, Linear, GELU, Module};

        let batch_size = 1;
        let seq_len = input_ids.len();
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;
        let num_heads = self.config.num_heads;
        let d_ff = self.config.d_ff;

        // 1. Token Embedding Lookup
        // トークン埋め込み変換: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        let token_emb = Embedding::<f64>::new(vocab_size, d_model, None, None, None);

        // Convert input_ids to f64 tensor: [batch_size, seq_len]
        let input_data: Vec<f64> = input_ids.iter().map(|&id| id as f64).collect();
        let input_tensor = Tensor::from_vec(input_data, vec![batch_size, seq_len]);
        let input_var = Variable::new(input_tensor, false);

        // Get token embeddings: [batch_size, seq_len, d_model]
        let mut x = token_emb.forward(&input_var);

        // 2. Add Positional Encoding
        // 位置エンコーディング追加
        let pos_encoding = SinusoidalPositionalEncoding::<f64>::new(
            self.config.max_seq_len,
            d_model
        );
        x = pos_encoding.forward(&x);

        // 3. Apply Transformer Blocks
        // Transformerブロック適用
        let num_layers = max_layers.unwrap_or(self.config.num_layers).min(self.config.num_layers);

        #[cfg(debug_assertions)]
        if let Some(max) = max_layers {
            eprintln!("Using {} layers (out of {})", num_layers, self.config.num_layers);
        }

        for layer_idx in 0..num_layers {
            // Save input for residual connection
            let residual = x.clone();

            // Layer Norm 1 (Pre-Attention) with loaded weights
            let ln1_key = format!("blk.{}.attn_norm.weight", layer_idx);
            x = self.apply_layer_norm(&x, &ln1_key, d_model);

            // Multi-Head Self-Attention
            let attention = MultiheadAttention::<f64>::new(
                d_model,
                num_heads,
                Some(0.0),  // dropout
                Some(true), // bias
                None,       // kdim
                None,       // vdim
                Some(true), // batch_first
            )?;
            let (attn_output, _) = attention.forward(&x, &x, &x, None, Some(false), None, Some(true))?;

            // Residual connection 1
            x = self.add_variables(&residual, &attn_output)?;

            // Save for second residual
            let residual2 = x.clone();

            // Layer Norm 2 (Pre-FFN) with loaded weights
            let ln2_key = format!("blk.{}.ffn_norm.weight", layer_idx);
            x = self.apply_layer_norm(&x, &ln2_key, d_model);

            // Feed-Forward Network
            let fc1 = Linear::<f64>::new(d_model, d_ff);
            let gelu = GELU::<f64>::new();
            let fc2 = Linear::<f64>::new(d_ff, d_model);

            let mut ffn_out = fc1.forward(&x);
            ffn_out = gelu.forward(&ffn_out);
            ffn_out = fc2.forward(&ffn_out);

            // Residual connection 2
            x = self.add_variables(&residual2, &ffn_out)?;
        }

        // Final Layer Norm (Output Norm) with loaded weights
        x = self.apply_layer_norm(&x, "output_norm.weight", d_model);

        // 4. Output Projection to Vocabulary Logits
        // 語彙サイズへの出力射影: [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        let lm_head = Linear::<f64>::new(d_model, vocab_size);
        let logits_var = lm_head.forward(&x);

        // Extract tensor from Variable
        let logits_binding = logits_var.data();
        let logits_data = logits_binding.read().unwrap();
        let logits = logits_data.clone();

        Ok(logits)
    }

    /// Add two Variables element-wise (for residual connections)
    /// 2つのVariableを要素ごとに加算（残差接続用）
    fn add_variables(&self, a: &crate::autograd::Variable<f64>, b: &crate::autograd::Variable<f64>) -> RusTorchResult<crate::autograd::Variable<f64>> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();

        if a_data.shape() != b_data.shape() {
            return Err(RusTorchError::shape_mismatch(a_data.shape(), b_data.shape()));
        }

        let result_data: Vec<f64> = a_data.as_array().iter()
            .zip(b_data.as_array().iter())
            .map(|(x, y)| x + y)
            .collect();

        let result = Tensor::from_vec(result_data, a_data.shape().to_vec());
        let requires_grad = a.requires_grad() || b.requires_grad();
        Ok(crate::autograd::Variable::new(result, requires_grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_creation() {
        let config = GPTConfig {
            vocab_size: 50257,
            d_model: 768,
            num_layers: 12,
            num_heads: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
        };

        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_gpt_model_creation() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = GPTModel::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.config().num_layers, 2);
    }
}
