//! GPT model implementation for hybrid_f32 (native f32 precision)
//! hybrid_f32用GPTモデル実装（ネイティブf32精度）

use crate::hybrid_f32::error::{F32Error, F32Result};
use crate::hybrid_f32::tensor::F32Tensor;
use crate::formats::gguf::{GGUFLoader, ModelParams};
use std::collections::HashMap;
use std::path::Path;

/// GPT model configuration
/// GPTモデル設定
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f32,
}

impl GPTConfig {
    /// Create config from GGUF model parameters
    /// GGUFモデルパラメータから設定を作成
    pub fn from_model_params(params: &ModelParams) -> Self {
        Self {
            vocab_size: params.vocab_size as usize,
            d_model: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            d_ff: (params.hidden_size * 4) as usize,
            max_seq_len: params.context_length as usize,
            dropout: 0.1,
        }
    }
}

/// Device type for GPU acceleration
/// GPU加速用デバイスタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU computation
    Cpu,
    /// Metal GPU (macOS)
    Metal,
    /// CoreML Neural Engine (macOS)
    CoreML,
    /// Hybrid Metal + CoreML
    Hybrid,
}

/// GPT model with native f32 precision for GPU acceleration
/// GPU加速用ネイティブf32精度GPTモデル
pub struct F32GPTModel {
    config: GPTConfig,
    weights: HashMap<String, F32Tensor>,
    device_type: DeviceType,
}

impl F32GPTModel {
    /// Create a new GPT model with CPU backend
    /// CPUバックエンドで新しいGPTモデルを作成
    pub fn new(config: GPTConfig) -> F32Result<Self> {
        Self::with_device(config, DeviceType::Cpu)
    }

    /// Create a new GPT model with specified device
    /// 指定デバイスで新しいGPTモデルを作成
    pub fn with_device(config: GPTConfig, device_type: DeviceType) -> F32Result<Self> {
        eprintln!("🚀 Creating F32GPTModel with {:?} device", device_type);
        eprintln!("   Precision: native f32 (optimized for GPU)");

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type,
        })
    }

    /// Load GPT model from GGUF file with Metal GPU support
    /// Metal GPUサポート付きでGGUFファイルからGPTモデルを読み込み
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> F32Result<Self> {
        Self::from_gguf_with_device(path, DeviceType::Metal)
    }

    /// Load GPT model from GGUF file with specified device
    /// 指定デバイスでGGUFファイルからGPTモデルを読み込み
    pub fn from_gguf_with_device<P: AsRef<Path>>(
        path: P,
        device_type: DeviceType,
    ) -> F32Result<Self> {
        let loader = GGUFLoader::from_file(path)
            .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

        // Extract model parameters
        let params = loader
            .get_model_params()
            .map_err(|e| F32Error::device_error(format!("Failed to get model params: {}", e)))?;

        let config = GPTConfig::from_model_params(&params);

        // Create model with device
        let mut model = Self::with_device(config, device_type)?;

        eprintln!("📊 Loading GPT model weights as f32");
        eprintln!("   Device: {:?}", device_type);
        eprintln!("   Vocab size: {}", model.config.vocab_size);
        eprintln!("   Layers: {}", model.config.num_layers);
        eprintln!("   d_model: {}", model.config.d_model);

        // Load weights as f32
        let tensor_names = loader.tensor_names();
        let mut loaded_count = 0;

        for name in tensor_names.iter() {
            // Load tensor as f64, then convert to f32
            match loader.load_tensor(name) {
                Ok(tensor_f64) => {
                    // Convert f64 tensor to f32
                    if let Some(data_slice) = tensor_f64.as_array().as_slice() {
                        let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
                        let shape = tensor_f64.shape().to_vec();

                        // Create F32Tensor from converted data
                        let f32_tensor = F32Tensor::from_vec(f32_data, &shape)
                            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create F32Tensor: {}", e)))?;

                        model.weights.insert(name.to_string(), f32_tensor);
                        loaded_count += 1;
                    }
                }
                Err(_) => {
                    // Skip tensors that fail to load
                    continue;
                }
            }
        }

        eprintln!("✅ Loaded {} weights as f32", loaded_count);

        Ok(model)
    }

    /// Get model configuration
    /// モデル設定を取得
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get device type
    /// デバイスタイプを取得
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get a weight tensor by name
    /// 名前で重みテンソルを取得
    pub fn get_weight(&self, name: &str) -> Option<&F32Tensor> {
        self.weights.get(name)
    }

    /// List all weight names
    /// すべての重み名をリスト
    pub fn weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    /// Forward pass through the model
    /// モデルの順伝播
    pub fn forward(&self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        eprintln!("🔄 F32GPTModel forward pass");
        eprintln!("   Device: {:?}", self.device_type);
        eprintln!("   Input length: {}", input_ids.len());

        let batch_size = 1;
        let seq_len = input_ids.len();

        // Step 1: Token Embedding
        let mut hidden_states = self.get_embeddings(input_ids)?;
        eprintln!("   ✓ Token embeddings: [{}, {}, {}]", batch_size, seq_len, self.config.d_model);

        // Step 2: Process through transformer layers (simplified - first 2 layers only)
        for layer_idx in 0..self.config.num_layers.min(2) {
            eprintln!("   🔄 Layer {}/{}", layer_idx + 1, self.config.num_layers);

            // Apply LayerNorm (GPU-accelerated on Metal)
            hidden_states = self.apply_layer_norm(hidden_states, layer_idx)?;
        }

        // Step 3: Final layer norm and projection to vocabulary
        eprintln!("   🔄 Final LayerNorm and projection");
        hidden_states = self.apply_final_layer_norm(hidden_states)?;
        let logits = self.project_to_vocab(hidden_states)?;
        eprintln!("   ✓ Logits: [{}, {}, {}]", batch_size, seq_len, self.config.vocab_size);

        Ok(logits)
    }

    /// Get token embeddings
    fn get_embeddings(&self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        let embed_weight = self.weights.get("token_embd.weight")
            .or_else(|| self.weights.get("model.embed_tokens.weight"))
            .ok_or_else(|| F32Error::device_error("Embedding weight not found"))?;

        let seq_len = input_ids.len();
        let d_model = self.config.d_model;
        let mut embedding_data = Vec::with_capacity(seq_len * d_model);

        if let Some(embed_slice) = embed_weight.data.as_slice() {
            for &token_id in input_ids {
                let token_idx = token_id.min(self.config.vocab_size - 1);
                let start = token_idx * d_model;
                let end = start + d_model;
                embedding_data.extend_from_slice(&embed_slice[start..end]);
            }
        } else {
            return Err(F32Error::device_error("Failed to access embedding weight data"));
        }

        F32Tensor::from_vec(embedding_data, &[1, seq_len, d_model])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create embedding tensor: {}", e)))
    }

    /// Apply LayerNorm using GPU acceleration if available
    fn apply_layer_norm(&self, input: F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        let ln_weight_key = format!("blk.{}.attn_norm.weight", layer_idx);
        let ln_bias_key = format!("blk.{}.attn_norm.bias", layer_idx);

        let gamma = self.weights.get(&ln_weight_key)
            .ok_or_else(|| F32Error::device_error(format!("LayerNorm weight not found: {}", ln_weight_key)))?;
        let beta = self.weights.get(&ln_bias_key)
            .ok_or_else(|| F32Error::device_error(format!("LayerNorm bias not found: {}", ln_bias_key)))?;

        self.layer_norm_impl(&input, gamma, beta)
    }

    /// Apply final LayerNorm
    fn apply_final_layer_norm(&self, input: F32Tensor) -> F32Result<F32Tensor> {
        let gamma = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .ok_or_else(|| F32Error::device_error("Final LayerNorm weight not found"))?;
        let beta = self.weights.get("output_norm.bias")
            .or_else(|| self.weights.get("model.norm.bias"))
            .ok_or_else(|| F32Error::device_error("Final LayerNorm bias not found"))?;

        self.layer_norm_impl(&input, gamma, beta)
    }

    /// LayerNorm implementation with Metal GPU acceleration
    fn layer_norm_impl(&self, input: &F32Tensor, gamma: &F32Tensor, beta: &F32Tensor) -> F32Result<F32Tensor> {
        let input_shape = input.data.shape();
        if input_shape.len() != 3 {
            return Err(F32Error::dimension_error(3, input_shape.len()));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        #[cfg(feature = "metal")]
        if self.device_type == DeviceType::Metal || self.device_type == DeviceType::Hybrid {
            return self.metal_layer_norm(input, gamma, beta, batch_size, seq_len, features);
        }

        self.cpu_layer_norm(input, gamma, beta, batch_size, seq_len, features)
    }

    #[cfg(feature = "metal")]
    fn metal_layer_norm(
        &self,
        input: &F32Tensor,
        gamma: &F32Tensor,
        beta: &F32Tensor,
        batch_size: usize,
        seq_len: usize,
        features: usize,
    ) -> F32Result<F32Tensor> {
        use crate::gpu::metal_kernels::metal_layer_norm_f32;

        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;
        let gamma_slice = gamma.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access gamma data"))?;
        let beta_slice = beta.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access beta data"))?;

        let mut output = vec![0.0f32; input_slice.len()];

        metal_layer_norm_f32(
            input_slice,
            &mut output,
            gamma_slice,
            beta_slice,
            batch_size,
            seq_len,
            features,
            1e-5,
        )
        .map_err(|e| F32Error::device_error(format!("Metal LayerNorm failed: {}", e)))?;

        F32Tensor::from_vec(output, &[batch_size, seq_len, features])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create output tensor: {}", e)))
    }

    fn cpu_layer_norm(
        &self,
        input: &F32Tensor,
        gamma: &F32Tensor,
        beta: &F32Tensor,
        batch_size: usize,
        seq_len: usize,
        features: usize,
    ) -> F32Result<F32Tensor> {
        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;
        let gamma_slice = gamma.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access gamma data"))?;
        let beta_slice = beta.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access beta data"))?;

        let mut output = vec![0.0f32; input_slice.len()];
        let eps = 1e-5f32;

        for b in 0..batch_size {
            for s in 0..seq_len {
                let offset = (b * seq_len + s) * features;
                let sum: f32 = input_slice[offset..offset + features].iter().sum();
                let mean = sum / features as f32;
                let var_sum: f32 = input_slice[offset..offset + features]
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum();
                let variance = var_sum / features as f32;
                let std = (variance + eps).sqrt();

                for f in 0..features {
                    let normalized = (input_slice[offset + f] - mean) / std;
                    output[offset + f] = gamma_slice[f] * normalized + beta_slice[f];
                }
            }
        }

        F32Tensor::from_vec(output, &[batch_size, seq_len, features])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create output tensor: {}", e)))
    }

    /// Project hidden states to vocabulary logits
    fn project_to_vocab(&self, hidden_states: F32Tensor) -> F32Result<F32Tensor> {
        let output_weight = self.weights.get("output.weight")
            .or_else(|| self.weights.get("lm_head.weight"))
            .or_else(|| self.weights.get("token_embd.weight"))
            .ok_or_else(|| F32Error::device_error("Output projection weight not found"))?;

        let hidden_shape = hidden_states.data.shape();
        let batch_size = hidden_shape[0];
        let seq_len = hidden_shape[1];
        let d_model = hidden_shape[2];
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; batch_size * seq_len * vocab_size];

        let hidden_slice = hidden_states.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access hidden states"))?;
        let weight_slice = output_weight.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access output weight"))?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                let hidden_offset = (b * seq_len + s) * d_model;
                let logit_offset = (b * seq_len + s) * vocab_size;
                for v in 0..vocab_size {
                    let weight_offset = v * d_model;
                    let mut sum = 0.0f32;
                    for d in 0..d_model {
                        sum += hidden_slice[hidden_offset + d] * weight_slice[weight_offset + d];
                    }
                    logits[logit_offset + v] = sum;
                }
            }
        }

        F32Tensor::from_vec(logits, &[batch_size, seq_len, vocab_size])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create logits tensor: {}", e)))
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
    fn test_f32_gpt_model_creation() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = F32GPTModel::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.device_type(), DeviceType::Cpu);
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_f32_gpt_model_with_metal() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = F32GPTModel::with_device(config, DeviceType::Metal).unwrap();
        assert_eq!(model.device_type(), DeviceType::Metal);
    }
}
