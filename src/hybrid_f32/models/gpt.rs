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

    /// Forward pass (simplified for testing)
    /// フォワードパス（テスト用簡略版）
    pub fn forward(&self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        eprintln!("🔄 F32GPTModel forward pass");
        eprintln!("   Device: {:?}", self.device_type);
        eprintln!("   Input length: {}", input_ids.len());

        // TODO: Implement full forward pass with Metal LayerNorm
        // For now, return a dummy output tensor
        let output_size = input_ids.len() * self.config.vocab_size;
        let dummy_output = vec![0.0f32; output_size];
        let shape = vec![1, input_ids.len(), self.config.vocab_size];

        F32Tensor::from_vec(dummy_output, &shape)
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create output tensor: {}", e)))
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
