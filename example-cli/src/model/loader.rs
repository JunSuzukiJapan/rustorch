use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use rustorch::prelude::Tensor;

use super::format_loader::FormatLoader;
use super::loaders::{GGUFFormatLoader, MLXFormatLoader, SafetensorsFormatLoader};
use super::{ModelFormat, ModelMetadata};
use crate::tokenizer::{Tokenizer, TokenizerWrapper};

pub struct ModelLoader {
    metadata: ModelMetadata,
    weights: HashMap<String, Tensor<f64>>,
    tokenizer: Box<dyn Tokenizer>,
}

impl ModelLoader {
    /// Load a model from file using the refactored loader architecture
    /// リファクタリングされたローダーアーキテクチャを使用してモデルを読み込み
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            anyhow::bail!("Model file not found: {}", path.display());
        }

        tracing::info!("Loading model from: {}", path.display());

        // Load metadata using appropriate format loader
        let metadata = Self::load_metadata_with_format_detection(path)?;

        tracing::info!("Detected format: {}", metadata.format.as_str());

        // Try to find tokenizer
        let tokenizer = Self::load_tokenizer(path, &metadata.format)?;

        Ok(Self {
            metadata,
            weights: HashMap::new(),
            tokenizer,
        })
    }

    /// Create a dummy model for testing only
    /// テスト専用のダミーモデルを作成
    #[cfg(test)]
    pub fn dummy() -> Self {
        Self::load_dummy().expect("Failed to create dummy model")
    }

    /// Load metadata with automatic format detection
    /// 自動フォーマット検出でメタデータを読み込み
    fn load_metadata_with_format_detection(path: &Path) -> Result<ModelMetadata> {
        // Try each format loader in order of preference
        if GGUFFormatLoader::can_load(path) {
            return GGUFFormatLoader::load_metadata(path);
        }

        if SafetensorsFormatLoader::can_load(path) {
            return SafetensorsFormatLoader::load_metadata(path);
        }

        if MLXFormatLoader::can_load(path) {
            return MLXFormatLoader::load_metadata(path);
        }

        // Fall back to format detection by extension
        let format = ModelFormat::from_path(path)?;
        match format {
            ModelFormat::PyTorch => Self::load_pytorch_metadata(path),
            ModelFormat::ONNX => Self::load_onnx_metadata(path),
            _ => anyhow::bail!("Unsupported model format for path: {}", path.display()),
        }
    }

    /// Load PyTorch metadata (legacy support)
    /// PyTorchメタデータを読み込み（レガシーサポート）
    fn load_pytorch_metadata(path: &Path) -> Result<ModelMetadata> {
        use super::formats::PyTorchLoader;

        tracing::info!("Loading PyTorch model from: {}", path.display());
        let (_state_dict, metadata) = PyTorchLoader::load(path)?;
        tracing::info!("Successfully loaded PyTorch model with {} tensors", _state_dict.len());

        Ok(metadata)
    }

    /// Load ONNX metadata (legacy support)
    /// ONNXメタデータを読み込み（レガシーサポート）
    fn load_onnx_metadata(path: &Path) -> Result<ModelMetadata> {
        use super::formats::ONNXLoader;

        tracing::info!("Loading ONNX model from: {}", path.display());
        let loader = ONNXLoader::from_file(path)?;
        let meta = loader.metadata();
        tracing::info!("ONNX model metadata: {:?}", meta);

        Ok(ModelMetadata {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::ONNX,
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            context_length: 2048,
        })
    }

    /// Load tokenizer for the model
    /// モデル用のトークナイザーを読み込み
    fn load_tokenizer(path: &Path, format: &ModelFormat) -> Result<Box<dyn Tokenizer>> {
        // Try format-specific tokenizer path
        let tokenizer_path = match format {
            ModelFormat::GGUF => GGUFFormatLoader::default_tokenizer_path(path),
            ModelFormat::Safetensors => SafetensorsFormatLoader::default_tokenizer_path(path),
            ModelFormat::MLX => MLXFormatLoader::default_tokenizer_path(path),
            _ => None,
        };

        if let Some(tokenizer_path) = tokenizer_path {
            tracing::info!("Loading tokenizer from: {}", tokenizer_path.display());
            return Ok(Box::new(TokenizerWrapper::from_file(&tokenizer_path)?));
        }

        tracing::warn!("Tokenizer file not found, using dummy tokenizer");
        Ok(Box::new(TokenizerWrapper::dummy()?))
    }

    fn load_dummy() -> Result<Self> {
        tracing::info!("Creating dummy model for testing");

        let metadata = ModelMetadata {
            name: "dummy-model".to_string(),
            format: ModelFormat::Dummy,
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            context_length: 2048,
        };

        let tokenizer = Box::new(TokenizerWrapper::dummy()?);

        Ok(Self {
            metadata,
            weights: HashMap::new(),
            tokenizer,
        })
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get model weights
    pub fn weights(&self) -> &HashMap<String, Tensor<f64>> {
        &self.weights
    }

    /// Get a specific weight by name
    pub fn weight(&self, name: &str) -> Option<&Tensor<f64>> {
        self.weights.get(name)
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_model_creation() {
        let loader = ModelLoader::dummy();
        assert_eq!(loader.metadata().name, "dummy-model");
        assert_eq!(loader.metadata().format, ModelFormat::Dummy);
        assert_eq!(loader.metadata().vocab_size, 32000);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = ModelLoader::from_file("nonexistent.gguf");
        assert!(result.is_err());
    }
}
