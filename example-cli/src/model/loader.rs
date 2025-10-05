use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use rustorch::prelude::Tensor;

use super::{ModelFormat, ModelMetadata};
use crate::tokenizer::{Tokenizer, TokenizerWrapper};

pub struct ModelLoader {
    metadata: ModelMetadata,
    weights: HashMap<String, Tensor<f64>>,
    tokenizer: Box<dyn Tokenizer>,
}

impl ModelLoader {
    /// Load a model from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            anyhow::bail!("Model file not found: {}", path.display());
        }

        let format = ModelFormat::from_path(path)?;

        tracing::info!("Loading model from: {}", path.display());
        tracing::info!("Detected format: {}", format.as_str());

        match format {
            ModelFormat::GGUF => Self::load_gguf(path),
            ModelFormat::Safetensors => Self::load_safetensors(path),
            ModelFormat::ONNX => Self::load_onnx(path),
            ModelFormat::MLX => Self::load_mlx(path),
            ModelFormat::PyTorch => Self::load_pytorch(path),
            ModelFormat::Dummy => Self::load_dummy(),
        }
    }

    /// Create a dummy model for testing only
    #[cfg(test)]
    pub fn dummy() -> Self {
        Self::load_dummy().expect("Failed to create dummy model")
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

    fn load_gguf(path: &Path) -> Result<Self> {
        tracing::info!("Loading GGUF model from: {}", path.display());

        // Use RusTorch's GGUF loader
        let gguf_loader = rustorch::formats::gguf::GGUFLoader::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF file: {}", e))?;

        // Extract model parameters from GGUF metadata
        let model_params = gguf_loader.get_model_params()
            .map_err(|e| anyhow::anyhow!("Failed to extract model parameters: {}", e))?;

        tracing::info!(
            "GGUF model parameters: vocab={}, hidden={}, layers={}, heads={}, context={}",
            model_params.vocab_size,
            model_params.hidden_size,
            model_params.num_layers,
            model_params.num_heads,
            model_params.context_length
        );

        let tensor_names = gguf_loader.tensor_names();
        tracing::info!("Found {} tensors in GGUF file", tensor_names.len());

        // For now, we'll keep weights empty until we implement tensor loading
        // TODO: Implement tensor data loading from RusTorch GGUF loader
        let weights = HashMap::new();

        tracing::info!("Successfully loaded GGUF metadata");

        let metadata = ModelMetadata {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::GGUF,
            vocab_size: model_params.vocab_size as usize,
            hidden_size: model_params.hidden_size as usize,
            num_layers: model_params.num_layers as usize,
            num_heads: model_params.num_heads as usize,
            context_length: model_params.context_length as usize,
        };

        // Try to load tokenizer from same directory
        let tokenizer_path = path.with_extension("").with_extension("tokenizer.json");
        let tokenizer: Box<dyn Tokenizer> = if tokenizer_path.exists() {
            Box::new(TokenizerWrapper::from_file(&tokenizer_path)?)
        } else {
            tracing::warn!("Tokenizer file not found, using dummy tokenizer");
            Box::new(TokenizerWrapper::dummy()?)
        };

        Ok(Self {
            metadata,
            weights,
            tokenizer,
        })
    }

    fn load_safetensors(path: &Path) -> Result<Self> {
        tracing::info!("Loading Safetensors model from: {}", path.display());

        // Use RusTorch's Safetensors loader
        let loader = rustorch::formats::safetensors::SafetensorsLoader::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load Safetensors file: {}", e))?;

        let tensor_names = loader.tensor_names();
        tracing::info!("Found {} tensors in Safetensors file", tensor_names.len());

        // Try to infer model architecture from tensor names and shapes
        let vocab_size = tensor_names.iter()
            .find(|&name| name.contains("embed") && name.contains("weight"))
            .and_then(|name| loader.tensor_info(name).ok())
            .and_then(|info| info.shape.first().copied())
            .unwrap_or(32000);

        let hidden_size = tensor_names.iter()
            .find(|&name| name.contains("embed") && name.contains("weight"))
            .and_then(|name| loader.tensor_info(name).ok())
            .and_then(|info| info.shape.get(1).copied())
            .unwrap_or(512);

        // Count layers by looking for layer.{N}. patterns
        let num_layers = tensor_names.iter()
            .filter_map(|name| {
                name.split('.').find_map(|part| part.parse::<usize>().ok())
            })
            .max()
            .map(|n| n + 1)
            .unwrap_or(6);

        tracing::info!(
            "Safetensors model parameters: vocab={}, hidden={}, layers={}",
            vocab_size,
            hidden_size,
            num_layers
        );

        let metadata = ModelMetadata {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::Safetensors,
            vocab_size,
            hidden_size,
            num_layers,
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

    fn load_onnx(path: &Path) -> Result<Self> {
        use super::formats::ONNXLoader;

        tracing::info!("Loading ONNX model from: {}", path.display());

        // Load ONNX file metadata
        let loader = ONNXLoader::from_file(path)?;
        let meta = loader.metadata();

        tracing::info!("ONNX model metadata: {:?}", meta);

        // Note: Full ONNX inference requires ONNX Runtime
        // This is just metadata loading for now
        let metadata = ModelMetadata {
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
        };

        let tokenizer = Box::new(TokenizerWrapper::dummy()?);

        Ok(Self {
            metadata,
            weights: HashMap::new(),
            tokenizer,
        })
    }

    fn load_mlx(path: &Path) -> Result<Self> {
        use super::formats::MLXLoader;

        tracing::info!("Loading MLX model from: {}", path.display());

        // Load MLX file
        let (_tensors, metadata) = MLXLoader::load(path)?;

        tracing::info!(
            "Successfully loaded MLX model with {} tensors",
            _tensors.len()
        );

        let tokenizer = Box::new(TokenizerWrapper::dummy()?);

        Ok(Self {
            metadata,
            weights: HashMap::new(),
            tokenizer,
        })
    }

    fn load_pytorch(path: &Path) -> Result<Self> {
        use super::formats::PyTorchLoader;

        tracing::info!("Loading PyTorch model from: {}", path.display());

        // Load PyTorch state_dict
        let (_state_dict, metadata) = PyTorchLoader::load(path)?;

        tracing::info!(
            "Successfully loaded PyTorch model with {} tensors",
            _state_dict.len()
        );

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
