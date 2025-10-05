use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use rustorch::prelude::Tensor;

use super::{ModelFormat, ModelMetadata};
use super::formats::{GGUFLoader, TensorLoader};
use crate::tokenizer::{Tokenizer, TokenizerWrapper};

pub struct ModelLoader {
    metadata: ModelMetadata,
    gguf_loader: Option<GGUFLoader>,
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

    /// Create a dummy model for testing
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
            gguf_loader: None,
            weights: HashMap::new(),
            tokenizer,
        })
    }

    fn load_gguf(path: &Path) -> Result<Self> {
        tracing::info!("Loading GGUF model from: {}", path.display());

        // Load GGUF file
        let mut loader = GGUFLoader::new(path)?;

        // Get model metadata from GGUF
        let metadata_map = loader.metadata();

        // Extract model parameters from metadata
        let vocab_size = metadata_map
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;

        let hidden_size = metadata_map
            .get("llama.embedding_length")
            .or_else(|| metadata_map.get("gpt2.embedding_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let num_layers = metadata_map
            .get("llama.block_count")
            .or_else(|| metadata_map.get("gpt2.block_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(22) as usize;

        let num_heads = metadata_map
            .get("llama.attention.head_count")
            .or_else(|| metadata_map.get("gpt2.attention.head_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let context_length = metadata_map
            .get("llama.context_length")
            .or_else(|| metadata_map.get("gpt2.context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        tracing::info!(
            "GGUF model parameters: vocab={}, hidden={}, layers={}, heads={}, context={}",
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length
        );

        let tensor_names = loader.tensor_names();
        tracing::info!("Found {} tensors in GGUF file", tensor_names.len());

        // Load all tensors into memory
        let mut weights = HashMap::new();
        let tensor_loader = TensorLoader::new();

        for tensor_name in &tensor_names {
            tracing::debug!("Loading tensor: {}", tensor_name);

            // Load tensor data from GGUF
            let tensor_bytes = loader.load_tensor_data(tensor_name)?;

            // Get tensor info for shape and type
            let tensor_info = loader.tensor_info(tensor_name)?;

            // Convert to RusTorch Tensor
            let tensor = tensor_loader.load_tensor(
                &tensor_bytes,
                &tensor_info.dims,
                tensor_info.ggml_type,
            )?;

            weights.insert(tensor_name.clone(), tensor);
        }

        tracing::info!("Successfully loaded {} tensors", weights.len());

        let metadata = ModelMetadata {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::GGUF,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length,
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
            gguf_loader: Some(loader),
            weights,
            tokenizer,
        })
    }

    fn load_safetensors(path: &Path) -> Result<Self> {
        use super::formats::SafetensorsLoader;

        tracing::info!("Loading Safetensors model from: {}", path.display());

        // Load safetensors file
        let loader = SafetensorsLoader::from_file(path)?;
        let tensor_names = loader.tensor_names();

        tracing::info!("Found {} tensors in Safetensors file", tensor_names.len());

        // Extract basic model info from tensor shapes
        // This is a simplified approach - real models would have config.json
        let vocab_size = 32000; // Default, should come from config
        let hidden_size = 512; // Default, should come from config

        let metadata = ModelMetadata {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::Safetensors,
            vocab_size,
            hidden_size,
            num_layers: 6,
            num_heads: 8,
            context_length: 2048,
        };

        let tokenizer = Box::new(TokenizerWrapper::dummy()?);

        Ok(Self {
            metadata,
            gguf_loader: None,
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
            gguf_loader: None,
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
            gguf_loader: None,
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
            gguf_loader: None,
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
