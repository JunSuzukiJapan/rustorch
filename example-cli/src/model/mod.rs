pub mod formats;
pub mod loader;
pub mod inference;
pub mod transformer;

pub use loader::ModelLoader;
pub use inference::InferenceEngine;
pub use transformer::{TransformerConfig, TransformerModel, KVCache};

use anyhow::Result;
use std::path::Path;

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub format: ModelFormat,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub context_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Dummy,
    GGUF,
    Safetensors,
    ONNX,
    MLX,
}

impl ModelFormat {
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("No file extension found"))?;

        match extension.to_lowercase().as_str() {
            "gguf" => Ok(ModelFormat::GGUF),
            "safetensors" => Ok(ModelFormat::Safetensors),
            "onnx" => Ok(ModelFormat::ONNX),
            "mlx" => Ok(ModelFormat::MLX),
            _ => Err(anyhow::anyhow!("Unsupported model format: {}", extension)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFormat::Dummy => "dummy",
            ModelFormat::GGUF => "gguf",
            ModelFormat::Safetensors => "safetensors",
            ModelFormat::ONNX => "onnx",
            ModelFormat::MLX => "mlx",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_format_from_path() {
        let path = PathBuf::from("model.gguf");
        assert_eq!(ModelFormat::from_path(&path).unwrap(), ModelFormat::GGUF);

        let path = PathBuf::from("model.safetensors");
        assert_eq!(ModelFormat::from_path(&path).unwrap(), ModelFormat::Safetensors);

        let path = PathBuf::from("model.onnx");
        assert_eq!(ModelFormat::from_path(&path).unwrap(), ModelFormat::ONNX);
    }

    #[test]
    fn test_model_format_as_str() {
        assert_eq!(ModelFormat::GGUF.as_str(), "gguf");
        assert_eq!(ModelFormat::Safetensors.as_str(), "safetensors");
        assert_eq!(ModelFormat::Dummy.as_str(), "dummy");
    }
}
