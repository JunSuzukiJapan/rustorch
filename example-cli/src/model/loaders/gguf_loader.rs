// GGUF format loader with common interface implementation
use anyhow::Result;
use std::path::Path;

use crate::model::format_loader::{metadata_utils, FormatLoader};
use crate::model::{ModelFormat, ModelMetadata};

pub struct GGUFFormatLoader;

impl FormatLoader for GGUFFormatLoader {
    fn load_metadata(path: &Path) -> Result<ModelMetadata> {
        tracing::info!("Loading GGUF metadata from: {}", path.display());

        // Use RusTorch's GGUF loader
        let gguf_loader = rustorch::formats::gguf::GGUFLoader::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF file: {}", e))?;

        // Extract model parameters from GGUF metadata
        let model_params = gguf_loader
            .get_model_params()
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

        Ok(ModelMetadata {
            name: metadata_utils::extract_model_name(path),
            format: ModelFormat::GGUF,
            vocab_size: model_params.vocab_size as usize,
            hidden_size: model_params.hidden_size as usize,
            num_layers: model_params.num_layers as usize,
            num_heads: model_params.num_heads as usize,
            context_length: model_params.context_length as usize,
        })
    }

    fn can_load(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_load() {
        assert!(GGUFFormatLoader::can_load(Path::new("model.gguf")));
        assert!(GGUFFormatLoader::can_load(Path::new("model.GGUF")));
        assert!(!GGUFFormatLoader::can_load(Path::new("model.safetensors")));
        assert!(!GGUFFormatLoader::can_load(Path::new("model.pt")));
    }
}
