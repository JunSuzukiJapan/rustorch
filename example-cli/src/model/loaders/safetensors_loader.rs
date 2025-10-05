// Safetensors format loader with common interface implementation
use anyhow::Result;
use std::path::Path;

use crate::model::format_loader::{metadata_utils, FormatLoader};
use crate::model::{ModelFormat, ModelMetadata};

pub struct SafetensorsFormatLoader;

impl FormatLoader for SafetensorsFormatLoader {
    fn load_metadata(path: &Path) -> Result<ModelMetadata> {
        tracing::info!("Loading Safetensors metadata from: {}", path.display());

        // Use RusTorch's Safetensors loader
        let loader = rustorch::formats::safetensors::SafetensorsLoader::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load Safetensors file: {}", e))?;

        let tensor_names_vec: Vec<String> = loader.tensor_names().iter().map(|&s| s.to_string()).collect();
        tracing::info!("Found {} tensors in Safetensors file", tensor_names_vec.len());

        // Infer model architecture from tensor names and shapes
        let vocab_size = Self::infer_vocab_size(&loader, &tensor_names_vec);
        let hidden_size = Self::infer_hidden_size(&loader, &tensor_names_vec);
        let num_layers = metadata_utils::count_layers(&tensor_names_vec);

        tracing::info!(
            "Safetensors inferred parameters: vocab={}, hidden={}, layers={}",
            vocab_size,
            hidden_size,
            num_layers
        );

        Ok(ModelMetadata {
            name: metadata_utils::extract_model_name(path),
            format: ModelFormat::Safetensors,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads: 8, // Default, could be inferred from attention weights
            context_length: 2048, // Default, could be inferred from position embeddings
        })
    }

    fn can_load(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
    }
}

impl SafetensorsFormatLoader {
    /// Infer vocabulary size from embedding layer
    /// 埋め込み層から語彙サイズを推論
    fn infer_vocab_size(
        loader: &rustorch::formats::safetensors::SafetensorsLoader,
        tensor_names: &[String],
    ) -> usize {
        tensor_names
            .iter()
            .find(|name| name.contains("embed") && name.contains("weight"))
            .and_then(|name| loader.tensor_info(name).ok())
            .and_then(|info| info.shape.first().copied())
            .unwrap_or(32000)
    }

    /// Infer hidden size from embedding layer
    /// 埋め込み層から隠れ層サイズを推論
    fn infer_hidden_size(
        loader: &rustorch::formats::safetensors::SafetensorsLoader,
        tensor_names: &[String],
    ) -> usize {
        tensor_names
            .iter()
            .find(|name| name.contains("embed") && name.contains("weight"))
            .and_then(|name| loader.tensor_info(name).ok())
            .and_then(|info| info.shape.get(1).copied())
            .unwrap_or(512)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_load() {
        assert!(SafetensorsFormatLoader::can_load(Path::new("model.safetensors")));
        assert!(SafetensorsFormatLoader::can_load(Path::new("model.SAFETENSORS")));
        assert!(!SafetensorsFormatLoader::can_load(Path::new("model.gguf")));
        assert!(!SafetensorsFormatLoader::can_load(Path::new("model.pt")));
    }
}
