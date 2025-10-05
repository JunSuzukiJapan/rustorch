// MLX format loader with common interface implementation
use anyhow::Result;
use std::path::Path;

use crate::model::format_loader::{metadata_utils, FormatLoader};
use crate::model::{ModelFormat, ModelMetadata};

pub struct MLXFormatLoader;

impl FormatLoader for MLXFormatLoader {
    fn load_metadata(path: &Path) -> Result<ModelMetadata> {
        tracing::info!("Loading MLX metadata from: {}", path.display());

        // Use RusTorch's MLX loader
        let mlx_loader = rustorch::formats::mlx::MLXLoader::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load MLX file: {}", e))?;

        // Extract model metadata
        let mlx_meta = mlx_loader.model_metadata();

        tracing::info!(
            "Successfully loaded MLX model with {} tensors",
            mlx_loader.tensor_names().len()
        );

        Ok(ModelMetadata {
            name: metadata_utils::extract_model_name(path),
            format: ModelFormat::MLX,
            vocab_size: mlx_meta.and_then(|m| m.vocab_size).unwrap_or(32000),
            hidden_size: mlx_meta.and_then(|m| m.hidden_size).unwrap_or(4096),
            num_layers: mlx_meta
                .and_then(|m| m.num_hidden_layers)
                .unwrap_or(32),
            num_heads: mlx_meta
                .and_then(|m| m.num_attention_heads)
                .unwrap_or(32),
            context_length: mlx_meta
                .and_then(|m| m.max_position_embeddings)
                .unwrap_or(2048),
        })
    }

    fn can_load(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("mlx") || ext.eq_ignore_ascii_case("npz"))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_load() {
        assert!(MLXFormatLoader::can_load(Path::new("model.mlx")));
        assert!(MLXFormatLoader::can_load(Path::new("model.MLX")));
        assert!(MLXFormatLoader::can_load(Path::new("model.npz")));
        assert!(!MLXFormatLoader::can_load(Path::new("model.gguf")));
        assert!(!MLXFormatLoader::can_load(Path::new("model.pt")));
    }
}
