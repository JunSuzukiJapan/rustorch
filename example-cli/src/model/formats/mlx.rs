// MLX format loader for Apple's MLX framework
use super::tensor_loader::TensorLoader;
use crate::model::ModelMetadata;
use anyhow::{Context, Result};
use rustorch::prelude::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// MLX model loader
pub struct MLXLoader;

impl MLXLoader {
    /// Load MLX model from file
    pub fn load(path: &Path) -> Result<(HashMap<String, Tensor<f64>>, ModelMetadata)> {
        tracing::info!("Loading MLX model from {:?}", path);

        // MLX uses safetensors format internally with .mlx extension
        // Read the file and parse as safetensors-compatible format
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open MLX file: {:?}", path))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| "Failed to read MLX file")?;

        // Parse MLX header (first 8 bytes contain metadata size)
        if buffer.len() < 8 {
            anyhow::bail!("MLX file too small: {} bytes", buffer.len());
        }

        let metadata_size = u64::from_le_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ]) as usize;

        if buffer.len() < 8 + metadata_size {
            anyhow::bail!(
                "MLX file corrupted: metadata size {} exceeds file size {}",
                metadata_size,
                buffer.len()
            );
        }

        // Parse metadata JSON
        let metadata_bytes = &buffer[8..8 + metadata_size];
        let metadata_json: serde_json::Value = serde_json::from_slice(metadata_bytes)
            .with_context(|| "Failed to parse MLX metadata")?;

        tracing::debug!("MLX metadata: {:#?}", metadata_json);

        // Extract tensor information from metadata
        let tensors_meta = metadata_json
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid MLX metadata format"))?;

        let mut tensors = HashMap::new();
        let data_start = 8 + metadata_size;

        for (name, meta) in tensors_meta {
            if name == "__metadata__" {
                continue; // Skip metadata entry
            }

            let dtype = meta["dtype"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing dtype for tensor {}", name))?;

            let shape = meta["shape"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing shape for tensor {}", name))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| anyhow::anyhow!("Invalid shape dimension"))
                })
                .collect::<Result<Vec<i64>>>()?;

            let data_offsets = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing data_offsets for tensor {}", name))?;

            let start = data_offsets[0]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("Invalid start offset"))?
                as usize;
            let end = data_offsets[1]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("Invalid end offset"))?
                as usize;

            // Extract tensor data
            let tensor_data = &buffer[data_start + start..data_start + end];

            // Convert to RusTorch tensor based on dtype
            let tensor = Self::parse_tensor(tensor_data, &shape, dtype)
                .with_context(|| format!("Failed to parse tensor {}", name))?;

            tensors.insert(name.clone(), tensor);
        }

        // Extract model metadata
        let model_metadata = Self::extract_metadata(&metadata_json)?;

        tracing::info!("Successfully loaded {} tensors from MLX model", tensors.len());

        Ok((tensors, model_metadata))
    }

    /// Parse tensor data based on dtype
    fn parse_tensor(data: &[u8], shape: &[i64], dtype: &str) -> Result<Tensor<f64>> {
        // Convert shape to usize
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();

        match dtype {
            "F32" | "float32" => {
                let values = TensorLoader::bytes_to_f32_vec(data)?;
                let f64_values: Vec<f64> = values.iter().map(|&v| v as f64).collect();
                Ok(Tensor::from_vec(f64_values, shape_usize))
            }
            "F16" | "float16" => {
                let values = TensorLoader::bytes_to_f16_vec(data)?;
                let f64_values: Vec<f64> = values.iter().map(|&v| v.to_f64()).collect();
                Ok(Tensor::from_vec(f64_values, shape_usize))
            }
            "I32" | "int32" => {
                let values = TensorLoader::bytes_to_i32_vec(data)?;
                let f64_values: Vec<f64> = values.iter().map(|&v| v as f64).collect();
                Ok(Tensor::from_vec(f64_values, shape_usize))
            }
            "I64" | "int64" => {
                let values = TensorLoader::bytes_to_i64_vec(data)?;
                let f64_values: Vec<f64> = values.iter().map(|&v| v as f64).collect();
                Ok(Tensor::from_vec(f64_values, shape_usize))
            }
            _ => anyhow::bail!("Unsupported MLX dtype: {}", dtype),
        }
    }

    /// Extract model metadata from MLX metadata JSON
    fn extract_metadata(metadata_json: &serde_json::Value) -> Result<ModelMetadata> {
        let meta = metadata_json
            .get("__metadata__")
            .and_then(|m| m.as_object());

        let (vocab_size, hidden_size, num_layers, num_heads, context_length) = if let Some(meta) =
            meta
        {
            (
                meta.get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32000) as usize,
                meta.get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as usize,
                meta.get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize,
                meta.get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize,
                meta.get("max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize,
            )
        } else {
            // Default values if metadata not available
            (32000, 4096, 32, 32, 2048)
        };

        Ok(ModelMetadata {
            name: "mlx_model".to_string(),
            format: crate::model::ModelFormat::MLX,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_f32_tensor() {
        let data: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64]; // [1.0, 2.0, 3.0]
        let shape = vec![3];
        let tensor = MLXLoader::parse_tensor(&data, &shape, "F32").unwrap();
        assert_eq!(tensor.size(), &[3]);
    }

    #[test]
    fn test_extract_metadata_with_defaults() {
        let json = serde_json::json!({});
        let metadata = MLXLoader::extract_metadata(&json).unwrap();
        assert_eq!(metadata.vocab_size, 32000);
        assert_eq!(metadata.hidden_size, 4096);
        assert_eq!(metadata.num_layers, 32);
    }

    #[test]
    fn test_extract_metadata_with_values() {
        let json = serde_json::json!({
            "__metadata__": {
                "vocab_size": 50000,
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "max_position_embeddings": 4096
            }
        });
        let metadata = MLXLoader::extract_metadata(&json).unwrap();
        assert_eq!(metadata.vocab_size, 50000);
        assert_eq!(metadata.hidden_size, 2048);
        assert_eq!(metadata.num_layers, 24);
        assert_eq!(metadata.num_heads, 16);
        assert_eq!(metadata.context_length, 4096);
    }
}
