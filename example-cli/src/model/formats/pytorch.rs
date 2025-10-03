// PyTorch format loader (.pt/.pth files)
use super::tensor_loader::TensorLoader;
use crate::model::ModelMetadata;
use anyhow::{Context, Result};
use rustorch::prelude::Tensor;
use serde_pickle::{DeOptions, HashableValue};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// PyTorch model loader
pub struct PyTorchLoader;

impl PyTorchLoader {
    /// Load PyTorch model from .pt or .pth file
    pub fn load(path: &Path) -> Result<(HashMap<String, Tensor<f64>>, ModelMetadata)> {
        tracing::info!("Loading PyTorch model from {:?}", path);

        let file = File::open(path)
            .with_context(|| format!("Failed to open PyTorch file: {:?}", path))?;

        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .with_context(|| "Failed to read PyTorch file")?;

        // PyTorch files are pickle format with a specific structure
        // Try to deserialize as a pickle object
        let state_dict = Self::deserialize_state_dict(&buffer)
            .with_context(|| "Failed to deserialize PyTorch state_dict")?;

        tracing::info!(
            "Successfully loaded {} tensors from PyTorch model",
            state_dict.len()
        );

        // Extract model metadata
        let model_metadata = Self::extract_metadata(&state_dict)?;

        Ok((state_dict, model_metadata))
    }

    /// Deserialize PyTorch state_dict from pickle format
    fn deserialize_state_dict(data: &[u8]) -> Result<HashMap<String, Tensor<f64>>> {
        // Parse pickle data
        let value = serde_pickle::from_slice(data, DeOptions::default())
            .with_context(|| "Failed to parse pickle data")?;

        let mut tensors = HashMap::new();

        // PyTorch state_dict is typically a dictionary
        match value {
            serde_pickle::Value::Dict(dict) => {
                for (key, val) in dict {
                    // Extract tensor name from HashableValue
                    let name = match key {
                        HashableValue::String(s) => s,
                        HashableValue::Bytes(b) => {
                            String::from_utf8(b).unwrap_or_else(|_| "unknown".to_string())
                        }
                        _ => continue,
                    };

                    // Parse tensor value
                    if let Some(tensor) = Self::parse_tensor_value(val)? {
                        tensors.insert(name, tensor);
                    }
                }
            }
            _ => {
                anyhow::bail!("Expected dictionary as PyTorch state_dict, got other type");
            }
        }

        Ok(tensors)
    }

    /// Parse a pickle value into a RusTorch tensor
    fn parse_tensor_value(value: serde_pickle::Value) -> Result<Option<Tensor<f64>>> {
        match value {
            serde_pickle::Value::Tuple(tuple) => {
                // PyTorch tensors are often stored as tuples with metadata
                if tuple.len() < 2 {
                    return Ok(None);
                }

                // Try to extract shape and data
                if let Some(shape) = Self::extract_shape(&tuple[0]) {
                    if let Some(data) = Self::extract_data(&tuple[1]) {
                        // Default to F32 for now
                        return Self::create_tensor_from_data(&data, &shape);
                    }
                }

                Ok(None)
            }
            serde_pickle::Value::List(list) => {
                // Sometimes tensors are stored as flat lists
                if list.is_empty() {
                    return Ok(None);
                }

                // Convert list to f64 values
                let values: Vec<f64> = list
                    .iter()
                    .filter_map(|v| match v {
                        serde_pickle::Value::F64(f) => Some(*f),
                        serde_pickle::Value::I64(i) => Some(*i as f64),
                        _ => None,
                    })
                    .collect();

                if values.is_empty() {
                    return Ok(None);
                }

                let shape = vec![values.len()];
                Ok(Some(Tensor::from_vec(values, shape)))
            }
            _ => Ok(None),
        }
    }

    /// Extract shape from pickle value
    fn extract_shape(value: &serde_pickle::Value) -> Option<Vec<i64>> {
        match value {
            serde_pickle::Value::Tuple(tuple) => {
                let shape: Vec<i64> = tuple
                    .iter()
                    .filter_map(|v| match v {
                        serde_pickle::Value::I64(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                if shape.is_empty() {
                    None
                } else {
                    Some(shape)
                }
            }
            serde_pickle::Value::List(list) => {
                let shape: Vec<i64> = list
                    .iter()
                    .filter_map(|v| match v {
                        serde_pickle::Value::I64(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                if shape.is_empty() {
                    None
                } else {
                    Some(shape)
                }
            }
            _ => None,
        }
    }

    /// Extract tensor data from pickle value
    fn extract_data(value: &serde_pickle::Value) -> Option<Vec<u8>> {
        match value {
            serde_pickle::Value::Bytes(bytes) => Some(bytes.clone()),
            serde_pickle::Value::String(s) => Some(s.as_bytes().to_vec()),
            _ => None,
        }
    }

    /// Create RusTorch tensor from raw data and shape
    fn create_tensor_from_data(data: &[u8], shape: &[i64]) -> Result<Option<Tensor<f64>>> {
        // Convert shape to usize
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();

        // Try F32 first (most common)
        if data.len() % 4 == 0 {
            let values = TensorLoader::bytes_to_f32_vec(data)?;
            let f64_values: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            return Ok(Some(Tensor::from_vec(f64_values, shape_usize)));
        }

        // Try F64
        if data.len() % 8 == 0 {
            let values = TensorLoader::bytes_to_f64_vec(data)?;
            return Ok(Some(Tensor::from_vec(values, shape_usize)));
        }

        Ok(None)
    }

    /// Extract model metadata from state_dict
    fn extract_metadata(state_dict: &HashMap<String, Tensor<f64>>) -> Result<ModelMetadata> {
        // Try to infer metadata from tensor names and shapes
        let mut vocab_size = 32000;
        let mut hidden_size = 4096;
        let mut num_layers = 0;
        let mut num_heads = 32;
        let mut context_length = 2048;

        for (name, tensor) in state_dict {
            // Look for embedding layer to get vocab size
            if name.contains("embed") && name.contains("weight") {
                let shape = tensor.size();
                if !shape.is_empty() {
                    vocab_size = shape[0];
                    if shape.len() > 1 {
                        hidden_size = shape[1];
                    }
                }
            }

            // Count layers
            if name.contains("layer") || name.contains("block") {
                // Extract layer number from name
                if let Some(layer_num) = Self::extract_layer_number(name) {
                    num_layers = num_layers.max(layer_num + 1);
                }
            }

            // Look for attention heads
            if name.contains("attention") && name.contains("weight") {
                let shape = tensor.size();
                if shape.len() >= 2 {
                    // Infer num_heads from attention weight dimensions
                    let dim = shape[0];
                    if dim % hidden_size == 0 {
                        num_heads = dim / hidden_size;
                    }
                }
            }

            // Look for positional embeddings
            if name.contains("position") && name.contains("embed") {
                let shape = tensor.size();
                if !shape.is_empty() {
                    context_length = shape[0];
                }
            }
        }

        // Set defaults if not found
        if num_layers == 0 {
            num_layers = 32;
        }

        Ok(ModelMetadata {
            name: "pytorch_model".to_string(),
            format: crate::model::ModelFormat::PyTorch,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length,
        })
    }

    /// Extract layer number from tensor name
    fn extract_layer_number(name: &str) -> Option<usize> {
        // Try to find patterns like "layer.0", "block_1", "layers.23", etc.
        for part in name.split(&['.', '_'][..]) {
            if let Ok(num) = part.parse::<usize>() {
                return Some(num);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(PyTorchLoader::extract_layer_number("layer.0.weight"), Some(0));
        assert_eq!(PyTorchLoader::extract_layer_number("block_12.attn"), Some(12));
        assert_eq!(
            PyTorchLoader::extract_layer_number("layers.23.mlp.weight"),
            Some(23)
        );
        assert_eq!(PyTorchLoader::extract_layer_number("embed.weight"), None);
    }

    #[test]
    fn test_extract_shape_from_tuple() {
        let tuple = vec![
            serde_pickle::Value::I64(10),
            serde_pickle::Value::I64(20),
            serde_pickle::Value::I64(30),
        ];
        let value = serde_pickle::Value::Tuple(tuple);
        let shape = PyTorchLoader::extract_shape(&value).unwrap();
        assert_eq!(shape, vec![10, 20, 30]);
    }

    #[test]
    fn test_extract_shape_from_list() {
        let list = vec![
            serde_pickle::Value::I64(5),
            serde_pickle::Value::I64(15),
        ];
        let value = serde_pickle::Value::List(list);
        let shape = PyTorchLoader::extract_shape(&value).unwrap();
        assert_eq!(shape, vec![5, 15]);
    }

    #[test]
    fn test_extract_data_from_bytes() {
        let bytes = vec![1, 2, 3, 4];
        let value = serde_pickle::Value::Bytes(bytes.clone());
        let data = PyTorchLoader::extract_data(&value).unwrap();
        assert_eq!(data, bytes);
    }
}
