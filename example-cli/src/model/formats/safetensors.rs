//! Safetensors format loader for Hugging Face models

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Safetensors model loader
pub struct SafetensorsLoader {
    data: Vec<u8>,
    tensors: HashMap<String, TensorInfo>,
}

/// Tensor metadata information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: usize,
    pub size: usize,
}

impl SafetensorsLoader {
    /// Load a Safetensors file from path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Read entire file into memory
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open Safetensors file: {}", path.display()))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .with_context(|| "Failed to read Safetensors file")?;

        Self::from_bytes(data)
    }

    /// Load from raw bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        // Parse safetensors format
        let st = SafeTensors::deserialize(&data)
            .with_context(|| "Failed to parse Safetensors format")?;

        // Extract tensor metadata
        let mut tensors = HashMap::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .with_context(|| format!("Failed to get tensor: {}", name))?;

            let info = TensorInfo {
                name: name.to_string(),
                shape: view.shape().to_vec(),
                dtype: format!("{:?}", view.dtype()),
                offset: view.data().as_ptr() as usize - data.as_ptr() as usize,
                size: view.data().len(),
            };

            tensors.insert(name.to_string(), info);
        }

        Ok(Self { data, tensors })
    }

    /// Get list of tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get tensor metadata by name
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get tensor data as raw bytes
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        Some(&self.data[info.offset..info.offset + info.size])
    }

    /// Get all tensor metadata
    pub fn all_tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    /// Convert tensor data to f32 vector
    pub fn tensor_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor_info(name)
            .with_context(|| format!("Tensor not found: {}", name))?;

        let data = self
            .tensor_data(name)
            .with_context(|| format!("Failed to get tensor data: {}", name))?;

        // Parse based on dtype
        match info.dtype.as_str() {
            "F32" => {
                // Already f32, just reinterpret bytes
                let float_data = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(float_data)
            }
            "F16" => {
                // Convert f16 to f32
                let float_data = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(float_data)
            }
            "I8" => {
                // Convert i8 to f32
                let float_data = data
                    .iter()
                    .map(|&b| i8::from_le_bytes([b]) as f32)
                    .collect();
                Ok(float_data)
            }
            _ => {
                anyhow::bail!("Unsupported dtype for conversion: {}", info.dtype)
            }
        }
    }

    /// Get model metadata if available
    pub fn metadata(&self) -> HashMap<String, String> {
        // Safetensors can include metadata in the header
        // For now, return basic info
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "safetensors".to_string());
        meta.insert("tensor_count".to_string(), self.tensors.len().to_string());
        meta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors() -> Vec<u8> {
        // Create a minimal valid safetensors file
        // Header: {"key": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
        let header = r#"{"test_tensor":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Create full file: [8-byte length][header][data]
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);

        // Add 16 bytes of f32 data (4 floats: 1.0, 2.0, 3.0, 4.0)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());

        data
    }

    #[test]
    fn test_safetensors_from_bytes() {
        let data = create_test_safetensors();
        let loader = SafetensorsLoader::from_bytes(data);
        assert!(loader.is_ok());
    }

    #[test]
    fn test_safetensors_from_file() {
        let data = create_test_safetensors();
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();

        let loader = SafetensorsLoader::from_file(temp_file.path());
        assert!(loader.is_ok());
    }

    #[test]
    fn test_tensor_names() {
        let data = create_test_safetensors();
        let loader = SafetensorsLoader::from_bytes(data).unwrap();
        let names = loader.tensor_names();
        assert!(names.contains(&"test_tensor".to_string()));
    }

    #[test]
    fn test_tensor_info() {
        let data = create_test_safetensors();
        let loader = SafetensorsLoader::from_bytes(data).unwrap();
        let info = loader.tensor_info("test_tensor");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.shape, vec![2, 2]);
    }

    #[test]
    fn test_nonexistent_file() {
        let result = SafetensorsLoader::from_file("/nonexistent/file.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata() {
        let data = create_test_safetensors();
        let loader = SafetensorsLoader::from_bytes(data).unwrap();
        let meta = loader.metadata();
        assert_eq!(meta.get("format"), Some(&"safetensors".to_string()));
    }
}
