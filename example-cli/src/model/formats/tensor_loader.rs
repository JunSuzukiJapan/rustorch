// Tensor data loader for GGUF format

use super::gguf::{GGMLType, GGUFLoader, GGUFTensorInfo};
use anyhow::{Context, Result};
use rustorch::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Load tensors from GGUF file
pub struct TensorLoader {
    loader: GGUFLoader,
    reader: BufReader<File>,
}

impl TensorLoader {
    /// Create new tensor loader from GGUF file
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let loader = GGUFLoader::new(path.as_ref())?;

        // Re-open file for tensor data reading
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
        let reader = BufReader::new(file);

        Ok(Self { loader, reader })
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.loader.get_tensor(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.loader.tensor_names()
    }

    /// Load tensor data by name and convert to RusTorch Tensor
    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor<f64>> {
        // Clone tensor_info to avoid borrow checker issues
        let tensor_info = self
            .get_tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?
            .clone();

        tracing::debug!(
            "Loading tensor '{}': type={:?}, dims={:?}, offset={}",
            name,
            tensor_info.ggml_type,
            tensor_info.dims,
            tensor_info.offset
        );

        // Read raw tensor data
        let raw_data = self.read_tensor_data(&tensor_info)?;

        // Dequantize to f64
        let f64_data = dequantize_tensor(&raw_data, tensor_info.ggml_type)?;

        // Convert dims from u64 to usize
        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();

        // Create RusTorch tensor
        Ok(Tensor::from_vec(f64_data, shape))
    }

    /// Load all tensors and return as a map
    pub fn load_all_tensors(&mut self) -> Result<HashMap<String, Tensor<f64>>> {
        let mut tensors = HashMap::new();
        let names = self.tensor_names();

        for name in names {
            let tensor = self.load_tensor(&name)?;
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    /// Read raw tensor data from file
    fn read_tensor_data(&mut self, info: &GGUFTensorInfo) -> Result<Vec<u8>> {
        // Calculate tensor data size in bytes
        let num_elements: u64 = info.dims.iter().product();
        let bytes_per_element = bytes_per_element(info.ggml_type);
        let total_bytes = (num_elements as usize) * bytes_per_element;

        // Seek to tensor data offset
        self.reader.seek(SeekFrom::Start(info.offset))?;

        // Read tensor data
        let mut buffer = vec![0u8; total_bytes];
        self.reader
            .read_exact(&mut buffer)
            .with_context(|| format!("Failed to read tensor data for '{}'", info.name))?;

        Ok(buffer)
    }
}

/// Get bytes per element for a GGML type
fn bytes_per_element(ggml_type: GGMLType) -> usize {
    match ggml_type {
        GGMLType::F32 => 4,
        GGMLType::F16 => 2,
        GGMLType::I8 => 1,
        GGMLType::I16 => 2,
        GGMLType::I32 => 4,
        // Quantized types use block-based storage
        // These are approximate - actual implementation would handle blocks
        GGMLType::Q4_0 | GGMLType::Q4_1 => 1, // ~0.5 bytes per element (4-bit)
        GGMLType::Q5_0 | GGMLType::Q5_1 => 1, // ~0.625 bytes per element (5-bit)
        GGMLType::Q8_0 | GGMLType::Q8_1 => 1, // 1 byte per element (8-bit)
        GGMLType::Q2_K => 1,                   // ~0.25 bytes per element (2-bit)
        GGMLType::Q3_K => 1,                   // ~0.375 bytes per element (3-bit)
        GGMLType::Q4_K => 1,                   // ~0.5 bytes per element (4-bit)
        GGMLType::Q5_K => 1,                   // ~0.625 bytes per element (5-bit)
        GGMLType::Q6_K => 1,                   // ~0.75 bytes per element (6-bit)
        GGMLType::Q8_K => 1,                   // 1 byte per element (8-bit)
    }
}

/// Dequantize tensor data to f64
fn dequantize_tensor(data: &[u8], ggml_type: GGMLType) -> Result<Vec<f64>> {
    match ggml_type {
        GGMLType::F32 => dequantize_f32(data),
        GGMLType::F16 => dequantize_f16(data),
        GGMLType::I8 => dequantize_i8(data),
        GGMLType::I16 => dequantize_i16(data),
        GGMLType::I32 => dequantize_i32(data),
        // Quantized types - simplified dequantization (would need proper implementation)
        _ => {
            tracing::warn!(
                "Quantized type {:?} not fully implemented, using placeholder dequantization",
                ggml_type
            );
            dequantize_quantized_placeholder(data)
        }
    }
}

/// Dequantize F32 data
fn dequantize_f32(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 4 != 0 {
        anyhow::bail!("Invalid F32 data length");
    }

    let mut result = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        let value = f32::from_le_bytes(bytes);
        result.push(value as f64);
    }

    Ok(result)
}

/// Dequantize F16 data (simplified - would use proper half-precision library)
fn dequantize_f16(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 2 != 0 {
        anyhow::bail!("Invalid F16 data length");
    }

    let mut result = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        // Simplified: treat as scaled integer
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let value = i16::from_le_bytes(bytes);
        result.push((value as f64) / 32768.0);
    }

    Ok(result)
}

/// Dequantize I8 data
fn dequantize_i8(data: &[u8]) -> Result<Vec<f64>> {
    Ok(data.iter().map(|&b| (b as i8) as f64).collect())
}

/// Dequantize I16 data
fn dequantize_i16(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 2 != 0 {
        anyhow::bail!("Invalid I16 data length");
    }

    let mut result = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let value = i16::from_le_bytes(bytes);
        result.push(value as f64);
    }

    Ok(result)
}

/// Dequantize I32 data
fn dequantize_i32(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 4 != 0 {
        anyhow::bail!("Invalid I32 data length");
    }

    let mut result = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        let value = i32::from_le_bytes(bytes);
        result.push(value as f64);
    }

    Ok(result)
}

/// Placeholder dequantization for quantized types
/// Real implementation would properly decode the quantization format
fn dequantize_quantized_placeholder(data: &[u8]) -> Result<Vec<f64>> {
    // For now, treat each byte as a quantized value and scale to [-1, 1]
    Ok(data
        .iter()
        .map(|&b| (b as f64 - 128.0) / 128.0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_element_f32() {
        assert_eq!(bytes_per_element(GGMLType::F32), 4);
    }

    #[test]
    fn test_bytes_per_element_f16() {
        assert_eq!(bytes_per_element(GGMLType::F16), 2);
    }

    #[test]
    fn test_bytes_per_element_i8() {
        assert_eq!(bytes_per_element(GGMLType::I8), 1);
    }

    #[test]
    fn test_dequantize_f32() {
        let data = vec![0u8, 0, 128, 63]; // 1.0 in f32 little-endian
        let result = dequantize_f32(&data).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_i8() {
        let data = vec![0u8, 127, 255];
        let result = dequantize_i8(&data).unwrap();
        assert_eq!(result, vec![0.0, 127.0, -1.0]);
    }

    #[test]
    fn test_dequantize_i16() {
        let data = vec![0u8, 0, 1, 0]; // [0, 1] in i16 little-endian
        let result = dequantize_i16(&data).unwrap();
        assert_eq!(result, vec![0.0, 1.0]);
    }

    #[test]
    fn test_dequantize_i32() {
        let data = vec![0u8, 0, 0, 0, 1, 0, 0, 0]; // [0, 1] in i32 little-endian
        let result = dequantize_i32(&data).unwrap();
        assert_eq!(result, vec![0.0, 1.0]);
    }

    #[test]
    fn test_dequantize_invalid_length() {
        let data = vec![0u8, 0, 0]; // Invalid length for f32 (not multiple of 4)
        assert!(dequantize_f32(&data).is_err());
    }

    #[test]
    fn test_tensor_loader_nonexistent_file() {
        let result = TensorLoader::new("nonexistent.gguf");
        assert!(result.is_err());
    }
}
