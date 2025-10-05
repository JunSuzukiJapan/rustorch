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
    loader: Option<GGUFLoader>,
    reader: Option<BufReader<File>>,
}

impl TensorLoader {
    /// Create new tensor loader (without file - for standalone use)
    pub fn new() -> Self {
        Self {
            loader: None,
            reader: None,
        }
    }

    /// Create new tensor loader from GGUF file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let loader = GGUFLoader::new(path.as_ref())?;

        // Re-open file for tensor data reading
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
        let reader = BufReader::new(file);

        Ok(Self {
            loader: Some(loader),
            reader: Some(reader),
        })
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.loader.as_ref()?.get_tensor(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.loader
            .as_ref()
            .map(|l| l.tensor_names())
            .unwrap_or_default()
    }

    /// Load tensor from raw data (dequantize and convert to RusTorch Tensor)
    pub fn load_tensor(
        &self,
        data: &[u8],
        shape: &[u64],
        ggml_type: GGMLType,
    ) -> Result<Tensor<f64>> {
        // Dequantize to f64
        let f64_data = dequantize_tensor(data, ggml_type)?;

        // Convert dims from u64 to usize
        let tensor_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create RusTorch tensor
        Ok(Tensor::from_vec(f64_data, tensor_shape))
    }

    /// Load tensor data by name and convert to RusTorch Tensor (requires file-based loader)
    pub fn load_tensor_by_name(&mut self, name: &str) -> Result<Tensor<f64>> {
        let _loader = self
            .loader
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Loader not initialized"))?;

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

        // Use load_tensor to convert
        self.load_tensor(&raw_data, &tensor_info.dims, tensor_info.ggml_type)
    }

    /// Load all tensors and return as a map
    pub fn load_all_tensors(&mut self) -> Result<HashMap<String, Tensor<f64>>> {
        let mut tensors = HashMap::new();
        let names = self.tensor_names();

        for name in names {
            let tensor = self.load_tensor_by_name(&name)?;
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    /// Read raw tensor data from file
    fn read_tensor_data(&mut self, info: &GGUFTensorInfo) -> Result<Vec<u8>> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Reader not initialized"))?;

        // Calculate tensor data size in bytes
        let num_elements: u64 = info.dims.iter().product();
        let bytes_per_element = bytes_per_element(info.ggml_type);
        let total_bytes = (num_elements as usize) * bytes_per_element;

        // Seek to tensor data offset
        reader.seek(SeekFrom::Start(info.offset))?;

        // Read tensor data
        let mut buffer = vec![0u8; total_bytes];
        reader
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
        GGMLType::Q2_K => 1,                  // ~0.25 bytes per element (2-bit)
        GGMLType::Q3_K => 1,                  // ~0.375 bytes per element (3-bit)
        GGMLType::Q4_K => 1,                  // ~0.5 bytes per element (4-bit)
        GGMLType::Q5_K => 1,                  // ~0.625 bytes per element (5-bit)
        GGMLType::Q6_K => 1,                  // ~0.75 bytes per element (6-bit)
        GGMLType::Q8_K => 1,                  // 1 byte per element (8-bit)
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
        // Quantized types with proper implementation
        GGMLType::Q4_0 => dequantize_q4_0(data),
        GGMLType::Q8_0 => dequantize_q8_0(data),
        // Other quantized types - use placeholder for now
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

/// Dequantize Q4_0 format (4-bit quantization with block-wise scaling)
/// Block format: f16 scale + 16 bytes (32 x 4-bit values)
/// Block size: 32 elements, 18 bytes per block
fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f64>> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18; // 2 bytes scale + 16 bytes data

    if data.len() % BYTES_PER_BLOCK != 0 {
        anyhow::bail!("Invalid Q4_0 data length: {}", data.len());
    }

    let num_blocks = data.len() / BYTES_PER_BLOCK;
    let mut result = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        let block = &data[block_start..block_start + BYTES_PER_BLOCK];

        // Read scale as f16
        let scale_bytes: [u8; 2] = [block[0], block[1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f64();

        // Decode 32 x 4-bit values from 16 bytes
        for byte_idx in 0..16 {
            let byte = block[2 + byte_idx];
            // Lower 4 bits
            let v0 = (byte & 0x0F) as i8 - 8; // Convert to signed (-8 to 7)
            // Upper 4 bits
            let v1 = ((byte >> 4) & 0x0F) as i8 - 8;

            result.push(v0 as f64 * scale);
            result.push(v1 as f64 * scale);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 format (8-bit quantization with block-wise scaling)
/// Block format: f16 scale + 32 bytes (32 x 8-bit values)
/// Block size: 32 elements, 34 bytes per block
fn dequantize_q8_0(data: &[u8]) -> Result<Vec<f64>> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34; // 2 bytes scale + 32 bytes data

    if data.len() % BYTES_PER_BLOCK != 0 {
        anyhow::bail!("Invalid Q8_0 data length: {}", data.len());
    }

    let num_blocks = data.len() / BYTES_PER_BLOCK;
    let mut result = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        let block = &data[block_start..block_start + BYTES_PER_BLOCK];

        // Read scale as f16
        let scale_bytes: [u8; 2] = [block[0], block[1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f64();

        // Decode 32 x 8-bit values
        for i in 0..BLOCK_SIZE {
            let v = block[2 + i] as i8; // Interpret as signed byte
            result.push(v as f64 * scale);
        }
    }

    Ok(result)
}

/// Placeholder dequantization for other quantized types
/// This is used as fallback for formats we haven't implemented yet
fn dequantize_quantized_placeholder(data: &[u8]) -> Result<Vec<f64>> {
    // For now, treat each byte as a quantized value and scale to [-1, 1]
    Ok(data.iter().map(|&b| (b as f64 - 128.0) / 128.0).collect())
}

// Public helper methods for MLX and PyTorch loaders

impl TensorLoader {
    /// Convert bytes to f32 vector
    pub fn bytes_to_f32_vec(data: &[u8]) -> Result<Vec<f32>> {
        if data.len() % 4 != 0 {
            anyhow::bail!("Invalid F32 data length: {}", data.len());
        }

        let mut result = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            let value = f32::from_le_bytes(bytes);
            result.push(value);
        }

        Ok(result)
    }

    /// Convert bytes to f64 vector
    pub fn bytes_to_f64_vec(data: &[u8]) -> Result<Vec<f64>> {
        if data.len() % 8 != 0 {
            anyhow::bail!("Invalid F64 data length: {}", data.len());
        }

        let mut result = Vec::with_capacity(data.len() / 8);
        for chunk in data.chunks_exact(8) {
            let bytes: [u8; 8] = chunk.try_into().unwrap();
            let value = f64::from_le_bytes(bytes);
            result.push(value);
        }

        Ok(result)
    }

    /// Convert bytes to f16 vector (using half crate)
    pub fn bytes_to_f16_vec(data: &[u8]) -> Result<Vec<half::f16>> {
        if data.len() % 2 != 0 {
            anyhow::bail!("Invalid F16 data length: {}", data.len());
        }

        let mut result = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            let bytes: [u8; 2] = chunk.try_into().unwrap();
            let value = half::f16::from_le_bytes(bytes);
            result.push(value);
        }

        Ok(result)
    }

    /// Convert bytes to i32 vector
    pub fn bytes_to_i32_vec(data: &[u8]) -> Result<Vec<i32>> {
        if data.len() % 4 != 0 {
            anyhow::bail!("Invalid I32 data length: {}", data.len());
        }

        let mut result = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            let value = i32::from_le_bytes(bytes);
            result.push(value);
        }

        Ok(result)
    }

    /// Convert bytes to i64 vector
    pub fn bytes_to_i64_vec(data: &[u8]) -> Result<Vec<i64>> {
        if data.len() % 8 != 0 {
            anyhow::bail!("Invalid I64 data length: {}", data.len());
        }

        let mut result = Vec::with_capacity(data.len() / 8);
        for chunk in data.chunks_exact(8) {
            let bytes: [u8; 8] = chunk.try_into().unwrap();
            let value = i64::from_le_bytes(bytes);
            result.push(value);
        }

        Ok(result)
    }
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
    fn test_tensor_loader_creation() {
        // TensorLoader can be created without a file
        let loader = TensorLoader::new();
        assert!(loader.loader.is_none());
        assert!(loader.reader.is_none());
    }
}
