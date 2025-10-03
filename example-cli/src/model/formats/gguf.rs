// GGUF format loader
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub key: String,
    pub value: GGUFValue,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFValue::UInt32(v) => Some(*v),
            GGUFValue::UInt64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::UInt64(v) => Some(*v),
            GGUFValue::UInt32(v) => Some(*v as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl GGMLType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2_K),
            11 => Ok(GGMLType::Q3_K),
            12 => Ok(GGMLType::Q4_K),
            13 => Ok(GGMLType::Q5_K),
            14 => Ok(GGMLType::Q6_K),
            15 => Ok(GGMLType::Q8_K),
            16 => Ok(GGMLType::I8),
            17 => Ok(GGMLType::I16),
            18 => Ok(GGMLType::I32),
            _ => anyhow::bail!("Unknown GGML type: {}", value),
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::I32 => 4,
            GGMLType::F16 | GGMLType::I16 => 2,
            GGMLType::I8 => 1,
            // Quantized types - approximate block sizes
            GGMLType::Q4_0 => 18, // 32 values per block
            GGMLType::Q4_1 => 20,
            GGMLType::Q5_0 => 22,
            GGMLType::Q5_1 => 24,
            GGMLType::Q8_0 => 34,
            GGMLType::Q8_1 => 36,
            GGMLType::Q2_K => 82,
            GGMLType::Q3_K => 110,
            GGMLType::Q4_K => 144,
            GGMLType::Q5_K => 176,
            GGMLType::Q6_K => 210,
            GGMLType::Q8_K => 292,
        }
    }
}

pub struct GGUFLoader {
    #[allow(dead_code)]
    reader: BufReader<File>,
    header: GGUFHeader,
    metadata: HashMap<String, GGUFValue>,
    tensors: Vec<GGUFTensorInfo>,
    #[allow(dead_code)]
    tensor_data_offset: u64,
}

impl GGUFLoader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file = File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;

        let mut reader = BufReader::new(file);

        tracing::info!("Parsing GGUF file: {}", path.display());

        let header = Self::read_header(&mut reader)?;
        tracing::debug!(
            "GGUF header: version={}, tensor_count={}, metadata_count={}",
            header.version,
            header.tensor_count,
            header.metadata_kv_count
        );

        let metadata_list = Self::read_metadata(&mut reader, header.metadata_kv_count)?;
        let metadata: HashMap<String, GGUFValue> = metadata_list
            .into_iter()
            .map(|m| (m.key, m.value))
            .collect();

        tracing::debug!("Loaded {} metadata entries", metadata.len());

        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;
        tracing::debug!("Loaded {} tensor definitions", tensors.len());

        // Calculate tensor data offset
        let current_pos = reader.stream_position()?;
        let alignment = 32; // GGUF uses 32-byte alignment
        let tensor_data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(Self {
            reader,
            header,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    fn read_header(reader: &mut BufReader<File>) -> Result<GGUFHeader> {
        let magic = Self::read_u32(reader)?;

        if magic != GGUF_MAGIC {
            anyhow::bail!(
                "Invalid GGUF magic number: 0x{:08x} (expected 0x{:08x})",
                magic,
                GGUF_MAGIC
            );
        }

        let version = Self::read_u32(reader)?;

        if version != GGUF_VERSION {
            tracing::warn!(
                "GGUF version mismatch: {} (expected {})",
                version,
                GGUF_VERSION
            );
        }

        let tensor_count = Self::read_u64(reader)?;
        let metadata_kv_count = Self::read_u64(reader)?;

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_metadata(reader: &mut BufReader<File>, count: u64) -> Result<Vec<GGUFMetadata>> {
        let mut metadata = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let key = Self::read_string(reader)?;
            let value_type = Self::read_u32(reader)?;
            let value = Self::read_value(reader, value_type)?;

            metadata.push(GGUFMetadata { key, value });
        }

        Ok(metadata)
    }

    fn read_tensor_info(reader: &mut BufReader<File>, count: u64) -> Result<Vec<GGUFTensorInfo>> {
        let mut tensors = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let name = Self::read_string(reader)?;
            let n_dims = Self::read_u32(reader)?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(reader)?);
            }

            let ggml_type_value = Self::read_u32(reader)?;
            let ggml_type = GGMLType::from_u32(ggml_type_value)?;

            let offset = Self::read_u64(reader)?;

            tensors.push(GGUFTensorInfo {
                name,
                n_dims,
                dims,
                ggml_type,
                offset,
            });
        }

        Ok(tensors)
    }

    fn read_value(reader: &mut BufReader<File>, value_type: u32) -> Result<GGUFValue> {
        match value_type {
            0 => Ok(GGUFValue::UInt8(Self::read_u8(reader)?)),
            1 => Ok(GGUFValue::Int8(Self::read_i8(reader)?)),
            2 => Ok(GGUFValue::UInt16(Self::read_u16(reader)?)),
            3 => Ok(GGUFValue::Int16(Self::read_i16(reader)?)),
            4 => Ok(GGUFValue::UInt32(Self::read_u32(reader)?)),
            5 => Ok(GGUFValue::Int32(Self::read_i32(reader)?)),
            6 => Ok(GGUFValue::Float32(Self::read_f32(reader)?)),
            7 => Ok(GGUFValue::Bool(Self::read_u8(reader)? != 0)),
            8 => Ok(GGUFValue::String(Self::read_string(reader)?)),
            9 => {
                let array_type = Self::read_u32(reader)?;
                let array_len = Self::read_u64(reader)?;
                let mut array = Vec::with_capacity(array_len as usize);
                for _ in 0..array_len {
                    array.push(Self::read_value(reader, array_type)?);
                }
                Ok(GGUFValue::Array(array))
            }
            10 => Ok(GGUFValue::UInt64(Self::read_u64(reader)?)),
            11 => Ok(GGUFValue::Int64(Self::read_i64(reader)?)),
            12 => Ok(GGUFValue::Float64(Self::read_f64(reader)?)),
            _ => anyhow::bail!("Unknown value type: {}", value_type),
        }
    }

    fn read_string(reader: &mut BufReader<File>) -> Result<String> {
        let len = Self::read_u64(reader)?;
        let mut buf = vec![0u8; len as usize];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).context("Invalid UTF-8 in string")
    }

    // Helper methods for reading primitive types
    fn read_u8(reader: &mut BufReader<File>) -> Result<u8> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(reader: &mut BufReader<File>) -> Result<i8> {
        Ok(Self::read_u8(reader)? as i8)
    }

    fn read_u16(reader: &mut BufReader<File>) -> Result<u16> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(reader: &mut BufReader<File>) -> Result<i16> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(reader: &mut BufReader<File>) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(reader: &mut BufReader<File>) -> Result<i32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(reader: &mut BufReader<File>) -> Result<u64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(reader: &mut BufReader<File>) -> Result<i64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(reader: &mut BufReader<File>) -> Result<f32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(reader: &mut BufReader<File>) -> Result<f64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    // Public API
    pub fn header(&self) -> &GGUFHeader {
        &self.header
    }

    pub fn metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[GGUFTensorInfo] {
        &self.tensors
    }

    pub fn get_metadata(&self, key: &str) -> Option<&GGUFValue> {
        self.metadata.get(key)
    }

    pub fn find_tensor(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Alias for find_tensor for compatibility
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.find_tensor(name)
    }

    /// Get list of all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }

    /// Load tensor data (placeholder - will be implemented when integrating with RusTorch)
    pub fn load_tensor_data(&mut self, _tensor_name: &str) -> Result<Vec<u8>> {
        // TODO: Implement actual tensor data loading
        // This requires seeking to tensor_data_offset + tensor.offset
        // and reading the appropriate number of bytes
        anyhow::bail!("Tensor data loading not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GGMLType::from_u32(0).unwrap(), GGMLType::F32);
        assert_eq!(GGMLType::from_u32(1).unwrap(), GGMLType::F16);
        assert_eq!(GGMLType::from_u32(2).unwrap(), GGMLType::Q4_0);
        assert!(GGMLType::from_u32(999).is_err());
    }

    #[test]
    fn test_ggml_type_size() {
        assert_eq!(GGMLType::F32.type_size(), 4);
        assert_eq!(GGMLType::F16.type_size(), 2);
        assert_eq!(GGMLType::I8.type_size(), 1);
    }

    #[test]
    fn test_gguf_value_as_str() {
        let val = GGUFValue::String("test".to_string());
        assert_eq!(val.as_str(), Some("test"));

        let val = GGUFValue::UInt32(42);
        assert_eq!(val.as_str(), None);
    }

    #[test]
    fn test_gguf_value_as_u32() {
        let val = GGUFValue::UInt32(42);
        assert_eq!(val.as_u32(), Some(42));

        let val = GGUFValue::String("test".to_string());
        assert_eq!(val.as_u32(), None);
    }

    #[test]
    fn test_gguf_loader_nonexistent_file() {
        let result = GGUFLoader::new("nonexistent.gguf");
        assert!(result.is_err());
    }
}
