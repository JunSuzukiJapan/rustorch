//! GGUF format support for loading quantized models
//! 量子化モデル読み込み用GGUF形式サポート
//!
//! Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION_V3: u32 = 3;
const GGUF_VERSION_V2: u32 = 2;

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// GGUF metadata key-value pair
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub key: String,
    pub value: GGUFValue,
}

/// GGUF value types
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
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFValue::UInt32(v) => Some(*v),
            GGUFValue::Int32(v) if *v >= 0 => Some(*v as u32),
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

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// GGML quantization types
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
}

impl GGMLType {
    pub fn from_u32(value: u32) -> RusTorchResult<Self> {
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
            _ => Err(RusTorchError::ParseError(format!(
                "Unknown GGML type: {}",
                value
            ))),
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18, // 16 nibbles + 2 bytes for scale
            GGMLType::Q4_1 => 20, // 16 nibbles + 2*2 bytes for scale and min
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

/// GGUF file loader
pub struct GGUFLoader {
    header: GGUFHeader,
    metadata: HashMap<String, GGUFValue>,
    tensors: Vec<GGUFTensorInfo>,
    data_offset: u64,
}

impl GGUFLoader {
    /// Load GGUF file from path
    pub fn from_file<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            RusTorchError::IoError(format!(
                "Failed to open GGUF file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read header
        let header = Self::read_header(&mut reader)?;

        // Read metadata
        let metadata_vec = Self::read_metadata(&mut reader, header.metadata_kv_count)?;
        let metadata: HashMap<String, GGUFValue> = metadata_vec
            .into_iter()
            .map(|m| (m.key, m.value))
            .collect();

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;

        // Calculate data offset
        let data_offset = reader.stream_position().map_err(|e| {
            RusTorchError::IoError(format!("Failed to get stream position: {}", e))
        })?;

        // Align to 32 bytes
        let aligned_offset = (data_offset + 31) & !31;
        reader.seek(SeekFrom::Start(aligned_offset)).map_err(|e| {
            RusTorchError::IoError(format!("Failed to seek to data: {}", e))
        })?;

        let final_offset = reader.stream_position().map_err(|e| {
            RusTorchError::IoError(format!("Failed to get final position: {}", e))
        })?;

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset: final_offset,
        })
    }

    /// Get model parameters from metadata
    pub fn get_model_params(&self) -> RusTorchResult<ModelParams> {
        // Get vocab_size from tokenizer.ggml.tokens array length
        let vocab_size = self
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GGUFValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
            .or_else(|| {
                self.metadata
                    .get("llama.vocab_size")
                    .and_then(|v| v.as_u32())
            })
            .ok_or_else(|| RusTorchError::ParseError("Missing vocab_size or tokenizer.ggml.tokens".to_string()))?;

        let hidden_size = self
            .metadata
            .get("llama.embedding_length")
            .or_else(|| self.metadata.get("llama.embed_length"))
            .and_then(|v| v.as_u32())
            .ok_or_else(|| RusTorchError::ParseError("Missing embedding_length".to_string()))?;

        let num_layers = self
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| RusTorchError::ParseError("Missing block_count".to_string()))?;

        let num_heads = self
            .metadata
            .get("llama.attention.head_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| RusTorchError::ParseError("Missing head_count".to_string()))?;

        let context_length = self
            .metadata
            .get("llama.context_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(2048);

        Ok(ModelParams {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length,
        })
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn read_header(reader: &mut BufReader<File>) -> RusTorchResult<GGUFHeader> {
        let magic = Self::read_u32(reader)?;

        if magic != GGUF_MAGIC {
            return Err(RusTorchError::ParseError(format!(
                "Invalid GGUF magic number: 0x{:08x} (expected 0x{:08x})",
                magic, GGUF_MAGIC
            )));
        }

        let version = Self::read_u32(reader)?;

        if version != GGUF_VERSION_V3 && version != GGUF_VERSION_V2 {
            return Err(RusTorchError::ParseError(format!(
                "Unsupported GGUF version: {} (supported: v2, v3)",
                version
            )));
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

    fn read_metadata(
        reader: &mut BufReader<File>,
        count: u64,
    ) -> RusTorchResult<Vec<GGUFMetadata>> {
        let mut metadata = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let key = Self::read_string(reader)?;
            let value_type = Self::read_u32(reader)?;
            let value = Self::read_value(reader, value_type)?;

            metadata.push(GGUFMetadata { key, value });
        }

        Ok(metadata)
    }

    fn read_tensor_info(
        reader: &mut BufReader<File>,
        count: u64,
    ) -> RusTorchResult<Vec<GGUFTensorInfo>> {
        let mut tensors = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let name = Self::read_string(reader)?;
            let n_dims = Self::read_u32(reader)?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(reader)?);
            }

            let ggml_type = Self::read_u32(reader)?;
            let offset = Self::read_u64(reader)?;

            tensors.push(GGUFTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        Ok(tensors)
    }

    fn read_value(
        reader: &mut BufReader<File>,
        value_type: u32,
    ) -> RusTorchResult<GGUFValue> {
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
            _ => Err(RusTorchError::ParseError(format!(
                "Unknown value type: {}",
                value_type
            ))),
        }
    }

    fn read_string(reader: &mut BufReader<File>) -> RusTorchResult<String> {
        let len = Self::read_u64(reader)?;
        if len == 0 {
            return Ok(String::new());
        }
        if len > 10_000_000 {
            return Err(RusTorchError::ParseError(format!(
                "String length too large: {}",
                len
            )));
        }
        let mut buf = vec![0u8; len as usize];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read string: {}", e)))?;
        String::from_utf8(buf)
            .map_err(|e| RusTorchError::ParseError(format!("Invalid UTF-8 in string: {}", e)))
    }

    // Helper methods for reading primitive types
    fn read_u8(reader: &mut BufReader<File>) -> RusTorchResult<u8> {
        let mut buf = [0u8; 1];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read u8: {}", e)))?;
        Ok(buf[0])
    }

    fn read_i8(reader: &mut BufReader<File>) -> RusTorchResult<i8> {
        Ok(Self::read_u8(reader)? as i8)
    }

    fn read_u16(reader: &mut BufReader<File>) -> RusTorchResult<u16> {
        let mut buf = [0u8; 2];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read u16: {}", e)))?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(reader: &mut BufReader<File>) -> RusTorchResult<i16> {
        let mut buf = [0u8; 2];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read i16: {}", e)))?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(reader: &mut BufReader<File>) -> RusTorchResult<u32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read u32: {}", e)))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(reader: &mut BufReader<File>) -> RusTorchResult<i32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read i32: {}", e)))?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(reader: &mut BufReader<File>) -> RusTorchResult<u64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read u64: {}", e)))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(reader: &mut BufReader<File>) -> RusTorchResult<i64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read i64: {}", e)))?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(reader: &mut BufReader<File>) -> RusTorchResult<f32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read f32: {}", e)))?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(reader: &mut BufReader<File>) -> RusTorchResult<f64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read f64: {}", e)))?;
        Ok(f64::from_le_bytes(buf))
    }
}

/// Model parameters extracted from GGUF metadata
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub context_length: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_element_size() {
        assert_eq!(GGMLType::F32.element_size(), 4);
        assert_eq!(GGMLType::F16.element_size(), 2);
        assert_eq!(GGMLType::Q4_K.element_size(), 144);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GGMLType::from_u32(0).unwrap(), GGMLType::F32);
        assert_eq!(GGMLType::from_u32(12).unwrap(), GGMLType::Q4_K);
        assert!(GGMLType::from_u32(999).is_err());
    }
}
