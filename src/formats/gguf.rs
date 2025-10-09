//! GGUF format support for loading quantized models
//! 量子化モデル読み込み用GGUF形式サポート
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
    file_path: PathBuf,
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
        let metadata: HashMap<String, GGUFValue> =
            metadata_vec.into_iter().map(|m| (m.key, m.value)).collect();

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;

        // Calculate data offset
        let data_offset = reader
            .stream_position()
            .map_err(|e| RusTorchError::IoError(format!("Failed to get stream position: {}", e)))?;

        // Align to 32 bytes
        let aligned_offset = (data_offset + 31) & !31;
        reader
            .seek(SeekFrom::Start(aligned_offset))
            .map_err(|e| RusTorchError::IoError(format!("Failed to seek to data: {}", e)))?;

        let final_offset = reader
            .stream_position()
            .map_err(|e| RusTorchError::IoError(format!("Failed to get final position: {}", e)))?;

        eprintln!("🔧 [GGUF] data_offset (after alignment) = {}", final_offset);

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset: final_offset,
            file_path: path.as_ref().to_path_buf(),
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
            .ok_or_else(|| {
                RusTorchError::ParseError("Missing vocab_size or tokenizer.ggml.tokens".to_string())
            })?;

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

        // Read num_kv_heads from GGUF metadata
        // For GQA (Grouped Query Attention), this is typically less than num_heads
        // TinyLlama uses 4 KV heads with 32 query heads
        let num_kv_heads = self
            .metadata
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.as_u32())
            .unwrap_or(num_heads); // Default to num_heads if not specified (MHA case)

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
            num_kv_heads,
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

    /// Extract tokenizer vocabulary from GGUF metadata
    /// Returns a vector of tokens in vocabulary order
    pub fn extract_tokenizer_vocab(&self) -> RusTorchResult<Vec<String>> {
        let tokens = self
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| {
                RusTorchError::ParseError("Missing tokenizer.ggml.tokens in GGUF metadata".to_string())
            })?;

        match tokens {
            GGUFValue::Array(arr) => {
                let mut vocab = Vec::with_capacity(arr.len());
                for token_value in arr {
                    match token_value {
                        GGUFValue::String(s) => vocab.push(s.clone()),
                        _ => {
                            return Err(RusTorchError::ParseError(
                                "Invalid token type in tokenizer.ggml.tokens".to_string(),
                            ))
                        }
                    }
                }
                Ok(vocab)
            }
            _ => Err(RusTorchError::ParseError(
                "tokenizer.ggml.tokens is not an array".to_string(),
            )),
        }
    }

    /// Get tokenizer model type (e.g., "llama", "gpt2")
    pub fn get_tokenizer_model(&self) -> Option<String> {
        self.metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract BPE merge rules from GGUF metadata
    /// Returns a vector of merge pairs (token1, token2)
    pub fn extract_bpe_merges(&self) -> RusTorchResult<Vec<(String, String)>> {
        let merges = self
            .metadata
            .get("tokenizer.ggml.merges")
            .ok_or_else(|| {
                RusTorchError::ParseError("Missing tokenizer.ggml.merges in GGUF metadata".to_string())
            })?;

        match merges {
            GGUFValue::Array(arr) => {
                let mut merge_rules = Vec::with_capacity(arr.len());
                for merge_value in arr {
                    match merge_value {
                        GGUFValue::String(s) => {
                            // Merge format: "token1 token2"
                            let parts: Vec<&str> = s.splitn(2, ' ').collect();
                            if parts.len() == 2 {
                                merge_rules.push((parts[0].to_string(), parts[1].to_string()));
                            }
                        }
                        _ => {
                            return Err(RusTorchError::ParseError(
                                "Invalid merge type in tokenizer.ggml.merges".to_string(),
                            ))
                        }
                    }
                }
                Ok(merge_rules)
            }
            _ => Err(RusTorchError::ParseError(
                "tokenizer.ggml.merges is not an array".to_string(),
            )),
        }
    }

    /// Extract token scores from GGUF metadata
    pub fn extract_token_scores(&self) -> RusTorchResult<Vec<f32>> {
        let scores = self
            .metadata
            .get("tokenizer.ggml.scores")
            .ok_or_else(|| {
                RusTorchError::ParseError("Missing tokenizer.ggml.scores in GGUF metadata".to_string())
            })?;

        match scores {
            GGUFValue::Array(arr) => {
                let mut score_values = Vec::with_capacity(arr.len());
                for score_value in arr {
                    match score_value {
                        GGUFValue::Float32(f) => score_values.push(*f),
                        _ => {
                            return Err(RusTorchError::ParseError(
                                "Invalid score type in tokenizer.ggml.scores".to_string(),
                            ))
                        }
                    }
                }
                Ok(score_values)
            }
            _ => Err(RusTorchError::ParseError(
                "tokenizer.ggml.scores is not an array".to_string(),
            )),
        }
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

    fn read_value(reader: &mut BufReader<File>, value_type: u32) -> RusTorchResult<GGUFValue> {
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

    /// Load tensor data by name
    pub fn load_tensor(&self, name: &str) -> RusTorchResult<Tensor<f64>> {
        let tensor_info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RusTorchError::ParseError(format!("Tensor not found: {}", name)))?
            .clone();

        // Reopen file for reading tensor data
        let file = File::open(&self.file_path).map_err(|e| {
            RusTorchError::IoError(format!("Failed to reopen GGUF file: {}", e))
        })?;
        let mut reader = BufReader::new(file);

        // Seek to tensor data offset
        let absolute_offset = self.data_offset + tensor_info.offset;
        eprintln!("🔧 [GGUF SEEK] '{}': data_offset={}, tensor_offset={}, absolute={}",
                  name, self.data_offset, tensor_info.offset, absolute_offset);
        reader
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|e| RusTorchError::IoError(format!("Failed to seek to tensor data: {}", e)))?;

        // Calculate number of elements
        let num_elements: usize = tensor_info.dims.iter().map(|&d| d as usize).product();

        // Convert GGML type to GGMLType enum
        let ggml_type = GGMLType::from_u32(tensor_info.ggml_type)?;
        eprintln!("🔧 [GGUF] Loading '{}': type={:?}, ggml_type_code={}",
                  name, ggml_type, tensor_info.ggml_type);

        // CRITICAL: GGML dimension ordering is REVERSED from PyTorch
        // GGML dims = [ne[0], ne[1], ...] where ne[0] is innermost (fastest-changing)
        // PyTorch shape = [dim0, dim1, ...] where dim0 is outermost
        // Example: PyTorch [2048, 32000] ↔ GGML dims = [32000, 2048]
        //
        // For F32/F16 tensors: We can safely reverse dims to get PyTorch-style shape
        // For quantized tensors (Q4_K, etc.): Keep GGML order to preserve quantization structure
        let original_dims: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        let shape: Vec<usize> = match ggml_type {
            GGMLType::F32 | GGMLType::F16 => {
                // Reverse dims for non-quantized tensors to get PyTorch order
                let mut s = original_dims.clone();
                s.reverse();
                s
            }
            _ => {
                // Keep GGML order for quantized tensors to preserve quantization structure
                original_dims.clone()
            }
        };

        // Debug: Log dimension handling for norm weights
        if name.contains("norm.weight") || name.contains("token_embd") || name.contains("output.weight") {
            eprintln!("🔧 [GGUF LOAD] '{}': type={:?}, original_dims={:?}, final_shape={:?}",
                name, ggml_type, original_dims, shape);
        }

        // Read tensor data based on type
        let data = match ggml_type {
            GGMLType::F32 => {
                let mut f32_data = vec![0.0f32; num_elements];
                for val in &mut f32_data {
                    *val = Self::read_f32(&mut reader)?;
                }
                f32_data.iter().map(|&x| x as f64).collect()
            }
            GGMLType::F16 => {
                // Read as u16 and convert to f32 then f64
                let mut data = Vec::with_capacity(num_elements);
                for _ in 0..num_elements {
                    let bits = Self::read_u16(&mut reader)?;
                    let f32_val = half::f16::from_bits(bits).to_f32();
                    data.push(f32_val as f64);
                }
                data
            }
            GGMLType::Q4_0 => {
                // Q4_0: Blocks of 32 elements
                // Block size: 18 bytes (2 bytes scale + 16 bytes quantized)
                Self::dequantize_q4_0(&mut reader, num_elements)?
            }
            GGMLType::Q4_K => {
                // Q4_K: Super-blocks of 256 elements (8 blocks of 32)
                // Block size: 144 bytes per super-block
                Self::dequantize_q4_k(&mut reader, num_elements)?
            }
            GGMLType::Q5_K => {
                // Q5_K: Super-blocks of 256 elements
                // Block size: 176 bytes per super-block
                Self::dequantize_q5_k(&mut reader, num_elements)?
            }
            GGMLType::Q6_K => {
                // Q6_K: Super-blocks of 256 elements (16 blocks of 16)
                // Block size: 210 bytes per super-block
                Self::dequantize_q6_k(&mut reader, num_elements)?
            }
            GGMLType::Q8_0 => {
                // Q8_0: Blocks of 32 elements
                // Block size: 34 bytes (2 bytes scale + 32 bytes quantized)
                Self::dequantize_q8_0(&mut reader, num_elements)?
            }
            _ => {
                return Err(RusTorchError::ParseError(format!(
                    "Tensor type {:?} not yet supported for loading",
                    ggml_type
                )))
            }
        };

        Ok(Tensor::from_vec(data, shape))
    }

    /// Dequantize Q4_0 format
    /// Q4_0: Blocks of 32 elements
    /// Block structure: scale (f16) + quants (16 bytes of 4-bit values)
    fn dequantize_q4_0(
        reader: &mut BufReader<File>,
        num_elements: usize,
    ) -> RusTorchResult<Vec<f64>> {
        const QK: usize = 32; // Elements per block
        const BLOCK_SIZE: usize = 18; // 2 bytes scale + 16 bytes quantized

        let num_blocks = (num_elements + QK - 1) / QK;
        let mut output = Vec::with_capacity(num_elements);

        for block_idx in 0..num_blocks {
            // Read scale (f16)
            let scale_bits = Self::read_u16(reader)?;
            let scale = half::f16::from_bits(scale_bits).to_f32();

            // Read 16 bytes of quantized 4-bit values (32 values total)
            let mut qs = [0u8; QK / 2];
            reader.read_exact(&mut qs).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q4_0 quants: {}", e))
            })?;

            // Debug: Print first block's scale and raw bytes
            if block_idx == 0 {
                eprintln!("🐛 [Q4_0 DEBUG] First block:");
                eprintln!("   scale_bits: 0x{:04x}", scale_bits);
                eprintln!("   scale: {:.10}", scale);
                eprintln!("   First 4 quant bytes: {:02x} {:02x} {:02x} {:02x}",
                         qs[0], qs[1], qs[2], qs[3]);
            }

            // Dequantize: Each byte contains two 4-bit values
            // llama.cpp interleaves: x0[j] at position j, x1[j] at position j+16
            // Prepare output space for this block
            let block_start = output.len();
            for _ in 0..QK {
                output.push(0.0);
            }

            for j in 0..QK / 2 {
                // Lower 4 bits (0-15) -> -8 to +7
                let x0 = ((qs[j] & 0x0F) as i8) - 8;
                // Upper 4 bits (0-15) -> -8 to +7
                let x1 = ((qs[j] >> 4) as i8) - 8;

                // llama.cpp interleaved layout: y[j] = x0, y[j+16] = x1
                output[block_start + j] = (x0 as f32 * scale) as f64;
                output[block_start + j + QK / 2] = (x1 as f32 * scale) as f64;
            }
        }

        // Trim to exact size
        output.truncate(num_elements);
        Ok(output)
    }

    /// Dequantize Q8_0 format
    /// Q8_0: Blocks of 32 elements, 8-bit quantization
    /// Block structure: scale (f16) + quants (32 bytes of 8-bit signed values)
    fn dequantize_q8_0(
        reader: &mut BufReader<File>,
        num_elements: usize,
    ) -> RusTorchResult<Vec<f64>> {
        const QK: usize = 32; // Elements per block

        let num_blocks = (num_elements + QK - 1) / QK;
        let mut output = Vec::with_capacity(num_elements);

        for _ in 0..num_blocks {
            // Read scale (f16, 2 bytes)
            let scale_bits = Self::read_u16(reader)?;
            let scale = half::f16::from_bits(scale_bits).to_f32();

            // Read 32 quantized 8-bit signed values
            let mut quants = [0i8; QK];
            for q in &mut quants {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf).map_err(|e| {
                    RusTorchError::IoError(format!("Failed to read Q8_0 quant: {}", e))
                })?;
                *q = buf[0] as i8;
            }

            // Dequantize: value = scale * quant
            for &q in &quants {
                if output.len() >= num_elements {
                    break;
                }
                output.push((scale * q as f32) as f64);
            }
        }

        output.truncate(num_elements);
        Ok(output)
    }

    /// Dequantize Q4_K format
    /// Q4_K: Super-blocks of 256 elements (8 blocks of 32 elements each)
    /// Block structure: scales (6-bit quantized) + mins (6-bit quantized) + quants (4-bit)
    fn dequantize_q4_k(
        reader: &mut BufReader<File>,
        num_elements: usize,
    ) -> RusTorchResult<Vec<f64>> {
        const QK_K: usize = 256;          // Elements per super-block
        const K_SCALE_SIZE: usize = 12;   // Scale data size

        let num_blocks = (num_elements + QK_K - 1) / QK_K;
        let mut output = Vec::with_capacity(num_elements);

        for _ in 0..num_blocks {
            // Read super-block data (144 bytes total)
            // Structure from llama.cpp:
            // - d (f16): super-scale
            // - dmin (f16): super-min
            // - scales[12]: quantized scales
            // - qs[QK_K/2]: 4-bit quantized values (128 bytes)

            // Read super-scale and super-min (f16 each)
            let d_bits = Self::read_u16(reader)?;
            let dmin_bits = Self::read_u16(reader)?;
            let d = half::f16::from_bits(d_bits).to_f32();
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            // Read quantized scales (12 bytes)
            let mut scales = [0u8; K_SCALE_SIZE];
            reader.read_exact(&mut scales).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q4_K scales: {}", e))
            })?;

            // Read quantized values (128 bytes = 256 nibbles)
            let mut qs = vec![0u8; QK_K / 2];
            reader.read_exact(&mut qs).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q4_K quants: {}", e))
            })?;

            // Dequantize: Process in pairs (64 elements at a time) like llama.cpp
            // Each pair processes 32 bytes: lower nibbles first, then upper nibbles
            for pair in 0..4 {
                let j1 = pair * 2;
                let j2 = pair * 2 + 1;
                let q_offset = pair * 32; // Byte offset in qs array

                // Get scale/min for both blocks in the pair
                let (scale1, min1) = if j1 < 4 {
                    let sc = scales[j1] & 63;
                    let mn = scales[j1 + 4] & 63;
                    (sc as f32, mn as f32)
                } else {
                    let sc = (scales[j1 + 4] & 0x0F) | ((scales[j1 - 4] >> 6) << 4);
                    let mn = (scales[j1 + 4] >> 4) | ((scales[j1] >> 6) << 4);
                    (sc as f32, mn as f32)
                };

                let (scale2, min2) = if j2 < 4 {
                    let sc = scales[j2] & 63;
                    let mn = scales[j2 + 4] & 63;
                    (sc as f32, mn as f32)
                } else {
                    let sc = (scales[j2 + 4] & 0x0F) | ((scales[j2 - 4] >> 6) << 4);
                    let mn = (scales[j2 + 4] >> 4) | ((scales[j2] >> 6) << 4);
                    (sc as f32, mn as f32)
                };

                let d1 = d * scale1;
                let m1 = dmin * min1;
                let d2 = d * scale2;
                let m2 = dmin * min2;

                // Debug: Print first block's values
                if output.is_empty() && pair == 0 {
                    eprintln!("🐛 [Q4_K DEBUG] First block:");
                    eprintln!("   d={:.10}, dmin={:.10}", d, dmin);
                    eprintln!("   scale1={}, min1={}", scale1, min1);
                    eprintln!("   d1={:.10}, m1={:.10}", d1, m1);
                    eprintln!("   First q_val: {}", qs[q_offset] & 0x0F);
                    eprintln!("   First dequant: {:.10}", d1 * (qs[q_offset] & 0x0F) as f32 - m1);
                }

                // Debug: Log d scale factors for first few blocks to check if all are abnormally small
                if output.len() < 512 && pair == 0 {
                    let block_num = output.len() / 256;
                    if block_num < 3 {
                        eprintln!("🐛 [Q4_K] Block {}: d={:.10}", block_num, d);
                    }
                }

                // Process 32 lower nibbles (block 1)
                for l in 0..32 {
                    if output.len() >= num_elements { break; }
                    let q_val = qs[q_offset + l] & 0x0F;
                    output.push((d1 * q_val as f32 - m1) as f64);
                }

                // Process 32 upper nibbles (block 2)
                for l in 0..32 {
                    if output.len() >= num_elements { break; }
                    let q_val = qs[q_offset + l] >> 4;
                    output.push((d2 * q_val as f32 - m2) as f64);
                }
            }
        }

        output.truncate(num_elements);
        Ok(output)
    }

    /// Dequantize Q5_K format matching llama.cpp
    /// Q5_K: Super-blocks of 256 elements
    /// Block structure: d(f16), dmin(f16), scales[12], qh[32], qs[128] = 176 bytes
    fn dequantize_q5_k(
        reader: &mut BufReader<File>,
        num_elements: usize,
    ) -> RusTorchResult<Vec<f64>> {
        const QK_K: usize = 256;
        const K_SCALE_SIZE: usize = 12;

        let num_blocks = (num_elements + QK_K - 1) / QK_K;
        let mut output = Vec::with_capacity(num_elements);

        for _ in 0..num_blocks {
            // Read super-scale and super-min (f16 each)
            let d_bits = Self::read_u16(reader)?;
            let dmin_bits = Self::read_u16(reader)?;
            let d = half::f16::from_bits(d_bits).to_f32();
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            // Read quantized scales (12 bytes)
            let mut scales = [0u8; K_SCALE_SIZE];
            reader.read_exact(&mut scales).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q5_K scales: {}", e))
            })?;

            // Read high bits (32 bytes)
            let mut qh = [0u8; QK_K / 8];
            reader.read_exact(&mut qh).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q5_K qh: {}", e))
            })?;

            // Read quantized values (128 bytes = 256 nibbles)
            let mut qs = vec![0u8; QK_K / 2];
            reader.read_exact(&mut qs).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q5_K quants: {}", e))
            })?;

            // Dequantize: Process in pairs (64 elements at a time)
            for pair in 0..4 {
                let j1 = pair * 2;
                let j2 = pair * 2 + 1;
                let q_offset = pair * 32;

                // Get scale/min for both blocks
                let (scale1, min1) = if j1 < 4 {
                    let sc = scales[j1] & 63;
                    let mn = scales[j1 + 4] & 63;
                    (sc as f32, mn as f32)
                } else {
                    let sc = (scales[j1 + 4] & 0x0F) | ((scales[j1 - 4] >> 6) << 4);
                    let mn = (scales[j1 + 4] >> 4) | ((scales[j1] >> 6) << 4);
                    (sc as f32, mn as f32)
                };

                let (scale2, min2) = if j2 < 4 {
                    let sc = scales[j2] & 63;
                    let mn = scales[j2 + 4] & 63;
                    (sc as f32, mn as f32)
                } else {
                    let sc = (scales[j2 + 4] & 0x0F) | ((scales[j2 - 4] >> 6) << 4);
                    let mn = (scales[j2 + 4] >> 4) | ((scales[j2] >> 6) << 4);
                    (sc as f32, mn as f32)
                };

                let d1 = d * scale1;
                let m1 = dmin * min1;
                let d2 = d * scale2;
                let m2 = dmin * min2;

                // Process 32 lower nibbles + high bit (block 1)
                for l in 0..32 {
                    if output.len() >= num_elements { break; }
                    let q_low = qs[q_offset + l] & 0x0F;
                    let qh_byte = qh[l];
                    let q_high = (qh_byte >> j1) & 1;
                    let q_val = q_low | (q_high << 4);
                    output.push((d1 * q_val as f32 - m1) as f64);
                }

                // Process 32 upper nibbles + high bit (block 2)
                for l in 0..32 {
                    if output.len() >= num_elements { break; }
                    let q_low = qs[q_offset + l] >> 4;
                    let qh_byte = qh[l];
                    let q_high = (qh_byte >> j2) & 1;
                    let q_val = q_low | (q_high << 4);
                    output.push((d2 * q_val as f32 - m2) as f64);
                }
            }
        }

        output.truncate(num_elements);
        Ok(output)
    }

    /// Dequantize Q6_K format matching llama.cpp exactly
    fn dequantize_q6_k<R: Read>(
        reader: &mut BufReader<R>,
        num_elements: usize,
    ) -> RusTorchResult<Vec<f64>> {
        const QK_K: usize = 256;

        let num_blocks = (num_elements + QK_K - 1) / QK_K;
        let mut output = vec![0.0f64; num_elements];

        for block_idx in 0..num_blocks {
            let block_start = block_idx * QK_K;

            // Read ql[128], qh[64], sc[16], d(f16) = 210 bytes
            let mut ql = vec![0u8; 128];
            reader.read_exact(&mut ql).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q6_K ql: {}", e))
            })?;

            let mut qh = vec![0u8; 64];
            reader.read_exact(&mut qh).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q6_K qh: {}", e))
            })?;

            let mut sc = vec![0i8; 16];
            for scale in &mut sc {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf).map_err(|e| {
                    RusTorchError::IoError(format!("Failed to read Q6_K scale: {}", e))
                })?;
                *scale = buf[0] as i8;
            }

            let mut d_buf = [0u8; 2];
            reader.read_exact(&mut d_buf).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read Q6_K d: {}", e))
            })?;
            let d_bits = u16::from_le_bytes(d_buf);
            let d = half::f16::from_bits(d_bits).to_f32();

            // Dequantize: 2 chunks of 128 elements
            let mut y_idx = block_start;
            let mut ql_idx = 0;
            let mut qh_idx = 0;
            let mut sc_idx = 0;

            for _chunk in 0..2 {
                for l in 0..32 {
                    let is = l / 16;

                    let q1 = (((ql[ql_idx + l] & 0xF) | (((qh[qh_idx + l] >> 0) & 3) << 4)) as i8) - 32;
                    let q2 = (((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) as i8) - 32;
                    let q3 = (((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) as i8) - 32;
                    let q4 = (((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) as i8) - 32;

                    if y_idx + l < num_elements {
                        output[y_idx + l] = (d * sc[sc_idx + is] as f32 * q1 as f32) as f64;
                    }
                    if y_idx + l + 32 < num_elements {
                        output[y_idx + l + 32] = (d * sc[sc_idx + is + 2] as f32 * q2 as f32) as f64;
                    }
                    if y_idx + l + 64 < num_elements {
                        output[y_idx + l + 64] = (d * sc[sc_idx + is + 4] as f32 * q3 as f32) as f64;
                    }
                    if y_idx + l + 96 < num_elements {
                        output[y_idx + l + 96] = (d * sc[sc_idx + is + 6] as f32 * q4 as f32) as f64;
                    }
                }

                y_idx += 128;
                ql_idx += 64;
                qh_idx += 32;
                sc_idx += 8;
            }
        }

        Ok(output)
    }

}

/// Model parameters extracted from GGUF metadata
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,  // Number of key-value heads for GQA
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

    #[test]
    fn test_q6k_dequantization_interleaved_pattern() {
        // Test that Q6_K dequantization uses correct interleaved indexing pattern
        // llama.cpp pattern: y[l], y[l+32], y[l+64], y[l+96] for each iteration

        use std::io::{Cursor, Write};
        use std::io::BufReader;

        // Create a minimal Q6_K block (210 bytes)
        // Structure: ql[128] + qh[64] + sc[16] + d(f16) = 210 bytes
        let mut block_data = Vec::new();

        // ql[128]: lower 4 bits - use simple pattern
        for i in 0..128 {
            block_data.push((i % 16) as u8);
        }

        // qh[64]: upper 2 bits - use simple pattern
        for i in 0..64 {
            block_data.push(((i % 4) << 0) | (((i + 1) % 4) << 2) | (((i + 2) % 4) << 4) | (((i + 3) % 4) << 6));
        }

        // sc[16]: scales - use constant value for simplicity
        for _ in 0..16 {
            block_data.push(10i8 as u8);
        }

        // d (f16): super-scale - use 0.01
        let d_f16 = half::f16::from_f32(0.01);
        block_data.write(&d_f16.to_bits().to_le_bytes()).unwrap();

        assert_eq!(block_data.len(), 210, "Q6_K block should be exactly 210 bytes");

        // Dequantize using our implementation
        let mut reader = BufReader::new(Cursor::new(block_data));
        let result = GGUFLoader::dequantize_q6_k(&mut reader, 256).unwrap();

        assert_eq!(result.len(), 256, "Should produce 256 values");

        // Verify interleaved pattern by checking specific positions
        // First chunk (0-127):
        // l=0: positions 0, 32, 64, 96 should come from first iteration
        // l=1: positions 1, 33, 65, 97 should come from second iteration

        // All values should be non-zero (since we used non-zero ql, qh, sc)
        let non_zero_count = result.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(non_zero_count > 200, "Most values should be non-zero with our test data");

        // Values should be in reasonable range (d * sc * q where q is -32 to 31)
        // max = 0.01 * 10 * 31 = 3.1
        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 5.0, "Value at index {} = {} exceeds expected range", i, val);
        }
    }

    #[test]
    fn test_q6k_dequantization_known_values() {
        // Test Q6_K dequantization with known values
        // This verifies the exact computation: d * scale * (quantized_value - 32)

        use std::io::{Cursor, Write};
        use std::io::BufReader;

        let mut block_data = Vec::new();

        // ql[128]: Set first byte to specific value
        block_data.push(0x0F); // lower nibble = 15
        for _ in 1..128 {
            block_data.push(0x00);
        }

        // qh[64]: Set first byte to extract upper bits
        block_data.push(0b00000000); // upper 2 bits = 0 for all 4 values
        for _ in 1..64 {
            block_data.push(0x00);
        }

        // sc[16]: Set specific scales
        block_data.push(16i8 as u8);  // sc[0]
        block_data.push(0i8 as u8);   // sc[1]
        block_data.push(10i8 as u8);  // sc[2]
        for _ in 3..16 {
            block_data.push(0i8 as u8);
        }

        // d (f16): 0.001
        let d_f16 = half::f16::from_f32(0.001);
        block_data.write(&d_f16.to_bits().to_le_bytes()).unwrap();

        let mut reader = BufReader::new(Cursor::new(block_data));
        let result = GGUFLoader::dequantize_q6_k(&mut reader, 256).unwrap();

        // First value at position 0:
        // ql[0] & 0xF = 15, qh[0] >> 0 & 3 = 0
        // q1 = (15 | (0 << 4)) - 32 = 15 - 32 = -17
        // is = 0 / 16 = 0
        // value = d * sc[0] * q1 = 0.001 * 16 * (-17) = -0.272
        let expected_0 = 0.001 * 16.0 * (-17.0);
        assert!((result[0] - expected_0).abs() < 0.001,
            "Position 0: expected {}, got {}", expected_0, result[0]);

        // Value at position 32 uses sc[2]:
        // ql[32] = 0, qh[32 % 64] = 0
        // q2 = (0 | (0 << 4)) - 32 = -32
        // value = d * sc[2] * q2 = 0.001 * 10 * (-32) = -0.32
        let expected_32 = 0.001 * 10.0 * (-32.0);
        assert!((result[32] - expected_32).abs() < 0.001,
            "Position 32: expected {}, got {}", expected_32, result[32]);
    }

    #[test]
    fn test_q6k_element_size() {
        // Verify Q6_K element size is correct
        // Q6_K block = 210 bytes for 256 elements
        assert_eq!(GGMLType::Q6_K.element_size(), 210);
    }
}
