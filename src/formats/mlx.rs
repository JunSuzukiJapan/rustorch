//! MLX format support for RusTorch (Apple's MLX framework)
//! RusTorch向けMLX形式サポート（AppleのMLXフレームワーク）

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// MLX tensor metadata
/// MLXテンソルメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLXTensorInfo {
    /// Data type (F32, F16, I32, I64, etc.)
    /// データ型
    pub dtype: String,
    /// Tensor shape
    /// テンソル形状
    pub shape: Vec<usize>,
    /// Data offsets in file [start, end]
    /// ファイル内のデータオフセット [開始, 終了]
    pub data_offsets: Vec<usize>,
}

/// MLX model metadata
/// MLXモデルメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLXModelMetadata {
    /// Vocabulary size
    /// 語彙サイズ
    pub vocab_size: Option<usize>,
    /// Hidden layer size
    /// 隠れ層サイズ
    pub hidden_size: Option<usize>,
    /// Number of hidden layers
    /// 隠れ層の数
    pub num_hidden_layers: Option<usize>,
    /// Number of attention heads
    /// アテンションヘッド数
    pub num_attention_heads: Option<usize>,
    /// Maximum position embeddings
    /// 最大位置埋め込み
    pub max_position_embeddings: Option<usize>,
}

/// MLX model loader for Apple's MLX framework
/// AppleのMLXフレームワーク用MLXモデルローダー
pub struct MLXLoader {
    /// Tensor metadata
    /// テンソルメタデータ
    tensors_info: HashMap<String, MLXTensorInfo>,
    /// Model metadata
    /// モデルメタデータ
    model_metadata: Option<MLXModelMetadata>,
    /// Raw file data
    /// 生ファイルデータ
    file_data: Vec<u8>,
    /// Data offset in file
    /// ファイル内のデータオフセット
    data_offset: usize,
}

impl MLXLoader {
    /// Load MLX model from file
    /// ファイルからMLXモデルを読み込み
    pub fn from_file<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| RusTorchError::IoError(format!("Failed to open MLX file: {}", e)))?;

        let mut file_data = Vec::new();
        file.read_to_end(&mut file_data)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read MLX file: {}", e)))?;

        // Parse MLX header (first 8 bytes contain metadata size)
        if file_data.len() < 8 {
            return Err(RusTorchError::ParseError(format!(
                "MLX file too small: {} bytes",
                file_data.len()
            )));
        }

        let metadata_size = u64::from_le_bytes([
            file_data[0],
            file_data[1],
            file_data[2],
            file_data[3],
            file_data[4],
            file_data[5],
            file_data[6],
            file_data[7],
        ]) as usize;

        if file_data.len() < 8 + metadata_size {
            return Err(RusTorchError::ParseError(format!(
                "MLX file corrupted: metadata size {} exceeds file size {}",
                metadata_size,
                file_data.len()
            )));
        }

        // Parse metadata JSON
        let metadata_bytes = &file_data[8..8 + metadata_size];
        let metadata_json: serde_json::Value = serde_json::from_slice(metadata_bytes)
            .map_err(|e| RusTorchError::DeserializationError(format!("Failed to parse MLX metadata: {}", e)))?;

        let mut tensors_info = HashMap::new();
        let mut model_metadata = None;

        // Extract tensor information from metadata
        if let Some(metadata_obj) = metadata_json.as_object() {
            for (name, meta) in metadata_obj {
                if name == "__metadata__" {
                    // Extract model metadata
                    if let Some(meta_obj) = meta.as_object() {
                        model_metadata = Some(MLXModelMetadata {
                            vocab_size: meta_obj
                                .get("vocab_size")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            hidden_size: meta_obj
                                .get("hidden_size")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            num_hidden_layers: meta_obj
                                .get("num_hidden_layers")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            num_attention_heads: meta_obj
                                .get("num_attention_heads")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            max_position_embeddings: meta_obj
                                .get("max_position_embeddings")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                        });
                    }
                    continue;
                }

                // Extract tensor info
                let dtype = meta["dtype"]
                    .as_str()
                    .ok_or_else(|| RusTorchError::ParseError(format!("Missing dtype for tensor {}", name)))?
                    .to_string();

                let shape = meta["shape"]
                    .as_array()
                    .ok_or_else(|| RusTorchError::ParseError(format!("Missing shape for tensor {}", name)))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .ok_or_else(|| RusTorchError::ParseError("Invalid shape dimension".to_string()))
                            .map(|v| v as usize)
                    })
                    .collect::<RusTorchResult<Vec<usize>>>()?;

                let data_offsets = meta["data_offsets"]
                    .as_array()
                    .ok_or_else(|| RusTorchError::ParseError(format!("Missing data_offsets for tensor {}", name)))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .ok_or_else(|| RusTorchError::ParseError("Invalid data offset".to_string()))
                            .map(|v| v as usize)
                    })
                    .collect::<RusTorchResult<Vec<usize>>>()?;

                if data_offsets.len() != 2 {
                    return Err(RusTorchError::ParseError(format!(
                        "Invalid data_offsets length for tensor {}: expected 2, got {}",
                        name,
                        data_offsets.len()
                    )));
                }

                tensors_info.insert(
                    name.clone(),
                    MLXTensorInfo {
                        dtype,
                        shape,
                        data_offsets,
                    },
                );
            }
        }

        let data_offset = 8 + metadata_size;

        Ok(Self {
            tensors_info,
            model_metadata,
            file_data,
            data_offset,
        })
    }

    /// Get all tensor names
    /// 全テンソル名を取得
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors_info.keys().collect()
    }

    /// Get tensor info
    /// テンソル情報を取得
    pub fn tensor_info(&self, name: &str) -> Option<&MLXTensorInfo> {
        self.tensors_info.get(name)
    }

    /// Get model metadata
    /// モデルメタデータを取得
    pub fn model_metadata(&self) -> Option<&MLXModelMetadata> {
        self.model_metadata.as_ref()
    }

    /// Load tensor by name
    /// 名前でテンソルを読み込み
    pub fn load_tensor<T: Float + 'static>(&self, name: &str) -> RusTorchResult<Tensor<T>> {
        let info = self.tensors_info.get(name).ok_or_else(|| {
            RusTorchError::ParseError(format!("Tensor '{}' not found in MLX file", name))
        })?;

        let start = self.data_offset + info.data_offsets[0];
        let end = self.data_offset + info.data_offsets[1];

        if end > self.file_data.len() {
            return Err(RusTorchError::ParseError(format!(
                "Tensor '{}' data range [{}, {}) exceeds file size {}",
                name,
                start,
                end,
                self.file_data.len()
            )));
        }

        let tensor_data = &self.file_data[start..end];

        Self::parse_tensor_data(tensor_data, &info.shape, &info.dtype)
    }

    /// Parse tensor data based on dtype
    /// dtypeに基づいてテンソルデータを解析
    fn parse_tensor_data<T: Float + 'static>(
        data: &[u8],
        shape: &[usize],
        dtype: &str,
    ) -> RusTorchResult<Tensor<T>> {
        match dtype {
            "F32" | "float32" => {
                if data.len() % 4 != 0 {
                    return Err(RusTorchError::ParseError(
                        "Invalid F32 data length".to_string(),
                    ));
                }
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let converted: Vec<T> = values
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect();
                Ok(Tensor::from_vec(converted, shape.to_vec()))
            }
            "F16" | "float16" => {
                if data.len() % 2 != 0 {
                    return Err(RusTorchError::ParseError(
                        "Invalid F16 data length".to_string(),
                    ));
                }
                let values: Vec<T> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let f16_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                        T::from(f16_val.to_f32()).unwrap()
                    })
                    .collect();
                Ok(Tensor::from_vec(values, shape.to_vec()))
            }
            "I32" | "int32" => {
                if data.len() % 4 != 0 {
                    return Err(RusTorchError::ParseError(
                        "Invalid I32 data length".to_string(),
                    ));
                }
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let converted: Vec<T> = values
                    .iter()
                    .map(|&v| T::from(v as f64).unwrap())
                    .collect();
                Ok(Tensor::from_vec(converted, shape.to_vec()))
            }
            "I64" | "int64" => {
                if data.len() % 8 != 0 {
                    return Err(RusTorchError::ParseError(
                        "Invalid I64 data length".to_string(),
                    ));
                }
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                let converted: Vec<T> = values
                    .iter()
                    .map(|&v| T::from(v as f64).unwrap())
                    .collect();
                Ok(Tensor::from_vec(converted, shape.to_vec()))
            }
            _ => Err(RusTorchError::ParseError(format!(
                "Unsupported MLX dtype: {}",
                dtype
            ))),
        }
    }

    /// Load all tensors
    /// 全テンソルを読み込み
    pub fn load_all_tensors<T: Float + 'static>(
        &self,
    ) -> RusTorchResult<HashMap<String, Tensor<T>>> {
        let mut tensors = HashMap::new();
        for name in self.tensor_names() {
            let tensor = self.load_tensor(name)?;
            tensors.insert(name.clone(), tensor);
        }
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_model_metadata() {
        let metadata = MLXModelMetadata {
            vocab_size: Some(32000),
            hidden_size: Some(4096),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            max_position_embeddings: Some(2048),
        };

        assert_eq!(metadata.vocab_size, Some(32000));
        assert_eq!(metadata.hidden_size, Some(4096));
    }

    #[test]
    fn test_mlx_tensor_info() {
        let info = MLXTensorInfo {
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            data_offsets: vec![0, 800],
        };

        assert_eq!(info.dtype, "F32");
        assert_eq!(info.shape, vec![10, 20]);
        assert_eq!(info.data_offsets, vec![0, 800]);
    }

    #[test]
    fn test_parse_f32_tensor_data() {
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0 in little-endian F32
            0x00, 0x00, 0x00, 0x40, // 2.0 in little-endian F32
            0x00, 0x00, 0x40, 0x40, // 3.0 in little-endian F32
        ];
        let shape = vec![3];
        let tensor: Tensor<f32> =
            MLXLoader::parse_tensor_data(&data, &shape, "F32").unwrap();

        assert_eq!(tensor.shape(), &[3]);
        let tensor_data = tensor.data.as_slice().unwrap();
        assert!((tensor_data[0] - 1.0).abs() < 1e-6);
        assert!((tensor_data[1] - 2.0).abs() < 1e-6);
        assert!((tensor_data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_i32_tensor_data() {
        let data: Vec<u8> = vec![
            0x01, 0x00, 0x00, 0x00, // 1 in little-endian I32
            0x02, 0x00, 0x00, 0x00, // 2 in little-endian I32
        ];
        let shape = vec![2];
        let tensor: Tensor<f64> =
            MLXLoader::parse_tensor_data(&data, &shape, "I32").unwrap();

        assert_eq!(tensor.shape(), &[2]);
        let tensor_data = tensor.data.as_slice().unwrap();
        assert!((tensor_data[0] - 1.0).abs() < 1e-6);
        assert!((tensor_data[1] - 2.0).abs() < 1e-6);
    }
}
