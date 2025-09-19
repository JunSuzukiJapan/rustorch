//! Various serialization formats support for Phase 9
//! フェーズ9用各種シリアライゼーション形式サポート

use super::core::{Loadable, Saveable, SerializationError, SerializationResult};
use crate::tensor::Tensor;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// PyTorch-compatible format handler
/// PyTorch互換形式ハンドラー
pub struct PyTorchFormat;

impl PyTorchFormat {
    /// Load PyTorch .pth file
    /// PyTorchの.pthファイルを読み込み
    pub fn load_pth<P: AsRef<Path>, T: Float + 'static>(
        path: P,
    ) -> SerializationResult<HashMap<String, Tensor<T>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic header to detect pickle format
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != [0x80, 0x02, b'q', 0x00] && magic != [0x80, 0x03, b'q', 0x00] {
            return Err(SerializationError::FormatError(
                "Invalid PyTorch pickle format".to_string(),
            ));
        }

        // For now, return placeholder - real implementation would use pickle parser
        Err(SerializationError::UnsupportedOperation(
            "PyTorch pickle format parsing not yet implemented".to_string(),
        ))
    }

    /// Save in PyTorch-compatible format
    /// PyTorch互換形式で保存
    pub fn save_pth<P: AsRef<Path>, T: Float + 'static>(
        tensors: &HashMap<String, Tensor<T>>,
        path: P,
    ) -> SerializationResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Simple binary format without serde
        writer.write_all(b"PYTORCH1")?; // Magic header
        writer.write_all(&(tensors.len() as u32).to_le_bytes())?;

        for (name, tensor) in tensors {
            // Write tensor name
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // Write tensor shape
            let shape = tensor.shape();
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Write tensor data
            if let Some(data_slice) = tensor.data.as_slice() {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data_slice.as_ptr() as *const u8,
                        data_slice.len() * std::mem::size_of::<T>(),
                    )
                };
                writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
                writer.write_all(bytes)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

/// ONNX format handler
/// ONNX形式ハンドラー
pub struct ONNXFormat;

impl ONNXFormat {
    /// Export to ONNX format
    /// ONNX形式にエクスポート
    pub fn export_onnx<P: AsRef<Path>, T: Float>(
        _model: &dyn Saveable,
        _path: P,
    ) -> SerializationResult<()> {
        // Placeholder for ONNX export
        Err(SerializationError::UnsupportedOperation(
            "ONNX export not yet implemented".to_string(),
        ))
    }

    /// Import from ONNX format
    /// ONNX形式からインポート
    pub fn import_onnx<P: AsRef<Path>, T: Float>(
        _path: P,
    ) -> SerializationResult<HashMap<String, Tensor<T>>> {
        // Placeholder for ONNX import
        Err(SerializationError::UnsupportedOperation(
            "ONNX import not yet implemented".to_string(),
        ))
    }
}

/// SafeTensors format handler
/// SafeTensors形式ハンドラー
pub struct SafeTensorsFormat;

impl SafeTensorsFormat {
    /// Save in SafeTensors format
    /// SafeTensors形式で保存
    pub fn save_safetensors<P: AsRef<Path>, T: Float + 'static>(
        tensors: &HashMap<String, Tensor<T>>,
        path: P,
    ) -> SerializationResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Create header with tensor metadata
        let mut header_data = HashMap::new();
        let mut current_offset = 0u64;

        for (name, tensor) in tensors {
            let data_size = tensor.numel() * std::mem::size_of::<T>();
            header_data.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": Self::get_dtype_string::<T>(),
                    "shape": tensor.shape(),
                    "data_offsets": [current_offset, current_offset + data_size as u64]
                }),
            );
            current_offset += data_size as u64;
        }

        let header_json = serde_json::to_string(&header_data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        // Write header size and header
        let header_size = header_json.len() as u64;
        writer.write_all(&header_size.to_le_bytes())?;
        writer.write_all(header_json.as_bytes())?;

        // Write tensor data
        for (_, tensor) in tensors {
            if let Some(data_slice) = tensor.data.as_slice() {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data_slice.as_ptr() as *const u8,
                        data_slice.len() * std::mem::size_of::<T>(),
                    )
                };
                writer.write_all(bytes)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from SafeTensors format
    /// SafeTensors形式から読み込み
    pub fn load_safetensors<P: AsRef<Path>, T: Float + 'static>(
        path: P,
    ) -> SerializationResult<HashMap<String, Tensor<T>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header size
        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes);

        // Read header
        let mut header_data = vec![0u8; header_size as usize];
        reader.read_exact(&mut header_data)?;
        let header_str = String::from_utf8(header_data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        let header: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        // Read all tensor data
        let mut tensor_data = Vec::new();
        reader.read_to_end(&mut tensor_data)?;

        // Parse tensors
        let mut tensors = HashMap::new();
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }

            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| SerializationError::FormatError("Invalid shape".to_string()))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();

            let data_offsets = meta["data_offsets"].as_array().ok_or_else(|| {
                SerializationError::FormatError("Invalid data_offsets".to_string())
            })?;
            let start_offset = data_offsets[0].as_u64().unwrap_or(0) as usize;
            let end_offset = data_offsets[1].as_u64().unwrap_or(0) as usize;

            if end_offset > tensor_data.len() {
                return Err(SerializationError::CorruptionError(
                    "Data offset out of bounds".to_string(),
                ));
            }

            let tensor_bytes = &tensor_data[start_offset..end_offset];
            let float_data = unsafe {
                std::slice::from_raw_parts(
                    tensor_bytes.as_ptr() as *const T,
                    tensor_bytes.len() / std::mem::size_of::<T>(),
                )
            };

            let tensor = Tensor::from_vec(float_data.to_vec(), shape);
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    fn get_dtype_string<T: Float>() -> String {
        match std::mem::size_of::<T>() {
            4 => "F32".to_string(),
            8 => "F64".to_string(),
            _ => "UNKNOWN".to_string(),
        }
    }
}

/// HuggingFace format handler
/// HuggingFace形式ハンドラー
pub struct HuggingFaceFormat;

impl HuggingFaceFormat {
    /// Save in HuggingFace format
    /// HuggingFace形式で保存
    pub fn save_hf<P: AsRef<Path>, T: Float + 'static>(
        model_state: &HashMap<String, Tensor<T>>,
        config: &HashMap<String, String>,
        path: P,
    ) -> SerializationResult<()> {
        let model_dir = path.as_ref();
        std::fs::create_dir_all(model_dir)?;

        // Save model weights in SafeTensors format
        let weights_path = model_dir.join("model.safetensors");
        SafeTensorsFormat::save_safetensors(model_state, weights_path)?;

        // Save config.json
        let config_path = model_dir.join("config.json");
        let config_file = File::create(config_path)?;
        serde_json::to_writer_pretty(config_file, config)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        Ok(())
    }

    /// Load from HuggingFace format
    /// HuggingFace形式から読み込み
    pub fn load_hf<P: AsRef<Path>, T: Float + 'static>(
        path: P,
    ) -> SerializationResult<(HashMap<String, Tensor<T>>, HashMap<String, String>)> {
        let model_dir = path.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_file = File::open(config_path)?;
        let config: HashMap<String, String> = serde_json::from_reader(config_file)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let tensors = SafeTensorsFormat::load_safetensors(weights_path)?;

        Ok((tensors, config))
    }
}

/// Legacy format conversion utilities
/// レガシー形式変換ユーティリティ
pub mod conversion {
    use super::*;

    /// Convert from older RusTorch format
    /// 古いRusTorch形式から変換
    pub fn upgrade_legacy_format<P: AsRef<Path>, T: Float + 'static>(
        path: P,
    ) -> SerializationResult<HashMap<String, Tensor<T>>> {
        // Placeholder for legacy format conversion
        Err(SerializationError::UnsupportedOperation(
            "Legacy format conversion not yet implemented".to_string(),
        ))
    }

    /// Convert between different precision formats
    /// 異なる精度形式間で変換
    pub fn convert_precision<F: Float + 'static, T: Float + 'static>(
        tensors: &HashMap<String, Tensor<F>>,
    ) -> HashMap<String, Tensor<T>>
    where
        F: Into<f64> + Copy,
        T: From<f64>,
    {
        let mut converted = HashMap::new();

        for (name, tensor) in tensors {
            // Use to_owned() to get owned data instead of as_slice() which may return None for non-contiguous arrays
            let data = tensor.data.to_owned();
            let (flat_data, _offset) = data.into_raw_vec_and_offset();

            let converted_data: Vec<T> = flat_data
                .iter()
                .map(|&x| <T as From<f64>>::from(x.into()))
                .collect();
            let converted_tensor = Tensor::from_vec(converted_data, tensor.shape().to_vec());
            converted.insert(name.clone(), converted_tensor);
        }

        converted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_format() {
        let mut tensors = HashMap::new();
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        tensors.insert("test_tensor".to_string(), tensor);

        let temp_path = std::env::temp_dir().join("test_safetensors.st");

        // Test save
        assert!(SafeTensorsFormat::save_safetensors(&tensors, &temp_path).is_ok());

        // Test load
        let loaded_tensors = SafeTensorsFormat::load_safetensors::<_, f32>(&temp_path).unwrap();
        assert!(loaded_tensors.contains_key("test_tensor"));

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_huggingface_format() {
        let mut tensors = HashMap::new();
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        tensors.insert("weight".to_string(), tensor);

        let mut config = HashMap::new();
        config.insert("model_type".to_string(), "test_model".to_string());

        let temp_dir = std::env::temp_dir().join("test_hf_model");

        // Test save
        assert!(HuggingFaceFormat::save_hf(&tensors, &config, &temp_dir).is_ok());

        // Test load
        let (loaded_tensors, loaded_config) =
            HuggingFaceFormat::load_hf::<_, f32>(&temp_dir).unwrap();
        assert!(loaded_tensors.contains_key("weight"));
        assert_eq!(loaded_config.get("model_type").unwrap(), "test_model");

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_precision_conversion() {
        let mut tensors_f32 = HashMap::new();
        let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.5, 3.7], vec![3]);
        tensors_f32.insert("test".to_string(), tensor_f32);

        let tensors_f64: HashMap<String, Tensor<f64>> = conversion::convert_precision(&tensors_f32);

        assert!(tensors_f64.contains_key("test"));

        let converted_tensor = tensors_f64.get("test").unwrap();
        assert_eq!(converted_tensor.shape(), &[3]);

        if let Some(data) = converted_tensor.data.as_slice() {
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.5).abs() < 1e-6);
        }
    }
}
