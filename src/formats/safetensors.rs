//! Safetensors format support for RusTorch
//! RusTorch向けSafetensors形式サポート

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use memmap2::Mmap;
use num_traits::Float;
use safetensors::{tensor::Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Safetensors model loader for RusTorch
/// RusTorch向けSafetensorsモデルローダー
pub struct SafetensorsLoader {
    /// Memory-mapped file to keep data alive
    /// データを保持するためのメモリマップファイル
    _mmap: Mmap, // Keep mmap alive
    /// Parsed safetensors data
    /// 解析されたsafetensorsデータ
    tensors: SafeTensors<'static>,
}

impl SafetensorsLoader {
    /// Load a safetensors file
    /// safetensorsファイルを読み込み
    pub fn from_file<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| RusTorchError::IoError(format!("Failed to open safetensors file: {}", e)))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| RusTorchError::IoError(format!("Failed to mmap file: {}", e)))?;

        // SAFETY: We keep the mmap alive for the lifetime of SafetensorsLoader
        // The transmute extends the lifetime to 'static, which is safe because
        // we own the mmap and keep it alive
        let tensors = unsafe {
            let data: &'static [u8] = std::mem::transmute(mmap.as_ref());
            SafeTensors::deserialize(data)
                .map_err(|e| RusTorchError::DeserializationError(format!("Safetensors parse error: {}", e)))?
        };

        Ok(Self {
            _mmap: mmap,
            tensors,
        })
    }

    /// Get tensor names
    /// テンソル名を取得
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names().iter().map(|s| s.as_str()).collect()
    }

    /// Get tensor metadata
    /// テンソルメタデータを取得
    pub fn tensor_info(&self, name: &str) -> RusTorchResult<TensorInfo> {
        let tensor = self.tensors.tensor(name)
            .map_err(|e| RusTorchError::tensor_op(format!("Tensor not found: {}", e)))?;

        Ok(TensorInfo {
            name: name.to_string(),
            shape: tensor.shape().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
        })
    }

    /// Load a tensor by name
    /// 名前でテンソルを読み込み
    pub fn load_tensor<T: Float + 'static>(
        &self,
        name: &str,
    ) -> RusTorchResult<Tensor<T>> {
        let tensor = self.tensors.tensor(name)
            .map_err(|e| RusTorchError::tensor_op(format!("Tensor '{}' not found: {}", name, e)))?;
        let shape = tensor.shape().to_vec();

        // Convert data based on tensor dtype
        let data = match tensor.dtype() {
            Dtype::F32 => {
                let data_slice = tensor.data();
                let f32_data: &[f32] = bytemuck::cast_slice(data_slice);
                f32_data.iter().map(|&x| T::from(x).unwrap()).collect()
            }
            Dtype::F64 => {
                let data_slice = tensor.data();
                let f64_data: &[f64] = bytemuck::cast_slice(data_slice);
                f64_data.iter().map(|&x| T::from(x).unwrap()).collect()
            }
            Dtype::F16 => {
                let data_slice = tensor.data();
                // F16 is stored as u16 little-endian
                let u16_data: &[u16] = bytemuck::cast_slice(data_slice);
                u16_data
                    .iter()
                    .map(|&x| {
                        let f32_val = half::f16::from_bits(x).to_f32();
                        T::from(f32_val).unwrap()
                    })
                    .collect()
            }
            Dtype::I32 => {
                let data_slice = tensor.data();
                let i32_data: &[i32] = bytemuck::cast_slice(data_slice);
                i32_data
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            Dtype::I64 => {
                let data_slice = tensor.data();
                let i64_data: &[i64] = bytemuck::cast_slice(data_slice);
                i64_data
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            Dtype::U8 => {
                let data_slice = tensor.data();
                data_slice
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            Dtype::I8 => {
                let data_slice = tensor.data();
                let i8_data: &[i8] = bytemuck::cast_slice(data_slice);
                i8_data
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            Dtype::U16 => {
                let data_slice = tensor.data();
                let u16_data: &[u16] = bytemuck::cast_slice(data_slice);
                u16_data
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            Dtype::I16 => {
                let data_slice = tensor.data();
                let i16_data: &[i16] = bytemuck::cast_slice(data_slice);
                i16_data
                    .iter()
                    .map(|&x| T::from(x as f64).unwrap())
                    .collect()
            }
            _ => {
                return Err(RusTorchError::tensor_op(format!(
                    "Unsupported dtype for tensor '{}': {:?}",
                    name,
                    tensor.dtype()
                )))
            }
        };

        Ok(Tensor::from_vec(data, shape))
    }

    /// Load all tensors
    /// 全テンソルを読み込み
    pub fn load_all_tensors<T: Float + 'static>(
        &self,
    ) -> RusTorchResult<HashMap<String, Tensor<T>>> {
        let mut tensors = HashMap::new();

        for name in self.tensor_names() {
            let tensor = self.load_tensor::<T>(name)?;
            tensors.insert(name.to_string(), tensor);
        }

        Ok(tensors)
    }
}

/// Tensor metadata information
/// テンソルメタデータ情報
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Safetensors model saver for RusTorch
/// RusTorch向けSafetensorsモデルセーバー
pub struct SafetensorsSaver;

impl SafetensorsSaver {
    /// Save tensors to safetensors format
    /// テンソルをsafetensors形式で保存
    pub fn save_to_file<T: Float + 'static, P: AsRef<Path>>(
        tensors: &HashMap<String, Tensor<T>>,
        path: P,
    ) -> RusTorchResult<()> {
        use safetensors::tensor::TensorView;

        // Pre-convert all tensor data to avoid borrowing issues
        let mut tensor_data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        for (name, tensor) in tensors {
            let shape = tensor.shape().to_vec();
            let data_f32: Vec<f32> = tensor.data.iter().map(|&x| x.to_f32().unwrap()).collect();
            let bytes = bytemuck::cast_slice(&data_f32).to_vec();
            tensor_data.push((name.clone(), shape, bytes));
        }

        let mut data_map = HashMap::new();

        for (name, shape, bytes) in &tensor_data {
            let dtype = Dtype::F32;
            let tensor_view =
                TensorView::new(dtype, shape.clone(), bytes.as_slice())
                    .map_err(|e| RusTorchError::SerializationError(format!("TensorView creation error: {}", e)))?;
            data_map.insert(name.clone(), tensor_view);
        }

        // Create safetensors data
        let safetensor_data = safetensors::serialize(&data_map, &None)
            .map_err(|e| RusTorchError::SerializationError(format!("Safetensors serialization error: {}", e)))?;

        // Write to file
        std::fs::write(path.as_ref(), safetensor_data)
            .map_err(|e| RusTorchError::IoError(format!("Failed to write safetensors file: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_safetensors_save_load() {
        let mut tensors = HashMap::new();
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tensors.insert("test_tensor".to_string(), tensor);

        // Save to temporary file
        let temp_file = NamedTempFile::new().unwrap();
        SafetensorsSaver::save_to_file(&tensors, temp_file.path()).unwrap();

        // Load back
        let loader = SafetensorsLoader::from_file(temp_file.path()).unwrap();
        let loaded_tensor: Tensor<f32> = loader.load_tensor("test_tensor").unwrap();

        assert_eq!(loaded_tensor.shape(), &[2, 2]);
        let expected_data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            loaded_tensor.data.iter().copied().collect::<Vec<_>>(),
            expected_data
        );
    }

    #[test]
    fn test_load_all_tensors() {
        let mut tensors = HashMap::new();
        let tensor1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::<f32>::from_vec(vec![3.0, 4.0, 5.0], vec![3]);
        tensors.insert("tensor1".to_string(), tensor1);
        tensors.insert("tensor2".to_string(), tensor2);

        let temp_file = NamedTempFile::new().unwrap();
        SafetensorsSaver::save_to_file(&tensors, temp_file.path()).unwrap();

        let loader = SafetensorsLoader::from_file(temp_file.path()).unwrap();
        let loaded_tensors: HashMap<String, Tensor<f32>> = loader.load_all_tensors().unwrap();

        assert_eq!(loaded_tensors.len(), 2);
        assert!(loaded_tensors.contains_key("tensor1"));
        assert!(loaded_tensors.contains_key("tensor2"));
    }

    #[test]
    fn test_tensor_info() {
        let mut tensors = HashMap::new();
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        tensors.insert("test".to_string(), tensor);

        let temp_file = NamedTempFile::new().unwrap();
        SafetensorsSaver::save_to_file(&tensors, temp_file.path()).unwrap();

        let loader = SafetensorsLoader::from_file(temp_file.path()).unwrap();
        let info = loader.tensor_info("test").unwrap();

        assert_eq!(info.name, "test");
        assert_eq!(info.shape, vec![3]);
        assert_eq!(info.dtype, "F32");
    }
}
