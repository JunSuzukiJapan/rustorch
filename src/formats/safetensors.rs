//! Safetensors format support for RusTorch
//! RusTorch向けSafetensors形式サポート

#[cfg(feature = "safetensors")]
use crate::tensor::Tensor;
#[cfg(feature = "safetensors")]
use safetensors::SafeTensors;
#[cfg(feature = "safetensors")]
use memmap2::Mmap;
#[cfg(feature = "safetensors")]
use std::collections::HashMap;
#[cfg(feature = "safetensors")]
use std::fs::File;
#[cfg(feature = "safetensors")]
use std::path::Path;
#[cfg(feature = "safetensors")]
use num_traits::Float;

#[cfg(feature = "safetensors")]
#[derive(Debug)]
pub enum SafetensorsError {
    IoError(std::io::Error),
    SafetensorsError(safetensors::SafeTensorError),
    ConversionError(String),
}

#[cfg(feature = "safetensors")]
impl From<std::io::Error> for SafetensorsError {
    fn from(error: std::io::Error) -> Self {
        SafetensorsError::IoError(error)
    }
}

#[cfg(feature = "safetensors")]
impl From<safetensors::SafeTensorError> for SafetensorsError {
    fn from(error: safetensors::SafeTensorError) -> Self {
        SafetensorsError::SafetensorsError(error)
    }
}

#[cfg(feature = "safetensors")]
impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetensorsError::IoError(e) => write!(f, "IO Error: {}", e),
            SafetensorsError::SafetensorsError(e) => write!(f, "Safetensors Error: {}", e),
            SafetensorsError::ConversionError(e) => write!(f, "Conversion Error: {}", e),
        }
    }
}

#[cfg(feature = "safetensors")]
impl std::error::Error for SafetensorsError {}

/// Safetensors model loader for RusTorch
/// RusTorch向けSafetensorsモデルローダー
#[cfg(feature = "safetensors")]
pub struct SafetensorsLoader {
    /// Parsed safetensors data
    /// 解析されたsafetensorsデータ
    tensors: SafeTensors<'static>,
    /// Memory-mapped file to keep data alive
    /// データを保持するためのメモリマップファイル
    _mmap: Mmap, // Keep mmap alive
}

#[cfg(feature = "safetensors")]
impl SafetensorsLoader {
    /// Load a safetensors file
    /// safetensorsファイルを読み込み
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, SafetensorsError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // SAFETY: We keep the mmap alive for the lifetime of SafetensorsLoader
        let tensors = SafeTensors::deserialize(&mmap)?;
        
        Ok(Self {
            tensors,
            _mmap: mmap,
        })
    }

    /// Get tensor names
    /// テンソル名を取得
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.names()
    }

    /// Load a tensor by name
    /// 名前でテンソルを読み込み
    pub fn load_tensor<T: Float + 'static>(&self, name: &str) -> Result<Tensor<T>, SafetensorsError> {
        let tensor = self.tensors.tensor(name)?;
        let shape = tensor.shape().to_vec();
        
        // Convert data based on tensor dtype
        let data = match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let data_slice = tensor.data();
                let f32_data: &[f32] = bytemuck::cast_slice(data_slice);
                f32_data.iter().map(|&x| T::from(x).unwrap()).collect()
            },
            safetensors::Dtype::F64 => {
                let data_slice = tensor.data();
                let f64_data: &[f64] = bytemuck::cast_slice(data_slice);
                f64_data.iter().map(|&x| T::from(x).unwrap()).collect()
            },
            safetensors::Dtype::I32 => {
                let data_slice = tensor.data();
                let i32_data: &[i32] = bytemuck::cast_slice(data_slice);
                i32_data.iter().map(|&x| T::from(x as f64).unwrap()).collect()
            },
            safetensors::Dtype::I64 => {
                let data_slice = tensor.data();
                let i64_data: &[i64] = bytemuck::cast_slice(data_slice);
                i64_data.iter().map(|&x| T::from(x as f64).unwrap()).collect()
            },
            _ => return Err(SafetensorsError::ConversionError(
                format!("Unsupported dtype: {:?}", tensor.dtype())
            )),
        };

        Ok(Tensor::from_vec(data, shape))
    }

    /// Load all tensors
    /// 全テンソルを読み込み
    pub fn load_all_tensors<T: Float + 'static>(&self) -> Result<HashMap<String, Tensor<T>>, SafetensorsError> {
        let mut tensors = HashMap::new();
        
        for name in self.tensor_names() {
            let tensor = self.load_tensor::<T>(name)?;
            tensors.insert(name.to_string(), tensor);
        }

        Ok(tensors)
    }
}

/// Safetensors model saver for RusTorch
/// RusTorch向けSafetensorsモデルセーバー
#[cfg(feature = "safetensors")]
pub struct SafetensorsSaver;

#[cfg(feature = "safetensors")]
impl SafetensorsSaver {
    /// Save tensors to safetensors format
    /// テンソルをsafetensors形式で保存
    pub fn save_to_file<T: Float + 'static, P: AsRef<Path>>(
        tensors: &HashMap<String, Tensor<T>>,
        path: P,
    ) -> Result<(), SafetensorsError> {
        let mut data_map = HashMap::new();
        
        for (name, tensor) in tensors {
            // Convert tensor data to bytes
            let shape = tensor.shape().to_vec();
            let data_f32: Vec<f32> = tensor.data.iter()
                .map(|&x| x.to_f32().unwrap())
                .collect();
            
            let dtype = safetensors::Dtype::F32;
            let bytes = bytemuck::cast_slice(&data_f32);
            
            data_map.insert(name.clone(), (dtype, shape, bytes.to_vec()));
        }

        // Create safetensors data
        let safetensor_data = safetensors::serialize(&data_map, &None)?;
        
        // Write to file
        std::fs::write(path, safetensor_data)?;
        
        Ok(())
    }
}

// Add bytemuck dependency
#[cfg(feature = "safetensors")]
extern crate bytemuck;

#[cfg(test)]
#[cfg(feature = "safetensors")]
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
        assert_eq!(loaded_tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
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
}