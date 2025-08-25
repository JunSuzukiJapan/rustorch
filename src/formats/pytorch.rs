//! PyTorch compatibility format for RusTorch
//! RusTorch向けPyTorch互換フォーマット

use crate::tensor::Tensor;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// PyTorch state dict format for RusTorch
/// RusTorch向けPyTorchステートディクト形式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDict {
    /// Map of tensor names to tensor data
    /// テンソル名からテンソルデータへのマップ
    pub tensors: HashMap<String, TensorData>,
    /// Optional metadata for the state dict
    /// ステートディクトのオプショナルメタデータ
    pub metadata: Option<HashMap<String, String>>,
}

/// Serializable tensor data
/// シリアライズ可能なテンソルデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Shape of the tensor
    /// テンソルの形状
    pub shape: Vec<usize>,
    /// Tensor data as f64 for maximum precision
    /// 最大精度のためのf64テンソルデータ
    pub data: Vec<f64>, // Use f64 for maximum precision
    /// Data type string identifier
    /// データ型文字列識別子
    pub dtype: String,
}

impl StateDict {
    /// Create new empty state dict
    /// 空のステートディクトを作成
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: None,
        }
    }

    /// Add tensor to state dict
    /// ステートディクトにテンソルを追加
    pub fn add_tensor<T: Float + 'static>(&mut self, name: String, tensor: &Tensor<T>) {
        let tensor_data = TensorData {
            shape: tensor.shape().to_vec(),
            data: tensor.data.iter().map(|&x| x.to_f64().unwrap()).collect(),
            dtype: std::any::type_name::<T>().to_string(),
        };
        self.tensors.insert(name, tensor_data);
    }

    /// Get tensor from state dict
    /// ステートディクトからテンソルを取得
    pub fn get_tensor<T: Float + 'static>(&self, name: &str) -> Option<Tensor<T>> {
        self.tensors.get(name).map(|tensor_data| {
            let data: Vec<T> = tensor_data
                .data
                .iter()
                .map(|&x| T::from(x).unwrap())
                .collect();
            Tensor::from_vec(data, tensor_data.shape.clone())
        })
    }

    /// Get all tensor names
    /// 全テンソル名を取得
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Save state dict to file (JSON format)
    /// ステートディクトをファイルに保存（JSON形式）
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load state dict from file (JSON format)
    /// ファイルからステートディクトを読み込み（JSON形式）
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state_dict = serde_json::from_reader(reader)?;
        Ok(state_dict)
    }

    /// Add metadata
    /// メタデータを追加
    pub fn add_metadata(&mut self, key: String, value: String) {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key, value);
    }

    /// Get metadata
    /// メタデータを取得
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.as_ref()?.get(key)
    }
}

impl Default for StateDict {
    fn default() -> Self {
        Self::new()
    }
}

/// PyTorch model wrapper for RusTorch
/// RusTorch向けPyTorchモデルラッパー
pub struct PyTorchModel {
    /// The model's state dictionary containing parameters
    /// パラメータを含むモデルのステートディクショナリ
    pub state_dict: StateDict,
    /// Optional architecture description
    /// オプショナルなアーキテクチャ記述
    pub architecture: Option<String>,
}

impl PyTorchModel {
    /// Create new PyTorch model
    /// 新しいPyTorchモデルを作成
    pub fn new() -> Self {
        Self {
            state_dict: StateDict::new(),
            architecture: None,
        }
    }

    /// Load model from PyTorch state dict
    /// PyTorchステートディクトからモデルを読み込み
    pub fn from_state_dict(state_dict: StateDict) -> Self {
        Self {
            state_dict,
            architecture: None,
        }
    }

    /// Set model architecture description
    /// モデルアーキテクチャ記述を設定
    pub fn set_architecture(&mut self, architecture: String) {
        self.architecture = Some(architecture);
    }

    /// Get layer names
    /// レイヤー名を取得
    pub fn layer_names(&self) -> Vec<&String> {
        self.state_dict.tensor_names()
    }

    /// Get layer weights
    /// レイヤー重みを取得
    pub fn get_layer_weights<T: Float + 'static>(&self, layer_name: &str) -> Option<Tensor<T>> {
        // Look for weight tensor (common PyTorch naming)
        let weight_key = format!("{}.weight", layer_name);
        self.state_dict
            .get_tensor(&weight_key)
            .or_else(|| self.state_dict.get_tensor(layer_name))
    }

    /// Get layer bias
    /// レイヤーバイアスを取得
    pub fn get_layer_bias<T: Float + 'static>(&self, layer_name: &str) -> Option<Tensor<T>> {
        let bias_key = format!("{}.bias", layer_name);
        self.state_dict.get_tensor(&bias_key)
    }

    /// Set layer weights
    /// レイヤー重みを設定
    pub fn set_layer_weights<T: Float + 'static>(&mut self, layer_name: &str, weights: &Tensor<T>) {
        let weight_key = format!("{}.weight", layer_name);
        self.state_dict.add_tensor(weight_key, weights);
    }

    /// Set layer bias
    /// レイヤーバイアスを設定
    pub fn set_layer_bias<T: Float + 'static>(&mut self, layer_name: &str, bias: &Tensor<T>) {
        let bias_key = format!("{}.bias", layer_name);
        self.state_dict.add_tensor(bias_key, bias);
    }

    /// Save model to file
    /// モデルをファイルに保存
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        self.state_dict.save_to_file(path)
    }

    /// Load model from file
    /// ファイルからモデルを読み込み
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let state_dict = StateDict::load_from_file(path)?;
        Ok(Self::from_state_dict(state_dict))
    }
}

impl Default for PyTorchModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for PyTorch compatibility
/// PyTorch互換性のためのユーティリティ関数
pub mod utils {
    use super::*;

    /// Convert layer name from PyTorch to RusTorch convention
    /// レイヤー名をPyTorchからRusTorch慣例に変換
    pub fn convert_layer_name(pytorch_name: &str) -> String {
        pytorch_name
            .replace(".", "_")
            .replace("weight", "w")
            .replace("bias", "b")
    }

    /// Convert layer name from RusTorch to PyTorch convention
    /// レイヤー名をRusTorchからPyTorch慣例に変換
    pub fn to_pytorch_layer_name(rustorch_name: &str) -> String {
        rustorch_name
            .replace("_w", ".weight")
            .replace("_b", ".bias")
            .replace("_", ".")
    }

    /// Extract model statistics
    /// モデル統計を抽出
    pub fn model_statistics(model: &PyTorchModel) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        let mut total_params = 0;
        let mut layer_count = 0;

        for name in model.layer_names() {
            if let Some(tensor_data) = model.state_dict.tensors.get(name) {
                let param_count: usize = tensor_data.shape.iter().product();
                total_params += param_count;
                layer_count += 1;
            }
        }

        stats.insert("total_parameters".to_string(), total_params);
        stats.insert("layer_count".to_string(), layer_count);
        stats
    }

    /// Validate model structure
    /// モデル構造を検証
    pub fn validate_model(model: &PyTorchModel) -> Result<(), String> {
        if model.state_dict.tensors.is_empty() {
            return Err("Model has no tensors".to_string());
        }

        // Check for common issues
        for (name, tensor_data) in &model.state_dict.tensors {
            if tensor_data.shape.is_empty() {
                return Err(format!("Tensor '{}' has empty shape", name));
            }

            if tensor_data.data.is_empty() {
                return Err(format!("Tensor '{}' has no data", name));
            }

            let expected_size: usize = tensor_data.shape.iter().product();
            if tensor_data.data.len() != expected_size {
                return Err(format!(
                    "Tensor '{}' data size mismatch: expected {}, got {}",
                    name,
                    expected_size,
                    tensor_data.data.len()
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_state_dict_operations() {
        let mut state_dict = StateDict::new();
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        state_dict.add_tensor("test_layer.weight".to_string(), &tensor);

        let loaded_tensor: Tensor<f32> = state_dict.get_tensor("test_layer.weight").unwrap();
        assert_eq!(loaded_tensor.shape(), &[2, 2]);
        assert_eq!(
            loaded_tensor.data.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_pytorch_model_operations() {
        let mut model = PyTorchModel::new();
        let weights = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        let bias = Tensor::<f32>::from_vec(vec![0.1, 0.2], vec![2]);

        model.set_layer_weights("linear1", &weights);
        model.set_layer_bias("linear1", &bias);

        let loaded_weights: Tensor<f32> = model.get_layer_weights("linear1").unwrap();
        let loaded_bias: Tensor<f32> = model.get_layer_bias("linear1").unwrap();

        assert_eq!(
            loaded_weights.data.as_slice().unwrap(),
            weights.data.as_slice().unwrap()
        );
        assert_eq!(
            loaded_bias.data.as_slice().unwrap(),
            bias.data.as_slice().unwrap()
        );
    }

    #[test]
    fn test_save_load_state_dict() {
        let mut state_dict = StateDict::new();
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        state_dict.add_tensor("test".to_string(), &tensor);
        state_dict.add_metadata("version".to_string(), "1.0".to_string());

        let temp_file = NamedTempFile::new().unwrap();
        state_dict.save_to_file(temp_file.path()).unwrap();

        let loaded_state_dict = StateDict::load_from_file(temp_file.path()).unwrap();
        let loaded_tensor: Tensor<f32> = loaded_state_dict.get_tensor("test").unwrap();

        assert_eq!(loaded_tensor.data.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(
            loaded_state_dict.get_metadata("version"),
            Some(&"1.0".to_string())
        );
    }

    #[test]
    fn test_model_statistics() {
        let mut model = PyTorchModel::new();
        let weights1 = Tensor::<f32>::from_vec(vec![1.0; 12], vec![3, 4]); // 12 params
        let weights2 = Tensor::<f32>::from_vec(vec![1.0; 8], vec![2, 4]); // 8 params

        model.set_layer_weights("layer1", &weights1);
        model.set_layer_weights("layer2", &weights2);

        let stats = utils::model_statistics(&model);
        assert_eq!(stats["total_parameters"], 20);
        assert_eq!(stats["layer_count"], 2);
    }

    #[test]
    fn test_model_validation() {
        let mut model = PyTorchModel::new();
        let weights = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        model.set_layer_weights("linear", &weights);

        assert!(utils::validate_model(&model).is_ok());

        // Test empty model
        let empty_model = PyTorchModel::new();
        assert!(utils::validate_model(&empty_model).is_err());
    }
}
