//! Model save/load functionality for Phase 9
//! フェーズ9用モデル保存・読み込み機能

use super::core::{
    compute_checksum, ComputationGraph, FileHeader, Loadable, ModelMetadata, Saveable,
    SerializationError, SerializationResult, TensorMetadata,
};
use crate::tensor::Tensor;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Main save function for objects
/// オブジェクト用メイン保存関数
pub fn save<P: AsRef<Path>>(obj: &dyn Saveable, path: P) -> SerializationResult<()> {
    let path = path.as_ref();

    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);

    // Write RUSTORCH magic bytes first for format detection
    writer.write_all(b"RUSTORCH")?;

    // Create header
    let metadata = obj.metadata();
    let mut header = FileHeader::new(obj.type_id().to_string(), metadata);

    // Serialize object data
    let object_data = obj.save_binary()?;
    header.checksum = compute_checksum(&object_data);

    // Write header
    let header_data =
        bincode::serialize(&header).map_err(|e| SerializationError::FormatError(e.to_string()))?;
    let header_size = header_data.len() as u64;

    writer.write_all(&header_size.to_le_bytes())?;
    writer.write_all(&header_data)?;

    // Write object data
    writer.write_all(&object_data)?;
    writer.flush()?;

    Ok(())
}

/// Main load function for objects
/// オブジェクト用メイン読み込み関数
pub fn load<P: AsRef<Path>, T: Loadable>(path: P) -> SerializationResult<T> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    // Read and verify RUSTORCH magic bytes
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"RUSTORCH" {
        return Err(SerializationError::FormatError(
            "Invalid RusTorch file format".to_string(),
        ));
    }

    // Read header size
    let mut header_size_bytes = [0u8; 8];
    reader.read_exact(&mut header_size_bytes)?;
    let header_size = u64::from_le_bytes(header_size_bytes);

    // Read header
    let mut header_data = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_data)?;
    let header: FileHeader = bincode::deserialize(&header_data)
        .map_err(|e| SerializationError::FormatError(e.to_string()))?;

    // Validate header
    header.validate()?;

    // Check type compatibility
    if header.object_type != T::expected_type_id() {
        return Err(SerializationError::TypeMismatch {
            expected: T::expected_type_id().to_string(),
            found: header.object_type,
        });
    }

    // Validate version
    T::validate_version(&header.version)?;

    // Read object data
    let mut object_data = Vec::new();
    reader.read_to_end(&mut object_data)?;

    // Verify checksum
    let computed_checksum = compute_checksum(&object_data);
    if computed_checksum != header.checksum {
        return Err(SerializationError::CorruptionError(
            "Checksum mismatch".to_string(),
        ));
    }

    // Deserialize object
    T::load_binary(&object_data)
}

/// Model state dictionary for PyTorch compatibility
/// PyTorch互換性用モデル状態辞書
#[derive(Debug, Clone)]
pub struct StateDict<T: Float> {
    pub parameters: HashMap<String, Tensor<T>>,
    pub buffers: HashMap<String, Tensor<T>>,
    pub metadata: ModelMetadata,
}

impl<T: Float + 'static> StateDict<T> {
    /// Create new state dictionary
    /// 新しい状態辞書を作成
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            buffers: HashMap::new(),
            metadata: ModelMetadata {
                model_type: "unknown".to_string(),
                parameters: HashMap::new(),
                buffers: HashMap::new(),
                config: HashMap::new(),
                training_state: false,
            },
        }
    }

    /// Add parameter to state dict
    /// 状態辞書にパラメータを追加
    pub fn add_parameter(&mut self, name: String, tensor: Tensor<T>) {
        let metadata = TensorMetadata {
            shape: tensor.shape().to_vec(),
            dtype: std::any::type_name::<T>().to_string(),
            device: "cpu".to_string(), // Default to CPU for now
            requires_grad: true,
            data_offset: 0, // Will be computed during save
            data_size: tensor.numel() as u64 * std::mem::size_of::<T>() as u64,
        };

        self.metadata.parameters.insert(name.clone(), metadata);
        self.parameters.insert(name, tensor);
    }

    /// Add buffer to state dict
    /// 状態辞書にバッファを追加
    pub fn add_buffer(&mut self, name: String, tensor: Tensor<T>) {
        let metadata = TensorMetadata {
            shape: tensor.shape().to_vec(),
            dtype: std::any::type_name::<T>().to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            data_offset: 0,
            data_size: tensor.numel() as u64 * std::mem::size_of::<T>() as u64,
        };

        self.metadata.buffers.insert(name.clone(), metadata);
        self.buffers.insert(name, tensor);
    }

    /// Get parameter by name
    /// 名前でパラメータを取得
    pub fn get_parameter(&self, name: &str) -> Option<&Tensor<T>> {
        self.parameters.get(name)
    }

    /// Get buffer by name
    /// 名前でバッファを取得
    pub fn get_buffer(&self, name: &str) -> Option<&Tensor<T>> {
        self.buffers.get(name)
    }

    /// Check if training mode
    /// トレーニングモードかチェック
    pub fn is_training(&self) -> bool {
        self.metadata.training_state
    }

    /// Set training mode
    /// トレーニングモードを設定
    pub fn set_training(&mut self, training: bool) {
        self.metadata.training_state = training;
    }
}

impl<T: Float + 'static> Saveable for StateDict<T> {
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Save metadata first
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        let metadata_bytes = metadata_json.as_bytes();
        buffer.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(metadata_bytes);

        // Save parameters count and data
        buffer.extend_from_slice(&(self.parameters.len() as u32).to_le_bytes());
        for (name, tensor) in &self.parameters {
            let name_bytes = name.as_bytes();
            buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(name_bytes);

            let tensor_data = tensor.save_binary()?;
            buffer.extend_from_slice(&(tensor_data.len() as u64).to_le_bytes());
            buffer.extend_from_slice(&tensor_data);
        }

        // Save buffers count and data
        buffer.extend_from_slice(&(self.buffers.len() as u32).to_le_bytes());
        for (name, tensor) in &self.buffers {
            let name_bytes = name.as_bytes();
            buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(name_bytes);

            let tensor_data = tensor.save_binary()?;
            buffer.extend_from_slice(&(tensor_data.len() as u64).to_le_bytes());
            buffer.extend_from_slice(&tensor_data);
        }

        Ok(buffer)
    }

    fn type_id(&self) -> &'static str {
        "state_dict"
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("model_type".to_string(), self.metadata.model_type.clone());
        meta.insert(
            "num_parameters".to_string(),
            self.parameters.len().to_string(),
        );
        meta.insert("num_buffers".to_string(), self.buffers.len().to_string());
        meta.insert(
            "training_state".to_string(),
            self.metadata.training_state.to_string(),
        );
        meta
    }
}

impl<T: Float + 'static> Loadable for StateDict<T> {
    fn load_binary(data: &[u8]) -> SerializationResult<Self> {
        if data.is_empty() {
            return Ok(Self::new());
        }

        let mut offset = 0;
        let mut state_dict = Self::new();

        // Read metadata
        if data.len() < offset + 8 {
            return Ok(state_dict);
        }
        let metadata_len =
            u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid metadata length".to_string())
            })?) as usize;
        offset += 8;

        if data.len() < offset + metadata_len {
            return Ok(state_dict);
        }
        let metadata_str =
            std::str::from_utf8(&data[offset..offset + metadata_len]).map_err(|_| {
                SerializationError::FormatError("Invalid metadata encoding".to_string())
            })?;
        state_dict.metadata = serde_json::from_str(metadata_str)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        offset += metadata_len;

        // Read parameters
        if data.len() < offset + 4 {
            return Ok(state_dict);
        }
        let params_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid parameters count".to_string())
            })?);
        offset += 4;

        for _ in 0..params_count {
            // Read parameter name
            if data.len() < offset + 4 {
                break;
            }
            let name_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid parameter name length".to_string())
                })?) as usize;
            offset += 4;

            if data.len() < offset + name_len {
                break;
            }
            let name =
                String::from_utf8(data[offset..offset + name_len].to_vec()).map_err(|_| {
                    SerializationError::FormatError("Invalid parameter name encoding".to_string())
                })?;
            offset += name_len;

            // Read tensor data
            if data.len() < offset + 8 {
                break;
            }
            let tensor_data_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid tensor data length".to_string())
                })?) as usize;
            offset += 8;

            if data.len() < offset + tensor_data_len {
                break;
            }
            let tensor_data = &data[offset..offset + tensor_data_len];
            if let Ok(tensor) = Tensor::<T>::load_binary(tensor_data) {
                state_dict.parameters.insert(name, tensor);
            }
            offset += tensor_data_len;
        }

        // Read buffers (similar to parameters)
        if data.len() < offset + 4 {
            return Ok(state_dict);
        }
        let buffers_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid buffers count".to_string())
            })?);
        offset += 4;

        for _ in 0..buffers_count {
            // Read buffer name
            if data.len() < offset + 4 {
                break;
            }
            let name_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid buffer name length".to_string())
                })?) as usize;
            offset += 4;

            if data.len() < offset + name_len {
                break;
            }
            let name =
                String::from_utf8(data[offset..offset + name_len].to_vec()).map_err(|_| {
                    SerializationError::FormatError("Invalid buffer name encoding".to_string())
                })?;
            offset += name_len;

            // Read tensor data
            if data.len() < offset + 8 {
                break;
            }
            let tensor_data_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid tensor data length".to_string())
                })?) as usize;
            offset += 8;

            if data.len() < offset + tensor_data_len {
                break;
            }
            let tensor_data = &data[offset..offset + tensor_data_len];
            if let Ok(tensor) = Tensor::<T>::load_binary(tensor_data) {
                state_dict.buffers.insert(name, tensor);
            }
            offset += tensor_data_len;
        }

        Ok(state_dict)
    }

    fn expected_type_id() -> &'static str {
        "state_dict"
    }
}

/// Safe tensor format for large models
/// 大規模モデル用セーフテンソル形式
#[derive(Debug, Clone)]
pub struct SafeTensorFormat<T: Float> {
    pub tensors: HashMap<String, Tensor<T>>,
    pub metadata: HashMap<String, String>,
}

impl<T: Float + 'static> SafeTensorFormat<T> {
    /// Create new safe tensor format
    /// 新しいセーフテンソル形式を作成
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add tensor with name
    /// 名前付きテンソルを追加
    pub fn add_tensor(&mut self, name: String, tensor: Tensor<T>) {
        self.tensors.insert(name, tensor);
    }

    /// Save in safetensors format
    /// safetensors形式で保存
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> SerializationResult<()> {
        // Create safetensors-compatible format
        let mut header_data = HashMap::new();

        for (name, tensor) in &self.tensors {
            let shape: Vec<usize> = tensor.shape().to_vec();
            header_data.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": self.get_dtype_string(),
                    "shape": shape,
                    "data_offsets": [0, tensor.numel() * std::mem::size_of::<T>()]
                }),
            );
        }

        // Add metadata
        header_data.insert("__metadata__".to_string(), serde_json::json!(self.metadata));

        let header_json = serde_json::to_string(&header_data)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        // Write header size and header
        let header_size = header_json.len() as u64;
        writer.write_all(&header_size.to_le_bytes())?;
        writer.write_all(header_json.as_bytes())?;

        // Write tensor data
        for (_, tensor) in &self.tensors {
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

    fn get_dtype_string(&self) -> String {
        match std::mem::size_of::<T>() {
            4 => "F32".to_string(),
            8 => "F64".to_string(),
            _ => "UNKNOWN".to_string(),
        }
    }
}

/// Model checkpoint system
/// モデルチェックポイントシステム
#[derive(Debug, Clone)]
pub struct ModelCheckpoint<T: Float> {
    pub epoch: usize,
    pub step: usize,
    pub model_state: StateDict<T>,
    pub optimizer_state: HashMap<String, Vec<u8>>,
    pub scheduler_state: HashMap<String, Vec<u8>>,
    pub metrics: HashMap<String, f64>,
    pub timestamp: u64,
}

impl<T: Float + 'static> ModelCheckpoint<T> {
    /// Create new model checkpoint
    /// 新しいモデルチェックポイントを作成
    pub fn new(epoch: usize, step: usize, model_state: StateDict<T>) -> Self {
        Self {
            epoch,
            step,
            model_state,
            optimizer_state: HashMap::new(),
            scheduler_state: HashMap::new(),
            metrics: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add optimizer state
    /// オプティマイザー状態を追加
    pub fn add_optimizer_state(&mut self, name: String, state: Vec<u8>) {
        self.optimizer_state.insert(name, state);
    }

    /// Add training metrics
    /// トレーニングメトリクスを追加
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Save checkpoint to file
    /// チェックポイントをファイルに保存
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> SerializationResult<()> {
        save(self, path)
    }

    /// Load checkpoint from file
    /// ファイルからチェックポイントを読み込み
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        load(path)
    }
}

impl<T: Float + 'static> Saveable for ModelCheckpoint<T> {
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Save epoch and step
        buffer.extend_from_slice(&(self.epoch as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.step as u64).to_le_bytes());

        // Save model state
        let model_state_data = self.model_state.save_binary()?;
        buffer.extend_from_slice(&(model_state_data.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&model_state_data);

        // Save optimizer state
        buffer.extend_from_slice(&(self.optimizer_state.len() as u32).to_le_bytes());
        for (key, value) in &self.optimizer_state {
            let key_bytes = key.as_bytes();
            buffer.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(key_bytes);
            buffer.extend_from_slice(&(value.len() as u64).to_le_bytes());
            buffer.extend_from_slice(value);
        }

        // Save scheduler state
        buffer.extend_from_slice(&(self.scheduler_state.len() as u32).to_le_bytes());
        for (key, value) in &self.scheduler_state {
            let key_bytes = key.as_bytes();
            buffer.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(key_bytes);
            buffer.extend_from_slice(&(value.len() as u64).to_le_bytes());
            buffer.extend_from_slice(value);
        }

        // Save metrics
        buffer.extend_from_slice(&(self.metrics.len() as u32).to_le_bytes());
        for (key, value) in &self.metrics {
            let key_bytes = key.as_bytes();
            buffer.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(key_bytes);
            buffer.extend_from_slice(&value.to_le_bytes());
        }

        // Save timestamp
        buffer.extend_from_slice(&self.timestamp.to_le_bytes());

        Ok(buffer)
    }

    fn type_id(&self) -> &'static str {
        "model_checkpoint"
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("epoch".to_string(), self.epoch.to_string());
        meta.insert("step".to_string(), self.step.to_string());
        meta.insert("timestamp".to_string(), self.timestamp.to_string());
        meta.insert(
            "model_type".to_string(),
            self.model_state.metadata.model_type.clone(),
        );
        meta
    }
}

impl<T: Float + 'static> Loadable for ModelCheckpoint<T> {
    fn load_binary(data: &[u8]) -> SerializationResult<Self> {
        if data.is_empty() {
            return Ok(Self::new(0, 0, StateDict::new()));
        }

        let mut offset = 0;
        let mut checkpoint = Self::new(0, 0, StateDict::new());

        // Read epoch and step
        if data.len() < offset + 16 {
            return Ok(checkpoint);
        }
        checkpoint.epoch = u64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| SerializationError::FormatError("Invalid epoch".to_string()))?,
        ) as usize;
        offset += 8;

        checkpoint.step = u64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| SerializationError::FormatError("Invalid step".to_string()))?,
        ) as usize;
        offset += 8;

        // Read model state
        if data.len() < offset + 8 {
            return Ok(checkpoint);
        }
        let model_state_len =
            u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid model state length".to_string())
            })?) as usize;
        offset += 8;

        if data.len() < offset + model_state_len {
            return Ok(checkpoint);
        }
        let model_state_data = &data[offset..offset + model_state_len];
        if let Ok(model_state) = StateDict::<T>::load_binary(model_state_data) {
            checkpoint.model_state = model_state;
        }
        offset += model_state_len;

        // Read optimizer state
        if data.len() < offset + 4 {
            return Ok(checkpoint);
        }
        let optimizer_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid optimizer count".to_string())
            })?);
        offset += 4;

        for _ in 0..optimizer_count {
            // Read key
            if data.len() < offset + 4 {
                break;
            }
            let key_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid key length".to_string())
                })?) as usize;
            offset += 4;

            if data.len() < offset + key_len {
                break;
            }
            let key = String::from_utf8(data[offset..offset + key_len].to_vec())
                .map_err(|_| SerializationError::FormatError("Invalid key encoding".to_string()))?;
            offset += key_len;

            // Read value
            if data.len() < offset + 8 {
                break;
            }
            let value_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid value length".to_string())
                })?) as usize;
            offset += 8;

            if data.len() < offset + value_len {
                break;
            }
            let value = data[offset..offset + value_len].to_vec();
            checkpoint.optimizer_state.insert(key, value);
            offset += value_len;
        }

        // Read scheduler state (similar pattern)
        if data.len() < offset + 4 {
            return Ok(checkpoint);
        }
        let scheduler_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid scheduler count".to_string())
            })?);
        offset += 4;

        for _ in 0..scheduler_count {
            // Read key
            if data.len() < offset + 4 {
                break;
            }
            let key_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid key length".to_string())
                })?) as usize;
            offset += 4;

            if data.len() < offset + key_len {
                break;
            }
            let key = String::from_utf8(data[offset..offset + key_len].to_vec())
                .map_err(|_| SerializationError::FormatError("Invalid key encoding".to_string()))?;
            offset += key_len;

            // Read value
            if data.len() < offset + 8 {
                break;
            }
            let value_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid value length".to_string())
                })?) as usize;
            offset += 8;

            if data.len() < offset + value_len {
                break;
            }
            let value = data[offset..offset + value_len].to_vec();
            checkpoint.scheduler_state.insert(key, value);
            offset += value_len;
        }

        // Read metrics
        if data.len() < offset + 4 {
            return Ok(checkpoint);
        }
        let metrics_count =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid metrics count".to_string())
            })?);
        offset += 4;

        for _ in 0..metrics_count {
            // Read key
            if data.len() < offset + 4 {
                break;
            }
            let key_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid key length".to_string())
                })?) as usize;
            offset += 4;

            if data.len() < offset + key_len {
                break;
            }
            let key = String::from_utf8(data[offset..offset + key_len].to_vec())
                .map_err(|_| SerializationError::FormatError("Invalid key encoding".to_string()))?;
            offset += key_len;

            // Read value (f64)
            if data.len() < offset + 8 {
                break;
            }
            let value = f64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                SerializationError::FormatError("Invalid metric value".to_string())
            })?);
            checkpoint.metrics.insert(key, value);
            offset += 8;
        }

        // Read timestamp
        if data.len() >= offset + 8 {
            checkpoint.timestamp =
                u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    SerializationError::FormatError("Invalid timestamp".to_string())
                })?);
        }

        Ok(checkpoint)
    }

    fn expected_type_id() -> &'static str {
        "model_checkpoint"
    }
}

/// Tensor serialization utilities
/// テンソルシリアライゼーションユーティリティ
impl<T: Float + 'static> Saveable for Tensor<T> {
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Serialize shape
        let shape = self.shape();
        buffer.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            buffer.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Serialize data
        if let Some(data_slice) = self.data.as_slice() {
            let byte_len = data_slice.len() * std::mem::size_of::<T>();
            buffer.extend_from_slice(&(byte_len as u64).to_le_bytes());
            let bytes =
                unsafe { std::slice::from_raw_parts(data_slice.as_ptr() as *const u8, byte_len) };
            buffer.extend_from_slice(bytes);
        } else {
            buffer.extend_from_slice(&(0u64).to_le_bytes());
        }

        Ok(buffer)
    }

    fn type_id(&self) -> &'static str {
        "tensor"
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.get_metadata()
    }
}

impl<T: Float + 'static> Loadable for Tensor<T> {
    fn load_binary(data: &[u8]) -> SerializationResult<Self> {
        let mut cursor = 0;

        if data.len() < 4 {
            return Err(SerializationError::FormatError(
                "Insufficient data for tensor shape".to_string(),
            ));
        }

        // Read shape length
        let shape_len = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Read shape
        let mut shape = Vec::new();
        for _ in 0..shape_len {
            if cursor + 8 > data.len() {
                return Err(SerializationError::FormatError(
                    "Insufficient data for tensor shape".to_string(),
                ));
            }
            let dim = u64::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
                data[cursor + 4],
                data[cursor + 5],
                data[cursor + 6],
                data[cursor + 7],
            ]) as usize;
            shape.push(dim);
            cursor += 8;
        }

        // Read data length
        if cursor + 8 > data.len() {
            return Err(SerializationError::FormatError(
                "Insufficient data for tensor data length".to_string(),
            ));
        }
        let data_len = u64::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]) as usize;
        cursor += 8;

        // Read tensor data
        if cursor + data_len > data.len() {
            return Err(SerializationError::FormatError(
                "Insufficient data for tensor data".to_string(),
            ));
        }

        let expected_elements = shape.iter().product::<usize>();
        let actual_elements = data_len / std::mem::size_of::<T>();

        if actual_elements != expected_elements {
            return Err(SerializationError::FormatError(format!(
                "Shape/data mismatch: shape requires {} elements, data has {}",
                expected_elements, actual_elements
            )));
        }

        // Ensure proper alignment for T
        let element_size = std::mem::size_of::<T>();
        let ptr = data[cursor..cursor + data_len].as_ptr();

        // Check alignment
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            // If not aligned, copy to properly aligned buffer
            let mut aligned_data = vec![0u8; data_len];
            aligned_data.copy_from_slice(&data[cursor..cursor + data_len]);
            let float_data = unsafe {
                std::slice::from_raw_parts(aligned_data.as_ptr() as *const T, actual_elements)
            };
            return Ok(Tensor::from_vec(float_data.to_vec(), shape));
        }

        let float_data = unsafe { std::slice::from_raw_parts(ptr as *const T, actual_elements) };

        Ok(Tensor::from_vec(float_data.to_vec(), shape))
    }

    fn expected_type_id() -> &'static str {
        "tensor"
    }
}

/// Model format detection utilities
/// モデル形式検出ユーティリティ
pub fn detect_format<P: AsRef<Path>>(path: P) -> SerializationResult<String> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    // Read first 16 bytes to detect format
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;

    if &magic[0..8] == b"RUSTORCH" {
        Ok("rustorch".to_string())
    } else if &magic[0..4] == b"PKG\x00" {
        Ok("pickle".to_string())
    } else if &magic[0..8] == b"safetens" {
        Ok("safetensors".to_string())
    } else {
        Err(SerializationError::FormatError(
            "Unknown file format".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_state_dict_creation() {
        let mut state_dict = StateDict::<f32>::new();

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        state_dict.add_parameter("weight".to_string(), tensor);

        assert!(state_dict.get_parameter("weight").is_some());
        assert_eq!(state_dict.parameters.len(), 1);
    }

    #[test]
    fn test_tensor_save_load() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);

        let binary_data = tensor.save_binary().unwrap();
        let loaded_tensor = Tensor::<f32>::load_binary(&binary_data).unwrap();

        assert_eq!(tensor.shape(), loaded_tensor.shape());
        assert_eq!(tensor.data.as_slice(), loaded_tensor.data.as_slice());
    }

    #[test]
    fn test_format_detection() {
        // Test would require actual files, so this is a placeholder
        // In a real scenario, we would create test files with different formats
    }

    #[test]
    fn test_model_checkpoint() {
        let mut state_dict = StateDict::<f32>::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        state_dict.add_parameter("test_param".to_string(), tensor);

        let checkpoint = ModelCheckpoint::new(5, 100, state_dict);

        assert_eq!(checkpoint.epoch, 5);
        assert_eq!(checkpoint.step, 100);
        assert!(checkpoint.model_state.get_parameter("test_param").is_some());
    }

    #[test]
    fn test_safe_tensor_format() {
        let mut safe_format = SafeTensorFormat::<f32>::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        safe_format.add_tensor("test_tensor".to_string(), tensor);

        assert_eq!(safe_format.tensors.len(), 1);
        assert!(safe_format.tensors.contains_key("test_tensor"));
    }
}
