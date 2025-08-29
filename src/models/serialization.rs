//! モデル保存・読み込み機能
//! Model serialization and deserialization

use crate::autograd::Variable;
use crate::models::Model;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// シリアライゼーション形式
/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SerializationFormat {
    /// JSON形式
    Json,
    /// バイナリ形式
    Binary,
    /// MessagePack形式
    MessagePack,
}

/// モデル保存エラー
/// Model save error
#[derive(Debug)]
pub enum SaveError {
    /// ファイルI/Oエラー
    IoError(std::io::Error),
    /// シリアライゼーションエラー
    SerializationError(String),
    /// 形式エラー
    FormatError(String),
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaveError::IoError(e) => write!(f, "IO error: {}", e),
            SaveError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            SaveError::FormatError(e) => write!(f, "Format error: {}", e),
        }
    }
}

impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SaveError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SaveError {
    fn from(error: std::io::Error) -> Self {
        SaveError::IoError(error)
    }
}

impl From<serde_json::Error> for SaveError {
    fn from(error: serde_json::Error) -> Self {
        SaveError::SerializationError(error.to_string())
    }
}

/// モデル読み込みエラー
/// Model load error
#[derive(Debug)]
pub enum LoadError {
    /// ファイルI/Oエラー
    IoError(std::io::Error),
    /// デシリアライゼーションエラー
    DeserializationError(String),
    /// 形式エラー
    FormatError(String),
    /// バージョン非互換エラー
    VersionError(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::IoError(e) => write!(f, "IO error: {}", e),
            LoadError::DeserializationError(e) => write!(f, "Deserialization error: {}", e),
            LoadError::FormatError(e) => write!(f, "Format error: {}", e),
            LoadError::VersionError(e) => write!(f, "Version error: {}", e),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(error: std::io::Error) -> Self {
        LoadError::IoError(error)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(error: serde_json::Error) -> Self {
        LoadError::DeserializationError(error.to_string())
    }
}

/// モデル状態辞書
/// Model state dictionary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStateDict {
    /// パラメータ
    pub parameters: HashMap<String, Vec<f64>>,
    /// メタデータ
    pub metadata: HashMap<String, String>,
    /// モデル設定
    pub config: HashMap<String, String>,
    /// バージョン情報
    pub version: String,
    /// 作成日時
    pub created_at: String,
}

impl ModelStateDict {
    /// 新しい状態辞書を作成
    /// Create a new state dictionary
    pub fn new() -> Self {
        ModelStateDict {
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            config: HashMap::new(),
            version: std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.2".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// パラメータを追加
    /// Add parameter
    pub fn add_parameter(&mut self, name: String, values: Vec<f64>) {
        self.parameters.insert(name, values);
    }

    /// メタデータを追加
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// 設定を追加
    /// Add configuration
    pub fn add_config(&mut self, key: String, value: String) {
        self.config.insert(key, value);
    }

    /// パラメータ数を取得
    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.values().map(|v| v.len()).sum()
    }

    /// サイズを取得（バイト）
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        // 概算サイズ計算
        self.parameter_count() * std::mem::size_of::<f64>()
            + self
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + self
                .config
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }
}

impl Default for ModelStateDict {
    fn default() -> Self {
        Self::new()
    }
}

/// モデル保存器
/// Model saver
pub struct ModelSaver;

impl ModelSaver {
    /// モデルを保存
    /// Save model
    pub fn save<T, M>(model: &M, path: &Path, format: SerializationFormat) -> Result<(), SaveError>
    where
        T: Float
            + 'static
            + Send
            + Sync
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
        M: Model<T>,
    {
        let state_dict = Self::extract_state_dict(model);

        match format {
            SerializationFormat::Json => Self::save_json(&state_dict, path),
            SerializationFormat::Binary => Self::save_binary(&state_dict, path),
            SerializationFormat::MessagePack => Self::save_messagepack(&state_dict, path),
        }
    }

    /// モデルから状態辞書を抽出
    /// Extract state dictionary from model
    fn extract_state_dict<T, M>(model: &M) -> ModelStateDict
    where
        T: Float
            + 'static
            + Send
            + Sync
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
        M: Model<T>,
    {
        let mut state_dict = ModelStateDict::new();

        // モデル設定を追加
        for (key, value) in model.config() {
            state_dict.add_config(key, value);
        }

        // パラメータを抽出（簡略化実装）
        let parameters = model.parameters();
        for (i, param) in parameters.iter().enumerate() {
            let param_name = format!("param_{}", i);
            let values = Self::extract_parameter_values(param);
            state_dict.add_parameter(param_name, values);
        }

        // メタデータを追加
        state_dict.add_metadata("model_summary".to_string(), model.summary());
        state_dict.add_metadata("mode".to_string(), format!("{:?}", model.mode()));

        state_dict
    }

    /// パラメータ値を抽出
    /// Extract parameter values
    fn extract_parameter_values<T>(_param: &Variable<T>) -> Vec<f64>
    where
        T: Float
            + 'static
            + Send
            + Sync
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        // 実装は簡略化 - 実際にはテンソルから値を抽出
        vec![0.0; 100] // プレースホルダー
    }

    /// JSON形式で保存
    /// Save in JSON format
    fn save_json(state_dict: &ModelStateDict, path: &Path) -> Result<(), SaveError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, state_dict)?;
        Ok(())
    }

    /// バイナリ形式で保存
    /// Save in binary format
    fn save_binary(state_dict: &ModelStateDict, path: &Path) -> Result<(), SaveError> {
        let json_data = serde_json::to_vec(state_dict)?;
        let mut file = File::create(path)?;
        file.write_all(&json_data)?;
        Ok(())
    }

    /// MessagePack形式で保存
    /// Save in MessagePack format
    fn save_messagepack(_state_dict: &ModelStateDict, _path: &Path) -> Result<(), SaveError> {
        // MessagePack実装は省略（rmp-serdeクレートが必要）
        Err(SaveError::FormatError(
            "MessagePack not implemented".to_string(),
        ))
    }

    /// チェックポイントを保存
    /// Save checkpoint
    pub fn save_checkpoint<T, M>(
        model: &M,
        epoch: usize,
        loss: f64,
        optimizer_state: Option<&HashMap<String, Vec<f64>>>,
        path: &Path,
    ) -> Result<(), SaveError>
    where
        T: Float
            + 'static
            + Send
            + Sync
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
        M: Model<T>,
    {
        let mut state_dict = Self::extract_state_dict(model);

        // チェックポイント情報を追加
        state_dict.add_metadata("checkpoint_epoch".to_string(), epoch.to_string());
        state_dict.add_metadata("checkpoint_loss".to_string(), loss.to_string());

        if let Some(opt_state) = optimizer_state {
            for (key, values) in opt_state {
                state_dict.add_parameter(format!("optimizer_{}", key), values.clone());
            }
        }

        Self::save_json(&state_dict, path)
    }
}

/// モデル読み込み器
/// Model loader
pub struct ModelLoader;

impl ModelLoader {
    /// モデルを読み込み
    /// Load model
    pub fn load(path: &Path, format: SerializationFormat) -> Result<ModelStateDict, LoadError> {
        match format {
            SerializationFormat::Json => Self::load_json(path),
            SerializationFormat::Binary => Self::load_binary(path),
            SerializationFormat::MessagePack => Self::load_messagepack(path),
        }
    }

    /// JSON形式で読み込み
    /// Load from JSON format
    fn load_json(path: &Path) -> Result<ModelStateDict, LoadError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state_dict: ModelStateDict = serde_json::from_reader(reader)?;
        Self::validate_state_dict(&state_dict)?;
        Ok(state_dict)
    }

    /// バイナリ形式で読み込み
    /// Load from binary format
    fn load_binary(path: &Path) -> Result<ModelStateDict, LoadError> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let state_dict: ModelStateDict = serde_json::from_slice(&buffer)?;
        Self::validate_state_dict(&state_dict)?;
        Ok(state_dict)
    }

    /// MessagePack形式で読み込み
    /// Load from MessagePack format
    fn load_messagepack(_path: &Path) -> Result<ModelStateDict, LoadError> {
        // MessagePack実装は省略
        Err(LoadError::FormatError(
            "MessagePack not implemented".to_string(),
        ))
    }

    /// 状態辞書を検証
    /// Validate state dictionary
    fn validate_state_dict(state_dict: &ModelStateDict) -> Result<(), LoadError> {
        // バージョン互換性チェック
        let current_version =
            std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.2".to_string());
        if state_dict.version != current_version {
            println!(
                "Warning: Model version {} differs from current version {}",
                state_dict.version, current_version
            );
        }

        // 必須フィールドの存在チェック
        if state_dict.parameters.is_empty() {
            return Err(LoadError::DeserializationError(
                "No parameters found in state dictionary".to_string(),
            ));
        }

        Ok(())
    }

    /// チェックポイントを読み込み
    /// Load checkpoint
    pub fn load_checkpoint(path: &Path) -> Result<(ModelStateDict, CheckpointInfo), LoadError> {
        let state_dict = Self::load_json(path)?;

        let epoch = state_dict
            .metadata
            .get("checkpoint_epoch")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let loss = state_dict
            .metadata
            .get("checkpoint_loss")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let checkpoint_info = CheckpointInfo { epoch, loss };

        Ok((state_dict, checkpoint_info))
    }

    /// 形式を自動検出
    /// Auto-detect format
    pub fn detect_format(path: &Path) -> Result<SerializationFormat, LoadError> {
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("json") => Ok(SerializationFormat::Json),
                Some("bin" | "dat") => Ok(SerializationFormat::Binary),
                Some("msgpack" | "mp") => Ok(SerializationFormat::MessagePack),
                _ => Err(LoadError::FormatError(format!(
                    "Unknown file extension: {:?}",
                    extension
                ))),
            }
        } else {
            Err(LoadError::FormatError(
                "No file extension found".to_string(),
            ))
        }
    }
}

/// チェックポイント情報
/// Checkpoint information
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// エポック数
    pub epoch: usize,
    /// 損失値
    pub loss: f64,
}

/// モデル変換器
/// Model converter
pub struct ModelConverter;

impl ModelConverter {
    /// 形式を変換
    /// Convert format
    pub fn convert(
        input_path: &Path,
        output_path: &Path,
        input_format: SerializationFormat,
        _output_format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state_dict = ModelLoader::load(input_path, input_format)?;
        ModelSaver::save_json(&state_dict, output_path)?;
        Ok(())
    }

    /// モデルを圧縮
    /// Compress model
    pub fn compress(
        input_path: &Path,
        output_path: &Path,
        compression_ratio: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut state_dict = ModelLoader::load(input_path, SerializationFormat::Json)?;

        // パラメータを量子化（簡略化実装）
        for (_name, params) in &mut state_dict.parameters {
            Self::quantize_parameters(params, compression_ratio);
        }

        state_dict.add_metadata("compressed".to_string(), "true".to_string());
        state_dict.add_metadata(
            "compression_ratio".to_string(),
            compression_ratio.to_string(),
        );

        ModelSaver::save_json(&state_dict, output_path)?;
        Ok(())
    }

    /// パラメータを量子化
    /// Quantize parameters
    fn quantize_parameters(params: &mut Vec<f64>, ratio: f64) {
        // 簡単な量子化実装
        for param in params.iter_mut() {
            *param = (*param * ratio).round() / ratio;
        }
    }
}

/// モデル情報表示器
/// Model info displayer
pub struct ModelInfo;

impl ModelInfo {
    /// モデル情報を表示
    /// Display model information
    pub fn display(path: &Path) -> Result<(), LoadError> {
        let format = ModelLoader::detect_format(path)?;
        let state_dict = ModelLoader::load(path, format)?;

        println!("Model Information:");
        println!("  Path: {:?}", path);
        println!("  Format: {:?}", format);
        println!("  Version: {}", state_dict.version);
        println!("  Created: {}", state_dict.created_at);
        println!("  Parameters: {}", state_dict.parameter_count());
        println!(
            "  Size: {:.2} MB",
            state_dict.size_bytes() as f64 / 1024.0 / 1024.0
        );

        println!("\nConfiguration:");
        for (key, value) in &state_dict.config {
            println!("  {}: {}", key, value);
        }

        println!("\nMetadata:");
        for (key, value) in &state_dict.metadata {
            println!("  {}: {}", key, value);
        }

        Ok(())
    }

    /// モデルを比較
    /// Compare models
    pub fn compare(path1: &Path, path2: &Path) -> Result<(), LoadError> {
        let format1 = ModelLoader::detect_format(path1)?;
        let format2 = ModelLoader::detect_format(path2)?;

        let state_dict1 = ModelLoader::load(path1, format1)?;
        let state_dict2 = ModelLoader::load(path2, format2)?;

        println!("Model Comparison:");
        println!("  Model 1: {:?}", path1);
        println!("  Model 2: {:?}", path2);

        println!("\nParameter Count:");
        println!("  Model 1: {}", state_dict1.parameter_count());
        println!("  Model 2: {}", state_dict2.parameter_count());

        println!("\nSize:");
        println!(
            "  Model 1: {:.2} MB",
            state_dict1.size_bytes() as f64 / 1024.0 / 1024.0
        );
        println!(
            "  Model 2: {:.2} MB",
            state_dict2.size_bytes() as f64 / 1024.0 / 1024.0
        );

        // 設定の違いをチェック
        let mut config_diffs = Vec::new();
        for (key, value1) in &state_dict1.config {
            if let Some(value2) = state_dict2.config.get(key) {
                if value1 != value2 {
                    config_diffs.push((key.clone(), value1.clone(), value2.clone()));
                }
            } else {
                config_diffs.push((key.clone(), value1.clone(), "N/A".to_string()));
            }
        }

        if !config_diffs.is_empty() {
            println!("\nConfiguration Differences:");
            for (key, val1, val2) in config_diffs {
                println!("  {}: {} vs {}", key, val1, val2);
            }
        }

        Ok(())
    }
}
