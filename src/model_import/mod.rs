/// Model format definitions and utilities
/// モデルフォーマット定義とユーティリティ
pub mod formats;
/// Model import functionality for PyTorch and ONNX models
/// PyTorchとONNXモデルのインポート機能

/// ONNX model import functionality
/// ONNXモデルインポート機能
pub mod onnx;
/// PyTorch model import functionality
/// PyTorchモデルインポート機能
pub mod pytorch;

use crate::error::RusTorchError;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path; // RusTorchResult,

// ImportError enum removed - now using unified RusTorchError system
// ImportErrorエナム削除 - 統一RusTorchErrorシステムを使用

/// Result type for model import operations (統一済み)
/// モデルインポート操作の結果型 (統一済み)
pub type ImportResult<T> = crate::error::RusTorchResult<T>;

/// Represents an imported model with weights and structure
/// 重みと構造を持つインポートされたモデルを表現
#[derive(Debug, Clone)]
pub struct ImportedModel {
    /// Model structure metadata
    pub metadata: ModelMetadata,
    /// Model weights/parameters
    pub weights: HashMap<String, Tensor<f32>>,
    /// Model architecture information
    pub architecture: ModelArchitecture,
}

/// Model metadata containing version and format information
/// バージョンとフォーマット情報を含むモデルメタデータ
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Original framework (PyTorch, TensorFlow, etc.)
    pub framework: String,
    /// Model format (ONNX, PyTorch, etc.)
    pub format: String,
    /// Description
    pub description: Option<String>,
    /// Author/creator information
    pub author: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Creation date
    pub created: Option<String>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// Model structure information
/// モデル構造情報
#[derive(Debug, Clone)]
pub struct ModelStructure {
    /// Model input shape
    /// モデル入力形状
    pub input_shape: Vec<usize>,
    /// Model output shape
    /// モデル出力形状
    pub output_shape: Vec<usize>,
    /// Total number of parameters
    /// 総パラメータ数
    pub num_parameters: usize,
}

/// Model architecture description
/// モデルアーキテクチャの説明
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
    /// Model layers/operations
    pub layers: Vec<LayerInfo>,
    /// Total parameter count
    pub parameter_count: usize,
    /// Model size in bytes
    pub model_size: usize,
}

/// Tensor specification for model inputs/outputs
/// モデル入出力のテンソル仕様
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape (None for dynamic dimensions)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub dtype: crate::dtype::DType,
    /// Description
    pub description: Option<String>,
}

/// Layer information for model architecture
/// モデルアーキテクチャのレイヤー情報
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    /// Layer type (Linear, Conv2d, etc.)
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<Option<usize>>,
    /// Output shape
    pub output_shape: Vec<Option<usize>>,
    /// Number of parameters
    pub params: usize,
    /// Layer-specific attributes
    pub attributes: HashMap<String, String>,
}

/// Main interface for importing models
/// モデルインポートのメインインターフェース
pub struct ModelImporter {
    /// Cache for downloaded models
    cache_dir: Option<std::path::PathBuf>,
    /// Progress callback for downloads
    progress_callback: Option<Box<dyn Fn(u64, u64) + Send + Sync>>,
}

impl ModelImporter {
    /// Create a new model importer
    /// 新しいモデルインポーターを作成
    pub fn new() -> Self {
        Self {
            cache_dir: None,
            progress_callback: None,
        }
    }

    /// Set cache directory for downloaded models
    /// ダウンロードしたモデルのキャッシュディレクトリを設定
    pub fn with_cache_dir<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = Some(cache_dir.as_ref().to_path_buf());
        self
    }

    /// Set progress callback for downloads
    /// ダウンロード用のプログレスコールバックを設定
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Import model from file path
    /// ファイルパスからモデルをインポート
    pub fn import_from_file<P: AsRef<Path>>(&self, path: P) -> ImportResult<ImportedModel> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(RusTorchError::model_io(format!(
                "File not found: {}",
                path.display()
            )));
        }

        // Determine format from file extension
        let format = self.detect_format(path)?;

        match format.as_str() {
            "onnx" => onnx::import_onnx_model(path),
            "pytorch" | "pth" | "pt" => pytorch::import_pytorch_model(path),
            _ => Err(RusTorchError::unsupported_format(format)),
        }
    }

    /// Import model from URL
    /// URLからモデルをインポート
    pub fn import_from_url(&self, url: &str) -> ImportResult<ImportedModel> {
        // Download to cache if available
        let local_path = if let Some(cache_dir) = &self.cache_dir {
            self.download_model(url, cache_dir)?
        } else {
            // Download to temporary location
            let temp_dir = std::env::temp_dir();
            self.download_model(url, &temp_dir)?
        };

        self.import_from_file(local_path)
    }

    /// Import pretrained model by name
    /// 名前で事前学習済みモデルをインポート
    pub fn import_pretrained(&self, model_name: &str) -> ImportResult<ImportedModel> {
        // Check if it's a known pretrained model
        if let Some(url) = self.get_pretrained_url(model_name) {
            self.import_from_url(&url)
        } else {
            Err(RusTorchError::model_io(format!(
                "Unknown pretrained model: {}",
                model_name
            )))
        }
    }

    /// Convert imported model to RusTorch module
    /// インポートしたモデルをRusTorchモジュールに変換
    pub fn to_module(&self, model: &ImportedModel) -> ImportResult<Box<dyn Module<f32>>> {
        // This would be implemented based on the specific architecture
        // For now, return a simple linear model as example
        let input_size = model
            .architecture
            .inputs
            .get(0)
            .and_then(|spec| spec.shape.last())
            .and_then(|&size| size)
            .unwrap_or(784);

        let output_size = model
            .architecture
            .outputs
            .get(0)
            .and_then(|spec| spec.shape.last())
            .and_then(|&size| size)
            .unwrap_or(10);

        Ok(Box::new(crate::nn::Linear::new(input_size, output_size)))
    }

    /// Detect model format from file extension
    /// ファイル拡張子からモデル形式を検出
    fn detect_format(&self, path: &Path) -> ImportResult<String> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| RusTorchError::unsupported_format("No file extension"))?;

        match extension.to_lowercase().as_str() {
            "onnx" => Ok("onnx".to_string()),
            "pth" | "pt" => Ok("pytorch".to_string()),
            "pb" => Ok("tensorflow".to_string()),
            "h5" => Ok("keras".to_string()),
            _ => Err(RusTorchError::unsupported_format(extension)),
        }
    }

    /// Download model from URL
    /// URLからモデルをダウンロード
    fn download_model(&self, url: &str, cache_dir: &Path) -> ImportResult<std::path::PathBuf> {
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(cache_dir).map_err(|e| RusTorchError::model_io(e.to_string()))?;

        // Extract filename from URL
        let filename = url
            .split('/')
            .last()
            .ok_or_else(|| RusTorchError::model_io("Invalid URL"))?;

        let local_path = cache_dir.join(filename);

        // Check if file already exists in cache
        if local_path.exists() {
            return Ok(local_path);
        }

        // Mock download implementation
        // In a real implementation, this would use reqwest or similar
        std::fs::write(&local_path, b"mock model data")
            .map_err(|e| RusTorchError::model_io(e.to_string()))?;

        if let Some(callback) = &self.progress_callback {
            callback(100, 100); // Report completion
        }

        Ok(local_path)
    }

    /// Get URL for pretrained model
    /// 事前学習済みモデルのURLを取得
    fn get_pretrained_url(&self, model_name: &str) -> Option<String> {
        // In a real implementation, this would have URLs for popular models
        match model_name {
            "resnet18" => {
                Some("https://download.pytorch.org/models/resnet18-5c106cde.pth".to_string())
            }
            "resnet50" => {
                Some("https://download.pytorch.org/models/resnet50-19c8e357.pth".to_string())
            }
            "mobilenet_v2" => {
                Some("https://download.pytorch.org/models/mobilenet_v2-b0353104.pth".to_string())
            }
            "bert-base-uncased" => Some(
                "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
                    .to_string(),
            ),
            _ => None,
        }
    }
}

impl Default for ModelImporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to import model from file
/// ファイルからモデルをインポートする便利関数
pub fn import_model<P: AsRef<Path>>(path: P) -> ImportResult<ImportedModel> {
    ModelImporter::new().import_from_file(path)
}

/// Convenience function to import pretrained model
/// 事前学習済みモデルをインポートする便利関数
pub fn import_pretrained(model_name: &str) -> ImportResult<ImportedModel> {
    ModelImporter::new().import_pretrained(model_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_importer_creation() {
        let importer = ModelImporter::new();
        assert!(importer.cache_dir.is_none());
        assert!(importer.progress_callback.is_none());
    }

    #[test]
    fn test_format_detection() {
        let importer = ModelImporter::new();

        let onnx_path = std::path::Path::new("model.onnx");
        let pytorch_path = std::path::Path::new("model.pth");

        // These would normally check file extensions
        assert_eq!(importer.detect_format(onnx_path).unwrap(), "onnx");
        assert_eq!(importer.detect_format(pytorch_path).unwrap(), "pytorch");
    }

    #[test]
    fn test_pretrained_url_lookup() {
        let importer = ModelImporter::new();

        assert!(importer.get_pretrained_url("resnet18").is_some());
        assert!(importer.get_pretrained_url("unknown_model").is_none());
    }
}
