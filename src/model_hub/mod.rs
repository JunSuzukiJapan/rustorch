//! Model Hub - Pretrained model download and management
//! モデルハブ - 事前学習済みモデルのダウンロードと管理

pub mod cache;
pub mod downloader;
pub mod registry;
pub mod verification;

pub use cache::{CacheConfig, ModelCache};
pub use downloader::{DownloadError, DownloadProgress, ModelDownloader};
pub use registry::{ModelInfo, ModelRegistry, ModelSource};
pub use verification::{Checksum, ModelVerifier};

use crate::error::RusTorchResult;
use crate::model_import::ImportedModel;
use std::path::PathBuf;

/// High-level model hub interface
/// 高レベルモデルハブインターface
pub struct ModelHub {
    downloader: ModelDownloader,
    registry: ModelRegistry,
    cache: ModelCache,
    verifier: ModelVerifier,
}

impl ModelHub {
    /// Create new model hub with default configuration
    /// デフォルト設定で新しいモデルハブを作成
    pub fn new() -> RusTorchResult<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::env::temp_dir())
            .join("rustorch")
            .join("models");

        let downloader = ModelDownloader::new();
        let registry = ModelRegistry::default();
        let cache = ModelCache::new(cache_dir)?;
        let verifier = ModelVerifier::new();

        Ok(Self {
            downloader,
            registry,
            cache,
            verifier,
        })
    }

    /// Create model hub with custom cache directory
    /// カスタムキャッシュディレクトリでモデルハブを作成
    pub fn with_cache_dir<P: Into<PathBuf>>(cache_dir: P) -> RusTorchResult<Self> {
        let downloader = ModelDownloader::new();
        let registry = ModelRegistry::default();
        let cache = ModelCache::new(cache_dir)?;
        let verifier = ModelVerifier::new();

        Ok(Self {
            downloader,
            registry,
            cache,
            verifier,
        })
    }

    /// Load pretrained model by name
    /// 名前で事前学習済みモデルを読み込み
    pub async fn load_model(&mut self, model_name: &str) -> RusTorchResult<ImportedModel> {
        // Check if model is already cached
        if let Some(cached_path) = self.cache.get_model_path(model_name) {
            if cached_path.exists() {
                println!("Loading cached model: {}", model_name);
                return crate::model_import::pytorch::import_pytorch_model(&cached_path);
            }
        }

        // Get model info from registry
        let model_info = self.registry.get_model_info(model_name)?;

        // Download model
        println!("Downloading model: {} from {}", model_name, model_info.url);
        let download_path = self.cache.get_download_path(model_name);

        self.downloader
            .download_with_progress(&model_info.url, &download_path, |progress| {
                println!("Download progress: {:.1}%", progress.percentage());
            })
            .await?;

        // Verify checksum if available
        if let Some(expected_hash) = &model_info.checksum {
            println!("Verifying model integrity...");
            self.verifier.verify_file(&download_path, expected_hash)?;
            println!("Model verification successful");
        }

        // Cache the model
        let cached_path = self.cache.cache_model(model_name, &download_path)?;

        // Import the model
        println!("Importing model...");
        crate::model_import::pytorch::import_pytorch_model(&cached_path)
    }

    /// List available models
    /// 利用可能なモデルをリスト表示
    pub fn list_models(&self) -> Vec<&str> {
        self.registry.list_models()
    }

    /// Get model information
    /// モデル情報を取得
    pub fn get_model_info(&self, model_name: &str) -> RusTorchResult<&ModelInfo> {
        self.registry.get_model_info(model_name)
    }

    /// Clear model cache
    /// モデルキャッシュをクリア
    pub fn clear_cache(&mut self) -> RusTorchResult<()> {
        self.cache.clear()
    }

    /// Get cache statistics
    /// キャッシュ統計を取得
    pub fn cache_stats(&self) -> (usize, u64) {
        self.cache.stats()
    }
}

impl Default for ModelHub {
    fn default() -> Self {
        Self::new().expect("Failed to create default ModelHub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_hub_creation() {
        let hub = ModelHub::new();
        assert!(hub.is_ok());
    }

    #[test]
    fn test_model_hub_with_custom_cache() {
        let temp_dir = TempDir::new().unwrap();
        let hub = ModelHub::with_cache_dir(temp_dir.path());
        assert!(hub.is_ok());
    }

    #[test]
    fn test_list_models() {
        let hub = ModelHub::new().unwrap();
        let models = hub.list_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"resnet18"));
        assert!(models.contains(&"resnet50"));
    }

    #[tokio::test]
    async fn test_model_loading_mock() {
        // This test would work with actual models
        // For now, it tests the structure
        let temp_dir = TempDir::new().unwrap();
        let _hub = ModelHub::with_cache_dir(temp_dir.path()).unwrap();

        // Test would download and load actual model
        // let result = hub.load_model("resnet18").await;
        // assert!(result.is_ok());

        // For now, just test that the structure works
        assert!(hub.list_models().contains(&"resnet18"));
    }
}
