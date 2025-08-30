//! Model cache management
//! モデルキャッシュ管理

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crate::error::{RusTorchError, RusTorchResult};
use serde::{Serialize, Deserialize};

/// Cache configuration
/// キャッシュ設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    /// 最大キャッシュサイズ（バイト）
    pub max_size_bytes: u64,
    /// Maximum number of cached models
    /// 最大キャッシュモデル数
    pub max_models: usize,
    /// Cache expiration time in days
    /// キャッシュ有効期限（日）
    pub expiration_days: u64,
    /// Auto-cleanup on startup
    /// 起動時自動クリーンアップ
    pub auto_cleanup: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10GB
            max_models: 50,
            expiration_days: 30,
            auto_cleanup: true,
        }
    }
}

/// Cache entry metadata
/// キャッシュエントリメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Model name
    /// モデル名
    pub model_name: String,
    /// File path in cache
    /// キャッシュ内ファイルパス
    pub file_path: PathBuf,
    /// File size in bytes
    /// ファイルサイズ（バイト）
    pub file_size: u64,
    /// Download timestamp
    /// ダウンロードタイムスタンプ
    pub downloaded_at: chrono::DateTime<chrono::Utc>,
    /// Last accessed timestamp
    /// 最終アクセスタイムスタンプ
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Model checksum for integrity verification
    /// 整合性検証用モデルチェックサム
    pub checksum: Option<String>,
}

/// Model cache manager
/// モデルキャッシュマネージャー
pub struct ModelCache {
    /// Cache directory path
    /// キャッシュディレクトリパス
    cache_dir: PathBuf,
    /// Cache configuration
    /// キャッシュ設定
    config: CacheConfig,
    /// Cache entries metadata
    /// キャッシュエントリメタデータ
    entries: HashMap<String, CacheEntry>,
    /// Metadata file path
    /// メタデータファイルパス
    metadata_file: PathBuf,
}

impl ModelCache {
    /// Create new cache manager
    /// 新しいキャッシュマネージャーを作成
    pub fn new<P: Into<PathBuf>>(cache_dir: P) -> RusTorchResult<Self> {
        let cache_dir = cache_dir.into();
        let metadata_file = cache_dir.join("cache_metadata.json");
        
        // Create cache directory
        std::fs::create_dir_all(&cache_dir)?;

        let config = CacheConfig::default();
        let mut cache = Self {
            cache_dir,
            config,
            entries: HashMap::new(),
            metadata_file,
        };

        // Load existing metadata
        cache.load_metadata()?;

        // Auto-cleanup if enabled
        if cache.config.auto_cleanup {
            cache.cleanup_expired()?;
        }

        Ok(cache)
    }

    /// Create cache with custom configuration
    /// カスタム設定でキャッシュを作成
    pub fn with_config<P: Into<PathBuf>>(cache_dir: P, config: CacheConfig) -> RusTorchResult<Self> {
        let cache_dir = cache_dir.into();
        let metadata_file = cache_dir.join("cache_metadata.json");
        
        std::fs::create_dir_all(&cache_dir)?;

        let mut cache = Self {
            cache_dir,
            config,
            entries: HashMap::new(),
            metadata_file,
        };

        cache.load_metadata()?;

        if cache.config.auto_cleanup {
            cache.cleanup_expired()?;
        }

        Ok(cache)
    }

    /// Get cached model path if available
    /// キャッシュされたモデルパスを取得（利用可能な場合）
    pub fn get_model_path(&mut self, model_name: &str) -> Option<PathBuf> {
        if let Some(entry) = self.entries.get_mut(model_name) {
            if entry.file_path.exists() {
                // Update last accessed time
                entry.last_accessed = chrono::Utc::now();
                let path = entry.file_path.clone();
                self.save_metadata().ok()?;
                return Some(path);
            } else {
                // File no longer exists, remove from cache
                self.entries.remove(model_name);
                self.save_metadata().ok()?;
            }
        }
        None
    }

    /// Get download path for a model
    /// モデルのダウンロードパスを取得
    pub fn get_download_path(&self, model_name: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.pth", model_name))
    }

    /// Cache a downloaded model
    /// ダウンロードしたモデルをキャッシュ
    pub fn cache_model<P: AsRef<Path>>(&mut self, model_name: &str, source_path: P) -> RusTorchResult<PathBuf> {
        let source_path = source_path.as_ref();
        let target_path = self.get_download_path(model_name);

        // Copy file to cache if it's not already there
        if source_path != target_path {
            std::fs::copy(source_path, &target_path)?;
        }

        // Get file metadata
        let metadata = std::fs::metadata(&target_path)?;
        let file_size = metadata.len();
        
        // Calculate checksum
        let checksum = self.calculate_checksum(&target_path)?;

        // Create cache entry
        let entry = CacheEntry {
            model_name: model_name.to_string(),
            file_path: target_path.clone(),
            file_size,
            downloaded_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            checksum: Some(checksum),
        };

        // Add to cache
        self.entries.insert(model_name.to_string(), entry);

        // Check cache limits and cleanup if necessary
        self.enforce_cache_limits()?;

        // Save metadata
        self.save_metadata()?;

        Ok(target_path)
    }

    /// Remove model from cache
    /// キャッシュからモデルを削除
    pub fn remove_model(&mut self, model_name: &str) -> RusTorchResult<bool> {
        if let Some(entry) = self.entries.remove(model_name) {
            if entry.file_path.exists() {
                std::fs::remove_file(&entry.file_path)?;
            }
            self.save_metadata()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear entire cache
    /// キャッシュ全体をクリア
    pub fn clear(&mut self) -> RusTorchResult<()> {
        for entry in self.entries.values() {
            if entry.file_path.exists() {
                std::fs::remove_file(&entry.file_path).ok();
            }
        }
        
        self.entries.clear();
        self.save_metadata()?;
        Ok(())
    }

    /// Get cache statistics (model_count, total_size_bytes)
    /// キャッシュ統計を取得（モデル数、総サイズバイト）
    pub fn stats(&self) -> (usize, u64) {
        let model_count = self.entries.len();
        let total_size = self.entries.values().map(|e| e.file_size).sum();
        (model_count, total_size)
    }

    /// List cached models
    /// キャッシュされたモデルをリスト表示
    pub fn list_cached_models(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Load cache metadata from disk
    /// ディスクからキャッシュメタデータを読み込み
    fn load_metadata(&mut self) -> RusTorchResult<()> {
        if !self.metadata_file.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.metadata_file)?;
        let entries: HashMap<String, CacheEntry> = serde_json::from_str(&content)
            .map_err(|e| RusTorchError::DeserializationError(e.to_string()))?;

        // Verify that cached files still exist
        for (name, entry) in entries {
            if entry.file_path.exists() {
                self.entries.insert(name, entry);
            }
        }

        Ok(())
    }

    /// Save cache metadata to disk
    /// キャッシュメタデータをディスクに保存
    fn save_metadata(&self) -> RusTorchResult<()> {
        let content = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| RusTorchError::SerializationError(e.to_string()))?;
        
        std::fs::write(&self.metadata_file, content)?;
        Ok(())
    }

    /// Calculate file checksum (SHA-256)
    /// ファイルチェックサムを計算（SHA-256）
    fn calculate_checksum<P: AsRef<Path>>(&self, path: P) -> RusTorchResult<String> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)?;
        let mut hasher = sha2::Sha256::new();
        let mut buffer = [0; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        use sha2::Digest;
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }

    /// Cleanup expired cache entries
    /// 期限切れキャッシュエントリをクリーンアップ
    fn cleanup_expired(&mut self) -> RusTorchResult<()> {
        let expiration_threshold = chrono::Utc::now() 
            - chrono::Duration::days(self.config.expiration_days as i64);

        let expired_models: Vec<String> = self.entries
            .iter()
            .filter(|(_, entry)| entry.last_accessed < expiration_threshold)
            .map(|(name, _)| name.clone())
            .collect();

        for model_name in expired_models {
            println!("Removing expired cached model: {}", model_name);
            self.remove_model(&model_name)?;
        }

        Ok(())
    }

    /// Enforce cache size and count limits
    /// キャッシュサイズと数の制限を適用
    fn enforce_cache_limits(&mut self) -> RusTorchResult<()> {
        // Check total size
        let total_size: u64 = self.entries.values().map(|e| e.file_size).sum();
        
        if total_size > self.config.max_size_bytes || self.entries.len() > self.config.max_models {
            // Remove least recently used models
            let mut entries_by_access: Vec<_> = self.entries.iter().collect();
            entries_by_access.sort_by_key(|(_, entry)| entry.last_accessed);

            let mut current_size = total_size;
            let mut current_count = self.entries.len();

            for (model_name, entry) in entries_by_access {
                if current_size <= self.config.max_size_bytes && current_count <= self.config.max_models {
                    break;
                }

                println!("Removing LRU cached model: {}", model_name);
                if entry.file_path.exists() {
                    std::fs::remove_file(&entry.file_path).ok();
                }
                
                current_size -= entry.file_size;
                current_count -= 1;
            }

            // Rebuild entries map without removed models
            self.entries.retain(|_, entry| entry.file_path.exists());
            self.save_metadata()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_size_bytes, 10 * 1024 * 1024 * 1024);
        assert_eq!(config.max_models, 50);
        assert_eq!(config.expiration_days, 30);
        assert!(config.auto_cleanup);
    }

    #[test]
    fn test_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::new(temp_dir.path());
        assert!(cache.is_ok());
        
        let cache = cache.unwrap();
        assert_eq!(cache.stats(), (0, 0));
    }

    #[test]
    fn test_cache_with_custom_config() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            max_size_bytes: 1024 * 1024, // 1MB
            max_models: 5,
            expiration_days: 7,
            auto_cleanup: false,
        };
        
        let cache = ModelCache::with_config(temp_dir.path(), config.clone());
        assert!(cache.is_ok());
        
        let cache = cache.unwrap();
        assert_eq!(cache.config.max_size_bytes, 1024 * 1024);
        assert_eq!(cache.config.max_models, 5);
    }

    #[test]
    fn test_cache_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Create a test file
        let test_file = temp_dir.path().join("test_model.pth");
        let test_data = b"test model data";
        std::fs::write(&test_file, test_data).unwrap();

        // Cache the model
        let result = cache.cache_model("test_model", &test_file);
        assert!(result.is_ok());

        let cached_path = result.unwrap();
        assert!(cached_path.exists());
        assert_eq!(cache.stats().0, 1); // One model cached
    }

    #[test]
    fn test_get_model_path() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Create and cache a test model
        let test_file = temp_dir.path().join("test_model.pth");
        std::fs::write(&test_file, b"test data").unwrap();
        cache.cache_model("test_model", &test_file).unwrap();

        // Get model path
        let path = cache.get_model_path("test_model");
        assert!(path.is_some());
        assert!(path.unwrap().exists());

        // Non-existent model should return None
        let no_path = cache.get_model_path("nonexistent");
        assert!(no_path.is_none());
    }

    #[test]
    fn test_remove_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Create and cache a test model
        let test_file = temp_dir.path().join("test_model.pth");
        std::fs::write(&test_file, b"test data").unwrap();
        cache.cache_model("test_model", &test_file).unwrap();

        assert_eq!(cache.stats().0, 1);

        // Remove model
        let removed = cache.remove_model("test_model").unwrap();
        assert!(removed);
        assert_eq!(cache.stats().0, 0);

        // Try to remove non-existent model
        let not_removed = cache.remove_model("nonexistent").unwrap();
        assert!(!not_removed);
    }

    #[test]
    fn test_list_cached_models() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Start with empty cache
        assert!(cache.list_cached_models().is_empty());

        // Add some models
        for i in 0..3 {
            let test_file = temp_dir.path().join(format!("model_{}.pth", i));
            std::fs::write(&test_file, b"test data").unwrap();
            cache.cache_model(&format!("model_{}", i), &test_file).unwrap();
        }

        let models = cache.list_cached_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"model_0"));
        assert!(models.contains(&"model_1"));
        assert!(models.contains(&"model_2"));
    }

    #[test]
    fn test_clear_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Add some models
        for i in 0..3 {
            let test_file = temp_dir.path().join(format!("model_{}.pth", i));
            std::fs::write(&test_file, b"test data").unwrap();
            cache.cache_model(&format!("model_{}", i), &test_file).unwrap();
        }

        assert_eq!(cache.stats().0, 3);

        // Clear cache
        cache.clear().unwrap();
        assert_eq!(cache.stats().0, 0);
    }

    #[test]
    fn test_cache_persistence() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create cache and add model
        {
            let mut cache = ModelCache::new(temp_dir.path()).unwrap();
            let test_file = temp_dir.path().join("test_model.pth");
            std::fs::write(&test_file, b"test data").unwrap();
            cache.cache_model("test_model", &test_file).unwrap();
        }

        // Create new cache instance and verify persistence
        {
            let cache = ModelCache::new(temp_dir.path()).unwrap();
            assert_eq!(cache.stats().0, 1);
            assert!(cache.list_cached_models().contains(&"test_model"));
        }
    }
}