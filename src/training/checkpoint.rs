//! チェックポイント管理
//! Checkpoint management

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// チェックポイント保存設定
/// Checkpoint save configuration
#[derive(Debug, Clone)]
pub struct SaveConfig {
    /// 保存ディレクトリ
    pub save_dir: PathBuf,
    /// ファイル名のプレフィックス
    pub prefix: String,
    /// 最大保存数（0で無制限）
    pub max_checkpoints: usize,
    /// 最良のモデルのみ保存するかどうか
    pub save_best_only: bool,
    /// 監視するメトリクス名
    pub monitor: String,
    /// より良い方向（true: 大きい方が良い, false: 小さい方が良い）
    pub mode_max: bool,
}

impl Default for SaveConfig {
    fn default() -> Self {
        Self {
            save_dir: PathBuf::from("checkpoints"),
            prefix: "model".to_string(),
            max_checkpoints: 5,
            save_best_only: false,
            monitor: "val_loss".to_string(),
            mode_max: false,
        }
    }
}

/// チェックポイントメタデータ
/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// エポック番号
    pub epoch: usize,
    /// 訓練損失
    pub train_loss: f64,
    /// 検証損失
    pub val_loss: Option<f64>,
    /// 学習率
    pub learning_rate: Option<f64>,
    /// その他のメトリクス
    pub metrics: HashMap<String, f64>,
    /// 保存日時（Unix timestamp）
    pub timestamp: u64,
    /// モデルの説明
    pub description: Option<String>,
    /// 追加のメタデータ
    pub extra: HashMap<String, String>,
}

impl CheckpointMetadata {
    /// 新しいメタデータを作成
    /// Create new metadata
    pub fn new(epoch: usize, train_loss: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            epoch,
            train_loss,
            val_loss: None,
            learning_rate: None,
            metrics: HashMap::new(),
            timestamp,
            description: None,
            extra: HashMap::new(),
        }
    }

    /// メトリクスを設定
    /// Set metric
    pub fn set_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// 追加メタデータを設定
    /// Set extra metadata
    pub fn set_extra(&mut self, key: String, value: String) {
        self.extra.insert(key, value);
    }
}

/// チェックポイント情報
/// Checkpoint information
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// ファイルパス
    pub path: PathBuf,
    /// メタデータ
    pub metadata: CheckpointMetadata,
    /// ファイルサイズ（バイト）
    pub file_size: u64,
}

impl CheckpointInfo {
    /// ファイル名を取得
    /// Get filename
    pub fn filename(&self) -> Option<&str> {
        self.path.file_name()?.to_str()
    }

    /// 監視するメトリクスの値を取得
    /// Get monitored metric value
    pub fn get_monitored_value(&self, monitor: &str) -> Option<f64> {
        match monitor {
            "train_loss" => Some(self.metadata.train_loss),
            "val_loss" => self.metadata.val_loss,
            _ => self.metadata.metrics.get(monitor).copied(),
        }
    }
}

/// チェックポイント管理者
/// Checkpoint manager
#[derive(Debug)]
pub struct CheckpointManager {
    /// 設定
    config: SaveConfig,
    /// 保存されたチェックポイント一覧
    checkpoints: Vec<CheckpointInfo>,
    /// 最良のスコア
    best_score: Option<f64>,
}

impl CheckpointManager {
    /// 新しいチェックポイント管理者を作成
    /// Create a new checkpoint manager
    pub fn new(config: SaveConfig) -> anyhow::Result<Self> {
        // 保存ディレクトリを作成
        fs::create_dir_all(&config.save_dir)?;

        let mut manager = Self {
            config,
            checkpoints: Vec::new(),
            best_score: None,
        };

        // 既存のチェックポイントをスキャン
        manager.scan_existing_checkpoints()?;

        Ok(manager)
    }

    /// デフォルト設定で作成
    /// Create with default configuration
    pub fn default() -> anyhow::Result<Self> {
        Self::new(SaveConfig::default())
    }

    /// チェックポイントを保存すべきかどうかを判定
    /// Determine if checkpoint should be saved
    pub fn should_save(&mut self, metadata: &CheckpointMetadata) -> bool {
        if !self.config.save_best_only {
            return true;
        }

        let current_score = match self.config.monitor.as_str() {
            "train_loss" => metadata.train_loss,
            "val_loss" => metadata.val_loss.unwrap_or(f64::INFINITY),
            _ => metadata.metrics.get(&self.config.monitor).copied().unwrap_or(0.0),
        };

        match self.best_score {
            None => {
                self.best_score = Some(current_score);
                true
            }
            Some(best) => {
                let is_better = if self.config.mode_max {
                    current_score > best
                } else {
                    current_score < best
                };

                if is_better {
                    self.best_score = Some(current_score);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// チェックポイントを保存
    /// Save checkpoint
    pub fn save_checkpoint(
        &mut self,
        metadata: CheckpointMetadata,
        model_data: &[u8],
    ) -> anyhow::Result<PathBuf> {
        if !self.should_save(&metadata) {
            return Err(anyhow::anyhow!("Checkpoint does not meet save criteria"));
        }

        // ファイル名を生成
        let filename = self.generate_filename(&metadata);
        let checkpoint_path = self.config.save_dir.join(&filename);

        // メタデータを保存
        let metadata_path = checkpoint_path.with_extension("meta.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;

        // モデルデータを保存
        let model_path = checkpoint_path.with_extension("bin");
        fs::write(&model_path, model_data)?;

        // チェックポイント情報を作成
        let file_size = model_data.len() as u64;
        let checkpoint_info = CheckpointInfo {
            path: checkpoint_path.clone(),
            metadata,
            file_size,
        };

        // リストに追加
        self.checkpoints.push(checkpoint_info);

        // 古いチェックポイントを削除
        self.cleanup_old_checkpoints()?;

        println!("Checkpoint saved: {}", checkpoint_path.display());
        Ok(checkpoint_path)
    }

    /// チェックポイントを読み込み
    /// Load checkpoint
    pub fn load_checkpoint(&self, path: &Path) -> anyhow::Result<(CheckpointMetadata, Vec<u8>)> {
        // メタデータを読み込み
        let metadata_path = path.with_extension("meta.json");
        let metadata_json = fs::read_to_string(&metadata_path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

        // モデルデータを読み込み
        let model_path = path.with_extension("bin");
        let model_data = fs::read(&model_path)?;

        Ok((metadata, model_data))
    }

    /// 最新のチェックポイントを取得
    /// Get latest checkpoint
    pub fn latest_checkpoint(&self) -> Option<&CheckpointInfo> {
        self.checkpoints
            .iter()
            .max_by_key(|info| info.metadata.epoch)
    }

    /// 最良のチェックポイントを取得
    /// Get best checkpoint
    pub fn best_checkpoint(&self) -> Option<&CheckpointInfo> {
        if self.checkpoints.is_empty() {
            return None;
        }

        self.checkpoints
            .iter()
            .min_by(|a, b| {
                let a_val = a.get_monitored_value(&self.config.monitor)
                    .unwrap_or(if self.config.mode_max { f64::NEG_INFINITY } else { f64::INFINITY });
                let b_val = b.get_monitored_value(&self.config.monitor)
                    .unwrap_or(if self.config.mode_max { f64::NEG_INFINITY } else { f64::INFINITY });

                if self.config.mode_max {
                    b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
    }

    /// すべてのチェックポイントを取得
    /// Get all checkpoints
    pub fn all_checkpoints(&self) -> &[CheckpointInfo] {
        &self.checkpoints
    }

    /// チェックポイントを削除
    /// Delete checkpoint
    pub fn delete_checkpoint(&mut self, path: &Path) -> anyhow::Result<()> {
        // ファイルを削除
        let metadata_path = path.with_extension("meta.json");
        let model_path = path.with_extension("bin");

        if metadata_path.exists() {
            fs::remove_file(&metadata_path)?;
        }
        if model_path.exists() {
            fs::remove_file(&model_path)?;
        }

        // リストから削除
        self.checkpoints.retain(|info| info.path != path);

        println!("Checkpoint deleted: {}", path.display());
        Ok(())
    }

    /// 古いチェックポイントをクリーンアップ
    /// Cleanup old checkpoints
    fn cleanup_old_checkpoints(&mut self) -> anyhow::Result<()> {
        if self.config.max_checkpoints == 0 {
            return Ok(());
        }

        // エポック順でソート
        self.checkpoints.sort_by_key(|info| info.metadata.epoch);

        // 古いものから削除
        while self.checkpoints.len() > self.config.max_checkpoints {
            if let Some(oldest) = self.checkpoints.first().cloned() {
                self.delete_checkpoint(&oldest.path)?;
            }
        }

        Ok(())
    }

    /// 既存のチェックポイントをスキャン
    /// Scan existing checkpoints
    fn scan_existing_checkpoints(&mut self) -> anyhow::Result<()> {
        if !self.config.save_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.config.save_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("meta") {
                continue; // メタデータファイルはスキップ
            }

            if path.is_file() && path.file_stem().is_some() {
                if let Ok((metadata, model_data)) = self.load_checkpoint(&path) {
                    let checkpoint_info = CheckpointInfo {
                        path: path.clone(),
                        metadata,
                        file_size: model_data.len() as u64,
                    };
                    self.checkpoints.push(checkpoint_info);
                }
            }
        }

        // エポック順でソート
        self.checkpoints.sort_by_key(|info| info.metadata.epoch);

        println!("Scanned {} existing checkpoints", self.checkpoints.len());
        Ok(())
    }

    /// ファイル名を生成
    /// Generate filename
    fn generate_filename(&self, metadata: &CheckpointMetadata) -> String {
        format!(
            "{}_epoch_{:04}_loss_{:.4}",
            self.config.prefix,
            metadata.epoch,
            metadata.train_loss
        )
    }

    /// チェックポイント統計を取得
    /// Get checkpoint statistics
    pub fn statistics(&self) -> CheckpointStatistics {
        let total_size: u64 = self.checkpoints.iter().map(|info| info.file_size).sum();
        let avg_size = if self.checkpoints.is_empty() {
            0
        } else {
            total_size / self.checkpoints.len() as u64
        };

        CheckpointStatistics {
            total_checkpoints: self.checkpoints.len(),
            total_size_bytes: total_size,
            average_size_bytes: avg_size,
            oldest_epoch: self.checkpoints.first().map(|info| info.metadata.epoch),
            newest_epoch: self.checkpoints.last().map(|info| info.metadata.epoch),
        }
    }
}

/// チェックポイント統計
/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStatistics {
    /// 総チェックポイント数
    pub total_checkpoints: usize,
    /// 総サイズ（バイト）
    pub total_size_bytes: u64,
    /// 平均サイズ（バイト）
    pub average_size_bytes: u64,
    /// 最古のエポック
    pub oldest_epoch: Option<usize>,
    /// 最新のエポック
    pub newest_epoch: Option<usize>,
}

impl CheckpointStatistics {
    /// 統計サマリーを表示
    /// Display statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Checkpoint Statistics:\n\
             - Total checkpoints: {}\n\
             - Total size: {:.2} MB\n\
             - Average size: {:.2} MB\n\
             - Epoch range: {} - {}",
            self.total_checkpoints,
            self.total_size_bytes as f64 / 1024.0 / 1024.0,
            self.average_size_bytes as f64 / 1024.0 / 1024.0,
            self.oldest_epoch.map_or("N/A".to_string(), |e| e.to_string()),
            self.newest_epoch.map_or("N/A".to_string(), |e| e.to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_metadata_creation() {
        let metadata = CheckpointMetadata::new(5, 0.25);
        assert_eq!(metadata.epoch, 5);
        assert_eq!(metadata.train_loss, 0.25);
        assert!(metadata.val_loss.is_none());
        assert!(metadata.timestamp > 0);
    }

    #[test]
    fn test_save_config_default() {
        let config = SaveConfig::default();
        assert_eq!(config.prefix, "model");
        assert_eq!(config.max_checkpoints, 5);
        assert!(!config.save_best_only);
        assert_eq!(config.monitor, "val_loss");
        assert!(!config.mode_max);
    }

    #[test]
    fn test_checkpoint_manager_creation() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = SaveConfig {
            save_dir: temp_dir.path().to_path_buf(),
            ..SaveConfig::default()
        };

        let manager = CheckpointManager::new(config)?;
        assert!(manager.checkpoints.is_empty());
        assert!(manager.best_score.is_none());

        Ok(())
    }

    #[test]
    fn test_should_save_logic() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = SaveConfig {
            save_dir: temp_dir.path().to_path_buf(),
            save_best_only: true,
            monitor: "val_loss".to_string(),
            mode_max: false,
            ..SaveConfig::default()
        };

        let mut manager = CheckpointManager::new(config)?;

        // 最初は保存される
        let metadata1 = CheckpointMetadata {
            val_loss: Some(0.5),
            ..CheckpointMetadata::new(1, 0.6)
        };
        assert!(manager.should_save(&metadata1));

        // より良いスコアは保存される
        let metadata2 = CheckpointMetadata {
            val_loss: Some(0.3),
            ..CheckpointMetadata::new(2, 0.4)
        };
        assert!(manager.should_save(&metadata2));

        // 悪いスコアは保存されない
        let metadata3 = CheckpointMetadata {
            val_loss: Some(0.8),
            ..CheckpointMetadata::new(3, 0.9)
        };
        assert!(!manager.should_save(&metadata3));

        Ok(())
    }

    #[test]
    fn test_filename_generation() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = SaveConfig {
            save_dir: temp_dir.path().to_path_buf(),
            prefix: "test_model".to_string(),
            ..SaveConfig::default()
        };

        let manager = CheckpointManager::new(config)?;
        let metadata = CheckpointMetadata::new(42, 0.1234);
        let filename = manager.generate_filename(&metadata);

        assert!(filename.contains("test_model"));
        assert!(filename.contains("epoch_0042"));
        assert!(filename.contains("loss_0.1234"));

        Ok(())
    }

    #[test]
    fn test_checkpoint_info_methods() {
        let path = PathBuf::from("test_model_epoch_0005.bin");
        let metadata = CheckpointMetadata::new(5, 0.25);
        let info = CheckpointInfo {
            path,
            metadata,
            file_size: 1024,
        };

        assert_eq!(info.filename(), Some("test_model_epoch_0005.bin"));
        assert_eq!(info.get_monitored_value("train_loss"), Some(0.25));
        assert_eq!(info.get_monitored_value("val_loss"), None);
    }

    #[test]
    fn test_checkpoint_statistics() {
        let statistics = CheckpointStatistics {
            total_checkpoints: 5,
            total_size_bytes: 5 * 1024 * 1024, // 5MB
            average_size_bytes: 1024 * 1024,   // 1MB
            oldest_epoch: Some(1),
            newest_epoch: Some(5),
        };

        let summary = statistics.summary();
        assert!(summary.contains("Total checkpoints: 5"));
        assert!(summary.contains("5.00 MB"));
        assert!(summary.contains("1.00 MB"));
        assert!(summary.contains("1 - 5"));
    }
}