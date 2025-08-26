//! 学習状態の管理
//! Training state management

use num_traits::Float;
use std::collections::HashMap;
use std::time::Duration;

/// 訓練全体の状態
/// Overall training state
#[derive(Debug, Clone)]
pub struct TrainingState<T: Float> {
    /// 総エポック数
    pub total_epochs: usize,
    /// 現在のエポック
    pub current_epoch: usize,
    /// エポック履歴
    pub epoch_history: Vec<EpochState<T>>,
    /// グローバルメタデータ
    pub metadata: HashMap<String, String>,
    /// 総実行時間
    pub total_duration: Duration,
    /// 最良の検証損失
    pub best_val_loss: Option<T>,
    /// 最良の検証精度
    pub best_val_accuracy: Option<T>,
    /// 最後に改善したエポック
    pub last_improvement_epoch: Option<usize>,
}

impl<T: Float> TrainingState<T> {
    /// 新しい訓練状態を作成
    /// Create a new training state
    pub fn new(total_epochs: usize) -> Self {
        Self {
            total_epochs,
            current_epoch: 0,
            epoch_history: Vec::new(),
            metadata: HashMap::new(),
            total_duration: Duration::new(0, 0),
            best_val_loss: None,
            best_val_accuracy: None,
            last_improvement_epoch: None,
        }
    }

    /// エポック状態を追加
    /// Add epoch state
    pub fn add_epoch(&mut self, epoch_state: EpochState<T>) {
        // 最良メトリクスの更新
        if let Some(ref val_metrics) = epoch_state.val_metrics {
            let val_loss = val_metrics.avg_loss;

            if self.best_val_loss.is_none() || val_loss < self.best_val_loss.unwrap() {
                self.best_val_loss = Some(val_loss);
                self.last_improvement_epoch = Some(epoch_state.epoch);
            }
        }

        self.epoch_history.push(epoch_state);
        self.current_epoch = self.epoch_history.len();
    }

    /// メタデータを設定
    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// メタデータを取得
    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// 進捗率を取得
    /// Get progress percentage
    pub fn progress(&self) -> f64 {
        if self.total_epochs == 0 {
            0.0
        } else {
            (self.current_epoch as f64 / self.total_epochs as f64) * 100.0
        }
    }

    /// 訓練が完了したかどうか
    /// Whether training is completed
    pub fn is_completed(&self) -> bool {
        self.current_epoch >= self.total_epochs
    }

    /// 最新のエポック状態を取得
    /// Get the latest epoch state
    pub fn latest_epoch(&self) -> Option<&EpochState<T>> {
        self.epoch_history.last()
    }

    /// 訓練サマリーを生成
    /// Generate training summary
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Training Summary:\n\
             - Total epochs: {}\n\
             - Completed epochs: {}\n\
             - Progress: {:.1}%\n\
             - Total time: {:.2}s\n",
            self.total_epochs,
            self.current_epoch,
            self.progress(),
            self.total_duration.as_secs_f64()
        );

        if let Some(best_loss) = self.best_val_loss {
            summary.push_str(&format!(
                " - Best validation loss: {:.4}\n",
                best_loss.to_f64().unwrap_or(0.0)
            ));
        }

        if let Some(best_acc) = self.best_val_accuracy {
            summary.push_str(&format!(
                " - Best validation accuracy: {:.4}\n",
                best_acc.to_f64().unwrap_or(0.0)
            ));
        }

        if let Some(last_improvement) = self.last_improvement_epoch {
            summary.push_str(&format!(
                " - Last improvement at epoch: {}\n",
                last_improvement + 1
            ));
        }

        summary
    }
}

/// エポックレベルの状態
/// Epoch-level state
#[derive(Debug, Clone)]
pub struct EpochState<T: Float> {
    /// エポック番号
    pub epoch: usize,
    /// 訓練メトリクス
    pub train_metrics: Option<super::trainer::EpochMetrics<T>>,
    /// 検証メトリクス
    pub val_metrics: Option<super::trainer::EpochMetrics<T>>,
    /// エポック実行時間
    pub duration: Duration,
    /// エポック固有のメタデータ
    pub metadata: HashMap<String, String>,
    /// 学習率
    pub learning_rate: Option<f64>,
    /// バッチ履歴
    pub batch_history: Vec<BatchState>,
}

impl<T: Float> EpochState<T> {
    /// 新しいエポック状態を作成
    /// Create a new epoch state
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            train_metrics: None,
            val_metrics: None,
            duration: Duration::new(0, 0),
            metadata: HashMap::new(),
            learning_rate: None,
            batch_history: Vec::new(),
        }
    }

    /// メタデータを設定
    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// バッチ状態を追加
    /// Add batch state
    pub fn add_batch(&mut self, batch_state: BatchState) {
        self.batch_history.push(batch_state);
    }

    /// 平均バッチ時間を計算
    /// Calculate average batch time
    pub fn avg_batch_time(&self) -> Duration {
        if self.batch_history.is_empty() {
            Duration::new(0, 0)
        } else {
            let total_time: Duration = self.batch_history.iter().map(|b| b.duration).sum();
            total_time / self.batch_history.len() as u32
        }
    }

    /// エポックサマリーを生成
    /// Generate epoch summary
    pub fn summary(&self) -> String {
        let mut summary = format!("Epoch {} Summary:\n", self.epoch + 1);

        if let Some(ref train_metrics) = self.train_metrics {
            summary.push_str(&format!(
                " - Train Loss: {:.4} ({} batches)\n",
                train_metrics.avg_loss.to_f64().unwrap_or(0.0),
                train_metrics.batch_count
            ));
        }

        if let Some(ref val_metrics) = self.val_metrics {
            summary.push_str(&format!(
                " - Val Loss: {:.4} ({} batches)\n",
                val_metrics.avg_loss.to_f64().unwrap_or(0.0),
                val_metrics.batch_count
            ));
        }

        summary.push_str(&format!(
            " - Duration: {:.2}s\n",
            self.duration.as_secs_f64()
        ));

        if let Some(lr) = self.learning_rate {
            summary.push_str(&format!(" - Learning Rate: {:.2e}\n", lr));
        }

        if !self.batch_history.is_empty() {
            summary.push_str(&format!(
                " - Avg Batch Time: {:.3}s\n",
                self.avg_batch_time().as_secs_f64()
            ));
        }

        summary
    }
}

/// バッチレベルの状態
/// Batch-level state
#[derive(Debug, Clone)]
pub struct BatchState {
    /// バッチ番号
    pub batch: usize,
    /// 損失値
    pub loss: Option<f64>,
    /// その他のメトリクス
    pub metrics: HashMap<String, f64>,
    /// バッチ実行時間
    pub duration: Duration,
    /// バッチサイズ
    pub batch_size: Option<usize>,
    /// バッチ固有のメタデータ
    pub metadata: HashMap<String, String>,
}

impl BatchState {
    /// 新しいバッチ状態を作成
    /// Create a new batch state
    pub fn new(batch: usize) -> Self {
        Self {
            batch,
            loss: None,
            metrics: HashMap::new(),
            duration: Duration::new(0, 0),
            batch_size: None,
            metadata: HashMap::new(),
        }
    }

    /// メトリクスを設定
    /// Set metric
    pub fn set_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// メトリクスを取得
    /// Get metric
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// メタデータを設定
    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// バッチサマリーを生成
    /// Generate batch summary
    pub fn summary(&self) -> String {
        let mut summary = format!("Batch {}: ", self.batch);

        if let Some(loss) = self.loss {
            summary.push_str(&format!("Loss = {:.4}, ", loss));
        }

        if let Some(batch_size) = self.batch_size {
            summary.push_str(&format!("Size = {}, ", batch_size));
        }

        summary.push_str(&format!("Time = {:.3}s", self.duration.as_secs_f64()));

        if !self.metrics.is_empty() {
            summary.push_str(" | ");
            let metric_strings: Vec<String> = self
                .metrics
                .iter()
                .map(|(k, v)| format!("{} = {:.4}", k, v))
                .collect();
            summary.push_str(&metric_strings.join(", "));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_state_creation() {
        let state: TrainingState<f32> = TrainingState::new(10);
        assert_eq!(state.total_epochs, 10);
        assert_eq!(state.current_epoch, 0);
        assert!(state.epoch_history.is_empty());
        assert_eq!(state.progress(), 0.0);
        assert!(!state.is_completed());
    }

    #[test]
    fn test_epoch_state_creation() {
        let epoch_state: EpochState<f32> = EpochState::new(0);
        assert_eq!(epoch_state.epoch, 0);
        assert!(epoch_state.train_metrics.is_none());
        assert!(epoch_state.val_metrics.is_none());
        assert!(epoch_state.batch_history.is_empty());
    }

    #[test]
    fn test_batch_state_creation() {
        let batch_state = BatchState::new(0);
        assert_eq!(batch_state.batch, 0);
        assert!(batch_state.loss.is_none());
        assert!(batch_state.metrics.is_empty());
        assert!(batch_state.batch_size.is_none());
    }

    #[test]
    fn test_training_state_progress() {
        let mut state: TrainingState<f32> = TrainingState::new(10);
        assert_eq!(state.progress(), 0.0);

        // エポックを追加してテスト
        let epoch_state = EpochState::new(0);
        state.add_epoch(epoch_state);
        assert_eq!(state.progress(), 10.0);

        // 5エポック追加
        for i in 1..5 {
            let epoch_state = EpochState::new(i);
            state.add_epoch(epoch_state);
        }
        assert_eq!(state.progress(), 50.0);
    }

    #[test]
    fn test_batch_state_metrics() {
        let mut batch_state = BatchState::new(0);

        batch_state.set_metric("accuracy".to_string(), 0.85);
        batch_state.set_metric("precision".to_string(), 0.82);

        assert_eq!(batch_state.get_metric("accuracy"), Some(0.85));
        assert_eq!(batch_state.get_metric("precision"), Some(0.82));
        assert_eq!(batch_state.get_metric("recall"), None);
    }

    #[test]
    fn test_epoch_state_metadata() {
        let mut epoch_state: EpochState<f32> = EpochState::new(0);

        epoch_state.set_metadata("optimizer".to_string(), "adam".to_string());
        epoch_state.set_metadata("lr_schedule".to_string(), "cosine".to_string());

        assert_eq!(
            epoch_state.metadata.get("optimizer"),
            Some(&"adam".to_string())
        );
        assert_eq!(
            epoch_state.metadata.get("lr_schedule"),
            Some(&"cosine".to_string())
        );
    }

    #[test]
    fn test_training_state_completion() {
        let mut state: TrainingState<f32> = TrainingState::new(3);
        assert!(!state.is_completed());

        // 3エポック追加
        for i in 0..3 {
            let epoch_state = EpochState::new(i);
            state.add_epoch(epoch_state);
        }

        assert!(state.is_completed());
        assert_eq!(state.progress(), 100.0);
    }
}
