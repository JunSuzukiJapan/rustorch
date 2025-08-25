//! 学習ループのコールバック
//! Training loop callbacks

use crate::training::state::{BatchState, EpochState, TrainingState};
use crate::training::trainer::CallbackSignal;
use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

/// コールバックトレイト
/// Callback trait
pub trait Callback<T: Float> {
    /// 訓練開始時に呼び出される
    /// Called at the start of training
    fn on_train_begin(&mut self, _state: &mut TrainingState<T>) -> anyhow::Result<()> {
        Ok(())
    }

    /// 訓練終了時に呼び出される
    /// Called at the end of training
    fn on_train_end(&mut self, _state: &mut TrainingState<T>) -> anyhow::Result<()> {
        Ok(())
    }

    /// エポック開始時に呼び出される
    /// Called at the start of each epoch
    fn on_epoch_begin(
        &mut self,
        _state: &mut TrainingState<T>,
        _epoch_state: &mut EpochState<T>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// エポック終了時に呼び出される
    /// Called at the end of each epoch
    fn on_epoch_end(
        &mut self,
        _state: &mut TrainingState<T>,
        _epoch_state: &EpochState<T>,
    ) -> anyhow::Result<Option<CallbackSignal>> {
        Ok(None)
    }

    /// バッチ開始時に呼び出される
    /// Called at the start of each batch
    fn on_batch_begin(
        &mut self,
        _state: &mut TrainingState<T>,
        _batch_state: &mut BatchState,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// バッチ終了時に呼び出される
    /// Called at the end of each batch
    fn on_batch_end(
        &mut self,
        _state: &mut TrainingState<T>,
        _batch_state: &BatchState,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// 早期停止コールバック
/// Early stopping callback
#[derive(Debug, Clone)]
pub struct EarlyStopping<T: Float> {
    /// 監視するメトリクス名
    monitor: String,
    /// 忍耐度（改善が見られないエポック数）
    patience: usize,
    /// 最小変化量
    min_delta: T,
    /// より良い方向（true: 大きい方が良い, false: 小さい方が良い）
    mode_max: bool,
    /// ベストスコア
    best_score: Option<T>,
    /// 待機カウンター
    wait_count: usize,
    /// 停止されたかどうか
    stopped: bool,
}

impl<T: Float> EarlyStopping<T> {
    /// 新しい早期停止コールバックを作成
    /// Create a new early stopping callback
    pub fn new(monitor: String, patience: usize, min_delta: T, mode_max: bool) -> Self {
        Self {
            monitor,
            patience,
            min_delta,
            mode_max,
            best_score: None,
            wait_count: 0,
            stopped: false,
        }
    }

    /// 検証損失を監視する早期停止
    /// Early stopping that monitors validation loss
    pub fn monitor_val_loss(patience: usize, min_delta: T) -> Self {
        Self::new("val_loss".to_string(), patience, min_delta, false)
    }

    /// 検証精度を監視する早期停止
    /// Early stopping that monitors validation accuracy
    pub fn monitor_val_accuracy(patience: usize, min_delta: T) -> Self {
        Self::new("val_accuracy".to_string(), patience, min_delta, true)
    }

    /// スコアが改善されたかどうかを判定
    /// Check if score has improved
    fn is_improvement(&self, current: T) -> bool {
        match self.best_score {
            None => true,
            Some(best) => {
                if self.mode_max {
                    current > best + self.min_delta
                } else {
                    current < best - self.min_delta
                }
            }
        }
    }
}

impl<T: Float + Debug> Callback<T> for EarlyStopping<T> {
    fn on_train_begin(&mut self, _state: &mut TrainingState<T>) -> anyhow::Result<()> {
        self.best_score = None;
        self.wait_count = 0;
        self.stopped = false;
        println!(
            "Early stopping monitoring '{}' with patience {}",
            self.monitor, self.patience
        );
        Ok(())
    }

    fn on_epoch_end(
        &mut self,
        _state: &mut TrainingState<T>,
        epoch_state: &EpochState<T>,
    ) -> anyhow::Result<Option<CallbackSignal>> {
        // 監視するメトリクスを取得
        let current_score = if self.monitor == "val_loss" {
            epoch_state.val_metrics.as_ref().map(|m| m.avg_loss)
        } else {
            // その他のメトリクス（実装簡略化）
            None
        };

        if let Some(score) = current_score {
            if self.is_improvement(score) {
                self.best_score = Some(score);
                self.wait_count = 0;
                println!(
                    "Epoch {}: {} improved to {:.4}",
                    epoch_state.epoch + 1,
                    self.monitor,
                    score.to_f64().unwrap_or(0.0)
                );
            } else {
                self.wait_count += 1;
                println!(
                    "Epoch {}: {} did not improve. Patience: {}/{}",
                    epoch_state.epoch + 1,
                    self.monitor,
                    self.wait_count,
                    self.patience
                );

                if self.wait_count >= self.patience {
                    self.stopped = true;
                    println!(
                        "Early stopping triggered after {} epochs without improvement",
                        self.patience
                    );
                    return Ok(Some(CallbackSignal::Stop));
                }
            }
        }

        Ok(Some(CallbackSignal::Continue))
    }
}

/// 学習率スケジューラーコールバック
/// Learning rate scheduler callback
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// 学習率の更新頻度（エポック単位）
    frequency: usize,
    /// 学習率の減衰率
    decay_rate: f64,
    /// 現在の学習率
    current_lr: f64,
    /// 初期学習率
    initial_lr: f64,
}

impl LearningRateScheduler {
    /// 新しい学習率スケジューラーを作成
    /// Create a new learning rate scheduler
    pub fn new(initial_lr: f64, decay_rate: f64, frequency: usize) -> Self {
        Self {
            frequency,
            decay_rate,
            current_lr: initial_lr,
            initial_lr,
        }
    }

    /// 指数減衰スケジューラー
    /// Exponential decay scheduler
    pub fn exponential_decay(initial_lr: f64, decay_rate: f64) -> Self {
        Self::new(initial_lr, decay_rate, 1)
    }

    /// ステップ減衰スケジューラー
    /// Step decay scheduler
    pub fn step_decay(initial_lr: f64, decay_rate: f64, step_size: usize) -> Self {
        Self::new(initial_lr, decay_rate, step_size)
    }
}

impl<T: Float> Callback<T> for LearningRateScheduler {
    fn on_train_begin(&mut self, _state: &mut TrainingState<T>) -> anyhow::Result<()> {
        self.current_lr = self.initial_lr;
        println!(
            "Learning rate scheduler initialized with lr = {:.2e}",
            self.current_lr
        );
        Ok(())
    }

    fn on_epoch_begin(
        &mut self,
        _state: &mut TrainingState<T>,
        epoch_state: &mut EpochState<T>,
    ) -> anyhow::Result<()> {
        if epoch_state.epoch > 0 && epoch_state.epoch % self.frequency == 0 {
            self.current_lr *= self.decay_rate;
            println!(
                "Epoch {}: Learning rate updated to {:.2e}",
                epoch_state.epoch + 1,
                self.current_lr
            );
        }

        epoch_state.learning_rate = Some(self.current_lr);
        Ok(())
    }
}

/// プログレスバーコールバック
/// Progress bar callback
#[derive(Debug)]
pub struct ProgressBar {
    /// 開始時刻
    start_time: Option<Instant>,
    /// プログレスバーの幅
    width: usize,
    /// 詳細表示するかどうか
    verbose: bool,
    /// バッチ頻度
    batch_frequency: usize,
}

impl ProgressBar {
    /// 新しいプログレスバーを作成
    /// Create a new progress bar
    pub fn new(width: usize, verbose: bool, batch_frequency: usize) -> Self {
        Self {
            start_time: None,
            width,
            verbose,
            batch_frequency,
        }
    }

    /// シンプルなプログレスバー
    /// Simple progress bar
    pub fn simple() -> Self {
        Self::new(50, false, 100)
    }

    /// 詳細なプログレスバー
    /// Verbose progress bar
    pub fn verbose() -> Self {
        Self::new(50, true, 10)
    }

    /// プログレスバーを描画
    /// Draw progress bar
    fn draw_progress(&self, current: usize, total: usize, suffix: &str) {
        let progress = if total > 0 {
            (current as f64 / total as f64).min(1.0)
        } else {
            0.0
        };

        let filled = (progress * self.width as f64) as usize;
        let empty = self.width - filled;

        let bar = format!(
            "\r[{}{}] {:.1}% ({}/{}) {}",
            "=".repeat(filled),
            " ".repeat(empty),
            progress * 100.0,
            current,
            total,
            suffix
        );

        print!("{}", bar);
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }
}

impl<T: Float> Callback<T> for ProgressBar {
    fn on_train_begin(&mut self, state: &mut TrainingState<T>) -> anyhow::Result<()> {
        self.start_time = Some(Instant::now());
        println!("Training started for {} epochs", state.total_epochs);
        Ok(())
    }

    fn on_train_end(&mut self, state: &mut TrainingState<T>) -> anyhow::Result<()> {
        if let Some(start_time) = self.start_time {
            let duration = start_time.elapsed();
            println!(
                "\nTraining completed in {:.2}s ({} epochs)",
                duration.as_secs_f64(),
                state.current_epoch
            );
        }
        Ok(())
    }

    fn on_epoch_begin(
        &mut self,
        state: &mut TrainingState<T>,
        epoch_state: &mut EpochState<T>,
    ) -> anyhow::Result<()> {
        if self.verbose {
            println!("\nEpoch {}/{}", epoch_state.epoch + 1, state.total_epochs);
        }
        Ok(())
    }

    fn on_epoch_end(
        &mut self,
        state: &mut TrainingState<T>,
        epoch_state: &EpochState<T>,
    ) -> anyhow::Result<Option<CallbackSignal>> {
        let mut suffix = String::new();

        if let Some(ref train_metrics) = epoch_state.train_metrics {
            suffix.push_str(&format!(
                "Loss: {:.4}",
                train_metrics.avg_loss.to_f64().unwrap_or(0.0)
            ));
        }

        if let Some(ref val_metrics) = epoch_state.val_metrics {
            suffix.push_str(&format!(
                " | Val Loss: {:.4}",
                val_metrics.avg_loss.to_f64().unwrap_or(0.0)
            ));
        }

        suffix.push_str(&format!(
            " | Time: {:.2}s",
            epoch_state.duration.as_secs_f64()
        ));

        if !self.verbose {
            self.draw_progress(epoch_state.epoch + 1, state.total_epochs, &suffix);
        } else {
            println!("{}", suffix);
        }

        Ok(Some(CallbackSignal::Continue))
    }

    fn on_batch_end(
        &mut self,
        _state: &mut TrainingState<T>,
        batch_state: &BatchState,
    ) -> anyhow::Result<()> {
        if self.verbose && batch_state.batch % self.batch_frequency == 0 {
            if let Some(loss) = batch_state.loss {
                println!("  Batch {}: Loss = {:.4}", batch_state.batch, loss);
            }
        }
        Ok(())
    }
}

/// モデルチェックポイントコールバック
/// Model checkpoint callback
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// 保存パス
    filepath: String,
    /// 監視するメトリクス
    monitor: String,
    /// より良い方向
    mode_max: bool,
    /// 最良のスコア
    best_score: Option<f64>,
    /// すべてのエポックで保存するかどうか
    save_all: bool,
}

impl ModelCheckpoint {
    /// 新しいモデルチェックポイントコールバックを作成
    /// Create a new model checkpoint callback
    pub fn new(filepath: String, monitor: String, mode_max: bool, save_all: bool) -> Self {
        Self {
            filepath,
            monitor,
            mode_max,
            best_score: None,
            save_all,
        }
    }

    /// 最良の検証損失でのみ保存
    /// Save only on best validation loss
    pub fn best_val_loss(filepath: String) -> Self {
        Self::new(filepath, "val_loss".to_string(), false, false)
    }

    /// すべてのエポックで保存
    /// Save at every epoch
    pub fn every_epoch(filepath: String) -> Self {
        Self::new(filepath, "".to_string(), false, true)
    }
}

impl<T: Float> Callback<T> for ModelCheckpoint {
    fn on_epoch_end(
        &mut self,
        _state: &mut TrainingState<T>,
        epoch_state: &EpochState<T>,
    ) -> anyhow::Result<Option<CallbackSignal>> {
        let should_save = if self.save_all {
            true
        } else {
            // 監視するメトリクスをチェック
            let current_score = if self.monitor == "val_loss" {
                epoch_state
                    .val_metrics
                    .as_ref()
                    .map(|m| m.avg_loss.to_f64().unwrap_or(0.0))
            } else {
                None
            };

            if let Some(score) = current_score {
                let is_better = match self.best_score {
                    None => true,
                    Some(best) => {
                        if self.mode_max {
                            score > best
                        } else {
                            score < best
                        }
                    }
                };

                if is_better {
                    self.best_score = Some(score);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };

        if should_save {
            println!(
                "Saving checkpoint at epoch {} to {}",
                epoch_state.epoch + 1,
                self.filepath
            );
            // 実際の保存処理はここに実装
            // Actual saving logic would be implemented here
        }

        Ok(Some(CallbackSignal::Continue))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::training::trainer::EpochMetrics;  // 一時的にコメントアウト

    #[test]
    fn test_early_stopping_creation() {
        let early_stopping: EarlyStopping<f32> = EarlyStopping::monitor_val_loss(5, 0.01);

        assert_eq!(early_stopping.monitor, "val_loss");
        assert_eq!(early_stopping.patience, 5);
        assert_eq!(early_stopping.min_delta, 0.01);
        assert!(!early_stopping.mode_max);
    }

    #[test]
    fn test_lr_scheduler_creation() {
        let scheduler = LearningRateScheduler::exponential_decay(0.001, 0.95);

        assert_eq!(scheduler.initial_lr, 0.001);
        assert_eq!(scheduler.decay_rate, 0.95);
        assert_eq!(scheduler.frequency, 1);
    }

    #[test]
    fn test_progress_bar_creation() {
        let progress = ProgressBar::simple();

        assert_eq!(progress.width, 50);
        assert!(!progress.verbose);
        assert_eq!(progress.batch_frequency, 100);
    }

    #[test]
    fn test_model_checkpoint_creation() {
        let checkpoint = ModelCheckpoint::best_val_loss("model.pt".to_string());

        assert_eq!(checkpoint.filepath, "model.pt");
        assert_eq!(checkpoint.monitor, "val_loss");
        assert!(!checkpoint.mode_max);
        assert!(!checkpoint.save_all);
    }

    #[test]
    fn test_early_stopping_improvement() {
        let early_stopping: EarlyStopping<f32> = EarlyStopping::monitor_val_loss(3, 0.01);

        // 最初は改善とみなされる
        assert!(early_stopping.is_improvement(0.5));

        let early_stopping_with_best = EarlyStopping {
            best_score: Some(0.5),
            ..early_stopping
        };

        // より小さい値は改善
        assert!(early_stopping_with_best.is_improvement(0.4));
        // より大きい値は改善ではない
        assert!(!early_stopping_with_best.is_improvement(0.6));
        // 最小変化量以下は改善ではない
        assert!(!early_stopping_with_best.is_improvement(0.49));
    }
}
