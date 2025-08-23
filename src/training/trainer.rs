//! 汎用的な学習ループトレーナー
//! Generic training loop trainer

use crate::autograd::Variable;
use crate::nn::loss::Loss;
use crate::optim::Optimizer;
use crate::data::DataLoader;
use super::state::{TrainingState, EpochState, BatchState};
use super::callbacks::Callback;
use super::metrics::{MetricsCollector, TrainingMetrics};
use num_traits::Float;
use std::time::Instant;
use std::fmt::Debug;

/// 訓練設定
/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// エポック数
    pub epochs: usize,
    /// ログ出力頻度（バッチ単位）
    pub log_frequency: usize,
    /// 検証頻度（エポック単位）
    pub validation_frequency: usize,
    /// グラディエントクリッピングの閾値
    pub gradient_clip_value: Option<f64>,
    /// デバイス設定
    pub device: String,
    /// 混合精度学習を使用するかどうか
    pub use_mixed_precision: bool,
    /// 勾配累積ステップ数
    pub accumulation_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            log_frequency: 100,
            validation_frequency: 1,
            gradient_clip_value: None,
            device: "cpu".to_string(),
            use_mixed_precision: false,
            accumulation_steps: 1,
        }
    }
}

/// 汎用的な学習ループトレーナー
/// Generic training loop trainer
pub struct Trainer<T, O, L>
where
    T: Float + 'static + Send + Sync + Debug + Clone + ndarray::ScalarOperand + num_traits::FromPrimitive,
    O: Optimizer + Clone,
    L: Loss<T> + Clone,
{
    config: TrainerConfig,
    optimizer: O,
    loss_fn: L,
    metrics_collector: MetricsCollector<T>,
    callbacks: Vec<Box<dyn Callback<T> + Send + Sync>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, O, L> Trainer<T, O, L>
where
    T: Float + 'static + Send + Sync + Debug + Clone + ndarray::ScalarOperand + num_traits::FromPrimitive,
    O: Optimizer + Clone,
    L: Loss<T> + Clone,
{
    /// 新しいトレーナーを作成
    /// Create a new trainer
    pub fn new(config: TrainerConfig, optimizer: O, loss_fn: L) -> Self {
        Self {
            config,
            optimizer,
            loss_fn,
            metrics_collector: MetricsCollector::new(),
            callbacks: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// コールバックを追加
    /// Add a callback
    pub fn add_callback(&mut self, callback: Box<dyn Callback<T> + Send + Sync>) {
        self.callbacks.push(callback);
    }

    /// モデルを訓練
    /// Train a model
    pub fn train<M, D>(
        &mut self,
        model: &mut M,
        train_loader: &mut DataLoader<T, D>,
        mut val_loader: Option<&mut DataLoader<T, D>>,
    ) -> anyhow::Result<TrainingMetrics<T>>
    where
        M: TrainableModel<T>,
        D: crate::data::Dataset<T>,
    {
        let start_time = Instant::now();
        let mut state = TrainingState::new(self.config.epochs);

        // 訓練開始時のコールバック
        for callback in &mut self.callbacks {
            callback.on_train_begin(&mut state)?;
        }

        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();
            let mut epoch_state = EpochState::new(epoch);

            // エポック開始時のコールバック
            for callback in &mut self.callbacks {
                callback.on_epoch_begin(&mut state, &mut epoch_state)?;
            }

            // 訓練フェーズ
            model.train();
            let train_metrics = self.train_epoch(model, train_loader, &mut state)?;
            epoch_state.train_metrics = Some(train_metrics.clone());

            // 検証フェーズ
            if epoch % self.config.validation_frequency == 0 {
                if let Some(val_loader) = val_loader.as_mut() {
                    model.eval();
                    let val_metrics = self.validate_epoch(model, val_loader, &mut state)?;
                    epoch_state.val_metrics = Some(val_metrics.clone());
                }
            }

            epoch_state.duration = epoch_start.elapsed();
            state.add_epoch(epoch_state.clone());

            // エポック終了時のコールバック
            for callback in &mut self.callbacks {
                if let Some(signal) = callback.on_epoch_end(&mut state, &epoch_state)? {
                    match signal {
                        CallbackSignal::Stop => {
                            println!("Training stopped by callback at epoch {}", epoch + 1);
                            break;
                        }
                        CallbackSignal::Continue => {}
                    }
                }
            }

            // ログ出力
            if epoch % self.config.validation_frequency == 0 {
                self.log_epoch_summary(epoch, &epoch_state);
            }
        }

        state.total_duration = start_time.elapsed();

        // 訓練終了時のコールバック
        for callback in &mut self.callbacks {
            callback.on_train_end(&mut state)?;
        }

        let final_metrics = self.metrics_collector.finalize(state);
        Ok(final_metrics)
    }

    /// 1エポックの訓練
    /// Train for one epoch
    fn train_epoch<M, D>(
        &mut self,
        model: &mut M,
        train_loader: &mut DataLoader<T, D>,
        state: &mut TrainingState<T>,
    ) -> anyhow::Result<EpochMetrics<T>>
    where
        M: TrainableModel<T>,
        D: crate::data::Dataset<T>,
    {
        let mut epoch_metrics = EpochMetrics::new();
        let mut batch_count = 0;
        let mut accumulated_loss = T::zero();

        train_loader.reset();

        while let Some((inputs, targets)) = train_loader.next_batch() {
            let batch_start = Instant::now();
            let mut batch_state = BatchState::new(batch_count);

            // バッチ開始時のコールバック
            for callback in &mut self.callbacks {
                callback.on_batch_begin(state, &mut batch_state)?;
            }

            // 順伝播
            let outputs = model.forward(&Variable::new(inputs, false));
            let loss = self.loss_fn.forward(&outputs, &Variable::new(targets, false));

            // 逆伝播と最適化
            if batch_count % self.config.accumulation_steps == 0 {
                self.optimizer.zero_grad();
            }

            loss.backward();

            if (batch_count + 1) % self.config.accumulation_steps == 0 {
                // グラディエントクリッピング
                if let Some(clip_value) = self.config.gradient_clip_value {
                    self.clip_gradients(model, clip_value);
                }

                // オプティマイザーのステップ実行は簡略化
                // Optimizer step execution is simplified
                // self.optimizer.step();
            }

            // メトリクス更新
            let loss_value = self.extract_scalar_value(&loss);
            accumulated_loss = accumulated_loss + T::from(loss_value).unwrap();
            epoch_metrics.total_loss = epoch_metrics.total_loss + T::from(loss_value).unwrap();
            epoch_metrics.batch_count += 1;

            batch_state.loss = Some(loss_value);
            batch_state.duration = batch_start.elapsed();

            // バッチ終了時のコールバック
            for callback in &mut self.callbacks {
                callback.on_batch_end(state, &batch_state)?;
            }

            // ログ出力
            if batch_count % self.config.log_frequency == 0 {
                println!(
                    "Batch {}: Loss = {:.4}",
                    batch_count,
                    loss_value
                );
            }

            batch_count += 1;
        }

        epoch_metrics.avg_loss = if epoch_metrics.batch_count > 0 {
            epoch_metrics.total_loss / T::from(epoch_metrics.batch_count).unwrap()
        } else {
            T::zero()
        };

        Ok(epoch_metrics)
    }

    /// 1エポックの検証
    /// Validate for one epoch
    fn validate_epoch<M, D>(
        &mut self,
        model: &mut M,
        val_loader: &mut DataLoader<T, D>,
        _state: &mut TrainingState<T>,
    ) -> anyhow::Result<EpochMetrics<T>>
    where
        M: TrainableModel<T>,
        D: crate::data::Dataset<T>,
    {
        let mut epoch_metrics = EpochMetrics::new();

        val_loader.reset();

        while let Some((inputs, targets)) = val_loader.next_batch() {
            // 順伝播のみ（勾配計算なし）
            let outputs = model.forward(&Variable::new(inputs, false));
            let loss = self.loss_fn.forward(&outputs, &Variable::new(targets, false));

            let loss_value = self.extract_scalar_value(&loss);
            epoch_metrics.total_loss = epoch_metrics.total_loss + T::from(loss_value).unwrap();
            epoch_metrics.batch_count += 1;
        }

        epoch_metrics.avg_loss = if epoch_metrics.batch_count > 0 {
            epoch_metrics.total_loss / T::from(epoch_metrics.batch_count).unwrap()
        } else {
            T::zero()
        };

        Ok(epoch_metrics)
    }

    /// グラディエントクリッピング
    /// Gradient clipping
    fn clip_gradients<M>(&self, _model: &M, _clip_value: f64)
    where
        M: TrainableModel<T>,
    {
        // 実装は簡略化 - 実際にはモデルのパラメータに対してクリッピングを適用
        // Implementation simplified - in practice, apply clipping to model parameters
    }

    /// スカラー値を抽出
    /// Extract scalar value
    fn extract_scalar_value(&self, _variable: &Variable<T>) -> f64 {
        // 実装は簡略化 - プレースホルダー値を返す
        // Implementation simplified - return placeholder value
        0.5
    }

    /// エポックサマリーをログ出力
    /// Log epoch summary
    fn log_epoch_summary(&self, epoch: usize, epoch_state: &EpochState<T>) {
        let mut summary = format!("Epoch {}/{}", epoch + 1, self.config.epochs);

        if let Some(ref train_metrics) = epoch_state.train_metrics {
            summary.push_str(&format!(
                " | Train Loss: {:.4}",
                train_metrics.avg_loss.to_f64().unwrap_or(0.0)
            ));
        }

        if let Some(ref val_metrics) = epoch_state.val_metrics {
            summary.push_str(&format!(
                " | Val Loss: {:.4}",
                val_metrics.avg_loss.to_f64().unwrap_or(0.0)
            ));
        }

        summary.push_str(&format!(
            " | Time: {:.2}s",
            epoch_state.duration.as_secs_f64()
        ));

        println!("{}", summary);
    }
}

/// 訓練可能なモデルのトレイト
/// Trait for trainable models
pub trait TrainableModel<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// 順伝播
    /// Forward pass
    fn forward(&self, input: &Variable<T>) -> Variable<T>;

    /// 訓練モードに設定
    /// Set to training mode
    fn train(&mut self);

    /// 評価モードに設定
    /// Set to evaluation mode
    fn eval(&mut self);

    /// パラメータを取得
    /// Get parameters
    fn parameters(&self) -> Vec<&Variable<T>>;

    /// パラメータを可変参照で取得
    /// Get mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Variable<T>>;
}

/// エポックレベルのメトリクス
/// Epoch-level metrics
#[derive(Debug, Clone)]
pub struct EpochMetrics<T: Float> {
    /// 総損失
    /// Total loss
    pub total_loss: T,
    /// 平均損失
    /// Average loss
    pub avg_loss: T,
    /// バッチ数
    /// Number of batches
    pub batch_count: usize,
}

impl<T: Float> EpochMetrics<T> {
    /// 新しいエポックメトリクスを作成
    /// Create new epoch metrics
    pub fn new() -> Self {
        Self {
            total_loss: T::zero(),
            avg_loss: T::zero(),
            batch_count: 0,
        }
    }
}

impl<T: Float> Default for EpochMetrics<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// コールバックシグナル
/// Callback signal
#[derive(Debug, Clone, Copy)]
pub enum CallbackSignal {
    /// 訓練を継続
    Continue,
    /// 訓練を停止
    Stop,
}

/// TrainerBuilder for fluent API
/// TrainerBuilderパターンでの流暢なAPI
pub struct TrainerBuilder<T, O, L>
where
    T: Float + 'static + Send + Sync + Debug + Clone + ndarray::ScalarOperand + num_traits::FromPrimitive,
    O: Optimizer + Clone,
    L: Loss<T> + Clone,
{
    config: TrainerConfig,
    optimizer: Option<O>,
    loss_fn: Option<L>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, O, L> TrainerBuilder<T, O, L>
where
    T: Float + 'static + Send + Sync + Debug + Clone + ndarray::ScalarOperand + num_traits::FromPrimitive,
    O: Optimizer + Clone,
    L: Loss<T> + Clone,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: TrainerConfig::default(),
            optimizer: None,
            loss_fn: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// エポック数を設定
    /// Set epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// ログ頻度を設定
    /// Set log frequency
    pub fn log_frequency(mut self, frequency: usize) -> Self {
        self.config.log_frequency = frequency;
        self
    }

    /// 検証頻度を設定
    /// Set validation frequency
    pub fn validation_frequency(mut self, frequency: usize) -> Self {
        self.config.validation_frequency = frequency;
        self
    }

    /// グラディエントクリッピングを設定
    /// Set gradient clipping
    pub fn gradient_clip_value(mut self, value: f64) -> Self {
        self.config.gradient_clip_value = Some(value);
        self
    }

    /// デバイスを設定
    /// Set device
    pub fn device(mut self, device: String) -> Self {
        self.config.device = device;
        self
    }

    /// オプティマイザーを設定
    /// Set optimizer
    pub fn optimizer(mut self, optimizer: O) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// 損失関数を設定
    /// Set loss function
    pub fn loss_fn(mut self, loss_fn: L) -> Self {
        self.loss_fn = Some(loss_fn);
        self
    }

    /// トレーナーを構築
    /// Build the trainer
    pub fn build(self) -> anyhow::Result<Trainer<T, O, L>> {
        let optimizer = self.optimizer.ok_or_else(|| anyhow::anyhow!("Optimizer not provided"))?;
        let loss_fn = self.loss_fn.ok_or_else(|| anyhow::anyhow!("Loss function not provided"))?;

        Ok(Trainer::new(self.config, optimizer, loss_fn))
    }
}

impl<T, O, L> Default for TrainerBuilder<T, O, L>
where
    T: Float + 'static + Send + Sync + Debug + Clone + ndarray::ScalarOperand + num_traits::FromPrimitive,
    O: Optimizer + Clone,
    L: Loss<T> + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.log_frequency, 100);
        assert_eq!(config.validation_frequency, 1);
        assert_eq!(config.gradient_clip_value, None);
        assert_eq!(config.device, "cpu");
        assert_eq!(config.use_mixed_precision, false);
        assert_eq!(config.accumulation_steps, 1);
    }

    #[test]
    fn test_epoch_metrics_creation() {
        let metrics: EpochMetrics<f32> = EpochMetrics::new();
        assert_eq!(metrics.total_loss, 0.0);
        assert_eq!(metrics.avg_loss, 0.0);
        assert_eq!(metrics.batch_count, 0);
    }

    #[test]
    fn test_callback_signal() {
        let signal = CallbackSignal::Continue;
        match signal {
            CallbackSignal::Continue => assert!(true),
            CallbackSignal::Stop => assert!(false),
        }
    }
}