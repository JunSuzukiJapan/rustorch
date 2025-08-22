//! 学習ループのテストとベンチマーク
//! Training loop tests and benchmarks

use super::*;
use crate::tensor::Tensor;
use crate::autograd::Variable;
use crate::nn::loss::MSELoss;
use crate::optim::sgd::SGD;
use crate::data::{TensorDataset, DataLoader};
use std::time::{Duration, Instant};

/// テスト用のシンプルなモデル
/// Simple model for testing
#[derive(Debug, Clone)]
pub struct SimpleTestModel {
    weights: Variable<f32>,
    bias: Variable<f32>,
    training: bool,
}

impl SimpleTestModel {
    /// 新しいテストモデルを作成
    /// Create a new test model
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Variable::new(Tensor::randn(&[input_size, output_size]));
        let bias = Variable::new(Tensor::randn(&[output_size]));
        
        Self {
            weights,
            bias,
            training: true,
        }
    }
}

impl trainer::TrainableModel<f32> for SimpleTestModel {
    fn forward(&self, input: &Variable<f32>) -> Variable<f32> {
        // Simplified forward pass: input @ weights + bias
        let output_data = if let (Some(input_data), Some(weight_data), Some(bias_data)) = (
            input.data().as_slice(),
            self.weights.data().as_slice(),
            self.bias.data().as_slice(),
        ) {
            let input_shape = input.data().shape();
            let weight_shape = self.weights.data().shape();
            
            if input_shape.len() >= 2 && weight_shape.len() == 2 {
                let batch_size = input_shape[0];
                let input_size = input_shape[1];
                let output_size = weight_shape[1];
                
                if input_size == weight_shape[0] && output_size == bias_data.len() {
                    let mut output = vec![0.0f32; batch_size * output_size];
                    
                    for b in 0..batch_size {
                        for o in 0..output_size {
                            let mut sum = bias_data[o];
                            for i in 0..input_size {
                                sum += input_data[b * input_size + i] * weight_data[i * output_size + o];
                            }
                            output[b * output_size + o] = sum;
                        }
                    }
                    
                    Tensor::from_vec(output, vec![batch_size, output_size])
                } else {
                    Tensor::zeros(&[1, 1])
                }
            } else {
                Tensor::zeros(&[1, 1])
            }
        } else {
            Tensor::zeros(&[1, 1])
        };

        Variable::new(output_data)
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn parameters(&self) -> Vec<&Variable<f32>> {
        vec![&self.weights, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Variable<f32>> {
        vec![&mut self.weights, &mut self.bias]
    }
}

/// テスト用のデータセットを作成
/// Create a test dataset
pub fn create_test_dataset(size: usize, input_dim: usize, output_dim: usize) -> TensorDataset<f32> {
    let features: Vec<Tensor<f32>> = (0..size)
        .map(|_| Tensor::randn(&[input_dim]))
        .collect();
    
    let targets: Vec<Tensor<f32>> = (0..size)
        .map(|_| Tensor::randn(&[output_dim]))
        .collect();
    
    TensorDataset::new(features, targets).unwrap()
}

/// 基本的な学習ループのテスト
/// Basic training loop test
#[cfg(test)]
mod trainer_tests {
    use super::*;

    #[test]
    fn test_trainer_config_creation() {
        let config = TrainerConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.log_frequency, 100);
        assert_eq!(config.validation_frequency, 1);
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_trainer_builder() {
        let optimizer = SGD::new(0.01);
        let loss_fn = MSELoss::new();
        
        let trainer_result = TrainerBuilder::new()
            .epochs(5)
            .log_frequency(50)
            .validation_frequency(2)
            .gradient_clip_value(1.0)
            .optimizer(optimizer)
            .loss_fn(loss_fn)
            .build();
        
        assert!(trainer_result.is_ok());
        let trainer = trainer_result.unwrap();
        assert_eq!(trainer.config.epochs, 5);
        assert_eq!(trainer.config.log_frequency, 50);
        assert_eq!(trainer.config.validation_frequency, 2);
        assert_eq!(trainer.config.gradient_clip_value, Some(1.0));
    }

    #[test]
    fn test_simple_model() {
        let model = SimpleTestModel::new(4, 2);
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]));
        
        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[1, 2]);
        
        let params = model.parameters();
        assert_eq!(params.len(), 2); // weights and bias
    }

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::<f32>::new(5);
        assert_eq!(state.total_epochs, 5);
        assert_eq!(state.current_epoch, 0);
        assert!(!state.is_completed());
        
        let epoch_state = EpochState::new(0);
        state.add_epoch(epoch_state);
        assert_eq!(state.current_epoch, 1);
        assert_eq!(state.progress(), 20.0);
    }

    #[test]
    fn test_early_stopping_callback() {
        let early_stopping: callbacks::EarlyStopping<f32> = 
            callbacks::EarlyStopping::monitor_val_loss(3, 0.01);
        
        assert_eq!(early_stopping.monitor, "val_loss");
        assert_eq!(early_stopping.patience, 3);
        assert_eq!(early_stopping.min_delta, 0.01);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::<f32>::new();
        
        let predictions = Variable::new(Tensor::from_vec(vec![0.8, 0.3, 0.9, 0.1], vec![4]));
        let targets = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]));
        
        let accuracy = collector.accuracy(&predictions, &targets);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        
        let metrics = collector.calculate_metrics(&predictions, &targets);
        assert!(metrics.contains_key("accuracy"));
        assert!(metrics.contains_key("precision"));
        assert!(metrics.contains_key("recall"));
        assert!(metrics.contains_key("f1_score"));
    }

    #[test]
    fn test_checkpoint_manager() -> anyhow::Result<()> {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new()?;
        let config = checkpoint::SaveConfig {
            save_dir: temp_dir.path().to_path_buf(),
            max_checkpoints: 3,
            save_best_only: false,
            ..checkpoint::SaveConfig::default()
        };
        
        let manager = checkpoint::CheckpointManager::new(config)?;
        assert!(manager.all_checkpoints().is_empty());
        
        Ok(())
    }
}

/// パフォーマンステスト
/// Performance tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_training_performance() {
        let dataset = create_test_dataset(1000, 10, 5);
        let mut train_loader = DataLoader::new(&dataset, 32, true);
        
        let mut model = SimpleTestModel::new(10, 5);
        let optimizer = SGD::new(0.01);
        let loss_fn = MSELoss::new();
        
        let config = TrainerConfig {
            epochs: 3,
            log_frequency: 100,
            validation_frequency: 1,
            ..TrainerConfig::default()
        };
        
        let mut trainer = Trainer::new(config, optimizer, loss_fn);
        
        let start_time = Instant::now();
        let result = trainer.train(&mut model, &mut train_loader, None);
        let duration = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(duration < Duration::from_secs(10)); // Should complete within 10 seconds
        
        println!("Training completed in: {:?}", duration);
    }

    #[test]
    fn test_metrics_computation_performance() {
        let collector = MetricsCollector::<f32>::new();
        
        // Large dataset for performance testing
        let size = 10000;
        let predictions = Variable::new(Tensor::randn(&[size]));
        let targets = Variable::new(Tensor::randn(&[size]));
        
        let start_time = Instant::now();
        let _metrics = collector.calculate_metrics(&predictions, &targets);
        let duration = start_time.elapsed();
        
        assert!(duration < Duration::from_millis(100)); // Should be fast
        println!("Metrics computation for {} samples: {:?}", size, duration);
    }

    #[test]
    fn test_callback_overhead() {
        let mut state = TrainingState::<f32>::new(100);
        let mut epoch_state = EpochState::new(0);
        
        let mut callbacks: Vec<Box<dyn callbacks::Callback<f32> + Send + Sync>> = vec![
            Box::new(callbacks::EarlyStopping::monitor_val_loss(10, 0.01)),
            Box::new(callbacks::LearningRateScheduler::exponential_decay(0.01, 0.95)),
            Box::new(callbacks::ProgressBar::simple()),
        ];
        
        let start_time = Instant::now();
        
        // Simulate callback overhead
        for callback in &mut callbacks {
            let _ = callback.on_epoch_begin(&mut state, &mut epoch_state);
            let _ = callback.on_epoch_end(&mut state, &epoch_state);
        }
        
        let duration = start_time.elapsed();
        assert!(duration < Duration::from_millis(10)); // Should have minimal overhead
        
        println!("Callback overhead for 3 callbacks: {:?}", duration);
    }
}

/// 統合テスト
/// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_training_pipeline() -> anyhow::Result<()> {
        // データセットを作成
        let train_dataset = create_test_dataset(200, 8, 3);
        let val_dataset = create_test_dataset(50, 8, 3);
        
        let mut train_loader = DataLoader::new(&train_dataset, 16, true);
        let mut val_loader = DataLoader::new(&val_dataset, 16, false);
        
        // モデルを作成
        let mut model = SimpleTestModel::new(8, 3);
        
        // トレーナーを構築
        let optimizer = SGD::new(0.01);
        let loss_fn = MSELoss::new();
        
        let mut trainer = TrainerBuilder::new()
            .epochs(5)
            .log_frequency(10)
            .validation_frequency(1)
            .optimizer(optimizer)
            .loss_fn(loss_fn)
            .build()?;
        
        // コールバックを追加
        trainer.add_callback(Box::new(callbacks::EarlyStopping::monitor_val_loss(3, 0.001)));
        trainer.add_callback(Box::new(callbacks::LearningRateScheduler::exponential_decay(0.01, 0.9)));
        
        // 訓練実行
        let result = trainer.train(&mut model, &mut train_loader, Some(&mut val_loader))?;
        
        // 結果検証
        assert!(!result.epoch_history.is_empty());
        assert!(result.training_state.total_duration > Duration::new(0, 0));
        
        println!("Training completed successfully:");
        println!("{}", result.summary());
        
        Ok(())
    }

    #[test]
    fn test_checkpoint_integration() -> anyhow::Result<()> {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new()?;
        
        // チェックポイント管理者を作成
        let config = checkpoint::SaveConfig {
            save_dir: temp_dir.path().to_path_buf(),
            max_checkpoints: 3,
            save_best_only: true,
            monitor: "val_loss".to_string(),
            ..checkpoint::SaveConfig::default()
        };
        
        let mut manager = checkpoint::CheckpointManager::new(config)?;
        
        // 複数のチェックポイントを保存
        for epoch in 0..5 {
            let mut metadata = checkpoint::CheckpointMetadata::new(epoch, 0.5 - epoch as f64 * 0.1);
            metadata.val_loss = Some(0.6 - epoch as f64 * 0.08);
            metadata.set_metric("accuracy".to_string(), 0.7 + epoch as f64 * 0.05);
            
            let dummy_model_data = vec![0u8; 1024]; // 1KB dummy data
            
            if manager.should_save(&metadata) {
                let _path = manager.save_checkpoint(metadata, &dummy_model_data)?;
            }
        }
        
        // 結果検証
        let checkpoints = manager.all_checkpoints();
        assert!(!checkpoints.is_empty());
        assert!(checkpoints.len() <= 3); // max_checkpoints respect
        
        // 最良のチェックポイントを確認
        let best = manager.best_checkpoint();
        assert!(best.is_some());
        
        // 統計を表示
        let stats = manager.statistics();
        println!("Checkpoint statistics: {}", stats.summary());
        
        Ok(())
    }

    #[test]
    fn test_custom_metrics() -> anyhow::Result<()> {
        let mut collector = MetricsCollector::<f32>::new();
        
        // カスタムメトリクスを追加
        collector.add_metric(
            "mean_squared_error".to_string(),
            |predictions, targets| {
                let pred_data = predictions.data().as_slice().unwrap_or(&[]);
                let target_data = targets.data().as_slice().unwrap_or(&[]);
                
                if pred_data.len() != target_data.len() || pred_data.is_empty() {
                    return 0.0;
                }
                
                let mse = pred_data.iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| (p - t).powi(2))
                    .sum::<f32>() / pred_data.len() as f32;
                
                mse as f64
            }
        );
        
        // テストデータ
        let predictions = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let targets = Variable::new(Tensor::from_vec(vec![1.1, 1.9, 3.2, 3.8], vec![4]));
        
        let metrics = collector.calculate_metrics(&predictions, &targets);
        
        assert!(metrics.contains_key("mean_squared_error"));
        assert!(metrics.contains_key("accuracy"));
        
        let mse = metrics["mean_squared_error"];
        assert!(mse >= 0.0 && mse < 1.0); // Should be a small error
        
        println!("Custom metrics: {:?}", metrics);
        
        Ok(())
    }
}

/// ベンチマーク用のユーティリティ
/// Benchmark utilities
pub mod benchmarks {
    use super::*;

    /// 学習ループのベンチマーク
    /// Training loop benchmark
    pub fn benchmark_training_loop(
        dataset_size: usize,
        batch_size: usize,
        epochs: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Duration {
        let dataset = create_test_dataset(dataset_size, input_dim, output_dim);
        let mut train_loader = DataLoader::new(&dataset, batch_size, true);
        
        let mut model = SimpleTestModel::new(input_dim, output_dim);
        let optimizer = SGD::new(0.01);
        let loss_fn = MSELoss::new();
        
        let config = TrainerConfig {
            epochs,
            log_frequency: usize::MAX, // Disable logging for benchmark
            validation_frequency: usize::MAX,
            ..TrainerConfig::default()
        };
        
        let mut trainer = Trainer::new(config, optimizer, loss_fn);
        
        let start_time = Instant::now();
        let _ = trainer.train(&mut model, &mut train_loader, None);
        start_time.elapsed()
    }
    
    /// メトリクス計算のベンチマーク
    /// Metrics computation benchmark
    pub fn benchmark_metrics_computation(data_size: usize) -> Duration {
        let collector = MetricsCollector::<f32>::new();
        let predictions = Variable::new(Tensor::randn(&[data_size]));
        let targets = Variable::new(Tensor::randn(&[data_size]));
        
        let start_time = Instant::now();
        let _ = collector.calculate_metrics(&predictions, &targets);
        start_time.elapsed()
    }
    
    /// コールバックのベンチマーク
    /// Callback benchmark
    pub fn benchmark_callbacks(num_epochs: usize, num_callbacks: usize) -> Duration {
        let mut state = TrainingState::<f32>::new(num_epochs);
        
        let mut callbacks: Vec<Box<dyn callbacks::Callback<f32> + Send + Sync>> = Vec::new();
        for _ in 0..num_callbacks {
            callbacks.push(Box::new(callbacks::EarlyStopping::monitor_val_loss(10, 0.01)));
        }
        
        let start_time = Instant::now();
        
        for epoch in 0..num_epochs {
            let mut epoch_state = EpochState::new(epoch);
            
            for callback in &mut callbacks {
                let _ = callback.on_epoch_begin(&mut state, &mut epoch_state);
                let _ = callback.on_epoch_end(&mut state, &epoch_state);
            }
            
            state.add_epoch(epoch_state);
        }
        
        start_time.elapsed()
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::benchmarks::*;
    use std::time::Duration;

    #[test]
    fn test_training_benchmark() {
        let duration = benchmark_training_loop(100, 10, 2, 5, 3);
        println!("Training benchmark (100 samples, 2 epochs): {:?}", duration);
        assert!(duration < Duration::from_secs(5));
    }

    #[test]
    fn test_metrics_benchmark() {
        let duration = benchmark_metrics_computation(1000);
        println!("Metrics benchmark (1000 samples): {:?}", duration);
        assert!(duration < Duration::from_millis(50));
    }

    #[test]
    fn test_callbacks_benchmark() {
        let duration = benchmark_callbacks(10, 5);
        println!("Callbacks benchmark (10 epochs, 5 callbacks): {:?}", duration);
        assert!(duration < Duration::from_millis(10));
    }
}