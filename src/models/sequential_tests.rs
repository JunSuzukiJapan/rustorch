//! Sequential APIの統合テスト
//! Sequential API integration tests

use super::sequential::*;
use super::high_level::*;
use crate::nn::{Linear, Module};
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::data::{DataLoader, TensorDataset};

#[cfg(test)]
mod sequential_integration_tests {
    use super::*;

    #[test]
    fn test_sequential_basic_operations() {
        let mut model: Sequential<f32> = Sequential::new();
        
        // 基本操作テスト
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
        assert!(!model.is_compiled());
        
        // レイヤー追加
        model.add_layer(Linear::new(10, 5));
        assert_eq!(model.len(), 1);
        assert!(!model.is_empty());
        
        // さらにレイヤー追加
        model.add_layer(Linear::new(5, 2));
        assert_eq!(model.len(), 2);
        
        // レイヤー削除
        assert!(model.remove(1).is_ok());
        assert_eq!(model.len(), 1);
        
        // 範囲外削除
        assert!(model.remove(5).is_err());
    }

    #[test]
    fn test_sequential_builder_pattern() {
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(784, 128))
            .add(Linear::new(128, 64))
            .add(Linear::new(64, 10));
        
        assert_eq!(model.len(), 3);
        assert!(!model.is_compiled());
        assert!(model.total_parameters() > 0);
    }

    #[test]
    fn test_sequential_builder_with_name() {
        let model: Sequential<f32> = SequentialBuilder::with_name("test_model")
            .add(Linear::new(100, 50))
            .add(Linear::new(50, 25))
            .build();
        
        assert_eq!(model.len(), 2);
        let summary = model.summary();
        assert!(summary.contains("test_model"));
    }

    #[test]
    fn test_sequential_forward_pass() {
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(5, 3))
            .add(Linear::new(3, 1));
        
        // テスト入力
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input_tensor = Tensor::from_vec(input_data, vec![1, 5]);
        let input_var = Variable::new(input_tensor, false);
        
        // 順伝播
        let output = model.forward(&input_var);
        
        // 出力の形状をチェック（実際の実装では詳細なチェックが必要）
        assert!(output.data().read().unwrap().shape().len() > 0);
    }

    #[test]
    fn test_sequential_training_mode() {
        let mut model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5));
        
        // 初期状態では訓練モードではない
        // 訓練モードに設定
        model.train();
        
        // 評価モードに設定
        model.eval();
        
        // エラーなく完了することを確認
        assert_eq!(model.len(), 1);
    }

    #[test]
    fn test_sequential_summary() {
        let model: Sequential<f32> = Sequential::with_name("summary_test")
            .add(Linear::new(784, 256))
            .add(Linear::new(256, 128))
            .add(Linear::new(128, 10));
        
        let summary = model.summary();
        
        // サマリーに期待される内容が含まれているかチェック
        assert!(summary.contains("summary_test"));
        assert!(summary.contains("Sequential Model") || summary.contains("summary_test"));
        assert!(summary.contains("Total params"));
        assert!(summary.contains("Trainable params"));
    }

    #[test]
    fn test_sequential_parameter_counting() {
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5))    // 10*5 + 5 = 55 parameters
            .add(Linear::new(5, 2));    // 5*2 + 2 = 12 parameters
        
        let total_params = model.total_parameters();
        let trainable_params = model.trainable_parameters();
        
        // パラメータ数が0より大きいことを確認
        assert!(total_params > 0);
        assert_eq!(total_params, trainable_params); // 簡略化実装では同じ
    }

    #[test]
    fn test_sequential_validation() {
        // 空のモデルは無効
        let empty_model: Sequential<f32> = Sequential::new();
        assert!(empty_model.validate().is_err());
        
        // レイヤーを持つモデルでもコンパイルされていないと無効
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5));
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_sequential_clear() {
        let mut model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 2));
        
        assert_eq!(model.len(), 2);
        
        model.clear();
        
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
        assert!(!model.is_compiled());
    }

    #[test]
    fn test_sequential_layer_access() {
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 2));
        
        // レイヤーアクセス
        assert!(model.get_layer(0).is_some());
        assert!(model.get_layer(1).is_some());
        assert!(model.get_layer(2).is_none());
        
        // 全レイヤーアクセス
        let layers = model.layers();
        assert_eq!(layers.len(), 2);
    }

    #[test]
    fn test_sequential_insert() {
        let mut model: Sequential<f32> = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 2));
        
        // 中間にレイヤーを挿入
        assert!(model.insert(1, Linear::new(5, 3)).is_ok());
        assert_eq!(model.len(), 3);
        
        // 範囲外挿入
        assert!(model.insert(10, Linear::new(2, 1)).is_err());
    }
}

#[cfg(test)]
mod high_level_api_tests {
    use super::*;

    #[test]
    fn test_training_history_creation() {
        let mut history: TrainingHistory<f32> = TrainingHistory::new();
        
        assert!(history.train_loss.is_empty());
        assert!(history.val_loss.is_empty());
        assert!(history.metrics.is_empty());
        assert_eq!(history.training_time, 0.0);
        assert!(history.best_val_loss.is_none());
        assert!(history.best_epoch.is_none());
    }

    #[test]
    fn test_training_history_add_epoch() {
        let mut history: TrainingHistory<f32> = TrainingHistory::new();
        let mut epoch_metrics = std::collections::HashMap::new();
        epoch_metrics.insert("accuracy".to_string(), 0.85);
        epoch_metrics.insert("precision".to_string(), 0.82);

        // 最初のエポック
        history.add_epoch(0.6, Some(0.7), epoch_metrics.clone());
        
        assert_eq!(history.train_loss.len(), 1);
        assert_eq!(history.val_loss.len(), 1);
        assert_eq!(history.best_val_loss, Some(0.7));
        assert_eq!(history.best_epoch, Some(0));

        // 改善されたエポック
        history.add_epoch(0.5, Some(0.6), epoch_metrics.clone());
        
        assert_eq!(history.train_loss.len(), 2);
        assert_eq!(history.val_loss.len(), 2);
        assert_eq!(history.best_val_loss, Some(0.6));
        assert_eq!(history.best_epoch, Some(1));
    }

    #[test]
    fn test_training_history_summary() {
        let mut history: TrainingHistory<f32> = TrainingHistory::new();
        history.training_time = 120.5;
        
        let mut epoch_metrics = std::collections::HashMap::new();
        epoch_metrics.insert("accuracy".to_string(), 0.85);
        
        history.add_epoch(0.5, Some(0.6), epoch_metrics);
        history.add_epoch(0.4, Some(0.5), std::collections::HashMap::new());

        let summary = history.summary();
        
        assert!(summary.contains("Training History Summary"));
        assert!(summary.contains("Total epochs: 2"));
        assert!(summary.contains("Training time: 120.50 seconds"));
        assert!(summary.contains("Final training loss"));
        assert!(summary.contains("Best validation loss"));
    }

    #[test]
    fn test_training_history_plot_data() {
        let mut history: TrainingHistory<f32> = TrainingHistory::new();
        
        history.add_epoch(0.6, Some(0.7), std::collections::HashMap::new());
        history.add_epoch(0.5, Some(0.6), std::collections::HashMap::new());
        history.add_epoch(0.4, Some(0.5), std::collections::HashMap::new());

        let (epochs, train_losses, val_losses) = history.plot_data();
        
        assert_eq!(epochs.len(), 3);
        assert_eq!(train_losses.len(), 3);
        assert_eq!(val_losses.len(), 3);
        
        assert_eq!(epochs, vec![1.0, 2.0, 3.0]);
        assert_eq!(train_losses, vec![0.6, 0.5, 0.4]);
        assert_eq!(val_losses, vec![0.7, 0.6, 0.5]);
    }

    #[test]
    fn test_fit_config_builder() {
        let config = FitConfig::new()
            .epochs(20)
            .batch_size(64)
            .verbose(false)
            .early_stopping(5);

        assert_eq!(config.epochs, 20);
        assert_eq!(config.batch_size, 64);
        assert!(!config.verbose);
        assert_eq!(config.patience, Some(5));
        assert_eq!(config.validation_freq, 1);
    }

    #[test]
    fn test_fit_config_default() {
        let config = FitConfig::default();

        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
        assert!(config.verbose);
        assert_eq!(config.validation_freq, 1);
        assert!(config.patience.is_none());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_sequential_with_training() {
        let mut model: Sequential<f32> = Sequential::with_name("integration_test")
            .add(Linear::new(5, 10))
            .add(Linear::new(10, 3));
        
        // モデル概要確認
        let summary = model.summary();
        assert!(summary.contains("integration_test"));
        assert!(summary.contains("Not compiled"));
        
        // 訓練・評価モード切り替え
        model.train();
        model.eval();
        
        // 順伝播テスト
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input_tensor = Tensor::from_vec(input_data, vec![1, 5]);
        let input_var = Variable::new(input_tensor, false);
        
        let output = model.forward(&input_var);
        assert!(output.data().read().unwrap().shape().len() > 0);
    }

    #[test]
    fn test_model_prediction_workflow() {
        let model: Sequential<f32> = Sequential::new()
            .add(Linear::new(3, 8))
            .add(Linear::new(8, 4))
            .add(Linear::new(4, 1));
        
        // 予測テスト
        let input_data = vec![1.0, 2.0, 3.0];
        let input_tensor = Tensor::from_vec(input_data, vec![1, 3]);
        let input_var = Variable::new(input_tensor, false);
        
        let prediction = model.predict(&input_var);
        assert!(prediction.is_ok());
    }

    #[test]
    fn test_model_save_load_workflow() {
        let model: Sequential<f32> = Sequential::with_name("save_load_test")
            .add(Linear::new(5, 10))
            .add(Linear::new(10, 2));
        
        // 保存テスト
        let save_result = model.save("test_model.rustorch");
        assert!(save_result.is_ok());
        
        // 読み込みテスト
        let mut loaded_model: Sequential<f32> = Sequential::new();
        let load_result = loaded_model.load("test_model.rustorch");
        assert!(load_result.is_ok());
    }

    #[test]
    fn test_full_workflow_simulation() {
        // 1. モデル作成
        let mut model: Sequential<f32> = Sequential::with_name("full_workflow")
            .add(Linear::new(4, 16))
            .add(Linear::new(16, 8))
            .add(Linear::new(8, 1));
        
        // 2. データ準備（ダミーデータ）
        let mut input_tensors = Vec::new();
        let mut target_tensors = Vec::new();
        
        let train_data = vec![
            (vec![1.0, 2.0, 3.0, 4.0], vec![10.0]),
            (vec![2.0, 3.0, 4.0, 5.0], vec![14.0]),
            (vec![3.0, 4.0, 5.0, 6.0], vec![18.0]),
        ];
        
        for (input, target) in train_data {
            input_tensors.push(Tensor::from_vec(input, vec![4]));
            target_tensors.push(Tensor::from_vec(target, vec![1]));
        }
        
        let dataset = TensorDataset::new(input_tensors, target_tensors);
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();
        
        let mut train_loader = DataLoader::new(&dataset, 2, false);
        
        // 3. 順伝播テスト
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_tensor = Tensor::from_vec(input_data, vec![1, 4]);
        let input_var = Variable::new(input_tensor, false);
        
        let output = model.forward(&input_var);
        assert!(output.data().read().unwrap().shape().len() > 0);
        
        // 4. 予測テスト
        let prediction = model.predict(&input_var);
        assert!(prediction.is_ok());
        
        // 5. バッチ予測テスト
        let batch_predictions = model.predict_batch(&mut train_loader);
        assert!(batch_predictions.is_ok());
        
        // 6. モデル概要
        let summary = model.summary();
        assert!(summary.contains("full_workflow"));
        assert!(summary.len() > 0);
    }
}