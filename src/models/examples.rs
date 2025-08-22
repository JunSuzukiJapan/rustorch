//! Sequential APIの使用例
//! Sequential API usage examples
//!
//! このモジュールはKeras風のSequential APIの使用例を提供し、
//! 実際のニューラルネットワークの構築と訓練方法を示します。

use crate::models::sequential::Sequential;
use crate::models::high_level::{HighLevelModel, FitConfig, TrainingHistory};
use crate::nn::{Linear, Module};
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::data::{DataLoader, TensorDataset};
use num_traits::Float;
use std::fmt::Debug;
use anyhow::Result;

/// シンプルな多層パーセプトロンの例
/// Simple Multi-Layer Perceptron example
pub fn simple_mlp_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Simple MLP Example ===");

    // 1. モデル構築
    let mut model = Sequential::<T>::with_name("simple_mlp")
        .add(Linear::new(784, 128))  // 入力層から隠れ層
        .add(Linear::new(128, 64))   // 隠れ層1から隠れ層2
        .add(Linear::new(64, 10));   // 隠れ層2から出力層

    // 2. モデルサマリー表示
    println!("{}", model.summary());

    // 3. モデルコンパイル（実際の実装では適切なオプティマイザーと損失関数を使用）
    // model.compile(
    //     optimizer::Adam::new(0.001),
    //     loss::CrossEntropyLoss::new(),
    //     vec!["accuracy".to_string()]
    // )?;

    println!("Model created successfully!");
    println!("Total parameters: {}", model.total_parameters());

    Ok(())
}

/// 畳み込みニューラルネットワークの例（概念的）
/// Convolutional Neural Network example (conceptual)
pub fn cnn_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== CNN Example (Conceptual) ===");

    // 実際の畳み込み層が実装されたときの使用例
    let mut model = Sequential::<T>::with_name("cnn_model");
    
    // 概念的な構造（実際の畳み込み層実装が必要）
    /*
    model = model
        .add(Conv2d::new(1, 32, 3))      // 畳み込み層1
        .add(ReLU::new())                // 活性化関数
        .add(MaxPool2d::new(2))          // プーリング層
        .add(Conv2d::new(32, 64, 3))     // 畳み込み層2
        .add(ReLU::new())                // 活性化関数
        .add(MaxPool2d::new(2))          // プーリング層
        .add(Flatten::new())             // 平坦化
        .add(Linear::new(64 * 7 * 7, 128)) // 全結合層
        .add(ReLU::new())                // 活性化関数
        .add(Dropout::new(0.5))          // ドロップアウト
        .add(Linear::new(128, 10));      // 出力層
    */

    // 現在は線形層のみで代替
    model = model
        .add(Linear::new(28 * 28, 128))  // 入力を平坦化済みと仮定
        .add(Linear::new(128, 64))
        .add(Linear::new(64, 10));

    println!("{}", model.summary());
    println!("CNN model structure created (using Linear layers as placeholder)");

    Ok(())
}

/// 自動エンコーダーの例
/// Autoencoder example
pub fn autoencoder_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Autoencoder Example ===");

    // エンコーダー部分
    let mut encoder = Sequential::<T>::with_name("encoder")
        .add(Linear::new(784, 256))   // 入力層
        .add(Linear::new(256, 128))   // 隠れ層1
        .add(Linear::new(128, 64))    // 隠れ層2
        .add(Linear::new(64, 32));    // ボトルネック層

    // デコーダー部分
    let mut decoder = Sequential::<T>::with_name("decoder")
        .add(Linear::new(32, 64))     // ボトルネック層から
        .add(Linear::new(64, 128))    // 隠れ層1
        .add(Linear::new(128, 256))   // 隠れ層2
        .add(Linear::new(256, 784));  // 出力層（再構築）

    println!("Encoder:");
    println!("{}", encoder.summary());
    println!("\nDecoder:");
    println!("{}", decoder.summary());

    // 実際の使用では、エンコーダーとデコーダーを組み合わせて使用
    println!("Autoencoder models created successfully!");

    Ok(())
}

/// 訓練プロセスの例
/// Training process example
pub fn training_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Training Process Example ===");

    // 1. モデル構築
    let mut model = Sequential::<T>::with_name("training_demo")
        .add(Linear::new(10, 64))
        .add(Linear::new(64, 32))
        .add(Linear::new(32, 1));

    println!("Model created:");
    println!("{}", model.summary());

    // 2. ダミーデータセット作成
    let train_data = create_dummy_dataset::<T>(1000, 10)?;
    let val_data = create_dummy_dataset::<T>(200, 10)?;

    // 3. データローダー作成
    let mut train_loader = DataLoader::new(train_data, 32);
    let mut val_loader = DataLoader::new(val_data, 32);

    // 4. モデルコンパイル（実際の実装では適切なオプティマイザーと損失関数を使用）
    // model.compile(
    //     optimizer::SGD::new(0.01),
    //     loss::MeanSquaredError::new(),
    //     vec!["mae".to_string()]
    // )?;

    // 5. 訓練設定
    let config = FitConfig::new()
        .epochs(10)
        .batch_size(32)
        .verbose(true)
        .early_stopping(3);

    println!("Training configuration:");
    println!(" - Epochs: {}", config.epochs);
    println!(" - Batch size: {}", config.batch_size);
    println!(" - Early stopping patience: {:?}", config.patience);

    // 6. 訓練実行（実際の実装が完了したときに有効化）
    /*
    let history = model.fit(
        &mut train_loader,
        Some(&mut val_loader),
        config.epochs,
        config.batch_size,
        config.verbose
    )?;

    // 7. 結果表示
    println!("\nTraining completed!");
    println!("{}", history.summary());
    */

    println!("Training example structure prepared!");

    Ok(())
}

/// 予測とモデル評価の例
/// Prediction and model evaluation example
pub fn prediction_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Prediction and Evaluation Example ===");

    // 1. 訓練済みモデル（簡略化）
    let model = Sequential::<T>::with_name("prediction_demo")
        .add(Linear::new(5, 32))
        .add(Linear::new(32, 16))
        .add(Linear::new(16, 3));

    println!("Model for prediction:");
    println!("{}", model.summary());

    // 2. 単一予測の例
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input_tensor = Tensor::from_vec(input_data, vec![1, 5]); // バッチサイズ1
    let input_var = Variable::new(input_tensor, false);

    let prediction = model.predict(&input_var)?;
    println!("Single prediction completed");

    // 3. バッチ予測の例
    let test_data = create_dummy_dataset::<T>(100, 5)?;
    let mut test_loader = DataLoader::new(test_data, 10);

    let predictions = model.predict_batch(&mut test_loader)?;
    println!("Batch prediction completed: {} batches", predictions.len());

    // 4. モデル評価の例
    /*
    let eval_metrics = model.evaluate(&mut test_loader)?;
    println!("Evaluation metrics:");
    for (metric_name, value) in eval_metrics {
        println!(" - {}: {:.4}", metric_name, value);
    }
    */

    println!("Prediction and evaluation examples completed!");

    Ok(())
}

/// モデル保存・読み込みの例
/// Model save/load example
pub fn model_persistence_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Model Persistence Example ===");

    // 1. モデル作成と訓練（簡略化）
    let mut model = Sequential::<T>::with_name("persistence_demo")
        .add(Linear::new(20, 64))
        .add(Linear::new(64, 32))
        .add(Linear::new(32, 5));

    println!("Original model:");
    println!("{}", model.summary());

    // 2. モデル保存
    let model_path = "models/my_model.rustorch";
    model.save(model_path)?;
    println!("Model saved to: {}", model_path);

    // 3. 新しいモデル作成と読み込み
    let mut loaded_model = Sequential::<T>::new();
    loaded_model.load(model_path)?;
    println!("Model loaded from: {}", model_path);

    // 4. アーキテクチャの比較
    println!("Loaded model parameters: {}", loaded_model.total_parameters());

    Ok(())
}

/// 転移学習の例（概念的）
/// Transfer learning example (conceptual)
pub fn transfer_learning_example<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("=== Transfer Learning Example (Conceptual) ===");

    // 1. 事前訓練モデルの読み込み（概念的）
    let mut pretrained_model = Sequential::<T>::with_name("pretrained_model");
    // pretrained_model.load("pretrained/imagenet_model.rustorch")?;

    // 2. 特徴抽出部分の固定（概念的）
    // for (i, layer) in pretrained_model.layers().iter().enumerate() {
    //     if i < pretrained_model.len() - 2 {  // 最後の2層以外を固定
    //         layer.freeze();
    //     }
    // }

    // 3. 新しい分類ヘッドの追加
    let mut transfer_model = Sequential::<T>::with_name("transfer_model")
        // .add_from_model(pretrained_model.layers()[..pretrained_model.len()-1])  // 最後の層以外をコピー
        .add(Linear::new(512, 256))   // 新しい隠れ層
        .add(Linear::new(256, 10));   // 新しい出力層（10クラス分類）

    println!("Transfer learning model:");
    println!("{}", transfer_model.summary());

    // 4. ファインチューニング設定
    println!("Transfer learning configuration:");
    println!(" - Frozen layers: {} (feature extraction)", 0); // プレースホルダー
    println!(" - Trainable layers: {} (classification head)", transfer_model.len());

    Ok(())
}

/// ダミーデータセットを作成するヘルパー関数
/// Helper function to create dummy dataset
fn create_dummy_dataset<T>(size: usize, input_dim: usize) -> Result<TensorDataset<T>>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    let mut input_tensors = Vec::new();
    let mut target_tensors = Vec::new();

    for i in 0..size {
        // ダミー入力データ
        let input_data: Vec<T> = (0..input_dim)
            .map(|j| T::from((i + j) as f64 * 0.01).unwrap())
            .collect();
        let input_tensor = Tensor::from_vec(input_data, vec![input_dim]);
        
        // ダミーターゲットデータ
        let target_data: Vec<T> = vec![T::from(i as f64 * 0.1).unwrap()];
        let target_tensor = Tensor::from_vec(target_data, vec![1]);
        
        input_tensors.push(input_tensor);
        target_tensors.push(target_tensor);
    }

    TensorDataset::new(input_tensors, target_tensors)
}

/// 全ての例を実行
/// Run all examples
pub fn run_all_examples<T>() -> Result<()>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    println!("Running Sequential API Examples");
    println!("================================\n");

    simple_mlp_example::<T>()?;
    println!();

    cnn_example::<T>()?;
    println!();

    autoencoder_example::<T>()?;
    println!();

    training_example::<T>()?;
    println!();

    prediction_example::<T>()?;
    println!();

    model_persistence_example::<T>()?;
    println!();

    transfer_learning_example::<T>()?;
    println!();

    println!("All examples completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mlp_example() {
        assert!(simple_mlp_example::<f32>().is_ok());
    }

    #[test]
    fn test_cnn_example() {
        assert!(cnn_example::<f32>().is_ok());
    }

    #[test]
    fn test_autoencoder_example() {
        assert!(autoencoder_example::<f32>().is_ok());
    }

    #[test]
    fn test_training_example() {
        assert!(training_example::<f32>().is_ok());
    }

    #[test]
    fn test_prediction_example() {
        assert!(prediction_example::<f32>().is_ok());
    }

    #[test]
    fn test_model_persistence_example() {
        assert!(model_persistence_example::<f32>().is_ok());
    }

    #[test]
    fn test_transfer_learning_example() {
        assert!(transfer_learning_example::<f32>().is_ok());
    }

    #[test]
    fn test_run_all_examples() {
        assert!(run_all_examples::<f32>().is_ok());
    }

    #[test]
    fn test_create_dummy_dataset() {
        let dataset = create_dummy_dataset::<f32>(100, 5);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        assert_eq!(dataset.len(), 100);
    }
}