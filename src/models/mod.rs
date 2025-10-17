//! 機械学習モデルアーキテクチャ
//! Machine learning model architectures

pub mod architecture;
pub mod cnn;
pub mod factory;
pub mod gpt;
pub mod llama;
pub mod rnn;
pub mod sampling;
pub mod serialization;
pub mod training;
pub mod transformer_models;

/// Keras風のSequential API
/// Keras-like Sequential API
pub mod sequential;

/// 高レベルモデルAPI
/// High-level model API
pub mod high_level;

// Sequential APIの使用例
// Sequential API examples
// pub mod examples;  // 一時的にコメントアウト - 型制約の問題のため

/// 基本的なSequential API実装
/// Basic Sequential API implementation
pub mod sequential_basic;

/// Sequential APIの統合テスト
/// Sequential API integration tests
#[cfg(test)]
pub mod sequential_tests;

use crate::autograd::Variable;
use crate::nn::Module;
use num_traits::Float;
use std::collections::HashMap;

/// モデルの訓練・評価モード
/// Training and evaluation modes for models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelMode {
    /// 訓練モード
    Train,
    /// 評価モード
    Eval,
}

/// 基本モデルトレイト
/// Base model trait
pub trait Model<T>: Module<T>
where
    T: Float + 'static + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    /// モデルを訓練モードに設定
    /// Set model to training mode
    fn train(&mut self);

    /// モデルを評価モードに設定
    /// Set model to evaluation mode
    fn eval(&mut self);

    /// 現在のモードを取得
    /// Get current mode
    fn mode(&self) -> ModelMode;

    /// モデルの設定を取得
    /// Get model configuration
    fn config(&self) -> HashMap<String, String>;

    /// モデルの概要を表示
    /// Display model summary
    fn summary(&self) -> String;
}

/// モデル構築のためのビルダーパターン
/// Builder pattern for model construction
pub trait ModelBuilder<T>
where
    T: Float + 'static + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    /// The model type that this builder creates
    /// このビルダーが作成するモデルの型
    type Model: Model<T>;

    /// モデルを構築
    /// Build the model
    fn build(self) -> Self::Model;
}

// Re-export model architectures
pub use architecture::ModelArchitecture;
pub use cnn::{CNNBuilder, ResNet, ResNetBuilder, CNN};
pub use factory::{create_model_from_gguf, detect_architecture};
pub use gpt::{GPTConfig, GPTModel};
pub use llama::{LlamaConfig, LlamaModel};
pub use rnn::{LSTMModel, LSTMModelBuilder, RNNModel, RNNModelBuilder};
pub use serialization::{ModelLoader, ModelSaver, SerializationFormat};
pub use training::{Trainer, TrainingConfig, TrainingResult};
pub use transformer_models::{BERTBuilder, TransformerModel, TransformerModelBuilder, BERT};

// Re-export Sequential API
pub use high_level::{FitConfig, HighLevelModel, TrainingHistory};
pub use sequential::{Sequential, SequentialBuilder};
pub use sequential_basic::{BasicSequential, BasicSequentialBuilder};
// pub use examples::run_all_examples;  // 一時的にコメントアウト

/// Inference engine for model evaluation
/// モデル評価用の推論エンジン
#[derive(Debug)]
pub struct InferenceEngine<T: Float + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    InferenceEngine<T>
{
    /// Create a new inference engine
    /// 新しい推論エンジンを作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform prediction using the given model
    /// 指定されたモデルを使用して予測を実行
    pub fn predict<M: Model<T>>(&self, model: &M, input: &Variable<T>) -> Variable<T> {
        model.forward(input)
    }
}

/// Evaluation metrics for model performance
/// モデル性能の評価メトリクス
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Model accuracy score
    /// モデルの精度スコア
    pub accuracy: f64,
    /// Model precision score
    /// モデルの適合率スコア
    pub precision: f64,
    /// Model recall score
    /// モデルの再現率スコア
    pub recall: f64,
    /// Model F1 score
    /// モデルのF1スコア
    pub f1_score: f64,
    /// Model loss value
    /// モデルの損失値
    pub loss: f64,
}

impl Metrics {
    /// Create new metrics with default values
    /// デフォルト値で新しいメトリクスを作成
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: 0.0,
        }
    }

    /// Create metrics with specified values
    /// 指定された値でメトリクスを作成
    pub fn with_values(
        accuracy: f64,
        precision: f64,
        recall: f64,
        f1_score: f64,
        loss: f64,
    ) -> Self {
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            loss,
        }
    }
}
