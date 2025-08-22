//! Keras風のSequential API
//! Keras-like Sequential API
//!
//! このモジュールは、レイヤーを順次積み重ねてニューラルネットワークを
//! 構築するためのシンプルで直感的なAPIを提供します。
//!
//! ## 使用例
//!
//! ```no_run
//! use rustorch::models::sequential::Sequential;
//! use rustorch::nn::{Linear, ReLU, Dropout};
//! use rustorch::optim::Adam;
//! use rustorch::nn::loss::CrossEntropyLoss;
//!
//! let mut model = Sequential::new()
//!     .add(Linear::new(784, 128))
//!     .add(ReLU::new())
//!     .add(Dropout::new(0.2, false))
//!     .add(Linear::new(128, 64))
//!     .add(ReLU::new())
//!     .add(Linear::new(64, 10));
//!
//! // モデルのコンパイル
//! model.compile(
//!     Adam::default_params(0.001),
//!     CrossEntropyLoss::new(),
//!     vec!["accuracy".to_string()]
//! );
//!
//! // 訓練（実際の実装では DataLoader を使用）
//! // model.fit(train_data, validation_data, epochs, batch_size, verbose);
//! ```

use crate::autograd::Variable;
use crate::nn::Module;
use crate::optim::Optimizer;
use crate::nn::loss::Loss;
use crate::training::TrainableModel;
use num_traits::Float;
use std::fmt::Debug;
use anyhow::Result;

/// Sequential APIの中核となるモデルクラス
/// Core model class for Sequential API
pub struct Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    /// レイヤーのリスト
    layers: Vec<Box<dyn Module<T> + Send + Sync>>,
    /// コンパイル済みかどうか
    compiled: bool,
    /// オプティマイザー（コンパイル後に設定）
    optimizer: Option<Box<dyn Optimizer + Send + Sync>>,
    /// 損失関数（コンパイル後に設定）
    loss_fn: Option<Box<dyn Loss<T> + Send + Sync>>,
    /// メトリクス（コンパイル後に設定）
    metrics: Vec<String>,
    /// 訓練モードかどうか
    training: bool,
    /// モデルの名前（オプション）
    name: Option<String>,
}

impl<T> std::fmt::Debug for Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &self.layers.len())
            .field("compiled", &self.compiled)
            .field("metrics", &self.metrics)
            .field("training", &self.training)
            .field("name", &self.name)
            .finish()
    }
}

impl<T> Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    /// 新しいSequentialモデルを作成
    /// Create a new Sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            compiled: false,
            optimizer: None,
            loss_fn: None,
            metrics: Vec::new(),
            training: false,
            name: None,
        }
    }

    /// 名前付きでSequentialモデルを作成
    /// Create a named Sequential model
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            layers: Vec::new(),
            compiled: false,
            optimizer: None,
            loss_fn: None,
            metrics: Vec::new(),
            training: false,
            name: Some(name.into()),
        }
    }

    /// レイヤーを追加（メソッドチェーン対応）
    /// Add a layer (method chaining support)
    pub fn add<M>(mut self, layer: M) -> Self
    where
        M: Module<T> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }

    /// レイヤーを追加（可変参照版）
    /// Add a layer (mutable reference version)
    pub fn add_layer<M>(&mut self, layer: M)
    where
        M: Module<T> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// レイヤーを挿入
    /// Insert a layer at specified position
    pub fn insert<M>(&mut self, index: usize, layer: M) -> Result<()>
    where
        M: Module<T> + Send + Sync + 'static,
    {
        if index > self.layers.len() {
            return Err(anyhow::anyhow!("Index {} out of bounds for {} layers", index, self.layers.len()));
        }
        self.layers.insert(index, Box::new(layer));
        Ok(())
    }

    /// レイヤーを削除
    /// Remove a layer at specified position
    pub fn remove(&mut self, index: usize) -> Result<()> {
        if index >= self.layers.len() {
            return Err(anyhow::anyhow!("Index {} out of bounds for {} layers", index, self.layers.len()));
        }
        self.layers.remove(index);
        Ok(())
    }

    /// レイヤー数を取得
    /// Get number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// モデルが空かどうか
    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// モデルをコンパイル
    /// Compile the model
    pub fn compile<O, L>(&mut self, optimizer: O, loss_fn: L, metrics: Vec<String>) -> Result<()>
    where
        O: Optimizer + Send + Sync + 'static,
        L: Loss<T> + Send + Sync + 'static,
    {
        if self.layers.is_empty() {
            return Err(anyhow::anyhow!("Cannot compile empty model. Add layers first."));
        }

        self.optimizer = Some(Box::new(optimizer));
        self.loss_fn = Some(Box::new(loss_fn));
        self.metrics = metrics;
        self.compiled = true;

        Ok(())
    }

    /// コンパイル済みかどうかを確認
    /// Check if model is compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// モデルの概要を表示
    /// Display model summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if let Some(ref name) = self.name {
            summary.push_str(&format!("Model: \"{}\"\n", name));
        } else {
            summary.push_str("Sequential Model\n");
        }
        
        summary.push_str("_________________________________________________________________\n");
        summary.push_str("Layer (type)                 Output Shape              Param #   \n");
        summary.push_str("=================================================================\n");

        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_info = format!("Layer_{} (type)", i);
            summary.push_str(&format!("{:<2} {}\n", i, layer_info));
            total_params += layer.parameters().len();
        }

        summary.push_str("=================================================================\n");
        summary.push_str(&format!("Total params: {}\n", total_params));
        summary.push_str(&format!("Trainable params: {}\n", total_params)); // 簡略化
        summary.push_str(&format!("Non-trainable params: 0\n"));
        summary.push_str("_________________________________________________________________\n");

        if self.compiled {
            summary.push_str("\nModel compiled with:\n");
            summary.push_str(&format!(" - Optimizer: {}\n", "Configured")); // 簡略化
            summary.push_str(&format!(" - Loss: {}\n", "Configured")); // 簡略化
            summary.push_str(&format!(" - Metrics: {:?}\n", self.metrics));
        } else {
            summary.push_str("\nModel not compiled yet. Call compile() before training.\n");
        }

        summary
    }

    /// モデルをクリア
    /// Clear the model
    pub fn clear(&mut self) {
        self.layers.clear();
        self.compiled = false;
        self.optimizer = None;
        self.loss_fn = None;
        self.metrics.clear();
    }

    /// レイヤーの参照を取得
    /// Get reference to layer
    pub fn get_layer(&self, index: usize) -> Option<&Box<dyn Module<T> + Send + Sync>> {
        self.layers.get(index)
    }

    /// 全レイヤーの参照を取得
    /// Get references to all layers
    pub fn layers(&self) -> &[Box<dyn Module<T> + Send + Sync>] {
        &self.layers
    }

    /// モデルの設定を検証
    /// Validate model configuration
    pub fn validate(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(anyhow::anyhow!("Model has no layers"));
        }

        if !self.compiled {
            return Err(anyhow::anyhow!("Model is not compiled"));
        }

        // レイヤー間の次元整合性チェック（簡略化実装）
        // Layer dimension compatibility check (simplified implementation)
        
        Ok(())
    }

    /// パラメータの総数を計算
    /// Calculate total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters().len()).sum()
    }

    /// 訓練可能なパラメータの総数を計算
    /// Calculate total number of trainable parameters
    pub fn trainable_parameters(&self) -> usize {
        // 簡略化: 全パラメータが訓練可能と仮定
        self.total_parameters()
    }
}

impl<T> Default for Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    /// 順伝播
    /// Forward pass
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut output = input.clone();
        
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        
        output
    }

    /// 訓練モードに設定
    /// Set training mode
    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    /// 評価モードに設定
    /// Set evaluation mode  
    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    /// パラメータを取得
    /// Get parameters
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            let mut layer_params = layer.parameters();
            params.append(&mut layer_params);
        }
        params
    }

    /// Any型への変換
    /// Convert to Any type
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> TrainableModel<T> for Sequential<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    /// 順伝播
    /// Forward pass
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        Module::forward(self, input)
    }

    /// 訓練モードに設定
    /// Set training mode
    fn train(&mut self) {
        Module::train(self);
    }

    /// 評価モードに設定
    /// Set evaluation mode
    fn eval(&mut self) {
        Module::eval(self);
    }

    /// パラメータを取得
    /// Get parameters
    fn parameters(&self) -> Vec<&Variable<T>> {
        let params = Vec::new();
        for _layer in &self.layers {
            // 各レイヤーからパラメータを収集
            // 実装は簡略化 - 実際にはレイヤーのパラメータアクセスメソッドを使用
        }
        params
    }

    /// パラメータを可変参照で取得
    /// Get mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Variable<T>> {
        let params = Vec::new();
        // 実装は簡略化 - 実際にはレイヤーの可変パラメータアクセスメソッドを使用
        params
    }
}

/// Sequential モデルのビルダー
/// Sequential model builder
pub struct SequentialBuilder<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    model: Sequential<T>,
}

impl<T> SequentialBuilder<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            model: Sequential::new(),
        }
    }

    /// 名前付きビルダーを作成
    /// Create a named builder
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            model: Sequential::with_name(name),
        }
    }

    /// レイヤーを追加
    /// Add a layer
    pub fn add<M>(mut self, layer: M) -> Self
    where
        M: Module<T> + Send + Sync + 'static,
    {
        self.model = self.model.add(layer);
        self
    }

    /// モデルを構築
    /// Build the model
    pub fn build(self) -> Sequential<T> {
        self.model
    }
}

impl<T> Default for SequentialBuilder<T>
where
    T: Float + Send + Sync + 'static + Debug + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_creation() {
        let model: Sequential<f32> = Sequential::new();
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
        assert!(!model.is_compiled());
    }

    #[test]
    fn test_sequential_with_name() {
        let model: Sequential<f32> = Sequential::with_name("test_model");
        assert_eq!(model.name, Some("test_model".to_string()));
    }

    #[test]
    fn test_sequential_summary() {
        let model: Sequential<f32> = Sequential::new();
        let summary = model.summary();
        assert!(summary.contains("Sequential Model"));
        assert!(summary.contains("Total params: 0"));
    }

    #[test]
    fn test_sequential_builder() {
        let model: Sequential<f32> = SequentialBuilder::new()
            .build();
        
        assert_eq!(model.len(), 0);
        assert!(!model.is_compiled());
    }

    #[test]
    fn test_sequential_validation() {
        let model: Sequential<f32> = Sequential::new();
        
        // 空のモデルは検証に失敗
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_sequential_clear() {
        let mut model: Sequential<f32> = Sequential::new();
        model.clear();
        
        assert_eq!(model.len(), 0);
        assert!(!model.is_compiled());
    }
}