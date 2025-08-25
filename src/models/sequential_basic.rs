//! 基本的なSequential API実装
//! Basic Sequential API implementation

use crate::autograd::Variable;
use crate::nn::Module;
use crate::training::trainer::TrainableModel;
use num_traits::Float;
use std::fmt::Debug;

/// 基本的なSequentialモデル
/// Basic Sequential model
#[derive(Debug)]
pub struct BasicSequential<
    T: Float
        + Send
        + Sync
        + Debug
        + Clone
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
> {
    /// レイヤーのリスト
    layers: Vec<Box<dyn Module<T> + Send + Sync>>,
    /// 訓練モードかどうか
    training: bool,
    /// モデルの名前
    name: Option<String>,
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > BasicSequential<T>
{
    /// 新しいSequentialモデルを作成
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            training: false,
            name: None,
        }
    }

    /// 名前付きモデルを作成
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            layers: Vec::new(),
            training: false,
            name: Some(name.into()),
        }
    }

    /// レイヤーを追加
    pub fn add_layer<M>(&mut self, layer: M)
    where
        M: Module<T> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// レイヤー数を取得
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// モデルが空かどうか
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// モデルをクリア
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// パラメータの総数を計算
    pub fn total_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameters().len())
            .sum()
    }

    /// モデルの概要を表示
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        if let Some(ref name) = self.name {
            summary.push_str(&format!("Model: \"{}\"\n", name));
        } else {
            summary.push_str("Basic Sequential Model\n");
        }

        summary.push_str(&format!("Layers: {}\n", self.layers.len()));
        summary.push_str(&format!("Total parameters: {}\n", self.total_parameters()));
        summary.push_str(&format!("Training mode: {}\n", self.training));

        summary
    }
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > Default for BasicSequential<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > Module<T> for BasicSequential<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            let mut layer_params = layer.parameters();
            params.append(&mut layer_params);
        }
        params
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > TrainableModel<T> for BasicSequential<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        Module::forward(self, input)
    }

    fn train(&mut self) {
        Module::train(self);
    }

    fn eval(&mut self) {
        Module::eval(self);
    }

    fn parameters(&self) -> Vec<&Variable<T>> {
        // 簡略化実装
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Variable<T>> {
        // 簡略化実装
        Vec::new()
    }
}

/// Sequentialビルダー
pub struct BasicSequentialBuilder<
    T: Float
        + Send
        + Sync
        + Debug
        + Clone
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
> {
    model: BasicSequential<T>,
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > BasicSequentialBuilder<T>
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            model: BasicSequential::new(),
        }
    }

    /// 名前付きビルダーを作成
    /// Create a named builder
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            model: BasicSequential::with_name(name),
        }
    }

    /// レイヤーを追加
    /// Add a layer
    pub fn add<M>(mut self, layer: M) -> Self
    where
        M: Module<T> + Send + Sync + 'static,
    {
        self.model.add_layer(layer);
        self
    }

    /// モデルを構築
    /// Build the model
    pub fn build(self) -> BasicSequential<T> {
        self.model
    }
}

impl<
        T: Float
            + Send
            + Sync
            + Debug
            + Clone
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    > Default for BasicSequentialBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_basic_sequential_creation() {
        let model: BasicSequential<f32> = BasicSequential::new();
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
    }

    #[test]
    fn test_basic_sequential_with_name() {
        let model: BasicSequential<f32> = BasicSequential::with_name("test_model");
        let summary = model.summary();
        assert!(summary.contains("test_model"));
    }

    #[test]
    fn test_basic_sequential_builder() {
        let model: BasicSequential<f32> = BasicSequentialBuilder::new().build();

        assert_eq!(model.len(), 0);
    }

    #[test]
    fn test_basic_sequential_summary() {
        let model: BasicSequential<f32> = BasicSequential::new();
        let summary = model.summary();
        assert!(summary.contains("Basic Sequential Model"));
        assert!(summary.contains("Layers: 0"));
    }

    #[test]
    fn test_basic_sequential_clear() {
        let mut model: BasicSequential<f32> = BasicSequential::new();
        model.clear();

        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
    }

    #[test]
    fn test_basic_sequential_training_mode() {
        let mut model: BasicSequential<f32> = BasicSequential::new();

        // 訓練モードに設定
        Module::train(&mut model);

        // 評価モードに設定
        Module::eval(&mut model);

        // エラーなく完了することを確認
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn test_basic_sequential_forward() {
        let model: BasicSequential<f32> = BasicSequential::new();

        // テスト入力
        let input_data = vec![1.0, 2.0, 3.0];
        let input_tensor = Tensor::from_vec(input_data, vec![1, 3]);
        let input_var = Variable::new(input_tensor, false);

        // 空のモデルでは入力がそのまま出力される
        let output = Module::forward(&model, &input_var);
        assert!(output.data().read().unwrap().shape().len() > 0);
    }
}
