//! CNN (畳み込みニューラルネットワーク) モデル実装
//! CNN (Convolutional Neural Network) model implementations

use crate::autograd::Variable;
use crate::models::{Model, ModelBuilder, ModelMode};
use crate::nn::{
    BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Softmax,
};
use num_traits::Float;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

/// 基本的な CNN モデル
/// Basic CNN model
#[derive(Debug)]
pub struct CNN<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    features: Sequential<T>,
    classifier: Sequential<T>,
    mode: ModelMode,
    num_classes: usize,
}

impl<T> CNN<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    /// 新しい CNN モデルを作成
    /// Create a new CNN model
    pub fn new(
        input_channels: usize,
        num_classes: usize,
        hidden_channels: Vec<usize>,
        dropout_rate: f64,
    ) -> Self {
        let mut features = Sequential::new();
        let mut in_channels = input_channels;

        // 畳み込み層の構築
        for (i, &out_channels) in hidden_channels.iter().enumerate() {
            // Conv2d -> BatchNorm -> ReLU -> MaxPool
            features.add_module(Conv2d::new(
                in_channels,
                out_channels,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                Some(true),
            ));
            features.add_module(BatchNorm2d::new(out_channels, None, None, None));
            features.add_module(ReLU::new());

            if i % 2 == 1 {
                // 2層ごとにプーリング
                features.add_module(MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0))));
            }

            in_channels = out_channels;
        }

        // 分類器の構築
        let mut classifier = Sequential::new();
        let final_feature_size = hidden_channels.last().unwrap_or(&64) * 7 * 7; // 仮定: 224x224 -> 7x7

        classifier.add_module(Dropout::new(
            <T as From<f32>>::from(dropout_rate as f32),
            false,
        ));
        classifier.add_module(Linear::new(final_feature_size, 512));
        classifier.add_module(ReLU::new());
        classifier.add_module(Dropout::new(
            <T as From<f32>>::from(dropout_rate as f32),
            false,
        ));
        classifier.add_module(Linear::new(512, num_classes));
        classifier.add_module(Softmax::new(1));

        CNN {
            features,
            classifier,
            mode: ModelMode::Train,
            num_classes,
        }
    }
}

impl<T> Module<T> for CNN<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let features = self.features.forward(input);

        // Global Average Pooling または Flatten
        let flattened = self.flatten_features(&features);

        self.classifier.forward(&flattened)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.features.parameters();
        params.extend(self.classifier.parameters());
        params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for CNN<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    fn train(&mut self) {
        self.mode = ModelMode::Train;
    }

    fn eval(&mut self) {
        self.mode = ModelMode::Eval;
    }

    fn mode(&self) -> ModelMode {
        self.mode
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("model_type".to_string(), "CNN".to_string());
        config.insert("num_classes".to_string(), self.num_classes.to_string());
        config.insert(
            "feature_layers".to_string(),
            self.features.len().to_string(),
        );
        config.insert(
            "classifier_layers".to_string(),
            self.classifier.len().to_string(),
        );
        config
    }

    fn summary(&self) -> String {
        format!(
            "CNN Model:\n  - Feature layers: {}\n  - Classifier layers: {}\n  - Classes: {}\n  - Mode: {:?}",
            self.features.len(),
            self.classifier.len(),
            self.num_classes,
            self.mode
        )
    }
}

impl<T> CNN<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    /// 特徴量をフラット化
    /// Flatten features
    fn flatten_features(&self, features: &Variable<T>) -> Variable<T> {
        // 実装は簡略化 - 実際にはテンソルの形状変換が必要
        features.clone()
    }
}

/// CNN モデルビルダー
/// CNN model builder
#[derive(Debug, Default)]
pub struct CNNBuilder<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    input_channels: Option<usize>,
    num_classes: Option<usize>,
    hidden_channels: Vec<usize>,
    dropout_rate: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> CNNBuilder<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        CNNBuilder {
            input_channels: None,
            num_classes: None,
            hidden_channels: vec![64, 128, 256, 512],
            dropout_rate: 0.5,
            _phantom: std::marker::PhantomData,
        }
    }

    /// 入力チャンネル数を設定
    /// Set input channels
    pub fn input_channels(mut self, channels: usize) -> Self {
        self.input_channels = Some(channels);
        self
    }

    /// クラス数を設定
    /// Set number of classes
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.num_classes = Some(classes);
        self
    }

    /// 隠れ層のチャンネル数を設定
    /// Set hidden layer channels
    pub fn hidden_channels(mut self, channels: Vec<usize>) -> Self {
        self.hidden_channels = channels;
        self
    }

    /// ドロップアウト率を設定
    /// Set dropout rate
    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
}

impl<T> ModelBuilder<T> for CNNBuilder<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    type Model = CNN<T>;

    fn build(self) -> Self::Model {
        let input_channels = self
            .input_channels
            .expect("Input channels must be specified");
        let num_classes = self
            .num_classes
            .expect("Number of classes must be specified");

        CNN::new(
            input_channels,
            num_classes,
            self.hidden_channels,
            self.dropout_rate,
        )
    }
}

/// ResNet ブロック
/// ResNet block
#[derive(Debug)]
pub struct ResNetBlock<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    conv2: Conv2d<T>,
    bn2: BatchNorm2d<T>,
    relu: ReLU<T>,
    downsample: Option<Sequential<T>>,
}

impl<T> ResNetBlock<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    /// 新しい ResNet ブロックを作成
    /// Create a new ResNet block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<Sequential<T>>,
    ) -> Self {
        ResNetBlock {
            conv1: Conv2d::new(
                in_channels,
                out_channels,
                (3, 3),
                Some((stride, stride)),
                Some((1, 1)),
                Some(true),
            ),
            bn1: BatchNorm2d::new(out_channels, None, None, None),
            conv2: Conv2d::new(
                out_channels,
                out_channels,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                Some(true),
            ),
            bn2: BatchNorm2d::new(out_channels, None, None, None),
            relu: ReLU::new(),
            downsample,
        }
    }
}

impl<T> Module<T> for ResNetBlock<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let identity = input.clone();

        let mut out = self.conv1.forward(input);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);

        out = self.conv2.forward(&out);
        out = self.bn2.forward(&out);

        // ダウンサンプリングが必要な場合
        let _identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)
        } else {
            identity
        };

        // 残差接続 (簡略化実装)
        // out = out + identity;
        out = self.relu.forward(&out);

        out
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.conv1.parameters();
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());

        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }

        params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// ResNet モデル
/// ResNet model
#[derive(Debug)]
pub struct ResNet<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    relu: ReLU<T>,
    maxpool: MaxPool2d,
    layers: Vec<Sequential<T>>,
    avgpool: Sequential<T>, // Global Average Pooling の代替
    fc: Linear<T>,
    mode: ModelMode,
    num_classes: usize,
}

impl<T> ResNet<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    /// 新しい ResNet モデルを作成
    /// Create a new ResNet model
    pub fn new(
        layers: Vec<usize>, // 各レイヤーのブロック数
        num_classes: usize,
    ) -> Self {
        let conv1 = Conv2d::new(3, 64, (7, 7), Some((2, 2)), Some((3, 3)), Some(true));
        let bn1 = BatchNorm2d::new(64, None, None, None);
        let relu = ReLU::new();
        let maxpool = MaxPool2d::new((3, 3), Some((2, 2)), Some((0, 0)));

        // ResNet レイヤーの構築
        let mut resnet_layers = Vec::new();
        let channels = [64, 128, 256, 512];
        let mut in_channels = 64;

        for (i, &num_blocks) in layers.iter().enumerate() {
            let out_channels = channels[i];
            let stride = if i == 0 { 1 } else { 2 };

            let mut layer = Sequential::new();

            // 最初のブロック（ストライドあり）
            let downsample = if stride != 1 || in_channels != out_channels {
                let mut ds = Sequential::new();
                ds.add_module(Conv2d::new(
                    in_channels,
                    out_channels,
                    (1, 1),
                    Some((stride, stride)),
                    Some((0, 0)),
                    Some(true),
                ));
                ds.add_module(BatchNorm2d::new(out_channels, None, None, None));
                Some(ds)
            } else {
                None
            };

            layer.add_module(ResNetBlock::new(
                in_channels,
                out_channels,
                stride,
                downsample,
            ));

            // 残りのブロック
            for _ in 1..num_blocks {
                layer.add_module(ResNetBlock::new(out_channels, out_channels, 1, None));
            }

            resnet_layers.push(layer);
            in_channels = out_channels;
        }

        let avgpool = Sequential::new(); // Global Average Pooling の実装が必要
        let fc = Linear::new(512, num_classes);

        ResNet {
            conv1,
            bn1,
            relu,
            maxpool,
            layers: resnet_layers,
            avgpool,
            fc,
            mode: ModelMode::Train,
            num_classes,
        }
    }

    /// ResNet-18 を作成
    /// Create ResNet-18
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(vec![2, 2, 2, 2], num_classes)
    }

    /// ResNet-34 を作成
    /// Create ResNet-34
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(vec![3, 4, 6, 3], num_classes)
    }

    /// ResNet-50 を作成
    /// Create ResNet-50
    pub fn resnet50(num_classes: usize) -> Self {
        Self::new(vec![3, 4, 6, 3], num_classes)
    }
}

impl<T> Module<T> for ResNet<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut x = self.conv1.forward(input);
        x = self.bn1.forward(&x);
        x = self.relu.forward(&x);
        x = self.maxpool.forward(&x);

        // ResNet レイヤーを順次適用
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        x = self.avgpool.forward(&x);
        x = self.fc.forward(&x);

        x
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.conv1.parameters();
        params.extend(self.bn1.parameters());

        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        params.extend(self.avgpool.parameters());
        params.extend(self.fc.parameters());

        params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for ResNet<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    fn train(&mut self) {
        self.mode = ModelMode::Train;
    }

    fn eval(&mut self) {
        self.mode = ModelMode::Eval;
    }

    fn mode(&self) -> ModelMode {
        self.mode
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("model_type".to_string(), "ResNet".to_string());
        config.insert("num_classes".to_string(), self.num_classes.to_string());
        config.insert("num_layers".to_string(), self.layers.len().to_string());
        config
    }

    fn summary(&self) -> String {
        format!(
            "ResNet Model:\n  - Layers: {}\n  - Classes: {}\n  - Mode: {:?}",
            self.layers.len(),
            self.num_classes,
            self.mode
        )
    }
}

/// ResNet ビルダー
/// ResNet builder
#[derive(Debug, Default)]
pub struct ResNetBuilder<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    layers: Vec<usize>,
    num_classes: Option<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ResNetBuilder<T>
where
    T: Float + 'static + Send + Sync + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        ResNetBuilder {
            layers: vec![2, 2, 2, 2], // ResNet-18 デフォルト
            num_classes: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// レイヤー構成を設定
    /// Set layer configuration
    pub fn layers(mut self, layers: Vec<usize>) -> Self {
        self.layers = layers;
        self
    }

    /// クラス数を設定
    /// Set number of classes
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.num_classes = Some(classes);
        self
    }

    /// ResNet-18 設定
    /// ResNet-18 configuration
    pub fn resnet18(mut self) -> Self {
        self.layers = vec![2, 2, 2, 2];
        self
    }

    /// ResNet-34 設定
    /// ResNet-34 configuration
    pub fn resnet34(mut self) -> Self {
        self.layers = vec![3, 4, 6, 3];
        self
    }

    /// ResNet-50 設定
    /// ResNet-50 configuration
    pub fn resnet50(mut self) -> Self {
        self.layers = vec![3, 4, 6, 3];
        self
    }
}

impl<T> ModelBuilder<T> for ResNetBuilder<T>
where
    T: Float
        + 'static
        + Send
        + Sync
        + Debug
        + Default
        + Copy
        + From<f32>
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::fmt::Display,
{
    type Model = ResNet<T>;

    fn build(self) -> Self::Model {
        let num_classes = self
            .num_classes
            .expect("Number of classes must be specified");
        ResNet::new(self.layers, num_classes)
    }
}
