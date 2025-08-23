//! RNN/LSTM モデル実装
//! RNN/LSTM model implementations

use crate::autograd::Variable;
use crate::nn::{Module, Linear, LSTM, RNN, Dropout, Embedding};
use crate::models::{Model, ModelMode, ModelBuilder};
use num_traits::Float;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

/// 基本的な RNN モデル
/// Basic RNN model
#[derive(Debug)]
pub struct RNNModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    embedding: Option<Embedding<T>>,
    rnn: RNN<T>,
    dropout: Dropout<T>,
    classifier: Linear<T>,
    mode: ModelMode,
    vocab_size: Option<usize>,
    hidden_size: usize,
    num_classes: usize,
}

impl<T> RNNModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい RNN モデルを作成
    /// Create a new RNN model
    pub fn new(
        vocab_size: Option<usize>,
        embedding_dim: Option<usize>,
        hidden_size: usize,
        num_layers: usize,
        num_classes: usize,
        dropout_rate: f64,
        bidirectional: bool,
    ) -> Self {
        let embedding = if let (Some(vocab), Some(embed_dim)) = (vocab_size, embedding_dim) {
            Some(Embedding::new(vocab, embed_dim, None, None, None))
        } else {
            None
        };
        
        let input_size = embedding_dim.unwrap_or(hidden_size);
        let rnn = RNN::new(input_size, hidden_size, Some(num_layers), Some(true), Some(true), Some(<T as From<f32>>::from(dropout_rate as f32)), Some(bidirectional));
        let dropout = Dropout::new(<T as From<f32>>::from(dropout_rate as f32), false);
        
        let classifier_input_size = if bidirectional { hidden_size * 2 } else { hidden_size };
        let classifier = Linear::new(classifier_input_size, num_classes);
        
        RNNModel {
            embedding,
            rnn,
            dropout,
            classifier,
            mode: ModelMode::Train,
            vocab_size,
            hidden_size,
            num_classes,
        }
    }
}

impl<T> Module<T> for RNNModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut x = input.clone();
        
        // 埋め込み層（オプション）
        if let Some(ref embedding) = self.embedding {
            x = embedding.forward(&x);
        }
        
        // RNN 層
        let output = self.rnn.forward(&x);
        
        // 最後の時刻の出力を使用（分類タスクの場合）
        let last_output = self.extract_last_output(&output);
        
        // ドロップアウト
        let dropped = self.dropout.forward(&last_output);
        
        // 分類器
        self.classifier.forward(&dropped)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        
        if let Some(ref embedding) = self.embedding {
            params.extend(embedding.parameters());
        }
        
        params.extend(self.rnn.parameters());
        params.extend(self.classifier.parameters());
        
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for RNNModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
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
        config.insert("model_type".to_string(), "RNN".to_string());
        config.insert("hidden_size".to_string(), self.hidden_size.to_string());
        config.insert("num_classes".to_string(), self.num_classes.to_string());
        if let Some(vocab_size) = self.vocab_size {
            config.insert("vocab_size".to_string(), vocab_size.to_string());
        }
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "RNN Model:\n  - Hidden size: {}\n  - Classes: {}\n  - Vocab size: {:?}\n  - Mode: {:?}",
            self.hidden_size,
            self.num_classes,
            self.vocab_size,
            self.mode
        )
    }
}

impl<T> RNNModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 最後の時刻の出力を抽出
    /// Extract last timestep output
    fn extract_last_output(&self, output: &Variable<T>) -> Variable<T> {
        // 実装は簡略化 - 実際にはテンソルの最後の時刻を抽出
        output.clone()
    }
}

/// LSTM モデル
/// LSTM model
#[derive(Debug)]
pub struct LSTMModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    embedding: Option<Embedding<T>>,
    lstm: LSTM<T>,
    dropout: Dropout<T>,
    classifier: Linear<T>,
    mode: ModelMode,
    vocab_size: Option<usize>,
    hidden_size: usize,
    num_classes: usize,
}

impl<T> LSTMModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい LSTM モデルを作成
    /// Create a new LSTM model
    pub fn new(
        vocab_size: Option<usize>,
        embedding_dim: Option<usize>,
        hidden_size: usize,
        num_layers: usize,
        num_classes: usize,
        dropout_rate: f64,
        bidirectional: bool,
    ) -> Self {
        let embedding = if let (Some(vocab), Some(embed_dim)) = (vocab_size, embedding_dim) {
            Some(Embedding::new(vocab, embed_dim, None, None, None))
        } else {
            None
        };
        
        let input_size = embedding_dim.unwrap_or(hidden_size);
        let lstm = LSTM::new(input_size, hidden_size, num_layers, true, true, <T as From<f32>>::from(dropout_rate as f32), bidirectional);
        let dropout = Dropout::new(<T as From<f32>>::from(dropout_rate as f32), false);
        
        let classifier_input_size = if bidirectional { hidden_size * 2 } else { hidden_size };
        let classifier = Linear::new(classifier_input_size, num_classes);
        
        LSTMModel {
            embedding,
            lstm,
            dropout,
            classifier,
            mode: ModelMode::Train,
            vocab_size,
            hidden_size,
            num_classes,
        }
    }
}

impl<T> Module<T> for LSTMModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut x = input.clone();
        
        // 埋め込み層（オプション）
        if let Some(ref embedding) = self.embedding {
            x = embedding.forward(&x);
        }
        
        // LSTM 層
        let (output, _hidden) = self.lstm.forward(&x, None);
        
        // 最後の時刻の出力を使用
        let last_output = self.extract_last_output(&output);
        
        // ドロップアウト
        let dropped = self.dropout.forward(&last_output);
        
        // 分類器
        self.classifier.forward(&dropped)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        
        if let Some(ref embedding) = self.embedding {
            params.extend(embedding.parameters());
        }
        
        params.extend(self.lstm.parameters());
        params.extend(self.classifier.parameters());
        
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for LSTMModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
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
        config.insert("model_type".to_string(), "LSTM".to_string());
        config.insert("hidden_size".to_string(), self.hidden_size.to_string());
        config.insert("num_classes".to_string(), self.num_classes.to_string());
        if let Some(vocab_size) = self.vocab_size {
            config.insert("vocab_size".to_string(), vocab_size.to_string());
        }
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "LSTM Model:\n  - Hidden size: {}\n  - Classes: {}\n  - Vocab size: {:?}\n  - Mode: {:?}",
            self.hidden_size,
            self.num_classes,
            self.vocab_size,
            self.mode
        )
    }
}

impl<T> LSTMModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 最後の時刻の出力を抽出
    /// Extract last timestep output
    fn extract_last_output(&self, output: &Variable<T>) -> Variable<T> {
        // 実装は簡略化 - 実際にはテンソルの最後の時刻を抽出
        output.clone()
    }
}

/// Seq2Seq モデル（エンコーダー・デコーダー）
/// Seq2Seq model (encoder-decoder)
#[derive(Debug)]
pub struct Seq2SeqModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    encoder: LSTMModel<T>,
    decoder: LSTMModel<T>,
    mode: ModelMode,
}

impl<T> Seq2SeqModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい Seq2Seq モデルを作成
    /// Create a new Seq2Seq model
    pub fn new(
        input_vocab_size: usize,
        output_vocab_size: usize,
        embedding_dim: usize,
        hidden_size: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Self {
        let encoder = LSTMModel::new(
            Some(input_vocab_size),
            Some(embedding_dim),
            hidden_size,
            num_layers,
            hidden_size, // エンコーダーは隠れ状態を出力
            dropout_rate,
            false,
        );
        
        let decoder = LSTMModel::new(
            Some(output_vocab_size),
            Some(embedding_dim),
            hidden_size,
            num_layers,
            output_vocab_size,
            dropout_rate,
            false,
        );
        
        Seq2SeqModel {
            encoder,
            decoder,
            mode: ModelMode::Train,
        }
    }
}

impl<T> Module<T> for Seq2SeqModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // エンコーダーで入力をエンコード
        let encoded = self.encoder.forward(input);
        
        // デコーダーで出力を生成（簡略化実装）
        self.decoder.forward(&encoded)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for Seq2SeqModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn train(&mut self) {
        self.mode = ModelMode::Train;
        Model::train(&mut self.encoder);
        Model::train(&mut self.decoder);
    }
    
    fn eval(&mut self) {
        self.mode = ModelMode::Eval;
        Model::eval(&mut self.encoder);
        Model::eval(&mut self.decoder);
    }
    
    fn mode(&self) -> ModelMode {
        self.mode
    }
    
    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("model_type".to_string(), "Seq2Seq".to_string());
        config.extend(self.encoder.config());
        config.extend(self.decoder.config());
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "Seq2Seq Model:\n  - Encoder: {}\n  - Decoder: {}\n  - Mode: {:?}",
            self.encoder.summary(),
            self.decoder.summary(),
            self.mode
        )
    }
}

/// RNN モデルビルダー
/// RNN model builder
#[derive(Debug, Default)]
pub struct RNNModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    vocab_size: Option<usize>,
    embedding_dim: Option<usize>,
    hidden_size: usize,
    num_layers: usize,
    num_classes: Option<usize>,
    dropout_rate: f64,
    bidirectional: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RNNModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        RNNModelBuilder {
            vocab_size: None,
            embedding_dim: None,
            hidden_size: 128,
            num_layers: 1,
            num_classes: None,
            dropout_rate: 0.5,
            bidirectional: false,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// 語彙サイズを設定
    /// Set vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }
    
    /// 埋め込み次元を設定
    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }
    
    /// 隠れ層サイズを設定
    /// Set hidden size
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }
    
    /// レイヤー数を設定
    /// Set number of layers
    pub fn num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }
    
    /// クラス数を設定
    /// Set number of classes
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.num_classes = Some(classes);
        self
    }
    
    /// ドロップアウト率を設定
    /// Set dropout rate
    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
    
    /// 双方向を設定
    /// Set bidirectional
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }
}

impl<T> ModelBuilder<T> for RNNModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    type Model = RNNModel<T>;
    
    fn build(self) -> Self::Model {
        let num_classes = self.num_classes.expect("Number of classes must be specified");
        
        RNNModel::new(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            num_classes,
            self.dropout_rate,
            self.bidirectional,
        )
    }
}

/// LSTM モデルビルダー
/// LSTM model builder
#[derive(Debug, Default)]
pub struct LSTMModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    vocab_size: Option<usize>,
    embedding_dim: Option<usize>,
    hidden_size: usize,
    num_layers: usize,
    num_classes: Option<usize>,
    dropout_rate: f64,
    bidirectional: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LSTMModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        LSTMModelBuilder {
            vocab_size: None,
            embedding_dim: None,
            hidden_size: 128,
            num_layers: 1,
            num_classes: None,
            dropout_rate: 0.5,
            bidirectional: false,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// 語彙サイズを設定
    /// Set vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }
    
    /// 埋め込み次元を設定
    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }
    
    /// 隠れ層サイズを設定
    /// Set hidden size
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }
    
    /// レイヤー数を設定
    /// Set number of layers
    pub fn num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }
    
    /// クラス数を設定
    /// Set number of classes
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.num_classes = Some(classes);
        self
    }
    
    /// ドロップアウト率を設定
    /// Set dropout rate
    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
    
    /// 双方向を設定
    /// Set bidirectional
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }
}

impl<T> ModelBuilder<T> for LSTMModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    type Model = LSTMModel<T>;
    
    fn build(self) -> Self::Model {
        let num_classes = self.num_classes.expect("Number of classes must be specified");
        
        LSTMModel::new(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            num_classes,
            self.dropout_rate,
            self.bidirectional,
        )
    }
}
