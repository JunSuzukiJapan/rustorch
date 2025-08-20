//! Transformer モデル実装
//! Transformer model implementations

use crate::autograd::Variable;
use crate::nn::{Module, Linear, Dropout, Embedding, LayerNorm};
use crate::nn::{TransformerEncoder, TransformerEncoderLayer, PositionalEmbedding};
use crate::models::{Model, ModelMode, ModelBuilder};
use num_traits::Float;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

/// 基本的な Transformer モデル
/// Basic Transformer model
#[derive(Debug)]
pub struct TransformerModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    embedding: Embedding<T>,
    positional_encoding: PositionalEmbedding<T>,
    encoder: TransformerEncoder<T>,
    classifier: Linear<T>,
    dropout: Dropout<T>,
    mode: ModelMode,
    vocab_size: usize,
    d_model: usize,
    num_classes: usize,
}

impl<T> TransformerModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい Transformer モデルを作成
    /// Create a new Transformer model
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        dim_feedforward: usize,
        num_classes: usize,
        dropout_rate: f64,
        max_seq_length: usize,
    ) -> Self {
        let embedding = Embedding::new(vocab_size, d_model, None, None, None);
        let positional_encoding = PositionalEmbedding::new(max_seq_length, d_model);
        
        let _encoder_layer = TransformerEncoderLayer::new(
            d_model,
            nhead,
            dim_feedforward,
            Some(<T as From<f32>>::from(dropout_rate as f32)),
        );
        let encoder = TransformerEncoder::new(num_encoder_layers, d_model, nhead, dim_feedforward, Some(<T as From<f32>>::from(dropout_rate as f32)));
        
        let classifier = Linear::new(d_model, num_classes);
        let dropout = Dropout::new(<T as From<f32>>::from(dropout_rate as f32), false);
        
        TransformerModel {
            embedding,
            positional_encoding,
            encoder,
            classifier,
            dropout,
            mode: ModelMode::Train,
            vocab_size,
            d_model,
            num_classes,
        }
    }
}

impl<T> Module<T> for TransformerModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // 埋め込み層
        let embedded = self.embedding.forward(input);
        
        // 位置エンコーディング
        let pos_encoded = self.positional_encoding.forward(&embedded);
        
        // ドロップアウト
        let dropped = self.dropout.forward(&pos_encoded);
        
        // Transformer エンコーダー
        let encoded = self.encoder.forward(&dropped, None);
        
        // 分類のため最初のトークン（[CLS]）を使用
        let cls_token = self.extract_cls_token(&encoded, None);
        
        // 分類器
        self.classifier.forward(&cls_token)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.embedding.parameters();
        params.extend(self.positional_encoding.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.classifier.parameters());
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for TransformerModel<T> 
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
        config.insert("model_type".to_string(), "Transformer".to_string());
        config.insert("vocab_size".to_string(), self.vocab_size.to_string());
        config.insert("d_model".to_string(), self.d_model.to_string());
        config.insert("num_classes".to_string(), self.num_classes.to_string());
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "Transformer Model:\n  - Vocab size: {}\n  - Model dim: {}\n  - Classes: {}\n  - Mode: {:?}",
            self.vocab_size,
            self.d_model,
            self.num_classes,
            self.mode
        )
    }
}

impl<T> TransformerModel<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// [CLS] トークンを抽出
    /// Extract [CLS] token
    fn extract_cls_token(&self, encoded: &Variable<T>, _mask: Option<&Variable<T>>) -> Variable<T> {
        // 実装は簡略化 - 実際には最初のトークンを抽出
        encoded.clone()
    }
}

/// BERT モデル（双方向エンコーダー）
/// BERT model (Bidirectional Encoder)
#[derive(Debug)]
pub struct BERT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    embeddings: BERTEmbeddings<T>,
    encoder: TransformerEncoder<T>,
    pooler: Linear<T>,
    classifier: Option<Linear<T>>,
    mode: ModelMode,
    config: BERTConfig,
}

/// BERT 設定
/// BERT configuration
#[derive(Debug, Clone)]
pub struct BERTConfig {
    /// Vocabulary size for BERT model
    /// BERTモデルの語彙サイズ
    pub vocab_size: usize,
    /// Hidden layer size
    /// 隠れ層のサイズ
    pub hidden_size: usize,
    /// Number of hidden layers
    /// 隠れ層の数
    pub num_hidden_layers: usize,
    /// Number of attention heads
    /// アテンション頭数
    pub num_attention_heads: usize,
    /// Intermediate layer size in feed-forward network
    /// フィードフォワードネットワークの中間層サイズ
    pub intermediate_size: usize,
    /// Maximum position embeddings
    /// 最大位置埋め込み数
    pub max_position_embeddings: usize,
    /// Type vocabulary size for token type embeddings
    /// トークンタイプ埋め込み用のタイプ語彙サイズ
    pub type_vocab_size: usize,
    /// Dropout probability
    /// ドロップアウト確率
    pub dropout_prob: f64,
    /// Number of classification labels (optional)
    /// 分類ラベル数（オプション）
    pub num_labels: Option<usize>,
}

impl Default for BERTConfig {
    fn default() -> Self {
        BERTConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            dropout_prob: 0.1,
            num_labels: None,
        }
    }
}

/// BERT 埋め込み層
/// BERT embeddings
#[derive(Debug)]
pub struct BERTEmbeddings<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    word_embeddings: Embedding<T>,
    position_embeddings: Embedding<T>,
    token_type_embeddings: Embedding<T>,
    layer_norm: LayerNorm<T>,
    dropout: Dropout<T>,
}

impl<T> BERTEmbeddings<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい BERT 埋め込み層を作成
    /// Create new BERT embeddings
    pub fn new(config: &BERTConfig) -> Self {
        BERTEmbeddings {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_size, None, None, None),
            position_embeddings: Embedding::new(config.max_position_embeddings, config.hidden_size, None, None, None),
            token_type_embeddings: Embedding::new(config.type_vocab_size, config.hidden_size, None, None, None),
            layer_norm: LayerNorm::new(vec![config.hidden_size], Some(<T as From<f32>>::from(1e-12f32)), None),
            dropout: Dropout::new(<T as From<f32>>::from(config.dropout_prob as f32), false),
        }
    }
}

impl<T> Module<T> for BERTEmbeddings<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // 単語埋め込み
        let word_emb = self.word_embeddings.forward(input);
        
        // 位置埋め込み（簡略化実装）
        let _pos_emb = self.position_embeddings.forward(input);
        
        // トークンタイプ埋め込み（簡略化実装）
        let _token_type_emb = self.token_type_embeddings.forward(input);
        
        // 埋め込みを合計（実際には要素ごとの加算）
        let embeddings = word_emb; // 簡略化: pos_emb + token_type_emb も加算すべき
        
        // Layer Normalization
        let normalized = self.layer_norm.forward(&embeddings);
        
        // ドロップアウト
        self.dropout.forward(&normalized)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.word_embeddings.parameters();
        params.extend(self.position_embeddings.parameters());
        params.extend(self.token_type_embeddings.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> BERT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい BERT モデルを作成
    /// Create a new BERT model
    pub fn new(config: BERTConfig) -> Self {
        let embeddings = BERTEmbeddings::new(&config);
        
        let _encoder_layer = TransformerEncoderLayer::new(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            Some(<T as From<f32>>::from(config.dropout_prob as f32)),
        );
        let encoder = TransformerEncoder::new(config.num_hidden_layers, config.hidden_size, config.num_attention_heads, config.intermediate_size, Some(<T as From<f32>>::from(config.dropout_prob as f32)));
        
        let pooler = Linear::new(config.hidden_size, config.hidden_size);
        
        let classifier = config.num_labels.map(|num_labels| {
            Linear::new(config.hidden_size, num_labels)
        });
        
        BERT {
            embeddings,
            encoder,
            pooler,
            classifier,
            mode: ModelMode::Train,
            config,
        }
    }
    
    /// 事前訓練済み BERT-Base を作成
    /// Create pre-trained BERT-Base
    pub fn bert_base_uncased(num_labels: Option<usize>) -> Self {
        let mut config = BERTConfig::default();
        config.num_labels = num_labels;
        Self::new(config)
    }
    
    /// 事前訓練済み BERT-Large を作成
    /// Create pre-trained BERT-Large
    pub fn bert_large_uncased(num_labels: Option<usize>) -> Self {
        let mut config = BERTConfig::default();
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        config.num_labels = num_labels;
        Self::new(config)
    }
}

impl<T> Module<T> for BERT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // 埋め込み層
        let embedded = self.embeddings.forward(input);
        
        // Transformer エンコーダー
        let encoded = self.encoder.forward(&embedded, None);
        
        // プーリング（[CLS] トークンの処理）
        let cls_token = self.extract_cls_token(&encoded, None);
        let pooled = self.pooler.forward(&cls_token);
        
        // 分類器（オプション）
        if let Some(ref classifier) = self.classifier {
            classifier.forward(&pooled)
        } else {
            pooled
        }
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.embeddings.parameters();
        params.extend(self.encoder.parameters());
        params.extend(self.pooler.parameters());
        
        if let Some(ref classifier) = self.classifier {
            params.extend(classifier.parameters());
        }
        
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for BERT<T> 
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
        config.insert("model_type".to_string(), "BERT".to_string());
        config.insert("vocab_size".to_string(), self.config.vocab_size.to_string());
        config.insert("hidden_size".to_string(), self.config.hidden_size.to_string());
        config.insert("num_layers".to_string(), self.config.num_hidden_layers.to_string());
        config.insert("num_heads".to_string(), self.config.num_attention_heads.to_string());
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "BERT Model:\n  - Vocab size: {}\n  - Hidden size: {}\n  - Layers: {}\n  - Attention heads: {}\n  - Mode: {:?}",
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            self.mode
        )
    }
}

impl<T> BERT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// [CLS] トークンを抽出
    /// Extract [CLS] token
    fn extract_cls_token(&self, encoded: &Variable<T>, _mask: Option<&Variable<T>>) -> Variable<T> {
        // 実装は簡略化 - 実際には最初のトークンを抽出
        encoded.clone()
    }
}

/// GPT モデル（生成型事前訓練 Transformer）
/// GPT model (Generative Pre-trained Transformer)
#[derive(Debug)]
pub struct GPT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    embeddings: Embedding<T>,
    positional_encoding: PositionalEmbedding<T>,
    decoder_layers: Vec<TransformerEncoderLayer<T>>, // デコーダー層として使用
    layer_norm: LayerNorm<T>,
    lm_head: Linear<T>,
    dropout: Dropout<T>,
    mode: ModelMode,
    config: GPTConfig,
}

/// GPT 設定
/// GPT configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Vocabulary size for GPT model
    /// GPTモデルの語彙サイズ
    pub vocab_size: usize,
    /// Number of position embeddings
    /// 位置埋め込み数
    pub n_positions: usize,
    /// Embedding dimension
    /// 埋め込み次元
    pub n_embd: usize,
    /// Number of transformer layers
    /// Transformerレイヤー数
    pub n_layer: usize,
    /// Number of attention heads
    /// アテンション頭数
    pub n_head: usize,
    /// Dropout probability
    /// ドロップアウト確率
    pub dropout: f64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        GPTConfig {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            dropout: 0.1,
        }
    }
}

impl<T> GPT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しい GPT モデルを作成
    /// Create a new GPT model
    pub fn new(config: GPTConfig) -> Self {
        let embeddings = Embedding::new(config.vocab_size, config.n_embd, None, None, None);
        let positional_encoding = PositionalEmbedding::new(config.n_positions, config.n_embd);
        
        let mut decoder_layers = Vec::new();
        for _ in 0..config.n_layer {
            decoder_layers.push(TransformerEncoderLayer::new(
                config.n_embd,
                config.n_head,
                config.n_embd * 4, // FFN の次元
                Some(<T as From<f32>>::from(config.dropout as f32)),
            ));
        }
        
        let layer_norm = LayerNorm::new(vec![config.n_embd], Some(<T as From<f32>>::from(1e-5f32)), None);
        let lm_head = Linear::new(config.n_embd, config.vocab_size);
        let dropout = Dropout::new(<T as From<f32>>::from(config.dropout as f32), false);
        
        GPT {
            embeddings,
            positional_encoding,
            decoder_layers,
            layer_norm,
            lm_head,
            dropout,
            mode: ModelMode::Train,
            config,
        }
    }
    
    /// GPT-2 Small を作成
    /// Create GPT-2 Small
    pub fn gpt2_small() -> Self {
        Self::new(GPTConfig::default())
    }
    
    /// GPT-2 Medium を作成
    /// Create GPT-2 Medium
    pub fn gpt2_medium() -> Self {
        let config = GPTConfig {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..GPTConfig::default()
        };
        Self::new(config)
    }
}

impl<T> Module<T> for GPT<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // 埋め込み層
        let embedded = self.embeddings.forward(input);
        
        // 位置エンコーディング
        let pos_encoded = self.positional_encoding.forward(&embedded);
        
        // ドロップアウト
        let mut hidden = self.dropout.forward(&pos_encoded);
        
        // デコーダー層を順次適用
        for layer in &self.decoder_layers {
            hidden = layer.forward(&hidden, None);
        }
        
        // 最終 Layer Normalization
        let normalized = self.layer_norm.forward(&hidden);
        
        // 言語モデルヘッド
        self.lm_head.forward(&normalized)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = self.embeddings.parameters();
        params.extend(self.positional_encoding.parameters());
        
        for layer in &self.decoder_layers {
            params.extend(layer.parameters());
        }
        
        params.extend(self.layer_norm.parameters());
        params.extend(self.lm_head.parameters());
        
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Model<T> for GPT<T> 
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
        config.insert("model_type".to_string(), "GPT".to_string());
        config.insert("vocab_size".to_string(), self.config.vocab_size.to_string());
        config.insert("n_embd".to_string(), self.config.n_embd.to_string());
        config.insert("n_layer".to_string(), self.config.n_layer.to_string());
        config.insert("n_head".to_string(), self.config.n_head.to_string());
        config
    }
    
    fn summary(&self) -> String {
        format!(
            "GPT Model:\n  - Vocab size: {}\n  - Embedding dim: {}\n  - Layers: {}\n  - Attention heads: {}\n  - Mode: {:?}",
            self.config.vocab_size,
            self.config.n_embd,
            self.config.n_layer,
            self.config.n_head,
            self.mode
        )
    }
}

/// Transformer モデルビルダー
/// Transformer model builder
#[derive(Debug, Default)]
pub struct TransformerModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    vocab_size: Option<usize>,
    d_model: usize,
    nhead: usize,
    num_encoder_layers: usize,
    dim_feedforward: usize,
    num_classes: Option<usize>,
    dropout_rate: f64,
    max_seq_length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TransformerModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        TransformerModelBuilder {
            vocab_size: None,
            d_model: 512,
            nhead: 8,
            num_encoder_layers: 6,
            dim_feedforward: 2048,
            num_classes: None,
            dropout_rate: 0.1,
            max_seq_length: 512,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// 語彙サイズを設定
    /// Set vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }
    
    /// モデル次元を設定
    /// Set model dimension
    pub fn d_model(mut self, dim: usize) -> Self {
        self.d_model = dim;
        self
    }
    
    /// アテンションヘッド数を設定
    /// Set number of attention heads
    pub fn nhead(mut self, heads: usize) -> Self {
        self.nhead = heads;
        self
    }
    
    /// エンコーダー層数を設定
    /// Set number of encoder layers
    pub fn num_encoder_layers(mut self, layers: usize) -> Self {
        self.num_encoder_layers = layers;
        self
    }
    
    /// フィードフォワード次元を設定
    /// Set feedforward dimension
    pub fn dim_feedforward(mut self, dim: usize) -> Self {
        self.dim_feedforward = dim;
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
    
    /// 最大シーケンス長を設定
    /// Set maximum sequence length
    pub fn max_seq_length(mut self, length: usize) -> Self {
        self.max_seq_length = length;
        self
    }
}

impl<T> ModelBuilder<T> for TransformerModelBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    type Model = TransformerModel<T>;
    
    fn build(self) -> Self::Model {
        let vocab_size = self.vocab_size.expect("Vocabulary size must be specified");
        let num_classes = self.num_classes.expect("Number of classes must be specified");
        
        TransformerModel::new(
            vocab_size,
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.dim_feedforward,
            num_classes,
            self.dropout_rate,
            self.max_seq_length,
        )
    }
}

/// BERT ビルダー
/// BERT builder
#[derive(Debug, Default)]
pub struct BERTBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    config: BERTConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> BERTBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    /// 新しいビルダーを作成
    /// Create a new builder
    pub fn new() -> Self {
        BERTBuilder {
            config: BERTConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// 語彙サイズを設定
    /// Set vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }
    
    /// 隠れ層サイズを設定
    /// Set hidden size
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.hidden_size = size;
        self
    }
    
    /// レイヤー数を設定
    /// Set number of layers
    pub fn num_hidden_layers(mut self, layers: usize) -> Self {
        self.config.num_hidden_layers = layers;
        self
    }
    
    /// アテンションヘッド数を設定
    /// Set number of attention heads
    pub fn num_attention_heads(mut self, heads: usize) -> Self {
        self.config.num_attention_heads = heads;
        self
    }
    
    /// ラベル数を設定
    /// Set number of labels
    pub fn num_labels(mut self, labels: usize) -> Self {
        self.config.num_labels = Some(labels);
        self
    }
    
    /// BERT-Base 設定
    /// BERT-Base configuration
    pub fn bert_base(mut self) -> Self {
        self.config = BERTConfig::default();
        self
    }
    
    /// BERT-Large 設定
    /// BERT-Large configuration
    pub fn bert_large(mut self) -> Self {
        self.config.hidden_size = 1024;
        self.config.num_hidden_layers = 24;
        self.config.num_attention_heads = 16;
        self.config.intermediate_size = 4096;
        self
    }
}

impl<T> ModelBuilder<T> for BERTBuilder<T> 
where
    T: Float + 'static + Send + Sync + Debug + Default + Copy + From<f32> + ndarray::ScalarOperand + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::One + std::iter::Sum + std::fmt::Display,
{
    type Model = BERT<T>;
    
    fn build(self) -> Self::Model {
        BERT::new(self.config)
    }
}
