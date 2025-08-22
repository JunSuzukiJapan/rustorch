//! ニューラルネットワークモジュールの定義
//! Neural network module definitions.

pub mod linear;
pub mod conv_base;
pub mod safe_ops;
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod conv_transpose;
pub mod pool2d;
pub mod adaptive_pool;
pub mod batchnorm;
pub mod dropout;
pub mod activation;
pub mod loss;
pub mod attention;
pub mod embedding;
pub mod transformer;
pub mod normalization;
pub mod rnn;
pub mod lstm;
pub mod gru;
pub mod quantization;
pub mod pruning;

use crate::autograd::Variable;
use num_traits::Float;
use std::any::Any;
use std::fmt::Debug;
use std::marker::{Send, Sync};

/// ニューラルネットワークモジュールのトレイト
/// A trait for neural network modules.
pub trait Module<T>: Send + Sync + Debug 
where
    T: Float + 'static + Send + Sync,
{
    /// モジュールの順伝搬を実行します。
    /// Performs the forward pass of the module.
    fn forward(&self, input: &Variable<T>) -> Variable<T>;
    
    /// モジュールのパラメータへの参照を返します。
    /// Returns a reference to the module's parameters.
    fn parameters(&self) -> Vec<Variable<T>>;
    
    /// ダウンキャストのための`&dyn Any`としてのモジュールへの参照を返します。
    /// Returns a reference to the module as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;
    
    /// モジュールを訓練モードに設定します。
    /// Sets the module to training mode.
    fn train(&mut self) {
        // Default implementation - do nothing
    }
    
    /// モジュールを評価モードに設定します。
    /// Sets the module to evaluation mode.
    fn eval(&mut self) {
        // Default implementation - do nothing
    }
}

/// モジュールを順番に適用するコンテナ
/// A sequential container for modules.
#[derive(Debug, Default)]
pub struct Sequential<T> {
    modules: Vec<Box<dyn Module<T>>>,
}

impl<T> Sequential<T> 
where
    T: Float + 'static + Send + Sync + Debug,
{
    /// 空のシーケンシャルコンテナを作成します。
    /// Creates a new empty sequential container.
    pub fn new() -> Self {
        Sequential { modules: Vec::new() }
    }
    
    /// モジュールをコンテナに追加します。
    /// Adds a module to the container.
    pub fn add_module<M: Module<T> + 'static>(&mut self, module: M) -> &mut Self {
        self.modules.push(Box::new(module));
        self
    }
    
    /// 指定されたインデックスのモジュールへの参照を返します。
    /// Returns a reference to the module at the given index.
    pub fn get_module(&self, index: usize) -> Option<&dyn Module<T>> {
        self.modules.get(index).map(|m| &**m)
    }
    
    /// コンテナ内のモジュールの数を返します。
    /// Returns the number of modules in the container.
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    /// コンテナが空の場合は`true`を返します。
    /// Returns `true` if the container is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl<T> Module<T> for Sequential<T> 
where
    T: Float + 'static + Send + Sync + Debug,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x);
        }
        x
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Re-export neural network modules
// ニューラルネットワークモジュールを再エクスポート
/// 線形（全結合）レイヤー
/// Linear (fully connected) layer
pub use linear::Linear;
/// 1次元畳み込みレイヤー
/// 1D convolution layer
pub use conv1d::Conv1d;
/// 2次元畳み込みレイヤー
/// 2D convolution layer
pub use conv2d::Conv2d;
/// 3次元畳み込みレイヤー
/// 3D convolution layer
pub use conv3d::Conv3d;
/// 2次元転置畳み込みレイヤー
/// 2D transposed convolution layer
pub use conv_transpose::ConvTranspose2d;
/// 2次元プーリングレイヤー
/// 2D pooling layers
pub use pool2d::{MaxPool2d, AvgPool2d};
/// 適応的プーリングレイヤー
/// Adaptive pooling layers
pub use adaptive_pool::{AdaptiveMaxPool2d, AdaptiveAvgPool2d};
/// バッチ正規化レイヤー
/// Batch normalization layers
pub use batchnorm::{BatchNorm1d, BatchNorm2d};
/// ドロップアウトレイヤー
/// Dropout layers
pub use dropout::{Dropout, AlphaDropout, dropout};
/// リカレントレイヤー
/// Recurrent layers
pub use rnn::{RNNCell, RNN};
/// LSTMレイヤー
/// LSTM layers
pub use lstm::{LSTMCell, LSTM};
/// GRUレイヤー
/// GRU layers
pub use gru::{GRUCell, GRU};
/// Embedding layers
/// 埋め込みレイヤー
pub use embedding::{Embedding, PositionalEmbedding, SinusoidalPositionalEncoding};
/// Normalization layers
/// 正規化レイヤー
pub use normalization::{LayerNorm, GroupNorm, RMSNorm};
/// Attention layers
/// アテンション層
pub use attention::{MultiHeadAttention, SelfAttention, CrossAttention};
/// Transformer layers
/// Transformer層
pub use transformer::{TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, Transformer};
/// Activation function modules
/// 活性化関数モジュール
pub use activation::{ReLU, Softmax, GELU, Tanh};
/// Quantization modules
/// 量子化モジュール
pub use quantization::{Quantizer, QuantizedTensor, QuantizationParams, QuantizationType, CalibrationMode, QuantizationAwareModule};
/// Pruning modules
/// プルーニングモジュール
pub use pruning::{Pruner, PruningMask, PruningMethod, PruningStructure, PruningSchedule, PruningAwareModule};
/// Loss functions
/// 損失関数
pub use loss::{Loss, MSELoss, CrossEntropyLoss, FocalLoss, TripletLoss, KLDivLoss, 
               mse_loss, cross_entropy_loss, focal_loss, triplet_loss, kl_div_loss};
