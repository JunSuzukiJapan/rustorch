//! Core data types for model parsing
//! モデル解析のためのコアデータタイプ

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Layer information extracted from PyTorch model
/// PyTorchモデルから抽出されたレイヤー情報
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Layer type (Linear, Conv2d, etc.)
    /// レイヤータイプ（Linear、Conv2dなど）
    pub layer_type: LayerType,
    /// Input shape
    /// 入力形状
    pub input_shape: Option<Vec<usize>>,
    /// Output shape
    /// 出力形状
    pub output_shape: Option<Vec<usize>>,
    /// Layer parameters
    /// レイヤーパラメータ
    pub parameters: HashMap<String, Vec<usize>>, // parameter name -> shape
    /// Number of parameters
    /// パラメータ数
    pub num_parameters: usize,
}

/// Supported layer types
/// サポートされているレイヤータイプ
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    /// Linear/Dense layer
    /// Linear/Denseレイヤー
    Linear {
        /// Input features
        in_features: usize,
        /// Output features
        out_features: usize,
    },
    /// 1D Convolution layer
    /// 1D畳み込みレイヤー
    Conv1d {
        /// Input channels
        in_channels: usize,
        /// Output channels
        out_channels: usize,
        /// Kernel size
        kernel_size: usize,
        /// Stride
        stride: usize,
        /// Padding
        padding: usize,
    },
    /// 2D Convolution layer
    /// 2D畳み込みレイヤー
    Conv2d {
        /// Input channels
        in_channels: usize,
        /// Output channels
        out_channels: usize,
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
        /// Padding
        padding: (usize, usize),
    },
    /// 3D Convolution layer
    /// 3D畳み込みレイヤー
    Conv3d {
        /// Input channels
        in_channels: usize,
        /// Output channels
        out_channels: usize,
        /// Kernel size
        kernel_size: (usize, usize, usize),
        /// Stride
        stride: (usize, usize, usize),
        /// Padding
        padding: (usize, usize, usize),
    },
    /// 2D Batch Normalization
    /// 2Dバッチ正規化
    BatchNorm2d {
        /// Number of features
        num_features: usize,
    },
    /// ReLU activation
    /// ReLU活性化
    ReLU,
    /// Dropout layer
    /// Dropoutレイヤー
    Dropout {
        /// Dropout probability
        p: f64,
    },
    /// 2D Max Pooling
    /// 2D最大プーリング
    MaxPool2d {
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
    },
    /// 2D Average Pooling
    /// 2D平均プーリング
    AvgPool2d {
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
    },
    /// Flatten layer
    /// Flattenレイヤー
    Flatten,
    /// Unknown layer type
    /// 不明なレイヤータイプ
    Unknown(String),
}

/// Model architecture graph
/// モデルアーキテクチャグラフ
#[derive(Debug)]
pub struct ModelGraph {
    /// Layers in the model
    /// モデル内のレイヤー
    pub layers: HashMap<String, LayerInfo>,
    /// Layer execution order
    /// レイヤー実行順序
    pub execution_order: Vec<String>,
    /// Layer connections (from -> to)
    /// レイヤー接続（from -> to）
    pub connections: HashMap<String, Vec<String>>,
    /// Model input layers
    /// モデル入力レイヤー
    pub input_layers: Vec<String>,
    /// Model output layers
    /// モデル出力レイヤー
    pub output_layers: Vec<String>,
}