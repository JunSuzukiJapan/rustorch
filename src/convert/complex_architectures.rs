//! Complex architecture support for PyTorch conversion
//! PyTorch変換の複雑アーキテクチャサポート

use crate::formats::pytorch::TensorData;
use crate::convert::SimpleLayerInfo;
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Transformer architecture components
/// Transformerアーキテクチャコンポーネント
#[derive(Debug, Clone)]
pub struct TransformerComponents {
    /// Multi-head attention layers
    /// マルチヘッドアテンションレイヤー
    pub attention_layers: Vec<MultiheadAttentionInfo>,
    /// Feed-forward network layers
    /// フィードフォワードネットワークレイヤー
    pub ffn_layers: Vec<FeedForwardInfo>,
    /// Layer normalization components
    /// レイヤー正規化コンポーネント
    pub layer_norms: Vec<LayerNormInfo>,
    /// Position embeddings
    /// 位置埋め込み
    pub position_embeddings: Option<PositionEmbeddingInfo>,
    /// Token embeddings
    /// トークン埋め込み
    pub token_embeddings: Option<TokenEmbeddingInfo>,
}

/// Multi-head attention layer information (Phase 6 - PyTorch compatible)
/// マルチヘッドアテンションレイヤー情報（フェーズ6 - PyTorch互換）
#[derive(Debug, Clone)]
pub struct MultiheadAttentionInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Number of attention heads
    /// アテンションヘッド数
    pub num_heads: usize,
    /// Embedding dimension
    /// 埋め込み次元
    pub embed_dim: usize,
    /// Dimension of each head
    /// 各ヘッドの次元
    pub head_dim: usize,
    /// Dropout probability
    /// ドロップアウト確率
    pub dropout: f32,
    /// Whether to use bias
    /// バイアス使用フラグ
    pub bias: bool,
    /// Batch first flag
    /// バッチファーストフラグ
    pub batch_first: bool,
    /// Query projection weights
    /// クエリ射影重み
    pub query_weights: Tensor<f32>,
    /// Key projection weights
    /// キー射影重み
    pub key_weights: Tensor<f32>,
    /// Value projection weights
    /// 値射影重み
    pub value_weights: Tensor<f32>,
    /// Output projection weights
    /// 出力射影重み
    pub output_weights: Tensor<f32>,
    /// Bias terms
    /// バイアス項
    pub biases: Option<Vec<Tensor<f32>>>,
}

/// Feed-forward network information
/// フィードフォワードネットワーク情報
#[derive(Debug, Clone)]
pub struct FeedForwardInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Input dimension
    /// 入力次元
    pub input_dim: usize,
    /// Hidden dimension
    /// 隠れ次元
    pub hidden_dim: usize,
    /// First linear layer weights
    /// 第一線形層重み
    pub linear1_weights: Tensor<f32>,
    /// Second linear layer weights
    /// 第二線形層重み
    pub linear2_weights: Tensor<f32>,
    /// Bias terms
    /// バイアス項
    pub biases: Option<Vec<Tensor<f32>>>,
    /// Activation function type
    /// 活性化関数タイプ
    pub activation: String,
}

/// Layer normalization information
/// レイヤー正規化情報
#[derive(Debug, Clone)]
pub struct LayerNormInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Normalized shape
    /// 正規化形状
    pub normalized_shape: Vec<usize>,
    /// Scale parameters (gamma)
    /// スケールパラメータ（ガンマ）
    pub weight: Tensor<f32>,
    /// Shift parameters (beta)
    /// シフトパラメータ（ベータ）
    pub bias: Option<Tensor<f32>>,
    /// Epsilon value
    /// イプシロン値
    pub eps: f32,
}

/// Position embedding information
/// 位置埋め込み情報
#[derive(Debug, Clone)]
pub struct PositionEmbeddingInfo {
    /// Maximum sequence length
    /// 最大シーケンス長
    pub max_length: usize,
    /// Embedding dimension
    /// 埋め込み次元
    pub embed_dim: usize,
    /// Embedding weights
    /// 埋め込み重み
    pub weights: Tensor<f32>,
}

/// Token embedding information
/// トークン埋め込み情報
#[derive(Debug, Clone)]
pub struct TokenEmbeddingInfo {
    /// Vocabulary size
    /// 語彙サイズ
    pub vocab_size: usize,
    /// Embedding dimension
    /// 埋め込み次元
    pub embed_dim: usize,
    /// Embedding weights
    /// 埋め込み重み
    pub weights: Tensor<f32>,
}

/// CNN architecture components
/// CNNアーキテクチャコンポーネント
#[derive(Debug, Clone)]
pub struct CNNComponents {
    /// Convolutional layers
    /// 畳み込みレイヤー
    pub conv_layers: Vec<ConvLayerInfo>,
    /// Pooling layers
    /// プーリングレイヤー
    pub pool_layers: Vec<PoolLayerInfo>,
    /// Batch normalization layers
    /// バッチ正規化レイヤー
    pub batch_norms: Vec<BatchNormInfo>,
    /// Classification head
    /// 分類ヘッド
    pub classifier: Option<ClassifierInfo>,
}

/// Convolutional layer information
/// 畳み込みレイヤー情報
#[derive(Debug, Clone)]
pub struct ConvLayerInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Input channels
    /// 入力チャンネル
    pub in_channels: usize,
    /// Output channels
    /// 出力チャンネル
    pub out_channels: usize,
    /// Kernel size
    /// カーネルサイズ
    pub kernel_size: (usize, usize),
    /// Stride
    /// ストライド
    pub stride: (usize, usize),
    /// Padding
    /// パディング
    pub padding: (usize, usize),
    /// Dilation
    /// 膨張
    pub dilation: (usize, usize),
    /// Groups
    /// グループ
    pub groups: usize,
    /// Weights
    /// 重み
    pub weights: Tensor<f32>,
    /// Bias
    /// バイアス
    pub bias: Option<Tensor<f32>>,
}

/// Pooling layer information
/// プーリングレイヤー情報
#[derive(Debug, Clone)]
pub struct PoolLayerInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Pooling type (Max, Average, Adaptive)
    /// プーリングタイプ（最大、平均、適応）
    pub pool_type: PoolType,
    /// Kernel size
    /// カーネルサイズ
    pub kernel_size: (usize, usize),
    /// Stride
    /// ストライド
    pub stride: Option<(usize, usize)>,
    /// Padding
    /// パディング
    pub padding: Option<(usize, usize)>,
}

/// Pooling types
/// プーリングタイプ
#[derive(Debug, Clone)]
pub enum PoolType {
    /// Max pooling
    /// 最大プーリング
    MaxPool,
    /// Average pooling
    /// 平均プーリング
    AvgPool,
    /// Adaptive max pooling
    /// 適応最大プーリング
    AdaptiveMaxPool(Vec<usize>),
    /// Adaptive average pooling
    /// 適応平均プーリング
    AdaptiveAvgPool(Vec<usize>),
}

/// Batch normalization information
/// バッチ正規化情報
#[derive(Debug, Clone)]
pub struct BatchNormInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Number of features
    /// 特徴数
    pub num_features: usize,
    /// Scale parameters
    /// スケールパラメータ
    pub weight: Tensor<f32>,
    /// Shift parameters
    /// シフトパラメータ
    pub bias: Tensor<f32>,
    /// Running mean
    /// 実行中平均
    pub running_mean: Tensor<f32>,
    /// Running variance
    /// 実行中分散
    pub running_var: Tensor<f32>,
    /// Momentum for running statistics
    /// 実行統計のモメンタム
    pub momentum: f32,
    /// Epsilon for numerical stability
    /// 数値安定性のためのイプシロン
    pub eps: f32,
}

/// Classifier information
/// 分類器情報
#[derive(Debug, Clone)]
pub struct ClassifierInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Input features
    /// 入力特徴
    pub in_features: usize,
    /// Number of classes
    /// クラス数
    pub num_classes: usize,
    /// Weights
    /// 重み
    pub weights: Tensor<f32>,
    /// Bias
    /// バイアス
    pub bias: Option<Tensor<f32>>,
}

/// Complex architecture parser
/// 複雑アーキテクチャパーサー
pub struct ComplexArchitectureParser;

impl ComplexArchitectureParser {
    /// Parse transformer architecture from layer parameters
    /// レイヤーパラメータからtransformerアーキテクチャを解析
    pub fn parse_transformer(
        layers: &HashMap<String, SimpleLayerInfo>
    ) -> Result<TransformerComponents, RusTorchError> {
        let mut attention_layers = Vec::new();
        let mut ffn_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let mut position_embeddings = None;
        let mut token_embeddings = None;

        for (layer_name, layer_info) in layers {
            if Self::is_attention_layer(layer_name) {
                if let Ok(attention) = Self::parse_attention_layer(layer_name, layer_info) {
                    attention_layers.push(attention);
                }
            } else if Self::is_ffn_layer(layer_name) {
                if let Ok(ffn) = Self::parse_ffn_layer(layer_name, layer_info) {
                    ffn_layers.push(ffn);
                }
            } else if Self::is_layer_norm(layer_name) {
                if let Ok(ln) = Self::parse_layer_norm(layer_name, layer_info) {
                    layer_norms.push(ln);
                }
            } else if Self::is_position_embedding(layer_name) {
                position_embeddings = Some(Self::parse_position_embedding(layer_name, layer_info)?);
            } else if Self::is_token_embedding(layer_name) {
                token_embeddings = Some(Self::parse_token_embedding(layer_name, layer_info)?);
            }
        }

        Ok(TransformerComponents {
            attention_layers,
            ffn_layers,
            layer_norms,
            position_embeddings,
            token_embeddings,
        })
    }

    /// Parse CNN architecture from layer parameters
    /// レイヤーパラメータからCNNアーキテクチャを解析
    pub fn parse_cnn(
        layers: &HashMap<String, SimpleLayerInfo>
    ) -> Result<CNNComponents, RusTorchError> {
        let mut conv_layers = Vec::new();
        let mut pool_layers = Vec::new();
        let mut batch_norms = Vec::new();
        let mut classifier = None;

        for (layer_name, layer_info) in layers {
            if Self::is_conv_layer(layer_name, layer_info) {
                if let Ok(conv) = Self::parse_conv_layer(layer_name, layer_info) {
                    conv_layers.push(conv);
                }
            } else if Self::is_pool_layer(layer_name) {
                if let Ok(pool) = Self::parse_pool_layer(layer_name, layer_info) {
                    pool_layers.push(pool);
                }
            } else if Self::is_batch_norm(layer_name, layer_info) {
                if let Ok(bn) = Self::parse_batch_norm(layer_name, layer_info) {
                    batch_norms.push(bn);
                }
            } else if Self::is_classifier(layer_name, layer_info) {
                classifier = Some(Self::parse_classifier(layer_name, layer_info)?);
            }
        }

        Ok(CNNComponents {
            conv_layers,
            pool_layers,
            batch_norms,
            classifier,
        })
    }

    /// Check if layer is attention layer
    /// レイヤーがアテンションレイヤーかチェック
    fn is_attention_layer(layer_name: &str) -> bool {
        layer_name.contains("attention") || 
        layer_name.contains("attn") ||
        layer_name.contains("self_attn") ||
        layer_name.contains("multi_head")
    }

    /// Parse attention layer
    /// アテンションレイヤーを解析
    fn parse_attention_layer(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<MultiheadAttentionInfo, RusTorchError> {
        // Extract attention parameters from tensors
        let q_proj = layer_info.tensors.get("q_proj.weight")
            .or_else(|| layer_info.tensors.get("query.weight"))
            .or_else(|| layer_info.tensors.get("wq.weight"))
            .ok_or_else(|| RusTorchError::import_error("query projection".to_string()))?;

        let k_proj = layer_info.tensors.get("k_proj.weight")
            .or_else(|| layer_info.tensors.get("key.weight"))
            .or_else(|| layer_info.tensors.get("wk.weight"))
            .ok_or_else(|| RusTorchError::import_error("key projection".to_string()))?;

        let v_proj = layer_info.tensors.get("v_proj.weight")
            .or_else(|| layer_info.tensors.get("value.weight"))
            .or_else(|| layer_info.tensors.get("wv.weight"))
            .ok_or_else(|| RusTorchError::import_error("value projection".to_string()))?;

        let out_proj = layer_info.tensors.get("out_proj.weight")
            .or_else(|| layer_info.tensors.get("output.weight"))
            .or_else(|| layer_info.tensors.get("wo.weight"))
            .ok_or_else(|| RusTorchError::import_error("output projection".to_string()))?;

        // Infer dimensions from weight shapes
        let embed_dim = q_proj.shape()[1];
        let total_head_dim = q_proj.shape()[0];
        
        // Common head configurations
        let num_heads = Self::infer_num_heads(embed_dim, total_head_dim);
        let head_dim = total_head_dim / num_heads;

        Ok(MultiheadAttentionInfo {
            name: layer_name.to_string(),
            num_heads,
            embed_dim,
            head_dim,
            dropout: 0.1, // Default dropout
            bias: true,   // Default bias
            batch_first: true, // Default batch_first
            query_weights: q_proj.clone(),
            key_weights: k_proj.clone(),
            value_weights: v_proj.clone(),
            output_weights: out_proj.clone(),
            biases: None, // Could be extracted if present
        })
    }

    /// Infer number of attention heads
    /// アテンションヘッド数を推論
    fn infer_num_heads(embed_dim: usize, total_head_dim: usize) -> usize {
        // Common head configurations
        if total_head_dim % embed_dim == 0 {
            total_head_dim / embed_dim
        } else if embed_dim % 64 == 0 && total_head_dim % 64 == 0 {
            embed_dim / 64 // Assume 64-dim heads
        } else if embed_dim % 32 == 0 && total_head_dim % 32 == 0 {
            embed_dim / 32 // Assume 32-dim heads
        } else {
            8 // Default fallback
        }
    }

    /// Check if layer is feed-forward network
    /// レイヤーがフィードフォワードネットワークかチェック
    fn is_ffn_layer(layer_name: &str) -> bool {
        layer_name.contains("ffn") ||
        layer_name.contains("feed_forward") ||
        layer_name.contains("mlp") ||
        (layer_name.contains("linear") && (layer_name.contains("1") || layer_name.contains("2")))
    }

    /// Parse feed-forward network layer
    /// フィードフォワードネットワークレイヤーを解析
    fn parse_ffn_layer(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<FeedForwardInfo, RusTorchError> {
        // Look for linear layer pairs
        let linear1 = layer_info.tensors.get("linear1.weight")
            .or_else(|| layer_info.tensors.get("fc1.weight"))
            .or_else(|| layer_info.tensors.get("w1.weight"))
            .ok_or_else(|| RusTorchError::import_error("first linear layer".to_string()))?;

        let linear2 = layer_info.tensors.get("linear2.weight")
            .or_else(|| layer_info.tensors.get("fc2.weight"))
            .or_else(|| layer_info.tensors.get("w2.weight"))
            .ok_or_else(|| RusTorchError::import_error("second linear layer".to_string()))?;

        let input_dim = linear1.shape()[1];
        let hidden_dim = linear1.shape()[0];

        Ok(FeedForwardInfo {
            name: layer_name.to_string(),
            input_dim,
            hidden_dim,
            linear1_weights: linear1.clone(),
            linear2_weights: linear2.clone(),
            biases: None,
            activation: "relu".to_string(), // Default, could be inferred
        })
    }

    /// Check if layer is layer normalization
    /// レイヤーがレイヤー正規化かチェック
    fn is_layer_norm(layer_name: &str) -> bool {
        layer_name.contains("layer_norm") ||
        layer_name.contains("ln") ||
        layer_name.contains("norm")
    }

    /// Parse layer normalization
    /// レイヤー正規化を解析
    fn parse_layer_norm(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<LayerNormInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("layer norm weight".to_string()))?;

        let bias = layer_info.tensors.get("bias");
        let normalized_shape = weight.shape().to_vec();

        Ok(LayerNormInfo {
            name: layer_name.to_string(),
            normalized_shape,
            weight: weight.clone(),
            bias: bias.cloned(),
            eps: 1e-5, // Default epsilon
        })
    }

    /// Check if layer is position embedding
    /// レイヤーが位置埋め込みかチェック
    fn is_position_embedding(layer_name: &str) -> bool {
        layer_name.contains("pos_emb") ||
        layer_name.contains("position") ||
        layer_name.contains("positional")
    }

    /// Parse position embedding
    /// 位置埋め込みを解析
    fn parse_position_embedding(
        _layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<PositionEmbeddingInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("position embedding weight".to_string()))?;

        let shape = weight.shape();
        let max_length = shape[0];
        let embed_dim = shape[1];

        Ok(PositionEmbeddingInfo {
            max_length,
            embed_dim,
            weights: weight.clone(),
        })
    }

    /// Check if layer is token embedding
    /// レイヤーがトークン埋め込みかチェック
    fn is_token_embedding(layer_name: &str) -> bool {
        layer_name.contains("token_emb") ||
        layer_name.contains("word_emb") ||
        layer_name.contains("embedding") ||
        layer_name == "embeddings"
    }

    /// Parse token embedding
    /// トークン埋め込みを解析
    fn parse_token_embedding(
        _layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<TokenEmbeddingInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("token embedding weight".to_string()))?;

        let shape = weight.shape();
        let vocab_size = shape[0];
        let embed_dim = shape[1];

        Ok(TokenEmbeddingInfo {
            vocab_size,
            embed_dim,
            weights: weight.clone(),
        })
    }

    /// Check if layer is convolutional
    /// レイヤーが畳み込みかチェック
    fn is_conv_layer(layer_name: &str, layer_info: &SimpleLayerInfo) -> bool {
        layer_name.contains("conv") && layer_info.layer_type == "Conv2d"
    }

    /// Parse convolutional layer
    /// 畳み込みレイヤーを解析
    fn parse_conv_layer(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<ConvLayerInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("conv weight".to_string()))?;

        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_h = shape[2];
        let kernel_w = shape[3];

        let bias = layer_info.tensors.get("bias");

        Ok(ConvLayerInfo {
            name: layer_name.to_string(),
            in_channels,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
            stride: (1, 1), // Default - could be inferred from layer name
            padding: (0, 0), // Default
            dilation: (1, 1), // Default
            groups: 1, // Default
            weights: weight.clone(),
            bias: bias.cloned(),
        })
    }

    /// Check if layer is pooling
    /// レイヤーがプーリングかチェック
    fn is_pool_layer(layer_name: &str) -> bool {
        layer_name.contains("pool") || layer_name.contains("avgpool") || layer_name.contains("maxpool")
    }

    /// Parse pooling layer
    /// プーリングレイヤーを解析
    fn parse_pool_layer(
        layer_name: &str,
        _layer_info: &SimpleLayerInfo
    ) -> Result<PoolLayerInfo, RusTorchError> {
        let pool_type = if layer_name.contains("max") {
            PoolType::MaxPool
        } else if layer_name.contains("avg") {
            PoolType::AvgPool
        } else if layer_name.contains("adaptive") {
            if layer_name.contains("max") {
                PoolType::AdaptiveMaxPool(vec![1, 1]) // Default output size
            } else {
                PoolType::AdaptiveAvgPool(vec![1, 1]) // Default output size
            }
        } else {
            PoolType::MaxPool // Default
        };

        Ok(PoolLayerInfo {
            name: layer_name.to_string(),
            pool_type,
            kernel_size: (2, 2), // Default
            stride: Some((2, 2)), // Default
            padding: Some((0, 0)), // Default
        })
    }

    /// Check if layer is batch normalization
    /// レイヤーがバッチ正規化かチェック
    fn is_batch_norm(layer_name: &str, layer_info: &SimpleLayerInfo) -> bool {
        (layer_name.contains("bn") || layer_name.contains("batch_norm")) && 
        layer_info.layer_type == "BatchNorm2d"
    }

    /// Parse batch normalization layer
    /// バッチ正規化レイヤーを解析
    fn parse_batch_norm(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<BatchNormInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("batch norm weight".to_string()))?;

        let bias = layer_info.tensors.get("bias")
            .ok_or_else(|| RusTorchError::import_error("batch norm bias".to_string()))?;

        let running_mean = layer_info.tensors.get("running_mean")
            .ok_or_else(|| RusTorchError::import_error("batch norm running mean".to_string()))?;

        let running_var = layer_info.tensors.get("running_var")
            .ok_or_else(|| RusTorchError::import_error("batch norm running var".to_string()))?;

        let num_features = weight.shape()[0];

        Ok(BatchNormInfo {
            name: layer_name.to_string(),
            num_features,
            weight: weight.clone(),
            bias: bias.clone(),
            running_mean: running_mean.clone(),
            running_var: running_var.clone(),
            momentum: 0.1, // Default
            eps: 1e-5, // Default
        })
    }

    /// Check if layer is classifier
    /// レイヤーが分類器かチェック
    fn is_classifier(layer_name: &str, layer_info: &SimpleLayerInfo) -> bool {
        (layer_name.contains("classifier") || layer_name.contains("fc") || layer_name == "head") &&
        layer_info.layer_type == "Linear"
    }

    /// Parse classifier layer
    /// 分類器レイヤーを解析
    fn parse_classifier(
        layer_name: &str,
        layer_info: &SimpleLayerInfo
    ) -> Result<ClassifierInfo, RusTorchError> {
        let weight = layer_info.tensors.get("weight")
            .ok_or_else(|| RusTorchError::import_error("classifier weight".to_string()))?;

        let shape = weight.shape();
        let num_classes = shape[0];
        let in_features = shape[1];
        let bias = layer_info.tensors.get("bias");

        Ok(ClassifierInfo {
            name: layer_name.to_string(),
            in_features,
            num_classes,
            weights: weight.clone(),
            bias: bias.cloned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_layer_detection() {
        assert!(ComplexArchitectureParser::is_attention_layer("encoder.layer.0.attention"));
        assert!(ComplexArchitectureParser::is_attention_layer("transformer.h.0.attn"));
        assert!(ComplexArchitectureParser::is_attention_layer("layers.0.self_attn"));
        assert!(!ComplexArchitectureParser::is_attention_layer("layers.0.linear"));
    }

    #[test]
    fn test_ffn_layer_detection() {
        assert!(ComplexArchitectureParser::is_ffn_layer("encoder.layer.0.ffn"));
        assert!(ComplexArchitectureParser::is_ffn_layer("transformer.h.0.mlp"));
        assert!(ComplexArchitectureParser::is_ffn_layer("layers.0.feed_forward"));
        assert!(!ComplexArchitectureParser::is_ffn_layer("layers.0.attention"));
    }

    #[test]
    fn test_conv_layer_detection() {
        let mut layer_info = SimpleLayerInfo {
            name: "features.0".to_string(),
            layer_type: "Conv2d".to_string(),
            parameter_shapes: HashMap::new(),
            num_parameters: 0,
            tensors: HashMap::new(),
        };
        
        assert!(ComplexArchitectureParser::is_conv_layer("features.0.conv", &layer_info));
        
        layer_info.layer_type = "Linear".to_string();
        assert!(!ComplexArchitectureParser::is_conv_layer("features.0.conv", &layer_info));
    }

    #[test]
    fn test_num_heads_inference() {
        assert_eq!(ComplexArchitectureParser::infer_num_heads(512, 512), 8); // 512/64 = 8
        assert_eq!(ComplexArchitectureParser::infer_num_heads(768, 768), 12); // 768/64 = 12
        assert_eq!(ComplexArchitectureParser::infer_num_heads(1024, 1024), 16); // 1024/64 = 16
    }
}