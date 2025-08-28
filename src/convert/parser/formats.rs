//! Architecture description formats and serialization
//! アーキテクチャ記述形式とシリアライゼーション

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Architecture description format for explicit model definition
/// 明示的なモデル定義のためのアーキテクチャ記述形式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDescription {
    /// Model metadata
    /// モデルメタデータ
    pub metadata: ModelMetadata,
    /// Layer definitions
    /// レイヤー定義
    pub layers: Vec<LayerDefinition>,
    /// Layer connections
    /// レイヤー接続
    pub connections: Vec<ConnectionDefinition>,
}

/// Model metadata information
/// モデルメタデータ情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    /// モデル名
    pub name: String,
    /// Model version
    /// モデルバージョン
    pub version: Option<String>,
    /// Framework (pytorch, tensorflow, etc.)
    /// フレームワーク
    pub framework: Option<String>,
    /// Description
    /// 説明
    pub description: Option<String>,
}

/// Layer definition in architecture description
/// アーキテクチャ記述でのレイヤー定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    /// Layer name/id
    /// レイヤー名/ID
    pub name: String,
    /// Layer type specification
    /// レイヤータイプ仕様
    #[serde(rename = "type")]
    pub layer_type: String,
    /// Layer parameters
    /// レイヤーパラメータ
    pub params: Option<HashMap<String, serde_json::Value>>,
    /// Input shape hint
    /// 入力形状ヒント
    pub input_shape: Option<Vec<usize>>,
    /// Output shape hint
    /// 出力形状ヒント
    pub output_shape: Option<Vec<usize>>,
}

/// Connection definition between layers
/// レイヤー間の接続定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionDefinition {
    /// Source layer name
    /// ソースレイヤー名
    pub from: String,
    /// Target layer name
    /// ターゲットレイヤー名
    pub to: String,
    /// Connection type (optional)
    /// 接続タイプ（オプション）
    #[serde(rename = "type")]
    pub connection_type: Option<String>,
}