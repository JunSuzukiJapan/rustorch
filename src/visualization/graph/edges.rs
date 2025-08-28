//! Graph edge definitions and styles
//! グラフエッジ定義とスタイル

/// Graph edge representation
/// グラフエッジ表現
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID
    /// 開始ノードID
    pub from: String,
    /// Target node ID
    /// 終了ノードID
    pub to: String,
    /// Edge label
    /// エッジラベル
    pub label: Option<String>,
    /// Edge style
    /// エッジスタイル
    pub style: crate::visualization::graph::styles::EdgeStyle,
}

/// 矢印タイプ
/// Arrow types for graph edges
#[derive(Debug, Clone, PartialEq)]
pub enum ArrowType {
    /// 矢印なし
    /// No arrow
    None,
    /// 通常の矢印
    /// Normal arrow
    Normal,
    /// 太い矢印
    /// Bold arrow
    Bold,
    /// 点線の矢印
    /// Dotted arrow
    Dotted,
    /// 双方向矢印
    /// Bidirectional arrow
    Bidirectional,
}