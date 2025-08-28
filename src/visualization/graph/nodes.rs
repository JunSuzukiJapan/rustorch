//! Graph node definitions and types
//! グラフノード定義と種別

/// Graph node representation
/// グラフノード表現
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node ID
    /// ノードID
    pub id: String,
    /// Node name
    /// ノード名
    pub name: String,
    /// Node type
    /// ノードタイプ
    pub node_type: NodeType,
    /// Shape information
    /// 形状情報
    pub shape: Vec<usize>,
    /// Position coordinates
    /// 位置座標
    pub position: (f32, f32),
    /// Style
    /// スタイル
    pub style: crate::visualization::graph::styles::NodeStyle,
}

/// ノードタイプ
/// Node types for computation graph
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// 入力ノード
    /// Input node
    Input,
    /// 演算ノード
    /// Operation node
    Operation(String),
    /// パラメータノード
    /// Parameter node
    Parameter,
    /// 出力ノード
    /// Output node
    Output,
    /// 損失ノード
    /// Loss node
    Loss,
    /// 勾配ノード
    /// Gradient node
    Gradient,
}

/// ノード形状
/// Node shapes for visualization
#[derive(Debug, Clone, PartialEq)]
pub enum NodeShape {
    /// 円形
    /// Circle
    Circle,
    /// 長方形
    /// Rectangle
    Rectangle,
    /// 楕円形
    /// Ellipse
    Ellipse,
    /// ひし形
    /// Diamond
    Diamond,
    /// 六角形
    /// Hexagon
    Hexagon,
}