//! Graph styling definitions
//! グラフスタイリング定義

/// ノードスタイル
/// Node styling options
#[derive(Debug, Clone)]
pub struct NodeStyle {
    /// 背景色 (RGB)
    /// Background color (RGB)
    pub background_color: (u8, u8, u8),
    /// 境界線色 (RGB)
    /// Border color (RGB)
    pub border_color: (u8, u8, u8),
    /// フォント色 (RGB)
    /// Font color (RGB)
    pub font_color: (u8, u8, u8),
    /// 境界線の幅
    /// Border width
    pub border_width: f32,
    /// フォントサイズ
    /// Font size
    pub font_size: f32,
    /// ノード形状
    /// Node shape
    pub shape: crate::visualization::graph::nodes::NodeShape,
}

/// エッジスタイル
/// Edge styling options
#[derive(Debug, Clone)]
pub struct EdgeStyle {
    /// 線の色 (RGB)
    /// Line color (RGB)
    pub color: (u8, u8, u8),
    /// 線の太さ
    /// Line thickness
    pub thickness: f32,
    /// 線のタイプ
    /// Line type
    pub line_type: LineType,
    /// 矢印タイプ
    /// Arrow type
    pub arrow_type: crate::visualization::graph::edges::ArrowType,
}

/// 線タイプ
/// Line types for graph edges
#[derive(Debug, Clone, PartialEq)]
pub enum LineType {
    /// 実線
    /// Solid line
    Solid,
    /// 点線
    /// Dotted line
    Dotted,
    /// 破線
    /// Dashed line
    Dashed,
    /// 一点鎖線
    /// Dash-dot line
    DashDot,
}