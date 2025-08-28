//! Graph layout algorithms
//! グラフレイアウトアルゴリズム

/// グラフレイアウトアルゴリズム
/// Graph layout algorithms for visualization
#[derive(Debug, Clone, PartialEq)]
pub enum GraphLayout {
    /// 階層レイアウト（上から下）
    /// Hierarchical layout (top to bottom)
    Hierarchical,
    /// 円形レイアウト
    /// Circular layout
    Circular,
    /// 力学レイアウト
    /// Force-directed layout
    ForceDirected,
    /// グリッドレイアウト
    /// Grid layout
    Grid,
    /// 左右レイアウト
    /// Left-to-right layout
    LeftToRight,
}