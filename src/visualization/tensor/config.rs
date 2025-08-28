//! Tensor plot configuration
//! テンソルプロット設定

use super::ColorMap;

/// Tensor plot configuration
/// テンソルプロット設定
#[derive(Debug, Clone)]
pub struct TensorPlotConfig {
    /// Color map
    /// カラーマップ
    pub colormap: ColorMap,
    /// Normalize values
    /// 正規化
    pub normalize: bool,
    /// Aspect ratio
    /// アスペクト比
    pub aspect: String,
    /// Title
    /// タイトル
    pub title: Option<String>,
    /// Show colorbar
    /// カラーバーを表示
    pub show_colorbar: bool,
    /// Show axes
    /// 軸を表示
    pub show_axes: bool,
    /// Figure size (width, height)
    /// 図のサイズ（幅、高さ）
    pub figsize: (f32, f32),
    /// DPI for high resolution
    /// 高解像度用DPI
    pub dpi: u32,
}

impl Default for TensorPlotConfig {
    fn default() -> Self {
        Self {
            colormap: ColorMap::default(),
            normalize: true,
            aspect: "equal".to_string(),
            title: None,
            show_colorbar: true,
            show_axes: true,
            figsize: (8.0, 6.0),
            dpi: 100,
        }
    }
}
