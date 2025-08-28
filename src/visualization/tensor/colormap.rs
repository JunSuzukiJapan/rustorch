//! Color map definitions for tensor visualization
//! テンソル可視化用カラーマップ定義

/// Color map types
/// カラーマップ種別
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMap {
    /// グレースケール
    /// Grayscale color map
    Grayscale,
    /// ジェット
    /// Jet color map
    Jet,
    /// ビリディス
    /// Viridis color map
    Viridis,
    /// プラズマ
    /// Plasma color map
    Plasma,
    /// インフェルノ
    /// Inferno color map
    Inferno,
    /// クールウォーム
    /// Cool-warm color map
    Coolwarm,
    /// カスタム
    /// Custom color map
    Custom(Vec<String>),
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

impl ColorMap {
    /// Get color for value in range [0, 1]
    /// [0, 1]範囲の値に対する色を取得
    pub fn get_color(&self, value: f64) -> String {
        match self {
            ColorMap::Grayscale => {
                let intensity = (value * 255.0) as u8;
                format!("rgb({},{},{})", intensity, intensity, intensity)
            }
            ColorMap::Jet => {
                // Jet colormap implementation
                if value < 0.25 {
                    let r = 0;
                    let g = (4.0 * value * 255.0) as u8;
                    let b = 255;
                    format!("rgb({},{},{})", r, g, b)
                } else if value < 0.5 {
                    let r = 0;
                    let g = 255;
                    let b = 255 - ((4.0 * (value - 0.25) * 255.0) as u8);
                    format!("rgb({},{},{})", r, g, b)
                } else if value < 0.75 {
                    let r = (4.0 * (value - 0.5) * 255.0) as u8;
                    let g = 255;
                    let b = 0;
                    format!("rgb({},{},{})", r, g, b)
                } else {
                    let r = 255;
                    let g = 255 - ((4.0 * (value - 0.75) * 255.0) as u8);
                    let b = 0;
                    format!("rgb({},{},{})", r, g, b)
                }
            }
            ColorMap::Viridis => {
                // Viridis colormap approximation
                let r = (255.0 * (0.267004 + value * (0.993248 - 0.267004))) as u8;
                let g = (255.0 * (0.004874 + value * (0.906157 - 0.004874))) as u8;
                let b = (255.0 * (0.329415 + value * (0.143936 - 0.329415))) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorMap::Plasma => {
                // Plasma colormap approximation
                let r = (255.0 * (0.050383 + value * (0.940015 - 0.050383))) as u8;
                let g = (255.0 * (0.029803 + value * (0.975158 - 0.029803))) as u8;
                let b = (255.0 * (0.527975 + value * (0.131326 - 0.527975))) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorMap::Inferno => {
                // Inferno colormap approximation
                let r = (255.0 * (0.001462 + value * (0.988362 - 0.001462))) as u8;
                let g = (255.0 * (0.000466 + value * (0.809003 - 0.000466))) as u8;
                let b = (255.0 * (0.013866 + value * (0.145728 - 0.013866))) as u8;
                format!("rgb({},{},{})", r, g, b)
            }
            ColorMap::Coolwarm => {
                // Cool-warm colormap
                if value < 0.5 {
                    let t = value * 2.0;
                    let r = (255.0 * (0.230 + t * (1.0 - 0.230))) as u8;
                    let g = (255.0 * (0.299 + t * (1.0 - 0.299))) as u8;
                    let b = 255;
                    format!("rgb({},{},{})", r, g, b)
                } else {
                    let t = (value - 0.5) * 2.0;
                    let r = 255;
                    let g = (255.0 * (1.0 - t * (1.0 - 0.299))) as u8;
                    let b = (255.0 * (1.0 - t * (1.0 - 0.230))) as u8;
                    format!("rgb({},{},{})", r, g, b)
                }
            }
            ColorMap::Custom(colors) => {
                if colors.is_empty() {
                    return "rgb(0,0,0)".to_string();
                }
                let index = (value * (colors.len() - 1) as f64) as usize;
                colors[index.min(colors.len() - 1)].clone()
            }
        }
    }
}