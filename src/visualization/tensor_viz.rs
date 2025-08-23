//! テンソル可視化機能
//! Tensor visualization functionality

use super::{PlotData, VisualizationResult, VisualizationError, Visualizable};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use num_traits::Float;
use std::collections::HashMap;

/// カラーマップ
/// Color map types
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMap {
    /// グレースケール
    Grayscale,
    /// ジェット
    Jet,
    /// ビリディス
    Viridis,
    /// プラズマ
    Plasma,
    /// インフェルノ
    Inferno,
    /// クールウォーム
    Coolwarm,
    /// カスタム
    Custom(Vec<String>),
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

/// テンソルプロット設定
/// Tensor plot configuration
#[derive(Debug, Clone)]
pub struct TensorPlotConfig {
    /// カラーマップ
    /// Color map
    pub colormap: ColorMap,
    /// 正規化
    /// Normalize values
    pub normalize: bool,
    /// アスペクト比
    /// Aspect ratio
    pub aspect: String,
    /// タイトル
    /// Title
    pub title: Option<String>,
    /// カラーバー表示
    /// Show colorbar
    pub show_colorbar: bool,
    /// 値の表示
    /// Show values
    pub show_values: bool,
    /// 小数点精度
    /// Decimal precision
    pub precision: usize,
}

impl Default for TensorPlotConfig {
    fn default() -> Self {
        Self {
            colormap: ColorMap::default(),
            normalize: true,
            aspect: "auto".to_string(),
            title: None,
            show_colorbar: true,
            show_values: false,
            precision: 2,
        }
    }
}

/// テンソル可視化クラス
/// Tensor visualizer
#[derive(Debug)]
pub struct TensorVisualizer {
    /// 設定
    /// Configuration
    pub config: TensorPlotConfig,
}

impl TensorVisualizer {
    /// 新しいビジュアライザーを作成
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            config: TensorPlotConfig::default(),
        }
    }
    
    /// 設定付きビジュアライザーを作成
    /// Create visualizer with configuration
    pub fn with_config(config: TensorPlotConfig) -> Self {
        Self { config }
    }
    
    /// 2Dテンソルをヒートマップとして可視化
    /// Visualize 2D tensor as heatmap
    pub fn plot_heatmap<T>(&self, tensor: &Tensor<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug + 'static,
    {
        let shape = tensor.shape();
        
        if shape.len() != 2 {
            return Err(VisualizationError::InvalidDataFormat(
                format!("Expected 2D tensor, got {}D tensor", shape.len())
            ));
        }
        
        let height = shape[0];
        let width = shape[1];
        let data = tensor.as_slice().ok_or_else(|| {
            VisualizationError::InvalidDataFormat("Tensor data not accessible as slice".to_string())
        })?;
        
        let mut svg = self.generate_heatmap_header(width, height);
        
        // データの正規化
        let normalized_data = if self.config.normalize {
            self.normalize_data(&data)?
        } else {
            data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect()
        };
        
        // ヒートマップの描画
        svg.push_str(&self.render_heatmap_cells(&normalized_data, width, height)?);
        
        // カラーバーの追加
        if self.config.show_colorbar {
            svg.push_str(&self.render_colorbar()?);
        }
        
        // タイトルの追加
        if let Some(ref title) = self.config.title {
            svg.push_str(&self.render_title(title, width)?);
        }
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    /// 1Dテンソルを棒グラフとして可視化
    /// Visualize 1D tensor as bar chart
    pub fn plot_bar_chart<T>(&self, tensor: &Tensor<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug + 'static,
    {
        let shape = tensor.shape();
        
        if shape.len() != 1 {
            return Err(VisualizationError::InvalidDataFormat(
                format!("Expected 1D tensor, got {}D tensor", shape.len())
            ));
        }
        
        let length = shape[0];
        let data = tensor.as_slice().ok_or_else(|| {
            VisualizationError::InvalidDataFormat("Tensor data not accessible as slice".to_string())
        })?;
        
        let mut svg = self.generate_bar_chart_header(length);
        
        // 棒グラフの描画
        svg.push_str(&self.render_bar_chart(&data, length)?);
        
        // タイトルの追加
        if let Some(ref title) = self.config.title {
            svg.push_str(&self.render_title(title, length)?);
        }
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    /// 3Dテンソルを複数のヒートマップとして可視化
    /// Visualize 3D tensor as multiple heatmaps
    pub fn plot_3d_slices<T>(&self, tensor: &Tensor<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug + 'static,
    {
        let shape = tensor.shape();
        
        if shape.len() != 3 {
            return Err(VisualizationError::InvalidDataFormat(
                format!("Expected 3D tensor, got {}D tensor", shape.len())
            ));
        }
        
        let depth = shape[0];
        let height = shape[1];
        let width = shape[2];
        let data = tensor.as_slice().ok_or_else(|| {
            VisualizationError::InvalidDataFormat("Tensor data not accessible as slice".to_string())
        })?;
        
        let mut svg = self.generate_3d_slices_header(depth, width, height);
        
        // 各スライスを描画
        for d in 0..depth {
            let slice_data = self.extract_slice(&data, d, height, width)?;
            let normalized_slice = if self.config.normalize {
                self.normalize_data(&slice_data)?
            } else {
                slice_data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect()
            };
            
            svg.push_str(&self.render_slice_heatmap(&normalized_slice, d, width, height)?);
        }
        
        // タイトルの追加
        if let Some(ref title) = self.config.title {
            svg.push_str(&self.render_title(title, width * depth)?);
        }
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    /// テンソルの統計情報を可視化
    /// Visualize tensor statistics
    pub fn plot_statistics<T>(&self, tensor: &Tensor<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug + 'static,
    {
        let data = tensor.as_slice().ok_or_else(|| {
            VisualizationError::InvalidDataFormat("Tensor data not accessible as slice".to_string())
        })?;
        let stats = self.compute_statistics(&data)?;
        
        let mut svg = self.generate_statistics_header();
        
        // 統計グラフの描画
        svg.push_str(&self.render_statistics_chart(&stats)?);
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    /// 変数の可視化（勾配情報付き）
    /// Visualize variable with gradient information
    pub fn plot_variable<T>(&self, variable: &Variable<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let tensor = variable.data();
        let tensor_guard = tensor.read().map_err(|e| {
            VisualizationError::PlottingError(format!("Failed to read tensor data: {}", e))
        })?;
        
        let mut svg = self.plot_heatmap(&tensor_guard)?;
        
        // 勾配情報の追加
        if variable.requires_grad() {
            let grad_info = self.render_gradient_info(variable)?;
            svg = svg.replace("</svg>", &format!("{}\n</svg>", grad_info));
        }
        
        Ok(svg)
    }
    
    // プライベートヘルパーメソッド
    
    fn generate_heatmap_header(&self, width: usize, height: usize) -> String {
        let svg_width = width * 20 + 100;  // セルサイズ20px + マージン
        let svg_height = height * 20 + 100;
        
        format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
.cell {{ stroke: #fff; stroke-width: 1; }}
.title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
.colorbar {{ font-family: Arial, sans-serif; font-size: 10px; }}
.value-text {{ font-family: Arial, sans-serif; font-size: 8px; text-anchor: middle; }}
</style>
"#,
            svg_width, svg_height
        )
    }
    
    fn generate_bar_chart_header(&self, length: usize) -> String {
        let svg_width = length * 30 + 100;
        let svg_height = 400;
        
        format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
.bar {{ stroke: none; }}
.axis {{ stroke: #333; stroke-width: 1; }}
.title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
.label {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }}
</style>
"#,
            svg_width, svg_height
        )
    }
    
    fn generate_3d_slices_header(&self, depth: usize, width: usize, height: usize) -> String {
        let cols = (depth as f32).sqrt().ceil() as usize;
        let rows = (depth + cols - 1) / cols;
        
        let svg_width = cols * (width * 15 + 20) + 50;
        let svg_height = rows * (height * 15 + 30) + 50;
        
        format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
.slice-cell {{ stroke: #fff; stroke-width: 0.5; }}
.slice-title {{ font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; }}
.main-title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
</style>
"#,
            svg_width, svg_height
        )
    }
    
    fn generate_statistics_header(&self) -> String {
        r#"<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
<style>
.stat-bar {{ stroke: none; }}
.stat-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: end; }}
.stat-value {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: start; }}
</style>
"#.to_string()
    }
    
    fn normalize_data<T>(&self, data: &[T]) -> VisualizationResult<Vec<f32>>
    where
        T: Float + std::fmt::Display,
    {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let min_val = data.iter().cloned().fold(T::infinity(), T::min);
        let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
        
        let range = max_val - min_val;
        
        if range == T::zero() {
            Ok(vec![0.5; data.len()])  // 全ての値が同じ場合
        } else {
            Ok(data.iter()
                .map(|&val| ((val - min_val) / range).to_f32().unwrap_or(0.0))
                .collect())
        }
    }
    
    fn render_heatmap_cells<T>(&self, data: &[T], width: usize, _height: usize) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display,
    {
        let mut cells = String::new();
        
        for (i, &value) in data.iter().enumerate() {
            let row = i / width;
            let col = i % width;
            let x = col * 20 + 50;
            let y = row * 20 + 50;
            
            let color = self.value_to_color(value.to_f32().unwrap_or(0.0));
            
            cells.push_str(&format!(
                r#"<rect x="{}" y="{}" width="20" height="20" fill="{}" class="cell" />"#,
                x, y, color
            ));
            
            if self.config.show_values {
                let text_color = if value.to_f32().unwrap_or(0.0) > 0.5 { "#fff" } else { "#000" };
                cells.push_str(&format!(
                    r#"<text x="{}" y="{}" fill="{}" class="value-text">{}</text>"#,
                    x + 10, y + 15, text_color, format!("{:.precision$}", value, precision = self.config.precision)
                ));
            }
        }
        
        Ok(cells)
    }
    
    fn render_bar_chart<T>(&self, data: &[T], length: usize) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display,
    {
        if data.is_empty() {
            return Ok(String::new());
        }
        
        let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
        let min_val = data.iter().cloned().fold(T::infinity(), T::min);
        let range = max_val - min_val;
        
        let mut bars = String::new();
        let bar_width = 25;
        let max_height = 300.0;
        
        for (i, &value) in data.iter().enumerate() {
            let x = i * 30 + 50;
            let normalized_value = if range != T::zero() {
                ((value - min_val) / range).to_f32().unwrap_or(0.0)
            } else {
                0.5
            };
            
            let bar_height = normalized_value * max_height;
            let y = 350.0 - bar_height;
            
            let color = self.value_to_color(normalized_value);
            
            bars.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="bar" />"#,
                x, y, bar_width, bar_height, color
            ));
            
            // インデックスラベル
            bars.push_str(&format!(
                r#"<text x="{}" y="375" class="label">{}</text>"#,
                x + bar_width / 2, i
            ));
        }
        
        // X軸
        bars.push_str(&format!(
            r#"<line x1="40" y1="350" x2="{}" y2="350" class="axis" />"#,
            length * 30 + 60
        ));
        
        // Y軸
        bars.push_str(r#"<line x1="50" y1="50" x2="50" y2="350" class="axis" />"#);
        
        Ok(bars)
    }
    
    fn render_slice_heatmap<T>(&self, data: &[T], slice_idx: usize, width: usize, height: usize) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display,
    {
        let cols = ((slice_idx + 1) as f32).sqrt().ceil() as usize;
        let col = slice_idx % cols;
        let row = slice_idx / cols;
        
        let offset_x = col * (width * 15 + 20) + 25;
        let offset_y = row * (height * 15 + 30) + 25;
        
        let mut slice_svg = String::new();
        
        // スライスタイトル
        slice_svg.push_str(&format!(
            r#"<text x="{}" y="{}" class="slice-title">Slice {}</text>"#,
            offset_x + (width * 15) / 2, offset_y - 5, slice_idx
        ));
        
        // セルの描画
        for (i, &value) in data.iter().enumerate() {
            let cell_row = i / width;
            let cell_col = i % width;
            let x = offset_x + cell_col * 15;
            let y = offset_y + cell_row * 15;
            
            let color = self.value_to_color(value.to_f32().unwrap_or(0.0));
            
            slice_svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="15" height="15" fill="{}" class="slice-cell" />"#,
                x, y, color
            ));
        }
        
        Ok(slice_svg)
    }
    
    fn render_colorbar(&self) -> VisualizationResult<String> {
        let mut colorbar = String::new();
        
        // カラーバーの実装（簡略化）
        let bar_x = 20;
        let bar_y = 50;
        let bar_width = 20;
        let bar_height = 200;
        let steps = 50;
        
        for i in 0..steps {
            let y = bar_y + i * (bar_height / steps);
            let value = i as f32 / steps as f32;
            let color = self.value_to_color(value);
            
            colorbar.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" />"#,
                bar_x, y, bar_width, bar_height / steps, color
            ));
        }
        
        // カラーバーラベル
        colorbar.push_str(&format!(
            r#"<text x="{}" y="{}" class="colorbar">1.0</text>"#,
            bar_x - 5, bar_y + 5
        ));
        colorbar.push_str(&format!(
            r#"<text x="{}" y="{}" class="colorbar">0.0</text>"#,
            bar_x - 5, bar_y + bar_height + 5
        ));
        
        Ok(colorbar)
    }
    
    fn render_title(&self, title: &str, width: usize) -> VisualizationResult<String> {
        Ok(format!(
            r#"<text x="{}" y="30" class="title">{}</text>"#,
            width * 10 + 50, title
        ))
    }
    
    fn render_gradient_info<T>(&self, _variable: &Variable<T>) -> VisualizationResult<String>
    where
        T: Float + std::fmt::Display + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        // 勾配情報の描画（簡略化）
        Ok(r#"<text x="10" y="15" style="font-size: 10px; fill: #666;">Gradient: Available</text>"#.to_string())
    }
    
    fn render_statistics_chart(&self, stats: &HashMap<String, f32>) -> VisualizationResult<String> {
        let mut chart = String::new();
        let mut y_offset = 50;
        
        for (name, &value) in stats {
            chart.push_str(&format!(
                r#"<text x="50" y="{}" class="stat-label">{}:</text>"#,
                y_offset, name
            ));
            chart.push_str(&format!(
                "<rect x=\"120\" y=\"{}\" width=\"{}\" height=\"15\" fill=\"#3498db\" class=\"stat-bar\"/>",
                y_offset - 12, value.abs() * 300.0
            ));
            chart.push_str(&format!(
                r#"<text x="430" y="{}" class="stat-value">{:.3}</text>"#,
                y_offset, value
            ));
            y_offset += 30;
        }
        
        Ok(chart)
    }
    
    fn value_to_color(&self, value: f32) -> String {
        let clamped_value = value.max(0.0).min(1.0);
        
        match self.config.colormap {
            ColorMap::Grayscale => {
                let gray = (255.0 * clamped_value) as u8;
                format!("rgb({},{},{})", gray, gray, gray)
            },
            ColorMap::Jet => {
                // ジェットカラーマップの実装（簡略化）
                if clamped_value < 0.25 {
                    let t = clamped_value * 4.0;
                    format!("rgb(0,{},255)", (255.0 * t) as u8)
                } else if clamped_value < 0.5 {
                    let t = (clamped_value - 0.25) * 4.0;
                    format!("rgb(0,255,{})", (255.0 * (1.0 - t)) as u8)
                } else if clamped_value < 0.75 {
                    let t = (clamped_value - 0.5) * 4.0;
                    format!("rgb({},255,0)", (255.0 * t) as u8)
                } else {
                    let t = (clamped_value - 0.75) * 4.0;
                    format!("rgb(255,{},0)", (255.0 * (1.0 - t)) as u8)
                }
            },
            ColorMap::Viridis => {
                // ビリディスカラーマップ（近似）
                let r = (255.0 * (0.267 + 0.329 * clamped_value)) as u8;
                let g = (255.0 * (0.004 + 0.678 * clamped_value)) as u8;
                let b = (255.0 * (0.329 + 0.431 * clamped_value)) as u8;
                format!("rgb({},{},{})", r, g, b)
            },
            ColorMap::Plasma => {
                // プラズマカラーマップ（近似）
                let r = (255.0 * (0.541 + 0.446 * clamped_value)) as u8;
                let g = (255.0 * (0.097 + 0.651 * clamped_value)) as u8;
                let b = (255.0 * (0.751 - 0.446 * clamped_value)) as u8;
                format!("rgb({},{},{})", r, g, b)
            },
            ColorMap::Inferno => {
                // インフェルノカラーマップ（近似）
                let r = (255.0 * (0.001 + 0.999 * clamped_value)) as u8;
                let g = (255.0 * (0.000 + 0.644 * clamped_value.powi(2))) as u8;
                let b = (255.0 * (0.000 + 0.361 * clamped_value.powi(3))) as u8;
                format!("rgb({},{},{})", r, g, b)
            },
            ColorMap::Coolwarm => {
                // クールウォームカラーマップ
                if clamped_value < 0.5 {
                    let t = clamped_value * 2.0;
                    let r = (255.0 * (0.230 + 0.299 * t)) as u8;
                    let g = (255.0 * (0.299 + 0.434 * t)) as u8;
                    let b = (255.0 * (0.754 + 0.178 * t)) as u8;
                    format!("rgb({},{},{})", r, g, b)
                } else {
                    let t = (clamped_value - 0.5) * 2.0;
                    let r = (255.0 * (0.706 + 0.216 * t)) as u8;
                    let g = (255.0 * (0.016 + 0.210 * t)) as u8;
                    let b = (255.0 * (0.150 - 0.118 * t)) as u8;
                    format!("rgb({},{},{})", r, g, b)
                }
            },
            ColorMap::Custom(ref colors) => {
                let idx = (clamped_value * (colors.len() - 1) as f32).round() as usize;
                colors.get(idx).unwrap_or(&"#000000".to_string()).clone()
            },
        }
    }
    
    fn extract_slice<T>(&self, data: &[T], slice_idx: usize, height: usize, width: usize) -> VisualizationResult<Vec<T>>
    where
        T: Float + Copy,
    {
        let start_idx = slice_idx * height * width;
        let end_idx = start_idx + height * width;
        
        if end_idx > data.len() {
            return Err(VisualizationError::InvalidDataFormat(
                "Slice index out of bounds".to_string()
            ));
        }
        
        Ok(data[start_idx..end_idx].to_vec())
    }
    
    fn compute_statistics<T>(&self, data: &[T]) -> VisualizationResult<HashMap<String, f32>>
    where
        T: Float + std::fmt::Display,
    {
        if data.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut stats = HashMap::new();
        
        let sum = data.iter().cloned().fold(T::zero(), |acc, x| acc + x);
        let mean = sum / T::from(data.len()).unwrap();
        
        let min_val = data.iter().cloned().fold(T::infinity(), T::min);
        let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
        
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .fold(T::zero(), |acc, x| acc + x) / T::from(data.len()).unwrap();
        let std_dev = variance.sqrt();
        
        stats.insert("Mean".to_string(), mean.to_f32().unwrap_or(0.0));
        stats.insert("Std Dev".to_string(), std_dev.to_f32().unwrap_or(0.0));
        stats.insert("Min".to_string(), min_val.to_f32().unwrap_or(0.0));
        stats.insert("Max".to_string(), max_val.to_f32().unwrap_or(0.0));
        stats.insert("Range".to_string(), (max_val - min_val).to_f32().unwrap_or(0.0));
        
        Ok(stats)
    }
}

impl Default for TensorVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Display + std::fmt::Debug + 'static> Visualizable<T> for Tensor<T> {
    fn to_plot_data(&self) -> VisualizationResult<PlotData<T>> {
        let shape = self.shape();
        if shape.len() != 1 {
            return Err(VisualizationError::InvalidDataFormat(
                "Can only convert 1D tensor to plot data".to_string()
            ));
        }
        
        let indices: Vec<T> = (0..shape[0])
            .map(|i| T::from(i).unwrap())
            .collect();
        
        let data = self.as_slice().ok_or_else(|| {
            VisualizationError::InvalidDataFormat("Tensor data not accessible as slice".to_string())
        })?;
        
        Ok(PlotData::new(indices, data.to_vec(), "Tensor Values".to_string()))
    }
    
    fn validate_config(&self, _config: &super::plotting::PlotConfig) -> VisualizationResult<()> {
        let shape = self.shape();
        if shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
            return Err(VisualizationError::ConfigError(
                "Cannot visualize empty tensor".to_string()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_tensor_visualizer_creation() {
        let visualizer = TensorVisualizer::new();
        assert_eq!(visualizer.config.colormap, ColorMap::Viridis);
        assert!(visualizer.config.normalize);
    }
    
    #[test]
    fn test_colormap_default() {
        let colormap = ColorMap::default();
        assert_eq!(colormap, ColorMap::Viridis);
    }
    
    #[test]
    fn test_value_to_color() {
        let visualizer = TensorVisualizer::new();
        
        let color = visualizer.value_to_color(0.5);
        assert!(color.starts_with("rgb("));
        
        let color_min = visualizer.value_to_color(0.0);
        let color_max = visualizer.value_to_color(1.0);
        assert_ne!(color_min, color_max);
    }
    
    #[test]
    fn test_plot_config_default() {
        let config = TensorPlotConfig::default();
        assert_eq!(config.colormap, ColorMap::Viridis);
        assert!(config.normalize);
        assert!(config.show_colorbar);
        assert!(!config.show_values);
    }
    
    #[test]
    fn test_tensor_visualization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]);
        let visualizer = TensorVisualizer::new();
        
        let result = visualizer.plot_heatmap(&tensor);
        assert!(result.is_ok());
        
        let svg = result.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
    
    #[test]
    fn test_1d_tensor_bar_chart() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 1.5];
        let tensor = Tensor::from_vec(data, vec![5]);
        let visualizer = TensorVisualizer::new();
        
        let result = visualizer.plot_bar_chart(&tensor);
        assert!(result.is_ok());
        
        let svg = result.unwrap();
        assert!(svg.contains("rect"));  // 棒グラフのバーが含まれる
        assert!(svg.contains("class=\"bar\""));
    }
}