//! 学習曲線プロット機能
//! Training curve plotting functionality

use crate::error::{RusTorchError, RusTorchResult};
use crate::models::high_level::TrainingHistory;
use crate::visualization::{PlotData, PlotMetadata, Visualizable};
use num_traits::Float;
use std::path::Path;

/// プロットスタイル
/// Plot styles
#[derive(Debug, Clone, PartialEq)]
pub enum PlotStyle {
    /// 線グラフ
    Line,
    /// 点グラフ
    Scatter,
    /// 線+点グラフ
    LineScatter,
    /// 棒グラフ
    Bar,
    /// エリアグラフ
    Area,
}

/// チャートタイプ
/// Chart types
#[derive(Debug, Clone, PartialEq)]
pub enum ChartType {
    /// 単一グラフ
    Single,
    /// サブプロット
    Subplots,
    /// 重ね合わせ
    Overlay,
    /// ダッシュボード
    Dashboard,
}

/// プロット設定
/// Plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// 幅
    /// Width
    pub width: u32,
    /// 高さ
    /// Height
    pub height: u32,
    /// DPI
    pub dpi: u32,
    /// チャートタイプ
    /// Chart type
    pub chart_type: ChartType,
    /// 背景色
    /// Background color
    pub background_color: String,
    /// フォントサイズ
    /// Font size
    pub font_size: u32,
    /// ライン幅
    /// Line width
    pub line_width: f32,
    /// マーカーサイズ
    /// Marker size
    pub marker_size: f32,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 300,
            chart_type: ChartType::Single,
            background_color: "#ffffff".to_string(),
            font_size: 12,
            line_width: 2.0,
            marker_size: 4.0,
        }
    }
}

/// 学習曲線プロッター
/// Training curve plotter
#[derive(Debug)]
pub struct TrainingPlotter {
    /// プロット設定
    /// Plot configuration
    pub config: PlotConfig,
    /// メタデータ
    /// Metadata
    pub metadata: PlotMetadata,
}

impl TrainingPlotter {
    /// 新しいプロッターを作成
    /// Create a new plotter
    pub fn new() -> Self {
        Self {
            config: PlotConfig::default(),
            metadata: PlotMetadata::new(),
        }
    }
    
    /// 設定付きプロッターを作成
    /// Create plotter with configuration
    pub fn with_config(config: PlotConfig) -> Self {
        Self {
            config,
            metadata: PlotMetadata::new(),
        }
    }
    
    /// メタデータを設定
    /// Set metadata
    pub fn with_metadata(mut self, metadata: PlotMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// 学習曲線をプロット
    /// Plot training curves
    pub fn plot_training_curves<T>(&self, history: &TrainingHistory<T>) -> RusTorchResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug,
    {
        // プロットデータの準備
        let _plot_data = history.to_plot_data()?;
        
        // SVG形式でプロットを生成
        let mut svg_content = self.generate_svg_header();
        
        // 学習損失プロット
        if !history.train_loss.is_empty() {
            let train_loss_data = self.prepare_loss_data(&history.train_loss, "Training Loss", "#1f77b4")?;
            svg_content.push_str(&self.render_line_plot(&train_loss_data)?);
        }
        
        // 検証損失プロット
        if !history.val_loss.is_empty() {
            let val_loss_data = self.prepare_loss_data(&history.val_loss, "Validation Loss", "#ff7f0e")?;
            svg_content.push_str(&self.render_line_plot(&val_loss_data)?);
        }
        
        // メトリクスプロット
        if !history.metrics.is_empty() {
            svg_content.push_str(&self.render_metrics_plots(history)?);
        }
        
        svg_content.push_str("</svg>");
        
        Ok(svg_content)
    }
    
    /// 損失比較プロット
    /// Plot loss comparison
    pub fn plot_loss_comparison<T>(&self, histories: Vec<(&str, &TrainingHistory<T>)>) -> RusTorchResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug,
    {
        let mut svg_content = self.generate_svg_header();
        
        let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];
        
        for (i, (name, history)) in histories.iter().enumerate() {
            if !history.train_loss.is_empty() {
                let color = colors.get(i % colors.len()).unwrap_or(&"#000000");
                let train_data = self.prepare_loss_data(&history.train_loss, &format!("{} - Training", name), color)?;
                svg_content.push_str(&self.render_line_plot(&train_data)?);
            }
            
            if !history.val_loss.is_empty() {
                let color = colors.get(i % colors.len()).unwrap_or(&"#000000");
                let val_data = self.prepare_loss_data(&history.val_loss, &format!("{} - Validation", name), color)?;
                svg_content.push_str(&self.render_line_plot(&val_data)?);
            }
        }
        
        svg_content.push_str("</svg>");
        Ok(svg_content)
    }
    
    /// メトリクス時系列プロット
    /// Plot metrics over time
    pub fn plot_metrics_timeline<T>(&self, history: &TrainingHistory<T>, metric_name: &str) -> RusTorchResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug,
    {
        let mut svg_content = self.generate_svg_header();
        
        // 指定されたメトリクスのデータを抽出
        let metric_values = self.extract_metric_values(history, metric_name)?;
        
        if !metric_values.is_empty() {
            let epochs: Vec<f32> = (1..=metric_values.len()).map(|i| i as f32).collect();
            let plot_data = PlotData::new(epochs, metric_values, metric_name.to_string())
                .with_color("#2ca02c".to_string())
                .with_style(PlotStyle::LineScatter);
            
            svg_content.push_str(&self.render_line_plot(&plot_data)?);
        }
        
        svg_content.push_str("</svg>");
        Ok(svg_content)
    }
    
    /// ファイルに保存
    /// Save to file
    pub fn save_plot<P: AsRef<Path>>(&self, content: &str, path: P) -> RusTorchResult<()> {
        std::fs::write(path, content)?;
        Ok(())
    }
    
    // プライベートヘルパーメソッド
    
    fn generate_svg_header(&self) -> String {
        format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
.axis {{ stroke: #333; stroke-width: 1; }}
.grid {{ stroke: #ddd; stroke-width: 0.5; stroke-dasharray: 3,3; }}
.line {{ fill: none; stroke-width: {}; }}
.text {{ font-family: Arial, sans-serif; font-size: {}px; fill: #333; }}
.title {{ font-size: {}px; font-weight: bold; text-anchor: middle; }}
.legend {{ font-size: 10px; }}
</style>
<rect width="100%" height="100%" fill="{}"/>
"#,
            self.config.width, 
            self.config.height,
            self.config.line_width,
            self.config.font_size,
            self.config.font_size + 4,
            self.config.background_color
        )
    }
    
    fn prepare_loss_data<T>(&self, loss_data: &[T], label: &str, color: &str) -> RusTorchResult<PlotData<f32>>
    where
        T: Float + std::fmt::Display,
    {
        let epochs: Vec<f32> = (1..=loss_data.len()).map(|i| i as f32).collect();
        let losses: Vec<f32> = loss_data.iter()
            .map(|&loss| loss.to_f32().unwrap_or(0.0))
            .collect();
        
        Ok(PlotData::new(epochs, losses, label.to_string())
            .with_color(color.to_string())
            .with_style(PlotStyle::Line))
    }
    
    fn render_line_plot(&self, data: &PlotData<f32>) -> RusTorchResult<String> {
        if data.x_data.len() != data.y_data.len() || data.x_data.is_empty() {
            return Err(RusTorchError::InvalidDataFormat(
                "X and Y data must have the same non-zero length".to_string()
            ).into());
        }
        
        // データの正規化とスケーリング
        let margin = 50.0;
        let plot_width = self.config.width as f32 - 2.0 * margin;
        let plot_height = self.config.height as f32 - 2.0 * margin;
        
        let x_min = data.x_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = data.x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_min = data.y_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = data.y_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let x_range = if x_max > x_min { x_max - x_min } else { 1.0 };
        let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };
        
        let mut path_data = String::new();
        
        for (i, (&x, &y)) in data.x_data.iter().zip(data.y_data.iter()).enumerate() {
            let screen_x = margin + (x - x_min) / x_range * plot_width;
            let screen_y = margin + plot_height - (y - y_min) / y_range * plot_height;
            
            if i == 0 {
                path_data.push_str(&format!("M {} {}", screen_x, screen_y));
            } else {
                path_data.push_str(&format!(" L {} {}", screen_x, screen_y));
            }
        }
        
        let default_color = "#1f77b4".to_string();
        let color = data.color.as_ref().unwrap_or(&default_color);
        
        Ok(format!(
            r#"<path d="{}" class="line" stroke="{}" stroke-width="{}"/>
"#,
            path_data, color, self.config.line_width
        ))
    }
    
    fn render_metrics_plots<T>(&self, _history: &TrainingHistory<T>) -> RusTorchResult<String>
    where
        T: Float + std::fmt::Display + std::fmt::Debug,
    {
        // メトリクスプロットの実装（簡略化）
        Ok(String::new())
    }
    
    fn extract_metric_values<T>(&self, history: &TrainingHistory<T>, metric_name: &str) -> RusTorchResult<Vec<f32>>
    where
        T: Float + std::fmt::Display,
    {
        if let Some(metric_values) = history.metrics.get(metric_name) {
            let values: Vec<f32> = metric_values.iter().map(|&v| v as f32).collect();
            if values.is_empty() {
                return Err(RusTorchError::InvalidDataFormat(
                    format!("Metric '{}' has no values", metric_name)
                ).into());
            }
            Ok(values)
        } else {
            Err(RusTorchError::InvalidDataFormat(
                format!("Metric '{}' not found in training history", metric_name)
            ).into())
        }
    }
}

impl Default for TrainingPlotter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Display + std::fmt::Debug> Visualizable<T> for TrainingHistory<T> {
    fn to_plot_data(&self) -> RusTorchResult<PlotData<T>> {
        if self.train_loss.is_empty() {
            return Err(RusTorchError::InvalidDataFormat(
                "Training history contains no data".to_string()
            ).into());
        }
        
        let epochs: Vec<T> = (1..=self.train_loss.len())
            .map(|i| T::from(i).unwrap())
            .collect();
        
        Ok(PlotData::new(epochs, self.train_loss.clone(), "Training Loss".to_string()))
    }
    
    fn validate_config(&self, _config: &PlotConfig) -> RusTorchResult<()> {
        if self.train_loss.is_empty() {
            return Err(RusTorchError::ConfigError(
                "Cannot plot empty training history".to_string()
            ).into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.dpi, 300);
        assert_eq!(config.chart_type, ChartType::Single);
    }
    
    #[test]
    fn test_training_plotter_creation() {
        let plotter = TrainingPlotter::new();
        assert_eq!(plotter.config.width, 800);
        assert_eq!(plotter.config.height, 600);
    }
    
    #[test]
    fn test_plot_data_creation() {
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data = vec![0.5, 0.3, 0.1];
        let plot_data = PlotData::new(x_data.clone(), y_data.clone(), "Test".to_string())
            .with_color("#ff0000".to_string())
            .with_style(PlotStyle::LineScatter);
        
        assert_eq!(plot_data.x_data, x_data);
        assert_eq!(plot_data.y_data, y_data);
        assert_eq!(plot_data.label, "Test");
        assert_eq!(plot_data.color, Some("#ff0000".to_string()));
        assert_eq!(plot_data.style, PlotStyle::LineScatter);
    }
    
    #[test]
    fn test_training_history_visualization() {
        let mut history = TrainingHistory::<f32>::new();
        history.add_epoch(0.8, Some(0.7), HashMap::new());
        history.add_epoch(0.6, Some(0.5), HashMap::new());
        history.add_epoch(0.4, Some(0.3), HashMap::new());
        
        let plot_data = history.to_plot_data().unwrap();
        assert_eq!(plot_data.x_data.len(), 3);
        assert_eq!(plot_data.y_data.len(), 3);
        assert_eq!(plot_data.label, "Training Loss");
    }
    
    #[test]
    fn test_svg_generation() {
        let plotter = TrainingPlotter::new();
        let mut history = TrainingHistory::<f32>::new();
        history.add_epoch(0.8, Some(0.7), HashMap::new());
        history.add_epoch(0.6, Some(0.5), HashMap::new());
        
        let svg_result = plotter.plot_training_curves(&history);
        assert!(svg_result.is_ok());
        
        let svg_content = svg_result.unwrap();
        assert!(svg_content.contains("<svg"));
        assert!(svg_content.contains("</svg>"));
    }
}