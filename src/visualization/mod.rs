//! 可視化ツール
//! Visualization tools
//!
//! このモジュールは、学習の進捗、テンソルデータ、計算グラフなどを
//! 可視化するためのツールを提供します。

/// 学習曲線のプロット機能
/// Training curve plotting functionality
pub mod plotting;

/// テンソルの可視化機能
/// Tensor visualization functionality
pub mod tensor_viz;

/// 計算グラフの可視化機能
/// Computation graph visualization functionality
pub mod graph_viz;

/// 可視化ユーティリティ
/// Visualization utilities
pub mod utils;

/// 可視化機能の統合テスト
/// Visualization integration tests
#[cfg(test)]
pub mod tests;

// Re-export main visualization types
pub use plotting::{TrainingPlotter, PlotConfig, PlotStyle, ChartType};
pub use tensor_viz::{TensorVisualizer, TensorPlotConfig, ColorMap};
pub use graph_viz::{GraphVisualizer, GraphNode, GraphEdge, GraphLayout};
pub use utils::{save_plot, export_format, PlotFormat};

use num_traits::Float;
use std::collections::HashMap;

/// 可視化エラー
/// Visualization errors
#[derive(Debug)]
pub enum VisualizationError {
    /// データ形式が無効
    /// Invalid data format
    InvalidDataFormat(String),
    /// I/Oエラー
    /// I/O error
    IoError(std::io::Error),
    /// プロッティングエラー
    /// Plotting error
    PlottingError(String),
    /// 設定エラー
    /// Configuration error
    ConfigError(String),
}

impl std::fmt::Display for VisualizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualizationError::InvalidDataFormat(msg) => write!(f, "Invalid data format: {}", msg),
            VisualizationError::IoError(err) => write!(f, "File I/O error: {}", err),
            VisualizationError::PlottingError(msg) => write!(f, "Plotting error: {}", msg),
            VisualizationError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for VisualizationError {}

impl From<std::io::Error> for VisualizationError {
    fn from(err: std::io::Error) -> Self {
        VisualizationError::IoError(err)
    }
}

/// 可視化結果
/// Visualization result
pub type VisualizationResult<T> = Result<T, VisualizationError>;

/// 基本的な可視化トレイト
/// Base visualization trait
pub trait Visualizable<T: Float> {
    /// データを可視化用フォーマットに変換
    /// Convert data to visualization format
    fn to_plot_data(&self) -> VisualizationResult<PlotData<T>>;
    
    /// 可視化設定を検証
    /// Validate visualization configuration
    fn validate_config(&self, config: &PlotConfig) -> VisualizationResult<()>;
}

/// プロットデータ
/// Plot data structure
#[derive(Debug, Clone)]
pub struct PlotData<T: Float> {
    /// X軸データ
    /// X-axis data
    pub x_data: Vec<T>,
    /// Y軸データ
    /// Y-axis data
    pub y_data: Vec<T>,
    /// ラベル
    /// Label
    pub label: String,
    /// 色
    /// Color
    pub color: Option<String>,
    /// スタイル
    /// Style
    pub style: PlotStyle,
}

impl<T: Float> PlotData<T> {
    /// 新しいプロットデータを作成
    /// Create new plot data
    pub fn new(x_data: Vec<T>, y_data: Vec<T>, label: String) -> Self {
        Self {
            x_data,
            y_data,
            label,
            color: None,
            style: PlotStyle::Line,
        }
    }
    
    /// 色を設定
    /// Set color
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }
    
    /// スタイルを設定
    /// Set style
    pub fn with_style(mut self, style: PlotStyle) -> Self {
        self.style = style;
        self
    }
}

/// メタデータ
/// Metadata for plots
#[derive(Debug, Clone, Default)]
pub struct PlotMetadata {
    /// タイトル
    /// Title
    pub title: Option<String>,
    /// X軸ラベル
    /// X-axis label
    pub xlabel: Option<String>,
    /// Y軸ラベル
    /// Y-axis label
    pub ylabel: Option<String>,
    /// 凡例表示
    /// Show legend
    pub show_legend: bool,
    /// グリッド表示
    /// Show grid
    pub show_grid: bool,
    /// 追加属性
    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

impl PlotMetadata {
    /// 新しいメタデータを作成
    /// Create new metadata
    pub fn new() -> Self {
        Self::default()
    }
    
    /// タイトルを設定
    /// Set title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }
    
    /// 軸ラベルを設定
    /// Set axis labels
    pub fn with_labels(mut self, xlabel: String, ylabel: String) -> Self {
        self.xlabel = Some(xlabel);
        self.ylabel = Some(ylabel);
        self
    }
    
    /// 凡例とグリッドを有効化
    /// Enable legend and grid
    pub fn with_legend_and_grid(mut self) -> Self {
        self.show_legend = true;
        self.show_grid = true;
        self
    }
}