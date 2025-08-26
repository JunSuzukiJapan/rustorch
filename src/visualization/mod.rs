//! # 可視化ツール / Visualization Tools
//!
//! RusTorchの包括的な可視化機能を提供するモジュールです。
//! 機械学習のワークフロー全体にわたって、データの理解とモデルの解釈を支援します。
//!
//! This module provides comprehensive visualization capabilities for RusTorch,
//! supporting data understanding and model interpretation throughout the machine learning workflow.
//!
//! ## ✨ Features / 機能
//!
//! - **📈 Training Curves**: Loss and metrics visualization with customizable styling
//! - **🔢 Tensor Visualization**: Heatmaps, bar charts, and 3D slice views
//! - **🕸️ Computation Graphs**: SVG and DOT format graph visualization
//! - **🎨 Color Palettes**: Professional colormaps (Viridis, Plasma, Jet, etc.)
//! - **📊 Dashboard Creation**: Multi-plot HTML dashboards
//! - **💾 Multiple Formats**: SVG, HTML, DOT output support
//!
//! ## 🚀 Quick Start / クイックスタート
//!
//! ```no_run
//! use rustorch::visualization::*;
//! use rustorch::models::high_level::TrainingHistory;
//! use rustorch::tensor::Tensor;
//! use rustorch::autograd::Variable;
//! use std::collections::HashMap;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create sample training history
//!     let mut history = TrainingHistory::<f32>::new();
//!     history.add_epoch(1.0, Some(1.2), HashMap::new());
//!
//!     // Training curve visualization
//!     let plotter = TrainingPlotter::new();
//!     let svg = plotter.plot_training_curves(&history)?;
//!
//!     // Tensor visualization
//!     let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
//!     let viz = TensorVisualizer::new();
//!     let heatmap = viz.plot_heatmap(&tensor)?;
//!
//!     // Computation graph
//!     let variable = Variable::new(tensor, true);
//!     let mut graph_viz = GraphVisualizer::new();
//!     graph_viz.build_graph(&variable)?;
//!     let graph_svg = graph_viz.to_svg()?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## 📊 Supported Visualizations / 対応する可視化
//!
//! | Type | Description | Output Formats |
//! |------|-------------|----------------|
//! | Training Curves | Loss and metrics over time | SVG, HTML |
//! | Tensor Heatmaps | 2D tensor value visualization | SVG, HTML |
//! | Bar Charts | 1D tensor value distribution | SVG, HTML |
//! | 3D Slices | Multi-dimensional tensor slicing | SVG, HTML |
//! | Computation Graph | Variable and operation flow | SVG, DOT |
//! | Dashboard | Multi-plot combination view | HTML |

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
pub use graph_viz::{GraphEdge, GraphLayout, GraphNode, GraphVisualizer};
pub use plotting::{ChartType, PlotConfig, PlotStyle, TrainingPlotter};
pub use tensor_viz::{ColorMap, TensorPlotConfig, TensorVisualizer};
pub use utils::{export_format, save_plot, PlotFormat};

use crate::error::RusTorchResult; // RusTorchError,
use num_traits::Float;
use std::collections::HashMap;

// PlotData is defined later with generic type parameter

// TensorSpec is defined in model_import::mod - import when needed
pub use crate::model_import::TensorSpec;

// VisualizationError enum removed - now using unified RusTorchError system
// VisualizationErrorエナム削除 - 統一RusTorchErrorシステムを使用

// 可視化結果 (統一済み)
// Visualization result (統一済み)
// RusTorchResult already imported - no need to redefine

/// 基本的な可視化トレイト
/// Base visualization trait
pub trait Visualizable<T: Float> {
    /// データを可視化用フォーマットに変換
    /// Convert data to visualization format
    fn to_plot_data(&self) -> RusTorchResult<PlotData<T>>;

    /// 可視化設定を検証
    /// Validate visualization configuration
    fn validate_config(&self, config: &PlotConfig) -> RusTorchResult<()>;
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
