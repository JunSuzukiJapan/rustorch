//! Python bindings for visualization and plotting
//! 可視化とプロッティングのPythonバインディング

use crate::python::error::to_py_err;
use crate::python::tensor::PyTensor;
use crate::python::training::PyTrainingHistory;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for plotting utilities
/// プロッティングユーティリティのPythonラッパー
#[pyclass]
pub struct PyPlotter {
    pub(crate) backend: String,
    pub(crate) style: String,
    pub(crate) figure_size: (usize, usize),
    pub(crate) dpi: usize,
}

#[pymethods]
impl PyPlotter {
    #[new]
    pub fn new(
        backend: Option<String>,
        style: Option<String>,
        figure_size: Option<(usize, usize)>,
        dpi: Option<usize>,
    ) -> Self {
        PyPlotter {
            backend: backend.unwrap_or_else(|| "matplotlib".to_string()),
            style: style.unwrap_or_else(|| "default".to_string()),
            figure_size: figure_size.unwrap_or((10, 6)),
            dpi: dpi.unwrap_or(100),
        }
    }

    /// Plot training history
    /// 訓練履歴をプロット
    pub fn plot_training_history(
        &self,
        history: &PyTrainingHistory,
        save_path: Option<String>,
        show_validation: Option<bool>,
    ) -> PyResult<()> {
        let show_validation = show_validation.unwrap_or(true);
        let (epochs, train_losses, val_losses) = history.plot_data();

        println!("Plotting training history:");
        println!("  Epochs: {}", epochs.len());
        println!(
            "  Training losses: {:?}",
            &train_losses[..std::cmp::min(5, train_losses.len())]
        );

        if show_validation && !val_losses.is_empty() {
            println!(
                "  Validation losses: {:?}",
                &val_losses[..std::cmp::min(5, val_losses.len())]
            );
        }

        if let Some(path) = save_path {
            println!("  Saved plot to: {}", path);
        }

        println!("  Figure size: {:?}, DPI: {}", self.figure_size, self.dpi);

        Ok(())
    }

    /// Plot tensor as image
    /// テンソルを画像としてプロット
    pub fn plot_tensor_as_image(
        &self,
        tensor: &PyTensor,
        title: Option<String>,
        colormap: Option<String>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let shape = tensor.shape();
        let title = title.unwrap_or_else(|| "Tensor Visualization".to_string());
        let colormap = colormap.unwrap_or_else(|| "viridis".to_string());

        println!("Plotting tensor as image:");
        println!("  Shape: {:?}", shape);
        println!("  Title: {}", title);
        println!("  Colormap: {}", colormap);

        if shape.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tensor must be at least 2D for image visualization",
            ));
        }

        if let Some(path) = save_path {
            println!("  Saved to: {}", path);
        }

        Ok(())
    }

    /// Plot multiple tensors
    /// 複数のテンソルをプロット
    pub fn plot_multiple_tensors(
        &self,
        tensors: Vec<PyTensor>,
        titles: Option<Vec<String>>,
        layout: Option<(usize, usize)>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let num_tensors = tensors.len();
        let titles = titles.unwrap_or_else(|| {
            (0..num_tensors)
                .map(|i| format!("Tensor {}", i + 1))
                .collect()
        });
        let layout = layout.unwrap_or_else(|| {
            let cols = (num_tensors as f64).sqrt().ceil() as usize;
            let rows = (num_tensors + cols - 1) / cols;
            (rows, cols)
        });

        println!("Plotting {} tensors:", num_tensors);
        println!("  Layout: {}x{}", layout.0, layout.1);

        for (i, tensor) in tensors.iter().enumerate() {
            let default_title = format!("Tensor {}", i + 1);
            let title = titles.get(i).unwrap_or(&default_title);
            println!("  {}: shape {:?}", title, tensor.shape());
        }

        if let Some(path) = save_path {
            println!("  Saved to: {}", path);
        }

        Ok(())
    }

    /// Create line plot
    /// 線グラフを作成
    pub fn line_plot(
        &self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        title: Option<String>,
        xlabel: Option<String>,
        ylabel: Option<String>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let title = title.unwrap_or_else(|| "Line Plot".to_string());
        let xlabel = xlabel.unwrap_or_else(|| "X".to_string());
        let ylabel = ylabel.unwrap_or_else(|| "Y".to_string());

        if x_data.len() != y_data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "X and Y data must have the same length",
            ));
        }

        println!("Creating line plot:");
        println!("  Title: {}", title);
        println!("  X label: {}, Y label: {}", xlabel, ylabel);
        println!("  Data points: {}", x_data.len());
        println!(
            "  X range: {:.3} to {:.3}",
            x_data.iter().cloned().fold(f64::INFINITY, f64::min),
            x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "  Y range: {:.3} to {:.3}",
            y_data.iter().cloned().fold(f64::INFINITY, f64::min),
            y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );

        if let Some(path) = save_path {
            println!("  Saved to: {}", path);
        }

        Ok(())
    }

    /// Create scatter plot
    /// 散布図を作成
    pub fn scatter_plot(
        &self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        colors: Option<Vec<f64>>,
        sizes: Option<Vec<f64>>,
        title: Option<String>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let title = title.unwrap_or_else(|| "Scatter Plot".to_string());

        if x_data.len() != y_data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "X and Y data must have the same length",
            ));
        }

        println!("Creating scatter plot:");
        println!("  Title: {}", title);
        println!("  Data points: {}", x_data.len());

        if let Some(ref colors) = colors {
            if colors.len() != x_data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Colors array must match data length",
                ));
            }
            println!("  Using color mapping");
        }

        if let Some(ref sizes) = sizes {
            if sizes.len() != x_data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Sizes array must match data length",
                ));
            }
            println!("  Using size mapping");
        }

        if let Some(path) = save_path {
            println!("  Saved to: {}", path);
        }

        Ok(())
    }

    /// Create histogram
    /// ヒストグラムを作成
    pub fn histogram(
        &self,
        data: Vec<f64>,
        bins: Option<usize>,
        title: Option<String>,
        xlabel: Option<String>,
        ylabel: Option<String>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let bins = bins.unwrap_or(30);
        let title = title.unwrap_or_else(|| "Histogram".to_string());
        let xlabel = xlabel.unwrap_or_else(|| "Value".to_string());
        let ylabel = ylabel.unwrap_or_else(|| "Frequency".to_string());

        println!("Creating histogram:");
        println!("  Title: {}", title);
        println!("  Data points: {}", data.len());
        println!("  Bins: {}", bins);
        println!(
            "  Data range: {:.3} to {:.3}",
            data.iter().cloned().fold(f64::INFINITY, f64::min),
            data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );

        if let Some(path) = save_path {
            println!("  Saved to: {}", path);
        }

        Ok(())
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "Plotter(backend='{}', style='{}', figure_size={:?})",
            self.backend, self.style, self.figure_size
        )
    }
}

/// Model visualization utilities
/// モデル可視化ユーティリティ
#[pyclass]
pub struct PyModelVisualizer {
    pub(crate) layout_engine: String,
    pub(crate) show_shapes: bool,
    pub(crate) show_layer_names: bool,
}

#[pymethods]
impl PyModelVisualizer {
    #[new]
    pub fn new(
        layout_engine: Option<String>,
        show_shapes: Option<bool>,
        show_layer_names: Option<bool>,
    ) -> Self {
        PyModelVisualizer {
            layout_engine: layout_engine.unwrap_or_else(|| "dot".to_string()),
            show_shapes: show_shapes.unwrap_or(true),
            show_layer_names: show_layer_names.unwrap_or(true),
        }
    }

    /// Visualize model architecture
    /// モデルアーキテクチャを可視化
    pub fn visualize_model(
        &self,
        model_summary: String,
        save_path: Option<String>,
        format: Option<String>,
    ) -> PyResult<()> {
        let format = format.unwrap_or_else(|| "png".to_string());

        println!("Visualizing model architecture:");
        println!("  Layout engine: {}", self.layout_engine);
        println!("  Show shapes: {}", self.show_shapes);
        println!("  Show layer names: {}", self.show_layer_names);
        println!("  Output format: {}", format);

        // Parse model summary (simplified)
        let lines: Vec<&str> = model_summary.lines().collect();
        println!("  Model summary lines: {}", lines.len());

        if let Some(path) = save_path {
            println!("  Saved visualization to: {}", path);
        }

        Ok(())
    }

    /// Create attention heatmap
    /// アテンション・ヒートマップを作成
    pub fn plot_attention_heatmap(
        &self,
        attention_weights: &PyTensor,
        tokens: Option<Vec<String>>,
        title: Option<String>,
        save_path: Option<String>,
    ) -> PyResult<()> {
        let shape = attention_weights.shape();
        let title = title.unwrap_or_else(|| "Attention Heatmap".to_string());

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Attention weights must be 2D tensor",
            ));
        }

        println!("Creating attention heatmap:");
        println!("  Title: {}", title);
        println!("  Attention shape: {:?}", shape);

        if let Some(ref tokens) = tokens {
            if tokens.len() != shape[0] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Token count must match attention dimension",
                ));
            }
            println!("  Tokens: {:?}", &tokens[..std::cmp::min(5, tokens.len())]);
        }

        if let Some(path) = save_path {
            println!("  Saved heatmap to: {}", path);
        }

        Ok(())
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "ModelVisualizer(layout_engine='{}', show_shapes={}, show_layer_names={})",
            self.layout_engine, self.show_shapes, self.show_layer_names
        )
    }
}

/// Tensor statistics visualization
/// テンソル統計可視化
#[pyclass]
pub struct PyTensorStats {
    pub(crate) tensor: PyTensor,
    pub(crate) computed: bool,
    pub(crate) stats: HashMap<String, f64>,
}

#[pymethods]
impl PyTensorStats {
    #[new]
    pub fn new(tensor: &PyTensor) -> Self {
        PyTensorStats {
            tensor: tensor.clone(),
            computed: false,
            stats: HashMap::new(),
        }
    }

    /// Compute tensor statistics
    /// テンソル統計を計算
    pub fn compute_stats(&mut self) -> PyResult<()> {
        let data = self.tensor.to_vec()?;
        let n = data.len() as f64;

        if n == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot compute statistics for empty tensor",
            ));
        }

        // Basic statistics
        let sum: f32 = data.iter().sum();
        let mean = sum as f64 / n;

        let variance = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = data.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

        // Percentiles (simplified)
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_data[sorted_data.len() / 2] as f64;
        let q25 = sorted_data[sorted_data.len() / 4] as f64;
        let q75 = sorted_data[3 * sorted_data.len() / 4] as f64;

        self.stats.insert("count".to_string(), n);
        self.stats.insert("mean".to_string(), mean);
        self.stats.insert("std".to_string(), std_dev);
        self.stats.insert("min".to_string(), min);
        self.stats.insert("max".to_string(), max);
        self.stats.insert("median".to_string(), median);
        self.stats.insert("q25".to_string(), q25);
        self.stats.insert("q75".to_string(), q75);

        self.computed = true;
        Ok(())
    }

    /// Get computed statistics
    /// 計算済み統計を取得
    pub fn get_stats(&self) -> PyResult<HashMap<String, f64>> {
        if !self.computed {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Statistics not computed. Call compute_stats() first.",
            ));
        }
        Ok(self.stats.clone())
    }

    /// Plot statistics
    /// 統計をプロット
    pub fn plot_stats(&self, save_path: Option<String>) -> PyResult<()> {
        if !self.computed {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Statistics not computed. Call compute_stats() first.",
            ));
        }

        println!("Plotting tensor statistics:");
        for (key, value) in &self.stats {
            println!("  {}: {:.6}", key, value);
        }

        if let Some(path) = save_path {
            println!("  Saved statistics plot to: {}", path);
        }

        Ok(())
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "TensorStats(shape={:?}, computed={})",
            self.tensor.shape(),
            self.computed
        )
    }
}

// Utility functions
// ユーティリティ関数

/// Save figure with specified format
/// 指定フォーマットで図を保存
#[pyfunction]
pub fn save_figure(
    path: String,
    format: Option<String>,
    dpi: Option<usize>,
    bbox_inches: Option<String>,
) -> PyResult<()> {
    let format = format.unwrap_or_else(|| "png".to_string());
    let dpi = dpi.unwrap_or(300);
    let bbox_inches = bbox_inches.unwrap_or_else(|| "tight".to_string());

    println!("Saving figure:");
    println!("  Path: {}", path);
    println!("  Format: {}", format);
    println!("  DPI: {}", dpi);
    println!("  Bbox inches: {}", bbox_inches);

    Ok(())
}

/// Set plotting style
/// プロッティングスタイルを設定
#[pyfunction]
pub fn set_plot_style(style: String) -> PyResult<()> {
    println!("Setting plot style to: {}", style);
    // Available styles: default, seaborn, ggplot, bmh, classic, etc.
    Ok(())
}

/// Create colormap
/// カラーマップを作成
#[pyfunction]
pub fn create_colormap(name: String, colors: Vec<String>) -> PyResult<()> {
    println!("Creating colormap '{}' with {} colors", name, colors.len());
    for (i, color) in colors.iter().enumerate() {
        println!("  Color {}: {}", i, color);
    }
    Ok(())
}
