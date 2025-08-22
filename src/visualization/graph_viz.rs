//! 計算グラフ可視化機能
//! Computation graph visualization functionality

use super::{VisualizationResult, VisualizationError};
use crate::autograd::Variable;
use num_traits::Float;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// グラフノード
/// Graph node representation
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// ノードID
    /// Node ID
    pub id: String,
    /// ノード名
    /// Node name
    pub name: String,
    /// ノードタイプ
    /// Node type
    pub node_type: NodeType,
    /// 形状情報
    /// Shape information
    pub shape: Vec<usize>,
    /// 位置座標
    /// Position coordinates
    pub position: (f32, f32),
    /// スタイル
    /// Style
    pub style: NodeStyle,
}

/// グラフエッジ
/// Graph edge representation
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// 開始ノードID
    /// Source node ID
    pub from: String,
    /// 終了ノードID
    /// Target node ID
    pub to: String,
    /// エッジラベル
    /// Edge label
    pub label: Option<String>,
    /// エッジスタイル
    /// Edge style
    pub style: EdgeStyle,
}

/// ノードタイプ
/// Node types
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// 入力変数
    Variable,
    /// 演算子
    Operation(String),
    /// 定数
    Constant,
    /// 出力
    Output,
    /// 勾配
    Gradient,
}

/// ノードスタイル
/// Node style
#[derive(Debug, Clone)]
pub struct NodeStyle {
    /// 色
    /// Color
    pub color: String,
    /// 境界色
    /// Border color
    pub border_color: String,
    /// 形状
    /// Shape
    pub shape: NodeShape,
    /// サイズ
    /// Size
    pub size: (f32, f32),
}

/// エッジスタイル
/// Edge style
#[derive(Debug, Clone)]
pub struct EdgeStyle {
    /// 色
    /// Color
    pub color: String,
    /// 線の太さ
    /// Line width
    pub width: f32,
    /// 線の種類
    /// Line style
    pub line_type: LineType,
    /// 矢印の種類
    /// Arrow type
    pub arrow_type: ArrowType,
}

/// ノード形状
/// Node shapes
#[derive(Debug, Clone, PartialEq)]
pub enum NodeShape {
    /// 矩形
    Rectangle,
    /// 楕円
    Ellipse,
    /// ダイアモンド
    Diamond,
    /// 円
    Circle,
}

/// 線の種類
/// Line types
#[derive(Debug, Clone, PartialEq)]
pub enum LineType {
    /// 実線
    Solid,
    /// 破線
    Dashed,
    /// 点線
    Dotted,
}

/// 矢印の種類
/// Arrow types
#[derive(Debug, Clone, PartialEq)]
pub enum ArrowType {
    /// 標準矢印
    Standard,
    /// 太い矢印
    Bold,
    /// 矢印なし
    None,
}

/// グラフレイアウト
/// Graph layout algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum GraphLayout {
    /// 階層レイアウト
    Hierarchical,
    /// 力学的レイアウト
    ForceDirected,
    /// 円形レイアウト
    Circular,
    /// グリッドレイアウト
    Grid,
}

/// 計算グラフ可視化クラス
/// Computation graph visualizer
#[derive(Debug)]
pub struct GraphVisualizer {
    /// ノードのリスト
    /// List of nodes
    pub nodes: Vec<GraphNode>,
    /// エッジのリスト
    /// List of edges
    pub edges: Vec<GraphEdge>,
    /// レイアウトアルゴリズム
    /// Layout algorithm
    pub layout: GraphLayout,
    /// キャンバスサイズ
    /// Canvas size
    pub canvas_size: (f32, f32),
}

impl GraphVisualizer {
    /// 新しいグラフビジュアライザーを作成
    /// Create a new graph visualizer
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            layout: GraphLayout::Hierarchical,
            canvas_size: (800.0, 600.0),
        }
    }
    
    /// レイアウト設定付きビジュアライザーを作成
    /// Create visualizer with layout
    pub fn with_layout(layout: GraphLayout) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            layout,
            canvas_size: (800.0, 600.0),
        }
    }
    
    /// 変数から計算グラフを構築
    /// Build computation graph from variable
    pub fn build_graph<T>(&mut self, variable: &Variable<T>) -> VisualizationResult<()>
    where
        T: Float + Debug + std::fmt::Display + Send + Sync + 'static,
    {
        self.nodes.clear();
        self.edges.clear();
        
        let mut visited = HashSet::new();
        let mut node_counter = 0;
        
        self.traverse_variable(variable, &mut visited, &mut node_counter)?;
        self.apply_layout()?;
        
        Ok(())
    }
    
    /// 複数の変数から計算グラフを構築
    /// Build computation graph from multiple variables
    pub fn build_graph_multi<T>(&mut self, variables: &[&Variable<T>]) -> VisualizationResult<()>
    where
        T: Float + Debug + std::fmt::Display + Send + Sync + 'static,
    {
        self.nodes.clear();
        self.edges.clear();
        
        let mut visited = HashSet::new();
        let mut node_counter = 0;
        
        for variable in variables {
            self.traverse_variable(variable, &mut visited, &mut node_counter)?;
        }
        
        self.apply_layout()?;
        Ok(())
    }
    
    /// グラフをSVGとして出力
    /// Export graph as SVG
    pub fn to_svg(&self) -> VisualizationResult<String> {
        let mut svg = self.generate_svg_header();
        
        // 背景の描画
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"#f9f9f9\" stroke=\"#ddd\" stroke-width=\"1\"/>",
            self.canvas_size.0, self.canvas_size.1
        ));
        
        // エッジの描画
        for edge in &self.edges {
            svg.push_str(&self.render_edge(edge)?);
        }
        
        // ノードの描画
        for node in &self.nodes {
            svg.push_str(&self.render_node(node)?);
        }
        
        // 凡例の追加
        svg.push_str(&self.render_legend()?);
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    /// グラフをDOT形式で出力
    /// Export graph as DOT format
    pub fn to_dot(&self) -> VisualizationResult<String> {
        let mut dot = String::from("digraph ComputationGraph {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    node [fontname=\"Arial\"];\n");
        dot.push_str("    edge [fontname=\"Arial\"];\n\n");
        
        // ノードの定義
        for node in &self.nodes {
            let shape = match node.style.shape {
                NodeShape::Rectangle => "box",
                NodeShape::Ellipse => "ellipse",
                NodeShape::Diamond => "diamond",
                NodeShape::Circle => "circle",
            };
            
            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\" shape={} fillcolor=\"{}\" style=filled];\n",
                node.id, node.name, shape, node.style.color
            ));
        }
        
        dot.push_str("\n");
        
        // エッジの定義
        for edge in &self.edges {
            let style = match edge.style.line_type {
                LineType::Solid => "solid",
                LineType::Dashed => "dashed",
                LineType::Dotted => "dotted",
            };
            
            let label = edge.label.as_ref().map(|l| format!(" label=\"{}\"", l)).unwrap_or_default();
            
            dot.push_str(&format!(
                "    \"{}\" -> \"{}\" [color=\"{}\" style={}{}];\n",
                edge.from, edge.to, edge.style.color, style, label
            ));
        }
        
        dot.push_str("}\n");
        Ok(dot)
    }
    
    /// ノードを追加
    /// Add node
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }
    
    /// エッジを追加
    /// Add edge
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }
    
    /// ノードの統計情報を取得
    /// Get node statistics
    pub fn get_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        stats.insert("total_nodes".to_string(), self.nodes.len());
        stats.insert("total_edges".to_string(), self.edges.len());
        
        let mut node_type_counts = HashMap::new();
        for node in &self.nodes {
            let type_name = match &node.node_type {
                NodeType::Variable => "Variable",
                NodeType::Operation(_) => "Operation",
                NodeType::Constant => "Constant",
                NodeType::Output => "Output",
                NodeType::Gradient => "Gradient",
            };
            *node_type_counts.entry(type_name.to_string()).or_insert(0) += 1;
        }
        
        stats.extend(node_type_counts);
        stats
    }
    
    // プライベートメソッド
    
    fn traverse_variable<T>(&mut self, _variable: &Variable<T>, visited: &mut HashSet<String>, node_counter: &mut usize) -> VisualizationResult<()>
    where
        T: Float + Debug + std::fmt::Display + Send + Sync + 'static,
    {
        // 簡略化された実装 - 実際にはVariable構造を走査
        let node_id = format!("var_{}", node_counter);
        *node_counter += 1;
        
        if !visited.contains(&node_id) {
            visited.insert(node_id.clone());
            
            let tensor_data = _variable.data();
            let tensor_guard = tensor_data.read().map_err(|e| {
                VisualizationError::PlottingError(format!("Failed to read variable data: {}", e))
            })?;
            
            let shape = tensor_guard.shape().to_vec();
            
            let node = GraphNode {
                id: node_id.clone(),
                name: format!("Var[{}]", format_shape(&shape)),
                node_type: NodeType::Variable,
                shape: shape.clone(),
                position: (0.0, 0.0), // レイアウトで設定
                style: self.get_default_node_style(&NodeType::Variable),
            };
            
            self.add_node(node);
            
            // 勾配情報がある場合
            if _variable.requires_grad() {
                let grad_id = format!("grad_{}", node_counter);
                *node_counter += 1;
                
                let grad_node = GraphNode {
                    id: grad_id.clone(),
                    name: "∇".to_string(),
                    node_type: NodeType::Gradient,
                    shape: shape.clone(),
                    position: (0.0, 0.0),
                    style: self.get_default_node_style(&NodeType::Gradient),
                };
                
                self.add_node(grad_node);
                
                let grad_edge = GraphEdge {
                    from: node_id,
                    to: grad_id,
                    label: Some("grad".to_string()),
                    style: self.get_default_edge_style(&EdgeType::Gradient),
                };
                
                self.add_edge(grad_edge);
            }
        }
        
        Ok(())
    }
    
    fn apply_layout(&mut self) -> VisualizationResult<()> {
        match self.layout {
            GraphLayout::Hierarchical => self.apply_hierarchical_layout(),
            GraphLayout::ForceDirected => self.apply_force_directed_layout(),
            GraphLayout::Circular => self.apply_circular_layout(),
            GraphLayout::Grid => self.apply_grid_layout(),
        }
    }
    
    fn apply_hierarchical_layout(&mut self) -> VisualizationResult<()> {
        // 階層的レイアウトの実装
        let levels = self.compute_node_levels()?;
        let margin = 50.0;
        let level_height = (self.canvas_size.1 - 2.0 * margin) / levels.len().max(1) as f32;
        
        for (level_idx, level_nodes) in levels.iter().enumerate() {
            let y = margin + level_idx as f32 * level_height;
            let node_width = (self.canvas_size.0 - 2.0 * margin) / level_nodes.len().max(1) as f32;
            
            for (node_idx, node_id) in level_nodes.iter().enumerate() {
                let x = margin + node_idx as f32 * node_width + node_width / 2.0;
                
                if let Some(node) = self.nodes.iter_mut().find(|n| &n.id == node_id) {
                    node.position = (x, y);
                }
            }
        }
        
        Ok(())
    }
    
    fn apply_force_directed_layout(&mut self) -> VisualizationResult<()> {
        // 力学的レイアウトの実装（簡略化）
        let center_x = self.canvas_size.0 / 2.0;
        let center_y = self.canvas_size.1 / 2.0;
        let radius = 200.0;
        let node_count = self.nodes.len();
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / node_count as f32;
            node.position = (
                center_x + radius * angle.cos(),
                center_y + radius * angle.sin(),
            );
        }
        
        Ok(())
    }
    
    fn apply_circular_layout(&mut self) -> VisualizationResult<()> {
        let center_x = self.canvas_size.0 / 2.0;
        let center_y = self.canvas_size.1 / 2.0;
        let radius = (self.canvas_size.0.min(self.canvas_size.1) / 2.0) * 0.8;
        let node_count = self.nodes.len();
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / node_count as f32;
            node.position = (
                center_x + radius * angle.cos(),
                center_y + radius * angle.sin(),
            );
        }
        
        Ok(())
    }
    
    fn apply_grid_layout(&mut self) -> VisualizationResult<()> {
        let cols = (self.nodes.len() as f32).sqrt().ceil() as usize;
        let rows = (self.nodes.len() + cols - 1) / cols;
        
        let margin = 50.0;
        let cell_width = (self.canvas_size.0 - 2.0 * margin) / cols as f32;
        let cell_height = (self.canvas_size.1 - 2.0 * margin) / rows as f32;
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let row = i / cols;
            let col = i % cols;
            
            node.position = (
                margin + col as f32 * cell_width + cell_width / 2.0,
                margin + row as f32 * cell_height + cell_height / 2.0,
            );
        }
        
        Ok(())
    }
    
    fn compute_node_levels(&self) -> VisualizationResult<Vec<Vec<String>>> {
        // トポロジカルソートによるレベル計算（簡略化）
        let mut levels = Vec::new();
        let mut current_level = Vec::new();
        
        for node in &self.nodes {
            current_level.push(node.id.clone());
        }
        
        if !current_level.is_empty() {
            levels.push(current_level);
        }
        
        Ok(levels)
    }
    
    fn generate_svg_header(&self) -> String {
        format!(
            "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
<defs>\n\
<marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n\
<polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#666\"/>\n\
</marker>\n\
<marker id=\"arrowhead_bold\" markerWidth=\"12\" markerHeight=\"9\" refX=\"12\" refY=\"4.5\" orient=\"auto\">\n\
<polygon points=\"0 0, 12 4.5, 0 9\" fill=\"#333\"/>\n\
</marker>\n\
</defs>\n\
<style>\n\
.node-text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; dominant-baseline: middle; }}\n\
.edge-text {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; fill: #666; }}\n\
.legend-text {{ font-family: Arial, sans-serif; font-size: 11px; }}\n\
.title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}\n\
</style>\n",
            self.canvas_size.0, self.canvas_size.1
        )
    }
    
    fn render_node(&self, node: &GraphNode) -> VisualizationResult<String> {
        let mut node_svg = String::new();
        
        let (x, y) = node.position;
        let (width, height) = node.style.size;
        
        match node.style.shape {
            NodeShape::Rectangle => {
                node_svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="2" rx="5"/>"#,
                    x - width / 2.0, y - height / 2.0, width, height,
                    node.style.color, node.style.border_color
                ));
            },
            NodeShape::Ellipse => {
                node_svg.push_str(&format!(
                    r#"<ellipse cx="{}" cy="{}" rx="{}" ry="{}" fill="{}" stroke="{}" stroke-width="2"/>"#,
                    x, y, width / 2.0, height / 2.0,
                    node.style.color, node.style.border_color
                ));
            },
            NodeShape::Circle => {
                let radius = width.min(height) / 2.0;
                node_svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="2"/>"#,
                    x, y, radius, node.style.color, node.style.border_color
                ));
            },
            NodeShape::Diamond => {
                let points = format!(
                    "{},{} {},{} {},{} {},{}",
                    x, y - height / 2.0,
                    x + width / 2.0, y,
                    x, y + height / 2.0,
                    x - width / 2.0, y
                );
                node_svg.push_str(&format!(
                    r#"<polygon points="{}" fill="{}" stroke="{}" stroke-width="2"/>"#,
                    points, node.style.color, node.style.border_color
                ));
            },
        }
        
        // ノードテキスト
        node_svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"node-text\" fill=\"#333\">{}</text>",
            x, y, node.name
        ));
        
        Ok(node_svg)
    }
    
    fn render_edge(&self, edge: &GraphEdge) -> VisualizationResult<String> {
        let from_node = self.nodes.iter().find(|n| n.id == edge.from)
            .ok_or_else(|| VisualizationError::PlottingError("Source node not found".to_string()))?;
        let to_node = self.nodes.iter().find(|n| n.id == edge.to)
            .ok_or_else(|| VisualizationError::PlottingError("Target node not found".to_string()))?;
        
        let (x1, y1) = from_node.position;
        let (x2, y2) = to_node.position;
        
        let stroke_dasharray = match edge.style.line_type {
            LineType::Solid => "",
            LineType::Dashed => "stroke-dasharray=\"5,5\"",
            LineType::Dotted => "stroke-dasharray=\"2,2\"",
        };
        
        let marker = match edge.style.arrow_type {
            ArrowType::Standard => "marker-end=\"url(#arrowhead)\"",
            ArrowType::Bold => "marker-end=\"url(#arrowhead_bold)\"",
            ArrowType::None => "",
        };
        
        let mut edge_svg = format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" {} {}/>"#,
            x1, y1, x2, y2, edge.style.color, edge.style.width, stroke_dasharray, marker
        );
        
        // エッジラベル
        if let Some(ref label) = edge.label {
            let mid_x = (x1 + x2) / 2.0;
            let mid_y = (y1 + y2) / 2.0;
            edge_svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="edge-text">{}</text>"#,
                mid_x, mid_y - 5.0, label
            ));
        }
        
        Ok(edge_svg)
    }
    
    fn render_legend(&self) -> VisualizationResult<String> {
        let mut legend = String::new();
        let legend_x = 20.0;
        let mut legend_y = self.canvas_size.1 - 120.0;
        
        legend.push_str(&format!(
            r#"<text x="{}" y="{}" class="title">Computation Graph</text>"#,
            self.canvas_size.0 / 2.0, 25.0
        ));
        
        // 凡例の項目
        let legend_items = [
            ("Variable", "#lightblue", NodeShape::Rectangle),
            ("Operation", "#lightgreen", NodeShape::Ellipse),
            ("Gradient", "#lightcoral", NodeShape::Circle),
        ];
        
        for (name, color, shape) in &legend_items {
            match shape {
                NodeShape::Rectangle => {
                    legend.push_str(&format!(
                        "<rect x=\"{}\" y=\"{}\" width=\"15\" height=\"10\" fill=\"{}\" stroke=\"#333\"/>",
                        legend_x, legend_y - 5.0, color
                    ));
                },
                NodeShape::Ellipse => {
                    legend.push_str(&format!(
                        "<ellipse cx=\"{}\" cy=\"{}\" rx=\"7.5\" ry=\"5\" fill=\"{}\" stroke=\"#333\"/>",
                        legend_x + 7.5, legend_y, color
                    ));
                },
                NodeShape::Circle => {
                    legend.push_str(&format!(
                        "<circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"{}\" stroke=\"#333\"/>",
                        legend_x + 7.5, legend_y, color
                    ));
                },
                _ => {}
            }
            
            legend.push_str(&format!(
                r#"<text x="{}" y="{}" class="legend-text">{}</text>"#,
                legend_x + 20.0, legend_y + 3.0, name
            ));
            
            legend_y += 20.0;
        }
        
        Ok(legend)
    }
    
    fn get_default_node_style(&self, node_type: &NodeType) -> NodeStyle {
        match node_type {
            NodeType::Variable => NodeStyle {
                color: "#e3f2fd".to_string(),
                border_color: "#1976d2".to_string(),
                shape: NodeShape::Rectangle,
                size: (80.0, 40.0),
            },
            NodeType::Operation(_) => NodeStyle {
                color: "#e8f5e8".to_string(),
                border_color: "#4caf50".to_string(),
                shape: NodeShape::Ellipse,
                size: (70.0, 35.0),
            },
            NodeType::Constant => NodeStyle {
                color: "#fff3e0".to_string(),
                border_color: "#ff9800".to_string(),
                shape: NodeShape::Rectangle,
                size: (60.0, 30.0),
            },
            NodeType::Output => NodeStyle {
                color: "#f3e5f5".to_string(),
                border_color: "#9c27b0".to_string(),
                shape: NodeShape::Diamond,
                size: (70.0, 40.0),
            },
            NodeType::Gradient => NodeStyle {
                color: "#ffebee".to_string(),
                border_color: "#f44336".to_string(),
                shape: NodeShape::Circle,
                size: (30.0, 30.0),
            },
        }
    }
    
    fn get_default_edge_style(&self, edge_type: &EdgeType) -> EdgeStyle {
        match edge_type {
            EdgeType::Forward => EdgeStyle {
                color: "#666".to_string(),
                width: 2.0,
                line_type: LineType::Solid,
                arrow_type: ArrowType::Standard,
            },
            EdgeType::Gradient => EdgeStyle {
                color: "#f44336".to_string(),
                width: 1.5,
                line_type: LineType::Dashed,
                arrow_type: ArrowType::Standard,
            },
        }
    }
}

impl Default for GraphVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum EdgeType {
    Forward,
    Gradient,
}

#[allow(dead_code)]
impl EdgeType {
    /// エッジタイプの文字列表現を取得
    /// Get string representation of edge type
    fn as_str(&self) -> &'static str {
        match self {
            EdgeType::Forward => "forward",
            EdgeType::Gradient => "gradient",
        }
    }
}

fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        "scalar".to_string()
    } else {
        format!("{:?}", shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_visualizer_creation() {
        let visualizer = GraphVisualizer::new();
        assert_eq!(visualizer.layout, GraphLayout::Hierarchical);
        assert_eq!(visualizer.canvas_size, (800.0, 600.0));
        assert!(visualizer.nodes.is_empty());
        assert!(visualizer.edges.is_empty());
    }
    
    #[test]
    fn test_node_creation() {
        let node = GraphNode {
            id: "test_node".to_string(),
            name: "Test Node".to_string(),
            node_type: NodeType::Variable,
            shape: vec![2, 3],
            position: (100.0, 200.0),
            style: NodeStyle {
                color: "#ffffff".to_string(),
                border_color: "#000000".to_string(),
                shape: NodeShape::Rectangle,
                size: (80.0, 40.0),
            },
        };
        
        assert_eq!(node.id, "test_node");
        assert_eq!(node.node_type, NodeType::Variable);
        assert_eq!(node.position, (100.0, 200.0));
    }
    
    #[test]
    fn test_edge_creation() {
        let edge = GraphEdge {
            from: "node1".to_string(),
            to: "node2".to_string(),
            label: Some("edge_label".to_string()),
            style: EdgeStyle {
                color: "#333".to_string(),
                width: 2.0,
                line_type: LineType::Solid,
                arrow_type: ArrowType::Standard,
            },
        };
        
        assert_eq!(edge.from, "node1");
        assert_eq!(edge.to, "node2");
        assert_eq!(edge.label, Some("edge_label".to_string()));
    }
    
    #[test]
    fn test_layout_types() {
        assert_eq!(GraphLayout::Hierarchical, GraphLayout::Hierarchical);
        assert_ne!(GraphLayout::Hierarchical, GraphLayout::Circular);
    }
    
    #[test]
    fn test_node_styles() {
        let visualizer = GraphVisualizer::new();
        
        let var_style = visualizer.get_default_node_style(&NodeType::Variable);
        assert_eq!(var_style.shape, NodeShape::Rectangle);
        
        let op_style = visualizer.get_default_node_style(&NodeType::Operation("add".to_string()));
        assert_eq!(op_style.shape, NodeShape::Ellipse);
        
        let grad_style = visualizer.get_default_node_style(&NodeType::Gradient);
        assert_eq!(grad_style.shape, NodeShape::Circle);
    }
    
    #[test]
    fn test_svg_generation() {
        let mut visualizer = GraphVisualizer::new();
        
        // テストノードを追加
        let node = GraphNode {
            id: "test".to_string(),
            name: "Test".to_string(),
            node_type: NodeType::Variable,
            shape: vec![2, 2],
            position: (400.0, 300.0),
            style: visualizer.get_default_node_style(&NodeType::Variable),
        };
        
        visualizer.add_node(node);
        
        let svg_result = visualizer.to_svg();
        assert!(svg_result.is_ok());
        
        let svg = svg_result.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Test"));
    }
}