//! Gradient flow visualization for computational graphs
//! 計算グラフの勾配フロー可視化

use crate::autograd::Variable;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;

/// Gradient flow visualizer
/// 勾配フロービジュアライザー
pub struct GradientFlowVisualizer {
    /// Graph nodes
    /// グラフノード
    nodes: Vec<NodeInfo>,
    /// Graph edges
    /// グラフエッジ
    edges: Vec<EdgeInfo>,
    /// Node ID counter
    /// ノードIDカウンタ
    node_counter: usize,
    /// Visited nodes tracking
    /// 訪問済みノード追跡
    visited: HashSet<usize>,
}

/// Node information for visualization
/// 可視化用ノード情報
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node ID
    /// ノードID
    id: usize,
    /// Node label
    /// ノードラベル
    label: String,
    /// Node type (variable, operation, etc.)
    /// ノードタイプ（変数、演算など）
    node_type: NodeType,
    /// Shape information
    /// 形状情報
    shape: Vec<usize>,
    /// Gradient value if available
    /// 利用可能な場合の勾配値
    gradient_norm: Option<f32>,
    /// Requires gradient flag
    /// 勾配要求フラグ
    _requires_grad: bool,
}

/// Edge information for visualization
/// 可視化用エッジ情報
#[derive(Debug, Clone)]
struct EdgeInfo {
    /// Source node ID
    /// ソースノードID
    from: usize,
    /// Target node ID
    /// ターゲットノードID
    to: usize,
    /// Edge label
    /// エッジラベル
    label: String,
    /// Gradient flow magnitude
    /// 勾配フローの大きさ
    gradient_magnitude: Option<f32>,
}

/// Node types in the computation graph
/// 計算グラフのノードタイプ
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// Input variable
    /// 入力変数
    Input,
    /// Parameter (trainable)
    /// パラメータ（訓練可能）
    Parameter,
    /// Operation node
    /// 演算ノード
    Operation(String),
    /// Loss/output node
    /// 損失/出力ノード
    Output,
}

impl GradientFlowVisualizer {
    /// Create a new gradient flow visualizer
    /// 新しい勾配フロービジュアライザーを作成
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_counter: 0,
            visited: HashSet::new(),
        }
    }

    /// Trace gradient flow from a variable
    /// 変数から勾配フローをトレース
    pub fn trace_from_variable<T>(&mut self, var: &Variable<T>, label: &str) -> usize 
    where
        T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        // Create node for this variable
        let node_id = self.node_counter;
        self.node_counter += 1;

        // Skip if already visited
        if self.visited.contains(&node_id) {
            return node_id;
        }
        self.visited.insert(node_id);

        // Extract shape and gradient information
        let shape = var.data.read().unwrap().shape().to_vec();
        let gradient_norm = if let Ok(grad_lock) = var.grad.read() {
            grad_lock.as_ref().map(|g| {
                let sum: f32 = g.data.iter()
                    .map(|&x| x.to_f32().unwrap_or(0.0).powi(2))
                    .sum();
                sum.sqrt()
            })
        } else {
            None
        };

        // Determine node type
        let node_type = if label.contains("loss") || label.contains("output") {
            NodeType::Output
        } else if var.requires_grad {
            NodeType::Parameter
        } else {
            NodeType::Input
        };

        // Add node
        self.nodes.push(NodeInfo {
            id: node_id,
            label: label.to_string(),
            node_type,
            shape,
            gradient_norm,
            _requires_grad: var.requires_grad,
        });

        node_id
    }

    /// Add an operation node
    /// 演算ノードを追加
    pub fn add_operation(&mut self, op_name: &str, inputs: Vec<usize>, output: usize) {
        let op_id = self.node_counter;
        self.node_counter += 1;

        // Add operation node
        self.nodes.push(NodeInfo {
            id: op_id,
            label: op_name.to_string(),
            node_type: NodeType::Operation(op_name.to_string()),
            shape: Vec::new(),
            gradient_norm: None,
            _requires_grad: true,
        });

        // Add edges from inputs to operation
        for input_id in inputs {
            self.edges.push(EdgeInfo {
                from: input_id,
                to: op_id,
                label: "forward".to_string(),
                gradient_magnitude: None,
            });
        }

        // Add edge from operation to output
        self.edges.push(EdgeInfo {
            from: op_id,
            to: output,
            label: "result".to_string(),
            gradient_magnitude: None,
        });
    }

    /// Generate DOT format for Graphviz
    /// Graphviz用のDOT形式を生成
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        writeln!(&mut dot, "digraph GradientFlow {{").unwrap();
        writeln!(&mut dot, "  rankdir=TB;").unwrap();
        writeln!(&mut dot, "  node [shape=box, style=\"rounded,filled\"];").unwrap();
        writeln!(&mut dot, "  edge [fontsize=10];").unwrap();
        writeln!(&mut dot).unwrap();

        // Add nodes
        for node in &self.nodes {
            let color = match node.node_type {
                NodeType::Input => "#e8f4f8",
                NodeType::Parameter => "#fff4e6",
                NodeType::Operation(_) => "#f0f8ff",
                NodeType::Output => "#ffe6e6",
            };

            let label = if let Some(grad_norm) = node.gradient_norm {
                format!("{}\\nshape: {:?}\\ngrad_norm: {:.4}", 
                    node.label, node.shape, grad_norm)
            } else {
                format!("{}\\nshape: {:?}", node.label, node.shape)
            };

            writeln!(&mut dot, "  n{} [label=\"{}\", fillcolor=\"{}\"];",
                node.id, label, color).unwrap();
        }

        writeln!(&mut dot).unwrap();

        // Add edges
        for edge in &self.edges {
            let style = if edge.gradient_magnitude.is_some() {
                let magnitude = edge.gradient_magnitude.unwrap();
                let width = (magnitude.log10() + 2.0).max(0.5).min(3.0);
                format!("penwidth={:.1}", width)
            } else {
                "".to_string()
            };

            writeln!(&mut dot, "  n{} -> n{} [label=\"{}\", {}];",
                edge.from, edge.to, edge.label, style).unwrap();
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    /// Save visualization to file
    /// 可視化をファイルに保存
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let dot_content = self.to_dot();
        let mut file = File::create(path)?;
        file.write_all(dot_content.as_bytes())?;
        Ok(())
    }

    /// Generate a summary of gradient flow statistics
    /// 勾配フロー統計のサマリーを生成
    pub fn gradient_flow_summary(&self) -> GradientFlowSummary {
        let total_nodes = self.nodes.len();
        let parameter_nodes = self.nodes.iter()
            .filter(|n| matches!(n.node_type, NodeType::Parameter))
            .count();
        
        let nodes_with_gradients = self.nodes.iter()
            .filter(|n| n.gradient_norm.is_some())
            .count();

        let gradient_norms: Vec<f32> = self.nodes.iter()
            .filter_map(|n| n.gradient_norm)
            .collect();

        let avg_gradient_norm = if !gradient_norms.is_empty() {
            gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32
        } else {
            0.0
        };

        let max_gradient_norm = gradient_norms.iter().cloned().fold(0.0f32, f32::max);
        let min_gradient_norm = gradient_norms.iter().cloned().fold(f32::INFINITY, f32::min);

        GradientFlowSummary {
            total_nodes,
            parameter_nodes,
            nodes_with_gradients,
            avg_gradient_norm,
            max_gradient_norm,
            min_gradient_norm: if min_gradient_norm.is_finite() { min_gradient_norm } else { 0.0 },
            total_edges: self.edges.len(),
        }
    }

    /// Detect potential gradient flow issues
    /// 潜在的な勾配フローの問題を検出
    pub fn detect_issues(&self) -> Vec<GradientFlowIssue> {
        let mut issues = Vec::new();

        // Check for vanishing gradients
        for node in &self.nodes {
            if let Some(grad_norm) = node.gradient_norm {
                if grad_norm < 1e-6 && matches!(node.node_type, NodeType::Parameter) {
                    issues.push(GradientFlowIssue::VanishingGradient {
                        node_label: node.label.clone(),
                        gradient_norm: grad_norm,
                    });
                }
                
                // Check for exploding gradients
                if grad_norm > 1e3 {
                    issues.push(GradientFlowIssue::ExplodingGradient {
                        node_label: node.label.clone(),
                        gradient_norm: grad_norm,
                    });
                }
            }
        }

        // Check for disconnected parameters
        for node in &self.nodes {
            if matches!(node.node_type, NodeType::Parameter) && node.gradient_norm.is_none() {
                issues.push(GradientFlowIssue::DisconnectedParameter {
                    node_label: node.label.clone(),
                });
            }
        }

        issues
    }

    /// Clear the visualizer for reuse
    /// 再利用のためにビジュアライザーをクリア
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.visited.clear();
        self.node_counter = 0;
    }
}

/// Summary of gradient flow statistics
/// 勾配フロー統計のサマリー
#[derive(Debug, Clone)]
pub struct GradientFlowSummary {
    /// Total number of nodes
    /// ノードの総数
    pub total_nodes: usize,
    /// Number of parameter nodes
    /// パラメータノードの数
    pub parameter_nodes: usize,
    /// Number of nodes with gradients
    /// 勾配を持つノードの数
    pub nodes_with_gradients: usize,
    /// Average gradient norm
    /// 平均勾配ノルム
    pub avg_gradient_norm: f32,
    /// Maximum gradient norm
    /// 最大勾配ノルム
    pub max_gradient_norm: f32,
    /// Minimum gradient norm
    /// 最小勾配ノルム
    pub min_gradient_norm: f32,
    /// Total number of edges
    /// エッジの総数
    pub total_edges: usize,
}

impl std::fmt::Display for GradientFlowSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gradient Flow Summary:")?;
        writeln!(f, "  Total nodes: {}", self.total_nodes)?;
        writeln!(f, "  Parameter nodes: {}", self.parameter_nodes)?;
        writeln!(f, "  Nodes with gradients: {}", self.nodes_with_gradients)?;
        writeln!(f, "  Average gradient norm: {:.6}", self.avg_gradient_norm)?;
        writeln!(f, "  Max gradient norm: {:.6}", self.max_gradient_norm)?;
        writeln!(f, "  Min gradient norm: {:.6}", self.min_gradient_norm)?;
        writeln!(f, "  Total edges: {}", self.total_edges)?;
        Ok(())
    }
}

/// Gradient flow issues detected
/// 検出された勾配フローの問題
#[derive(Debug, Clone)]
pub enum GradientFlowIssue {
    /// Vanishing gradient detected
    /// 勾配消失が検出された
    VanishingGradient {
        /// Node label
        node_label: String,
        /// Gradient norm value
        gradient_norm: f32,
    },
    /// Exploding gradient detected
    /// 勾配爆発が検出された
    ExplodingGradient {
        /// Node label
        node_label: String,
        /// Gradient norm value
        gradient_norm: f32,
    },
    /// Disconnected parameter (no gradient)
    /// 切断されたパラメータ（勾配なし）
    DisconnectedParameter {
        /// Node label
        node_label: String,
    },
}

impl std::fmt::Display for GradientFlowIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradientFlowIssue::VanishingGradient { node_label, gradient_norm } => {
                write!(f, "Vanishing gradient in '{}': norm = {:.2e}", node_label, gradient_norm)
            }
            GradientFlowIssue::ExplodingGradient { node_label, gradient_norm } => {
                write!(f, "Exploding gradient in '{}': norm = {:.2e}", node_label, gradient_norm)
            }
            GradientFlowIssue::DisconnectedParameter { node_label } => {
                write!(f, "Disconnected parameter '{}': no gradient computed", node_label)
            }
        }
    }
}

/// Interactive gradient flow analyzer
/// インタラクティブ勾配フロー解析器
pub struct GradientFlowAnalyzer {
    /// History of gradient norms
    /// 勾配ノルムの履歴
    gradient_history: HashMap<String, Vec<f32>>,
    /// Maximum history length
    /// 最大履歴長
    max_history_length: usize,
}

impl GradientFlowAnalyzer {
    /// Create a new gradient flow analyzer
    /// 新しい勾配フロー解析器を作成
    pub fn new(max_history_length: usize) -> Self {
        Self {
            gradient_history: HashMap::new(),
            max_history_length,
        }
    }

    /// Record gradient norm for a parameter
    /// パラメータの勾配ノルムを記録
    pub fn record_gradient<T>(&mut self, name: &str, tensor: &Tensor<T>)
    where
        T: num_traits::Float,
    {
        let norm = tensor.data.iter()
            .map(|&x| x.to_f32().unwrap_or(0.0).powi(2))
            .sum::<f32>()
            .sqrt();

        let history = self.gradient_history.entry(name.to_string()).or_insert_with(Vec::new);
        history.push(norm);
        
        // Maintain maximum history length
        if history.len() > self.max_history_length {
            history.remove(0);
        }
    }

    /// Get gradient history for a parameter
    /// パラメータの勾配履歴を取得
    pub fn get_history(&self, name: &str) -> Option<&Vec<f32>> {
        self.gradient_history.get(name)
    }

    /// Analyze gradient trends
    /// 勾配トレンドを分析
    pub fn analyze_trends(&self) -> HashMap<String, GradientTrend> {
        let mut trends = HashMap::new();

        for (name, history) in &self.gradient_history {
            if history.len() < 2 {
                continue;
            }

            let recent_avg = history[history.len().saturating_sub(10)..]
                .iter()
                .sum::<f32>() / history[history.len().saturating_sub(10)..].len() as f32;

            let overall_avg = history.iter().sum::<f32>() / history.len() as f32;

            let trend = if recent_avg < overall_avg * 0.1 {
                GradientTrend::Vanishing
            } else if recent_avg > overall_avg * 10.0 {
                GradientTrend::Exploding
            } else if (recent_avg - overall_avg).abs() < overall_avg * 0.1 {
                GradientTrend::Stable
            } else if recent_avg > overall_avg {
                GradientTrend::Increasing
            } else {
                GradientTrend::Decreasing
            };

            trends.insert(name.clone(), trend);
        }

        trends
    }

    /// Clear history
    /// 履歴をクリア
    pub fn clear(&mut self) {
        self.gradient_history.clear();
    }
}

/// Gradient trend analysis result
/// 勾配トレンド分析結果
#[derive(Debug, Clone, PartialEq)]
pub enum GradientTrend {
    /// Gradient is stable
    Stable,
    /// Gradient is increasing
    Increasing,
    /// Gradient is decreasing
    Decreasing,
    /// Gradient is vanishing
    Vanishing,
    /// Gradient is exploding
    Exploding,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_flow_visualizer() {
        let mut visualizer = GradientFlowVisualizer::new();
        
        // Add some nodes
        let input_id = visualizer.node_counter;
        visualizer.node_counter += 1;
        visualizer.nodes.push(NodeInfo {
            id: input_id,
            label: "input".to_string(),
            node_type: NodeType::Input,
            shape: vec![32, 10],
            gradient_norm: None,
            _requires_grad: false,
        });

        let param_id = visualizer.node_counter;
        visualizer.node_counter += 1;
        visualizer.nodes.push(NodeInfo {
            id: param_id,
            label: "weight".to_string(),
            node_type: NodeType::Parameter,
            shape: vec![10, 5],
            gradient_norm: Some(0.5),
            _requires_grad: true,
        });

        // Add operation
        visualizer.add_operation("matmul", vec![input_id, param_id], 2);

        // Generate DOT
        let dot = visualizer.to_dot();
        assert!(dot.contains("digraph GradientFlow"));
        assert!(dot.contains("weight"));
        assert!(dot.contains("matmul"));
    }

    #[test]
    fn test_gradient_flow_summary() {
        let mut visualizer = GradientFlowVisualizer::new();
        
        // Add nodes with different gradient norms
        for i in 0..5 {
            visualizer.nodes.push(NodeInfo {
                id: i,
                label: format!("param_{}", i),
                node_type: NodeType::Parameter,
                shape: vec![10, 10],
                gradient_norm: Some((i + 1) as f32 * 0.1),
                _requires_grad: true,
            });
        }

        let summary = visualizer.gradient_flow_summary();
        assert_eq!(summary.total_nodes, 5);
        assert_eq!(summary.parameter_nodes, 5);
        assert_eq!(summary.nodes_with_gradients, 5);
        assert!(summary.avg_gradient_norm > 0.0);
    }

    #[test]
    fn test_issue_detection() {
        let mut visualizer = GradientFlowVisualizer::new();
        
        // Add node with vanishing gradient
        visualizer.nodes.push(NodeInfo {
            id: 0,
            label: "vanishing_param".to_string(),
            node_type: NodeType::Parameter,
            shape: vec![10],
            gradient_norm: Some(1e-7),
            _requires_grad: true,
        });

        // Add node with exploding gradient
        visualizer.nodes.push(NodeInfo {
            id: 1,
            label: "exploding_param".to_string(),
            node_type: NodeType::Parameter,
            shape: vec![10],
            gradient_norm: Some(1e4),
            _requires_grad: true,
        });

        let issues = visualizer.detect_issues();
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn test_gradient_analyzer() {
        let mut analyzer = GradientFlowAnalyzer::new(100);
        
        // Simulate gradient recording
        let _tensor = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);
        
        for i in 0..20 {
            let scaled = Tensor::from_vec(
                vec![0.1 * (i as f32 + 1.0), 0.2 * (i as f32 + 1.0), 0.3 * (i as f32 + 1.0)],
                vec![3]
            );
            analyzer.record_gradient("weight", &scaled);
        }

        let history = analyzer.get_history("weight").unwrap();
        assert_eq!(history.len(), 20);

        let trends = analyzer.analyze_trends();
        assert!(trends.contains_key("weight"));
    }
}