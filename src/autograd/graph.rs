//! Computation graph for automatic differentiation
//! 自動微分のための計算グラフ

use super::function::Function;
use crate::tensor::Tensor;
use num_traits::Float;
use std::sync::{Arc, Weak};
use std::collections::HashMap;

/// Node in the computation graph
/// 計算グラフのノード
pub struct GraphNode<T: Float + Send + Sync + 'static> {
    /// Function that created this node
    /// このノードを作成した関数
    pub function: Option<Arc<dyn Function<T>>>,
    /// Input nodes
    /// 入力ノード
    pub inputs: Vec<Weak<GraphNode<T>>>,
    /// Gradient accumulator
    /// 勾配アキュムレータ
    pub grad: Option<Tensor<T>>,
    /// Input tensors (kept for backward pass)
    /// 入力テンソル（逆伝播のために保持）
    pub input_tensors: Vec<Tensor<T>>,
    /// Whether this node requires gradient computation
    /// このノードが勾配計算を必要とするか
    pub requires_grad: bool,
}

impl<T: Float + Send + Sync + 'static> GraphNode<T> {
    /// Create a new leaf node (no function, no inputs)
    /// 新しい葉ノードを作成（関数なし、入力なし）
    pub fn new_leaf(requires_grad: bool) -> Arc<Self> {
        Arc::new(GraphNode {
            function: None,
            inputs: Vec::new(),
            grad: None,
            input_tensors: Vec::new(),
            requires_grad,
        })
    }
    
    /// Create a new function node
    /// 新しい関数ノードを作成
    pub fn new_function(
        function: Arc<dyn Function<T>>,
        inputs: Vec<Weak<GraphNode<T>>>,
        input_tensors: Vec<Tensor<T>>,
        requires_grad: bool,
    ) -> Arc<Self> {
        Arc::new(GraphNode {
            function: Some(function),
            inputs,
            grad: None,
            input_tensors,
            requires_grad,
        })
    }
    
    /// Accumulate gradient for this node
    /// このノードの勾配を蓄積
    pub fn accumulate_grad(&mut self, grad: Tensor<T>) {
        match &mut self.grad {
            Some(existing_grad) => {
                *existing_grad = &*existing_grad + &grad;
            }
            None => {
                self.grad = Some(grad);
            }
        }
    }
    
    /// Clear gradient for this node
    /// このノードの勾配をクリア
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}

/// Computation graph for automatic differentiation
/// 自動微分のための計算グラフ
#[derive(Default)]
pub struct ComputationGraph<T: Float + Send + Sync + 'static> {
    /// Map from variable ID to graph node
    /// 変数IDからグラフノードへのマップ
    nodes: HashMap<usize, Arc<GraphNode<T>>>,
    /// Next variable ID
    /// 次の変数ID
    next_id: usize,
}

impl<T: Float + Send + Sync + 'static> ComputationGraph<T> {
    /// Create a new computation graph
    /// 新しい計算グラフを作成
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }
    
    /// Add a leaf node to the graph
    /// 葉ノードをグラフに追加
    pub fn add_leaf(&mut self, requires_grad: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let node = GraphNode::new_leaf(requires_grad);
        self.nodes.insert(id, node);
        id
    }
    
    /// Add a function node to the graph
    /// 関数ノードをグラフに追加
    pub fn add_function(
        &mut self,
        function: Arc<dyn Function<T>>,
        input_ids: Vec<usize>,
        input_tensors: Vec<Tensor<T>>,
        requires_grad: bool,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let input_nodes: Vec<Weak<GraphNode<T>>> = input_ids
            .iter()
            .filter_map(|&input_id| {
                self.nodes.get(&input_id).map(|node| Arc::downgrade(node))
            })
            .collect();
        
        let node = GraphNode::new_function(function, input_nodes, input_tensors, requires_grad);
        self.nodes.insert(id, node);
        id
    }
    
    /// Get a node by ID
    /// IDによってノードを取得
    pub fn get_node(&self, id: usize) -> Option<&Arc<GraphNode<T>>> {
        self.nodes.get(&id)
    }
    
    /// Get a mutable node by ID
    /// IDによって可変ノードを取得
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut Arc<GraphNode<T>>> {
        self.nodes.get_mut(&id)
    }
    
    /// Perform backward pass from the given node
    /// 指定されたノードから逆伝播を実行
    pub fn backward(&mut self, root_id: usize, grad_output: Option<Tensor<T>>) {
        // Initialize gradient for root node
        if let Some(_root_node_arc) = self.nodes.get(&root_id) {
            // We need to work with the node safely
            let initial_grad = grad_output.unwrap_or_else(|| Tensor::ones(&[]));
            
            // Use topological sort to traverse the graph in reverse order
            let mut visited = std::collections::HashSet::new();
            let mut stack = Vec::new();
            self.topological_sort(root_id, &mut visited, &mut stack);
            
            // Initialize root gradient
            if let Some(root_node_arc) = self.nodes.get_mut(&root_id) {
                if let Some(root_node) = Arc::get_mut(root_node_arc) {
                    root_node.accumulate_grad(initial_grad);
                }
            }
            
            // Process nodes in reverse topological order
            for &node_id in stack.iter().rev() {
                if let Some(node_arc) = self.nodes.get(&node_id).cloned() {
                    if let Some(function) = &node_arc.function {
                        if let Some(grad) = &node_arc.grad {
                            let grad_inputs = function.backward(grad, &node_arc.input_tensors.iter().collect::<Vec<_>>());
                            
                            // Propagate gradients to input nodes
                            for (i, input_weak) in node_arc.inputs.iter().enumerate() {
                                if let Some(input_arc) = input_weak.upgrade() {
                                    if let Some(grad_input) = &grad_inputs[i] {
                                        // Find the input node ID
                                        for (&input_id, input_node_arc) in &self.nodes {
                                            if Arc::ptr_eq(input_node_arc, &input_arc) {
                                                if let Some(input_node_arc_mut) = self.nodes.get_mut(&input_id) {
                                                    if let Some(input_node) = Arc::get_mut(input_node_arc_mut) {
                                                        if input_node.requires_grad {
                                                            input_node.accumulate_grad(grad_input.clone());
                                                        }
                                                    }
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Perform topological sort of the computation graph
    /// 計算グラフのトポロジカルソート
    fn topological_sort(&self, node_id: usize, visited: &mut std::collections::HashSet<usize>, stack: &mut Vec<usize>) {
        if visited.contains(&node_id) {
            return;
        }
        
        visited.insert(node_id);
        
        if let Some(node_arc) = self.nodes.get(&node_id) {
            for input_weak in &node_arc.inputs {
                if let Some(input_arc) = input_weak.upgrade() {
                    // Find the input node ID
                    for (&input_id, input_node_arc) in &self.nodes {
                        if Arc::ptr_eq(input_node_arc, &input_arc) {
                            self.topological_sort(input_id, visited, stack);
                            break;
                        }
                    }
                }
            }
        }
        
        stack.push(node_id);
    }
}