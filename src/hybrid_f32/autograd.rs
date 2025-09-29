//! f32統一ハイブリッド自動微分システム
//! f32 Unified Hybrid Automatic Differentiation System
//!
//! フェーズ5: 高度自動微分機能
//! Phase 5: Advanced Automatic Differentiation Features
//!
//! このモジュールは、f32精度での高効率自動微分を提供し、
//! Neural Engine、Metal GPU、CPUでの統一実行をサポートします。

use crate::error::RusTorchResult;
use crate::hybrid_f32::tensor::core::F32Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 計算グラフのノード識別子
/// Computation graph node identifier
pub type NodeId = usize;

/// 計算グラフのノード
/// Computation graph node
#[derive(Debug)]
pub struct ComputationNode {
    pub id: NodeId,
    pub operation: String,
    pub inputs: Vec<NodeId>,
    pub gradient_function: Option<Box<dyn GradientFunction>>,
}

/// 勾配計算関数のトレイト
/// Trait for gradient computation functions
pub trait GradientFunction: std::fmt::Debug + Send + Sync {
    /// 勾配を計算
    /// Compute gradient
    fn compute_gradient(
        &self,
        grad_output: &F32Tensor,
        inputs: &[&F32Tensor],
    ) -> RusTorchResult<Vec<F32Tensor>>;

    /// 関数名を取得
    /// Get function name
    fn name(&self) -> &str;
}

/// 加算の勾配関数
/// Addition gradient function
#[derive(Debug)]
pub struct AddBackward;

impl GradientFunction for AddBackward {
    fn compute_gradient(
        &self,
        grad_output: &F32Tensor,
        _inputs: &[&F32Tensor],
    ) -> RusTorchResult<Vec<F32Tensor>> {
        // 加算の勾配は入力にそのまま伝播
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }

    fn name(&self) -> &str {
        "AddBackward"
    }
}

/// 乗算の勾配関数
/// Multiplication gradient function
#[derive(Debug)]
pub struct MulBackward;

impl GradientFunction for MulBackward {
    fn compute_gradient(
        &self,
        grad_output: &F32Tensor,
        inputs: &[&F32Tensor],
    ) -> RusTorchResult<Vec<F32Tensor>> {
        if inputs.len() != 2 {
            return Err("MulBackward requires exactly 2 inputs".into());
        }

        // 乗算の勾配: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        let grad_a = grad_output.mul(inputs[1])?;
        let grad_b = grad_output.mul(inputs[0])?;

        Ok(vec![grad_a, grad_b])
    }

    fn name(&self) -> &str {
        "MulBackward"
    }
}

/// 行列乗算の勾配関数
/// Matrix multiplication gradient function
#[derive(Debug)]
pub struct MatMulBackward;

impl GradientFunction for MatMulBackward {
    fn compute_gradient(
        &self,
        grad_output: &F32Tensor,
        inputs: &[&F32Tensor],
    ) -> RusTorchResult<Vec<F32Tensor>> {
        if inputs.len() != 2 {
            return Err("MatMulBackward requires exactly 2 inputs".into());
        }

        // 行列乗算の勾配: ∂(A@B)/∂A = grad_output @ B.T, ∂(A@B)/∂B = A.T @ grad_output
        let grad_a = grad_output.matmul(&inputs[1].transpose()?)?;
        let grad_b = inputs[0].transpose()?.matmul(grad_output)?;

        Ok(vec![grad_a, grad_b])
    }

    fn name(&self) -> &str {
        "MatMulBackward"
    }
}

/// ReLUの勾配関数
/// ReLU gradient function
#[derive(Debug)]
pub struct ReLUBackward {
    pub input: F32Tensor,
}

impl GradientFunction for ReLUBackward {
    fn compute_gradient(
        &self,
        grad_output: &F32Tensor,
        _inputs: &[&F32Tensor],
    ) -> RusTorchResult<Vec<F32Tensor>> {
        // ReLUの勾配: x > 0 なら 1, x <= 0 なら 0
        let zero = F32Tensor::zeros(self.input.shape())?;
        let mask = self.input.gt(&zero)?;
        let grad_input = grad_output.mul(&mask)?;

        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "ReLUBackward"
    }
}

/// f32自動微分対応テンソル
/// f32 tensor with automatic differentiation support
#[derive(Debug)]
pub struct F32Variable {
    pub data: F32Tensor,
    pub grad: Option<F32Tensor>,
    pub requires_grad: bool,
    pub node_id: Option<NodeId>,
    pub is_leaf: bool,
}

impl F32Variable {
    /// 新しい変数を作成
    /// Create new variable
    pub fn new(data: F32Tensor, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
            node_id: None,
            is_leaf: true,
        }
    }

    /// 勾配をゼロに初期化
    /// Zero gradients
    pub fn zero_grad(&mut self) -> RusTorchResult<()> {
        if self.requires_grad {
            self.grad = Some(F32Tensor::zeros(self.data.shape())?);
        }
        Ok(())
    }

    /// 勾配を累積
    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: &F32Tensor) -> RusTorchResult<()> {
        if self.requires_grad {
            match &mut self.grad {
                Some(existing_grad) => {
                    *existing_grad = existing_grad.add(grad)?;
                }
                None => {
                    self.grad = Some(grad.clone());
                }
            }
        }
        Ok(())
    }

    /// デタッチ（勾配計算を停止）
    /// Detach (stop gradient computation)
    pub fn detach(&self) -> RusTorchResult<F32Variable> {
        Ok(F32Variable::new(self.data.clone(), false))
    }

    /// 形状を取得
    /// Get shape
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 要素数を取得
    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// スライスとして取得
    /// Get as slice
    pub fn as_slice(&self) -> Option<&[f32]> {
        Some(self.data.as_slice())
    }
}
impl Clone for F32Variable {
    fn clone(&self) -> Self {
        // Note: This clone() method for F32Variable does not return a Result
        // We handle F32Tensor's clone() which returns Result<F32Tensor, Error>
        F32Variable {
            data: self.data.clone(),
            grad: self.grad.as_ref().map(|g| g.clone()),
            requires_grad: self.requires_grad,
            node_id: self.node_id,
            is_leaf: self.is_leaf,
        }
    }
}

/// 自動微分エンジン
/// Automatic differentiation engine
pub struct F32AutogradEngine {
    computation_graph: HashMap<NodeId, ComputationNode>,
    next_node_id: NodeId,
    topological_order: Vec<NodeId>,
}

impl F32AutogradEngine {
    /// 新しい自動微分エンジンを作成
    /// Create new autograd engine
    pub fn new() -> Self {
        Self {
            computation_graph: HashMap::new(),
            next_node_id: 0,
            topological_order: Vec::new(),
        }
    }

    /// 新しいノードを追加
    /// Add new node
    pub fn add_node(
        &mut self,
        operation: String,
        inputs: Vec<NodeId>,
        gradient_function: Option<Box<dyn GradientFunction>>,
    ) -> NodeId {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = ComputationNode {
            id: node_id,
            operation,
            inputs,
            gradient_function,
        };

        self.computation_graph.insert(node_id, node);
        node_id
    }

    /// トポロジカルソートを実行
    /// Perform topological sort
    pub fn topological_sort(&mut self, root_node: NodeId) {
        let mut visited = std::collections::HashSet::new();
        let mut temp_mark = std::collections::HashSet::new();
        self.topological_order.clear();

        self.dfs_topological_sort(root_node, &mut visited, &mut temp_mark);
        self.topological_order.reverse();
    }

    fn dfs_topological_sort(
        &mut self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        temp_mark: &mut std::collections::HashSet<NodeId>,
    ) {
        if temp_mark.contains(&node_id) {
            return; // 循環検出
        }
        if visited.contains(&node_id) {
            return;
        }

        temp_mark.insert(node_id);

        if let Some(node) = self.computation_graph.get(&node_id) {
            let inputs = node.inputs.clone();
            for &input_id in &inputs {
                self.dfs_topological_sort(input_id, visited, temp_mark);
            }
        }

        temp_mark.remove(&node_id);
        visited.insert(node_id);
        self.topological_order.push(node_id);
    }

    /// 逆伝播を実行
    /// Perform backward pass
    pub fn backward(
        &self,
        variables: &mut HashMap<NodeId, F32Variable>,
        root_grad: &F32Tensor,
        root_node: NodeId,
    ) -> RusTorchResult<()> {
        let mut gradients: HashMap<NodeId, F32Tensor> = HashMap::new();
        gradients.insert(root_node, root_grad.clone());

        // トポロジカル順序の逆順で勾配を計算
        for &node_id in self.topological_order.iter().rev() {
            if let Some(grad_output) = gradients.get(&node_id) {
                if let Some(node) = self.computation_graph.get(&node_id) {
                    if let Some(ref grad_fn) = node.gradient_function {
                        // 入力テンソルを取得
                        let mut input_tensors = Vec::new();
                        for &input_id in &node.inputs {
                            if let Some(variable) = variables.get(&input_id) {
                                input_tensors.push(variable.data.clone());
                            }
                        }

                        let input_refs: Vec<&F32Tensor> = input_tensors.iter().collect();

                        // 勾配を計算
                        let input_grads = grad_fn.compute_gradient(grad_output, &input_refs)?;

                        // 各入力に勾配を累積
                        for (i, &input_id) in node.inputs.iter().enumerate() {
                            if i < input_grads.len() {
                                match gradients.get_mut(&input_id) {
                                    Some(existing_grad) => {
                                        *existing_grad = existing_grad.add(&input_grads[i])?;
                                    }
                                    None => {
                                        gradients.insert(input_id, input_grads[i].clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 変数に勾配を設定
        for (node_id, grad) in gradients {
            if let Some(var) = variables.get_mut(&node_id) {
                var.accumulate_grad(&grad)?;
            }
        }

        Ok(())
    }
}

/// f32自動微分コンテキスト
/// f32 autograd context
pub struct F32AutogradContext {
    engine: Arc<Mutex<F32AutogradEngine>>,
    variables: Arc<Mutex<HashMap<NodeId, F32Variable>>>,
    grad_enabled: bool,
}

impl F32AutogradContext {
    /// 新しいコンテキストを作成
    /// Create new context
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(F32AutogradEngine::new())),
            variables: Arc::new(Mutex::new(HashMap::new())),
            grad_enabled: true,
        }
    }

    /// 勾配計算を有効/無効にする
    /// Enable/disable gradient computation
    pub fn set_grad_enabled(&mut self, enabled: bool) {
        self.grad_enabled = enabled;
    }

    /// 勾配計算が有効かどうか
    /// Check if gradient computation is enabled
    pub fn is_grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// 変数を登録
    /// Register variable
    pub fn register_variable(&self, mut variable: F32Variable) -> RusTorchResult<NodeId> {
        let mut engine = self.engine.lock().unwrap();
        let mut variables = self.variables.lock().unwrap();

        let node_id = engine.add_node("leaf".to_string(), vec![], None);
        variable.node_id = Some(node_id);
        variables.insert(node_id, variable);

        Ok(node_id)
    }

    /// 演算を実行して新しい変数を作成
    /// Execute operation and create new variable
    pub fn execute_operation(
        &self,
        operation: &str,
        inputs: Vec<NodeId>,
        output_data: F32Tensor,
        gradient_function: Option<Box<dyn GradientFunction>>,
    ) -> RusTorchResult<F32Variable> {
        let mut engine = self.engine.lock().unwrap();
        let mut variables = self.variables.lock().unwrap();

        let requires_grad = self.grad_enabled
            && inputs
                .iter()
                .any(|&id| variables.get(&id).map_or(false, |v| v.requires_grad));

        let node_id = engine.add_node(operation.to_string(), inputs, gradient_function);

        let mut output_variable = F32Variable::new(output_data, requires_grad);
        output_variable.node_id = Some(node_id);
        output_variable.is_leaf = false;

        variables.insert(node_id, output_variable.clone());

        Ok(output_variable)
    }

    /// 逆伝播を実行
    /// Execute backward pass
    pub fn backward(&self, variable: &F32Variable) -> RusTorchResult<()> {
        if let Some(node_id) = variable.node_id {
            let mut engine = self.engine.lock().unwrap();
            let mut variables = self.variables.lock().unwrap();

            engine.topological_sort(node_id);

            let ones = F32Tensor::ones(variable.data.shape())?;
            engine.backward(&mut variables, &ones, node_id)?;
        }

        Ok(())
    }
}

// グローバル自動微分コンテキスト
// Global autograd context
lazy_static::lazy_static! {
    /// グローバル自動微分コンテキスト / Global autograd context
    pub static ref GLOBAL_AUTOGRAD_CONTEXT: Arc<Mutex<F32AutogradContext>> =
        Arc::new(Mutex::new(F32AutogradContext::new()));
}

/// no_grad コンテキストマネージャー
/// no_grad context manager
pub struct NoGrad {
    previous_state: bool,
}

impl NoGrad {
    pub fn new() -> Self {
        let mut context = GLOBAL_AUTOGRAD_CONTEXT.lock().unwrap();
        let previous_state = context.is_grad_enabled();
        context.set_grad_enabled(false);

        Self { previous_state }
    }
}

impl Drop for NoGrad {
    fn drop(&mut self) {
        let mut context = GLOBAL_AUTOGRAD_CONTEXT.lock().unwrap();
        context.set_grad_enabled(self.previous_state);
    }
}

/// 勾配計算無効化マクロ
/// Disable gradient computation macro
#[macro_export]
macro_rules! no_grad {
    ($block:block) => {{
        let _guard = $crate::hybrid_f32::autograd::NoGrad::new();
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_variable_creation() -> RusTorchResult<()> {
        let data = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let var = F32Variable::new(data, true);

        assert!(var.requires_grad);
        assert!(var.is_leaf);
        assert_eq!(var.shape(), &[3]);

        Ok(())
    }

    #[test]
    fn test_gradient_functions() -> RusTorchResult<()> {
        let a = F32Tensor::from_vec(vec![2.0, 3.0], &[2])?;
        let b = F32Tensor::from_vec(vec![4.0, 5.0], &[2])?;
        let grad_output = F32Tensor::from_vec(vec![1.0, 1.0], &[2])?;

        let mul_backward = MulBackward;
        let grads = mul_backward.compute_gradient(&grad_output, &[&a, &b])?;

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_slice(), &[4.0, 5.0][..]); // grad_a = grad_output * b
        assert_eq!(grads[1].as_slice(), &[2.0, 3.0][..]); // grad_b = grad_output * a

        Ok(())
    }

    #[test]
    fn test_autograd_context() -> RusTorchResult<()> {
        let context = F32AutogradContext::new();

        let data = F32Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let var = F32Variable::new(data, true);

        let _node_id = context.register_variable(var)?;

        assert!(context.is_grad_enabled());

        Ok(())
    }

    #[test]
    fn test_no_grad_macro() -> RusTorchResult<()> {
        let context = GLOBAL_AUTOGRAD_CONTEXT.lock().unwrap();
        assert!(context.is_grad_enabled());
        drop(context);

        no_grad!({
            let context = GLOBAL_AUTOGRAD_CONTEXT.lock().unwrap();
            assert!(!context.is_grad_enabled());
        });

        let context = GLOBAL_AUTOGRAD_CONTEXT.lock().unwrap();
        assert!(context.is_grad_enabled());

        Ok(())
    }
}
