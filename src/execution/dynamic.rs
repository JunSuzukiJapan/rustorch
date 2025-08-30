//! Dynamic computation graph execution engine
//! 動的計算グラフ実行エンジン

use crate::autograd::graph::{ComputationGraph, GraphNode};
use crate::autograd::function::Function;
use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Weak};
use std::time::Instant;

/// Dynamic operation types for runtime execution
/// 実行時動的演算タイプ
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicOp {
    /// Matrix multiplication
    MatMul,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Mul,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Convolution operation
    Conv2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    /// Linear transformation
    Linear { in_features: usize, out_features: usize },
    /// Batch normalization
    BatchNorm { num_features: usize },
    /// Dropout
    Dropout { p: f64 },
    /// Reshape operation
    Reshape { shape: Vec<usize> },
    /// Custom operation with name
    Custom(String),
}

/// Dynamic execution node containing operation and runtime information
/// 演算と実行時情報を含む動的実行ノード
pub struct DynamicNode<T: Float + Send + Sync + 'static> {
    /// Operation type
    pub op: DynamicOp,
    /// Input node references
    pub inputs: Vec<Arc<DynamicNode<T>>>,
    /// Cached output tensor
    pub cached_output: RwLock<Option<Tensor<T>>>,
    /// Whether this node needs recomputation
    pub dirty: RwLock<bool>,
    /// Node ID for tracking
    pub id: usize,
    /// Execution time tracking
    pub execution_time: RwLock<Option<std::time::Duration>>,
    /// Memory usage tracking
    pub memory_usage: RwLock<Option<usize>>,
}

impl<T: Float + Send + Sync + 'static> DynamicNode<T> {
    /// Create a new dynamic node
    pub fn new(op: DynamicOp, inputs: Vec<Arc<DynamicNode<T>>>, id: usize) -> Arc<Self> {
        Arc::new(DynamicNode {
            op,
            inputs,
            cached_output: RwLock::new(None),
            dirty: RwLock::new(true),
            id,
            execution_time: RwLock::new(None),
            memory_usage: RwLock::new(None),
        })
    }

    /// Mark this node as dirty (needs recomputation)
    pub fn mark_dirty(&self) {
        *self.dirty.write().unwrap() = true;
        *self.cached_output.write().unwrap() = None;
    }

    /// Check if node is dirty
    pub fn is_dirty(&self) -> bool {
        *self.dirty.read().unwrap()
    }

    /// Get cached output if available
    pub fn get_cached_output(&self) -> Option<Tensor<T>> {
        self.cached_output.read().unwrap().clone()
    }

    /// Set cached output
    pub fn set_cached_output(&self, output: Tensor<T>) {
        *self.cached_output.write().unwrap() = Some(output);
        *self.dirty.write().unwrap() = false;
    }
}

/// Dynamic execution context for runtime graph management
/// 実行時グラフ管理のための動的実行コンテキスト
pub struct DynamicExecutionContext<T: Float + Send + Sync + 'static> {
    /// Current computation graph
    graph: Arc<RwLock<ComputationGraph<T>>>,
    /// Dynamic nodes for runtime execution
    dynamic_nodes: HashMap<usize, Arc<DynamicNode<T>>>,
    /// Node execution order cache
    execution_order: RwLock<Option<Vec<usize>>>,
    /// JIT compilation cache
    compiled_ops: HashMap<Vec<DynamicOp>, Arc<dyn Function<T>>>,
    /// Next node ID
    next_node_id: usize,
    /// Execution statistics
    stats: DynamicExecutionStats,
}

/// Dynamic execution statistics
/// 動的実行統計
#[derive(Debug, Default)]
pub struct DynamicExecutionStats {
    /// Total operations executed
    pub total_ops: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total execution time
    pub total_execution_time: std::time::Duration,
    /// Memory allocations
    pub memory_allocations: usize,
    /// JIT compilations performed
    pub jit_compilations: usize,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> 
    DynamicExecutionContext<T> 
{
    /// Create new dynamic execution context
    pub fn new() -> Self {
        DynamicExecutionContext {
            graph: Arc::new(RwLock::new(ComputationGraph::new())),
            dynamic_nodes: HashMap::new(),
            execution_order: RwLock::new(None),
            compiled_ops: HashMap::new(),
            next_node_id: 0,
            stats: DynamicExecutionStats::default(),
        }
    }

    /// Add a dynamic operation node
    pub fn add_operation(&mut self, op: DynamicOp, input_ids: Vec<usize>) -> RusTorchResult<usize> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        // Get input nodes
        let input_nodes: Vec<Arc<DynamicNode<T>>> = input_ids
            .iter()
            .filter_map(|&id| self.dynamic_nodes.get(&id).cloned())
            .collect();

        if input_nodes.len() != input_ids.len() {
            return Err(RusTorchError::tensor_op("Some input nodes not found"));
        }

        // Create dynamic node
        let dynamic_node = DynamicNode::new(op, input_nodes, node_id);
        self.dynamic_nodes.insert(node_id, dynamic_node);

        // Invalidate execution order cache
        *self.execution_order.write().unwrap() = None;

        Ok(node_id)
    }

    /// Add a leaf node (input/parameter)
    pub fn add_leaf(&mut self, tensor: Tensor<T>) -> RusTorchResult<usize> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        // Create leaf node
        let dynamic_node = DynamicNode::new(DynamicOp::Custom("leaf".to_string()), vec![], node_id);
        dynamic_node.set_cached_output(tensor);
        
        self.dynamic_nodes.insert(node_id, dynamic_node);

        Ok(node_id)
    }

    /// Get dynamic node by ID
    pub fn get_dynamic_node(&self, id: &usize) -> Option<&Arc<DynamicNode<T>>> {
        self.dynamic_nodes.get(id)
    }

    /// Execute the graph and return output of specified node
    pub fn execute(&mut self, output_node_id: usize) -> RusTorchResult<Tensor<T>> {
        let start_time = Instant::now();

        // Build execution order if not cached
        self.build_execution_order(output_node_id)?;

        let execution_order = self.execution_order.read().unwrap().clone()
            .ok_or_else(|| RusTorchError::tensor_op("Failed to build execution order"))?;

        // Execute nodes in order
        for &node_id in &execution_order {
            if let Some(node) = self.dynamic_nodes.get(&node_id).cloned() {
                if node.is_dirty() || node.get_cached_output().is_none() {
                    let output = self.execute_node(&node)?;
                    node.set_cached_output(output);
                    self.stats.total_ops += 1;
                } else {
                    // Cache hit
                    self.stats.cache_hit_rate = 
                        (self.stats.cache_hit_rate * (self.stats.total_ops as f64) + 1.0) / 
                        (self.stats.total_ops as f64 + 1.0);
                }
            }
        }

        // Update stats
        self.stats.total_execution_time += start_time.elapsed();

        // Get final output
        if let Some(output_node) = self.dynamic_nodes.get(&output_node_id) {
            output_node.get_cached_output()
                .ok_or_else(|| RusTorchError::tensor_op("Output node has no result"))
        } else {
            Err(RusTorchError::tensor_op("Output node not found"))
        }
    }

    /// Execute a single node
    pub fn execute_node(&self, node: &DynamicNode<T>) -> RusTorchResult<Tensor<T>> {
        let start_time = Instant::now();

        // Get input tensors
        let mut input_tensors = Vec::new();
        for input_node in &node.inputs {
            if let Some(tensor) = input_node.get_cached_output() {
                input_tensors.push(tensor);
            } else {
                return Err(RusTorchError::tensor_op(
                    format!("Input node {} has no cached output", input_node.id)
                ));
            }
        }

        // Execute operation
        let output = match &node.op {
            DynamicOp::Add => {
                if input_tensors.len() != 2 {
                    return Err(RusTorchError::tensor_op("Add requires 2 inputs"));
                }
                &input_tensors[0] + &input_tensors[1]
            },
            DynamicOp::Mul => {
                if input_tensors.len() != 2 {
                    return Err(RusTorchError::tensor_op("Mul requires 2 inputs"));
                }
                &input_tensors[0] * &input_tensors[1]
            },
            DynamicOp::MatMul => {
                if input_tensors.len() != 2 {
                    return Err(RusTorchError::tensor_op("MatMul requires 2 inputs"));
                }
                input_tensors[0].matmul(&input_tensors[1])?
            },
            DynamicOp::ReLU => {
                if input_tensors.len() != 1 {
                    return Err(RusTorchError::tensor_op("ReLU requires 1 input"));
                }
                // Use element-wise operations instead of missing relu method
                let input_data = &input_tensors[0].data;
                let relu_data: Vec<T> = input_data.iter()
                    .map(|&x| if x > T::zero() { x } else { T::zero() })
                    .collect();
                Tensor::from_vec(relu_data, input_tensors[0].shape().to_vec())
            },
            DynamicOp::Sigmoid => {
                if input_tensors.len() != 1 {
                    return Err(RusTorchError::tensor_op("Sigmoid requires 1 input"));
                }
                // Use element-wise operations for sigmoid
                let input_data = &input_tensors[0].data;
                let sigmoid_data: Vec<T> = input_data.iter()
                    .map(|&x| T::one() / (T::one() + (-x).exp()))
                    .collect();
                Tensor::from_vec(sigmoid_data, input_tensors[0].shape().to_vec())
            },
            DynamicOp::Reshape { shape } => {
                if input_tensors.len() != 1 {
                    return Err(RusTorchError::tensor_op("Reshape requires 1 input"));
                }
                input_tensors[0].reshape(shape)?
            },
            DynamicOp::Linear { in_features: _, out_features: _ } => {
                if input_tensors.len() < 2 || input_tensors.len() > 3 {
                    return Err(RusTorchError::tensor_op("Linear requires 2-3 inputs (input, weight, [bias])"));
                }
                self.execute_linear(&input_tensors)?
            },
            _ => {
                return Err(RusTorchError::tensor_op(
                    format!("Operation {:?} not implemented yet", node.op)
                ));
            },
        };

        // Record execution metrics
        let execution_time = start_time.elapsed();
        *node.execution_time.write().unwrap() = Some(execution_time);
        
        // Estimate memory usage
        let memory_usage = output.data.len() * std::mem::size_of::<T>();
        *node.memory_usage.write().unwrap() = Some(memory_usage);

        Ok(output)
    }

    /// Execute Linear operation
    fn execute_linear(&self, inputs: &[Tensor<T>]) -> RusTorchResult<Tensor<T>> {
        let input = &inputs[0];
        let weight = &inputs[1];
        let bias = inputs.get(2);

        // Matrix multiplication: input @ weight.T
        let mut output = input.matmul(&weight.transpose()?)?;

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            output = &output + bias_tensor;
        }

        Ok(output)
    }

    /// Build execution order using topological sort
    fn build_execution_order(&mut self, output_node_id: usize) -> RusTorchResult<()> {
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        self.topological_sort(output_node_id, &mut visited, &mut temp_visited, &mut order)?;
        
        *self.execution_order.write().unwrap() = Some(order);
        Ok(())
    }

    /// Topological sort for dynamic graph
    fn topological_sort(
        &self,
        node_id: usize,
        visited: &mut std::collections::HashSet<usize>,
        temp_visited: &mut std::collections::HashSet<usize>,
        order: &mut Vec<usize>,
    ) -> RusTorchResult<()> {
        if temp_visited.contains(&node_id) {
            return Err(RusTorchError::tensor_op("Circular dependency detected"));
        }

        if visited.contains(&node_id) {
            return Ok(());
        }

        temp_visited.insert(node_id);

        if let Some(node) = self.dynamic_nodes.get(&node_id) {
            for input_node in &node.inputs {
                self.topological_sort(input_node.id, visited, temp_visited, order)?;
            }
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        order.push(node_id);

        Ok(())
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> &DynamicExecutionStats {
        &self.stats
    }

    /// Clear all cached outputs and force recomputation
    pub fn clear_cache(&mut self) {
        for node in self.dynamic_nodes.values() {
            node.mark_dirty();
        }
        *self.execution_order.write().unwrap() = None;
    }

    /// Create execution plan with memory optimization
    pub fn create_execution_plan(&self, output_node_id: usize) -> RusTorchResult<ExecutionPlan<T>> {
        let mut plan = ExecutionPlan::new();
        
        // Build dependency graph
        let mut visited = std::collections::HashSet::new();
        self.build_execution_plan_recursive(output_node_id, &mut visited, &mut plan)?;
        
        // Optimize plan
        plan.optimize_memory_usage();
        plan.optimize_execution_order();
        
        Ok(plan)
    }

    /// Recursively build execution plan
    fn build_execution_plan_recursive(
        &self,
        node_id: usize,
        visited: &mut std::collections::HashSet<usize>,
        plan: &mut ExecutionPlan<T>,
    ) -> RusTorchResult<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        if let Some(node) = self.dynamic_nodes.get(&node_id) {
            // Process dependencies first
            for input_node in &node.inputs {
                self.build_execution_plan_recursive(input_node.id, visited, plan)?;
            }

            // Add this node to plan
            plan.add_operation(node_id, node.op.clone(), 
                node.inputs.iter().map(|n| n.id).collect());
            visited.insert(node_id);
        }

        Ok(())
    }
}

/// Execution plan for optimized graph execution
/// 最適化されたグラフ実行のための実行プラン
#[derive(Clone)]
pub struct ExecutionPlan<T: Float + Send + Sync + 'static> {
    /// Ordered operations
    pub operations: Vec<PlannedOperation>,
    /// Memory allocation plan
    pub memory_plan: MemoryPlan,
    /// Parallel execution opportunities
    pub parallel_groups: Vec<Vec<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

/// Planned operation with optimization metadata
/// 最適化メタデータ付きの計画された演算
#[derive(Debug, Clone)]
pub struct PlannedOperation {
    /// Node ID
    pub node_id: usize,
    /// Operation type
    pub op: DynamicOp,
    /// Input node IDs
    pub input_ids: Vec<usize>,
    /// Estimated execution time
    pub estimated_time: Option<std::time::Duration>,
    /// Memory requirements
    pub memory_requirement: usize,
    /// Can be executed in parallel with previous operations
    pub parallel_safe: bool,
}

/// Memory allocation plan
/// メモリ割り当てプラン
#[derive(Debug, Default, Clone)]
pub struct MemoryPlan {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Memory allocation schedule
    pub allocations: Vec<MemoryAllocation>,
    /// Memory reuse opportunities
    pub reuse_map: HashMap<usize, usize>,
}

/// Memory allocation entry
/// メモリ割り当てエントリ
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Operation that needs this memory
    pub operation_id: usize,
    /// Size in bytes
    pub size: usize,
    /// Lifetime (when this memory can be freed)
    pub lifetime_end: usize,
    /// Can reuse memory from previous allocation
    pub reuse_from: Option<usize>,
}

impl<T: Float + Send + Sync + 'static> ExecutionPlan<T> {
    /// Create new execution plan
    pub fn new() -> Self {
        ExecutionPlan {
            operations: Vec::new(),
            memory_plan: MemoryPlan::default(),
            parallel_groups: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add operation to plan
    pub fn add_operation(&mut self, node_id: usize, op: DynamicOp, input_ids: Vec<usize>) {
        let planned_op = PlannedOperation {
            node_id,
            op,
            input_ids,
            estimated_time: None,
            memory_requirement: 0,
            parallel_safe: false,
        };
        self.operations.push(planned_op);
    }

    /// Optimize memory usage by analyzing lifetimes
    pub fn optimize_memory_usage(&mut self) {
        // Analyze when each tensor is last used
        let mut last_use = HashMap::new();
        
        for (op_idx, op) in self.operations.iter().enumerate() {
            for &input_id in &op.input_ids {
                last_use.insert(input_id, op_idx);
            }
        }

        // Plan memory reuse
        for (op_idx, op) in self.operations.iter().enumerate() {
            let allocation = MemoryAllocation {
                operation_id: op.node_id,
                size: op.memory_requirement,
                lifetime_end: last_use.get(&op.node_id).copied().unwrap_or(op_idx),
                reuse_from: None,
            };
            self.memory_plan.allocations.push(allocation);
        }
    }

    /// Optimize execution order for parallelism
    pub fn optimize_execution_order(&mut self) {
        // Group operations that can run in parallel
        let mut current_group = Vec::new();
        
        for (idx, op) in self.operations.iter().enumerate() {
            // Check if this operation depends on any operation in current group
            let has_dependency = current_group.iter().any(|&group_idx: &usize| {
                op.input_ids.contains(&self.operations[group_idx].node_id)
            });

            if has_dependency {
                // Start new group
                if !current_group.is_empty() {
                    self.parallel_groups.push(current_group.clone());
                    current_group.clear();
                }
            }
            
            current_group.push(idx);
        }

        if !current_group.is_empty() {
            self.parallel_groups.push(current_group);
        }
    }

    /// Get estimated total execution time
    pub fn estimated_execution_time(&self) -> std::time::Duration {
        let mut total_time = std::time::Duration::default();
        
        for group in &self.parallel_groups {
            // For parallel group, take the maximum time
            let group_time = group.iter()
                .filter_map(|&idx| self.operations[idx].estimated_time)
                .max()
                .unwrap_or_default();
            total_time += group_time;
        }
        
        total_time
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.memory_plan.peak_memory
    }
}

/// JIT compilation context for dynamic operations
/// 動的演算のためのJITコンパイルコンテキスト
pub struct JitCompiler<T: Float + Send + Sync + 'static> {
    /// Compiled operation cache
    compiled_cache: HashMap<String, Arc<dyn Function<T>>>,
    /// Compilation statistics
    compilation_stats: JitStats,
}

/// JIT compilation statistics
/// JITコンパイル統計
#[derive(Debug, Default)]
pub struct JitStats {
    /// Number of compilations
    pub compilations: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Total compilation time
    pub compilation_time: std::time::Duration,
    /// Average execution speedup
    pub average_speedup: f64,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> JitCompiler<T> {
    /// Create new JIT compiler
    pub fn new() -> Self {
        JitCompiler {
            compiled_cache: HashMap::new(),
            compilation_stats: JitStats::default(),
        }
    }

    /// Compile a sequence of operations into optimized function
    pub fn compile_operations(&mut self, ops: &[DynamicOp]) -> RusTorchResult<Arc<dyn Function<T>>> {
        let ops_key = format!("{:?}", ops);
        
        if let Some(cached) = self.compiled_cache.get(&ops_key) {
            self.compilation_stats.cache_hits += 1;
            return Ok(cached.clone());
        }

        let start_time = Instant::now();
        
        // Create fused operation
        let fused_op = self.create_fused_operation(ops)?;
        
        self.compilation_stats.compilations += 1;
        self.compilation_stats.compilation_time += start_time.elapsed();
        
        let fused_fn = Arc::new(fused_op);
        self.compiled_cache.insert(ops_key, fused_fn.clone());
        
        Ok(fused_fn)
    }

    /// Create a fused operation from multiple operations
    fn create_fused_operation(&self, ops: &[DynamicOp]) -> RusTorchResult<FusedOperation<T>> {
        Ok(FusedOperation::new(ops.to_vec()))
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> &JitStats {
        &self.compilation_stats
    }
}

/// Fused operation that combines multiple operations
/// 複数の演算を組み合わせた融合演算
pub struct FusedOperation<T: Float + Send + Sync + 'static> {
    operations: Vec<DynamicOp>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> FusedOperation<T> {
    /// Create new fused operation
    pub fn new(operations: Vec<DynamicOp>) -> Self {
        FusedOperation {
            operations,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> 
    Function<T> for FusedOperation<T> 
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
        // For simplicity, just return the first input
        // In a real implementation, this would execute the fused operations
        if inputs.is_empty() {
            Tensor::zeros(&[1])
        } else {
            inputs[0].clone()
        }
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Vec<Option<Tensor<T>>> {
        // Backward pass through fused operations
        // This would require careful gradient tracking through the fused sequence
        vec![Some(grad_output.clone()); inputs.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_execution_context_creation() {
        let mut ctx = DynamicExecutionContext::<f32>::new();
        
        // Add leaf nodes
        let input1 = Tensor::zeros(&[2, 3]);
        let input2 = Tensor::ones(&[2, 3]);
        
        let leaf1_id = ctx.add_leaf(input1).unwrap();
        let leaf2_id = ctx.add_leaf(input2).unwrap();
        
        // Add operation
        let add_id = ctx.add_operation(DynamicOp::Add, vec![leaf1_id, leaf2_id]).unwrap();
        
        // Execute
        let result = ctx.execute(add_id).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_execution_plan() {
        let mut plan = ExecutionPlan::<f32>::new();
        plan.add_operation(0, DynamicOp::Add, vec![]);
        plan.add_operation(1, DynamicOp::ReLU, vec![0]);
        
        plan.optimize_execution_order();
        assert!(!plan.parallel_groups.is_empty());
    }

    #[test]
    fn test_jit_compiler() {
        let mut compiler = JitCompiler::<f32>::new();
        
        let ops = vec![DynamicOp::Add, DynamicOp::ReLU];
        let compiled = compiler.compile_operations(&ops).unwrap();
        
        // Test cache hit
        let compiled2 = compiler.compile_operations(&ops).unwrap();
        assert_eq!(compiler.get_stats().cache_hits, 1);
    }

    #[test]
    fn test_relu_operation() {
        let mut ctx = DynamicExecutionContext::<f32>::new();
        
        // Test ReLU with mixed positive/negative values
        let input_data = vec![-1.0, 0.0, 1.0, 2.0];
        let input = Tensor::from_vec(input_data, vec![4]);
        let leaf_id = ctx.add_leaf(input).unwrap();
        let relu_id = ctx.add_operation(DynamicOp::ReLU, vec![leaf_id]).unwrap();
        
        let result = ctx.execute(relu_id).unwrap();
        let expected = vec![0.0, 0.0, 1.0, 2.0];
        
        if let Some(slice) = result.as_slice() {
            for (actual, expected) in slice.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sigmoid_operation() {
        let mut ctx = DynamicExecutionContext::<f32>::new();
        
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let leaf_id = ctx.add_leaf(input).unwrap();
        let sigmoid_id = ctx.add_operation(DynamicOp::Sigmoid, vec![leaf_id]).unwrap();
        
        let result = ctx.execute(sigmoid_id).unwrap();
        
        // sigmoid(0) should be 0.5
        if let Some(slice) = result.as_slice() {
            assert!((slice[0] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear_operation() {
        let mut ctx = DynamicExecutionContext::<f32>::new();
        
        let input = Tensor::ones(&[2, 3]);
        let weight = Tensor::ones(&[4, 3]); // 3 -> 4 features
        let bias = Tensor::zeros(&[4]);
        
        let input_id = ctx.add_leaf(input).unwrap();
        let weight_id = ctx.add_leaf(weight).unwrap();
        let bias_id = ctx.add_leaf(bias).unwrap();
        
        let linear_id = ctx.add_operation(
            DynamicOp::Linear { in_features: 3, out_features: 4 },
            vec![input_id, weight_id, bias_id]
        ).unwrap();
        
        let result = ctx.execute(linear_id).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        
        // With all ones input and weight, output should be 3.0 for each element
        if let Some(slice) = result.as_slice() {
            for &value in slice {
                assert!((value - 3.0).abs() < 1e-6);
            }
        }
    }
}