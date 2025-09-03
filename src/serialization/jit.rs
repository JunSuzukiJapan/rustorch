//! JIT compilation system for Phase 9
//! フェーズ9用JITコンパイルシステム

use super::core::{ComputationGraph, GraphNode, Loadable, Saveable, SerializationError, SerializationResult};
use super::model_io::{load, save};
use crate::tensor::Tensor;
use num_traits::{Float, Zero};
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Script module for JIT compilation
/// JITコンパイル用スクリプトモジュール
#[derive(Debug, Clone)]
pub struct ScriptModule<T: Float> {
    pub graph: ComputationGraph<T>,
    #[allow(dead_code)]
    pub parameters: HashMap<String, Tensor<T>>,
    #[allow(dead_code)]
    pub buffers: HashMap<String, Tensor<T>>,
    pub constants: HashMap<String, T>,
    pub metadata: HashMap<String, String>,
}

impl<T: Float + FromStr + 'static> ScriptModule<T> {
    /// Create new script module
    /// 新しいスクリプトモジュールを作成
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            parameters: HashMap::new(),
            buffers: HashMap::new(),
            constants: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add parameter to module
    /// モジュールにパラメータを追加
    pub fn add_parameter(&mut self, name: String, tensor: Tensor<T>) {
        self.parameters.insert(name, tensor);
    }
    
    /// Add buffer to module
    /// モジュールにバッファを追加
    pub fn add_buffer(&mut self, name: String, tensor: Tensor<T>) {
        self.buffers.insert(name, tensor);
    }
    
    /// Add constant to module
    /// モジュールに定数を追加
    pub fn add_constant(&mut self, name: String, value: T) {
        self.constants.insert(name, value);
    }
    
    /// Execute module with given inputs
    /// 指定された入力でモジュールを実行
    pub fn forward(&self, inputs: &[Tensor<T>]) -> SerializationResult<Vec<Tensor<T>>> {
        self.execute_graph(inputs)
    }
    
    /// Execute computation graph
    /// 計算グラフを実行
    fn execute_graph(&self, inputs: &[Tensor<T>]) -> SerializationResult<Vec<Tensor<T>>> {
        let mut tensor_cache: HashMap<usize, Tensor<T>> = HashMap::new();
        
        // Initialize inputs
        for (i, input) in inputs.iter().enumerate() {
            tensor_cache.insert(i, input.clone());
        }
        
        // Execute nodes in topological order
        for node in &self.graph.nodes {
            let input_tensors: Vec<&Tensor<T>> = node.inputs
                .iter()
                .map(|&id| tensor_cache.get(&id).ok_or_else(|| {
                    SerializationError::FormatError(
                        format!("Invalid input tensor ID: {}", id)
                    )
                }))
                .collect::<Result<Vec<_>, _>>()?;
            
            let output_tensor = self.execute_operation(&node.op_type, &input_tensors, &node.attributes)?;
            
            if !node.outputs.is_empty() {
                tensor_cache.insert(node.outputs[0], output_tensor);
            }
        }
        
        // Collect outputs
        let mut outputs = Vec::new();
        for output_name in &self.graph.outputs {
            if let Ok(output_id) = output_name.parse::<usize>() {
                if let Some(tensor) = tensor_cache.get(&output_id) {
                    outputs.push(tensor.clone());
                }
            }
        }
        
        Ok(outputs)
    }
    
    /// Execute single operation
    /// 単一操作を実行
    fn execute_operation(
        &self,
        op_type: &str,
        inputs: &[&Tensor<T>],
        attributes: &HashMap<String, String>,
    ) -> SerializationResult<Tensor<T>> {
        match op_type {
            "add" => {
                if inputs.len() != 2 {
                    return Err(SerializationError::FormatError(
                        "Add operation requires 2 inputs".to_string()
                    ));
                }
                // Use manual addition for now
                let mut result = inputs[0].clone();
                result.data = &inputs[0].data + &inputs[1].data;
                Ok(result)
            }
            "mul" => {
                if inputs.len() != 2 {
                    return Err(SerializationError::FormatError(
                        "Mul operation requires 2 inputs".to_string()
                    ));
                }
                // Use manual multiplication for now
                let mut result = inputs[0].clone();
                result.data = &inputs[0].data * &inputs[1].data;
                Ok(result)
            }
            "matmul" => {
                if inputs.len() != 2 {
                    return Err(SerializationError::FormatError(
                        "MatMul operation requires 2 inputs".to_string()
                    ));
                }
                // Simplified matrix multiplication for JIT - just return first input for now
                Ok(inputs[0].clone())
            }
            "relu" => {
                if inputs.len() != 1 {
                    return Err(SerializationError::FormatError(
                        "ReLU operation requires 1 input".to_string()
                    ));
                }
                // Use manual ReLU implementation
                let mut result = inputs[0].clone();
                result.data.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
                Ok(result)
            }
            "constant" => {
                let value_str = attributes.get("value")
                    .ok_or_else(|| SerializationError::MissingField("value".to_string()))?;
                let value: T = value_str.parse()
                    .map_err(|_| SerializationError::FormatError("Invalid constant value".to_string()))?;
                
                let shape_str = attributes.get("shape")
                    .ok_or_else(|| SerializationError::MissingField("shape".to_string()))?;
                let shape: Vec<usize> = serde_json::from_str(shape_str)
                    .map_err(|e| SerializationError::FormatError(e.to_string()))?;
                
                let numel: usize = shape.iter().product();
                let data = vec![value; numel];
                Ok(Tensor::from_vec(data, shape))
            }
            _ => Err(SerializationError::UnsupportedOperation(
                format!("Unsupported operation: {}", op_type)
            ))
        }
    }
}

impl<T: Float + 'static> Saveable for ScriptModule<T> {
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        // Custom serialization without serde
        let mut buffer = Vec::new();
        
        // Serialize metadata
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| SerializationError::FormatError(e.to_string()))?;
        let metadata_bytes = metadata_json.as_bytes();
        buffer.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(metadata_bytes);
        
        // Serialize graph nodes count
        buffer.extend_from_slice(&(self.graph.nodes.len() as u32).to_le_bytes());
        
        // Serialize basic graph structure (simplified)
        buffer.extend_from_slice(&(self.graph.inputs.len() as u32).to_le_bytes());
        for input in &self.graph.inputs {
            let input_bytes = input.as_bytes();
            buffer.extend_from_slice(&(input_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(input_bytes);
        }
        
        buffer.extend_from_slice(&(self.graph.outputs.len() as u32).to_le_bytes());
        for output in &self.graph.outputs {
            let output_bytes = output.as_bytes();
            buffer.extend_from_slice(&(output_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(output_bytes);
        }
        
        Ok(buffer)
    }
    
    fn type_id(&self) -> &'static str {
        "script_module"
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = self.metadata.clone();
        meta.insert("num_parameters".to_string(), self.parameters.len().to_string());
        meta.insert("num_buffers".to_string(), self.buffers.len().to_string());
        meta.insert("num_nodes".to_string(), self.graph.nodes.len().to_string());
        meta
    }
}

impl<T: Float + FromStr + 'static> Loadable for ScriptModule<T> {
    fn load_binary(_data: &[u8]) -> SerializationResult<Self> {
        // Simplified loading - return empty module
        Ok(Self::new())
    }
    
    fn expected_type_id() -> &'static str {
        "script_module"
    }
}

/// Function wrapper for JIT compilation
/// JITコンパイル用関数ラッパー
pub type JitFunction<T> = Box<dyn Fn(&[Tensor<T>]) -> Vec<Tensor<T>>>;

/// Script function from Rust closure
/// Rustクロージャからスクリプト関数を作成
pub fn script<T, F>(func: F) -> SerializationResult<ScriptModule<T>>
where
    T: Float + FromStr + 'static,
    F: Fn(&[Tensor<T>]) -> Vec<Tensor<T>> + 'static,
{
    // For now, create a simple module that wraps the function
    // In a full implementation, this would analyze the function and create a graph
    let mut module = ScriptModule::new();
    
    // Add metadata about the scripted function
    module.metadata.insert("type".to_string(), "scripted_function".to_string());
    module.metadata.insert("created_at".to_string(), 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    );
    
    // Create a simple linear graph representing the function
    // This is a simplified implementation - real JIT would analyze the closure
    let node = GraphNode {
        id: 0,
        op_type: "function_call".to_string(),
        inputs: vec![0], // Input tensor
        outputs: vec![1], // Output tensor
        attributes: HashMap::new(),
    };
    
    module.graph.add_node(node);
    module.graph.inputs.push("input_0".to_string());
    module.graph.outputs.push("output_0".to_string());
    
    Ok(module)
}

/// Trace function execution to create JIT module
/// 関数実行をトレースしてJITモジュールを作成
pub fn trace<T, F>(
    func: F, 
    example_inputs: &[Tensor<T>]
) -> SerializationResult<ScriptModule<T>>
where
    T: Float + FromStr + 'static,
    F: Fn(&[Tensor<T>]) -> Vec<Tensor<T>> + 'static,
{
    // Execute function with example inputs to trace operations
    let outputs = func(example_inputs);
    
    let mut module = ScriptModule::new();
    
    // Add metadata about the traced function
    module.metadata.insert("type".to_string(), "traced_function".to_string());
    module.metadata.insert("input_shapes".to_string(), 
        serde_json::to_string(&example_inputs.iter()
            .map(|t| t.shape().to_vec())
            .collect::<Vec<_>>())
            .unwrap_or_default()
    );
    module.metadata.insert("output_shapes".to_string(),
        serde_json::to_string(&outputs.iter()
            .map(|t| t.shape().to_vec())
            .collect::<Vec<_>>())
            .unwrap_or_default()
    );
    
    // Create graph nodes based on traced execution
    for (i, input) in example_inputs.iter().enumerate() {
        module.graph.inputs.push(format!("input_{}", i));
        module.add_buffer(format!("traced_input_{}", i), input.clone());
    }
    
    for (i, output) in outputs.iter().enumerate() {
        module.graph.outputs.push(format!("output_{}", i));
        module.add_buffer(format!("traced_output_{}", i), output.clone());
    }
    
    // Add a traced operation node
    let node = GraphNode {
        id: 0,
        op_type: "traced_function".to_string(),
        inputs: (0..example_inputs.len()).collect(),
        outputs: (0..outputs.len()).collect(),
        attributes: HashMap::new(),
    };
    
    module.graph.add_node(node);
    
    Ok(module)
}

/// JIT compilation cache
/// JITコンパイルキャッシュ
pub struct JitCache<T: Float> {
    cached_modules: HashMap<String, ScriptModule<T>>,
    cache_dir: std::path::PathBuf,
}

impl<T: Float + FromStr + 'static> JitCache<T> {
    /// Create new JIT cache
    /// 新しいJITキャッシュを作成
    pub fn new<P: AsRef<std::path::Path>>(cache_dir: P) -> SerializationResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;
        
        Ok(Self {
            cached_modules: HashMap::new(),
            cache_dir,
        })
    }
    
    /// Get cached module
    /// キャッシュされたモジュールを取得
    pub fn get(&mut self, key: &str) -> Option<&ScriptModule<T>> {
        if !self.cached_modules.contains_key(key) {
            // Try to load from disk
            let cache_path = self.cache_dir.join(format!("{}.rustorch", key));
            if let Ok(module) = load::<_, ScriptModule<T>>(cache_path) {
                self.cached_modules.insert(key.to_string(), module);
            }
        }
        
        self.cached_modules.get(key)
    }
    
    /// Store module in cache
    /// モジュールをキャッシュに保存
    pub fn store(&mut self, key: String, module: ScriptModule<T>) -> SerializationResult<()> {
        let cache_path = self.cache_dir.join(format!("{}.rustorch", key));
        save(&module, &cache_path)?;
        self.cached_modules.insert(key, module);
        Ok(())
    }
    
    /// Clear cache
    /// キャッシュをクリア
    pub fn clear(&mut self) -> SerializationResult<()> {
        self.cached_modules.clear();
        
        // Remove cache files
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("rustorch") {
                std::fs::remove_file(entry.path())?;
            }
        }
        
        Ok(())
    }
}

/// Graph optimization utilities
/// グラフ最適化ユーティリティ
pub mod optimization {
    use super::*;
    
    /// Optimize computation graph
    /// 計算グラフを最適化
    pub fn optimize_graph<T: Float>(graph: &mut ComputationGraph<T>) -> SerializationResult<()> {
        // Dead code elimination
        eliminate_dead_nodes(graph)?;
        
        // Constant folding
        fold_constants(graph)?;
        
        // Operation fusion
        fuse_operations(graph)?;
        
        Ok(())
    }
    
    /// Remove unreachable nodes
    /// 到達不可能ノードを削除
    fn eliminate_dead_nodes<T: Float>(graph: &mut ComputationGraph<T>) -> SerializationResult<()> {
        let mut reachable = std::collections::HashSet::new();
        let mut stack = Vec::new();
        
        // Mark output nodes as reachable
        for output_name in &graph.outputs {
            if let Ok(output_id) = output_name.parse::<usize>() {
                if output_id < graph.nodes.len() {
                    stack.push(output_id);
                    reachable.insert(output_id);
                }
            }
        }
        
        // Mark all nodes reachable from outputs
        while let Some(node_id) = stack.pop() {
            if let Some(node) = graph.nodes.get(node_id) {
                for &input_id in &node.inputs {
                    if input_id < graph.nodes.len() && !reachable.contains(&input_id) {
                        reachable.insert(input_id);
                        stack.push(input_id);
                    }
                }
            }
        }
        
        // Remove unreachable nodes
        graph.nodes.retain(|node| reachable.contains(&node.id));
        
        Ok(())
    }
    
    /// Fold constant operations
    /// 定数操作を畳み込み
    fn fold_constants<T: Float>(graph: &mut ComputationGraph<T>) -> SerializationResult<()> {
        // Simplified constant folding - in practice this would be more sophisticated
        for node in &mut graph.nodes {
            if node.op_type == "constant" {
                // Mark as foldable
                node.attributes.insert("folded".to_string(), "true".to_string());
            }
        }
        Ok(())
    }
    
    /// Fuse compatible operations
    /// 互換操作を融合
    fn fuse_operations<T: Float>(graph: &mut ComputationGraph<T>) -> SerializationResult<()> {
        // Look for fusion opportunities (e.g., add + relu -> add_relu)
        let mut fused_nodes = Vec::new();
        
        for i in 0..graph.nodes.len() {
            if graph.nodes[i].op_type == "add" {
                // Look for following ReLU
                for j in i + 1..graph.nodes.len() {
                    if graph.nodes[j].op_type == "relu" && 
                       graph.nodes[j].inputs.contains(&graph.nodes[i].outputs[0]) {
                        
                        // Create fused node
                        let fused_node = GraphNode {
                            id: graph.nodes.len(),
                            op_type: "add_relu".to_string(),
                            inputs: graph.nodes[i].inputs.clone(),
                            outputs: graph.nodes[j].outputs.clone(),
                            attributes: HashMap::new(),
                        };
                        
                        fused_nodes.push((i, j, fused_node));
                        break;
                    }
                }
            }
        }
        
        // Apply fusions (would need more sophisticated implementation)
        for (_add_idx, _relu_idx, fused_node) in fused_nodes {
            // In practice, we'd remove the original nodes and add the fused one
            // This is a simplified placeholder
            graph.nodes.push(fused_node);
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_script_module_creation() {
        let module = ScriptModule::<f32>::new();
        assert!(module.parameters.is_empty());
        assert!(module.graph.nodes.is_empty());
    }
    
    #[test]
    fn test_script_module_parameters() {
        let mut module = ScriptModule::<f32>::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        
        module.add_parameter("weight".to_string(), tensor);
        assert_eq!(module.parameters.len(), 1);
        assert!(module.parameters.contains_key("weight"));
    }
    
    #[test]
    fn test_jit_cache() {
        let temp_dir = std::env::temp_dir().join("rustorch_test_cache");
        let mut cache = JitCache::<f32>::new(&temp_dir).unwrap();
        
        let module = ScriptModule::<f32>::new();
        cache.store("test_module".to_string(), module).unwrap();
        
        assert!(cache.get("test_module").is_some());
        
        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }
    
    #[test]
    fn test_script_function() {
        let module = script(|inputs: &[Tensor<f32>]| {
            if inputs.is_empty() {
                vec![]
            } else {
                vec![inputs[0].clone()]
            }
        }).unwrap();
        
        assert_eq!(module.type_id(), "script_module");
        assert!(!module.graph.nodes.is_empty());
    }
    
    #[test]
    fn test_trace_function() {
        let example_input = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let inputs = vec![example_input];
        
        let module = trace(|inputs: &[Tensor<f32>]| {
            if inputs.is_empty() {
                vec![]
            } else {
                vec![inputs[0].clone()]
            }
        }, &inputs).unwrap();
        
        assert_eq!(module.type_id(), "script_module");
        assert!(!module.graph.inputs.is_empty());
        assert!(!module.graph.outputs.is_empty());
    }
}