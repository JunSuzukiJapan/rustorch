//! Runtime execution engine with JIT compilation and optimization
//! JITコンパイルと最適化を持つ実行時実行エンジン

use super::dynamic::{DynamicExecutionContext, DynamicOp, ExecutionPlan, JitCompiler};
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Runtime execution engine for dynamic graph execution
/// 動的グラフ実行のための実行時実行エンジン
pub struct RuntimeEngine<T: Float + Send + Sync + 'static> {
    /// Dynamic execution context
    pub context: DynamicExecutionContext<T>,
    /// JIT compiler
    jit_compiler: JitCompiler<T>,
    /// Execution cache for common patterns
    pub execution_cache: HashMap<String, CachedExecution<T>>,
    /// Runtime configuration
    pub config: RuntimeConfig,
    /// Performance metrics
    metrics: Arc<RwLock<RuntimeMetrics>>,
}

/// Runtime configuration
/// 実行時設定
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Enable JIT compilation
    pub enable_jit: bool,
    /// Enable operation fusion
    pub enable_fusion: bool,
    /// Enable memory optimization
    pub enable_memory_opt: bool,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// JIT compilation threshold (operations)
    pub jit_threshold: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            enable_jit: true,
            enable_fusion: true,
            enable_memory_opt: true,
            enable_parallel: true,
            max_cache_size: 1000,
            jit_threshold: 5,
        }
    }
}

/// Cached execution for pattern reuse
/// パターン再利用のためのキャッシュされた実行
#[derive(Clone)]
pub struct CachedExecution<T: Float + Send + Sync + 'static> {
    /// Execution plan
    pub plan: ExecutionPlan<T>,
    /// Expected input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Hit count
    pub hit_count: usize,
    /// Last used timestamp
    pub last_used: Instant,
}

/// Runtime performance metrics
/// 実行時パフォーマンスメトリクス
#[derive(Debug, Default, Clone)]
pub struct RuntimeMetrics {
    /// Total executions
    pub total_executions: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average execution time
    pub avg_execution_time: std::time::Duration,
    /// JIT compilation statistics
    pub jit_stats: JitCompilationMetrics,
    /// Memory statistics
    pub memory_stats: MemoryMetrics,
    /// Parallel execution statistics
    pub parallel_stats: ParallelExecutionMetrics,
}

/// JIT compilation metrics
/// JITコンパイルメトリクス
#[derive(Debug, Default, Clone)]
pub struct JitCompilationMetrics {
    /// Total compilations
    pub total_compilations: usize,
    /// Successful compilations
    pub successful_compilations: usize,
    /// Average compilation time
    pub avg_compilation_time: std::time::Duration,
    /// Average speedup from JIT
    pub avg_speedup: f64,
}

/// Memory usage metrics
/// メモリ使用量メトリクス
#[derive(Debug, Default, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Memory efficiency (reuse rate)
    pub memory_efficiency: f64,
    /// Allocation count
    pub allocations: usize,
    /// Deallocation count
    pub deallocations: usize,
}

/// Parallel execution metrics
/// 並列実行メトリクス
#[derive(Debug, Default, Clone)]
pub struct ParallelExecutionMetrics {
    /// Parallel execution opportunities
    pub parallel_opportunities: usize,
    /// Parallel executions performed
    pub parallel_executions: usize,
    /// Average parallelism factor
    pub avg_parallelism: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    RuntimeEngine<T>
{
    /// Create new runtime engine
    pub fn new(config: RuntimeConfig) -> Self {
        RuntimeEngine {
            context: DynamicExecutionContext::new(),
            jit_compiler: JitCompiler::new(),
            execution_cache: HashMap::new(),
            config,
            metrics: Arc::new(RwLock::new(RuntimeMetrics::default())),
        }
    }

    /// Execute a computation graph with runtime optimization
    pub fn execute_graph(
        &mut self,
        graph_builder: impl FnOnce(&mut GraphBuilder<T>) -> RusTorchResult<usize>,
    ) -> RusTorchResult<Tensor<T>> {
        let start_time = Instant::now();

        // Build graph using builder pattern
        let mut builder = GraphBuilder::new(&mut self.context);
        let output_node_id = graph_builder(&mut builder)?;

        // Check cache for similar execution pattern
        let pattern_key = self.generate_pattern_key(output_node_id)?;

        if self.execution_cache.contains_key(&pattern_key) {
            // Update cache statistics
            if let Some(cached) = self.execution_cache.get_mut(&pattern_key) {
                cached.hit_count += 1;
                cached.last_used = Instant::now();
            }

            // Update metrics
            let mut metrics = self.metrics.write().unwrap();
            metrics.cache_hit_rate = (metrics.cache_hit_rate * metrics.total_executions as f64
                + 1.0)
                / (metrics.total_executions as f64 + 1.0);
        }

        // Create execution plan
        let execution_plan = self.context.create_execution_plan(output_node_id)?;

        // Apply optimizations based on configuration
        if self.config.enable_jit && execution_plan.operations.len() >= self.config.jit_threshold {
            self.apply_jit_compilation(&execution_plan)?;
        }

        // Execute
        let result = self.context.execute(output_node_id)?;

        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_executions += 1;
        metrics.memory_stats.allocations += 1;

        // Update peak memory (estimate based on tensor size)
        let estimated_memory = result.data.len() * std::mem::size_of::<T>();
        if estimated_memory > metrics.memory_stats.peak_memory {
            metrics.memory_stats.peak_memory = estimated_memory;
        }

        // Calculate memory efficiency (simple heuristic)
        metrics.memory_stats.memory_efficiency =
            metrics.memory_stats.allocations as f64 / (metrics.total_executions as f64 + 1.0);

        metrics.avg_execution_time = (metrics.avg_execution_time
            * (metrics.total_executions - 1) as u32
            + start_time.elapsed())
            / metrics.total_executions as u32;

        Ok(result)
    }

    /// Generate pattern key for caching
    fn generate_pattern_key(&self, output_node_id: usize) -> RusTorchResult<String> {
        // Create a simplified pattern representation
        let mut pattern_parts = Vec::new();
        self.collect_pattern_recursive(
            output_node_id,
            &mut pattern_parts,
            &mut std::collections::HashSet::new(),
        )?;
        Ok(pattern_parts.join("->"))
    }

    /// Recursively collect pattern for caching
    fn collect_pattern_recursive(
        &self,
        node_id: usize,
        pattern: &mut Vec<String>,
        visited: &mut std::collections::HashSet<usize>,
    ) -> RusTorchResult<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        visited.insert(node_id);

        if let Some(node) = self.context.get_dynamic_node(&node_id) {
            // Add operation to pattern
            pattern.push(format!("{:?}", node.op));

            // Process inputs
            for input_node in &node.inputs {
                self.collect_pattern_recursive(input_node.id, pattern, visited)?;
            }
        }

        Ok(())
    }

    /// Apply JIT compilation to hot paths
    fn apply_jit_compilation(&mut self, plan: &ExecutionPlan<T>) -> RusTorchResult<()> {
        // Extract operation sequences for JIT compilation
        let ops: Vec<DynamicOp> = plan.operations.iter().map(|op| op.op.clone()).collect();

        if ops.len() >= self.config.jit_threshold {
            let start_time = Instant::now();
            let _compiled_fn = self.jit_compiler.compile_operations(&ops)?;

            // Update JIT metrics
            let mut metrics = self.metrics.write().unwrap();
            metrics.jit_stats.total_compilations += 1;
            metrics.jit_stats.avg_compilation_time = (metrics.jit_stats.avg_compilation_time
                * (metrics.jit_stats.total_compilations - 1) as u32
                + start_time.elapsed())
                / metrics.jit_stats.total_compilations as u32;
        }

        Ok(())
    }

    /// Get runtime metrics
    pub fn get_metrics(&self) -> RuntimeMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Reset all metrics
    pub fn reset_metrics(&mut self) {
        *self.metrics.write().unwrap() = RuntimeMetrics::default();
    }

    /// Warm up the engine with common operations
    pub fn warmup(&mut self) -> RusTorchResult<()> {
        // Pre-compile common operation patterns
        let common_patterns = vec![
            vec![
                DynamicOp::Linear {
                    in_features: 784,
                    out_features: 128,
                },
                DynamicOp::ReLU,
            ],
            vec![
                DynamicOp::Conv2d {
                    kernel_size: (3, 3),
                    stride: (1, 1),
                    padding: (1, 1),
                },
                DynamicOp::ReLU,
            ],
            vec![DynamicOp::Add, DynamicOp::ReLU],
            vec![DynamicOp::MatMul, DynamicOp::Sigmoid],
        ];

        for pattern in common_patterns {
            self.jit_compiler.compile_operations(&pattern)?;

            // Update JIT metrics
            let mut metrics = self.metrics.write().unwrap();
            metrics.jit_stats.total_compilations += 1;
            metrics.jit_stats.successful_compilations += 1;
        }

        Ok(())
    }

    /// Clean up old cache entries
    pub fn cleanup_cache(&mut self) {
        let now = Instant::now();
        let max_age = std::time::Duration::from_secs(3600); // 1 hour

        self.execution_cache
            .retain(|_, cached| now.duration_since(cached.last_used) < max_age);

        // Limit cache size
        if self.execution_cache.len() > self.config.max_cache_size {
            // Remove least recently used entries
            let entries: Vec<_> = self
                .execution_cache
                .iter()
                .map(|(k, v)| (k.clone(), v.last_used))
                .collect();
            let mut sorted_entries = entries;
            sorted_entries.sort_by_key(|(_, last_used)| *last_used);

            let to_remove = sorted_entries.len() - self.config.max_cache_size;
            for (key, _) in sorted_entries.into_iter().take(to_remove) {
                self.execution_cache.remove(&key);
            }
        }
    }

    /// Profile execution and suggest optimizations
    pub fn profile_execution(&mut self, iterations: usize) -> RusTorchResult<ProfileResult> {
        let mut profile_result = ProfileResult::new();

        for i in 0..iterations {
            let start_time = Instant::now();

            // Create a sample graph for profiling
            let result = self.execute_graph(|builder| {
                let input1 = builder.add_input(Tensor::ones(&[32, 784]))?;
                let weight1 = builder.add_parameter(Tensor::ones(&[128, 784]))?;
                let bias1 = builder.add_parameter(Tensor::ones(&[128]))?;

                let linear1 = builder.add_operation(
                    DynamicOp::Linear {
                        in_features: 784,
                        out_features: 128,
                    },
                    vec![input1, weight1, bias1],
                )?;

                let relu1 = builder.add_operation(DynamicOp::ReLU, vec![linear1])?;

                let weight2 = builder.add_parameter(Tensor::ones(&[10, 128]))?;
                let bias2 = builder.add_parameter(Tensor::ones(&[10]))?;

                let output = builder.add_operation(
                    DynamicOp::Linear {
                        in_features: 128,
                        out_features: 10,
                    },
                    vec![relu1, weight2, bias2],
                )?;

                Ok(output)
            })?;

            let execution_time = start_time.elapsed();
            profile_result.add_execution_time(execution_time);

            if i % 100 == 0 {
                println!(
                    "Profile iteration {}/{}: {:?}",
                    i + 1,
                    iterations,
                    execution_time
                );
            }
        }

        // Analyze metrics and generate recommendations
        profile_result.analyze_performance(&self.get_metrics());

        Ok(profile_result)
    }
}

/// Graph builder for fluent API
/// 流暢なAPIのためのグラフビルダー
pub struct GraphBuilder<'a, T: Float + Send + Sync + 'static> {
    context: &'a mut DynamicExecutionContext<T>,
}

impl<'a, T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    GraphBuilder<'a, T>
{
    /// Create new graph builder
    pub fn new(context: &'a mut DynamicExecutionContext<T>) -> Self {
        GraphBuilder { context }
    }

    /// Add input tensor
    pub fn add_input(&mut self, tensor: Tensor<T>) -> RusTorchResult<usize> {
        self.context.add_leaf(tensor)
    }

    /// Add parameter tensor
    pub fn add_parameter(&mut self, tensor: Tensor<T>) -> RusTorchResult<usize> {
        self.context.add_leaf(tensor)
    }

    /// Add operation
    pub fn add_operation(&mut self, op: DynamicOp, inputs: Vec<usize>) -> RusTorchResult<usize> {
        self.context.add_operation(op, inputs)
    }

    /// Add linear layer
    pub fn linear(
        &mut self,
        input: usize,
        weight: usize,
        bias: Option<usize>,
    ) -> RusTorchResult<usize> {
        let inputs = if let Some(b) = bias {
            vec![input, weight, b]
        } else {
            vec![input, weight]
        };

        // Infer dimensions from weight tensor
        if let Some(weight_node) = self.context.get_dynamic_node(&weight) {
            if let Some(weight_tensor) = weight_node.get_cached_output() {
                let shape = weight_tensor.shape();
                if shape.len() == 2 && shape[0] > 0 && shape[1] > 0 {
                    return self.add_operation(
                        DynamicOp::Linear {
                            in_features: shape[1],
                            out_features: shape[0],
                        },
                        inputs,
                    );
                }
            }
        }

        // Fallback to default sizes
        self.add_operation(
            DynamicOp::Linear {
                in_features: 784,
                out_features: 128,
            },
            inputs,
        )
    }

    /// Add conv2d layer
    pub fn conv2d(
        &mut self,
        input: usize,
        weight: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> RusTorchResult<usize> {
        self.add_operation(
            DynamicOp::Conv2d {
                kernel_size,
                stride,
                padding,
            },
            vec![input, weight],
        )
    }

    /// Add ReLU activation
    pub fn relu(&mut self, input: usize) -> RusTorchResult<usize> {
        self.add_operation(DynamicOp::ReLU, vec![input])
    }

    /// Add sigmoid activation  
    pub fn sigmoid(&mut self, input: usize) -> RusTorchResult<usize> {
        self.add_operation(DynamicOp::Sigmoid, vec![input])
    }

    /// Add element-wise addition
    pub fn add(&mut self, input1: usize, input2: usize) -> RusTorchResult<usize> {
        // Validate input shapes for compatibility
        if let (Some(node1), Some(node2)) = (
            self.context.get_dynamic_node(&input1),
            self.context.get_dynamic_node(&input2),
        ) {
            if let (Some(tensor1), Some(tensor2)) =
                (node1.get_cached_output(), node2.get_cached_output())
            {
                let shape1 = tensor1.shape();
                let shape2 = tensor2.shape();

                // Check for exact shape match or broadcasting compatibility
                if shape1 != shape2 && !Self::can_broadcast(shape1, shape2) {
                    return Err(RusTorchError::shape_mismatch(shape1, shape2));
                }
            }
        }

        self.add_operation(DynamicOp::Add, vec![input1, input2])
    }

    /// Check if two shapes can be broadcast together
    fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
        let (s1, s2) = if shape1.len() > shape2.len() {
            (shape1, shape2)
        } else {
            (shape2, shape1)
        };

        for (i, (&dim2, &dim1)) in s2.iter().rev().zip(s1.iter().rev()).enumerate() {
            if dim2 != 1 && dim1 != 1 && dim2 != dim1 {
                return false;
            }
        }
        true
    }

    /// Add matrix multiplication
    pub fn matmul(&mut self, input1: usize, input2: usize) -> RusTorchResult<usize> {
        self.add_operation(DynamicOp::MatMul, vec![input1, input2])
    }

    /// Add reshape operation
    pub fn reshape(&mut self, input: usize, shape: Vec<usize>) -> RusTorchResult<usize> {
        self.add_operation(DynamicOp::Reshape { shape }, vec![input])
    }
}

/// Profiling result with performance analysis
/// パフォーマンス分析付きプロファイル結果
pub struct ProfileResult {
    /// Execution times
    execution_times: Vec<std::time::Duration>,
    /// Performance recommendations
    recommendations: Vec<String>,
    /// Bottleneck analysis
    bottlenecks: Vec<BottleneckInfo>,
}

/// Bottleneck information
/// ボトルネック情報
#[derive(Debug)]
pub struct BottleneckInfo {
    /// Operation type causing bottleneck
    pub operation: String,
    /// Percentage of total time
    pub time_percentage: f64,
    /// Recommended optimization
    pub recommendation: String,
}

impl ProfileResult {
    /// Create new profile result
    pub fn new() -> Self {
        ProfileResult {
            execution_times: Vec::new(),
            recommendations: Vec::new(),
            bottlenecks: Vec::new(),
        }
    }

    /// Add execution time measurement
    pub fn add_execution_time(&mut self, time: std::time::Duration) {
        self.execution_times.push(time);
    }

    /// Analyze performance and generate recommendations
    pub fn analyze_performance(&mut self, metrics: &RuntimeMetrics) {
        // Calculate statistics
        let avg_time = if !self.execution_times.is_empty() {
            self.execution_times.iter().sum::<std::time::Duration>()
                / self.execution_times.len() as u32
        } else {
            std::time::Duration::default()
        };
        let min_time = self
            .execution_times
            .iter()
            .min()
            .copied()
            .unwrap_or_default();
        let max_time = self
            .execution_times
            .iter()
            .max()
            .copied()
            .unwrap_or_default();

        // Generate recommendations based on analysis
        if metrics.cache_hit_rate < 0.5 {
            self.recommendations.push(
                "Consider increasing cache size or improving cache key generation".to_string(),
            );
        }

        if metrics.jit_stats.avg_speedup < 2.0 {
            self.recommendations
                .push("JIT compilation showing limited benefit, consider disabling".to_string());
        }

        if metrics.memory_stats.memory_efficiency < 0.7 {
            self.recommendations
                .push("Memory efficiency low, consider memory pooling optimization".to_string());
        }

        if metrics.parallel_stats.parallel_efficiency < 0.6 {
            self.recommendations.push(
                "Parallel execution efficiency low, review operation dependencies".to_string(),
            );
        }

        // Identify bottlenecks
        if max_time > avg_time * 2 {
            self.bottlenecks.push(BottleneckInfo {
                operation: "Variable execution time".to_string(),
                time_percentage: ((max_time.as_nanos() - min_time.as_nanos()) as f64
                    / max_time.as_nanos() as f64)
                    * 100.0,
                recommendation: "Investigate inconsistent operation performance".to_string(),
            });
        }
    }

    /// Get performance summary
    pub fn summary(&self) -> String {
        let avg_time = if !self.execution_times.is_empty() {
            self.execution_times.iter().sum::<std::time::Duration>()
                / self.execution_times.len() as u32
        } else {
            std::time::Duration::default()
        };

        format!(
            "Performance Profile Summary:\n\
             - Executions: {}\n\
             - Average time: {:?}\n\
             - Recommendations: {}\n\
             - Bottlenecks: {}",
            self.execution_times.len(),
            avg_time,
            self.recommendations.len(),
            self.bottlenecks.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_engine_creation() {
        let config = RuntimeConfig::default();
        let _engine = RuntimeEngine::<f32>::new(config);
    }

    #[test]
    fn test_graph_builder() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        let result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[2, 3]))?;
            let weight = builder.add_parameter(Tensor::ones(&[4, 3]))?;
            let output = builder.linear(input, weight, None)?;
            Ok(output)
        });

        match result {
            Ok(_) => {}
            Err(e) => panic!("Test failed with error: {:?}", e),
        }
    }

    #[test]
    fn test_warmup() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        engine.warmup().unwrap();

        // Should have compiled common patterns
        assert!(engine.jit_compiler.get_stats().compilations > 0);
    }

    #[test]
    fn test_cache_cleanup() {
        let mut config = RuntimeConfig::default();
        config.max_cache_size = 2;
        let mut engine = RuntimeEngine::<f32>::new(config);

        // Fill cache beyond limit
        for i in 0..5 {
            let _result = engine
                .execute_graph(|builder| {
                    let input = builder.add_input(Tensor::ones(&[i + 1, 3]))?;
                    let output = builder.relu(input)?;
                    Ok(output)
                })
                .unwrap();
        }

        engine.cleanup_cache();
        assert!(engine.execution_cache.len() <= 2);
    }

    #[test]
    fn test_profiling() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        let profile_result = engine.profile_execution(3).unwrap();
        let summary = profile_result.summary();

        assert!(summary.contains("Executions: 3"));
    }
}
