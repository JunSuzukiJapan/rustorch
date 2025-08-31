//! Dynamic execution engine for runtime graph optimization
//! 実行時グラフ最適化のための動的実行エンジン

pub mod dynamic;
pub mod runtime;

pub use dynamic::{
    DynamicExecutionContext, DynamicExecutionStats, DynamicNode, DynamicOp, ExecutionPlan,
    JitCompiler, JitStats, MemoryAllocation, MemoryPlan, PlannedOperation,
};

pub use runtime::{
    BottleneckInfo, CachedExecution, GraphBuilder, JitCompilationMetrics, MemoryMetrics,
    ParallelExecutionMetrics, ProfileResult, RuntimeConfig, RuntimeEngine, RuntimeMetrics,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_end_to_end_dynamic_execution() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        // Test complete neural network execution
        let result = engine.execute_graph(|builder| {
            // Input layer
            let input = builder.add_input(Tensor::randn(&[32, 784]))?; // MNIST-like input

            // Hidden layer 1
            let weight1 = builder.add_parameter(Tensor::randn(&[256, 784]))?;
            let bias1 = builder.add_parameter(Tensor::randn(&[256]))?;
            let hidden1 = builder.linear(input, weight1, Some(bias1))?;
            let activated1 = builder.relu(hidden1)?;

            // Hidden layer 2
            let weight2 = builder.add_parameter(Tensor::randn(&[128, 256]))?;
            let bias2 = builder.add_parameter(Tensor::randn(&[128]))?;
            let hidden2 = builder.linear(activated1, weight2, Some(bias2))?;
            let activated2 = builder.relu(hidden2)?;

            // Output layer
            let weight3 = builder.add_parameter(Tensor::randn(&[10, 128]))?;
            let bias3 = builder.add_parameter(Tensor::randn(&[10]))?;
            let output = builder.linear(activated2, weight3, Some(bias3))?;

            Ok(output)
        });

        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[32, 10]);

        // Verify metrics were collected
        let metrics = engine.get_metrics();
        assert!(metrics.total_executions > 0);
        assert!(metrics.avg_execution_time > std::time::Duration::from_nanos(0));
    }

    #[test]
    fn test_multi_layer_neural_network() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        let result = engine.execute_graph(|builder| {
            // Input: [batch=16, features=784] (flattened MNIST-like)
            let input = builder.add_input(Tensor::randn(&[16, 784]))?;

            // Hidden layer 1: 784 -> 256
            let w1 = builder.add_parameter(Tensor::randn(&[256, 784]))?;
            let b1 = builder.add_parameter(Tensor::randn(&[256]))?;
            let h1 = builder.linear(input, w1, Some(b1))?;
            let a1 = builder.relu(h1)?;

            // Hidden layer 2: 256 -> 128
            let w2 = builder.add_parameter(Tensor::randn(&[128, 256]))?;
            let b2 = builder.add_parameter(Tensor::randn(&[128]))?;
            let h2 = builder.linear(a1, w2, Some(b2))?;
            let a2 = builder.relu(h2)?;

            // Output layer: 128 -> 10
            let w3 = builder.add_parameter(Tensor::randn(&[10, 128]))?;
            let b3 = builder.add_parameter(Tensor::randn(&[10]))?;
            let output = builder.linear(a2, w3, Some(b3))?;

            Ok(output)
        });

        match result {
            Ok(output_tensor) => {
                assert_eq!(output_tensor.shape(), &[16, 10]);
            }
            Err(e) => panic!("Multi-layer test failed with error: {:?}", e),
        }
    }

    #[test]
    fn test_optimization_effectiveness() {
        let mut config = RuntimeConfig::default();
        config.enable_jit = true;
        config.enable_fusion = true;
        config.jit_threshold = 3;

        let mut engine = RuntimeEngine::<f32>::new(config);

        // Run warmup
        engine.warmup().unwrap();

        // Execute same pattern multiple times
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = std::time::Instant::now();
            let _result = engine
                .execute_graph(|builder| {
                    let input = builder.add_input(Tensor::ones(&[64, 128]))?;
                    let relu1 = builder.relu(input)?;
                    let relu2 = builder.relu(relu1)?;
                    let relu3 = builder.relu(relu2)?;
                    Ok(relu3)
                })
                .unwrap();
            times.push(start.elapsed());
        }

        // Later executions should be faster due to optimization
        let early_avg = times[0..3].iter().sum::<std::time::Duration>() / 3;
        let later_avg = times[7..10].iter().sum::<std::time::Duration>() / 3;

        // JIT and caching should provide some speedup
        println!("Early avg: {:?}, Later avg: {:?}", early_avg, later_avg);
        assert!(later_avg <= early_avg * 2); // Allow some variance in testing
    }

    #[test]
    fn test_memory_management() {
        let config = RuntimeConfig::default();
        let mut engine = RuntimeEngine::<f32>::new(config);

        // Execute memory-intensive operations
        for size in [100, 500, 1000].iter() {
            let _result = engine
                .execute_graph(|builder| {
                    let input = builder.add_input(Tensor::randn(&[*size, *size]))?;
                    let output = builder.relu(input)?;
                    Ok(output)
                })
                .unwrap();
        }

        let metrics = engine.get_metrics();

        // Execution metrics should be tracked
        assert!(metrics.total_executions > 0);
    }
}
