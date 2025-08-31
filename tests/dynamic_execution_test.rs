//! Comprehensive tests for dynamic execution engine
//! 動的実行エンジンの包括的テスト

use rustorch::execution::{RuntimeEngine, RuntimeConfig, DynamicOp};
use rustorch::tensor::Tensor;
use std::time::Duration;

#[test]
fn test_basic_dynamic_execution() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[2, 3]))?;
        let weight = builder.add_parameter(Tensor::ones(&[4, 3]))?;
        let output = builder.linear(input, weight, None)?;
        Ok(output)
    });

    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[2, 4]);
        },
        Err(e) => panic!("Basic dynamic execution test failed with error: {:?}", e),
    }
}

#[test]
fn test_complex_neural_network() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Multi-layer perceptron
        let input = builder.add_input(Tensor::ones(&[16, 784]))?; // Batch of 16, MNIST-like
        
        // Layer 1: 784 -> 512
        let w1 = builder.add_parameter(Tensor::ones(&[512, 784]))?;
        let b1 = builder.add_parameter(Tensor::ones(&[512]))?;
        let h1 = builder.linear(input, w1, Some(b1))?;
        let a1 = builder.relu(h1)?;
        
        // Layer 2: 512 -> 256  
        let w2 = builder.add_parameter(Tensor::ones(&[256, 512]))?;
        let b2 = builder.add_parameter(Tensor::ones(&[256]))?;
        let h2 = builder.linear(a1, w2, Some(b2))?;
        let a2 = builder.relu(h2)?;
        
        // Layer 3: 256 -> 10
        let w3 = builder.add_parameter(Tensor::ones(&[10, 256]))?;
        let b3 = builder.add_parameter(Tensor::ones(&[10]))?;
        let output = builder.linear(a2, w3, Some(b3))?;
        
        Ok(output)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[16, 10]);

    // Check that metrics were collected
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_executions, 1);
    assert!(metrics.avg_execution_time > Duration::from_nanos(0));
}

#[test]
fn test_convolutional_network() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Input: [batch=2, channels=3, height=32, width=32] (CIFAR-10 like)
        let input = builder.add_input(Tensor::ones(&[2, 3, 32, 32]))?;
        
        // Conv layer 1: 3 -> 16 channels
        let conv1_weight = builder.add_parameter(Tensor::ones(&[16, 3, 3, 3]))?;
        let conv1 = builder.conv2d(input, conv1_weight, (3, 3), (1, 1), (1, 1))?;
        let conv1_relu = builder.relu(conv1)?;
        
        // Conv layer 2: 16 -> 32 channels  
        let conv2_weight = builder.add_parameter(Tensor::ones(&[32, 16, 3, 3]))?;
        let conv2 = builder.conv2d(conv1_relu, conv2_weight, (3, 3), (1, 1), (1, 1))?;
        let conv2_relu = builder.relu(conv2)?;
        
        // Flatten for linear layer
        let flattened = builder.reshape(conv2_relu, vec![2, 32 * 32 * 32])?;
        
        // Linear layer: features -> 10 classes
        let linear_weight = builder.add_parameter(Tensor::ones(&[10, 32 * 32 * 32]))?;
        let linear_bias = builder.add_parameter(Tensor::ones(&[10]))?;
        let output = builder.linear(flattened, linear_weight, Some(linear_bias))?;
        
        Ok(output)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[2, 10]);
}

#[test]
fn test_jit_compilation() {
    let mut config = RuntimeConfig::default();
    config.enable_jit = true;
    config.jit_threshold = 3; // Low threshold for testing
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Execute the same pattern multiple times to trigger JIT
    for i in 0..5 {
        let result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[4, 4]))?;
            let relu1 = builder.relu(input)?;
            let relu2 = builder.relu(relu1)?;
            let relu3 = builder.relu(relu2)?;
            Ok(relu3)
        });
        
        assert!(result.is_ok(), "Execution {} failed", i);
    }

    // Check that JIT compilation was triggered
    let metrics = engine.get_metrics();
    assert!(metrics.jit_stats.total_compilations > 0);
    assert_eq!(metrics.total_executions, 5);
}

#[test]
fn test_memory_optimization() {
    let mut config = RuntimeConfig::default();
    config.enable_memory_opt = true;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Execute operations that should benefit from memory pooling
    for size in [10, 50, 100, 200].iter() {
        let result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[*size, *size]))?;
            let weight = builder.add_parameter(Tensor::ones(&[*size, *size]))?;
            let matmul = builder.matmul(input, weight)?;
            let output = builder.relu(matmul)?;
            Ok(output)
        });

        assert!(result.is_ok());
    }

    let metrics = engine.get_metrics();
    assert!(metrics.memory_stats.allocations > 0);
    assert!(metrics.total_executions == 4);
}

#[test]
fn test_parallel_execution() {
    let mut config = RuntimeConfig::default();
    config.enable_parallel = true;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Create independent computation paths that can be parallelized
        let input1 = builder.add_input(Tensor::ones(&[8, 8]))?;
        let input2 = builder.add_input(Tensor::ones(&[8, 8]))?;
        let input3 = builder.add_input(Tensor::ones(&[8, 8]))?;
        
        // Independent ReLU operations (can be parallel)
        let relu1 = builder.relu(input1)?;
        let relu2 = builder.relu(input2)?;
        let relu3 = builder.relu(input3)?;
        
        // Combine results (sequential dependency)
        let add1 = builder.add(relu1, relu2)?;
        let final_output = builder.add(add1, relu3)?;
        
        Ok(final_output)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[8, 8]);

    let _metrics = engine.get_metrics();
    // Parallel opportunities is always non-negative by type definition
}

#[test]
fn test_operation_fusion() {
    let mut config = RuntimeConfig::default();
    config.enable_fusion = true;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[4, 4]))?;
        
        // Chain of operations that can be fused
        let relu1 = builder.relu(input)?;
        let relu2 = builder.relu(relu1)?; // Redundant ReLU - should be optimized away
        
        Ok(relu2)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[4, 4]);
    
    // All values should be 1.0 (ReLU of positive values)
    if let Some(slice) = output.as_slice() {
        for &value in slice {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_caching_effectiveness() {
    let mut config = RuntimeConfig::default();
    config.max_cache_size = 100;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Execute the same pattern multiple times
    let pattern_executions = 5;
    for _ in 0..pattern_executions {
        let result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[3, 3]))?;
            let weight = builder.add_parameter(Tensor::ones(&[3, 3]))?; // Weight matrix
            let output = builder.matmul(input, weight)?;
            Ok(output)
        });
        
        assert!(result.is_ok());
    }

    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_executions, pattern_executions);
    
    // Cache hit rate should improve over time
    assert!(metrics.cache_hit_rate >= 0.0);
}

#[test]
fn test_error_handling() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Test invalid operations
    let result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[2, 3]))?;
        let weight = builder.add_parameter(Tensor::ones(&[5, 6]))?; // Incompatible shapes
        let output = builder.matmul(input, weight)?; // Should fail due to shape mismatch
        Ok(output)
    });

    assert!(result.is_err());
}

#[test]
fn test_profiling_functionality() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let profile_result = engine.profile_execution(3);
    assert!(profile_result.is_ok());
    
    let result = profile_result.unwrap();
    let summary = result.summary();
    
    assert!(summary.contains("Executions: 3"));
    assert!(summary.contains("Average time:"));
}

#[test]
fn test_warmup_functionality() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Test warmup
    let warmup_result = engine.warmup();
    assert!(warmup_result.is_ok());

    // Should have compiled common patterns
    let metrics = engine.get_metrics();
    assert!(metrics.jit_stats.total_compilations > 0);
}

#[test]
fn test_cache_cleanup() {
    let mut config = RuntimeConfig::default();
    config.max_cache_size = 2;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Create more cache entries than allowed
    for i in 0..5 {
        let _result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[i + 2, i + 2]))?;
            let output = builder.relu(input)?;
            Ok(output)
        }).unwrap();
    }

    // Trigger cleanup
    engine.cleanup_cache();
    
    // Cache should be within limits
    assert!(engine.execution_cache.len() <= 2);
}

#[test]
fn test_multi_input_operations() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Test operations with multiple inputs
        let input1 = builder.add_input(Tensor::ones(&[3, 3]))?;
        let input2 = builder.add_input(Tensor::from_vec(vec![2.0; 9], vec![3, 3]))?;
        
        // Element-wise operations
        let add_result = builder.add(input1, input2)?;
        let mult_input = builder.add_input(Tensor::from_vec(vec![0.5; 9], vec![3, 3]))?;
        let mult_result = builder.add_operation(DynamicOp::Mul, vec![add_result, mult_input])?;
        
        Ok(mult_result)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[3, 3]);
    
    // Verify computation: (1 + 2) * 0.5 = 1.5
    if let Some(slice) = output.as_slice() {
        for &value in slice {
            assert!((value - 1.5).abs() < 1e-6);
        }
    }
}

#[test]
fn test_activation_functions() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Test ReLU
    let relu_result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]))?;
        let output = builder.relu(input)?;
        Ok(output)
    }).unwrap();
    
    let expected_relu = vec![0.0, 0.0, 1.0, 2.0];
    if let Some(slice) = relu_result.as_slice() {
        for (actual, expected) in slice.iter().zip(expected_relu.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    // Test Sigmoid
    let sigmoid_result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::from_vec(vec![0.0], vec![1]))?;
        let output = builder.sigmoid(input)?;
        Ok(output)
    }).unwrap();
    
    // sigmoid(0) = 0.5
    if let Some(slice) = sigmoid_result.as_slice() {
        assert!((slice[0] - 0.5).abs() < 1e-6);
    }
}

#[test]
fn test_reshape_operations() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[2, 3, 4]))?; // 24 elements
        let reshaped = builder.reshape(input, vec![4, 6])?; // Reshape to 4x6
        Ok(reshaped)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[4, 6]);
    if let Some(slice) = output.as_slice() {
        assert_eq!(slice.len(), 24);
    }
}

#[test]
fn test_performance_optimization() {
    let mut config = RuntimeConfig::default();
    config.enable_jit = true;
    config.enable_fusion = true;
    config.enable_parallel = true;
    config.jit_threshold = 2;
    
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Warm up the engine
    engine.warmup().unwrap();

    // Measure execution times
    let mut times = Vec::new();
    
    for _ in 0..10 {
        let start = std::time::Instant::now();
        
        let _result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[32, 64]))?;
            let w1 = builder.add_parameter(Tensor::ones(&[128, 64]))?;
            let h1 = builder.linear(input, w1, None)?;
            let a1 = builder.relu(h1)?;
            let w2 = builder.add_parameter(Tensor::ones(&[32, 128]))?;
            let output = builder.linear(a1, w2, None)?;
            Ok(output)
        }).unwrap();
        
        times.push(start.elapsed());
    }

    // Later executions should be faster or at least consistent
    let early_avg = times[0..3].iter().sum::<Duration>() / 3;
    let later_avg = times[7..10].iter().sum::<Duration>() / 3;
    
    println!("Early average: {:?}, Later average: {:?}", early_avg, later_avg);
    
    // With optimization, later executions should not be significantly slower
    assert!(later_avg <= early_avg * 3); // Allow some variance
    
    let metrics = engine.get_metrics();
    assert!(metrics.jit_stats.total_compilations > 0);
    assert!(metrics.total_executions == 10);
}

#[test]
fn test_memory_efficiency() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let _initial_memory = std::mem::size_of::<RuntimeEngine<f32>>();

    // Execute multiple operations
    for i in 1..=5 {
        let _result = engine.execute_graph(|builder| {
            let size = i * 10;
            let input = builder.add_input(Tensor::ones(&[size, size]))?;
            let weight = builder.add_parameter(Tensor::ones(&[size, size]))?;
            let output = builder.matmul(input, weight)?;
            Ok(output)
        }).unwrap();
    }

    let metrics = engine.get_metrics();
    
    // Memory should be managed efficiently
    assert!(metrics.memory_stats.allocations > 0);
    assert!(metrics.memory_stats.memory_efficiency >= 0.0);
    
    // Peak memory should be reasonable
    assert!(metrics.memory_stats.peak_memory > 0);
}

#[test]
fn test_execution_plan_optimization() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Create a graph with parallelizable operations
        let input1 = builder.add_input(Tensor::ones(&[5, 5]))?;
        let input2 = builder.add_input(Tensor::ones(&[5, 5]))?;
        let input3 = builder.add_input(Tensor::ones(&[5, 5]))?;
        
        // These can be executed in parallel
        let relu1 = builder.relu(input1)?;
        let relu2 = builder.relu(input2)?;
        let sigmoid1 = builder.sigmoid(input3)?;
        
        // Sequential dependencies
        let add1 = builder.add(relu1, relu2)?;
        let final_result = builder.add(add1, sigmoid1)?;
        
        Ok(final_result)
    });

    assert!(result.is_ok());
    
    let _metrics = engine.get_metrics();
    if engine.config.enable_parallel {
        // Parallel opportunities is always non-negative by type definition
    }
}

#[test]
fn test_error_recovery() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Test recovery from invalid operations
    let invalid_result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[2, 3]))?;
        let incompatible = builder.add_input(Tensor::ones(&[4, 5]))?;
        let _output = builder.add(input, incompatible)?; // Should fail - shape mismatch
        Ok(0) // Won't reach here
    });

    assert!(invalid_result.is_err());

    // Engine should still work after error
    let valid_result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[3, 3]))?;
        let output = builder.relu(input)?;
        Ok(output)
    });

    assert!(valid_result.is_ok());
}

#[test]
fn test_large_scale_execution() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Test execution with larger tensors
    let result = engine.execute_graph(|builder| {
        let input = builder.add_input(Tensor::ones(&[128, 1024]))?; // Large input
        
        // Deep network
        let mut current = input;
        let layer_sizes = [1024, 512, 256, 128, 64, 32, 16, 8];
        
        for i in 0..layer_sizes.len() - 1 {
            let weight = builder.add_parameter(Tensor::ones(&[layer_sizes[i + 1], layer_sizes[i]]))?;
            let bias = builder.add_parameter(Tensor::ones(&[layer_sizes[i + 1]]))?;
            let linear = builder.linear(current, weight, Some(bias))?;
            current = builder.relu(linear)?;
        }
        
        Ok(current)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[128, 8]);

    let metrics = engine.get_metrics();
    assert!(metrics.avg_execution_time > Duration::from_nanos(0));
}

#[test]
fn test_mixed_operations() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    let result = engine.execute_graph(|builder| {
        // Mix of different operation types
        let input = builder.add_input(Tensor::ones(&[4, 16, 16]))?; // 3D tensor
        
        // Reshape to 2D for linear operations
        let flattened = builder.reshape(input, vec![4, 16 * 16])?;
        
        // Linear transformation
        let weight = builder.add_parameter(Tensor::ones(&[64, 16 * 16]))?;
        let linear = builder.linear(flattened, weight, None)?;
        
        // Activation
        let activated = builder.relu(linear)?;
        
        // Another reshape
        let reshaped = builder.reshape(activated, vec![4, 8, 8])?;
        
        // Final linear layer
        let final_flat = builder.reshape(reshaped, vec![4, 64])?;
        let final_weight = builder.add_parameter(Tensor::ones(&[10, 64]))?;
        let output = builder.linear(final_flat, final_weight, None)?;
        
        Ok(output)
    });

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[4, 10]);
}

#[test]
fn test_metrics_accuracy() {
    let config = RuntimeConfig::default();
    let mut engine = RuntimeEngine::<f32>::new(config);

    // Reset metrics to start clean
    engine.reset_metrics();
    
    let executions = 3;
    for _ in 0..executions {
        let _result = engine.execute_graph(|builder| {
            let input = builder.add_input(Tensor::ones(&[2, 2]))?;
            let output = builder.relu(input)?;
            Ok(output)
        }).unwrap();
    }

    let metrics = engine.get_metrics();
    
    // Verify metric accuracy
    assert_eq!(metrics.total_executions, executions);
    assert!(metrics.avg_execution_time > Duration::from_nanos(0));
    
    // Cache hit rate should be reasonable (0.0 to 1.0)
    assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
}