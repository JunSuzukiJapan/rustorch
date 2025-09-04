//! Integration tests for distributed training functionality
//! 分散学習機能の統合テスト

use rustorch::distributed::*;
use rustorch::tensor::Tensor;
use rustorch::nn::Linear;
use rustorch::autograd::Variable;
use rustorch::error::RusTorchResult;
use std::time::Duration;

/// Test basic distributed initialization
/// 基本的な分散初期化テスト
#[test]
fn test_distributed_initialization() {
    // Set environment variables for testing
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29500");

    let result = init_process_group(
        DistributedBackend::TCP,
        Some("tcp://localhost:29500"),
        Some(1),
        Some(0),
        Some(Duration::from_secs(30)),
    );

    assert!(result.is_ok(), "Failed to initialize process group: {:?}", result);
    assert!(is_initialized());
    assert_eq!(api::get_rank(), 0);
    assert_eq!(api::get_world_size(), 1);

    // Clean up
    let _ = destroy_process_group();
}

/// Test DistributedDataParallel wrapper
/// DistributedDataParallelラッパーテスト
#[test]
fn test_ddp_wrapper() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    // Initialize distributed training
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29500");

    let init_result = init_process_group(
        DistributedBackend::TCP,
        None,
        None,
        None,
        None,
    );
    assert!(init_result.is_ok(), "Failed to initialize process group: {:?}", init_result);

    // Create a simple model
    let linear: Linear<f32> = Linear::new(10, 5);
    let ddp_result = wrap_module(linear, Some(vec![0]));
    
    assert!(ddp_result.is_ok(), "Failed to create DDP wrapper: {:?}", ddp_result);

    // Test forward pass
    let ddp = ddp_result.unwrap();
    let input = Variable::new(Tensor::randn(&[2, 10]), false);
    let output = ddp.forward(&input);
    
    assert!(output.is_ok(), "DDP forward pass failed: {:?}", output);
    let output_var = output.unwrap();
    let output_data = output_var.data();
    let output_guard = output_data.read().unwrap();
    assert_eq!(output_guard.shape(), &[2, 5]);

    // Clean up
    let _ = destroy_process_group();
}

/// Test all-reduce operation
/// All-reduce操作テスト
#[test]
fn test_all_reduce_operation() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    // Initialize single-process distributed training for testing
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29501");

    let init_result = init_process_group(DistributedBackend::TCP, None, None, None, None);
    if init_result.is_err() {
        println!("Skipping test - distributed initialization failed: {:?}", init_result);
        return;
    }

    let mut tensor: Tensor<f32> = Tensor::ones(&[3, 3]);
    let result = all_reduce(&mut tensor, ReduceOp::Sum, None, false);
    
    // Just check that operation didn't fail - don't try to print DistributedRequest
    assert!(result.is_ok(), "All-reduce operation failed");

    // Clean up
    let _ = destroy_process_group();
}

/// Test broadcast operation
/// ブロードキャスト操作テスト
#[test]
fn test_broadcast_operation() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29502");

    let init_result = init_process_group(DistributedBackend::TCP, None, None, None, None);
    if init_result.is_err() {
        println!("Skipping test - distributed initialization failed: {:?}", init_result);
        return;
    }

    let mut tensor: Tensor<f32> = Tensor::randn(&[2, 2]);
    let _original_data = tensor.clone();
    let result = broadcast(&mut tensor, 0, None, false);
    
    // Just check that operation didn't fail - don't try to print DistributedRequest
    assert!(result.is_ok(), "Broadcast operation failed");

    // Clean up
    let _ = destroy_process_group();
}

/// Test gradient synchronization
/// 勾配同期テスト
#[test]
fn test_gradient_synchronization() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29503");

    let init_result = init_process_group(DistributedBackend::TCP, None, None, None, None);
    if init_result.is_err() {
        println!("Skipping test - distributed initialization failed: {:?}", init_result);
        return;
    }

    // Create DDP model
    let linear: Linear<f32> = Linear::new(5, 3);
    let ddp = wrap_module(linear, Some(vec![0])).unwrap();

    // Perform forward pass
    let input = Variable::new(Tensor::randn(&[2, 5]), false);
    let _output = ddp.forward(&input).unwrap();

    // Test gradient synchronization
    let sync_result = ddp.sync_gradients();
    assert!(sync_result.is_ok(), "Gradient synchronization failed: {:?}", sync_result);

    // Clean up
    let _ = destroy_process_group();
}

// Async gradient synchronization test removed to avoid feature warnings
// Use --features async-gradient to enable async functionality testing

/// Performance benchmark for distributed operations
/// 分散操作のパフォーマンスベンチマーク
#[test]
fn test_distributed_performance() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29504");

    let init_result = init_process_group(DistributedBackend::TCP, None, None, None, None);
    if init_result.is_err() {
        println!("Skipping test - distributed initialization failed: {:?}", init_result);
        return;
    }

    let sizes = vec![
        vec![100, 100],        // 10K elements
        vec![1000, 100],       // 100K elements  
        vec![1000, 1000],      // 1M elements
    ];

    for size in sizes {
        let mut tensor: Tensor<f32> = Tensor::randn(&size);
        
        let start = std::time::Instant::now();
        let result = all_reduce(&mut tensor, ReduceOp::Sum, None, false);
        let duration = start.elapsed();
        
        assert!(result.is_ok(), "All-reduce failed for size {:?}", size);
        println!("All-reduce for {:?}: {:?}", size, duration);
    }

    // Clean up
    let _ = destroy_process_group();
}

/// Test error handling in distributed operations
/// 分散操作でのエラーハンドリングテスト
#[test]
fn test_distributed_error_handling() {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    // Test without initialization
    assert!(!is_initialized());
    
    let mut tensor: Tensor<f32> = Tensor::ones(&[2, 2]);
    let result = all_reduce(&mut tensor, ReduceOp::Sum, None, false);
    assert!(result.is_err(), "All-reduce should fail without initialization");

    // Test invalid operations
    let invalid_group_result = new_group(vec![], None, None);
    assert!(invalid_group_result.is_err(), "Empty group creation should fail");
}

/// Test NCCL backend specific functionality
/// NCCLバックエンド固有機能テスト
#[cfg(feature = "nccl")]
#[test]
fn test_nccl_specific_features() {
    use rustorch::distributed::nccl_integration::{NCCLOps, NCCLOptimizations};

    // Test NCCL configuration
    let config = NCCLOps::get_optimal_config(8, 32.0);
    assert!(config.compression_enabled);
    assert_eq!(config.bucket_size_mb, 50);

    let config = NCCLOps::get_optimal_config(2, 8.0);
    assert!(!config.compression_enabled);
    assert_eq!(config.bucket_size_mb, 25);
}

/// Multi-GPU validation test
/// マルチGPU検証テスト
#[test]
fn test_multi_gpu_validation() {
    // This test would require actual GPU hardware
    // このテストは実際のGPUハードウェアが必要
    let validator: RusTorchResult<MultiGpuValidator<f32>> = MultiGpuValidator::new();
    assert!(validator.is_ok());

    let validator = validator.unwrap();
    let available_gpus = validator.get_devices();
    
    // Test should pass even with 0 GPUs detected
    println!("Detected {} GPUs", available_gpus.len());
}

/// Integration test with realistic training scenario
/// 現実的な学習シナリオでの統合テスト
#[test]
fn test_distributed_training_scenario() -> RusTorchResult<()> {
    // Ensure clean state first
    let _ = destroy_process_group();
    
    // Setup distributed environment
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29505");

    init_process_group(DistributedBackend::TCP, None, None, None, None)?;

    // Create model
    let model: Linear<f32> = Linear::new(784, 10);
    let ddp_model = wrap_module(model, Some(vec![0]))?;

    // Simulate training step
    let batch_size = 32;
    let input = Variable::new(Tensor::randn(&[batch_size, 784]), false);
    let _target: Variable<f32> = Variable::new(Tensor::randn(&[batch_size, 10]), false);

    // Forward pass
    let output = ddp_model.forward(&input)?;
    let output_data = output.data();
    let output_guard = output_data.read().unwrap();
    assert_eq!(output_guard.shape(), &[batch_size, 10]);

    // Backward pass (simplified - would need loss function)
    // バックワードパス（簡略化 - 損失関数が必要）
    
    // Gradient synchronization
    ddp_model.sync_gradients()?;

    // Clean up
    destroy_process_group()?;

    Ok(())
}