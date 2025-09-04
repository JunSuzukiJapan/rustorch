//! Integration tests for distributed training functionality
//! 分散学習機能の統合テスト

use rustorch::distributed::*;
use rustorch::tensor::Tensor;
use rustorch::nn::{Module, Linear};
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
    assert_eq!(get_rank(), 0);
    assert_eq!(get_world_size(), 1);

    // Clean up
    let _ = destroy_process_group();
}

/// Test DistributedDataParallel wrapper
/// DistributedDataParallelラッパーテスト
#[test]
fn test_ddp_wrapper() {
    // Initialize distributed training
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29500");

    let _ = init_process_group(
        DistributedBackend::TCP,
        None,
        None,
        None,
        None,
    );

    // Create a simple model
    let linear = Linear::new(10, 5).unwrap();
    let ddp_result = wrap_module(linear, Some(vec![0]));
    
    assert!(ddp_result.is_ok(), "Failed to create DDP wrapper: {:?}", ddp_result);

    // Test forward pass
    let ddp = ddp_result.unwrap();
    let input: Tensor<f32> = Tensor::randn(&[2, 10]);
    let output = ddp.forward(&input);
    
    assert!(output.is_ok(), "DDP forward pass failed: {:?}", output);
    assert_eq!(output.unwrap().shape(), &[2, 5]);

    // Clean up
    let _ = destroy_process_group();
}

/// Test all-reduce operation
/// All-reduce操作テスト
#[test]
fn test_all_reduce_operation() {
    // Initialize single-process distributed training for testing
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29501");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let mut tensor: Tensor<f32> = Tensor::ones(&[3, 3]);
    let result = all_reduce(&mut tensor, ReduceOp::Sum, None, false);
    
    assert!(result.is_ok(), "All-reduce operation failed: {:?}", result);

    // Clean up
    let _ = destroy_process_group();
}

/// Test broadcast operation
/// ブロードキャスト操作テスト
#[test]
fn test_broadcast_operation() {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29502");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let mut tensor: Tensor<f32> = Tensor::randn(&[2, 2]);
    let original_data = tensor.clone();
    let result = broadcast(&mut tensor, 0, None, false);
    
    assert!(result.is_ok(), "Broadcast operation failed: {:?}", result);

    // Clean up
    let _ = destroy_process_group();
}

/// Test gradient synchronization
/// 勾配同期テスト
#[test]
fn test_gradient_synchronization() {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29503");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    // Create DDP model
    let linear = Linear::new(5, 3).unwrap();
    let ddp = wrap_module(linear, Some(vec![0])).unwrap();

    // Perform forward pass
    let input: Tensor<f32> = Tensor::randn(&[2, 5]);
    let output = ddp.forward(&input).unwrap();

    // Test gradient synchronization
    let sync_result = ddp.sync_gradients();
    assert!(sync_result.is_ok(), "Gradient synchronization failed: {:?}", sync_result);

    // Clean up
    let _ = destroy_process_group();
}

/// Test async gradient synchronization
/// 非同期勾配同期テスト
#[cfg(feature = "async")]
#[test]
fn test_async_gradient_sync() {
    use rustorch::distributed::async_gradient::{AsyncGradientSynchronizer, AsyncConfig, Priority};

    let config = AsyncConfig::default();
    let sync_result = AsyncGradientSynchronizer::new(config);
    assert!(sync_result.is_ok(), "Failed to create async synchronizer: {:?}", sync_result);

    let synchronizer = sync_result.unwrap();
    let gradient: Tensor<f32> = Tensor::randn(&[10, 10]);
    
    let submit_result = synchronizer.submit_gradient(
        "test_param".to_string(),
        gradient,
        Priority::Normal,
    );
    assert!(submit_result.is_ok(), "Failed to submit gradient: {:?}", submit_result);

    // Give some time for processing
    std::thread::sleep(Duration::from_millis(100));

    let completions = synchronizer.check_completions();
    assert!(!completions.is_empty(), "No completions received");
}

/// Performance benchmark for distributed operations
/// 分散操作のパフォーマンスベンチマーク
#[test]
fn test_distributed_performance() {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29504");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

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
    let validator = MultiGpuValidator::new();
    assert!(validator.is_ok());

    let validator = validator.unwrap();
    let available_gpus = validator.detect_gpus();
    
    // Test should pass even with 0 GPUs detected
    println!("Detected {} GPUs", available_gpus.len());
}

/// Integration test with realistic training scenario
/// 現実的な学習シナリオでの統合テスト
#[test]
fn test_distributed_training_scenario() -> RusTorchResult<()> {
    // Setup distributed environment
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29505");

    init_process_group(DistributedBackend::TCP, None, None, None, None)?;

    // Create model
    let model = Linear::new(784, 10)?;
    let ddp_model = wrap_module(model, Some(vec![0]))?;

    // Simulate training step
    let batch_size = 32;
    let input: Tensor<f32> = Tensor::randn(&[batch_size, 784]);
    let target: Tensor<f32> = Tensor::randn(&[batch_size, 10]);

    // Forward pass
    let output = ddp_model.forward(&input)?;
    assert_eq!(output.shape(), &[batch_size, 10]);

    // Backward pass (simplified - would need loss function)
    // バックワードパス（簡略化 - 損失関数が必要）
    
    // Gradient synchronization
    ddp_model.sync_gradients()?;

    // Clean up
    destroy_process_group()?;

    Ok(())
}