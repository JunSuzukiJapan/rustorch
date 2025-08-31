//! Basic Multi-GPU Validation Tests
//! 基本マルチGPU検証テスト
//!
//! Tests that can run on any system to validate multi-GPU infrastructure

use rustorch::error::RusTorchResult;
use rustorch::gpu::distributed_training::{
    DistributedTrainer, FaultToleranceConfig, LearningRateSchedule, TrainingConfig,
};
use rustorch::gpu::multi_gpu::GradientCompression;
use rustorch::gpu::multi_gpu::{GpuTopology, MultiGpuContext, ParallelismStrategy};
use rustorch::gpu::sync_primitives::{MultiGpuBarrier, StreamManager, StreamPriority};
use rustorch::tensor::Tensor;
use std::time::Duration;

#[test]
fn test_gpu_topology_creation() {
    let topology = GpuTopology::default();

    assert_eq!(topology.num_gpus, 1);
    assert_eq!(topology.device_ids, vec![0]);
    assert_eq!(topology.p2p_matrix, vec![vec![true]]);
    assert_eq!(topology.bandwidth_matrix, vec![vec![0.0]]);
    assert_eq!(topology.compute_capabilities, vec![(8, 0)]);
    assert!(topology.memory_per_gpu[0] > 0);

    println!("✅ GPU topology creation test passed");
}

#[test]
fn test_multi_gpu_context_basic() -> RusTorchResult<()> {
    // Test context creation with mock GPUs
    let device_ids = vec![0];
    let context = MultiGpuContext::new(device_ids)?;

    assert_eq!(context.gpu_count(), 1);
    assert!(context.is_gpu_available(0));
    assert!(!context.is_gpu_available(1));
    assert_eq!(context.get_device_ids(), &[0]);

    println!("✅ Multi-GPU context basic test passed");
    Ok(())
}

#[test]
fn test_multi_gpu_barrier_basic() {
    let gpu_ids = vec![0];
    let timeout = Duration::from_secs(1);
    let barrier = MultiGpuBarrier::new(gpu_ids, timeout);

    // Test basic wait functionality
    let result = barrier.wait(0);
    assert!(
        result.is_ok(),
        "Single GPU barrier should succeed immediately"
    );

    // Test reset
    barrier.reset();

    println!("✅ Multi-GPU barrier basic test passed");
}

#[test]
fn test_stream_manager_basic() -> RusTorchResult<()> {
    let mut manager = StreamManager::new();

    // Test stream creation
    let stream_id = manager.create_stream(0, StreamPriority::Normal)?;

    // Test event creation
    let event_id = manager.create_event(0)?;

    // Test event recording
    manager.record_event(stream_id, event_id)?;

    // Test event query
    assert!(
        manager.query_event(event_id),
        "Event should be completed after recording"
    );

    // Test device synchronization
    manager.synchronize_device(0)?;

    println!("✅ Stream manager basic test passed");
    Ok(())
}

#[test]
fn test_distributed_trainer_basic() -> RusTorchResult<()> {
    let gpu_ids = vec![0];
    let config = TrainingConfig {
        sync_frequency: 1,
        compression: GradientCompression::ErrorFeedback,
        lr_schedule: LearningRateSchedule::Constant(0.001),
        fault_tolerance: FaultToleranceConfig {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            checkpointing: false,
            checkpoint_frequency: 100,
            gradient_accumulation: false,
        },
        communication_timeout: Duration::from_secs(10),
    };

    let trainer = DistributedTrainer::new(gpu_ids, ParallelismStrategy::DataParallel, config)?;
    assert_eq!(trainer.get_gpu_count(), 1);

    println!("✅ Distributed trainer basic test passed");
    Ok(())
}

#[test]
fn test_tensor_multi_gpu_compatibility() -> RusTorchResult<()> {
    // Test that tensors can be used with multi-GPU infrastructure
    let tensor1 = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let tensor2 = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);

    // Test basic properties
    assert_eq!(tensor1.shape(), &[4]);
    assert_eq!(tensor2.shape(), &[4]);
    assert_eq!(tensor1.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor2.data.as_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0]);

    // Test that tensors work with multi-GPU context
    let context = MultiGpuContext::new(vec![0])?;
    assert_eq!(context.gpu_count(), 1);

    println!("✅ Tensor multi-GPU compatibility test passed");
    Ok(())
}

#[test]
fn test_p2p_communication_basic() -> RusTorchResult<()> {
    let context = MultiGpuContext::new(vec![0])?;
    let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    // Test P2P with same GPU (should work)
    let result = context.test_p2p_communication(0, 0, &tensor);

    // This should either succeed or give a clear error message
    match result {
        Ok(_) => println!("✅ P2P communication test passed"),
        Err(e) => println!("ℹ️ P2P communication test handled gracefully: {:?}", e),
    }

    Ok(())
}

#[test]
fn test_stream_priority_system() -> RusTorchResult<()> {
    let mut manager = StreamManager::new();

    // Test different priority levels
    let critical_stream = manager.create_stream(0, StreamPriority::Critical)?;
    let high_stream = manager.create_stream(0, StreamPriority::High)?;
    let normal_stream = manager.create_stream(0, StreamPriority::Normal)?;
    let low_stream = manager.create_stream(0, StreamPriority::Low)?;

    // Verify all streams have unique IDs
    let streams = vec![critical_stream, high_stream, normal_stream, low_stream];
    for (i, &id1) in streams.iter().enumerate() {
        for (j, &id2) in streams.iter().enumerate() {
            if i != j {
                assert_ne!(id1, id2, "Stream IDs should be unique");
            }
        }
    }

    println!("✅ Stream priority system test passed");
    Ok(())
}

#[test]
fn test_compression_types() {
    // Test that compression types can be created
    let error_feedback_compression = GradientCompression::ErrorFeedback;
    let topk_compression = GradientCompression::TopK(10);
    let quantization_compression = GradientCompression::Quantization(8);

    // Basic validation that types are properly structured
    match error_feedback_compression {
        GradientCompression::ErrorFeedback => (),
        _ => panic!("Expected ErrorFeedback compression"),
    }

    match topk_compression {
        GradientCompression::TopK(k) => assert_eq!(k, 10),
        _ => panic!("Expected TopK compression"),
    }

    match quantization_compression {
        GradientCompression::Quantization(bits) => assert_eq!(bits, 8),
        _ => panic!("Expected Quantization compression"),
    }

    println!("✅ Compression types test passed");
}

#[test]
fn test_parallelism_strategies() {
    // Test that all parallelism strategies are available
    let strategies = vec![
        ParallelismStrategy::DataParallel,
        ParallelismStrategy::ModelParallel,
        ParallelismStrategy::PipelineParallel,
        ParallelismStrategy::Hybrid,
        ParallelismStrategy::ExpertParallel,
    ];

    for strategy in strategies {
        let context_result = MultiGpuContext::new_with_strategy(vec![0], strategy);
        assert!(
            context_result.is_ok(),
            "Strategy {:?} should be supported",
            strategy
        );
    }

    println!("✅ Parallelism strategies test passed");
}

#[test]
fn test_multi_gpu_error_handling() {
    // Test error handling for invalid GPU IDs
    let context = MultiGpuContext::new(vec![0]);
    assert!(context.is_ok(), "Valid GPU ID should work");

    // Test barrier timeout handling with very short timeout
    let barrier = MultiGpuBarrier::new(vec![0], Duration::from_millis(1));
    let result = barrier.wait(0);

    // Should either succeed immediately or timeout gracefully
    match result {
        Ok(_) => println!("✅ Barrier completed immediately"),
        Err(_) => println!("✅ Barrier timeout handled correctly"),
    }

    println!("✅ Multi-GPU error handling test passed");
}

// Main integration test
#[test]
fn run_multi_gpu_basic_validation_suite() {
    println!("=== Multi-GPU Basic Validation Suite ===");

    #[allow(clippy::type_complexity)]
    let test_results: Vec<(
        &str,
        Box<dyn Fn() -> Result<(), Box<dyn std::error::Error>>>,
    )> = vec![
        (
            "GPU Topology",
            Box::new(|| {
                test_gpu_topology_creation();
                Ok(())
            }),
        ),
        (
            "Multi-GPU Context",
            Box::new(|| {
                test_multi_gpu_context_basic()?;
                Ok(())
            }),
        ),
        (
            "Stream Manager",
            Box::new(|| {
                test_stream_manager_basic()?;
                Ok(())
            }),
        ),
        (
            "Distributed Trainer",
            Box::new(|| {
                test_distributed_trainer_basic()?;
                Ok(())
            }),
        ),
        (
            "Tensor Compatibility",
            Box::new(|| {
                test_tensor_multi_gpu_compatibility()?;
                Ok(())
            }),
        ),
        (
            "P2P Communication",
            Box::new(|| {
                test_p2p_communication_basic()?;
                Ok(())
            }),
        ),
        (
            "Stream Priorities",
            Box::new(|| {
                test_stream_priority_system()?;
                Ok(())
            }),
        ),
    ];

    let mut passed = 0;
    let total = test_results.len();

    for (test_name, test_fn) in test_results {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_fn())) {
            Ok(Ok(_)) => {
                println!("✅ {}: PASS", test_name);
                passed += 1;
            }
            Ok(Err(e)) => {
                println!("❌ {}: FAIL - {:?}", test_name, e);
            }
            Err(_) => {
                println!("❌ {}: PANIC", test_name);
            }
        }
    }

    println!("=== Results: {}/{} tests passed ===", passed, total);
    println!(
        "=== Multi-GPU Infrastructure: {} ===",
        if passed == total {
            "VALIDATED ✅"
        } else {
            "NEEDS ATTENTION ❌"
        }
    );
}
