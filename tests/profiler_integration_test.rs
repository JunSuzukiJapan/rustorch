//! Performance Profiler Integration Tests
//! パフォーマンスプロファイラー統合テスト
//!
//! Comprehensive tests for the integrated profiling system including
//! multi-GPU profiling, metrics collection, and automated performance testing.

use rustorch::error::RusTorchResult;
use rustorch::gpu::distributed_training::{DistributedTrainer, TrainingConfig};
use rustorch::gpu::multi_gpu::{GradientCompression, ParallelismStrategy};
use rustorch::gpu::multi_gpu_profiler::MultiGpuProfiler;
use rustorch::profiler::ProfilerConfig;
use rustorch::profiler::{
    clear_profiler, disable_profiler, enable_profiler, ProfileContext, RusTorchProfiler,
};
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_distributed_trainer_profiling_integration() -> RusTorchResult<()> {
    // Clean profiler state
    disable_profiler();
    clear_profiler();
    enable_profiler();

    // Create distributed trainer with profiling enabled
    let config = TrainingConfig {
        sync_frequency: 1,
        compression: GradientCompression::TopK(100),
        lr_schedule: rustorch::gpu::distributed_training::LearningRateSchedule::Constant(0.01),
        fault_tolerance: rustorch::gpu::distributed_training::FaultToleranceConfig {
            max_retries: 2,
            retry_delay: Duration::from_millis(50),
            checkpointing: false,
            checkpoint_frequency: 100,
            gradient_accumulation: false,
        },
        communication_timeout: Duration::from_secs(10),
    };

    let mut trainer = DistributedTrainer::new(
        vec![0, 1], // Mock GPU IDs
        ParallelismStrategy::DataParallel,
        config,
    )?;

    // Enable profiling
    trainer.enable_profiling()?;

    // Simulate training step
    let model_params = HashMap::new();
    let gradients = HashMap::new();

    {
        let _ctx = ProfileContext::new("training_step_with_profiling");
        let _result = trainer.training_step(&model_params, gradients)?;
    }

    // Get profiling report from trainer
    let report = trainer.get_profiling_report();
    assert!(report.is_some(), "Profiling report should be available");

    let report = report.unwrap();
    assert!(
        report.session_duration > Duration::ZERO,
        "Session should have non-zero duration"
    );

    // Clean up
    disable_profiler();
    clear_profiler();

    Ok(())
}

#[test]
fn test_multi_gpu_profiler_standalone() -> RusTorchResult<()> {
    let config = ProfilerConfig {
        level: rustorch::profiler::ProfilingLevel::Comprehensive,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_system_metrics: true,
        enable_call_stack: false,
        max_session_duration: Some(300),
        metrics_buffer_size: 1000,
        sampling_rate: 20.0,
        export_chrome_trace: false,
        export_tensorboard: false,
        export_json: true,
    };

    let profiler = MultiGpuProfiler::new(vec![0, 1], config)?;

    // Record some mock training steps
    let _ = profiler.record_training_step(
        Duration::from_millis(50), // forward
        Duration::from_millis(30), // backward
        Duration::from_millis(10), // update
        Duration::from_millis(20), // sync
        100.0,                     // throughput
    );
    let _ = profiler.record_training_step(
        Duration::from_millis(55), // forward
        Duration::from_millis(35), // backward
        Duration::from_millis(12), // update
        Duration::from_millis(25), // sync
        110.0,                     // throughput
    );
    let _ = profiler.record_training_step(
        Duration::from_millis(45), // forward
        Duration::from_millis(28), // backward
        Duration::from_millis(8),  // update
        Duration::from_millis(18), // sync
        95.0,                      // throughput
    );

    let report = profiler.generate_report();

    assert!(report.session_duration > Duration::ZERO);
    assert_eq!(report.total_training_steps, 3);
    assert!(report.avg_step_time > Duration::ZERO);
    assert!(report.communication_overhead_ratio >= 0.0);
    assert!(report.communication_overhead_ratio <= 1.0);

    Ok(())
}

#[test]
fn test_rustorch_profiler_integration() -> RusTorchResult<()> {
    let config = ProfilerConfig {
        level: rustorch::profiler::ProfilingLevel::Standard,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_system_metrics: false,
        enable_call_stack: false,
        max_session_duration: Some(300),
        metrics_buffer_size: 1000,
        sampling_rate: 10.0,
        export_chrome_trace: false,
        export_tensorboard: false,
        export_json: true,
    };

    let mut profiler = RusTorchProfiler::new(config);

    // Enable multi-GPU profiling
    profiler.enable_multi_gpu_profiling(vec![0, 1])?;

    // Record some operations
    profiler.record_operation("tensor_multiply", Duration::from_millis(50));
    profiler.record_operation("tensor_add", Duration::from_millis(10));
    profiler.record_operation("tensor_multiply", Duration::from_millis(45));

    // Take memory snapshots
    let mut gpu_memory = HashMap::new();
    gpu_memory.insert(0, 1024 * 1024); // 1MB
    gpu_memory.insert(1, 2048 * 1024); // 2MB
    profiler.take_memory_snapshot(512 * 1024, gpu_memory); // 512KB host memory

    // Generate comprehensive report
    let report = profiler.generate_report();

    assert!(report.session_duration > Duration::ZERO);
    assert_eq!(report.total_operations, 3);
    assert!(report
        .operation_summary
        .operations
        .contains_key("tensor_multiply"));
    assert!(report
        .operation_summary
        .operations
        .contains_key("tensor_add"));

    // Check multiply operation stats
    let multiply_stats = &report.operation_summary.operations["tensor_multiply"];
    assert_eq!(multiply_stats.count, 2);
    assert!(multiply_stats.avg_time > Duration::ZERO);

    // Check memory analysis
    assert_eq!(report.memory_analysis.total_snapshots, 1);
    assert!(report.memory_analysis.average_usage > 0);

    // Check recommendations
    assert!(!report.recommendations.is_empty());

    Ok(())
}

#[test]
fn test_performance_metrics_collection() -> RusTorchResult<()> {
    // Test the metrics collection system integrated with profiling
    let config = ProfilerConfig {
        level: rustorch::profiler::ProfilingLevel::Standard,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_system_metrics: false,
        enable_call_stack: false,
        max_session_duration: Some(60),
        metrics_buffer_size: 1000,
        sampling_rate: 100.0,
        export_chrome_trace: false,
        export_tensorboard: false,
        export_json: true,
    };

    let profiler = RusTorchProfiler::new(config);

    // Simulate multiple operations with different performance characteristics
    for i in 0..10 {
        let operation_name = if i % 2 == 0 { "fast_op" } else { "slow_op" };
        let duration = if i % 2 == 0 {
            Duration::from_millis(10)
        } else {
            Duration::from_millis(100)
        };

        profiler.record_operation(operation_name, duration);

        // Simulate varying memory usage
        let memory_usage = 1024 * 1024 * (i + 1); // Increasing memory usage
        let gpu_memory = HashMap::from([(0, memory_usage / 2)]);
        profiler.take_memory_snapshot(memory_usage, gpu_memory);
    }

    let report = profiler.generate_report();

    // Verify metrics collection
    assert_eq!(report.total_operations, 10);
    assert!(report.operation_summary.operations.contains_key("fast_op"));
    assert!(report.operation_summary.operations.contains_key("slow_op"));

    // Check operation statistics
    let fast_stats = &report.operation_summary.operations["fast_op"];
    let slow_stats = &report.operation_summary.operations["slow_op"];

    assert_eq!(fast_stats.count, 5);
    assert_eq!(slow_stats.count, 5);
    assert!(slow_stats.avg_time > fast_stats.avg_time);

    // Verify memory analysis
    assert_eq!(report.memory_analysis.total_snapshots, 10);
    assert!(report.memory_analysis.peak_usage > report.memory_analysis.average_usage);

    // Check for performance recommendations
    assert!(!report.recommendations.is_empty());

    Ok(())
}

#[test]
fn test_automated_performance_testing() -> RusTorchResult<()> {
    // Test automated performance testing workflow
    let config = ProfilerConfig {
        level: rustorch::profiler::ProfilingLevel::Comprehensive,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_system_metrics: true,
        enable_call_stack: false,
        max_session_duration: Some(120),
        metrics_buffer_size: 2000,
        sampling_rate: 50.0,
        export_chrome_trace: false,
        export_tensorboard: false,
        export_json: true,
    };

    let profiler = RusTorchProfiler::new(config);

    // Simulate automated test workflow
    let test_operations = vec![
        ("tensor_creation", Duration::from_millis(5)),
        ("tensor_multiplication", Duration::from_millis(15)),
        ("gpu_transfer", Duration::from_millis(50)),
        ("gradient_computation", Duration::from_millis(30)),
        ("parameter_update", Duration::from_millis(10)),
    ];

    // Record operations multiple times to simulate real training
    for iteration in 0..5 {
        for (op_name, base_duration) in &test_operations {
            // Add some variance to make it realistic
            let variance = Duration::from_millis(iteration * 2);
            let actual_duration = *base_duration + variance;

            profiler.record_operation(op_name, actual_duration);
        }

        // Take memory snapshot for each iteration
        let memory_usage = 1024 * 1024 * (iteration + 1) as usize;
        let gpu_memory = HashMap::from([(0, memory_usage / 2), (1, memory_usage / 3)]);
        profiler.take_memory_snapshot(memory_usage, gpu_memory);
    }

    let report = profiler.generate_report();

    // Verify comprehensive testing results
    assert_eq!(report.total_operations, 25); // 5 operations × 5 iterations
    assert_eq!(report.operation_summary.operations.len(), 5);
    assert_eq!(report.memory_analysis.total_snapshots, 5);

    // Check that all operations are tracked
    for (op_name, _) in &test_operations {
        assert!(report.operation_summary.operations.contains_key(*op_name));
        let stats = &report.operation_summary.operations[*op_name];
        assert_eq!(stats.count, 5);
        assert!(stats.total_time > Duration::ZERO);
        assert!(stats.avg_time > Duration::ZERO);
    }

    // Verify performance analysis
    assert!(report.operation_summary.slowest_operation.is_some());
    let (slowest_name, _) = report.operation_summary.slowest_operation.unwrap();
    assert_eq!(slowest_name, "gpu_transfer"); // Should be the slowest operation

    // Check memory trend analysis
    use rustorch::profiler::MemoryTrend;
    assert!(matches!(
        report.memory_analysis.memory_trend,
        MemoryTrend::Increasing
    ));

    Ok(())
}

#[test]
fn test_profiler_recommendations() -> RusTorchResult<()> {
    let config = ProfilerConfig {
        level: rustorch::profiler::ProfilingLevel::Verbose,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_system_metrics: true,
        enable_call_stack: true,
        max_session_duration: Some(300),
        metrics_buffer_size: 5000,
        sampling_rate: 10.0,
        export_chrome_trace: true,
        export_tensorboard: false,
        export_json: true,
    };

    let profiler = RusTorchProfiler::new(config);

    // Create scenario that should trigger recommendations
    // Record a very slow operation
    profiler.record_operation("slow_tensor_operation", Duration::from_millis(2000)); // 2 seconds

    // Record normal operations
    profiler.record_operation("normal_operation", Duration::from_millis(10));
    profiler.record_operation("normal_operation", Duration::from_millis(12));

    // Create increasing memory usage pattern
    for i in 1..=5 {
        let memory = 1024 * 1024 * i * 2; // Rapidly increasing memory
        profiler.take_memory_snapshot(memory, HashMap::new());
    }

    let report = profiler.generate_report();

    // Should have recommendations for slow operation and memory growth
    assert!(!report.recommendations.is_empty());

    let slow_op_recommendation = report
        .recommendations
        .iter()
        .any(|r| r.contains("slow_tensor_operation") && r.contains("optimization"));
    assert!(
        slow_op_recommendation,
        "Should recommend optimizing slow operation"
    );

    let memory_recommendation = report
        .recommendations
        .iter()
        .any(|r| r.contains("Memory") && r.contains("increasing"));
    assert!(
        memory_recommendation,
        "Should recommend checking memory usage"
    );

    Ok(())
}
