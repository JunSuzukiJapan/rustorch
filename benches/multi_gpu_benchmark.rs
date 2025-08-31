//! Multi-GPU Performance Benchmarking Suite
//! マルチGPUパフォーマンスベンチマークスイート
//!
//! Comprehensive benchmarking suite for multi-GPU operations including
//! distributed training, communication primitives, and synchronization.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::error::RusTorchResult;
use rustorch::gpu::distributed_training::{
    DistributedTrainer, FaultToleranceConfig, LearningRateSchedule, TrainingConfig,
};
use rustorch::gpu::multi_gpu::{GradientCompression, MultiGpuContext, ParallelismStrategy};
use rustorch::gpu::multi_gpu_profiler::profile_multi_gpu_operation;
use rustorch::gpu::sync_primitives::{MultiGpuBarrier, StreamManager, StreamPriority};
use rustorch::tensor::Tensor;
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark multi-GPU context creation and management
fn bench_multi_gpu_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultiGPU Context");

    // Test different GPU counts (simulated)
    for gpu_count in [1, 2, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("context_creation", gpu_count),
            gpu_count,
            |b, &gpu_count| {
                b.iter(|| {
                    let gpu_ids: Vec<usize> = (0..gpu_count).collect();
                    let _context = MultiGpuContext::new(black_box(gpu_ids));
                    black_box(_context)
                });
            },
        );
    }

    // Test different parallelism strategies
    let strategies = [
        ("DataParallel", ParallelismStrategy::DataParallel),
        ("ModelParallel", ParallelismStrategy::ModelParallel),
        ("PipelineParallel", ParallelismStrategy::PipelineParallel),
        ("Hybrid", ParallelismStrategy::Hybrid),
    ];

    for (name, strategy) in strategies.iter() {
        group.bench_with_input(
            BenchmarkId::new("strategy_creation", name),
            strategy,
            |b, &strategy| {
                b.iter(|| {
                    let _context =
                        MultiGpuContext::new_with_strategy(black_box(vec![0]), black_box(strategy));
                    black_box(_context)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU synchronization primitives
fn bench_synchronization_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("Synchronization");

    // Benchmark barrier operations
    group.bench_function("barrier_single_gpu", |b| {
        let barrier = MultiGpuBarrier::new(vec![0], Duration::from_secs(1));
        b.iter(|| {
            let result = barrier.wait(black_box(0));
            black_box(result)
        });
    });

    // Benchmark stream manager operations
    group.bench_function("stream_creation", |b| {
        b.iter(|| {
            let mut manager = StreamManager::new();
            let stream_id = manager.create_stream(black_box(0), black_box(StreamPriority::Normal));
            black_box(stream_id)
        });
    });

    group.bench_function("event_creation_and_recording", |b| {
        b.iter(|| {
            let mut manager = StreamManager::new();
            let stream_id = manager.create_stream(0, StreamPriority::Normal).unwrap();
            let event_id = manager.create_event(0).unwrap();
            let result = manager.record_event(black_box(stream_id), black_box(event_id));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark distributed training operations
fn bench_distributed_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distributed Training");

    // Setup training configuration
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

    // Benchmark trainer creation
    group.bench_function("trainer_creation", |b| {
        b.iter(|| {
            let trainer = DistributedTrainer::new(
                black_box(vec![0]),
                black_box(ParallelismStrategy::DataParallel),
                black_box(config.clone()),
            );
            black_box(trainer)
        });
    });

    // Benchmark gradient compression
    let compression_types = [
        ("ErrorFeedback", GradientCompression::ErrorFeedback),
        ("TopK", GradientCompression::TopK(10)),
        ("Quantization", GradientCompression::Quantization(8)),
    ];

    for (name, compression) in compression_types.iter() {
        group.bench_with_input(
            BenchmarkId::new("gradient_compression", name),
            compression,
            |b, &compression| {
                let mut config = config.clone();
                config.compression = compression;

                b.iter(|| {
                    let trainer = DistributedTrainer::new(
                        vec![0],
                        ParallelismStrategy::DataParallel,
                        black_box(config.clone()),
                    );
                    black_box(trainer)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tensor operations for multi-GPU
fn bench_tensor_multi_gpu_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor MultiGPU Operations");

    // Test different tensor sizes
    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("tensor_creation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    let tensor = Tensor::<f32>::from_vec(black_box(data), black_box(vec![size]));
                    black_box(tensor)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multi_gpu_context_with_tensor", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    let tensor = Tensor::<f32>::from_vec(data, vec![size]);
                    let context = MultiGpuContext::new(vec![0]);
                    black_box((tensor, context))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark communication patterns
fn bench_communication_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Communication Patterns");

    // Setup test context
    let setup = || -> RusTorchResult<MultiGpuContext> { MultiGpuContext::new(vec![0]) };

    group.bench_function("p2p_communication_setup", |b| {
        b.iter(|| {
            let _context = setup().unwrap();
            let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
            let result =
                _context.test_p2p_communication(black_box(0), black_box(0), black_box(&tensor));
            black_box(result)
        });
    });

    // Benchmark all-reduce operations with different tensor sizes
    let tensor_sizes = [10, 100, 1000];
    for size in tensor_sizes.iter() {
        group.bench_with_input(BenchmarkId::new("all_reduce", size), size, |b, &size| {
            b.iter(|| {
                let _context = setup().unwrap();
                let tensors: Vec<Tensor<f32>> = (0..1)
                    .map(|_| {
                        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                        Tensor::<f32>::from_vec(data, vec![size])
                    })
                    .collect();

                let result = _context.all_reduce(black_box(tensors));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark performance profiling overhead
#[allow(dead_code)]
fn bench_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Profiling Overhead");

    // Benchmark with profiling disabled
    group.bench_function("operation_without_profiling", |b| {
        b.iter(|| {
            let _context = MultiGpuContext::new(vec![0]).unwrap();
            let tensor = Tensor::<f32>::from_vec(vec![1.0; 1000], vec![1000]);
            let tensors = vec![tensor];
            let result = _context.all_reduce(black_box(tensors));
            black_box(result)
        });
    });

    // Benchmark with profiling enabled
    group.bench_function("operation_with_profiling", |b| {
        b.iter(|| {
            let result = profile_multi_gpu_operation("benchmark_operation", &[0], || {
                let _context = MultiGpuContext::new(vec![0])?;
                let tensor = Tensor::<f32>::from_vec(vec![1.0; 1000], vec![1000]);
                let tensors = vec![tensor];
                _context.all_reduce(tensors)
            });
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark different gradient compression methods
#[allow(dead_code)]
fn bench_gradient_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient Compression");

    let compression_methods = [
        ("ErrorFeedback", GradientCompression::ErrorFeedback),
        ("TopK_10", GradientCompression::TopK(10)),
        ("TopK_50", GradientCompression::TopK(50)),
        ("Quantization_8bit", GradientCompression::Quantization(8)),
        ("Quantization_4bit", GradientCompression::Quantization(4)),
    ];

    for (name, compression) in compression_methods.iter() {
        group.bench_with_input(
            BenchmarkId::new("compression_method", name),
            compression,
            |b, &compression| {
                let config = TrainingConfig {
                    sync_frequency: 1,
                    compression,
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

                b.iter(|| {
                    let trainer = DistributedTrainer::new(
                        vec![0],
                        ParallelismStrategy::DataParallel,
                        black_box(config.clone()),
                    );
                    black_box(trainer)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark training step performance
#[allow(dead_code)]
fn bench_training_step_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Training Step Performance");

    // Setup test trainer
    let setup_trainer = || -> RusTorchResult<DistributedTrainer> {
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

        DistributedTrainer::new(vec![0], ParallelismStrategy::DataParallel, config)
    };

    // Benchmark different parameter sizes
    let param_sizes = [100, 1000, 5000];
    for size in param_sizes.iter() {
        group.bench_with_input(BenchmarkId::new("training_step", size), size, |b, &size| {
            b.iter(|| {
                let mut trainer = setup_trainer().unwrap();

                // Create mock parameters and gradients
                let mut parameters = HashMap::new();
                let mut gradients = HashMap::new();

                for i in 0..5 {
                    // 5 parameters
                    let param_name = format!("param_{}", i);
                    let param_data: Vec<f32> =
                        (0..size).map(|j| (i * size + j) as f32 * 0.01).collect();
                    let grad_data: Vec<f32> = (0..size).map(|j| (j as f32 * 0.001)).collect();

                    parameters.insert(
                        param_name.clone(),
                        Tensor::<f32>::from_vec(param_data, vec![size]),
                    );
                    gradients.insert(
                        param_name,
                        vec![Tensor::<f32>::from_vec(grad_data, vec![size])],
                    );
                }

                let result = trainer.training_step(black_box(&parameters), black_box(gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark stream priority performance
#[allow(dead_code)]
fn bench_stream_priorities(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stream Priorities");

    let priorities = [
        ("Low", StreamPriority::Low),
        ("Normal", StreamPriority::Normal),
        ("High", StreamPriority::High),
        ("Critical", StreamPriority::Critical),
    ];

    for (name, priority) in priorities.iter() {
        group.bench_with_input(
            BenchmarkId::new("stream_creation", name),
            priority,
            |b, &priority| {
                b.iter(|| {
                    let mut manager = StreamManager::new();
                    let result = manager.create_stream(black_box(0), black_box(priority));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark profiler performance impact
fn bench_profiler_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("Profiler Impact");

    // Create test operation
    let test_operation = || -> RusTorchResult<()> {
        let _context = MultiGpuContext::new(vec![0])?;
        let tensor = Tensor::<f32>::from_vec(vec![1.0; 1000], vec![1000]);
        let tensors = vec![tensor];
        let _result = _context.all_reduce(tensors)?;
        Ok(())
    };

    // Benchmark without profiling
    group.bench_function("without_profiling", |b| {
        b.iter(|| {
            let result = test_operation();
            black_box(result)
        });
    });

    // Benchmark with profiling
    group.bench_function("with_profiling", |b| {
        b.iter(|| {
            let result = profile_multi_gpu_operation("benchmark_test", &[0], || test_operation());
            black_box(result)
        });
    });

    group.finish();
}

/// Comprehensive multi-GPU benchmark suite
fn bench_comprehensive_multi_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive MultiGPU");

    // Full pipeline benchmark: context + training + profiling
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let result = profile_multi_gpu_operation(
                "full_pipeline_test",
                &[0],
                || -> RusTorchResult<f32> {
                    // Create context
                    let _context = MultiGpuContext::new(vec![0])?;

                    // Create trainer
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

                    let mut trainer = DistributedTrainer::new(
                        vec![0],
                        ParallelismStrategy::DataParallel,
                        config,
                    )?;

                    // Simulate training step
                    let mut parameters = HashMap::new();
                    let mut gradients = HashMap::new();

                    parameters.insert(
                        "weight".to_string(),
                        Tensor::<f32>::from_vec(vec![1.0; 100], vec![100]),
                    );
                    gradients.insert(
                        "weight".to_string(),
                        vec![Tensor::<f32>::from_vec(vec![0.01; 100], vec![100])],
                    );

                    let _updated_params = trainer.training_step(&parameters, gradients)?;

                    // Return performance score
                    Ok(95.5)
                },
            );
            black_box(result)
        });
    });

    group.finish();
}

/// Stress test for high-frequency operations
fn bench_stress_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stress Test");
    group.sample_size(20); // Reduce sample size for stress tests

    // High-frequency barrier operations
    group.bench_function("rapid_barrier_operations", |b| {
        let barrier = MultiGpuBarrier::new(vec![0], Duration::from_millis(100));
        b.iter(|| {
            for _ in 0..10 {
                let result = barrier.wait(black_box(0));
                black_box(result).ok();
                barrier.reset();
            }
        });
    });

    // High-frequency stream operations
    group.bench_function("rapid_stream_operations", |b| {
        b.iter(|| {
            let mut manager = StreamManager::new();
            for _ in 0..10 {
                let stream_id = manager.create_stream(0, StreamPriority::Normal).unwrap();
                let event_id = manager.create_event(0).unwrap();
                let _result = manager.record_event(stream_id, event_id);
            }
            black_box(manager)
        });
    });

    group.finish();
}

// Benchmark group definitions
criterion_group!(
    multi_gpu_benches,
    bench_multi_gpu_context,
    bench_synchronization_primitives,
    bench_distributed_training,
    bench_tensor_multi_gpu_ops,
    bench_communication_patterns,
    bench_profiler_impact,
    bench_comprehensive_multi_gpu,
    bench_stress_test
);

criterion_main!(multi_gpu_benches);
