//! Distributed training performance benchmarks
//! 分散学習パフォーマンスベンチマーク

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::autograd::Variable;
use rustorch::distributed::*;
use rustorch::nn::Linear;
use rustorch::tensor::Tensor;
use std::time::Duration;

/// Benchmark distributed initialization
/// 分散初期化ベンチマーク
fn bench_distributed_init(c: &mut Criterion) {
    c.bench_function("distributed_init", |b| {
        b.iter(|| {
            std::env::set_var("RANK", "0");
            std::env::set_var("WORLD_SIZE", "1");
            std::env::set_var("MASTER_ADDR", "localhost");
            std::env::set_var("MASTER_PORT", "29600");

            let _ = init_process_group(
                DistributedBackend::TCP,
                Some("tcp://localhost:29600"),
                Some(1),
                Some(0),
                Some(Duration::from_secs(10)),
            );

            let _ = destroy_process_group();
        });
    });
}

/// Benchmark all-reduce operations with different tensor sizes
/// 異なるテンソルサイズでのall-reduce操作ベンチマーク
fn bench_all_reduce_sizes(c: &mut Criterion) {
    // Initialize distributed once for all benchmarks
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29601");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let sizes = vec![
        (100, "small"),
        (1000, "medium"),
        (10000, "large"),
        (100000, "very_large"),
    ];

    let mut group = c.benchmark_group("all_reduce_sizes");

    for (size, label) in sizes {
        group.bench_with_input(BenchmarkId::new("all_reduce", label), &size, |b, &size| {
            let mut tensor: Tensor<f32> = Tensor::randn(&[size]);
            b.iter(|| {
                let _ = all_reduce(&mut tensor, ReduceOp::Sum, None, false);
            });
        });
    }

    group.finish();

    // Clean up
    let _ = destroy_process_group();
}

/// Benchmark DDP forward pass
/// DDPフォワードパスベンチマーク
fn bench_ddp_forward(c: &mut Criterion) {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29602");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let model_sizes = vec![
        (10, 5, "tiny"),
        (100, 50, "small"),
        (784, 128, "medium"),
        (2048, 1024, "large"),
    ];

    let mut group = c.benchmark_group("ddp_forward");

    for (input_size, output_size, label) in model_sizes {
        let linear: Linear<f32> = Linear::new(input_size, output_size);
        let ddp = wrap_simple(linear, Some(vec![0])).unwrap();
        let input_tensor: Tensor<f32> = Tensor::randn(&[32, input_size]);
        let input = Variable::new(input_tensor, false);

        group.bench_with_input(
            BenchmarkId::new("ddp_forward", label),
            &input_size,
            |b, _| {
                b.iter(|| {
                    let _ = ddp.forward(&input);
                });
            },
        );
    }

    group.finish();

    let _ = destroy_process_group();
}

/// Benchmark gradient synchronization
/// 勾配同期ベンチマーク
fn bench_gradient_sync(c: &mut Criterion) {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29603");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let param_counts = vec![
        (10, "few_params"),
        (100, "many_params"),
        (1000, "lots_params"),
    ];

    let mut group = c.benchmark_group("gradient_sync");

    for (param_count, label) in param_counts {
        // Create model with specified number of parameters
        let linear: Linear<f32> = Linear::new(param_count, param_count);
        let ddp = wrap_simple(linear, Some(vec![0])).unwrap();

        group.bench_with_input(
            BenchmarkId::new("grad_sync", label),
            &param_count,
            |b, _| {
                b.iter(|| {
                    let _ = ddp.sync_gradients();
                });
            },
        );
    }

    group.finish();

    let _ = destroy_process_group();
}

/// Benchmark async gradient operations
/// 非同期勾配操作ベンチマーク
#[cfg(feature = "nccl")]
fn bench_async_gradient_ops(c: &mut Criterion) {
    use rustorch::distributed::async_gradient::{AsyncConfig, AsyncGradientSynchronizer, Priority};

    let config = AsyncConfig::default();
    let synchronizer = AsyncGradientSynchronizer::new(config).unwrap();

    c.bench_function("async_gradient_submit", |b| {
        let gradient: Tensor<f32> = Tensor::randn(&[1000, 1000]);
        b.iter(|| {
            let _ = synchronizer.submit_gradient(
                "benchmark_param".to_string(),
                gradient.clone(),
                Priority::Normal,
            );
        });
    });
}

/// Benchmark communication patterns
/// 通信パターンベンチマーク
fn bench_communication_patterns(c: &mut Criterion) {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29604");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    let mut group = c.benchmark_group("communication_patterns");

    // Benchmark different communication operations
    let tensor: Tensor<f32> = Tensor::randn(&[1000, 1000]);

    group.bench_function("all_reduce", |b| {
        b.iter(|| {
            let mut tensor_copy = tensor.clone();
            let _ = all_reduce(&mut tensor_copy, ReduceOp::Sum, None, false);
        });
    });

    group.bench_function("broadcast", |b| {
        b.iter(|| {
            let mut tensor_copy = tensor.clone();
            let _ = broadcast(&mut tensor_copy, 0, None, false);
        });
    });

    group.bench_function("all_gather", |b| {
        b.iter(|| {
            let mut tensor_list = Vec::new();
            let _ = all_gather(&mut tensor_list, &tensor, None, false);
        });
    });

    group.finish();

    let _ = destroy_process_group();
}

/// Memory efficiency benchmark
/// メモリ効率ベンチマーク
fn bench_memory_efficiency(c: &mut Criterion) {
    std::env::set_var("RANK", "0");
    std::env::set_var("WORLD_SIZE", "1");
    std::env::set_var("MASTER_ADDR", "localhost");
    std::env::set_var("MASTER_PORT", "29605");

    let _ = init_process_group(DistributedBackend::TCP, None, None, None, None);

    c.bench_function("ddp_memory_overhead", |b| {
        b.iter(|| {
            let linear: Linear<f32> = Linear::new(1000, 1000);
            let _ddp = wrap_simple(linear, Some(vec![0])).unwrap();
            // Measure memory usage difference
        });
    });

    let _ = destroy_process_group();
}

#[cfg(feature = "nccl")]
criterion_group!(
    benches,
    bench_distributed_init,
    bench_all_reduce_sizes,
    bench_ddp_forward,
    bench_gradient_sync,
    bench_async_gradient_ops,
    bench_communication_patterns,
    bench_memory_efficiency,
);

#[cfg(not(feature = "nccl"))]
criterion_group!(
    benches,
    bench_distributed_init,
    bench_all_reduce_sizes,
    bench_ddp_forward,
    bench_gradient_sync,
    bench_communication_patterns,
    bench_memory_efficiency,
);

criterion_main!(benches);
