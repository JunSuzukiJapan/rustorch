//! GPU統合パフォーマンスベンチマーク
//! GPU integration performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::tensor::{Tensor, gpu_parallel::*};
use rustorch::gpu::{DeviceType, set_device, current_device};

/// GPU並列操作ベンチマーク
/// GPU parallel operations benchmarks
fn bench_gpu_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_parallel_operations");
    
    // テストサイズ
    let sizes = vec![100, 1000, 10000, 100000];
    
    for size in sizes {
        // GPU要素ごと演算ベンチマーク
        group.bench_with_input(
            BenchmarkId::new("gpu_elementwise_add", size),
            &size,
            |b, &size| {
                let tensor1 = Tensor::<f32>::ones(&[size]);
                let tensor2 = Tensor::<f32>::ones(&[size]);
                
                b.iter(|| {
                    let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
                    black_box(result)
                });
            },
        );
        
        // GPU行列乗算ベンチマーク
        if size <= 1000 { // 大きすぎる行列は避ける
            let matrix_size = (size as f64).sqrt() as usize;
            group.bench_with_input(
                BenchmarkId::new("gpu_matmul", matrix_size),
                &matrix_size,
                |b, &matrix_size| {
                    let tensor1 = Tensor::<f32>::ones(&[matrix_size, matrix_size]);
                    let tensor2 = Tensor::<f32>::ones(&[matrix_size, matrix_size]);
                    
                    b.iter(|| {
                        let result = tensor1.gpu_matmul(&tensor2);
                        black_box(result)
                    });
                },
            );
        }
        
        // GPUリダクションベンチマーク
        group.bench_with_input(
            BenchmarkId::new("gpu_reduce_sum", size),
            &size,
            |b, &size| {
                let tensor = Tensor::<f32>::ones(&[size]);
                
                b.iter(|| {
                    let result = tensor.gpu_reduce(0, 0.0f32, |acc, x| acc + x);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// GPU-CPU転送ベンチマーク
/// GPU-CPU transfer benchmarks
fn bench_gpu_cpu_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_transfer");
    
    let sizes = vec![1000, 10000, 100000, 1000000];
    
    for size in sizes {
        // CPU->GPU転送
        group.bench_with_input(
            BenchmarkId::new("cpu_to_gpu", size),
            &size,
            |b, &size| {
                let tensor = Tensor::<f32>::ones(&[size]);
                
                b.iter(|| {
                    let result = tensor.to_device(DeviceType::Cuda(0));
                    black_box(result)
                });
            },
        );
        
        // GPU->CPU転送
        group.bench_with_input(
            BenchmarkId::new("gpu_to_cpu", size),
            &size,
            |b, &size| {
                let tensor = Tensor::<f32>::ones(&[size]);
                // GPU上にあると仮定
                
                b.iter(|| {
                    let result = tensor.to_cpu();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// GPU戦略比較ベンチマーク
/// GPU strategy comparison benchmarks
fn bench_gpu_strategy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_strategy_comparison");
    
    let size = 10000;
    let tensor1 = Tensor::<f32>::ones(&[size]);
    let tensor2 = Tensor::<f32>::ones(&[size]);
    
    // CPU並列戦略
    group.bench_function("cpu_parallel_strategy", |b| {
        let mut ctx = GpuParallelContext::new(GpuParallelConfig {
            gpu_strategy: GpuParallelStrategy::CpuParallel,
            ..Default::default()
        });
        
        b.iter(|| {
            let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
            black_box(result)
        });
    });
    
    // GPU優先戦略
    group.bench_function("gpu_preferred_strategy", |b| {
        let mut ctx = GpuParallelContext::new(GpuParallelConfig {
            gpu_strategy: GpuParallelStrategy::GpuPreferred,
            ..Default::default()
        });
        
        b.iter(|| {
            let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
            black_box(result)
        });
    });
    
    // 自動戦略
    group.bench_function("auto_strategy", |b| {
        let mut ctx = GpuParallelContext::new(GpuParallelConfig {
            gpu_strategy: GpuParallelStrategy::Auto,
            ..Default::default()
        });
        
        b.iter(|| {
            let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
            black_box(result)
        });
    });
    
    group.finish();
}

/// GPUバッチ操作ベンチマーク
/// GPU batch operations benchmarks
fn bench_gpu_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_batch_operations");
    
    let batch_sizes = vec![1, 4, 16, 64];
    let feature_size = 1000;
    
    for batch_size in batch_sizes {
        // GPUバッチ正規化
        group.bench_with_input(
            BenchmarkId::new("gpu_batch_normalize", batch_size),
            &batch_size,
            |b, &batch_size| {
                let tensor = Tensor::<f32>::ones(&[batch_size, feature_size]);
                
                b.iter(|| {
                    let result = tensor.gpu_batch_normalize(1e-5);
                    black_box(result)
                });
            },
        );
        
        // GPUバッチ畳み込み
        group.bench_with_input(
            BenchmarkId::new("gpu_batch_conv2d", batch_size),
            &batch_size,
            |b, &batch_size| {
                let input = Tensor::<f32>::ones(&[batch_size, 3, 32, 32]); // NCHW format
                let kernel = Tensor::<f32>::ones(&[16, 3, 3, 3]); // 16 filters, 3x3
                
                b.iter(|| {
                    let result = input.gpu_batch_conv2d(&kernel, 1, 1);
                    black_box(result)
                });
            },
        );
        
        // GPUバッチアテンション
        group.bench_with_input(
            BenchmarkId::new("gpu_batch_attention", batch_size),
            &batch_size,
            |b, &batch_size| {
                let seq_len = 128;
                let hidden_dim = 512;
                
                let query = Tensor::<f32>::ones(&[batch_size, seq_len, hidden_dim]);
                let key = Tensor::<f32>::ones(&[batch_size, seq_len, hidden_dim]);
                let value = Tensor::<f32>::ones(&[batch_size, seq_len, hidden_dim]);
                
                b.iter(|| {
                    let result = query.gpu_batch_attention(&key, &value);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// デバイス選択最適化ベンチマーク
/// Device selection optimization benchmarks
fn bench_device_selection_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_selection_optimization");
    
    let sizes = vec![100, 1000, 10000, 100000];
    
    for size in sizes {
        // 最適デバイス選択
        group.bench_with_input(
            BenchmarkId::new("optimal_device_selection", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let device = gpu_parallel_utils::select_optimal_device(size);
                    black_box(device)
                });
            },
        );
        
        // GPU効率性評価
        group.bench_with_input(
            BenchmarkId::new("gpu_efficiency_evaluation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let efficiency = gpu_parallel_utils::evaluate_gpu_efficiency(size, 2.0);
                    black_box(efficiency)
                });
            },
        );
        
        // バッチサイズ最適化
        group.bench_with_input(
            BenchmarkId::new("batch_size_optimization", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let batch_size = gpu_parallel_utils::optimize_batch_size(size, DeviceType::Cuda(0));
                    black_box(batch_size)
                });
            },
        );
    }
    
    group.finish();
}

/// GPU統合パフォーマンス比較
/// GPU integration performance comparison
fn bench_gpu_integration_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_integration_comparison");
    
    let size = 10000;
    let tensor1 = Tensor::<f32>::ones(&[size]);
    let tensor2 = Tensor::<f32>::ones(&[size]);
    
    // CPU並列 vs GPU統合比較
    group.bench_function("cpu_parallel_baseline", |b| {
        b.iter(|| {
            let result = tensor1.batch_elementwise_op(&tensor2, |a, b| a + b);
            black_box(result)
        });
    });
    
    group.bench_function("gpu_integrated", |b| {
        b.iter(|| {
            let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b);
            black_box(result)
        });
    });
    
    // 行列乗算比較
    let matrix_size = 100;
    let matrix1 = Tensor::<f32>::ones(&[matrix_size, matrix_size]);
    let matrix2 = Tensor::<f32>::ones(&[matrix_size, matrix_size]);
    
    group.bench_function("cpu_matmul_baseline", |b| {
        b.iter(|| {
            let result = matrix1.batch_matmul(&matrix2);
            black_box(result)
        });
    });
    
    group.bench_function("gpu_matmul_integrated", |b| {
        b.iter(|| {
            let result = matrix1.gpu_matmul(&matrix2);
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    gpu_integration_benches,
    bench_gpu_parallel_operations,
    bench_gpu_cpu_transfer,
    bench_gpu_strategy_comparison,
    bench_gpu_batch_operations,
    bench_device_selection_optimization,
    bench_gpu_integration_comparison
);

criterion_main!(gpu_integration_benches);
