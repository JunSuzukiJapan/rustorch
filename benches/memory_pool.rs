use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::memory::get_f32_pool;
use rustorch::tensor::Tensor;

fn benchmark_tensor_creation_with_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    let sizes = vec![
        (vec![10, 10], "10x10"),
        (vec![100, 100], "100x100"),
        (vec![1000, 1000], "1000x1000"),
        (vec![50, 50, 10], "50x50x10"),
    ];

    for (shape, name) in sizes {
        // Benchmark with memory pool
        group.bench_with_input(BenchmarkId::new("with_pool", name), &shape, |b, shape| {
            b.iter(|| {
                let _tensor: Tensor<f32> = Tensor::zeros(shape);
            })
        });

        // Benchmark without memory pool (direct ndarray allocation)
        group.bench_with_input(
            BenchmarkId::new("without_pool", name),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let _array = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(shape));
                })
            },
        );
    }

    group.finish();
}

fn benchmark_memory_pool_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_reuse");

    // Test memory pool efficiency with repeated allocations
    group.bench_function("repeated_allocations", |b| {
        b.iter(|| {
            let mut tensors = Vec::new();

            // Allocate multiple tensors
            for _ in 0..100 {
                tensors.push(Tensor::<f32>::zeros(&[10, 10]));
            }

            // Drop all tensors (they should return to pool)
            drop(tensors);

            // Allocate again (should reuse from pool)
            for _ in 0..100 {
                let _tensor = Tensor::<f32>::zeros(&[10, 10]);
            }
        })
    });

    group.finish();
}

fn benchmark_pool_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_statistics");

    group.bench_function("get_pool_stats", |b| {
        // Pre-populate pool
        let pool = get_f32_pool();
        {
            let mut pool_guard = pool.lock().unwrap();
            for _ in 0..50 {
                let array = pool_guard.allocate(&[10, 10]);
                pool_guard.deallocate(array);
            }
        }

        b.iter(|| {
            let pool_guard = pool.lock().unwrap();
            let _stats = pool_guard.stats();
        })
    });

    group.finish();
}

fn benchmark_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("different_tensor_sizes");

    let sizes = vec![
        (vec![8], "1D_small"),
        (vec![64], "1D_medium"),
        (vec![512], "1D_large"),
        (vec![8, 8], "2D_small"),
        (vec![64, 64], "2D_medium"),
        (vec![256, 256], "2D_large"),
        (vec![4, 4, 4], "3D_small"),
        (vec![16, 16, 16], "3D_medium"),
        (vec![32, 32, 32], "3D_large"),
    ];

    for (shape, name) in sizes {
        group.bench_with_input(
            BenchmarkId::new("pooled_allocation", name),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let _tensor: Tensor<f32> = Tensor::zeros(shape);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory usage patterns
    group.bench_function("allocation_deallocation_cycle", |b| {
        b.iter(|| {
            // Simulate training loop memory pattern
            let mut batch_tensors = Vec::new();

            // Forward pass - allocate intermediate tensors
            for _i in 0..10 {
                batch_tensors.push(Tensor::<f32>::zeros(&[32, 128]));
                batch_tensors.push(Tensor::<f32>::zeros(&[32, 64]));
                batch_tensors.push(Tensor::<f32>::zeros(&[32, 32]));
            }

            // Backward pass - tensors go out of scope
            drop(batch_tensors);

            // Next iteration - should reuse memory
            let mut next_batch = Vec::new();
            for _i in 0..10 {
                next_batch.push(Tensor::<f32>::zeros(&[32, 128]));
                next_batch.push(Tensor::<f32>::zeros(&[32, 64]));
                next_batch.push(Tensor::<f32>::zeros(&[32, 32]));
            }
        })
    });

    group.finish();
}

criterion_group!(
    memory_benches,
    benchmark_tensor_creation_with_pool,
    benchmark_memory_pool_reuse,
    benchmark_pool_statistics,
    benchmark_different_sizes,
    benchmark_memory_efficiency
);
criterion_main!(memory_benches);
