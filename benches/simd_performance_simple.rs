//! Simplified SIMD Performance Benchmarks
//! 
//! This is a minimal version to test SIMD operations without complex dependencies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::tensor::Tensor;

/// Benchmark basic SIMD vs scalar element-wise operations
fn bench_simd_elementwise_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_elementwise_simple");
    
    let size = 1024;
    let tensor1 = Tensor::<f32>::ones(&[size]);
    let tensor2 = Tensor::<f32>::ones(&[size]);
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Scalar addition
    group.bench_with_input(
        BenchmarkId::new("scalar_add", size),
        &(size, &tensor1, &tensor2),
        |b, (_, t1, t2)| {
            b.iter(|| {
                let result = black_box(&*t1 + &*t2);
                black_box(result)
            });
        },
    );
    
    group.finish();
}

criterion_group!(simd_benches_simple, bench_simd_elementwise_simple);
criterion_main!(simd_benches_simple);
