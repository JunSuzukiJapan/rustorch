//! Optimized operations benchmarks
//! 最適化された演算のベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::prelude::*;

fn bench_optimized_matmul(c: &mut Criterion) {
    let a = Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
    let b = Tensor::from_vec((0..10000).map(|i| (i + 1) as f32).collect(), vec![100, 100]);

    c.bench_function("optimized_matmul_100x100", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b)))
    });

    let large_a = Tensor::from_vec((0..250000).map(|i| i as f32).collect(), vec![500, 500]);
    let large_b = Tensor::from_vec(
        (0..250000).map(|i| (i + 1) as f32).collect(),
        vec![500, 500],
    );

    c.bench_function("optimized_matmul_500x500", |bencher| {
        bencher.iter(|| black_box(large_a.matmul(&large_b)))
    });

    let xl_a = Tensor::from_vec((0..1_000_000).map(|i| i as f32).collect(), vec![1000, 1000]);
    let xl_b = Tensor::from_vec(
        (0..1_000_000).map(|i| (i + 1) as f32).collect(),
        vec![1000, 1000],
    );

    c.bench_function("optimized_matmul_1000x1000", |bencher| {
        bencher.iter(|| black_box(xl_a.matmul(&xl_b)))
    });
}

fn bench_inplace_operations(c: &mut Criterion) {
    c.bench_function("inplace_add_vs_regular", |bencher| {
        bencher.iter(|| {
            let mut a = Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
            let b = Tensor::from_vec((0..10000).map(|i| (i + 1) as f32).collect(), vec![100, 100]);

            let _result = &a + &b;
            black_box(a)
        })
    });

    c.bench_function("regular_add_for_comparison", |bencher| {
        bencher.iter(|| {
            let a = Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
            let b = Tensor::from_vec((0..10000).map(|i| (i + 1) as f32).collect(), vec![100, 100]);

            black_box(&a + &b)
        })
    });

    c.bench_function("inplace_mul_scalar", |bencher| {
        bencher.iter(|| {
            let mut tensor =
                Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
            let _result = tensor.mul_scalar(2.0);
            black_box(tensor)
        })
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    c.bench_function("batch_matmul_32x784x128", |bencher| {
        bencher.iter(|| {
            let batch_input =
                Tensor::from_vec((0..25088).map(|i| i as f32 * 0.01).collect(), vec![32, 784]);
            let weight = Tensor::from_vec(
                (0..100352).map(|i| i as f32 * 0.01).collect(),
                vec![784, 128],
            );

            black_box(batch_input.matmul(&weight))
        })
    });

    c.bench_function("large_batch_matmul_128x512x256", |bencher| {
        bencher.iter(|| {
            let batch_input = Tensor::from_vec(
                (0..65536).map(|i| i as f32 * 0.001).collect(),
                vec![128, 512],
            );
            let weight = Tensor::from_vec(
                (0..131072).map(|i| i as f32 * 0.001).collect(),
                vec![512, 256],
            );

            black_box(batch_input.matmul(&weight))
        })
    });
}

fn bench_memory_efficiency(c: &mut Criterion) {
    c.bench_function("memory_efficient_chain", |bencher| {
        bencher.iter(|| {
            let mut result = Tensor::from_vec(
                (0..10000).map(|i| i as f32 * 0.01).collect(),
                vec![100, 100],
            );
            let operand = Tensor::from_vec(
                (0..10000).map(|i| (i + 1) as f32 * 0.01).collect(),
                vec![100, 100],
            );

            // Chain of in-place operations
            for _ in 0..5 {
                let _temp = &result + &operand;
                let _result = result.mul_scalar(0.9);
            }

            black_box(result)
        })
    });
}

criterion_group!(
    optimized_benches,
    bench_optimized_matmul,
    bench_inplace_operations,
    bench_batch_operations,
    bench_memory_efficiency
);
criterion_main!(optimized_benches);
