// Backend performance comparison benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch_cli::backend::{cpu::CpuBackend, Backend};

fn benchmark_cpu_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_backend");

    group.bench_function("create_backend", |b| {
        b.iter(|| {
            let backend = CpuBackend::new();
            black_box(backend);
        });
    });

    group.bench_function("check_availability", |b| {
        let backend = CpuBackend::new();
        b.iter(|| {
            let available = backend.is_available();
            black_box(available);
        });
    });

    group.bench_function("create_zeros_2x2", |b| {
        let backend = CpuBackend::new();
        b.iter(|| {
            let tensor = backend.zeros(&[2, 2]).unwrap();
            black_box(tensor);
        });
    });

    group.bench_function("create_zeros_100x100", |b| {
        let backend = CpuBackend::new();
        b.iter(|| {
            let tensor = backend.zeros(&[100, 100]).unwrap();
            black_box(tensor);
        });
    });

    group.finish();
}

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    let backend = CpuBackend::new();

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &s| {
            b.iter(|| {
                let tensor = backend.zeros(&[s, s]).unwrap();
                black_box(tensor);
            });
        });

        group.bench_with_input(BenchmarkId::new("from_vec", size), size, |b, &s| {
            let data = vec![1.0; s * s];
            b.iter(|| {
                let tensor = backend.from_vec(data.clone(), &[s, s]).unwrap();
                black_box(tensor);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_cpu_backend, benchmark_tensor_creation);
criterion_main!(benches);
