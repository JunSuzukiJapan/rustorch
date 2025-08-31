use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::tensor::Tensor;

fn bench_matrix_decompositions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Decomposition Performance");
    group.sample_size(20); // Reduce sample size for faster benchmarks
    group.measurement_time(std::time::Duration::from_secs(5));

    // Test with 16x16 matrices for reasonable benchmark time
    let size = 16;

    // Create test matrix for all decompositions
    let matrix_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 * 1.4 + 2.8) % 8.0 + 1.0)
        .collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    // SVD benchmark
    group.bench_function("SVD_16x16", |b| {
        b.iter(|| {
            let result = black_box(&matrix).svd();
            black_box(result)
        });
    });

    // QR benchmark
    group.bench_function("QR_16x16", |b| {
        b.iter(|| {
            let result = black_box(&matrix).qr();
            black_box(result)
        });
    });

    // LU benchmark
    group.bench_function("LU_16x16", |b| {
        b.iter(|| {
            let result = black_box(&matrix).qr();
            black_box(result)
        });
    });

    // Create symmetric matrix for eigenvalue decomposition
    let mut sym_data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if i == j {
                sym_data[i * size + j] = (i + 1) as f32 * 2.0;
            } else if i < j {
                let val = ((i + j + 1) as f32 * 0.5) % 3.0;
                sym_data[i * size + j] = val;
                sym_data[j * size + i] = val; // Make symmetric
            }
        }
    }
    let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);

    // Symmetric eigenvalue benchmark
    group.bench_function("Symeig_16x16", |b| {
        b.iter(|| {
            let result = black_box(&sym_matrix).eigh();
            black_box(result)
        });
    });

    // General eigenvalue benchmark
    group.bench_function("Eig_16x16", |b| {
        b.iter(|| {
            let result = black_box(&matrix).eigh();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_matrix_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVD Scale Performance");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    for size in &[4, 8, 16, 32] {
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.7 + 3.14) % 10.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![*size, *size]);

        group.bench_function(&format!("SVD_{}x{}", size, size), |b| {
            b.iter(|| {
                let result = black_box(&matrix).svd(); // SVD decomposition
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_rectangular_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rectangular Matrix Performance");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    let test_cases = vec![(8, 4, "8x4"), (16, 8, "16x8"), (32, 16, "32x16")];

    for (rows, cols, label) in test_cases {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 2.1 + 1.41) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        group.bench_function(&format!("SVD_{}", label), |b| {
            b.iter(|| {
                let result = black_box(&matrix).svd();
                black_box(result)
            });
        });

        group.bench_function(&format!("QR_{}", label), |b| {
            b.iter(|| {
                let result = black_box(&matrix).qr();
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_decompositions,
    bench_matrix_sizes,
    bench_rectangular_matrices
);
criterion_main!(benches);
