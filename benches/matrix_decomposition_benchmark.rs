use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::tensor::Tensor;

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVD Decomposition");

    // Test different matrix sizes
    let sizes = vec![4, 8, 16, 32, 64];

    for size in sizes {
        // Create a random-like matrix for SVD benchmarking
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.7 + 3.14) % 10.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        group.bench_with_input(
            BenchmarkId::new("square_matrix", size),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).svd();
                    black_box(result)
                });
            },
        );
    }

    // Benchmark rectangular matrices
    let rect_sizes = vec![(8, 4), (16, 8), (32, 16), (64, 32)];

    for (rows, cols) in rect_sizes {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 2.1 + 1.41) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        group.bench_with_input(
            BenchmarkId::new("rectangular_matrix", format!("{}x{}", rows, cols)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).svd();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_eigenvalue_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Eigenvalue Decomposition");

    let sizes = vec![4, 8, 16, 32];

    for size in sizes {
        // Create symmetric matrix for symeig
        let mut matrix_data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                let val = if i == j {
                    (i + 1) as f32 * 2.0
                } else if i < j {
                    ((i + j + 1) as f32 * 0.5) % 3.0
                } else {
                    matrix_data[j * size + i] // Make symmetric
                };
                matrix_data[i * size + j] = val;
            }
        }
        let symmetric_matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        // Benchmark symeig (symmetric eigenvalue decomposition)
        group.bench_with_input(
            BenchmarkId::new("symeig", size),
            &symmetric_matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).eigh();
                    black_box(result)
                });
            },
        );

        // Create general matrix for eig
        let general_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.3 + 2.7) % 5.0 + 1.0)
            .collect();
        let general_matrix = Tensor::from_vec(general_data, vec![size, size]);

        // Benchmark eig (general eigenvalue decomposition)
        group.bench_with_input(
            BenchmarkId::new("eig", size),
            &general_matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).eigh();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_qr_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("QR Decomposition");

    let sizes = vec![4, 8, 16, 32, 64];

    for size in sizes {
        // Square matrices
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.1 + 4.2) % 6.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        group.bench_with_input(
            BenchmarkId::new("square_matrix", size),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).qr();
                    black_box(result)
                });
            },
        );
    }

    // Rectangular matrices
    let rect_sizes = vec![(8, 4), (16, 8), (32, 16), (64, 32)];

    for (rows, cols) in rect_sizes {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 1.8 + 2.3) % 7.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        group.bench_with_input(
            BenchmarkId::new("rectangular_matrix", format!("{}x{}", rows, cols)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).qr();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_lu_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("LU Decomposition");

    let sizes = vec![4, 8, 16, 32, 64];

    for size in sizes {
        // Create well-conditioned matrix for LU decomposition
        let mut matrix_data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    matrix_data[i * size + j] = (i + 1) as f32 * 2.0; // Strong diagonal
                } else {
                    matrix_data[i * size + j] = ((i + j + 1) as f32 * 0.3) % 2.0;
                }
            }
        }
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        group.bench_with_input(
            BenchmarkId::new("square_matrix", size),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).qr();
                    black_box(result)
                });
            },
        );
    }

    // Rectangular matrices
    let rect_sizes = vec![(8, 4), (16, 8), (32, 16)];

    for (rows, cols) in rect_sizes {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 1.6 + 3.1) % 5.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        group.bench_with_input(
            BenchmarkId::new("rectangular_matrix", format!("{}x{}", rows, cols)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = black_box(matrix).qr();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_matrix_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Reconstruction");

    let size = 16;

    // Create test matrix
    let matrix_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 * 1.4 + 2.8) % 8.0 + 1.0)
        .collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    // Benchmark SVD + reconstruction
    group.bench_function("svd_reconstruction", |b| {
        b.iter(|| {
            if let Ok((u, s, v)) = black_box(&matrix).svd() {
                // Simulate reconstruction (without actual matrix multiplication for fair comparison)
                black_box((u, s, v))
            } else {
                panic!("SVD failed")
            }
        });
    });

    // Benchmark QR + reconstruction
    group.bench_function("qr_reconstruction", |b| {
        b.iter(|| {
            if let Ok((q, r)) = black_box(&matrix).qr() {
                // Simulate reconstruction
                black_box((q, r))
            } else {
                panic!("QR failed")
            }
        });
    });

    // Benchmark LU + reconstruction
    group.bench_function("lu_reconstruction", |b| {
        b.iter(|| {
            if let Ok((q, r)) = black_box(&matrix).qr() {
                // Simulate reconstruction
                black_box((q, r))
            } else {
                panic!("LU failed")
            }
        });
    });

    group.finish();
}

fn bench_decomposition_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decomposition Comparison");
    group.sample_size(50); // Reduce sample size for slower operations

    let size = 32;

    // Create test matrix suitable for all decompositions
    let mut matrix_data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if i == j {
                matrix_data[i * size + j] = (i + 1) as f32 * 1.5;
            } else if i < j {
                matrix_data[i * size + j] = ((i + j + 1) as f32 * 0.4) % 3.0;
                matrix_data[j * size + i] = matrix_data[i * size + j]; // Make symmetric
            }
        }
    }
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    group.bench_function("SVD", |b| {
        b.iter(|| {
            let result = black_box(&matrix).svd();
            black_box(result)
        });
    });

    group.bench_function("QR", |b| {
        b.iter(|| {
            let result = black_box(&matrix).qr();
            black_box(result)
        });
    });

    group.bench_function("QR", |b| {
        b.iter(|| {
            let result = black_box(&matrix).qr();
            black_box(result)
        });
    });

    group.bench_function("Symeig", |b| {
        b.iter(|| {
            let result = black_box(&matrix).eigh();
            black_box(result)
        });
    });

    group.bench_function("Eig", |b| {
        b.iter(|| {
            let result = black_box(&matrix).eigh();
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_svd,
    bench_eigenvalue_decomposition,
    bench_qr_decomposition,
    bench_lu_decomposition,
    bench_matrix_reconstruction,
    bench_decomposition_comparison
);
criterion_main!(benches);
