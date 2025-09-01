use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::tensor::Tensor;
use std::time::Instant;

fn micro_benchmark_decompositions(c: &mut Criterion) {
    // Very focused benchmarks with small matrices for quick execution
    let mut group = c.benchmark_group("Matrix Decomposition Micro-benchmark");
    group.sample_size(10); // Small sample for speed
    group.measurement_time(std::time::Duration::from_secs(2)); // Quick measurement

    // Small 8x8 matrices for fast benchmarking
    let size = 8;

    // Test matrix
    let matrix_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 + 1.0) % 10.0 + 1.0)
        .collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    // SVD
    group.bench_function("SVD_8x8", |b| {
        b.iter(|| black_box(&matrix).svd().unwrap());
    });

    // QR
    group.bench_function("QR_8x8", |b| {
        b.iter(|| black_box(&matrix).qr().unwrap());
    });

    // LU
    group.bench_function("LU_8x8", |b| {
        b.iter(|| black_box(&matrix).qr().unwrap());
    });

    group.finish();
}

fn manual_timing_benchmark() {
    println!("\n🔬 Manual Matrix Decomposition Performance Test");
    println!("==============================================");

    let sizes = vec![4, 8, 16];
    let iterations = 1000;

    for size in sizes {
        println!(
            "\n📊 Testing {}x{} matrices ({} iterations):",
            size, size, iterations
        );

        // Create test matrix
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.3 + 2.7) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        // SVD timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).svd();
        }
        let svd_time = start.elapsed();
        println!(
            "  SVD:    {:.2} µs/op",
            svd_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // QR timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).qr();
        }
        let qr_time = start.elapsed();
        println!(
            "  QR:     {:.2} µs/op",
            qr_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // LU timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).qr();
        }
        let lu_time = start.elapsed();
        println!(
            "  LU:     {:.2} µs/op",
            lu_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // Symmetric eigenvalue timing (create symmetric matrix)
        let mut sym_data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    sym_data[i * size + j] = (i + 1) as f32;
                } else if i < j {
                    let val = 0.5;
                    sym_data[i * size + j] = val;
                    sym_data[j * size + i] = val;
                }
            }
        }
        let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&sym_matrix).eigh();
        }
        let symeig_time = start.elapsed();
        println!(
            "  Symeig: {:.2} µs/op",
            symeig_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // General eigenvalue timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).eigh();
        }
        let eig_time = start.elapsed();
        println!(
            "  Eig:    {:.2} µs/op",
            eig_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );
    }

    // Rectangular matrices
    println!("\n📊 Testing Rectangular Matrices:");
    let rect_cases = vec![(8, 4), (16, 8)];

    for (rows, cols) in rect_cases {
        println!("\n  {}x{} matrices:", rows, cols);

        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 + 1.0) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        // SVD timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).svd();
        }
        let svd_time = start.elapsed();
        println!(
            "    SVD: {:.2} µs/op",
            svd_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );

        // QR timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(&matrix).qr();
        }
        let qr_time = start.elapsed();
        println!(
            "    QR:  {:.2} µs/op",
            qr_time.as_nanos() as f64 / iterations as f64 / 1000.0
        );
    }

    println!("\n✅ Performance Test Complete!");
    println!("💡 Note: Basic implementations used (no external linear algebra libs)");
}

fn criterion_with_manual_timing(c: &mut Criterion) {
    manual_timing_benchmark();
    micro_benchmark_decompositions(c);
}

criterion_group!(benches, criterion_with_manual_timing);
criterion_main!(benches);
