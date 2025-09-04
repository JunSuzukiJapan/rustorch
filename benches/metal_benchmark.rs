//! Metal Performance Benchmark Suite  
//! Metalパフォーマンスベンチマークスイート

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::tensor::Tensor;

/// Print Metal system information
fn print_metal_system_info() {
    println!("🍎 === Metal Performance Benchmark ===");

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use metal::Device;
        if let Some(device) = Device::system_default() {
            println!("✅ Metal device: {}", device.name());
            println!(
                "   Max working set: {} MB",
                device.recommended_max_working_set_size() / (1024 * 1024)
            );
            println!("   Low power: {}", device.is_low_power());
        } else {
            println!("❌ No Metal device available");
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("ℹ️  Metal not available (requires macOS and metal feature)");
    }

    println!("=====================================");
}

/// Metal matrix operations benchmark
fn bench_metal_matrix(c: &mut Criterion) {
    print_metal_system_info();

    let mut group = c.benchmark_group("metal_matrix");

    let size = 128;
    let a = Tensor::<f32>::ones(&[size, size]);
    let b = Tensor::<f32>::ones(&[size, size]);

    // CPU benchmark (baseline)
    group.bench_function("cpu_matmul", |bench| {
        bench.iter(|| {
            let _result = black_box(a.matmul(&b)).unwrap();
        });
    });

    // Metal GPU benchmark (feature-gated)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use metal::Device;
        use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

        if Device::system_default().is_some() {
            group.bench_function("metal_gpu_matmul", |bench| {
                bench.iter(|| {
                    let _result = black_box(a.gpu_matmul(&b)).unwrap();
                });
            });
        }
    }

    group.finish();
}

/// Metal GPU operations benchmark
fn bench_metal_operations(c: &mut Criterion) {
    let group = c.benchmark_group("metal_operations");

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use metal::Device;
        use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

        if let Some(device) = Device::system_default() {
            println!("🚀 Testing Metal GPU operations on {}", device.name());

            // Various matrix sizes for GPU performance testing
            for size in [64, 128, 256, 512].iter() {
                let a = Tensor::<f32>::ones(&[*size, *size]);
                let b = Tensor::<f32>::ones(&[*size, *size]);

                group.bench_function(&format!("metal_matmul_{}x{}", size, size), |bench| {
                    bench.iter(|| {
                        let _result = black_box(a.gpu_matmul(&b)).unwrap();
                    });
                });
            }

            // Batch operations
            let tensors = vec![
                Tensor::<f32>::ones(&[64, 64]),
                Tensor::<f32>::ones(&[64, 64]),
                Tensor::<f32>::ones(&[64, 64]),
            ];

            group.bench_function("metal_batch_matmul", |bench| {
                bench.iter(|| {
                    for tensor in &tensors {
                        let _result = black_box(tensor.gpu_batch_matmul(&tensors[0])).unwrap();
                    }
                });
            });
        } else {
            println!("⚠️  No Metal device found - skipping GPU benchmarks");
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("ℹ️  Metal not available - compile with '--features metal' on macOS");
    }

    group.finish();
}

criterion_group!(
    name = metal_benches;
    config = Criterion::default();
    targets = bench_metal_matrix, bench_metal_operations
);
criterion_main!(metal_benches);
