//! Simple GPU Performance Benchmark
//! GPU性能の簡単なベンチマーク

use criterion::{black_box, criterion_group, Criterion};
use rustorch::tensor::Tensor;
use rustorch::gpu::DeviceManager;

// Simple matrix operations benchmark
fn bench_simple_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_gpu_operations");
    
    // Test matrix sizes
    let size = 256;
    let a = Tensor::<f32>::ones(&[size, size]);
    let b = Tensor::<f32>::ones(&[size, size]);
    
    // CPU matrix multiplication
    group.bench_function("cpu_matmul", |bench| {
        bench.iter(|| {
            let _result = black_box(a.matmul(&b)).unwrap();
        });
    });
    
    // GPU matrix multiplication
    group.bench_function("gpu_matmul", |bench| {
        bench.iter(|| {
            use rustorch::gpu::matrix_ops::GpuLinearAlgebra;
            let _result = black_box(a.gpu_matmul(&b)).unwrap();
        });
    });
    
    group.finish();
}

// Simple reduction operations benchmark
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_reductions");
    
    let tensor = Tensor::<f32>::ones(&[1000, 1000]);
    
    // CPU sum
    group.bench_function("cpu_sum", |bench| {
        bench.iter(|| {
            let _result = black_box(tensor.sum());
        });
    });
    
    // GPU sum
    group.bench_function("gpu_sum", |bench| {
        bench.iter(|| {
            let _result = black_box(tensor.gpu_sum(None)).unwrap();
        });
    });
    
    // CPU mean
    group.bench_function("cpu_mean", |bench| {
        bench.iter(|| {
            let _result = black_box(tensor.mean());
        });
    });
    
    // GPU mean
    group.bench_function("gpu_mean", |bench| {
        bench.iter(|| {
            let _result = black_box(tensor.gpu_mean(None)).unwrap();
        });
    });
    
    group.finish();
}

// Print system information
fn print_system_info() {
    println!("=== GPU Performance Benchmark ===");
    
    let manager = DeviceManager::new();
    let devices = manager.available_devices();
    
    println!("Available devices: {:?}", devices);
    println!("CUDA available: {}", DeviceManager::is_cuda_available());
    println!("Metal available: {}", DeviceManager::is_metal_available());
    
    #[cfg(target_os = "macos")]
    println!("Platform: macOS");
    
    #[cfg(target_os = "linux")]
    println!("Platform: Linux");
    
    #[cfg(target_os = "windows")]
    println!("Platform: Windows");
    
    println!("===================================\n");
}

criterion_group!(
    name = simple_benchmarks;
    config = Criterion::default();
    targets = bench_simple_operations, bench_reductions
);

// Custom main to print system info before benchmarks
fn main() {
    print_system_info();
    simple_benchmarks();
}