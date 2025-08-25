//! Metal Performance Shaders demonstration for AMD Radeon Pro Vega 56
//! AMD Radeon Pro Vega 56向けMetal Performance Shadersデモンストレーション

use rustorch::gpu::metal_kernels::{MetalKernelExecutor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RusTorch Metal Performance Demo");
    println!("AMD Radeon Pro Vega 56 最適化デモ");
    println!("====================================");
    
    #[cfg(feature = "metal")]
    {
        // Initialize Metal executor
        let executor = MetalKernelExecutor::new()?;
        println!("✓ Metal executor initialized");

        // Test element-wise addition
        test_elementwise_addition(&executor)?;
        
        // Test matrix multiplication performance
        test_matrix_multiplication(&executor)?;
        
        // Test tiled matrix multiplication for large matrices
        test_large_matrix_multiplication(&executor)?;
        
        // Test reduction operations
        test_reduction_operations(&executor)?;
        
        println!("\n🎯 All Metal Performance tests completed successfully!");
    }
    
    #[cfg(not(feature = "metal"))]
    {
        println!("❌ Metal feature not enabled");
        println!("Run with: cargo run --example metal_performance_demo --features metal");
    }
    
    Ok(())
}

#[cfg(feature = "metal")]
fn test_elementwise_addition(executor: &MetalKernelExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Element-wise Addition");
    println!("---------------------------------");
    
    let size = 1_000_000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut c = vec![0.0f32; size];
    
    let start = Instant::now();
    executor.elementwise_add_f32(&a, &b, &mut c)?;
    let duration = start.elapsed();
    
    // Verify results
    let expected: Vec<f32> = (0..size).map(|i| i as f32 + (i * 2) as f32).collect();
    let errors: Vec<_> = c.iter()
        .zip(expected.iter())
        .enumerate()
        .filter(|(_, (&actual, &expected))| (actual - expected).abs() > 1e-6)
        .collect();
    
    if errors.is_empty() {
        println!("✓ Element-wise addition: {:.3}ms ({:.2} GB/s)", 
                 duration.as_secs_f64() * 1000.0,
                 (size as f64 * 3.0 * 4.0) / (duration.as_secs_f64() * 1e9)); // 3 arrays * 4 bytes/f32
    } else {
        println!("❌ Verification failed: {} errors", errors.len());
    }
    
    Ok(())
}

#[cfg(feature = "metal")]
fn test_matrix_multiplication(executor: &MetalKernelExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Standard Matrix Multiplication");
    println!("----------------------------------------");
    
    let sizes = [64, 128, 256];
    
    for &size in &sizes {
        let m = size;
        let n = size;
        let k = size;
        
        // Create test matrices
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) / (m*k) as f32).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i * 2) as f32) / (k*n) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        
        let start = Instant::now();
        executor.matmul_f32(&a, &b, &mut c, m, n, k)?;
        let duration = start.elapsed();
        
        // Calculate GFLOPS (2 * M * N * K operations)
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        
        println!("✓ Matrix {}x{}x{}: {:.3}ms ({:.2} GFLOPS)", 
                 m, n, k, duration.as_secs_f64() * 1000.0, gflops);
    }
    
    Ok(())
}

#[cfg(feature = "metal")]
fn test_large_matrix_multiplication(executor: &MetalKernelExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Tiled Matrix Multiplication (Large)");
    println!("----------------------------------------------");
    
    let sizes = [512, 1024];
    
    for &size in &sizes {
        let m = size;
        let n = size;
        let k = size;
        
        // Create test matrices
        let a: Vec<f32> = (0..m*k).map(|i| ((i % 1000) as f32) / 1000.0).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (((i * 3) % 1000) as f32) / 1000.0).collect();
        let mut c = vec![0.0f32; m * n];
        
        let start = Instant::now();
        executor.tiled_matmul_f32(&a, &b, &mut c, m, n, k)?;
        let duration = start.elapsed();
        
        // Calculate GFLOPS and memory bandwidth
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0; // bytes
        let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);
        
        println!("✓ Tiled Matrix {}x{}x{}: {:.3}ms ({:.2} GFLOPS, {:.1} GB/s)", 
                 m, n, k, duration.as_secs_f64() * 1000.0, gflops, bandwidth_gb_s);
    }
    
    Ok(())
}

#[cfg(feature = "metal")]
fn test_reduction_operations(executor: &MetalKernelExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Reduction Operations");
    println!("-------------------------------");
    
    let sizes = [10_000, 100_000, 1_000_000];
    
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let expected_sum: f32 = data.iter().sum();
        
        let start = Instant::now();
        let result = executor.reduce_sum_f32(&data)?;
        let duration = start.elapsed();
        
        let error = (result - expected_sum).abs();
        let relative_error = error / expected_sum;
        
        if relative_error < 1e-4 {
            println!("✓ Reduction sum ({}): {:.3}ms (result: {:.1}, expected: {:.1})", 
                     size, duration.as_secs_f64() * 1000.0, result, expected_sum);
        } else {
            println!("❌ Reduction sum ({}): Large error {:.6}", size, relative_error);
        }
    }
    
    Ok(())
}

#[cfg(feature = "metal")]
fn benchmark_vega56_performance(executor: &MetalKernelExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 AMD Radeon Pro Vega 56 Performance Benchmark");
    println!("===============================================");
    
    // Theoretical peak performance for Vega 56:
    // - 10.5 TFLOPS FP32
    // - 410 GB/s memory bandwidth
    // - 3584 stream processors
    
    let large_size = 2048;
    let m = large_size;
    let n = large_size;
    let k = large_size;
    
    println!("Matrix size: {}x{}x{}", m, n, k);
    println!("Memory required: {:.1} MB", 
             (m * k + k * n + m * n) as f64 * 4.0 / 1_000_000.0);
    
    // Create large test matrices
    let a: Vec<f32> = (0..m*k).map(|i| (i % 1000) as f32 / 1000.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i * 3) % 1000) as f32 / 1000.0).collect();
    let mut c = vec![0.0f32; m * n];
    
    // Warm-up
    executor.tiled_matmul_f32(&a, &b, &mut c, m, n, k)?;
    
    // Benchmark multiple runs
    let num_runs = 5;
    let mut durations = Vec::new();
    
    for run in 1..=num_runs {
        let start = Instant::now();
        executor.tiled_matmul_f32(&a, &b, &mut c, m, n, k)?;
        let duration = start.elapsed();
        durations.push(duration);
        
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        let efficiency = (gflops / 10500.0) * 100.0; // Vega 56 peak: 10.5 TFLOPS
        
        println!("Run {}: {:.3}ms ({:.1} GFLOPS, {:.1}% efficiency)", 
                 run, duration.as_secs_f64() * 1000.0, gflops, efficiency);
    }
    
    // Calculate statistics
    let avg_duration: f64 = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / num_runs as f64;
    let avg_gflops = (2.0 * (m * n * k) as f64) / (avg_duration * 1e9);
    let avg_efficiency = (avg_gflops / 10500.0) * 100.0;
    
    println!("\n📊 Average Performance:");
    println!("Duration: {:.3}ms", avg_duration * 1000.0);
    println!("GFLOPS: {:.1}", avg_gflops);
    println!("Efficiency: {:.1}% of theoretical peak", avg_efficiency);
    
    Ok(())
}

