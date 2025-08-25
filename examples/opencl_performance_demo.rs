//! OpenCL Cross-platform Performance demonstration
//! クロスプラットフォームOpenCLパフォーマンスデモンストレーション

use rustorch::gpu::opencl_optimized::{OpenClMatrixExecutor, opencl_matmul_f32};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RusTorch OpenCL Cross-platform Performance Demo");
    println!("RusTorchクロスプラットフォームOpenCLパフォーマンスデモ");
    println!("===============================================");
    
    #[cfg(feature = "opencl")]
    {
        // Initialize OpenCL executor with automatic device selection
        match OpenClMatrixExecutor::new() {
            Ok(mut executor) => {
                println!("✓ OpenCL executor initialized");
                
                // Display device information
                display_device_info(&executor)?;
                
                // Test matrix multiplication across different sizes
                test_matrix_multiplication_performance(&mut executor)?;
                
                // Test vendor-specific optimizations
                test_vendor_optimizations(&mut executor)?;
                
                // Cross-platform compatibility test
                test_cross_platform_compatibility(&mut executor)?;
                
                // Memory and compute intensive benchmark
                benchmark_opencl_performance(&mut executor)?;
                
                println!("\n🎯 All OpenCL cross-platform tests completed successfully!");
            }
            Err(e) => {
                println!("❌ OpenCL initialization failed: {}", e);
                println!("This is expected if OpenCL is not available on this system.");
                
                // Test fallback interface
                test_fallback_interface()?;
            }
        }
    }
    
    #[cfg(not(feature = "opencl"))]
    {
        println!("❌ OpenCL feature not enabled");
        println!("Run with: cargo run --example opencl_performance_demo --features opencl");
        
        test_fallback_interface()?;
    }
    
    Ok(())
}

#[cfg(feature = "opencl")]
fn display_device_info(executor: &OpenClMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 OpenCL Device Information");
    println!("-----------------------------");
    
    let info = executor.get_device_info();
    
    println!("Device Name: {}", info.name);
    println!("Vendor: {}", info.vendor);
    println!("Compute Units: {}", info.compute_units);
    println!("Max Work Group Size: {}", info.max_work_group_size);
    println!("Global Memory: {:.1} GB", info.global_mem_size as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Local Memory: {:.1} KB", info.local_mem_size as f64 / 1024.0);
    println!("Max Clock Frequency: {} MHz", info.max_clock_frequency);
    println!("Device Type: {}", info.device_type);
    
    // Provide vendor-specific insights
    match info.vendor.as_str() {
        v if v.contains("AMD") => {
            println!("🔧 AMD GPU detected - Using GCN-optimized kernels with wavefront scheduling");
        }
        v if v.contains("NVIDIA") => {
            println!("🔧 NVIDIA GPU detected - Using CUDA core optimizations and warp execution");  
        }
        v if v.contains("Intel") => {
            println!("🔧 Intel GPU detected - Using conservative optimizations for integrated graphics");
        }
        _ => {
            println!("🔧 Generic GPU detected - Using standard OpenCL optimizations");
        }
    }
    
    Ok(())
}

#[cfg(feature = "opencl")]
fn test_matrix_multiplication_performance(executor: &mut OpenClMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Matrix Multiplication Performance");
    println!("-------------------------------------------");
    
    let sizes = [64, 128, 256, 512, 1024];
    
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
        
        // Calculate memory bandwidth utilization
        let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0; // bytes
        let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);
        
        println!("✓ Matrix {}x{}x{}: {:.3}ms ({:.1} GFLOPS, {:.1} GB/s)", 
                 m, n, k, duration.as_secs_f64() * 1000.0, gflops, bandwidth_gb_s);
        
        // Verify correctness (simple check)
        let non_zero_count = c.iter().filter(|&&x| x.abs() > 1e-6).count();
        let expected_non_zero = (m * n * 3) / 4; // Rough estimate
        
        if non_zero_count > expected_non_zero / 2 {
            println!("  ✓ Correctness check passed ({} non-zero elements)", non_zero_count);
        } else {
            println!("  ⚠️ Possible correctness issue ({} non-zero elements)", non_zero_count);
        }
    }
    
    Ok(())
}

#[cfg(feature = "opencl")]
fn test_vendor_optimizations(executor: &mut OpenClMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Vendor-Specific Optimizations");
    println!("----------------------------------------");
    
    let info = executor.get_device_info();
    
    println!("Testing optimizations for: {}", info.vendor);
    
    // Test different matrix sizes to verify optimization effectiveness
    let test_cases = [
        (128, 128, 128, "Small matrices"),
        (512, 512, 512, "Medium matrices"), 
        (1024, 1024, 1024, "Large matrices"),
        (2048, 1024, 512, "Non-square matrices"),
        (256, 1024, 256, "Tall matrices"),
        (1024, 256, 512, "Wide matrices"),
    ];
    
    for &(m, n, k, description) in &test_cases {
        println!("\nTesting {}: {}x{}x{}", description, m, n, k);
        
        // Create test matrices
        let a: Vec<f32> = (0..m*k).map(|i| ((i % 1000) as f32) / 1000.0).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (((i * 3) % 1000) as f32) / 1000.0).collect();
        let mut c = vec![0.0f32; m * n];
        
        let start = Instant::now();
        executor.matmul_f32(&a, &b, &mut c, m, n, k)?;
        let duration = start.elapsed();
        
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        
        // Calculate efficiency based on compute units
        let theoretical_gflops = info.compute_units as f64 * info.max_clock_frequency as f64 * 2.0 / 1000.0;
        let efficiency = (gflops / theoretical_gflops) * 100.0;
        
        println!("  ✓ {:.3}ms ({:.1} GFLOPS, {:.1}% efficiency)", 
                 duration.as_secs_f64() * 1000.0, gflops, efficiency);
    }
    
    Ok(())
}

#[cfg(feature = "opencl")]
fn test_cross_platform_compatibility(executor: &mut OpenClMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Cross-Platform Compatibility");
    println!("--------------------------------------");
    
    let info = executor.get_device_info();
    
    // Test various scenarios that might behave differently across platforms
    let test_scenarios = [
        (1, 1, 1, "Minimal matrix"),
        (2, 2, 2, "Power of 2 small"),
        (3, 3, 3, "Non-power of 2 small"),
        (7, 11, 13, "Prime dimensions"),
        (32, 32, 32, "Tile-aligned"),
        (33, 33, 33, "Tile-misaligned"),
        (64, 1, 64, "Vector-like"),
        (1, 64, 1, "Row vector"),
        (100, 100, 100, "Round number"),
        (127, 127, 127, "Odd size"),
    ];
    
    println!("Running compatibility tests on {} ({})", info.name, info.vendor);
    
    let mut passed = 0;
    let mut failed = 0;
    
    for &(m, n, k, description) in &test_scenarios {
        print!("  Testing {}: {}x{}x{} ... ", description, m, n, k);
        
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32 + 1.0) / (m*k) as f32).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32 + 1.0) / (k*n) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        
        match executor.matmul_f32(&a, &b, &mut c, m, n, k) {
            Ok(()) => {
                // Basic sanity check
                let all_finite = c.iter().all(|&x| x.is_finite());
                if all_finite {
                    println!("✓ PASS");
                    passed += 1;
                } else {
                    println!("❌ FAIL (non-finite values)");
                    failed += 1;
                }
            }
            Err(e) => {
                println!("❌ FAIL ({})", e);
                failed += 1;
            }
        }
    }
    
    println!("\nCompatibility Summary: {} passed, {} failed", passed, failed);
    
    if failed == 0 {
        println!("🎉 Perfect cross-platform compatibility!");
    } else if failed < 3 {
        println!("⚠️ Minor compatibility issues - mostly working");
    } else {
        println!("🚨 Significant compatibility problems detected");
    }
    
    Ok(())
}

#[cfg(feature = "opencl")]
fn benchmark_opencl_performance(executor: &mut OpenClMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 OpenCL Performance Benchmark");
    println!("===============================");
    
    let info = executor.get_device_info();
    
    // Estimate theoretical peak performance
    let theoretical_gflops = info.compute_units as f64 * info.max_clock_frequency as f64 * 2.0 / 1000.0;
    println!("Theoretical peak performance: {:.0} GFLOPS", theoretical_gflops);
    
    let benchmark_size = 1536; // Good balance for various devices
    let m = benchmark_size;
    let n = benchmark_size;
    let k = benchmark_size;
    
    println!("Benchmark matrix size: {}x{}x{}", m, n, k);
    println!("Memory required: {:.1} MB", 
             (m * k + k * n + m * n) as f64 * 4.0 / 1_000_000.0);
    
    // Create benchmark matrices
    let a: Vec<f32> = (0..m*k).map(|i| ((i % 997) as f32) / 997.0).collect(); // Use prime for better distribution
    let b: Vec<f32> = (0..k*n).map(|i| (((i * 7) % 991) as f32) / 991.0).collect();
    let mut c = vec![0.0f32; m * n];
    
    // Warm-up
    executor.matmul_f32(&a, &b, &mut c, m, n, k)?;
    
    // Benchmark multiple runs
    let num_runs = 5;
    let mut durations = Vec::new();
    
    println!("\nRunning {} benchmark iterations...", num_runs);
    
    for run in 1..=num_runs {
        let start = Instant::now();
        executor.matmul_f32(&a, &b, &mut c, m, n, k)?;
        let duration = start.elapsed();
        durations.push(duration);
        
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        let efficiency = (gflops / theoretical_gflops) * 100.0;
        
        println!("Run {}: {:.3}ms ({:.0} GFLOPS, {:.1}% efficiency)", 
                 run, duration.as_secs_f64() * 1000.0, gflops, efficiency);
    }
    
    // Calculate statistics
    let avg_duration: f64 = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / num_runs as f64;
    let min_duration = durations.iter().min().unwrap().as_secs_f64();
    let max_duration = durations.iter().max().unwrap().as_secs_f64();
    
    let avg_gflops = (2.0 * (m * n * k) as f64) / (avg_duration * 1e9);
    let peak_gflops = (2.0 * (m * n * k) as f64) / (min_duration * 1e9);
    let avg_efficiency = (avg_gflops / theoretical_gflops) * 100.0;
    
    println!("\n📊 Benchmark Results:");
    println!("Average: {:.3}ms ({:.0} GFLOPS, {:.1}% efficiency)", 
             avg_duration * 1000.0, avg_gflops, avg_efficiency);
    println!("Best: {:.3}ms ({:.0} GFLOPS)", min_duration * 1000.0, peak_gflops);
    println!("Worst: {:.3}ms", max_duration * 1000.0);
    println!("Variation: {:.1}%", ((max_duration - min_duration) / avg_duration) * 100.0);
    
    // Performance analysis
    println!("\n🎯 Performance Analysis:");
    if avg_efficiency > 50.0 {
        println!("🟢 Excellent performance! OpenCL implementation is well-optimized for this device.");
    } else if avg_efficiency > 30.0 {
        println!("🟡 Good performance. Further optimizations possible with device-specific tuning.");
    } else if avg_efficiency > 15.0 {
        println!("🟠 Moderate performance. Consider different tile sizes or memory access patterns.");
    } else {
        println!("🔴 Performance below expectations. Check for memory bandwidth limitations.");
    }
    
    // Device-specific recommendations
    match info.vendor.as_str() {
        v if v.contains("AMD") => {
            println!("💡 AMD GPU: Consider optimizing for wavefront size (64) and LDS usage.");
        }
        v if v.contains("NVIDIA") => {
            println!("💡 NVIDIA GPU: Consider optimizing for warp size (32) and shared memory.");
        }
        v if v.contains("Intel") => {
            println!("💡 Intel GPU: Consider lower tile sizes and reduced local memory usage.");
        }
        _ => {
            println!("💡 Generic optimization: Try different tile sizes and work group configurations.");
        }
    }
    
    Ok(())
}

fn test_fallback_interface() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Fallback Interface");
    println!("-----------------------------");
    
    let a = vec![1.0f32; 16];
    let b = vec![2.0f32; 16];
    let mut c = vec![0.0f32; 16];
    
    match opencl_matmul_f32(&a, &b, &mut c, 4, 4, 4) {
        Ok(()) => {
            println!("❌ Unexpected success in fallback mode");
        }
        Err(e) => {
            println!("✓ Fallback correctly returned error: {}", e);
        }
    }
    
    Ok(())
}

// Stub functions for non-OpenCL builds
#[cfg(not(feature = "opencl"))]
fn display_device_info(_: &()) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
#[cfg(not(feature = "opencl"))]
fn test_matrix_multiplication_performance(_: &mut ()) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
#[cfg(not(feature = "opencl"))]
fn test_vendor_optimizations(_: &mut ()) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
#[cfg(not(feature = "opencl"))]
fn test_cross_platform_compatibility(_: &mut ()) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
#[cfg(not(feature = "opencl"))]
fn benchmark_opencl_performance(_: &mut ()) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }