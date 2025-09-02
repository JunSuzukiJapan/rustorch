//! CUDA cuBLAS Performance demonstration for high-end NVIDIA GPUs
//! 高性能NVIDIA GPU向けCUDA cuBLASパフォーマンスデモンストレーション

use rustorch::gpu::cuda_enhanced::CudaMatrixExecutor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RusTorch CUDA cuBLAS Performance Demo");
    println!("高性能CUDA cuBLAS パフォーマンスデモ");
    println!("=====================================");

    #[cfg(feature = "cuda")]
    {
        // Initialize CUDA executor
        match CudaMatrixExecutor::new(0) {
            Ok(executor) => {
                println!("✓ CUDA executor initialized on device 0");

                // Display device information
                display_device_info(&executor)?;

                // Test standard matrix multiplication
                test_standard_matrix_multiplication(&executor)?;

                // Test Tensor Core acceleration (if available)
                test_tensor_core_acceleration(&executor)?;

                // Test batch matrix multiplication
                test_batch_matrix_multiplication(&executor)?;

                // Performance benchmark against theoretical peak
                benchmark_cuda_performance(&executor)?;

                println!("\n🎯 All CUDA cuBLAS tests completed successfully!");
            }
            Err(e) => {
                println!("❌ CUDA initialization failed: {}", e);
                println!("This is expected if CUDA is not available on this system.");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA feature not enabled");
        println!("Run with: cargo run --example cuda_performance_demo --features cuda");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn display_device_info(executor: &CudaMatrixExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 CUDA Device Information");
    println!("---------------------------");

    match executor.get_compute_capability() {
        Ok((major, minor)) => {
            println!("Compute Capability: {}.{}", major, minor);

            // Check for Tensor Core support
            let has_tensor_cores = (major > 7) || (major == 7 && minor >= 0);
            println!(
                "Tensor Core Support: {}",
                if has_tensor_cores {
                    "✓ Available"
                } else {
                    "❌ Not Available"
                }
            );

            // Architecture information
            let arch_name = match (major, minor) {
                (8, 6) => "Ada Lovelace (RTX 40xx)",
                (8, 0) => "Ampere (A100)",
                (7, 5) => "Turing (RTX 20xx/GTX 16xx)",
                (7, 0) => "Volta (V100)",
                (6, 1) => "Pascal (GTX 10xx)",
                (5, 2) => "Maxwell",
                _ => "Unknown Architecture",
            };
            println!("Architecture: {}", arch_name);
        }
        Err(e) => println!("Failed to get compute capability: {}", e),
    }

    match executor.get_memory_info() {
        Ok((free, total)) => {
            println!(
                "GPU Memory: {:.1} GB total, {:.1} GB free",
                total as f64 / 1024.0 / 1024.0 / 1024.0,
                free as f64 / 1024.0 / 1024.0 / 1024.0
            );
        }
        Err(e) => println!("Failed to get memory info: {}", e),
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn test_standard_matrix_multiplication(
    executor: &CudaMatrixExecutor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Standard cuBLAS SGEMM");
    println!("----------------------------------");

    let sizes = [128, 512, 1024, 2048];

    for &size in &sizes {
        let m = size;
        let n = size;
        let k = size;

        // Create test matrices
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 2) as f32) / (k * n) as f32)
            .collect();
        let mut c = vec![0.0f32; m * n];

        let start = Instant::now();
        executor.matmul_f32(&a, &b, &mut c, m, n, k, false)?;
        let duration = start.elapsed();

        // Calculate GFLOPS (2 * M * N * K operations)
        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);

        // Calculate memory bandwidth utilization
        let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0; // bytes
        let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);

        println!(
            "✓ Matrix {}x{}x{}: {:.3}ms ({:.1} GFLOPS, {:.1} GB/s)",
            m,
            n,
            k,
            duration.as_secs_f64() * 1000.0,
            gflops,
            bandwidth_gb_s
        );

        // Verify correctness (simple check)
        let expected_sum = a.iter().sum::<f32>() * b.iter().sum::<f32>() / (k as f32);
        let actual_sum = c.iter().sum::<f32>();
        let relative_error = ((actual_sum - expected_sum) / expected_sum).abs();

        if relative_error < 0.01 {
            println!(
                "  ✓ Correctness check passed (error: {:.4}%)",
                relative_error * 100.0
            );
        } else {
            println!("  ⚠️ Large numerical error: {:.4}%", relative_error * 100.0);
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn test_tensor_core_acceleration(
    executor: &CudaMatrixExecutor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Tensor Core Acceleration");
    println!("------------------------------------");

    // Check if Tensor Cores are supported
    match executor.get_compute_capability() {
        Ok((major, minor)) if major > 7 || (major == 7 && minor >= 0) => {
            println!("Tensor Cores available - testing mixed precision");

            let sizes = [512, 1024, 2048];

            for &size in &sizes {
                let _m = size;
                let _n = size;
                let _k = size;

                // Ensure dimensions are multiples of 8 for Tensor Core efficiency
                let aligned_size = (size + 7) & !7;
                let aligned_m = aligned_size;
                let aligned_n = aligned_size;
                let aligned_k = aligned_size;

                // Create test matrices
                let a: Vec<f32> = (0..aligned_m * aligned_k)
                    .map(|i| (i as f32) / 1000.0)
                    .collect();
                let b: Vec<f32> = (0..aligned_k * aligned_n)
                    .map(|i| ((i * 3) as f32) / 1000.0)
                    .collect();
                let mut c_standard = vec![0.0f32; aligned_m * aligned_n];
                let mut c_tensor = vec![0.0f32; aligned_m * aligned_n];

                // Standard SGEMM
                let start = Instant::now();
                executor.matmul_f32(
                    &a,
                    &b,
                    &mut c_standard,
                    aligned_m,
                    aligned_n,
                    aligned_k,
                    false,
                )?;
                let standard_time = start.elapsed();

                // Tensor Core accelerated (simulated)
                let start = Instant::now();
                executor.matmul_f32(
                    &a,
                    &b,
                    &mut c_tensor,
                    aligned_m,
                    aligned_n,
                    aligned_k,
                    true,
                )?;
                let tensor_time = start.elapsed();

                let flops = 2.0 * (aligned_m * aligned_n * aligned_k) as f64;
                let standard_gflops = flops / (standard_time.as_secs_f64() * 1e9);
                let tensor_gflops = flops / (tensor_time.as_secs_f64() * 1e9);

                let speedup = standard_time.as_secs_f64() / tensor_time.as_secs_f64();

                println!("✓ Matrix {}x{}x{}: Standard {:.1} GFLOPS, Tensor {:.1} GFLOPS ({:.2}x speedup)", 
                         aligned_m, aligned_n, aligned_k, standard_gflops, tensor_gflops, speedup);
            }
        }
        Ok((major, minor)) => {
            println!(
                "Tensor Cores not available on compute capability {}.{}",
                major, minor
            );
            println!("Requires Volta (7.0+) or newer architecture");
        }
        Err(e) => {
            println!("Could not determine compute capability: {}", e);
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn test_batch_matrix_multiplication(
    executor: &CudaMatrixExecutor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Batch Matrix Multiplication");
    println!("--------------------------------------");

    let batch_sizes = [4, 16, 64];
    let matrix_size = 256;

    for &batch_size in &batch_sizes {
        let m = matrix_size;
        let n = matrix_size;
        let k = matrix_size;

        // Create batch of test matrices
        let mut a_batch = Vec::new();
        let mut b_batch = Vec::new();
        let mut c_batch = Vec::new();

        for i in 0..batch_size {
            let a: Vec<f32> = (0..m * k)
                .map(|j| ((i * 1000 + j) as f32) / 10000.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|j| (((i + 1) * 1000 + j * 2) as f32) / 10000.0)
                .collect();
            let c = vec![0.0f32; m * n];

            a_batch.push(a);
            b_batch.push(b);
            c_batch.push(c);
        }

        let start = Instant::now();
        executor.batch_matmul_f32(&a_batch, &b_batch, &mut c_batch, m, n, k)?;
        let duration = start.elapsed();

        // Calculate total FLOPS for the entire batch
        let total_flops = batch_size as f64 * 2.0 * (m * n * k) as f64;
        let gflops = total_flops / (duration.as_secs_f64() * 1e9);

        println!(
            "✓ Batch {} matrices ({}x{}x{}): {:.3}ms ({:.1} GFLOPS)",
            batch_size,
            m,
            n,
            k,
            duration.as_secs_f64() * 1000.0,
            gflops
        );

        // Verify one result
        if !c_batch[0].iter().all(|&x| x.is_finite()) {
            println!("  ⚠️ Non-finite values detected in result");
        } else {
            println!("  ✓ All results are finite");
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_cuda_performance(
    executor: &CudaMatrixExecutor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 CUDA Performance Benchmark");
    println!("==============================");

    // Get compute capability for theoretical performance estimation
    let (major, minor) = executor.get_compute_capability()?;

    // Estimate theoretical peak performance (very rough approximation)
    let theoretical_gflops = match (major, minor) {
        (8, 6) => 35000.0, // RTX 4090: ~35 TFLOPS FP32
        (8, 0) => 19500.0, // A100: ~19.5 TFLOPS FP32
        (7, 5) => 16000.0, // RTX 2080 Ti: ~16 TFLOPS FP32
        (7, 0) => 15700.0, // V100: ~15.7 TFLOPS FP32
        (6, 1) => 11000.0, // GTX 1080 Ti: ~11 TFLOPS FP32
        _ => 5000.0,       // Conservative estimate for older architectures
    };

    println!(
        "Estimated peak performance: {:.0} GFLOPS",
        theoretical_gflops
    );

    let large_size = 4096;
    let m = large_size;
    let n = large_size;
    let k = large_size;

    println!("Matrix size: {}x{}x{}", m, n, k);
    println!(
        "Memory required: {:.1} GB",
        (m * k + k * n + m * n) as f64 * 4.0 / 1_000_000_000.0
    );

    // Create large test matrices
    let a: Vec<f32> = (0..m * k).map(|i| (i % 1000) as f32 / 1000.0).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 3) % 1000) as f32 / 1000.0)
        .collect();
    let mut c = vec![0.0f32; m * n];

    // Warm-up
    executor.matmul_f32(&a, &b, &mut c, m, n, k, false)?;

    // Benchmark multiple runs
    let num_runs = 3;
    let mut durations = Vec::new();

    for run in 1..=num_runs {
        let start = Instant::now();
        executor.matmul_f32(&a, &b, &mut c, m, n, k, false)?;
        let duration = start.elapsed();
        durations.push(duration);

        let flops = 2.0 * (m * n * k) as f64;
        let gflops = flops / (duration.as_secs_f64() * 1e9);
        let efficiency = (gflops / theoretical_gflops) * 100.0;

        println!(
            "Run {}: {:.3}ms ({:.0} GFLOPS, {:.1}% of peak)",
            run,
            duration.as_secs_f64() * 1000.0,
            gflops,
            efficiency
        );
    }

    // Calculate statistics
    let avg_duration: f64 =
        durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / num_runs as f64;
    let avg_gflops = (2.0 * (m * n * k) as f64) / (avg_duration * 1e9);
    let avg_efficiency = (avg_gflops / theoretical_gflops) * 100.0;

    println!("\n📊 Average Performance:");
    println!("Duration: {:.3}ms", avg_duration * 1000.0);
    println!("GFLOPS: {:.0}", avg_gflops);
    println!("Efficiency: {:.1}% of theoretical peak", avg_efficiency);

    // Performance analysis
    if avg_efficiency > 80.0 {
        println!("🟢 Excellent performance! Near optimal GPU utilization.");
    } else if avg_efficiency > 60.0 {
        println!("🟡 Good performance. Some room for optimization.");
    } else if avg_efficiency > 40.0 {
        println!("🟠 Moderate performance. Consider optimization strategies.");
    } else {
        println!("🔴 Performance below expectations. Check for bottlenecks.");
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn display_device_info(_: &()) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn test_standard_matrix_multiplication(_: &()) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn test_tensor_core_acceleration(_: &()) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn test_batch_matrix_multiplication(_: &()) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn benchmark_cuda_performance(_: &()) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
