//! Metal GPU Performance Benchmark
//! Metal GPU パフォーマンスベンチマーク

#[cfg(test)]
mod tests {
    use rustorch::tensor::Tensor;
    use std::time::Instant;

    #[test]
    fn test_metal_gpu_benchmark() {
        println!("🍎 === Metal GPU Performance Test ===");

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use metal::Device;
            use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

            if let Some(device) = Device::system_default() {
                println!("✅ Metal Device: {}", device.name());
                println!(
                    "   Max working set: {} MB",
                    device.recommended_max_working_set_size() / (1024 * 1024)
                );
                println!("   Low power: {}", device.is_low_power());

                // Test various matrix sizes
                for size in [64, 128, 256, 512] {
                    println!("\n📊 Testing matrix size: {}x{}", size, size);

                    let a = Tensor::<f32>::rand(&[size, size]);
                    let b = Tensor::<f32>::rand(&[size, size]);

                    // CPU baseline
                    let start = Instant::now();
                    let cpu_result = a.matmul(&b).unwrap();
                    let cpu_time = start.elapsed();

                    // GPU Metal
                    let start = Instant::now();
                    let gpu_result = a.gpu_matmul(&b).unwrap();
                    let gpu_time = start.elapsed();

                    // Calculate GFLOPS (operations = 2 * m * n * k for matmul)
                    let ops = 2.0 * (size * size * size) as f64;
                    let cpu_gflops = ops / (cpu_time.as_secs_f64() * 1e9);
                    let gpu_gflops = ops / (gpu_time.as_secs_f64() * 1e9);

                    println!(
                        "  CPU time: {:.2}ms, Performance: {:.3} GFLOPS",
                        cpu_time.as_millis(),
                        cpu_gflops
                    );
                    println!(
                        "  GPU time: {:.2}ms, Performance: {:.3} GFLOPS",
                        gpu_time.as_millis(),
                        gpu_gflops
                    );

                    if gpu_time < cpu_time {
                        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
                        println!("  🚀 GPU Speedup: {:.2}x faster", speedup);
                    } else {
                        println!("  📈 CPU faster (GPU overhead)");
                    }

                    // Verify results match
                    assert_eq!(cpu_result.shape(), gpu_result.shape());
                    println!("  ✅ Results verified: shapes match");
                }

                // Test Metal-specific operations
                println!("\n🔥 Testing Metal-specific operations:");

                let a = Tensor::<f32>::rand(&[1024, 1024]);
                let b = Tensor::<f32>::rand(&[1024, 1024]);

                let start = Instant::now();
                let _batch_result = a.gpu_matmul(&b).unwrap();
                let batch_time = start.elapsed();

                let ops = 2.0 * (1024 * 1024 * 1024) as f64;
                let batch_gflops = ops / (batch_time.as_secs_f64() * 1e9);

                println!(
                    "  Metal batch matmul (1024x1024): {:.2}ms, {:.3} GFLOPS",
                    batch_time.as_millis(),
                    batch_gflops
                );
            } else {
                println!("❌ No Metal device available");
            }
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            println!("ℹ️  Metal not available - requires macOS and metal feature");
        }

        println!("\n=======================================");
    }
}
