//! GPU Performance Demonstration
//! GPU性能デモンストレーション

#[cfg(not(target_arch = "wasm32"))]
use rustorch::gpu::DeviceManager;
use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Performance Demo ===");

    #[cfg(not(target_arch = "wasm32"))]
    {
        // System information
        let manager = DeviceManager::new();
        let devices = manager.available_devices();

        println!("Available devices: {:?}", devices);
        println!("CUDA available: {}", DeviceManager::is_cuda_available());
        println!("Metal available: {}", DeviceManager::is_metal_available());
        println!();
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("GPU operations are not available in WASM target.");
        println!("This demo shows CPU tensor operations only.\n");
    }

    // Test different sizes
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        println!("--- Testing {}x{} matrices ---", size, size);

        let a = Tensor::<f32>::ones(&[size, size]);
        let b = Tensor::<f32>::ones(&[size, size]);

        // CPU matrix multiplication
        let start = Instant::now();
        let _cpu_result = a.matmul(&b)?;
        let cpu_time = start.elapsed();

        #[cfg(not(target_arch = "wasm32"))]
        {
            // GPU matrix multiplication (with fallback)
            let start = Instant::now();
            use rustorch::gpu::matrix_ops::GpuLinearAlgebra;
            let _gpu_result = a.gpu_matmul(&b)?;
            let gpu_time = start.elapsed();

            println!("CPU time: {:?}", cpu_time);
            println!("GPU time: {:?}", gpu_time);

            if gpu_time.as_nanos() > 0 {
                let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
                println!("Speedup: {:.2}x", speedup);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            println!("CPU time: {:?}", cpu_time);
            println!("GPU acceleration not available in WASM");
        }

        println!();
    }

    // Test reduction operations
    println!("--- Testing Reduction Operations ---");
    let tensor = Tensor::<f32>::ones(&[1000, 1000]);

    // CPU sum
    let start = Instant::now();
    let cpu_sum = tensor.sum();
    let cpu_sum_time = start.elapsed();

    #[cfg(not(target_arch = "wasm32"))]
    {
        // GPU sum
        let start = Instant::now();
        let gpu_sum = tensor.gpu_sum(None)?;
        let gpu_sum_time = start.elapsed();

        println!("CPU sum result: {}, time: {:?}", cpu_sum, cpu_sum_time);
        println!(
            "GPU sum result: {}, time: {:?}",
            gpu_sum.data[0], gpu_sum_time
        );

        // Verify results match
        let diff = (cpu_sum - gpu_sum.data[0]).abs();
        println!("Result difference: {:.6}", diff);

        if diff < 1e-3 {
            println!("✅ Results match!");
        } else {
            println!("⚠️  Results differ significantly");
        }

        println!("\n--- GPU Operations Test Complete ---");
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("CPU sum result: {}, time: {:?}", cpu_sum, cpu_sum_time);
        println!("GPU sum not available in WASM");
        println!("\n--- CPU Operations Test Complete ---");
    }

    Ok(())
}
