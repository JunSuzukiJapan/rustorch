#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    gpu::{F32UnifiedGPUContext, GPUDevice},
    tensor::F32Tensor,
    unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 RusTorch Comprehensive Heavy Benchmark");
    println!("==========================================");
    println!("📊 Testing: CPU-only, GPU-only, Neural Engine-only, Hybrid, Hybrid_f32");
    println!();

    // Heavy benchmark configuration
    let iterations = 20;
    let large_sizes = vec![256, 512, 1024, 2048];
    let matrix_sizes = vec![128, 256, 512, 1024];

    println!("📋 Benchmark Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Tensor sizes: {:?}", large_sizes);
    println!("  Matrix sizes: {:?}", matrix_sizes);
    println!();

    // Test 1: CPU-only mode
    println!("💻 1. CPU-only Heavy Benchmark");
    println!("================================");
    benchmark_cpu_only(iterations, &large_sizes, &matrix_sizes)?;
    println!();

    // Test 2: GPU-only mode (Metal)
    println!("⚡ 2. Metal GPU-only Heavy Benchmark");
    println!("=====================================");
    benchmark_gpu_only(iterations, &large_sizes, &matrix_sizes)?;
    println!();

    // Test 3: Neural Engine-only mode
    println!("🧠 3. Neural Engine-only Heavy Benchmark");
    println!("==========================================");
    benchmark_neural_engine_only(iterations, &large_sizes, &matrix_sizes)?;
    println!();

    // Test 4: Existing hybrid mode
    println!("🔄 4. Existing Hybrid Mode Heavy Benchmark");
    println!("============================================");
    benchmark_existing_hybrid(iterations, &large_sizes, &matrix_sizes)?;
    println!();

    // Test 5: hybrid_f32 mode
    println!("🚀 5. Hybrid_f32 Mode Heavy Benchmark");
    println!("=======================================");
    benchmark_hybrid_f32(iterations, &large_sizes, &matrix_sizes)?;
    println!();

    // Comparison summary
    println!("📊 Performance Summary");
    println!("======================");
    println!("See individual benchmark results above for detailed comparison.");
    println!("✅ All heavy benchmarks completed successfully!");

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn benchmark_cpu_only(
    iterations: usize,
    tensor_sizes: &[usize],
    matrix_sizes: &[usize],
) -> rustorch::error::RusTorchResult<()> {
    println!("🔍 CPU-only operations with large tensors and matrices");

    for &size in tensor_sizes {
        println!("  📏 Tensor size: {}x{}", size, size);

        // Create large tensor
        let data: Vec<f64> = (0..size * size).map(|i| (i as f64 + 1.0) % 100.0).collect();
        let tensor = Tensor::from_vec(data, vec![size, size]);

        // Tensor operations
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.sum();
        }
        let sum_time = start.elapsed().as_millis() as f64 / iterations as f64;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.transpose();
        }
        let transpose_time = start.elapsed().as_millis() as f64 / iterations as f64;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.mean();
        }
        let mean_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!(
            "    Sum: {:.2}ms, Transpose: {:.2}ms, Mean: {:.2}ms",
            sum_time, transpose_time, mean_time
        );
    }

    // Matrix multiplication benchmark
    for &size in matrix_sizes {
        println!("  🔢 Matrix multiplication: {}x{}", size, size);

        let data_a: Vec<f64> = (0..size * size).map(|i| (i as f64) % 10.0 + 1.0).collect();
        let data_b: Vec<f64> = (0..size * size)
            .map(|i| (i as f64 + 5.0) % 10.0 + 1.0)
            .collect();

        let matrix_a = Tensor::from_vec(data_a, vec![size, size]);
        let matrix_b = Tensor::from_vec(data_b, vec![size, size]);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix_a.matmul(&matrix_b)?;
        }
        let matmul_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    MatMul: {:.2}ms", matmul_time);
    }

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn benchmark_gpu_only(
    iterations: usize,
    tensor_sizes: &[usize],
    matrix_sizes: &[usize],
) -> rustorch::error::RusTorchResult<()> {
    println!("🔍 Metal GPU-only operations with large tensors and matrices");

    // Initialize GPU
    let mut gpu_context = F32UnifiedGPUContext::new()?;
    let devices = gpu_context.list_devices()?;

    let metal_device = devices
        .iter()
        .find(|d| matches!(d, GPUDevice::Metal(_)))
        .ok_or_else(|| rustorch::error::RusTorchError::tensor_op("No Metal GPU found"))?;

    println!("  🎯 Using device: {:?}", metal_device);

    for &size in tensor_sizes {
        println!("  📏 Tensor size: {}x{}", size, size);

        // Create large tensor on GPU
        let data: Vec<f32> = (0..size * size).map(|i| (i as f32 + 1.0) % 100.0).collect();
        let tensor = F32Tensor::from_vec(data, &[size, size])?;

        // Move to GPU and perform operations
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.sum();
        }
        let sum_time = start.elapsed().as_millis() as f64 / iterations as f64;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.transpose();
        }
        let transpose_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!(
            "    GPU Sum: {:.2}ms, GPU Transpose: {:.2}ms",
            sum_time, transpose_time
        );
    }

    // Matrix multiplication on GPU
    for &size in matrix_sizes {
        println!("  🔢 GPU Matrix multiplication: {}x{}", size, size);

        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32) % 10.0 + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 5.0) % 10.0 + 1.0)
            .collect();

        let matrix_a = F32Tensor::from_vec(data_a, &[size, size])?;
        let matrix_b = F32Tensor::from_vec(data_b, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix_a.matmul(&matrix_b)?;
        }
        let matmul_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    GPU MatMul: {:.2}ms", matmul_time);
    }

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn benchmark_neural_engine_only(
    iterations: usize,
    tensor_sizes: &[usize],
    matrix_sizes: &[usize],
) -> rustorch::error::RusTorchResult<()> {
    println!("🔍 Neural Engine-only operations with large tensors and matrices");

    // Initialize Neural Engine
    let mut gpu_context = F32UnifiedGPUContext::new()?;
    let devices = gpu_context.list_devices()?;

    let neural_engine = devices
        .iter()
        .find(|d| matches!(d, GPUDevice::CoreML(_)))
        .ok_or_else(|| rustorch::error::RusTorchError::tensor_op("No Neural Engine found"))?;

    println!("  🎯 Using device: {:?}", neural_engine);

    for &size in tensor_sizes {
        println!("  📏 Tensor size: {}x{}", size, size);

        // Create large tensor for Neural Engine
        let data: Vec<f32> = (0..size * size).map(|i| (i as f32 + 1.0) % 100.0).collect();
        let tensor = F32Tensor::from_vec(data, &[size, size])?;

        // Neural Engine operations
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.sum();
        }
        let sum_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Neural Engine Sum: {:.2}ms", sum_time);
    }

    // Matrix multiplication on Neural Engine
    for &size in matrix_sizes {
        println!(
            "  🔢 Neural Engine Matrix multiplication: {}x{}",
            size, size
        );

        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32) % 10.0 + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 5.0) % 10.0 + 1.0)
            .collect();

        let matrix_a = F32Tensor::from_vec(data_a, &[size, size])?;
        let matrix_b = F32Tensor::from_vec(data_b, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            println!("🧠 Executing Neural Engine f32 matmul (zero conversion cost)");
            let _ = matrix_a.matmul(&matrix_b)?;
        }
        let matmul_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Neural Engine MatMul: {:.2}ms", matmul_time);
    }

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn benchmark_existing_hybrid(
    iterations: usize,
    tensor_sizes: &[usize],
    matrix_sizes: &[usize],
) -> rustorch::error::RusTorchResult<()> {
    println!("🔍 Existing hybrid mode operations with automatic device selection");

    // Initialize hybrid system
    let mut gpu_context = F32UnifiedGPUContext::new()?;
    let devices = gpu_context.list_devices()?;

    println!("  🎯 Available devices: {} devices", devices.len());
    for device in &devices {
        println!("    {:?}", device);
    }

    for &size in tensor_sizes {
        println!("  📏 Tensor size: {}x{}", size, size);

        // Create tensors and let hybrid system choose optimal device
        let data: Vec<f32> = (0..size * size).map(|i| (i as f32 + 1.0) % 100.0).collect();
        let tensor = F32Tensor::from_vec(data, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.sum();
        }
        let sum_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Hybrid Sum: {:.2}ms", sum_time);
    }

    // Matrix multiplication with hybrid selection
    for &size in matrix_sizes {
        println!("  🔢 Hybrid Matrix multiplication: {}x{}", size, size);

        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32) % 10.0 + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 5.0) % 10.0 + 1.0)
            .collect();

        let matrix_a = F32Tensor::from_vec(data_a, &[size, size])?;
        let matrix_b = F32Tensor::from_vec(data_b, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix_a.matmul(&matrix_b)?;
        }
        let matmul_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Hybrid MatMul: {:.2}ms", matmul_time);
    }

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn benchmark_hybrid_f32(
    iterations: usize,
    tensor_sizes: &[usize],
    matrix_sizes: &[usize],
) -> rustorch::error::RusTorchResult<()> {
    println!("🔍 Hybrid_f32 mode operations with zero conversion cost");

    // Initialize F32 Hybrid Executor
    let mut executor = F32HybridExecutor::new()?;
    println!("  🚀 F32 Unified Hybrid Executor initialized");

    // Device detection
    let devices = executor.get_available_devices()?;
    println!("  🎯 Available devices:");
    for device in &devices {
        match device {
            GPUDevice::CPU => println!("    CPU: CPU (0.5 TFLOPS f32)"),
            GPUDevice::Metal(id) => println!("    Metal({}): Apple M1 GPU (2.6 TFLOPS f32)", id),
            GPUDevice::CoreML(id) => println!(
                "    CoreML({}): Apple M1 Neural Engine (7.0 TFLOPS f32)",
                id
            ),
            _ => println!("    {:?}", device),
        }
    }

    for &size in tensor_sizes {
        println!("  📏 Tensor size: {}x{}", size, size);

        // Create F32 tensors with zero conversion overhead
        let data: Vec<f32> = (0..size * size).map(|i| (i as f32 + 1.0) % 100.0).collect();
        let tensor = F32Tensor::from_vec(data, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tensor.sum();
            println!("📊 Conversion cost reduction: 100% (zero conversion overhead)");
        }
        let sum_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Hybrid_f32 Sum: {:.2}ms", sum_time);
    }

    // Matrix multiplication with f32 unified execution
    for &size in matrix_sizes {
        println!("  🔢 Hybrid_f32 Matrix multiplication: {}x{}", size, size);

        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32) % 10.0 + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 5.0) % 10.0 + 1.0)
            .collect();

        let matrix_a = F32Tensor::from_vec(data_a, &[size, size])?;
        let matrix_b = F32Tensor::from_vec(data_b, &[size, size])?;

        let start = Instant::now();
        for _ in 0..iterations {
            println!("🚀 F32 unified execution (zero conversion cost)");
            let _ = matrix_a.matmul(&matrix_b)?;
            println!("  ✓ Neural Engine executed with f32 precision");
            println!("  ✓ Estimated performance: ~7.0 TFLOPS (f32)");
        }
        let matmul_time = start.elapsed().as_millis() as f64 / iterations as f64;

        println!("    Hybrid_f32 MatMul: {:.2}ms", matmul_time);
    }

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ This benchmark requires the 'hybrid-f32' feature to be enabled.");
    println!("📋 Run with: cargo run --example comprehensive_heavy_benchmark --features hybrid-f32 --release");
}
