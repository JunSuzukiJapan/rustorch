/// GPU kernel demonstration and validation example
/// GPUカーネルデモンストレーションと検証例
use rustorch::gpu::{
    kernels::{AddKernel, KernelExecutor, MatMulKernel},
    validation::print_gpu_validation_report,
    DeviceType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RusTorch GPU Kernel Demo ===\n");

    // Print available devices
    println!("Available GPU devices:");
    let devices = vec![
        DeviceType::Cpu,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(0),
        #[cfg(feature = "metal")]
        DeviceType::Metal(0),
        #[cfg(feature = "opencl")]
        DeviceType::OpenCL(0),
    ];

    for device in &devices {
        if device.is_available() {
            println!("  ✓ {}", device);
        } else {
            println!("  ✗ {} (not available)", device);
        }
    }
    println!();

    // Demonstrate element-wise addition
    demo_elementwise_addition()?;

    // Demonstrate matrix multiplication
    demo_matrix_multiplication()?;

    // Run comprehensive validation
    println!("=== Running GPU Validation ===");
    print_gpu_validation_report();

    // Performance comparison
    performance_comparison()?;

    // Optional: Metal specific demo
    #[cfg(feature = "metal")]
    demo_metal_specific()?;

    Ok(())
}

fn demo_elementwise_addition() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Element-wise Addition Demo ===");

    let size = 1024;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut c = vec![0.0f32; size];

    let kernel = AddKernel;

    for device in &[DeviceType::Cpu] {
        if !device.is_available() {
            continue;
        }

        let executor = KernelExecutor::new(*device);
        let start_time = std::time::Instant::now();

        let inputs = [a.as_slice(), b.as_slice()];
        let mut outputs = [c.as_mut_slice()];

        executor.execute_kernel(&kernel, &inputs, &mut outputs)?;

        let elapsed = start_time.elapsed();

        // Verify results
        let correct = c.iter().all(|&x| (x - 3.0).abs() < 1e-6);

        println!(
            "  {}: {} ({:.2}ms) - {}",
            device,
            if correct { "✓" } else { "✗" },
            elapsed.as_secs_f64() * 1000.0,
            if correct { "PASS" } else { "FAIL" }
        );

        // Reset output for next device
        c.fill(0.0);
    }

    println!();
    Ok(())
}

fn demo_matrix_multiplication() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Matrix Multiplication Demo ===");

    let n = 4;
    let size = n * n;

    // Create test matrices
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];

    // A = sequential values, B = identity matrix
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (i * n + j + 1) as f32;
            b[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let mut c = vec![0.0f32; size];
    let kernel = MatMulKernel;

    for device in &[DeviceType::Cpu] {
        if !device.is_available() {
            continue;
        }

        let executor = KernelExecutor::new(*device);
        let start_time = std::time::Instant::now();

        let inputs = [a.as_slice(), b.as_slice()];
        let mut outputs = [c.as_mut_slice()];

        executor.execute_kernel(&kernel, &inputs, &mut outputs)?;

        let elapsed = start_time.elapsed();

        // Verify results (A * I = A)
        let correct = a
            .iter()
            .zip(c.iter())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-5);

        println!(
            "  {} ({}x{}): {} ({:.2}ms) - {}",
            device,
            n,
            n,
            if correct { "✓" } else { "✗" },
            elapsed.as_secs_f64() * 1000.0,
            if correct { "PASS" } else { "FAIL" }
        );

        // Reset output for next device
        c.fill(0.0);
    }

    println!();
    Ok(())
}

fn performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Comparison ===");

    let sizes = vec![1024, 4096, 16384];
    let kernel = AddKernel;

    for size in sizes {
        println!("Size: {} elements", size);

        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut c = vec![0.0f32; size];

        for device in &[DeviceType::Cpu] {
            if !device.is_available() {
                continue;
            }

            let executor = KernelExecutor::new(*device);

            // Warmup
            let inputs = [a.as_slice(), b.as_slice()];
            let mut outputs = [c.as_mut_slice()];
            executor.execute_kernel(&kernel, &inputs, &mut outputs)?;

            // Benchmark
            let iterations = 10;
            let start_time = std::time::Instant::now();

            for _ in 0..iterations {
                let inputs = [a.as_slice(), b.as_slice()];
                let mut outputs = [c.as_mut_slice()];
                executor.execute_kernel(&kernel, &inputs, &mut outputs)?;
            }

            let elapsed = start_time.elapsed();
            let avg_time = elapsed.as_secs_f64() / iterations as f64;
            let throughput = size as f64 / avg_time / 1e6; // Million elements per second

            println!(
                "  {}: {:.2}ms avg, {:.1} Melem/s",
                device,
                avg_time * 1000.0,
                throughput
            );
        }
        println!();
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn demo_cuda_specific() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::gpu::cuda_kernels::{cuda_elementwise_add_f32, CudaKernelExecutor};

    println!("=== CUDA Specific Demo ===");

    if let Ok(executor) = CudaKernelExecutor::new(0) {
        let size = 1024;
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut c = vec![0.0f32; size];

        let start_time = std::time::Instant::now();
        cuda_elementwise_add_f32(&a, &b, &mut c)?;
        let elapsed = start_time.elapsed();

        let correct = c.iter().all(|&x| (x - 3.0).abs() < 1e-6);

        println!(
            "  CUDA Direct Call: {} ({:.2}ms)",
            if correct { "✓" } else { "✗" },
            elapsed.as_secs_f64() * 1000.0
        );
    } else {
        println!("  CUDA not available");
    }

    println!();
    Ok(())
}

#[cfg(feature = "metal")]
fn demo_metal_specific() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::gpu::metal_kernels::{metal_elementwise_add_f32, MetalKernelExecutor};

    println!("=== Metal Specific Demo ===");

    if let Ok(_executor) = MetalKernelExecutor::new() {
        let size = 1024;
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut c = vec![0.0f32; size];

        let start_time = std::time::Instant::now();
        metal_elementwise_add_f32(&a, &b, &mut c)?;
        let elapsed = start_time.elapsed();

        let correct = c.iter().all(|&x| (x - 3.0).abs() < 1e-6);

        println!(
            "  Metal Direct Call: {} ({:.2}ms)",
            if correct { "✓" } else { "✗" },
            elapsed.as_secs_f64() * 1000.0
        );
    } else {
        println!("  Metal not available");
    }

    println!();
    Ok(())
}
