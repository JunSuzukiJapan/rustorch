//! Comprehensive GPU Performance Benchmark
//! 包括的GPU性能ベンチマークスイート - 全バックエンド統合テスト
//!
//! Tests all GPU backends (Metal, CUDA, OpenCL) against OpenBLAS CPU baseline
//! to provide performance comparison and automatic backend selection guidance.

use rustorch::tensor::Tensor;
use std::time::Instant;

#[cfg(feature = "linalg-system")]
use rustorch::linalg::optimized_matmul;

#[cfg(feature = "metal")]
use rustorch::gpu::metal_kernels::MetalKernelExecutor;

#[cfg(feature = "cuda")]
use rustorch::gpu::cuda_enhanced::CudaMatrixExecutor;

#[cfg(feature = "opencl")]
use rustorch::gpu::opencl_optimized::OpenClMatrixExecutor;

#[derive(Debug, Clone)]
struct BenchmarkResult {
    backend: String,
    matrix_size: usize,
    duration_ms: f64,
    gflops: f64,
    memory_bandwidth_gb_s: f64,
    efficiency_percent: f64,
    success: bool,
    error_message: Option<String>,
}

#[derive(Debug)]
struct SystemInfo {
    os: String,
    total_memory_gb: f64,
    cpu_cores: usize,
    gpu_available: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Comprehensive GPU Benchmark Suite");
    println!("包括的GPU性能ベンチマークスイート");
    println!("=============================================");

    // Display system information
    let system_info = collect_system_info();
    display_system_info(&system_info);

    // Benchmark configuration
    let test_sizes = vec![128, 256, 512, 1024, 2048];
    let mut all_results = Vec::new();

    for &size in &test_sizes {
        println!("\n📊 Testing Matrix Size: {}x{}x{}", size, size, size);
        println!(
            "Memory required: {:.1} MB",
            (size * size * 3) as f64 * 4.0 / 1_000_000.0
        );
        println!(
            "Theoretical FLOPS: {:.1} GFLOPS",
            2.0 * (size as f64).powi(3) / 1e9
        );
        println!("{}", "─".repeat(60));

        // Test OpenBLAS CPU baseline
        match benchmark_cpu_openblas(size) {
            Ok(cpu_result) => {
                all_results.push(cpu_result.clone());
                display_single_result(&cpu_result);
            }
            Err(e) => {
                println!("❌ CPU benchmark failed: {}", e);
                continue;
            }
        }

        // Test Metal (if available)
        #[cfg(feature = "metal")]
        {
            let metal_result = benchmark_metal(size);
            all_results.push(metal_result.clone());
            display_single_result(&metal_result);
        }

        // Test CUDA (if available)
        #[cfg(feature = "cuda")]
        {
            let cuda_result = benchmark_cuda(size);
            all_results.push(cuda_result.clone());
            display_single_result(&cuda_result);
        }

        // Test OpenCL (if available)
        #[cfg(feature = "opencl")]
        {
            let opencl_result = benchmark_opencl(size);
            all_results.push(opencl_result.clone());
            display_single_result(&opencl_result);
        }

        // Show relative performance for this size
        show_relative_performance(&all_results, size);
    }

    // Final analysis and recommendations
    analyze_and_recommend(&all_results, &system_info);

    Ok(())
}

fn collect_system_info() -> SystemInfo {
    #[allow(clippy::vec_init_then_push)]
    #[allow(unused_mut)]
    let gpu_available = {
        let mut vec = Vec::new();
        #[cfg(feature = "metal")]
        vec.push("Metal".to_string());
        #[cfg(feature = "cuda")]
        vec.push("CUDA".to_string());
        #[cfg(feature = "opencl")]
        vec.push("OpenCL".to_string());
        vec
    };

    // Suppress unused variable warning if no GPU features are enabled
    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "opencl")))]
    let _ = &gpu_available;

    SystemInfo {
        os: std::env::consts::OS.to_string(),
        total_memory_gb: 16.0, // Default estimate
        cpu_cores: num_cpus::get(),
        gpu_available,
    }
}

fn display_system_info(info: &SystemInfo) {
    println!("\n💻 System Information");
    println!("OS: {}", info.os);
    println!("CPU Cores: {}", info.cpu_cores);
    println!("Estimated Memory: {:.1} GB", info.total_memory_gb);
    println!("Available GPU Backends: {:?}", info.gpu_available);

    if info.gpu_available.is_empty() {
        println!("⚠️ No GPU backends available - CPU only benchmarking");
    }
}

fn benchmark_cpu_openblas(size: usize) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let m = size;
    let n = size;
    let k = size;

    // Create test matrices
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i * 2) as f32) / (k * n) as f32)
        .collect();

    let _a = Tensor::from_vec(a_data, vec![m, k]);
    let _b = Tensor::from_vec(b_data, vec![k, n]);

    let _start = Instant::now();
    #[cfg(feature = "linalg-system")]
    {
        match optimized_matmul(&_a, &_b) {
            Ok(_result) => {
                let duration = _start.elapsed();
                let duration_ms = duration.as_secs_f64() * 1000.0;
                let flops = 2.0 * (m * n * k) as f64;
                let gflops = flops / (duration.as_secs_f64() * 1e9);
                let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0;
                let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);

                // CPU efficiency against theoretical peak (rough estimate)
                let theoretical_gflops = 100.0; // Conservative estimate
                let efficiency = (gflops / theoretical_gflops * 100.0).min(100.0);

                Ok(BenchmarkResult {
                    backend: "OpenBLAS CPU".to_string(),
                    matrix_size: size,
                    duration_ms,
                    gflops,
                    memory_bandwidth_gb_s: bandwidth_gb_s,
                    efficiency_percent: efficiency,
                    success: true,
                    error_message: None,
                })
            }
            Err(e) => Ok(BenchmarkResult {
                backend: "OpenBLAS CPU".to_string(),
                matrix_size: size,
                duration_ms: 0.0,
                gflops: 0.0,
                memory_bandwidth_gb_s: 0.0,
                efficiency_percent: 0.0,
                success: false,
                error_message: Some(format!("{}", e)),
            }),
        }
    }

    #[cfg(not(feature = "linalg-system"))]
    {
        Ok(BenchmarkResult {
            backend: "CPU (no BLAS)".to_string(),
            matrix_size: size,
            duration_ms: 0.0,
            gflops: 0.0,
            memory_bandwidth_gb_s: 0.0,
            efficiency_percent: 0.0,
            success: false,
            error_message: Some("linalg-system feature not enabled".to_string()),
        })
    }
}

#[cfg(feature = "metal")]
fn benchmark_metal(size: usize) -> BenchmarkResult {
    match MetalKernelExecutor::new() {
        Ok(executor) => {
            let m = size;
            let n = size;
            let k = size;

            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 2) as f32) / (k * n) as f32)
                .collect();
            let mut c = vec![0.0f32; m * n];

            let start = Instant::now();
            match executor.matmul_f32(&a, &b, &mut c, m, n, k) {
                Ok(()) => {
                    let duration = start.elapsed();
                    let duration_ms = duration.as_secs_f64() * 1000.0;
                    let flops = 2.0 * (m * n * k) as f64;
                    let gflops = flops / (duration.as_secs_f64() * 1e9);
                    let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0;
                    let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);

                    // AMD Radeon Pro Vega 56 theoretical peak: ~7000 GFLOPS
                    let theoretical_gflops = 7000.0;
                    let efficiency = (gflops / theoretical_gflops * 100.0).min(100.0);

                    BenchmarkResult {
                        backend: "Metal".to_string(),
                        matrix_size: size,
                        duration_ms,
                        gflops,
                        memory_bandwidth_gb_s: bandwidth_gb_s,
                        efficiency_percent: efficiency,
                        success: true,
                        error_message: None,
                    }
                }
                Err(e) => BenchmarkResult {
                    backend: "Metal".to_string(),
                    matrix_size: size,
                    duration_ms: 0.0,
                    gflops: 0.0,
                    memory_bandwidth_gb_s: 0.0,
                    efficiency_percent: 0.0,
                    success: false,
                    error_message: Some(format!("{}", e)),
                },
            }
        }
        Err(e) => BenchmarkResult {
            backend: "Metal".to_string(),
            matrix_size: size,
            duration_ms: 0.0,
            gflops: 0.0,
            memory_bandwidth_gb_s: 0.0,
            efficiency_percent: 0.0,
            success: false,
            error_message: Some(format!("Metal initialization failed: {}", e)),
        },
    }
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
fn benchmark_metal(_size: usize) -> BenchmarkResult {
    BenchmarkResult {
        backend: "Metal".to_string(),
        matrix_size: _size,
        duration_ms: 0.0,
        gflops: 0.0,
        memory_bandwidth_gb_s: 0.0,
        efficiency_percent: 0.0,
        success: false,
        error_message: Some("Metal feature not enabled".to_string()),
    }
}

#[cfg(feature = "cuda")]
fn benchmark_cuda(size: usize) -> BenchmarkResult {
    match CudaMatrixExecutor::new(0) {
        Ok(executor) => {
            let m = size;
            let n = size;
            let k = size;

            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 2) as f32) / (k * n) as f32)
                .collect();
            let mut c = vec![0.0f32; m * n];

            let start = Instant::now();
            match executor.matmul_f32(&a, &b, &mut c, m, n, k, false) {
                Ok(()) => {
                    let duration = start.elapsed();
                    let duration_ms = duration.as_secs_f64() * 1000.0;
                    let flops = 2.0 * (m * n * k) as f64;
                    let gflops = flops / (duration.as_secs_f64() * 1e9);
                    let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0;
                    let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);

                    // Estimate based on GPU architecture
                    let theoretical_gflops = 5000.0; // Conservative estimate
                    let efficiency = (gflops / theoretical_gflops * 100.0).min(100.0);

                    BenchmarkResult {
                        backend: "CUDA".to_string(),
                        matrix_size: size,
                        duration_ms,
                        gflops,
                        memory_bandwidth_gb_s: bandwidth_gb_s,
                        efficiency_percent: efficiency,
                        success: true,
                        error_message: None,
                    }
                }
                Err(e) => BenchmarkResult {
                    backend: "CUDA".to_string(),
                    matrix_size: size,
                    duration_ms: 0.0,
                    gflops: 0.0,
                    memory_bandwidth_gb_s: 0.0,
                    efficiency_percent: 0.0,
                    success: false,
                    error_message: Some(format!("{}", e)),
                },
            }
        }
        Err(e) => BenchmarkResult {
            backend: "CUDA".to_string(),
            matrix_size: size,
            duration_ms: 0.0,
            gflops: 0.0,
            memory_bandwidth_gb_s: 0.0,
            efficiency_percent: 0.0,
            success: false,
            error_message: Some(format!("CUDA initialization failed: {}", e)),
        },
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn benchmark_cuda(_size: usize) -> BenchmarkResult {
    BenchmarkResult {
        backend: "CUDA".to_string(),
        matrix_size: _size,
        duration_ms: 0.0,
        gflops: 0.0,
        memory_bandwidth_gb_s: 0.0,
        efficiency_percent: 0.0,
        success: false,
        error_message: Some("CUDA feature not enabled".to_string()),
    }
}

#[cfg(feature = "opencl")]
fn benchmark_opencl(size: usize) -> BenchmarkResult {
    match OpenClMatrixExecutor::new() {
        Ok(mut executor) => {
            let m = size;
            let n = size;
            let k = size;

            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 2) as f32) / (k * n) as f32)
                .collect();
            let mut c = vec![0.0f32; m * n];

            let start = Instant::now();
            match executor.matmul_f32(&a, &b, &mut c, m, n, k) {
                Ok(()) => {
                    let duration = start.elapsed();
                    let duration_ms = duration.as_secs_f64() * 1000.0;
                    let flops = 2.0 * (m * n * k) as f64;
                    let gflops = flops / (duration.as_secs_f64() * 1e9);
                    let memory_ops = ((m * k) + (k * n) + (m * n)) as f64 * 4.0;
                    let bandwidth_gb_s = memory_ops / (duration.as_secs_f64() * 1e9);

                    // OpenCL efficiency varies by device
                    let theoretical_gflops = 3000.0; // Conservative cross-platform estimate
                    let efficiency = (gflops / theoretical_gflops * 100.0).min(100.0);

                    BenchmarkResult {
                        backend: "OpenCL".to_string(),
                        matrix_size: size,
                        duration_ms,
                        gflops,
                        memory_bandwidth_gb_s: bandwidth_gb_s,
                        efficiency_percent: efficiency,
                        success: true,
                        error_message: None,
                    }
                }
                Err(e) => BenchmarkResult {
                    backend: "OpenCL".to_string(),
                    matrix_size: size,
                    duration_ms: 0.0,
                    gflops: 0.0,
                    memory_bandwidth_gb_s: 0.0,
                    efficiency_percent: 0.0,
                    success: false,
                    error_message: Some(format!("{}", e)),
                },
            }
        }
        Err(e) => BenchmarkResult {
            backend: "OpenCL".to_string(),
            matrix_size: size,
            duration_ms: 0.0,
            gflops: 0.0,
            memory_bandwidth_gb_s: 0.0,
            efficiency_percent: 0.0,
            success: false,
            error_message: Some(format!("OpenCL initialization failed: {}", e)),
        },
    }
}

#[cfg(not(feature = "opencl"))]
#[allow(dead_code)]
fn benchmark_opencl(_size: usize) -> BenchmarkResult {
    BenchmarkResult {
        backend: "OpenCL".to_string(),
        matrix_size: _size,
        duration_ms: 0.0,
        gflops: 0.0,
        memory_bandwidth_gb_s: 0.0,
        efficiency_percent: 0.0,
        success: false,
        error_message: Some("OpenCL feature not enabled".to_string()),
    }
}

fn display_single_result(result: &BenchmarkResult) {
    if result.success {
        println!(
            "✅ {}: {:.3}ms ({:.1} GFLOPS, {:.1} GB/s, {:.1}% eff)",
            result.backend,
            result.duration_ms,
            result.gflops,
            result.memory_bandwidth_gb_s,
            result.efficiency_percent
        );
    } else {
        println!(
            "❌ {}: {}",
            result.backend,
            result
                .error_message
                .as_ref()
                .unwrap_or(&"Unknown error".to_string())
        );
    }
}

fn show_relative_performance(results: &[BenchmarkResult], size: usize) {
    let size_results: Vec<_> = results
        .iter()
        .filter(|r| r.matrix_size == size && r.success)
        .collect();

    if size_results.is_empty() {
        return;
    }

    // Find fastest backend for this size
    let fastest = size_results
        .iter()
        .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap())
        .unwrap();

    println!("\n🏆 Size {} Performance Ranking:", size);
    let mut sorted_results = size_results.clone();
    sorted_results.sort_by(|a, b| b.gflops.partial_cmp(&a.gflops).unwrap());

    for (i, result) in sorted_results.iter().enumerate() {
        let speedup = result.gflops / fastest.gflops;
        let relative = if result.gflops == fastest.gflops {
            "FASTEST".to_string()
        } else {
            format!("{:.2}x faster than baseline", speedup)
        };

        println!(
            "{}. {}: {:.1} GFLOPS ({})",
            i + 1,
            result.backend,
            result.gflops,
            relative
        );
    }
}

fn analyze_and_recommend(results: &[BenchmarkResult], system_info: &SystemInfo) {
    println!();
    println!("{}", "=".repeat(70));
    println!("🎯 Final Analysis and Recommendations");
    println!("最終分析と推奨事項");
    println!("{}", "=".repeat(70));

    // Overall performance summary
    let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();

    if successful_results.is_empty() {
        println!("❌ No successful benchmarks completed");
        return;
    }

    // Find best performing backend across all sizes
    let overall_best = successful_results
        .iter()
        .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap())
        .unwrap();

    println!("\n📊 Overall Performance Leader:");
    println!("Backend: {}", overall_best.backend);
    println!(
        "Peak Performance: {:.1} GFLOPS @ {}x{} matrices",
        overall_best.gflops, overall_best.matrix_size, overall_best.matrix_size
    );
    println!(
        "Efficiency: {:.1}% of theoretical peak",
        overall_best.efficiency_percent
    );

    // Performance analysis by backend
    println!("\n🔍 Backend Analysis:");

    let backends = vec!["OpenBLAS CPU", "Metal", "CUDA", "OpenCL"];
    for backend in &backends {
        let backend_results: Vec<_> = successful_results
            .iter()
            .filter(|r| r.backend == *backend)
            .collect();

        if !backend_results.is_empty() {
            let avg_gflops: f64 = backend_results.iter().map(|r| r.gflops).sum::<f64>()
                / backend_results.len() as f64;
            let max_gflops = backend_results.iter().map(|r| r.gflops).fold(0.0, f64::max);
            let avg_efficiency: f64 = backend_results
                .iter()
                .map(|r| r.efficiency_percent)
                .sum::<f64>()
                / backend_results.len() as f64;

            println!(
                "  {}: Avg {:.1} GFLOPS, Peak {:.1} GFLOPS, Efficiency {:.1}%",
                backend, avg_gflops, max_gflops, avg_efficiency
            );

            // Specific recommendations
            match *backend {
                "Metal" => {
                    if max_gflops > 50.0 {
                        println!(
                            "    ✅ Metal shows excellent performance on AMD Radeon Pro Vega 56"
                        );
                        println!("    💡 Recommended for macOS with AMD GPU acceleration");
                    } else {
                        println!("    ⚠️ Metal performance below expectations - check GPU drivers");
                    }
                }
                "CUDA" => {
                    if !backend_results.is_empty() {
                        println!("    ✅ CUDA available - excellent for NVIDIA GPUs");
                        println!("    💡 Consider Tensor Core acceleration for mixed precision");
                    }
                }
                "OpenCL" => {
                    if max_gflops > 30.0 {
                        println!("    ✅ OpenCL shows good cross-platform compatibility");
                        println!("    💡 Good fallback option for diverse hardware");
                    } else {
                        println!("    ⚠️ OpenCL performance modest - device-specific optimization needed");
                    }
                }
                "OpenBLAS CPU" => {
                    println!("    ✅ Reliable CPU baseline with good multithreading");
                    println!("    💡 Excellent for systems without GPU acceleration");
                }
                _ => {}
            }
        } else {
            println!("  {}: ❌ Not available or failed to initialize", backend);
        }
    }

    // System-specific recommendations
    println!(
        "\n🖥️ System-Specific Recommendations for {}:",
        system_info.os
    );

    match system_info.os.as_str() {
        "macos" => {
            println!(
                "  1. 🥇 Primary: Metal Performance Shaders (optimized for AMD Radeon Pro Vega 56)"
            );
            println!("  2. 🥈 Secondary: OpenCL (cross-platform compatibility)");
            println!(
                "  3. 🥉 Fallback: OpenBLAS CPU ({} cores available)",
                system_info.cpu_cores
            );
            println!("  💡 CUDA not typically available on macOS");
        }
        "linux" => {
            println!("  1. 🥇 Primary: CUDA (if NVIDIA GPU available)");
            println!("  2. 🥈 Secondary: OpenCL (AMD/Intel GPU support)");
            println!(
                "  3. 🥉 Fallback: OpenBLAS CPU ({} cores)",
                system_info.cpu_cores
            );
            println!("  💡 Metal not available on Linux");
        }
        "windows" => {
            println!("  1. 🥇 Primary: CUDA (NVIDIA) or OpenCL (AMD/Intel)");
            println!("  2. 🥈 Secondary: OpenCL for cross-vendor compatibility");
            println!(
                "  3. 🥉 Fallback: OpenBLAS CPU ({} cores)",
                system_info.cpu_cores
            );
            println!("  💡 Metal not available on Windows");
        }
        _ => {
            println!("  Default recommendations: OpenCL → OpenBLAS CPU");
        }
    }

    // Usage guidelines
    println!("\n📋 Usage Guidelines:");
    println!("  • Small matrices (<512): CPU may outperform GPU due to overhead");
    println!("  • Medium matrices (512-1024): GPU acceleration becomes beneficial");
    println!("  • Large matrices (>1024): GPU acceleration essential for performance");
    println!("  • Batch operations: Always prefer GPU when available");
    println!("  • Memory constraints: Monitor GPU memory usage for large workloads");

    // Feature flag recommendations
    println!("\n🚩 Recommended Cargo Features:");

    let available_gpus = &system_info.gpu_available;
    if available_gpus.is_empty() {
        println!("  cargo run --features \"linalg-system\" # CPU-only optimization");
    } else {
        let mut features = vec!["linalg-system"];

        if available_gpus.contains(&"Metal".to_string()) {
            features.push("metal");
        }
        if available_gpus.contains(&"CUDA".to_string()) {
            features.push("cuda");
        }
        if available_gpus.contains(&"OpenCL".to_string()) {
            features.push("opencl");
        }

        println!(
            "  cargo run --features \"{}\" # Full acceleration suite",
            features.join(",")
        );
    }

    println!("\n✨ Benchmark Complete! RusTorchの包括的GPU最適化が完了しました！");
}
