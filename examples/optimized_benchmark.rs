//! Optimized Performance Benchmark with BLAS Integration
//! BLAS統合による最適化パフォーマンスベンチマーク

#[cfg(not(target_arch = "wasm32"))]
use rustorch::linalg::{benchmark_matmul_implementations, multithreaded_matmul};
use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Optimized Performance Benchmark");
    println!("============================================");

    #[cfg(not(target_arch = "wasm32"))]
    println!("AMD Radeon Pro Vega 56 + OpenBLAS Environment\n");

    #[cfg(target_arch = "wasm32")]
    println!("WASM CPU Environment\n");

    // Test small matrices first
    println!("📊 Small Matrix Tests (64x64):");
    test_small_matrix_performance()?;

    println!("\n📊 Medium Matrix Tests (128x128):");
    test_medium_matrix_performance()?;

    println!("\n📊 Large Matrix Tests (256x256):");
    test_large_matrix_performance()?;

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Run comprehensive benchmark
        println!("\n🔬 Comprehensive Benchmark Suite:");
        run_comprehensive_benchmark()?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("\n🔬 WASM CPU Benchmark Suite:");
        wasm_basic_benchmark()?;
    }

    println!("\n🎯 Performance Summary:");
    println!("- Standard CPU: Baseline performance");
    println!("- Multi-threaded: CPU parallelization gains");
    #[cfg(feature = "blas-optimized")]
    println!("- BLAS-optimized: Maximum CPU performance with OpenBLAS");
    #[cfg(not(feature = "blas-optimized"))]
    println!("- BLAS-optimized: Not enabled (use --features blas-optimized)");

    println!("\n💡 Next Steps:");
    println!("- Enable GPU acceleration with --features metal");
    println!("- Test with larger matrices for GPU benefits");
    println!("- Compare against PyTorch performance");

    Ok(())
}

fn test_small_matrix_performance() -> Result<(), Box<dyn std::error::Error>> {
    let size = 64;
    create_and_benchmark_matrices(size)?;
    Ok(())
}

fn test_medium_matrix_performance() -> Result<(), Box<dyn std::error::Error>> {
    let size = 128;
    create_and_benchmark_matrices(size)?;
    Ok(())
}

fn test_large_matrix_performance() -> Result<(), Box<dyn std::error::Error>> {
    let size = 256;
    create_and_benchmark_matrices(size)?;
    Ok(())
}

fn create_and_benchmark_matrices(size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Create test data
    let data_a: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
    let data_b: Vec<f32> = (0..(size * size))
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();

    let matrix_a = Tensor::<f32>::from_vec(data_a, vec![size, size]);
    let matrix_b = Tensor::<f32>::from_vec(data_b, vec![size, size]);

    // Calculate FLOPS
    let flops = 2.0 * (size as f64).powi(3);

    // Benchmark standard implementation
    let start = Instant::now();
    let _std_result = matrix_a
        .matmul(&matrix_b)
        .map_err(|e| format!("Standard matmul failed: {}", e))?;
    let std_time = start.elapsed();
    let std_gflops = flops / (std_time.as_secs_f64() * 1e9);

    #[cfg(not(target_arch = "wasm32"))]
    let (mt_time, mt_gflops) = {
        // Benchmark multi-threaded implementation
        let start = Instant::now();
        let _mt_result = multithreaded_matmul(&matrix_a, &matrix_b)?;
        let mt_time = start.elapsed();
        let mt_gflops = flops / (mt_time.as_secs_f64() * 1e9);
        (mt_time, mt_gflops)
    };

    #[cfg(target_arch = "wasm32")]
    let (mt_time, mt_gflops) = (std_time, std_gflops); // WASM doesn't support multithreading

    println!(
        "  Standard:      {:.2}ms ({:.2} GFLOPS)",
        std_time.as_secs_f64() * 1000.0,
        std_gflops
    );
    println!(
        "  Multi-thread:  {:.2}ms ({:.2} GFLOPS) - {:.2}x speedup",
        mt_time.as_secs_f64() * 1000.0,
        mt_gflops,
        std_time.as_secs_f64() / mt_time.as_secs_f64()
    );

    // Benchmark BLAS if available
    #[cfg(feature = "blas-optimized")]
    {
        let start = Instant::now();
        let _blas_result = optimized_matmul(&matrix_a, &matrix_b)?;
        let blas_time = start.elapsed();
        let blas_gflops = flops / (blas_time.as_secs_f64() * 1e9);

        println!(
            "  BLAS-optimized: {:.2}ms ({:.2} GFLOPS) - {:.2}x speedup",
            blas_time.as_secs_f64() * 1000.0,
            blas_gflops,
            std_time.as_secs_f64() / blas_time.as_secs_f64()
        );
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_comprehensive_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![32, 64, 128];

    for size in sizes {
        println!("\n  Testing {}x{} matrices:", size, size);
        benchmark_matmul_implementations::<f32>(size)?;
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn wasm_basic_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![32, 64, 128];

    for size in sizes {
        println!("\n  Testing {}x{} matrices (WASM CPU):", size, size);

        let data_a: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
        let data_b: Vec<f32> = (0..(size * size))
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();

        let matrix_a = Tensor::<f32>::from_vec(data_a, vec![size, size]);
        let matrix_b = Tensor::<f32>::from_vec(data_b, vec![size, size]);

        // Calculate FLOPS
        let flops = 2.0 * (size as f64).powi(3);

        let start = Instant::now();
        let _result = matrix_a
            .matmul(&matrix_b)
            .map_err(|e| format!("WASM matmul failed: {}", e))?;
        let time = start.elapsed();
        let gflops = flops / (time.as_secs_f64() * 1e9);

        println!(
            "    WASM CPU: {:.2}ms ({:.2} GFLOPS)",
            time.as_secs_f64() * 1000.0,
            gflops
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_benchmark() {
        let result = create_and_benchmark_matrices(32);
        assert!(result.is_ok());
    }
}
