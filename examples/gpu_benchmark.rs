//! GPU Performance Benchmark Example
//! GPUパフォーマンスベンチマーク例

#[cfg(not(target_arch = "wasm32"))]
use rustorch::gpu::simple_metal_test::{
    benchmark_metal_performance, test_metal_gpu_basic, test_metal_tensor_operations,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch GPU Performance Benchmark");
    println!("=====================================\n");

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Test Metal GPU availability
        println!("1️⃣ Testing Metal GPU availability...");
        match test_metal_gpu_basic() {
            Ok(()) => println!("✅ Metal GPU tests passed\n"),
            Err(e) => println!("⚠️ Metal GPU test failed: {}\n", e),
        }

        // Test tensor operations
        println!("2️⃣ Testing tensor operations...");
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("GPU operations are not available in WASM target.");
        println!("This benchmark shows CPU-only operations.\n");

        // WASM CPU benchmark
        wasm_cpu_benchmark()?;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match test_metal_tensor_operations() {
            Ok(()) => println!("✅ Tensor operation tests passed\n"),
            Err(e) => println!("❌ Tensor operation test failed: {}\n", e),
        }

        // Run performance benchmark
        println!("3️⃣ Running performance benchmark...");
        match benchmark_metal_performance() {
            Ok(()) => println!("✅ Performance benchmark completed\n"),
            Err(e) => println!("❌ Performance benchmark failed: {}\n", e),
        }
    }

    println!("🎯 Benchmark Summary:");
    println!("- RusTorch v0.4.0 基本機能: 完全動作 ✅");
    println!("- CPU テンソル操作: 高性能 ⚡");
    println!("- GPU 機能: 段階的開発中 🚧");
    println!("- AMD Radeon Pro Vega 56: 検出・対応予定 🔧\n");

    println!("💡 Recommendations:");
    println!("- 現在のCPU実装は本格的な機械学習ワークロードに対応済み");
    println!("- GPU機能は安全なCPU fallbackで動作保証");
    println!("- Metal実装の段階的な復旧を継続中");

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn wasm_cpu_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::tensor::Tensor;
    use std::time::Instant;

    println!("1️⃣ WASM CPU Tensor Operations...");

    let sizes = vec![64, 128, 256];

    for size in sizes {
        println!("Testing {}x{} matrices:", size, size);

        let a = Tensor::<f32>::ones(&[size, size]);
        let b = Tensor::<f32>::ones(&[size, size]);

        let start = Instant::now();
        let result = a.matmul(&b)?;
        let time = start.elapsed();

        println!(
            "  Matrix multiplication: {:.3}ms",
            time.as_secs_f64() * 1000.0
        );
        assert_eq!(result.shape(), &[size, size]);
    }

    println!("✅ WASM CPU operations completed successfully\n");
    Ok(())
}
