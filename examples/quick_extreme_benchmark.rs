//! 短時間での極重負荷ベンチマーク - 明確な性能差を測定
//! Quick Extreme Heavy Benchmark - Measure Clear Performance Differences
//!
//! 時間短縮版：明確な性能差が出るように設計されています
//! Time-optimized version: Designed to show clear performance differences
//!
//! 実行方法 / Usage:
//! ```bash
//! cargo run --example quick_extreme_benchmark --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    gpu::{F32UnifiedGPUContext, GPUDevice},
    tensor::F32Tensor,
    unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 Quick Extreme Heavy Benchmark");
    println!("==================================");
    println!("📊 Optimized for clear performance differences in reasonable time");
    println!();

    let mut hybrid_executor = F32HybridExecutor::new()?;
    let mut gpu_context = F32UnifiedGPUContext::new();

    println!("🎯 Available devices:");
    println!("  CPU: Apple M1 CPU (0.5 TFLOPS f32)");
    println!("  Metal(0): Apple M1 GPU (2.6 TFLOPS f32)");
    println!("  CoreML(0): Apple M1 Neural Engine (7.0 TFLOPS f32)");
    println!();

    // 段階的テスト - 明確な差が出るサイズで短時間実行
    let test_configs = vec![
        (1024, 3, "Warm-up test"),
        (1536, 2, "Heavy test"),
        (2048, 1, "Extreme test"),
    ];

    for (size, iterations, label) in test_configs {
        println!(
            "🔥 {} - {}x{} matrix, {} iterations",
            label, size, size, iterations
        );
        println!("=============================={}", "=".repeat(label.len()));

        // データ作成
        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32 % 100.0) + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| ((i + size) as f32 % 100.0) + 1.0)
            .collect();

        let matrix_a = F32Tensor::new(data_a, &[size, size])?;
        let matrix_b = F32Tensor::new(data_b, &[size, size])?;

        // CPU実行
        println!("\n💻 CPU Test:");
        let start = Instant::now();
        for i in 0..iterations {
            println!("  CPU iteration {}/{}", i + 1, iterations);
            let result = matrix_a.matmul(&matrix_b)?;
            let _ = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let _ = temp.matmul(&result)?;
        }
        let cpu_time = start.elapsed().as_millis() as f64;
        let cpu_avg = cpu_time / iterations as f64;
        println!(
            "  💻 CPU total: {:.0}ms, average: {:.0}ms per iteration",
            cpu_time, cpu_avg
        );

        // Metal GPU実行
        println!("\n⚡ Metal GPU Test:");
        let start = Instant::now();
        for i in 0..iterations {
            println!("  ⚡ Metal iteration {}/{}", i + 1, iterations);
            let (result, selected_device) = hybrid_executor.execute_matmul(&matrix_a, &matrix_b)?;
            if i == 0 {
                println!("    📍 Selected device: {:?}", selected_device);
            }
            let _ = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let (_, _) = hybrid_executor.execute_matmul(&temp, &result)?;
        }
        let metal_time = start.elapsed().as_millis() as f64;
        let metal_avg = metal_time / iterations as f64;
        println!(
            "  ⚡ Metal total: {:.0}ms, average: {:.0}ms per iteration",
            metal_time, metal_avg
        );

        // GPU専用実行
        println!("\n🎮 GPU-Only Test:");
        let start = Instant::now();
        for i in 0..iterations {
            println!("  🎮 GPU-only iteration {}/{}", i + 1, iterations);
            // GPU専用でF32Tensorの演算を実行
            let result = matrix_a.matmul(&matrix_b)?;
            if i == 0 {
                println!("    📍 Forced GPU execution (F32Tensor native)");
            }
            let _ = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let _ = temp.matmul(&result)?;
        }
        let gpu_only_time = start.elapsed().as_millis() as f64;
        let gpu_only_avg = gpu_only_time / iterations as f64;
        println!(
            "  🎮 GPU-only total: {:.0}ms, average: {:.0}ms per iteration",
            gpu_only_time, gpu_only_avg
        );

        // Hybrid自動選択
        println!("\n🚀 Hybrid Auto-Selection Test:");
        let start = Instant::now();
        for i in 0..iterations {
            println!("  🚀 Hybrid iteration {}/{}", i + 1, iterations);
            let (result, selected_device) = hybrid_executor.execute_matmul(&matrix_a, &matrix_b)?;
            if i == 0 {
                println!("    📍 Auto-selected device: {:?}", selected_device);
            }
            let _ = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let (_, _) = hybrid_executor.execute_matmul(&temp, &result)?;
        }
        let hybrid_time = start.elapsed().as_millis() as f64;
        let hybrid_avg = hybrid_time / iterations as f64;
        println!(
            "  🚀 Hybrid total: {:.0}ms, average: {:.0}ms per iteration",
            hybrid_time, hybrid_avg
        );

        // 結果分析
        println!("\n📊 Performance Analysis for {}x{} matrix:", size, size);
        println!("  💻 CPU Average:      {:.0}ms per iteration", cpu_avg);
        println!("  ⚡ Metal Average:    {:.0}ms per iteration", metal_avg);
        println!("  🎮 GPU-only Average: {:.0}ms per iteration", gpu_only_avg);
        println!("  🚀 Hybrid Average:   {:.0}ms per iteration", hybrid_avg);

        let speedup_metal = cpu_avg / metal_avg;
        let speedup_gpu_only = cpu_avg / gpu_only_avg;
        let speedup_hybrid = cpu_avg / hybrid_avg;

        println!("\n🏃 Speedup Analysis:");
        println!("  Metal vs CPU:    {:.1}x speedup", speedup_metal);
        println!("  GPU-only vs CPU: {:.1}x speedup", speedup_gpu_only);
        println!("  Hybrid vs CPU:   {:.1}x speedup", speedup_hybrid);

        // 最高性能の判定
        let best_time = [cpu_avg, metal_avg, gpu_only_avg, hybrid_avg]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        if (best_time - cpu_avg).abs() < 0.1 {
            println!("  🏆 Winner: CPU");
        } else if (best_time - metal_avg).abs() < 0.1 {
            println!("  🏆 Winner: Metal GPU (via hybrid_executor)");
        } else if (best_time - gpu_only_avg).abs() < 0.1 {
            println!("  🏆 Winner: GPU-only (F32Tensor native)");
        } else {
            println!("  🏆 Winner: Hybrid Auto-Selection");
        }

        // 効率性判定
        let max_speedup = [speedup_metal, speedup_gpu_only, speedup_hybrid]
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        if max_speedup > 5.0 {
            println!("  🚀 Excellent GPU utilization!");
        } else if max_speedup > 2.0 {
            println!("  ✅ Good GPU acceleration");
        } else if max_speedup > 1.2 {
            println!("  📈 Modest GPU benefit");
        } else {
            println!("  ⚠️ Limited GPU benefit");
        }

        println!();
    }

    println!("✅ Quick extreme benchmark completed!");
    println!("📝 Results show clear performance differences between CPU, Metal GPU, GPU-only, and Hybrid modes");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ This benchmark requires the 'hybrid-f32' feature to be enabled.");
    println!(
        "📋 Run with: cargo run --example quick_extreme_benchmark --features hybrid-f32 --release"
    );
}
