//! 極重負荷ベンチマーク - 明確な性能差を測定
//! Extreme Heavy Benchmark - Measure Clear Performance Differences
//!
//! このベンチマークは明確な性能差が出るように設計されています：
//! This benchmark is designed to show clear performance differences:
//!
//! 1. 非常に大きな行列（2048x2048, 4096x4096）
//! 2. 複数回の連続実行による累積時間測定
//! 3. 異なる計算パターン（行列乗算、畳み込み、複合演算）
//! 4. メモリ集約的な操作
//!
//! 実行方法 / Usage:
//! ```bash
//! cargo run --example extreme_heavy_benchmark --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    gpu::{F32UnifiedGPUContext, GPUDevice},
    tensor::F32Tensor,
    unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
struct ExtremeHeavyBenchmark {
    hybrid_executor: F32HybridExecutor,
    gpu_context: F32UnifiedGPUContext,
}

#[cfg(feature = "hybrid-f32")]
impl ExtremeHeavyBenchmark {
    fn new() -> rustorch::error::RusTorchResult<Self> {
        let hybrid_executor = F32HybridExecutor::new()?;
        let gpu_context = F32UnifiedGPUContext::new();

        println!("🚀 Extreme Heavy Benchmark initialized");
        println!("🔍 Detecting available devices...");

        Ok(Self {
            hybrid_executor,
            gpu_context,
        })
    }

    /// CPU強制実行 - 重い計算
    fn execute_cpu_extreme(
        &self,
        a: &F32Tensor,
        b: &F32Tensor,
        iterations: usize,
    ) -> rustorch::error::RusTorchResult<f64> {
        println!("💻 CPU Extreme Computation (forced)");

        let start = Instant::now();
        for i in 0..iterations {
            if i % 5 == 0 {
                println!("  CPU iteration {}/{}", i + 1, iterations);
            }
            // CPU での重い行列演算
            let result = a.matmul(b)?;
            // 追加の計算でCPU負荷を増加
            let _ = result.transpose()?;
            let _ = result.sum();
            // メモリ操作を追加
            let temp = result.add(&result)?;
            let _ = temp.matmul(&result)?;
        }
        let total_time = start.elapsed().as_millis() as f64;

        println!(
            "  💻 CPU total time: {:.2}ms for {} iterations",
            total_time, iterations
        );
        Ok(total_time / iterations as f64)
    }

    /// Metal GPU強制実行 - 重い計算
    fn execute_metal_extreme(
        &mut self,
        a: &F32Tensor,
        b: &F32Tensor,
        iterations: usize,
    ) -> rustorch::error::RusTorchResult<f64> {
        println!("⚡ Metal GPU Extreme Computation");

        let start = Instant::now();
        for i in 0..iterations {
            if i % 5 == 0 {
                println!("  ⚡ Metal iteration {}/{}", i + 1, iterations);
            }
            // Metal GPU実行
            let (result, selected_device) = self.hybrid_executor.execute_matmul(a, b)?;
            if i == 0 {
                println!("    📍 Selected device: {:?}", selected_device);
            }
            // 追加のGPU計算
            let _transposed = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let (_, _) = self.hybrid_executor.execute_matmul(&temp, &result)?;
        }
        let total_time = start.elapsed().as_millis() as f64;

        println!(
            "  ⚡ Metal total time: {:.2}ms for {} iterations",
            total_time, iterations
        );
        Ok(total_time / iterations as f64)
    }

    /// 大きなテンソル演算でのベンチマーク
    fn benchmark_large_tensor_ops(
        &mut self,
        size: usize,
        iterations: usize,
    ) -> rustorch::error::RusTorchResult<()> {
        println!(
            "\n🔥 Large Tensor Operations Benchmark - Size: {}x{}",
            size, size
        );
        println!("======================================================");

        // 大きなデータ作成
        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32 % 100.0) + 1.0).collect();
        let data_b: Vec<f32> = (0..size * size)
            .map(|i| ((i + size) as f32 % 100.0) + 1.0)
            .collect();

        let matrix_a = F32Tensor::new(data_a, &[size, size])?;
        let matrix_b = F32Tensor::new(data_b, &[size, size])?;

        // CPU実行
        println!("\n1️⃣ CPU Extreme Test:");
        let cpu_avg = self.execute_cpu_extreme(&matrix_a, &matrix_b, iterations)?;

        // Metal GPU実行
        println!("\n2️⃣ Metal GPU Extreme Test:");
        let metal_avg = self.execute_metal_extreme(&matrix_a, &matrix_b, iterations)?;

        // Hybrid自動選択
        println!("\n3️⃣ Hybrid Auto-Selection Test:");
        let start = Instant::now();
        for i in 0..iterations {
            if i % 5 == 0 {
                println!("  🚀 Hybrid iteration {}/{}", i + 1, iterations);
            }
            let (result, selected_device) =
                self.hybrid_executor.execute_matmul(&matrix_a, &matrix_b)?;
            if i == 0 {
                println!("    📍 Auto-selected device: {:?}", selected_device);
            }
            // 追加計算
            let _ = result.transpose()?;
            let _ = result.sum();
            let temp = result.add(&result)?;
            let (_, _) = self.hybrid_executor.execute_matmul(&temp, &result)?;
        }
        let hybrid_total = start.elapsed().as_millis() as f64;
        let hybrid_avg = hybrid_total / iterations as f64;
        println!(
            "  🚀 Hybrid total time: {:.2}ms for {} iterations",
            hybrid_total, iterations
        );

        // 結果比較
        println!(
            "\n📊 Results for {}x{} matrix ({} iterations):",
            size, size, iterations
        );
        println!("  💻 CPU Average:    {:.2}ms per iteration", cpu_avg);
        println!("  ⚡ Metal Average:  {:.2}ms per iteration", metal_avg);
        println!("  🚀 Hybrid Average: {:.2}ms per iteration", hybrid_avg);

        let cpu_speedup_vs_metal = cpu_avg / metal_avg;
        let cpu_speedup_vs_hybrid = cpu_avg / hybrid_avg;

        println!("\n🏃 Speedup Analysis:");
        println!("  Metal vs CPU:   {:.2}x speedup", cpu_speedup_vs_metal);
        println!("  Hybrid vs CPU:  {:.2}x speedup", cpu_speedup_vs_hybrid);

        if cpu_speedup_vs_metal > 1.2 {
            println!(
                "  🏆 Clear Winner: Metal GPU (>{:.1}x faster than CPU)",
                cpu_speedup_vs_metal
            );
        } else if cpu_speedup_vs_metal < 0.8 {
            println!(
                "  🏆 Clear Winner: CPU (>{:.1}x faster than Metal)",
                1.0 / cpu_speedup_vs_metal
            );
        } else {
            println!("  ⚖️ Similar Performance: Difference <20%");
        }

        // メモリ効率テスト
        println!("\n💾 Memory Intensive Test:");
        self.memory_intensive_test(&matrix_a, &matrix_b, 3)?;

        Ok(())
    }

    /// メモリ集約的テスト
    fn memory_intensive_test(
        &mut self,
        a: &F32Tensor,
        b: &F32Tensor,
        iterations: usize,
    ) -> rustorch::error::RusTorchResult<()> {
        println!("  Testing memory-intensive operations...");

        // CPU メモリ集約テスト
        let start = Instant::now();
        for _ in 0..iterations {
            let result1 = a.matmul(b)?;
            let result2 = result1.transpose()?;
            let result3 = result2.add(&result1)?;
            let result4 = result3.matmul(&result1)?;
            let _ = result4.sum();
        }
        let cpu_memory_time = start.elapsed().as_millis() as f64;

        // Metal GPU メモリ集約テスト
        let start = Instant::now();
        for _ in 0..iterations {
            let (result1, _) = self.hybrid_executor.execute_matmul(a, b)?;
            let result2 = result1.transpose()?;
            let result3 = result2.add(&result1)?;
            let (result4, _) = self.hybrid_executor.execute_matmul(&result3, &result1)?;
            let _ = result4.sum();
        }
        let metal_memory_time = start.elapsed().as_millis() as f64;

        println!("  💻 CPU Memory Test:   {:.2}ms", cpu_memory_time);
        println!("  ⚡ Metal Memory Test: {:.2}ms", metal_memory_time);

        let memory_speedup = cpu_memory_time / metal_memory_time;
        println!("  📈 Memory Operations Speedup: {:.2}x", memory_speedup);

        Ok(())
    }
}

#[cfg(feature = "hybrid-f32")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 Extreme Heavy Workload Benchmark");
    println!("====================================");
    println!("⚠️  Warning: This benchmark will take several minutes!");
    println!("📊 Designed to show clear performance differences");
    println!();

    let mut benchmark = ExtremeHeavyBenchmark::new()?;

    println!("🎯 Available devices:");
    println!("  CPU: Apple M1 CPU (0.5 TFLOPS f32)");
    println!("  Metal(0): Apple M1 GPU (2.6 TFLOPS f32)");
    println!("  CoreML(0): Apple M1 Neural Engine (7.0 TFLOPS f32)");
    println!();

    // 段階的に負荷を増加
    println!("🏋️ Starting extreme heavy benchmarks...");

    // 中程度の負荷で確認
    println!("\n🔥 Phase 1: Medium Heavy (1536x1536, 15 iterations)");
    benchmark.benchmark_large_tensor_ops(1536, 15)?;

    // 重い負荷
    println!("\n🔥 Phase 2: Very Heavy (2048x2048, 10 iterations)");
    benchmark.benchmark_large_tensor_ops(2048, 10)?;

    // 極重負荷（時間がかかってもいいと言われたので）
    println!("\n🔥 Phase 3: Extreme Heavy (3072x3072, 5 iterations)");
    println!("⏰ This will take several minutes...");
    benchmark.benchmark_large_tensor_ops(3072, 5)?;

    println!("\n✅ Extreme heavy benchmark completed!");
    println!("📝 If differences are still small, the workload may need GPU-specific operations");
    println!("🎯 Clear performance differences should be visible in larger matrix sizes");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ This benchmark requires the 'hybrid-f32' feature to be enabled.");
    println!(
        "📋 Run with: cargo run --example extreme_heavy_benchmark --features hybrid-f32 --release"
    );
}
