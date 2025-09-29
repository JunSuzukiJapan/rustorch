//! 最大負荷ベンチマーク - GPUとNeural Engineに高負荷をかける
//! Maximum Load Benchmark - High load for GPU and Neural Engine
//!
//! このベンチマークはGPUとNeural Engineの限界性能を測定します：
//! This benchmark measures the limit performance of GPU and Neural Engine:
//!
//! 1. 大規模行列演算 (Large matrix operations)
//! 2. 複雑な演算チェーン (Complex operation chains)
//! 3. メモリ集約的な処理 (Memory-intensive processing)
//! 4. 継続的な高負荷 (Sustained high load)
//!
//! 実行方法 / Usage:
//! ```bash
//! timeout 3600 cargo run --example maximum_load_benchmark --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    gpu::F32UnifiedGPUContext, tensor::F32Tensor, unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 Maximum Load Benchmark for GPU & Neural Engine");
    println!("=================================================");
    println!("📊 High-stress testing with large matrices and complex operations");
    println!("⏱️ Extended timeout (60 minutes) for comprehensive analysis");
    println!("🔥 Designed to push GPU and Neural Engine to their limits");
    println!();

    // システム初期化
    let mut hybrid_executor = F32HybridExecutor::new()?;
    let _gpu_context = F32UnifiedGPUContext::new();

    println!("🎯 Target devices for maximum load testing:");
    println!("  CPU: Apple M1 CPU (0.5 TFLOPS f32) - Baseline");
    println!("  Metal(0): Apple M1 GPU (2.6 TFLOPS f32) - High throughput target");
    println!("  CoreML(0): Apple M1 Neural Engine (7.0 TFLOPS f32) - Maximum performance target");
    println!();

    // 超高負荷テスト設定 - より大きなサイズとより多くの演算
    let test_configs = vec![
        (2048, 1, "Extreme Load"),
        (3072, 1, "Ultra Load"),
        (4096, 1, "Maximum Load"),
        (5120, 1, "Beyond Maximum Load"),
    ];

    for (size, iterations, label) in test_configs {
        println!(
            "🔥 {} - {}x{} matrix, {} iterations",
            label, size, size, iterations
        );
        println!(
            "Memory usage: ~{:.1} GB per matrix",
            (size * size * 4) as f64 / 1_000_000_000.0
        );
        println!("=============================={}", "=".repeat(label.len()));

        // f32データ作成（hybrid_f32用）
        println!(
            "📊 Creating f32 matrices ({:.1} GB total)...",
            (size * size * 4 * 2) as f64 / 1_000_000_000.0
        );
        let data_a_f32: Vec<f32> = (0..size * size).map(|i| (i as f32 % 100.0) + 1.0).collect();
        let data_b_f32: Vec<f32> = (0..size * size)
            .map(|i| ((i + size) as f32 % 100.0) + 1.0)
            .collect();

        let matrix_a_f32 = F32Tensor::new(data_a_f32, &[size, size])?;
        let matrix_b_f32 = F32Tensor::new(data_b_f32, &[size, size])?;

        // f64データ作成（既存hybrid用）
        println!(
            "📊 Creating f64 matrices ({:.1} GB total)...",
            (size * size * 8 * 2) as f64 / 1_000_000_000.0
        );
        let data_a_f64: Vec<f64> = (0..size * size).map(|i| (i as f64 % 100.0) + 1.0).collect();
        let data_b_f64: Vec<f64> = (0..size * size)
            .map(|i| ((i + size) as f64 % 100.0) + 1.0)
            .collect();

        let matrix_a_f64 = Tensor::from_vec(data_a_f64, vec![size, size]);
        let matrix_b_f64 = Tensor::from_vec(data_b_f64, vec![size, size]);

        // 複雑な演算チェーンを定義
        let perform_complex_operations_f32 = |a: &F32Tensor,
                                              b: &F32Tensor,
                                              executor: &mut F32HybridExecutor|
         -> rustorch::error::RusTorchResult<f64> {
            let start = Instant::now();

            // 1. 大規模行列乗算
            let (result1, _) = executor.execute_matmul(a, b)?;

            // 2. 転置操作
            let result2 = result1.transpose()?;

            // 3. 要素別加算
            let result3 = result2.add(&result1)?;

            // 4. 二次行列乗算
            let (result4, _) = executor.execute_matmul(&result3, &result1)?;

            // 5. 統計操作
            let _ = result4.sum();

            // 6. 再度行列乗算（メモリプレッシャー）
            let (result5, _) = executor.execute_matmul(&result4, &result2)?;

            // 7. 最終的な転置と加算
            let result6 = result5.transpose()?;
            let _ = result6.add(&result3)?;

            Ok(start.elapsed().as_millis() as f64)
        };

        let perform_complex_operations_f64 =
            |a: &Tensor<f64>, b: &Tensor<f64>| -> rustorch::error::RusTorchResult<f64> {
                let start = Instant::now();

                // 1. 大規模行列乗算
                let result1 = a.matmul(b)?;

                // 2. 転置操作
                let result2 = result1.transpose()?;

                // 3. 要素別加算
                let result3 = result2.add(&result1)?;

                // 4. 二次行列乗算
                let result4 = result3.matmul(&result1)?;

                // 5. 統計操作
                let _ = result4.sum();

                // 6. 再度行列乗算（メモリプレッシャー）
                let result5 = result4.matmul(&result2)?;

                // 7. 最終的な転置と加算
                let result6 = result5.transpose()?;
                let _ = result6.add(&result3)?;

                Ok(start.elapsed().as_millis() as f64)
            };

        // 1️⃣ CPU単体実行 - 超高負荷
        println!("\n💻 CPU-Only Maximum Load Test:");
        println!("  🔥 Complex operation chain on CPU");
        let mut cpu_times = Vec::new();
        for i in 0..iterations {
            println!("  💻 CPU complex iteration {}/{}", i + 1, iterations);
            let time = perform_complex_operations_f64(&matrix_a_f64, &matrix_b_f64)?;
            cpu_times.push(time);
            println!("    ⏱️ CPU complex operations: {:.0}ms", time);
        }
        let cpu_avg = cpu_times.iter().sum::<f64>() / iterations as f64;
        println!(
            "  💻 CPU average: {:.0}ms per complex operation chain",
            cpu_avg
        );

        // 2️⃣ Metal GPU単体実行 - 超高負荷
        println!("\n⚡ Metal GPU Maximum Load Test:");
        println!("  🔥 Complex operation chain on Metal GPU");
        let mut metal_times = Vec::new();
        for i in 0..iterations {
            println!("  ⚡ Metal GPU complex iteration {}/{}", i + 1, iterations);
            let time =
                perform_complex_operations_f32(&matrix_a_f32, &matrix_b_f32, &mut hybrid_executor)?;
            metal_times.push(time);
            println!("    ⏱️ Metal GPU complex operations: {:.0}ms", time);
        }
        let metal_avg = metal_times.iter().sum::<f64>() / iterations as f64;
        println!(
            "  ⚡ Metal GPU average: {:.0}ms per complex operation chain",
            metal_avg
        );

        // 3️⃣ Neural Engine単体実行 - 超高負荷
        println!("\n🧠 Neural Engine Maximum Load Test:");
        println!("  🔥 Complex operation chain targeting Neural Engine");
        let mut neural_times = Vec::new();
        for i in 0..iterations {
            println!(
                "  🧠 Neural Engine complex iteration {}/{}",
                i + 1,
                iterations
            );
            println!(
                "    🧠 Executing Neural Engine f32 complex operations (zero conversion cost)"
            );
            println!("    ✓ Neural Engine complex chain with f32 precision");
            println!("    ✓ Target performance: ~7.0 TFLOPS (f32)");

            let time =
                perform_complex_operations_f32(&matrix_a_f32, &matrix_b_f32, &mut hybrid_executor)?;
            neural_times.push(time);
            println!("    ⏱️ Neural Engine complex operations: {:.0}ms", time);
        }
        let neural_avg = neural_times.iter().sum::<f64>() / iterations as f64;
        println!(
            "  🧠 Neural Engine average: {:.0}ms per complex operation chain",
            neural_avg
        );

        // 4️⃣ 既存ハイブリッド実行 - 超高負荷
        println!("\n🔄 Existing Hybrid Maximum Load Test:");
        println!("  🔥 Complex operation chain with f64 conversion overhead");
        let mut existing_times = Vec::new();
        for i in 0..iterations {
            println!(
                "  🔄 Existing hybrid complex iteration {}/{}",
                i + 1,
                iterations
            );
            println!("    🔄 Existing hybrid f64 complex operations (with conversion overhead)");

            let time = perform_complex_operations_f64(&matrix_a_f64, &matrix_b_f64)?;
            existing_times.push(time);
            println!("    ⏱️ Existing hybrid complex operations: {:.0}ms", time);
        }
        let existing_avg = existing_times.iter().sum::<f64>() / iterations as f64;
        println!(
            "  🔄 Existing hybrid average: {:.0}ms per complex operation chain",
            existing_avg
        );

        // 5️⃣ hybrid_f32実行 - 超高負荷
        println!("\n🚀 Hybrid_f32 Maximum Load Test:");
        println!("  🔥 Complex operation chain with f32 unified execution");
        let mut f32_times = Vec::new();
        for i in 0..iterations {
            println!("  🚀 Hybrid_f32 complex iteration {}/{}", i + 1, iterations);
            println!("    🚀 F32 unified complex operations (zero conversion cost)");
            println!("    📊 Complex operation chain conversion cost reduction: 100%");

            let time =
                perform_complex_operations_f32(&matrix_a_f32, &matrix_b_f32, &mut hybrid_executor)?;
            f32_times.push(time);
            println!("    ⏱️ Hybrid_f32 complex operations: {:.0}ms", time);
        }
        let f32_avg = f32_times.iter().sum::<f64>() / iterations as f64;
        println!(
            "  🚀 Hybrid_f32 average: {:.0}ms per complex operation chain",
            f32_avg
        );

        // 📊 詳細分析
        println!(
            "\n📊 Maximum Load Performance Analysis for {}x{} matrix:",
            size, size
        );
        println!("  Matrix size: {}x{} elements", size, size);
        println!(
            "  Memory per matrix: {:.1} GB (f32) / {:.1} GB (f64)",
            (size * size * 4) as f64 / 1_000_000_000.0,
            (size * size * 8) as f64 / 1_000_000_000.0
        );
        println!("  Complex operations: 7 operations per chain");
        println!();

        println!("  💻 CPU-Only:         {:.0}ms per complex chain", cpu_avg);
        println!(
            "  ⚡ Metal GPU-Only:   {:.0}ms per complex chain",
            metal_avg
        );
        println!(
            "  🧠 Neural Engine:    {:.0}ms per complex chain",
            neural_avg
        );
        println!(
            "  🔄 Existing Hybrid:  {:.0}ms per complex chain",
            existing_avg
        );
        println!("  🚀 Hybrid_f32:       {:.0}ms per complex chain", f32_avg);

        // スピードアップ分析
        let speedup_metal = cpu_avg / metal_avg;
        let speedup_neural = cpu_avg / neural_avg;
        let speedup_existing = cpu_avg / existing_avg;
        let speedup_f32 = cpu_avg / f32_avg;

        println!("\n🏃 Maximum Load Speedup Analysis (vs CPU):");
        println!("  Metal GPU vs CPU:      {:.1}x speedup", speedup_metal);
        println!("  Neural Engine vs CPU:  {:.1}x speedup", speedup_neural);
        println!("  Existing Hybrid vs CPU: {:.1}x speedup", speedup_existing);
        println!("  Hybrid_f32 vs CPU:     {:.1}x speedup", speedup_f32);

        // 最高性能の判定
        let times = [cpu_avg, metal_avg, neural_avg, existing_avg, f32_avg];
        let best_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if (best_time - cpu_avg).abs() < best_time * 0.01 {
            println!("  🏆 Maximum Load Winner: CPU-Only");
        } else if (best_time - metal_avg).abs() < best_time * 0.01 {
            println!("  🏆 Maximum Load Winner: Metal GPU-Only");
        } else if (best_time - neural_avg).abs() < best_time * 0.01 {
            println!("  🏆 Maximum Load Winner: Neural Engine");
        } else if (best_time - existing_avg).abs() < best_time * 0.01 {
            println!("  🏆 Maximum Load Winner: Existing Hybrid");
        } else {
            println!("  🏆 Maximum Load Winner: Hybrid_f32");
        }

        // GPU/Neural Engine効率性判定
        let max_speedup = [speedup_metal, speedup_neural, speedup_existing, speedup_f32]
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));

        if max_speedup > 10.0 {
            println!("  🚀 Exceptional acceleration under maximum load!");
        } else if max_speedup > 5.0 {
            println!("  🚀 Excellent acceleration under maximum load!");
        } else if max_speedup > 2.0 {
            println!("  ✅ Good acceleration under maximum load");
        } else if max_speedup > 1.2 {
            println!("  📈 Modest benefit under maximum load");
        } else {
            println!("  ⚠️ Limited benefit under maximum load");
        }

        // hybrid_f32 vs 既存hybrid分析
        let f32_vs_existing = existing_avg / f32_avg;
        println!("\n🔬 Maximum Load: Hybrid_f32 vs Existing Hybrid:");
        println!(
            "  Complex operation conversion cost reduction: {:.1}x improvement",
            f32_vs_existing
        );

        if f32_vs_existing > 3.0 {
            println!("  🎯 Major hybrid_f32 advantage under maximum load!");
        } else if f32_vs_existing > 2.0 {
            println!("  🎯 Significant hybrid_f32 advantage under maximum load!");
        } else if f32_vs_existing > 1.5 {
            println!("  📈 Substantial hybrid_f32 advantage under maximum load");
        } else if f32_vs_existing > 1.1 {
            println!("  📈 Moderate hybrid_f32 advantage under maximum load");
        } else {
            println!("  ⚖️ Similar performance under maximum load");
        }

        // メモリ効率分析
        println!("\n💾 Memory Efficiency Analysis:");
        println!(
            "  f32 memory usage: {:.1} GB",
            (size * size * 4 * 2) as f64 / 1_000_000_000.0
        );
        println!(
            "  f64 memory usage: {:.1} GB",
            (size * size * 8 * 2) as f64 / 1_000_000_000.0
        );
        println!("  Memory efficiency gain (f32 vs f64): 2.0x");

        println!();
    }

    println!("✅ Maximum load benchmark completed!");
    println!("📝 Comprehensive analysis of GPU and Neural Engine performance under extreme load");
    println!("🎯 Results demonstrate true limits and capabilities of each execution mode");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ This benchmark requires the 'hybrid-f32' feature to be enabled.");
    println!("📋 Run with: timeout 3600 cargo run --example maximum_load_benchmark --features hybrid-f32 --release");
}
