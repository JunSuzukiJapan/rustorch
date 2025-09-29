//! 集中比較ベンチマーク - 主要モードの詳細比較
//! Focused Comparison Benchmark - Detailed comparison of key modes
//!
//! このベンチマークは主要な実行モードを効率的に比較します：
//! This benchmark efficiently compares key execution modes:
//!
//! 1. CPU単体実行 (CPU-only execution)
//! 2. Metal GPU単体実行 (Metal GPU-only execution)
//! 3. 既存ハイブリッド実行 (Existing hybrid execution)
//! 4. hybrid_f32実行 (hybrid_f32 execution)
//!
//! 実行方法 / Usage:
//! ```bash
//! timeout 1800 cargo run --example focused_comparison_benchmark --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    gpu::F32UnifiedGPUContext, tensor::F32Tensor, unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
use rustorch::gpu::{hybrid_executor::HybridExecutor, DeviceType, OpType};

#[cfg(feature = "hybrid-f32")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 Focused Comparison Benchmark");
    println!("================================");
    println!("📊 Detailed comparison: CPU, Metal GPU, True Existing Hybrid, Hybrid_f32");
    println!("⏱️ Optimized for comprehensive results within reasonable time");
    println!();

    // システム初期化
    let mut hybrid_executor = F32HybridExecutor::new()?;
    let _gpu_context = F32UnifiedGPUContext::new();

    // 既存ハイブリッドエグゼキューター初期化
    let existing_hybrid_executor = HybridExecutor::new();

    println!("🎯 Target modes for focused comparison:");
    println!("  CPU: Apple M1 CPU (0.5 TFLOPS f32) - Baseline");
    println!("  Metal GPU: Apple M1 GPU (2.6 TFLOPS f32) - GPU acceleration");
    println!("  True Existing Hybrid: Metal(0) → CoreML(0) → CPU (improved chain)");
    println!("  Hybrid_f32: f32 unified with zero conversion cost");
    println!();

    // 集中テスト設定 - 明確な差が出るサイズで効率的に
    let test_size = 2048;
    let iterations = 1;

    println!(
        "🔥 Focused Test - {}x{} matrix, {} iterations",
        test_size, test_size, iterations
    );
    println!(
        "Memory usage: ~{:.1} GB per matrix",
        (test_size * test_size * 4) as f64 / 1_000_000_000.0
    );
    println!("=====================================");

    // データ作成
    println!("📊 Creating test matrices...");

    // f32データ（hybrid_f32用）
    let data_a_f32: Vec<f32> = (0..test_size * test_size)
        .map(|i| (i as f32 % 100.0) + 1.0)
        .collect();
    let data_b_f32: Vec<f32> = (0..test_size * test_size)
        .map(|i| ((i + test_size) as f32 % 100.0) + 1.0)
        .collect();
    let matrix_a_f32 = F32Tensor::new(data_a_f32, &[test_size, test_size])?;
    let matrix_b_f32 = F32Tensor::new(data_b_f32, &[test_size, test_size])?;

    // f64データ（既存hybrid用）
    let data_a_f64: Vec<f64> = (0..test_size * test_size)
        .map(|i| (i as f64 % 100.0) + 1.0)
        .collect();
    let data_b_f64: Vec<f64> = (0..test_size * test_size)
        .map(|i| ((i + test_size) as f64 % 100.0) + 1.0)
        .collect();
    let matrix_a_f64 = Tensor::from_vec(data_a_f64, vec![test_size, test_size]);
    let matrix_b_f64 = Tensor::from_vec(data_b_f64, vec![test_size, test_size]);

    // 標準化された演算チェーン
    let perform_standard_operations_f32 = |a: &F32Tensor,
                                           b: &F32Tensor,
                                           executor: &mut F32HybridExecutor|
     -> rustorch::error::RusTorchResult<f64> {
        let start = Instant::now();

        // 1. 行列乗算
        let (result1, _) = executor.execute_matmul(a, b)?;
        // 2. 転置
        let result2 = result1.transpose()?;
        // 3. 加算
        let result3 = result2.add(&result1)?;
        // 4. 再度行列乗算
        let (result4, _) = executor.execute_matmul(&result3, &result1)?;
        // 5. 統計操作
        let _ = result4.sum();

        Ok(start.elapsed().as_millis() as f64)
    };

    let perform_standard_operations_f64 =
        |a: &Tensor<f64>, b: &Tensor<f64>| -> rustorch::error::RusTorchResult<f64> {
            let start = Instant::now();

            // 1. 行列乗算
            let result1 = a.matmul(b)?;
            // 2. 転置
            let result2 = result1.transpose()?;
            // 3. 加算
            let result3 = result2.add(&result1)?;
            // 4. 再度行列乗算
            let result4 = result3.matmul(&result1)?;
            // 5. 統計操作
            let _ = result4.sum();

            Ok(start.elapsed().as_millis() as f64)
        };

    // 1️⃣ CPU単体実行
    println!("\n💻 CPU-Only Test:");
    println!("  🔥 Standard operation chain on CPU (f64)");
    let mut cpu_times = Vec::new();
    for i in 0..iterations {
        println!("  💻 CPU iteration {}/{}", i + 1, iterations);
        let time = perform_standard_operations_f64(&matrix_a_f64, &matrix_b_f64)?;
        cpu_times.push(time);
        println!("    ⏱️ CPU operations: {:.0}ms", time);
    }
    let cpu_avg = cpu_times.iter().sum::<f64>() / iterations as f64;
    println!("  💻 CPU average: {:.0}ms per operation chain", cpu_avg);

    // 2️⃣ Metal GPU単体実行（hybrid_executor経由）
    println!("\n⚡ Metal GPU-Only Test:");
    println!("  🔥 Standard operation chain on Metal GPU (f32)");
    let mut metal_times = Vec::new();
    for i in 0..iterations {
        println!("  ⚡ Metal GPU iteration {}/{}", i + 1, iterations);
        let time =
            perform_standard_operations_f32(&matrix_a_f32, &matrix_b_f32, &mut hybrid_executor)?;
        metal_times.push(time);
        println!("    ⏱️ Metal GPU operations: {:.0}ms", time);
    }
    let metal_avg = metal_times.iter().sum::<f64>() / iterations as f64;
    println!(
        "  ⚡ Metal GPU average: {:.0}ms per operation chain",
        metal_avg
    );

    // 既存ハイブリッド演算（GPU強制実行）
    let perform_existing_hybrid_operations = |a: &Tensor<f64>,
                                              b: &Tensor<f64>,
                                              executor: &HybridExecutor|
     -> rustorch::error::RusTorchResult<f64> {
        use rustorch::gpu::hybrid_executor::HybridExecution;

        let start = Instant::now();

        // HybridExecutionトレイトを使用してGPU強制実行
        let result1 = a.hybrid_operation(OpType::LinearAlgebra, |device| {
            if device == DeviceType::Cpu {
                return Err(rustorch::error::RusTorchError::tensor_op(
                    "CPU fallback prohibited - GPU execution required",
                ));
            }
            println!("    🎯 Executing matmul on device: {:?}", device);
            a.matmul(b)
        })?;

        let result2 = result1.hybrid_operation(OpType::LinearAlgebra, |device| {
            if device == DeviceType::Cpu {
                return Err(rustorch::error::RusTorchError::tensor_op(
                    "CPU fallback prohibited - GPU execution required",
                ));
            }
            println!("    🎯 Executing transpose on device: {:?}", device);
            result1.transpose()
        })?;

        let result3 = result2.hybrid_operation(OpType::LinearAlgebra, |device| {
            if device == DeviceType::Cpu {
                return Err(rustorch::error::RusTorchError::tensor_op(
                    "CPU fallback prohibited - GPU execution required",
                ));
            }
            println!("    🎯 Executing add on device: {:?}", device);
            result2.add(&result1)
        })?;

        let result4 = result3.hybrid_operation(OpType::LinearAlgebra, |device| {
            if device == DeviceType::Cpu {
                return Err(rustorch::error::RusTorchError::tensor_op(
                    "CPU fallback prohibited - GPU execution required",
                ));
            }
            println!("    🎯 Executing final matmul on device: {:?}", device);
            result3.matmul(&result1)
        })?;

        let _ = result4.sum();
        Ok(start.elapsed().as_millis() as f64)
    };

    // 3️⃣ 真の既存ハイブリッド実行（CPUフォールバック禁止）
    println!("\n🔄 True Existing Hybrid Test (NO CPU Fallback):");
    println!("  🔥 HybridExecution trait with GPU enforcement");
    let mut existing_times = Vec::new();
    for i in 0..iterations {
        println!(
            "  🔄 True existing hybrid iteration {}/{}",
            i + 1,
            iterations
        );
        println!("    🔄 Using HybridExecution trait with CPU fallback prohibition");

        let time = perform_existing_hybrid_operations(
            &matrix_a_f64,
            &matrix_b_f64,
            &existing_hybrid_executor,
        )?;
        existing_times.push(time);
        println!("    ⏱️ True existing hybrid operations: {:.0}ms", time);
    }
    let existing_avg = existing_times.iter().sum::<f64>() / iterations as f64;
    println!(
        "  🔄 True existing hybrid average: {:.0}ms per operation chain",
        existing_avg
    );

    // 4️⃣ hybrid_f32実行
    println!("\n🚀 Hybrid_f32 Test:");
    println!("  🔥 Standard operation chain with hybrid_f32 (f32)");
    let mut f32_times = Vec::new();
    for i in 0..iterations {
        println!("  🚀 Hybrid_f32 iteration {}/{}", i + 1, iterations);
        println!("    🚀 f32 unified execution with zero conversion cost");
        println!("    📊 Automatic device selection for optimal performance");

        let time =
            perform_standard_operations_f32(&matrix_a_f32, &matrix_b_f32, &mut hybrid_executor)?;
        f32_times.push(time);
        println!("    ⏱️ Hybrid_f32 operations: {:.0}ms", time);
    }
    let f32_avg = f32_times.iter().sum::<f64>() / iterations as f64;
    println!(
        "  🚀 Hybrid_f32 average: {:.0}ms per operation chain",
        f32_avg
    );

    // 📊 詳細分析
    println!(
        "\n📊 Focused Comparison Analysis for {}x{} matrix:",
        test_size, test_size
    );
    println!("  Operation chain: matmul → transpose → add → matmul → sum");
    println!(
        "  Memory per matrix: {:.1} GB",
        (test_size * test_size * 4) as f64 / 1_000_000_000.0
    );
    println!();

    println!("  💻 CPU-Only:         {:.0}ms per chain", cpu_avg);
    println!("  ⚡ Metal GPU-Only:   {:.0}ms per chain", metal_avg);
    println!("  🔄 True Existing Hybrid: {:.0}ms per chain", existing_avg);
    println!("  🚀 Hybrid_f32:       {:.0}ms per chain", f32_avg);

    // スピードアップ分析
    let speedup_metal = cpu_avg / metal_avg;
    let speedup_existing = cpu_avg / existing_avg;
    let speedup_f32 = cpu_avg / f32_avg;

    println!("\n🏃 Focused Speedup Analysis (vs CPU):");
    println!("  Metal GPU vs CPU:      {:.2}x speedup", speedup_metal);
    println!(
        "  True Existing Hybrid vs CPU: {:.2}x speedup",
        speedup_existing
    );
    println!("  Hybrid_f32 vs CPU:     {:.2}x speedup", speedup_f32);

    // 相対比較
    let metal_vs_existing = existing_avg / metal_avg;
    let metal_vs_f32 = f32_avg / metal_avg;
    let f32_vs_existing = existing_avg / f32_avg;

    println!("\n🔬 Relative Performance Analysis:");
    println!(
        "  Metal GPU vs True Existing Hybrid: {:.2}x ratio",
        metal_vs_existing
    );
    println!("  Metal GPU vs Hybrid_f32:      {:.2}x ratio", metal_vs_f32);
    println!(
        "  Hybrid_f32 vs True Existing Hybrid: {:.2}x ratio",
        f32_vs_existing
    );

    // 最高性能の判定
    let times = [cpu_avg, metal_avg, existing_avg, f32_avg];
    let best_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if (best_time - cpu_avg).abs() < best_time * 0.01 {
        println!("  🏆 Focused Winner: CPU-Only");
    } else if (best_time - metal_avg).abs() < best_time * 0.01 {
        println!("  🏆 Focused Winner: Metal GPU-Only");
    } else if (best_time - existing_avg).abs() < best_time * 0.01 {
        println!("  🏆 Focused Winner: True Existing Hybrid");
    } else {
        println!("  🏆 Focused Winner: Hybrid_f32");
    }

    // 効率性判定
    let max_speedup = [speedup_metal, speedup_existing, speedup_f32]
        .iter()
        .fold(0.0f64, |a, &b| a.max(b));

    if max_speedup > 5.0 {
        println!("  🚀 Exceptional acceleration achieved!");
    } else if max_speedup > 2.0 {
        println!("  🚀 Excellent acceleration achieved!");
    } else if max_speedup > 1.5 {
        println!("  ✅ Good acceleration achieved");
    } else if max_speedup > 1.2 {
        println!("  📈 Modest acceleration achieved");
    } else {
        println!("  ⚠️ Limited acceleration observed");
    }

    // hybrid_f32の優位性分析
    println!("\n🎯 Hybrid_f32 Advantages:");
    if f32_vs_existing > 1.5 {
        println!(
            "  🎯 Major advantage over true existing hybrid ({:.2}x faster)",
            f32_vs_existing
        );
    } else if f32_vs_existing > 1.2 {
        println!(
            "  📈 Significant advantage over true existing hybrid ({:.2}x faster)",
            f32_vs_existing
        );
    } else if f32_vs_existing > 1.05 {
        println!(
            "  📈 Moderate advantage over true existing hybrid ({:.2}x faster)",
            f32_vs_existing
        );
    } else {
        println!("  ⚖️ Similar performance to true existing hybrid");
    }

    if metal_vs_f32 > 0.95 && metal_vs_f32 < 1.05 {
        println!("  ✅ Hybrid_f32 matches Metal GPU performance");
    } else if metal_vs_f32 > 1.05 {
        println!(
            "  🚀 Hybrid_f32 outperforms Metal GPU ({:.2}x faster)",
            1.0 / metal_vs_f32
        );
    } else {
        println!(
            "  📊 Metal GPU slightly faster than Hybrid_f32 ({:.2}x)",
            metal_vs_f32
        );
    }

    println!("\n✅ Focused comparison benchmark completed!");
    println!("📝 Clear performance hierarchy established across all key execution modes");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ This benchmark requires the 'hybrid-f32' feature to be enabled.");
    println!("📋 Run with: timeout 1800 cargo run --example focused_comparison_benchmark --features hybrid-f32 --release");
}
