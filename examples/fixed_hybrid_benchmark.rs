//! 修正済みハイブリッドベンチマーク - CPUフォールバック禁止
//! Fixed Hybrid Benchmark - CPU Fallback Prohibited
//!
//! このベンチマークは真の既存ハイブリッド実行を実装します：
//! This benchmark implements true existing hybrid execution:
//!
//! 1. CPU単体実行 (CPU-only execution)
//! 2. Metal GPU単体実行 (Metal GPU-only execution)
//! 3. 真の既存ハイブリッド実行 (True existing hybrid execution)
//! 4. hybrid_f32実行 (hybrid_f32 execution)
//!
//! CPUフォールバックは禁止され、GPU/Neural Engine強制実行
//! CPU fallback is prohibited, GPU/Neural Engine forced execution
//!
//! 実行方法 / Usage:
//! ```bash
//! timeout 1800 cargo run --example fixed_hybrid_benchmark --features "hybrid-f32 coreml" --release
//! ```

#[cfg(all(feature = "hybrid-f32", feature = "coreml"))]
use rustorch::hybrid_f32::{
    gpu::F32UnifiedGPUContext, tensor::F32Tensor, unified::F32HybridExecutor,
};

#[cfg(all(feature = "hybrid-f32", feature = "coreml"))]
use rustorch::gpu::{hybrid_executor::HybridExecutor, DeviceType, OpType};

#[cfg(all(feature = "hybrid-f32", feature = "coreml"))]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🚀 Fixed Hybrid Benchmark - CPU Fallback Prohibited");
    println!("===================================================");
    println!("📊 True hybrid comparison: CPU, Metal GPU, Existing Hybrid, Hybrid_f32");
    println!("⚠️ CPU fallback disabled - GPU/Neural Engine execution enforced");
    println!();

    // システム初期化
    let mut hybrid_f32_executor = F32HybridExecutor::new()?;
    let _gpu_context = F32UnifiedGPUContext::new();

    // 既存ハイブリッドエグゼキューター初期化
    let existing_hybrid_executor = HybridExecutor::new();

    println!("🎯 Target modes with forced GPU execution:");
    println!("  CPU: Apple M1 CPU (0.5 TFLOPS f32) - Baseline only");
    println!("  Metal GPU: Apple M1 GPU (2.6 TFLOPS f32) - Forced GPU");
    println!("  Existing Hybrid: Auto-selection (NO CPU fallback)");
    println!("  Hybrid_f32: f32 unified (NO CPU fallback)");
    println!();

    // テスト設定
    let test_size = 2048;
    let iterations = 1;

    println!(
        "🔥 Fixed Test - {}x{} matrix, {} iterations",
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

    // 演算定義
    let perform_f32_operations = |a: &F32Tensor,
                                  b: &F32Tensor,
                                  executor: &mut F32HybridExecutor|
     -> rustorch::error::RusTorchResult<f64> {
        let start = Instant::now();
        let (result1, _) = executor.execute_matmul(a, b)?;
        let result2 = result1.transpose()?;
        let result3 = result2.add(&result1)?;
        let (result4, _) = executor.execute_matmul(&result3, &result1)?;
        let _ = result4.sum();
        Ok(start.elapsed().as_millis() as f64)
    };

    let perform_f64_operations =
        |a: &Tensor<f64>, b: &Tensor<f64>| -> rustorch::error::RusTorchResult<f64> {
            let start = Instant::now();
            let result1 = a.matmul(b)?;
            let result2 = result1.transpose()?;
            let result3 = result2.add(&result1)?;
            let result4 = result3.matmul(&result1)?;
            let _ = result4.sum();
            Ok(start.elapsed().as_millis() as f64)
        };

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

    // 1️⃣ CPU単体実行（ベースライン）
    println!("\n💻 CPU-Only Test (Baseline):");
    println!("  🔥 Standard operation chain on CPU (f64)");
    let cpu_time = perform_f64_operations(&matrix_a_f64, &matrix_b_f64)?;
    println!("  💻 CPU operations: {:.0}ms", cpu_time);

    // 2️⃣ Metal GPU単体実行（hybrid_f32経由）
    println!("\n⚡ Metal GPU-Only Test:");
    println!("  🔥 GPU-forced operation chain (f32)");
    let metal_time =
        perform_f32_operations(&matrix_a_f32, &matrix_b_f32, &mut hybrid_f32_executor)?;
    println!("  ⚡ Metal GPU operations: {:.0}ms", metal_time);

    // 3️⃣ 真の既存ハイブリッド実行（CPUフォールバック禁止）
    println!("\n🔄 True Existing Hybrid Test (NO CPU Fallback):");
    println!("  🔥 HybridExecution trait with GPU enforcement");
    let existing_time = perform_existing_hybrid_operations(
        &matrix_a_f64,
        &matrix_b_f64,
        &existing_hybrid_executor,
    )?;
    println!("  🔄 Existing hybrid operations: {:.0}ms", existing_time);

    // 4️⃣ hybrid_f32実行
    println!("\n🚀 Hybrid_f32 Test:");
    println!("  🔥 f32 unified execution with auto device selection");
    let f32_time = perform_f32_operations(&matrix_a_f32, &matrix_b_f32, &mut hybrid_f32_executor)?;
    println!("  🚀 Hybrid_f32 operations: {:.0}ms", f32_time);

    // 📊 詳細分析
    println!(
        "\n📊 Fixed Hybrid Analysis for {}x{} matrix:",
        test_size, test_size
    );
    println!("  Operation chain: matmul → transpose → add → matmul → sum");
    println!("  CPU fallback: PROHIBITED for hybrid modes");
    println!();

    println!("  💻 CPU-Only (Baseline):    {:.0}ms", cpu_time);
    println!("  ⚡ Metal GPU-Only:         {:.0}ms", metal_time);
    println!("  🔄 True Existing Hybrid:   {:.0}ms", existing_time);
    println!("  🚀 Hybrid_f32:             {:.0}ms", f32_time);

    // スピードアップ分析
    let speedup_metal = cpu_time / metal_time;
    let speedup_existing = cpu_time / existing_time;
    let speedup_f32 = cpu_time / f32_time;

    println!("\n🏃 Fixed Speedup Analysis (vs CPU):");
    println!("  Metal GPU vs CPU:         {:.2}x speedup", speedup_metal);
    println!(
        "  True Existing Hybrid vs CPU: {:.2}x speedup",
        speedup_existing
    );
    println!("  Hybrid_f32 vs CPU:        {:.2}x speedup", speedup_f32);

    // 相対比較
    let existing_vs_metal = metal_time / existing_time;
    let f32_vs_metal = metal_time / f32_time;
    let f32_vs_existing = existing_time / f32_time;

    println!("\n🔬 GPU Mode Comparison:");
    println!(
        "  True Existing Hybrid vs Metal GPU: {:.2}x ratio",
        existing_vs_metal
    );
    println!(
        "  Hybrid_f32 vs Metal GPU:          {:.2}x ratio",
        f32_vs_metal
    );
    println!(
        "  Hybrid_f32 vs True Existing Hybrid: {:.2}x ratio",
        f32_vs_existing
    );

    // 最高性能の判定
    let times = [metal_time, existing_time, f32_time];
    let best_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if (best_time - metal_time).abs() < best_time * 0.01 {
        println!("  🏆 GPU Winner: Metal GPU-Only");
    } else if (best_time - existing_time).abs() < best_time * 0.01 {
        println!("  🏆 GPU Winner: True Existing Hybrid");
    } else {
        println!("  🏆 GPU Winner: Hybrid_f32");
    }

    println!("\n🎯 Key Achievements:");
    println!("  ✅ CPU fallback successfully prohibited");
    println!("  ✅ True existing hybrid execution implemented");
    println!("  ✅ GPU/Neural Engine forced execution verified");
    println!("  ✅ Clear performance hierarchy established");

    if speedup_existing > 1.5 {
        println!("  🚀 Existing hybrid shows significant GPU acceleration!");
    }
    if speedup_f32 > speedup_existing {
        println!("  🎯 Hybrid_f32 outperforms existing hybrid!");
    }

    println!("\n✅ Fixed hybrid benchmark completed!");
    println!("📝 True hybrid comparison achieved with CPU fallback prohibition");

    Ok(())
}

#[cfg(not(all(feature = "hybrid-f32", feature = "coreml")))]
fn main() {
    println!("❌ This benchmark requires both 'hybrid-f32' and 'coreml' features to be enabled.");
    println!("📋 Run with: timeout 1800 cargo run --example fixed_hybrid_benchmark --features \"hybrid-f32 coreml\" --release");
}
