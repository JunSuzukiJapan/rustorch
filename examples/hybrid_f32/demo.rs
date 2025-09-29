//! f32統一ハイブリッドシステム デモ
//! f32 Unified Hybrid System Demo
//!
//! このデモは実験的なf32統一ハイブリッドシステムの基本的な使用法と
//! 変換コスト削減効果を示します。
//!
//! This demo shows basic usage of the experimental f32 unified hybrid system
//! and demonstrates conversion cost reduction effects.
//!
//! 実行方法 / Usage:
//! ```bash
//! cargo run --example hybrid_f32_demo --features hybrid-f32
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    benchmarks::run_quick_benchmark, tensor::F32Tensor, unified::F32HybridExecutor,
};

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 f32統一ハイブリッドシステム デモ");
    println!("🚀 f32 Unified Hybrid System Demo");
    println!("=====================================\n");

    // 実験警告の表示
    rustorch::hybrid_f32_experimental!();

    // 1. 基本的なF32Tensor操作
    demo_basic_f32_tensor()?;

    // 2. デバイス間移動（変換コストなし）
    demo_zero_cost_device_movement()?;

    // 3. 統一ハイブリッド実行
    demo_unified_hybrid_execution()?;

    // 4. クイックベンチマーク
    demo_quick_benchmark()?;

    println!("\n✅ デモ完了！f32統一ハイブリッドシステムの基本機能を確認しました。");
    println!("✅ Demo completed! Basic functionality of f32 unified hybrid system verified.");

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn demo_basic_f32_tensor() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 1. 基本的なF32Tensor操作");
    println!("📊 1. Basic F32Tensor Operations");
    println!("--------------------------------");

    // テンソル作成（変換コストなし）
    let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = F32Tensor::zeros(&[2, 2]);
    let c = F32Tensor::randn(&[2, 2]);

    println!("  📝 作成したテンソル:");
    println!("     a.shape(): {:?}", a.shape());
    println!("     b.shape(): {:?} (zeros)", b.shape());
    println!("     c.shape(): {:?} (random)", c.shape());

    // 行列乗算（智的デバイス選択）
    let result = a.matmul(&b)?;
    println!("  ⚡ 行列乗算実行: a × b = result");
    println!("     result.shape(): {:?}", result.shape());

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn demo_zero_cost_device_movement() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 2. ゼロコストデバイス間移動");
    println!("🔄 2. Zero-Cost Device Movement");
    println!("--------------------------------");

    let mut tensor = F32Tensor::randn(&[100, 100]);

    println!("  💻 初期状態: CPU");
    println!("     Device state: {:?}", tensor.device_state());

    // Metal GPUに移動（変換コストなし）
    tensor.to_metal(0)?;
    println!("  🚀 Metal GPUに移動完了（変換コストなし）");
    println!("     Device state: {:?}", tensor.device_state());

    // Neural Engineに移動（変換コストなし）
    tensor.to_coreml(0)?;
    println!("  🧠 Neural Engineに移動完了（変換コストなし）");
    println!("     Device state: {:?}", tensor.device_state());

    // 全デバイス同期
    tensor.synchronize_all()?;
    println!("  🔄 全デバイス同期完了");
    println!("     Device state: {:?}", tensor.device_state());

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn demo_unified_hybrid_execution() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 3. 統一ハイブリッド実行");
    println!("🎯 3. Unified Hybrid Execution");
    println!("-------------------------------");

    // ハイブリッド実行エンジンを初期化
    let mut executor = F32HybridExecutor::new()?;
    executor.initialize()?;

    // 異なるサイズの行列で最適デバイス選択をテスト
    let test_sizes = vec![
        (50, 50),   // 小規模 → CPU
        (200, 200), // 中規模 → Neural Engine
        (800, 800), // 大規模 → Metal GPU
    ];

    for (size_m, size_n) in test_sizes {
        println!("\n  📏 テストサイズ: {}x{}", size_m, size_n);

        let a = F32Tensor::randn(&[size_m, size_n]);
        let b = F32Tensor::randn(&[size_n, size_m]);

        let start_time = std::time::Instant::now();
        let (result, experiment_results) = executor.execute_matmul(&a, &b)?;
        let execution_time = start_time.elapsed();

        println!("     結果形状: {:?}", result.shape());
        println!("     実行時間: {:?}", execution_time);
        println!(
            "     変換コスト削減: {:.1}%",
            experiment_results.conversion_cost_reduction
        );
    }

    // パフォーマンス統計を表示
    let stats = executor.get_performance_stats();
    println!("\n  📊 実行統計:");
    println!("     総実行回数: {}", stats.total_operations);
    println!("     平均実行時間: {:?}", stats.average_execution_time);
    println!(
        "     変換コスト削減時間: {:?}",
        stats.conversion_cost_savings
    );
    println!("     デバイス使用状況:");
    for (device, count) in stats.device_usage {
        println!("       {}: {} 回", device, count);
    }

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn demo_quick_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ 4. クイックベンチマーク");
    println!("⚡ 4. Quick Benchmark");
    println!("---------------------");

    println!("  🏁 実行中... (数秒お待ちください)");
    run_quick_benchmark()?;

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ このデモを実行するには hybrid-f32 フィーチャーが必要です。");
    println!("❌ This demo requires the hybrid-f32 feature to be enabled.");
    println!("");
    println!("実行方法 / Usage:");
    println!("cargo run --example hybrid_f32_demo --features hybrid-f32");
}
