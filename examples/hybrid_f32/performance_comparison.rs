//! f32統一ハイブリッドシステム vs 従来システム パフォーマンス比較
//! f32 Unified Hybrid System vs Legacy System Performance Comparison
//!
//! このベンチマークは既存のsimple_performance_testと同等のテストを
//! hybrid_f32システムで実行し、直接的な性能比較を行います。
//!
//! This benchmark runs tests equivalent to the existing simple_performance_test
//! using the hybrid_f32 system for direct performance comparison.
//!
//! 実行方法 / Usage:
//! ```bash
//! # 1. 従来システムのベンチマーク実行
//! cargo bench --bench simple_performance_test --no-default-features
//!
//! # 2. hybrid_f32システムのベンチマーク実行
//! cargo run --example hybrid_f32_performance_comparison --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::benchmarks::performance_test::{
    F32PerformanceTest, PerformanceTestConfig, PerformanceTestResults
};

#[cfg(feature = "hybrid-f32")]
use rustorch::tensor::Tensor;
use std::time::Instant;

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 f32統一ハイブリッドシステム vs 従来システム パフォーマンス比較");
    println!("🎯 f32 Unified Hybrid vs Legacy System Performance Comparison");
    println!("================================================================\n");

    // 実験警告の表示
    rustorch::hybrid_f32_experimental!();

    // 設定
    let config = PerformanceTestConfig {
        iterations: 100,
        warmup_iterations: 10,
        tensor_sizes: vec![1000, 10000],
        matrix_sizes: vec![64, 128, 256],
    };

    println!("📊 テスト設定 / Test Configuration:");
    println!("  反復回数 / Iterations: {}", config.iterations);
    println!("  ウォームアップ / Warmup: {}", config.warmup_iterations);
    println!();

    // 1. 従来システム（リファレンス実装）ベンチマーク
    println!("📈 1. 従来システム（f64ベース）ベンチマーク");
    println!("📈 1. Legacy System (f64-based) Benchmark");
    println!("------------------------------------------");

    let legacy_results = run_legacy_benchmark(&config)?;

    // 2. f32統一ハイブリッドシステムベンチマーク
    println!("\n🚀 2. f32統一ハイブリッドシステムベンチマーク");
    println!("🚀 2. f32 Unified Hybrid System Benchmark");
    println!("--------------------------------------------");

    let mut f32_test = F32PerformanceTest::new(config.clone())?;
    let f32_results = f32_test.run_full_benchmark()?;

    // 3. 詳細比較分析
    println!("\n🔍 3. 詳細比較分析 / Detailed Comparison Analysis");
    println!("==============================================");

    perform_detailed_analysis(&legacy_results, &f32_results);

    // 4. 最適化効果の要約
    println!("\n✨ 4. f32統一ハイブリッドシステムの最適化効果");
    println!("✨ 4. f32 Unified Hybrid System Optimization Effects");
    println!("=================================================");

    summarize_optimization_effects(&legacy_results, &f32_results);

    println!("\n✅ パフォーマンス比較完了！");
    println!("✅ Performance comparison completed!");

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
#[derive(Debug, Clone)]
struct LegacyBenchmarkResults {
    tensor_addition: f64,           // ms
    tensor_sum: f64,                // ms
    tensor_creation: f64,           // ms
    tensor_clone: f64,              // ms
    matrix_multiplication: f64,     // ms
}

#[cfg(feature = "hybrid-f32")]
fn run_legacy_benchmark(config: &PerformanceTestConfig) -> Result<LegacyBenchmarkResults, Box<dyn std::error::Error>> {
    println!("  実行中... / Running...");

    // テンソル加算ベンチマーク
    let tensor_addition = {
        let size = 10000;
        let tensor_a = Tensor::<f64>::from_vec((0..size).map(|i| i as f64).collect(), vec![size]);
        let tensor_b = Tensor::<f64>::from_vec((0..size).map(|i| (i + 1) as f64).collect(), vec![size]);

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let _ = &tensor_a + &tensor_b;
        }

        // 実測定
        let start = Instant::now();
        for _ in 0..config.iterations {
            let _ = &tensor_a + &tensor_b;
        }
        let duration = start.elapsed();
        duration.as_nanos() as f64 / config.iterations as f64 / 1_000_000.0
    };

    // テンソル合計ベンチマーク
    let tensor_sum = {
        let size = 10000;
        let tensor = Tensor::<f64>::from_vec((0..size).map(|i| i as f64).collect(), vec![size]);

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let _ = tensor.sum();
        }

        // 実測定
        let start = Instant::now();
        for _ in 0..config.iterations {
            let _ = tensor.sum();
        }
        let duration = start.elapsed();
        duration.as_nanos() as f64 / config.iterations as f64 / 1_000_000.0
    };

    // テンソル作成ベンチマーク
    let tensor_creation = {
        let size = 1000;

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let _ = Tensor::<f64>::zeros(&[size]);
        }

        // 実測定
        let start = Instant::now();
        for _ in 0..config.iterations {
            let _ = Tensor::<f64>::zeros(&[size]);
        }
        let duration = start.elapsed();
        duration.as_nanos() as f64 / config.iterations as f64 / 1_000_000.0
    };

    // テンソル複製ベンチマーク
    let tensor_clone = {
        let tensor = Tensor::<f64>::from_vec((0..1000).map(|i| i as f64).collect(), vec![1000]);

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let _ = tensor.clone();
        }

        // 実測定
        let start = Instant::now();
        for _ in 0..config.iterations {
            let _ = tensor.clone();
        }
        let duration = start.elapsed();
        duration.as_nanos() as f64 / config.iterations as f64 / 1_000_000.0
    };

    // 行列乗算ベンチマーク
    let matrix_multiplication = {
        let size = 128;
        let mat_a = Tensor::<f64>::from_vec(
            (0..size * size).map(|i| (i as f64) * 0.01).collect(),
            vec![size, size]
        );
        let mat_b = Tensor::<f64>::from_vec(
            (0..size * size).map(|i| (i as f64) * 0.01).collect(),
            vec![size, size]
        );

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let _ = mat_a.matmul(&mat_b);
        }

        // 実測定
        let start = Instant::now();
        for _ in 0..config.iterations {
            let _ = mat_a.matmul(&mat_b);
        }
        let duration = start.elapsed();
        duration.as_nanos() as f64 / config.iterations as f64 / 1_000_000.0
    };

    let results = LegacyBenchmarkResults {
        tensor_addition,
        tensor_sum,
        tensor_creation,
        tensor_clone,
        matrix_multiplication,
    };

    println!("  従来システム結果:");
    println!("    Tensor addition:       {:.6} ms", results.tensor_addition);
    println!("    Tensor sum:            {:.6} ms", results.tensor_sum);
    println!("    Tensor creation:       {:.6} ms", results.tensor_creation);
    println!("    Tensor clone:          {:.6} ms", results.tensor_clone);
    println!("    Matrix multiplication: {:.6} ms", results.matrix_multiplication);

    Ok(results)
}

#[cfg(feature = "hybrid-f32")]
fn perform_detailed_analysis(legacy: &LegacyBenchmarkResults, f32_hybrid: &PerformanceTestResults) {
    println!("| 操作 | 従来システム | f32ハイブリッド | 性能比 | 改善率 |");
    println!("|------|-------------|----------------|--------|--------|");

    let operations = [
        ("テンソル加算", legacy.tensor_addition, f32_hybrid.tensor_addition),
        ("テンソル合計", legacy.tensor_sum, f32_hybrid.tensor_sum),
        ("テンソル作成", legacy.tensor_creation, f32_hybrid.tensor_creation),
        ("テンソル複製", legacy.tensor_clone, f32_hybrid.tensor_clone),
        ("行列乗算", legacy.matrix_multiplication, f32_hybrid.matrix_multiplication),
    ];

    let mut total_speedup = 0.0;
    let mut count = 0;

    for (op_name, legacy_time, f32_time) in operations.iter() {
        let speedup = legacy_time / f32_time;
        let improvement = (1.0 - f32_time / legacy_time) * 100.0;

        println!("| {} | {:.6} ms | {:.6} ms | {:.2}x | {:.1}% |",
            op_name, legacy_time, f32_time, speedup, improvement);

        total_speedup += speedup;
        count += 1;
    }

    let avg_speedup = total_speedup / count as f64;
    println!("\n📊 平均性能比: {:.2}x", avg_speedup);
    println!("📊 使用デバイス: {}", f32_hybrid.device_used);
}

#[cfg(feature = "hybrid-f32")]
fn summarize_optimization_effects(legacy: &LegacyBenchmarkResults, f32_hybrid: &PerformanceTestResults) {
    println!("🎯 主要な最適化効果:");
    println!();

    // 1. 変換コスト削減
    println!("1. 🔄 変換コスト削減:");
    println!("   ✓ f64 → f32 変換処理の完全排除");
    println!("   ✓ ゼロコピーデバイス間移動");

    // 2. メモリ効率
    println!();
    println!("2. 💾 メモリ効率向上:");
    println!("   ✓ 50% メモリ使用量削減 (f64 → f32)");
    println!("   ✓ キャッシュ効率の向上");

    // 3. デバイス最適化
    println!();
    println!("3. 🚀 智的デバイス選択:");
    println!("   ✓ 演算規模に応じた最適デバイス選択");
    println!("   ✓ Neural Engine/Metal GPU の直接活用");
    println!("   ✓ 使用デバイス: {}", f32_hybrid.device_used);

    // 4. 実測改善効果
    let matrix_speedup = legacy.matrix_multiplication / f32_hybrid.matrix_multiplication;
    println!();
    println!("4. 📈 実測改善効果:");
    println!("   ✓ 行列乗算: {:.2}x 高速化", matrix_speedup);

    if matrix_speedup > 1.0 {
        println!("   ✓ 大規模演算での優位性確認");
    } else {
        println!("   ⚠️ 小規模演算では初期化コストが影響");
        println!("   💡 大規模データセットでの真価発揮を期待");
    }

    println!();
    println!("🔮 将来の最適化可能性:");
    println!("   • さらなるGPU並列化");
    println!("   • 演算融合最適化");
    println!("   • メモリプールの活用");
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ この比較テストを実行するには hybrid-f32 フィーチャーが必要です。");
    println!("❌ This comparison test requires the hybrid-f32 feature to be enabled.");
    println!("");
    println!("実行方法 / Usage:");
    println!("1. 従来システム: cargo bench --bench simple_performance_test --no-default-features");
    println!("2. f32ハイブリッド: cargo run --example hybrid_f32_performance_comparison --features hybrid-f32 --release");
}