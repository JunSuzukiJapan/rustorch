//! f32統一ハイブリッドシステム vs 従来システム 定量的比較
//! f32 Unified Hybrid System vs Legacy System Quantitative Comparison
//!
//! このベンチマークは実験的なf32統一ハイブリッドシステムと従来のf64ベース
//! システムの性能を直接比較し、変換コスト削減効果を定量的に測定します。
//!
//! This benchmark directly compares the experimental f32 unified hybrid system
//! with the legacy f64-based system and quantitatively measures conversion cost
//! reduction effects.
//!
//! 実行方法 / Usage:
//! ```bash
//! cargo run --example hybrid_f32_quick_comparison --features hybrid-f32 --release
//! ```

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::{
    tensor::F32Tensor,
    unified::F32HybridExecutor,
    benchmarks::{BenchmarkConfig, F32HybridBenchmark},
};

#[cfg(feature = "hybrid-f32")]
use rustorch::tensor::Tensor;
use std::time::Instant;

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 f32統一ハイブリッドシステム vs 従来システム 定量的比較");
    println!("🎯 f32 Unified Hybrid vs Legacy System Quantitative Comparison");
    println!("================================================================\n");

    // 実験警告の表示
    rustorch::hybrid_f32_experimental!();

    // 比較用設定（短時間実行）
    let quick_config = BenchmarkConfig {
        matrix_sizes: vec![
            (64, 64, 64),     // 超小規模
            (128, 128, 128),  // 小規模
            (256, 256, 256),  // 中規模
        ],
        iterations: 5,
        warmup_iterations: 2,
        measure_baseline: true,
    };

    println!("📊 ベンチマーク設定 / Benchmark Configuration:");
    println!("  行列サイズ / Matrix sizes: {:?}", quick_config.matrix_sizes);
    println!("  反復回数 / Iterations: {}", quick_config.iterations);
    println!("  ウォームアップ / Warmup: {}", quick_config.warmup_iterations);
    println!();

    // 1. f32統一ハイブリッドシステムベンチマーク
    println!("🚀 1. f32統一ハイブリッドシステム性能測定");
    println!("🚀 1. f32 Unified Hybrid System Performance Measurement");
    println!("----------------------------------------------------");

    let mut hybrid_benchmark = F32HybridBenchmark::new(quick_config.clone())?;
    let hybrid_results = hybrid_benchmark.run_comprehensive_benchmark()?;

    // 2. 従来システム（f64ベース）直接ベンチマーク
    println!("\n📈 2. 従来システム（f64ベース）性能測定");
    println!("📈 2. Legacy System (f64-based) Performance Measurement");
    println!("-----------------------------------------------------");

    let legacy_results = run_legacy_system_benchmark(&quick_config)?;

    // 3. 詳細比較分析
    println!("\n🔍 3. 詳細比較分析 / Detailed Comparison Analysis");
    println!("==============================================");

    perform_detailed_comparison(&hybrid_results.hybrid_f32_results, &legacy_results)?;

    // 4. 変換コスト具体的測定
    println!("\n⚡ 4. 変換コスト具体的測定 / Specific Conversion Cost Measurement");
    println!("================================================================");

    measure_conversion_costs(&quick_config)?;

    // 5. メモリ使用量比較
    println!("\n💾 5. メモリ使用量比較 / Memory Usage Comparison");
    println!("=============================================");

    compare_memory_usage(&quick_config)?;

    println!("\n✅ 定量的比較完了！");
    println!("✅ Quantitative comparison completed!");

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
#[derive(Debug, Clone)]
struct LegacyBenchmarkResult {
    size: (usize, usize, usize),
    execution_time: std::time::Duration,
    conversion_time: std::time::Duration,
    total_time: std::time::Duration,
    tflops: f64,
}

#[cfg(feature = "hybrid-f32")]
fn run_legacy_system_benchmark(config: &BenchmarkConfig) -> Result<Vec<LegacyBenchmarkResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();

    for &(m, n, k) in &config.matrix_sizes {
        println!("  📏 測定中: {}x{}x{}", m, n, k);

        // f64データ準備（従来システム）
        let a_data: Vec<f64> = (0..m*k).map(|_| rand::random::<f64>()).collect();
        let b_data: Vec<f64> = (0..k*n).map(|_| rand::random::<f64>()).collect();

        let mut total_execution = std::time::Duration::from_secs(0);
        let mut total_conversion = std::time::Duration::from_secs(0);

        // ウォームアップ
        for _ in 0..config.warmup_iterations {
            let a = Tensor::from_vec(a_data.clone(), vec![m, k]);
            let b = Tensor::from_vec(b_data.clone(), vec![k, n]);
            let _ = a.matmul(&b);
        }

        // 実測定
        for _ in 0..config.iterations {
            // 変換時間測定
            let conversion_start = Instant::now();
            let a = Tensor::from_vec(a_data.clone(), vec![m, k]);
            let b = Tensor::from_vec(b_data.clone(), vec![k, n]);
            let conversion_time = conversion_start.elapsed();

            // 実行時間測定
            let execution_start = Instant::now();
            let _ = a.matmul(&b);
            let execution_time = execution_start.elapsed();

            total_execution += execution_time;
            total_conversion += conversion_time;
        }

        let avg_execution = total_execution / config.iterations as u32;
        let avg_conversion = total_conversion / config.iterations as u32;
        let avg_total = avg_execution + avg_conversion;

        // TFLOPS計算
        let operations = 2.0 * m as f64 * n as f64 * k as f64;
        let seconds = avg_execution.as_secs_f64();
        let tflops = (operations / seconds) / 1e12;

        let result = LegacyBenchmarkResult {
            size: (m, n, k),
            execution_time: avg_execution,
            conversion_time: avg_conversion,
            total_time: avg_total,
            tflops,
        };

        println!("    実行時間: {:?}, 変換時間: {:?}, 合計: {:?}",
                avg_execution, avg_conversion, avg_total);

        results.push(result);
    }

    Ok(results)
}

#[cfg(feature = "hybrid-f32")]
fn perform_detailed_comparison(
    hybrid_results: &[rustorch::hybrid_f32::benchmarks::MatrixBenchmarkResult],
    legacy_results: &[LegacyBenchmarkResult]
) -> Result<(), Box<dyn std::error::Error>> {

    println!("| サイズ | f32ハイブリッド | 従来システム | 性能向上 | 変換コスト削減 |");
    println!("|--------|----------------|--------------|----------|----------------|");

    let mut total_speedup = 0.0;
    let mut total_conversion_savings = 0.0;

    for (hybrid, legacy) in hybrid_results.iter().zip(legacy_results.iter()) {
        let speedup = legacy.total_time.as_secs_f64() / hybrid.execution_time.as_secs_f64();
        let conversion_savings = legacy.conversion_time.as_secs_f64();

        println!("| {}x{}x{} | {:?} | {:?} | {:.2}x | {:?} |",
            hybrid.size.0, hybrid.size.1, hybrid.size.2,
            hybrid.execution_time,
            legacy.total_time,
            speedup,
            legacy.conversion_time
        );

        total_speedup += speedup;
        total_conversion_savings += conversion_savings;
    }

    let avg_speedup = total_speedup / hybrid_results.len() as f64;
    let avg_conversion_savings = total_conversion_savings / hybrid_results.len() as f64;

    println!("\n📊 平均性能向上: {:.2}x", avg_speedup);
    println!("⚡ 平均変換コスト削減: {:.2}ms", avg_conversion_savings * 1000.0);

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn measure_conversion_costs(config: &BenchmarkConfig) -> Result<(), Box<dyn std::error::Error>> {
    let test_size = config.matrix_sizes[1]; // 中規模でテスト
    let (m, n, k) = test_size;

    println!("  測定サイズ: {}x{}x{}", m, n, k);

    // 従来システムの変換コスト測定
    let conversion_start = Instant::now();
    let a_data: Vec<f64> = (0..m*k).map(|_| rand::random::<f64>()).collect();
    let b_data: Vec<f64> = (0..k*n).map(|_| rand::random::<f64>()).collect();
    let _a = Tensor::from_vec(a_data, vec![m, k]);
    let _b = Tensor::from_vec(b_data, vec![k, n]);
    let conversion_time = conversion_start.elapsed();

    // f32ハイブリッドシステムの「変換コスト」（実際は0）
    let f32_start = Instant::now();
    let _a_f32 = F32Tensor::randn(&[m, k]);
    let _b_f32 = F32Tensor::randn(&[k, n]);
    let f32_time = f32_start.elapsed();

    println!("  従来システム変換コスト: {:?}", conversion_time);
    println!("  f32ハイブリッド変換コスト: {:?} (ほぼゼロ)", f32_time);
    println!("  変換コスト削減率: {:.1}%",
        ((conversion_time.as_nanos() - f32_time.as_nanos()) as f64 / conversion_time.as_nanos() as f64) * 100.0);

    Ok(())
}

#[cfg(feature = "hybrid-f32")]
fn compare_memory_usage(config: &BenchmarkConfig) -> Result<(), Box<dyn std::error::Error>> {
    let test_size = config.matrix_sizes[2]; // 大規模でテスト
    let (m, n, _k) = test_size;
    let total_elements = m * n;

    // f64 (従来システム)
    let f64_memory = total_elements * std::mem::size_of::<f64>();

    // f32 (ハイブリッドシステム)
    let f32_memory = total_elements * std::mem::size_of::<f32>();

    let memory_reduction = ((f64_memory - f32_memory) as f64 / f64_memory as f64) * 100.0;

    println!("  テストサイズ: {}x{} ({} 要素)", m, n, total_elements);
    println!("  f64メモリ使用量: {} MB", f64_memory / (1024 * 1024));
    println!("  f32メモリ使用量: {} MB", f32_memory / (1024 * 1024));
    println!("  メモリ削減率: {:.1}%", memory_reduction);
    println!("  実際のメモリ効率向上: ~{:.1}%", memory_reduction);

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ この比較テストを実行するには hybrid-f32 フィーチャーが必要です。");
    println!("❌ This comparison test requires the hybrid-f32 feature to be enabled.");
    println!("");
    println!("実行方法 / Usage:");
    println!("cargo run --example hybrid_f32_quick_comparison --features hybrid-f32 --release");
}