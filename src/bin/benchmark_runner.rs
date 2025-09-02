//! フェーズ２リファクタリング効果検証用ベンチマーク実行ツール
//! Benchmark runner to verify Phase 2 refactoring effectiveness

use rustorch::optim::benchmarks::OptimizerBenchmark;

fn main() {
    println!("🚀 RusTorch フェーズ２ リファクタリング効果検証");
    println!("Phase 2 Refactoring Effectiveness Verification");
    println!("{}", "=".repeat(60));
    
    let mut benchmark = OptimizerBenchmark::new();
    
    // 1. Adam系最適化器の比較ベンチマーク実行
    println!("\n📊 Adam系最適化器比較ベンチマーク実行中...");
    let results = benchmark.run_adam_comparison();
    
    // 2. L-BFGS専用ベンチマーク実行
    println!("\n🔬 L-BFGS専用ベンチマーク実行中...");
    let lbfgs_results = benchmark.run_lbfgs_benchmark();
    
    // 3. パフォーマンスレポート生成
    println!("\n📈 パフォーマンスレポート生成中...");
    let report = benchmark.generate_report(&results);
    println!("{}", report);
    
    // 4. L-BFGS結果追加表示
    if !lbfgs_results.is_empty() {
        println!("## 🔬 L-BFGS専用結果\n");
        for (config_name, result) in lbfgs_results {
            println!("**{}**: {:.2}μs/step, {:.1} steps/sec, {}MB memory", 
                    config_name, result.avg_step_time_us, result.steps_per_second, result.peak_memory_mb);
        }
        println!();
    }
    
    println!("\n✅ ベンチマーク完了 - Benchmark Completed");
    println!("🎉 リファクタリング効果検証完了！");
}