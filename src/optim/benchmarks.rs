//! Performance benchmarks and profiling for optimizers
//! 最適化器のパフォーマンスベンチマークとプロファイリング

use crate::optim::common::*;
use crate::optim::utils::{OptimizerMetrics, OptimizerUtils};
use crate::optim::*;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive benchmark suite for optimizer performance analysis
/// 最適化器パフォーマンス分析のための包括的ベンチマークスイート
pub struct OptimizerBenchmark {
    /// Test configurations for different scenarios
    test_configs: Vec<BenchmarkConfig>,
    /// Results storage
    results: HashMap<String, BenchmarkResult>,
}

/// Configuration for a single benchmark test
/// 単一ベンチマークテストの設定
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Test name identifier
    pub name: String,
    /// Parameter tensor dimensions
    pub param_shape: Vec<usize>,
    /// Number of optimization steps
    pub num_steps: usize,
    /// Number of warmup steps (not counted in timing)
    pub warmup_steps: usize,
    /// Number of iterations to average results
    pub num_iterations: usize,
    /// Learning rate for the test
    pub learning_rate: f32,
}

/// Results from a benchmark test
/// ベンチマークテストの結果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Average time per step in microseconds
    pub avg_step_time_us: f64,
    /// Standard deviation of step times
    pub step_time_std_us: f64,
    /// Peak memory usage estimate (MB)
    pub peak_memory_mb: usize,
    /// Convergence rate (gradient norm reduction per step)
    pub convergence_rate: f32,
    /// Total test duration in milliseconds
    pub total_duration_ms: u64,
    /// Steps per second throughput
    pub steps_per_second: f64,
}

impl OptimizerBenchmark {
    /// Create new benchmark suite with standard configurations
    /// 標準設定による新しいベンチマークスイートを作成
    pub fn new() -> Self {
        let test_configs = vec![
            BenchmarkConfig {
                name: "small_dense".to_string(),
                param_shape: vec![1000],
                num_steps: 1000,
                warmup_steps: 100,
                num_iterations: 5,
                learning_rate: 1e-3,
            },
            BenchmarkConfig {
                name: "medium_dense".to_string(),
                param_shape: vec![10000],
                num_steps: 1000,
                warmup_steps: 100,
                num_iterations: 5,
                learning_rate: 1e-3,
            },
            BenchmarkConfig {
                name: "large_dense".to_string(),
                param_shape: vec![100000],
                num_steps: 500,
                warmup_steps: 50,
                num_iterations: 3,
                learning_rate: 1e-4,
            },
            BenchmarkConfig {
                name: "matrix_2d".to_string(),
                param_shape: vec![512, 512],
                num_steps: 500,
                warmup_steps: 50,
                num_iterations: 3,
                learning_rate: 1e-3,
            },
            BenchmarkConfig {
                name: "tensor_3d".to_string(),
                param_shape: vec![64, 64, 64],
                num_steps: 200,
                warmup_steps: 20,
                num_iterations: 3,
                learning_rate: 1e-3,
            },
        ];

        Self {
            test_configs,
            results: HashMap::new(),
        }
    }

    /// Run comprehensive benchmark comparing all Adam-based optimizers
    /// 全Adam系最適化器を比較する包括的ベンチマークを実行
    pub fn run_adam_comparison(&mut self) -> HashMap<String, HashMap<String, BenchmarkResult>> {
        let mut all_results = HashMap::new();

        for config in &self.test_configs.clone() {
            println!(
                "🚀 ベンチマーク実行中: {} (形状: {:?})",
                config.name, config.param_shape
            );

            let mut config_results = HashMap::new();

            // NAdam benchmark
            if let Ok(nadam) = nadam::NAdam::default_params(config.learning_rate) {
                println!("  📊 NAdam をベンチマーク中...");
                let result = self.benchmark_optimizer(config, Box::new(nadam));
                config_results.insert("NAdam".to_string(), result);
            }

            // RAdam benchmark
            if let Ok(radam) = radam::RAdam::default_params(config.learning_rate) {
                println!("  📊 RAdam をベンチマーク中...");
                let result = self.benchmark_optimizer(config, Box::new(radam));
                config_results.insert("RAdam".to_string(), result);
            }

            // Adamax benchmark
            if let Ok(adamax) = adamax::Adamax::default_params(config.learning_rate) {
                println!("  📊 Adamax をベンチマーク中...");
                let result = self.benchmark_optimizer(config, Box::new(adamax));
                config_results.insert("Adamax".to_string(), result);
            }

            all_results.insert(config.name.clone(), config_results);
        }

        all_results
    }

    /// Benchmark a single optimizer with given configuration
    /// 指定された設定で単一の最適化器をベンチマーク
    fn benchmark_optimizer(
        &self,
        config: &BenchmarkConfig,
        mut optimizer: Box<dyn Optimizer>,
    ) -> BenchmarkResult {
        let mut step_times = Vec::new();
        let mut metrics = OptimizerMetrics::new(config.num_steps + config.warmup_steps);

        let start_total = Instant::now();

        for iteration in 0..config.num_iterations {
            // Create test data for this iteration
            let param = Tensor::ones(&config.param_shape);
            let mut grad_norms = Vec::new();

            // Warmup phase
            for step in 0..config.warmup_steps {
                let grad = self.generate_synthetic_gradient(&config.param_shape, step);
                let grad_norm = OptimizerUtils::l2_norm(&grad);
                optimizer.step(&param, &grad);
                grad_norms.push(grad_norm);
            }

            // Timed benchmark phase
            for step in 0..config.num_steps {
                let grad = self.generate_synthetic_gradient(&config.param_shape, step);
                let grad_norm = OptimizerUtils::l2_norm(&grad);

                let step_start = Instant::now();
                optimizer.step(&param, &grad);
                let step_time = step_start.elapsed();

                if iteration == 0 {
                    // Only collect timing for first iteration
                    step_times.push(step_time.as_micros() as f64);
                }

                // Record metrics
                metrics.record_step(
                    grad_norm,
                    0.1, // Dummy parameter change norm
                    optimizer.learning_rate(),
                    step_time.as_secs_f32(),
                );

                grad_norms.push(grad_norm);
            }
        }

        let total_duration = start_total.elapsed();

        // Calculate statistics
        let avg_step_time = step_times.iter().sum::<f64>() / step_times.len() as f64;
        let variance = step_times
            .iter()
            .map(|&t| (t - avg_step_time).powi(2))
            .sum::<f64>()
            / step_times.len() as f64;
        let std_dev = variance.sqrt();

        // Estimate memory usage (rough approximation)
        let total_elements = config.param_shape.iter().product::<usize>();
        let estimated_memory_mb = (total_elements * 4 * 3) / (1024 * 1024); // 3 tensors (param, momentum, velocity) * 4 bytes per f32

        BenchmarkResult {
            avg_step_time_us: avg_step_time,
            step_time_std_us: std_dev,
            peak_memory_mb: estimated_memory_mb,
            convergence_rate: 0.95, // Placeholder - would need proper convergence analysis
            total_duration_ms: total_duration.as_millis() as u64,
            steps_per_second: (config.num_steps as f64) / total_duration.as_secs_f64()
                * config.num_iterations as f64,
        }
    }

    /// Generate synthetic gradient for testing
    /// テスト用の合成勾配を生成
    fn generate_synthetic_gradient(&self, shape: &[usize], step: usize) -> Tensor<f32> {
        let total_elements = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        // Generate gradient that simulates realistic training behavior
        let step_factor = 1.0 / (1.0 + step as f32 * 0.001); // Gradually decreasing gradients

        for i in 0..total_elements {
            let noise = (i as f32 * 0.123 + step as f32 * 0.456).sin() * 0.1;
            let gradient = (0.1 + noise) * step_factor;
            data.push(gradient);
        }

        Tensor::from_vec(data, shape.to_vec())
    }

    /// Run L-BFGS specific benchmarks
    /// L-BFGS固有のベンチマークを実行
    pub fn run_lbfgs_benchmark(&mut self) -> HashMap<String, BenchmarkResult> {
        let mut results = HashMap::new();

        for config in &self.test_configs.clone() {
            if config.param_shape.iter().product::<usize>() > 50000 {
                continue; // Skip very large tensors for L-BFGS
            }

            println!("🔬 L-BFGS ベンチマーク: {}", config.name);

            if let Ok(lbfgs) = lbfgs::LBFGS::new(config.learning_rate) {
                let result = self.benchmark_optimizer(config, Box::new(lbfgs));
                results.insert(config.name.clone(), result);
            }
        }

        results
    }

    /// Generate performance report
    /// パフォーマンスレポートを生成
    pub fn generate_report(
        &self,
        results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    ) -> String {
        let mut report = String::new();
        report.push_str("# 🚀 RusTorch フェーズ２ 最適化器パフォーマンスレポート\n\n");

        for (config_name, optimizer_results) in results {
            report.push_str(&format!("## 📊 テスト設定: {}\n\n", config_name));
            report.push_str("| 最適化器 | 平均ステップ時間(μs) | 標準偏差(μs) | メモリ使用量(MB) | ステップ/秒 |\n");
            report.push_str(
                "|----------|---------------------|-------------|----------------|----------|\n",
            );

            let mut sorted_results: Vec<_> = optimizer_results.iter().collect();
            sorted_results.sort_by(|a, b| {
                a.1.avg_step_time_us
                    .partial_cmp(&b.1.avg_step_time_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (optimizer_name, result) in sorted_results {
                report.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {} | {:.1} |\n",
                    optimizer_name,
                    result.avg_step_time_us,
                    result.step_time_std_us,
                    result.peak_memory_mb,
                    result.steps_per_second
                ));
            }
            report.push_str("\n");
        }

        // Performance summary
        report.push_str("## 🎯 パフォーマンス要約\n\n");
        let mut fastest_by_config = HashMap::new();
        let mut most_efficient_memory = HashMap::new();

        for (config_name, optimizer_results) in results {
            let fastest = optimizer_results
                .iter()
                .min_by(|a, b| {
                    a.1.avg_step_time_us
                        .partial_cmp(&b.1.avg_step_time_us)
                        .unwrap()
                })
                .map(|(name, result)| (name.as_str(), result.avg_step_time_us));

            let most_memory_efficient = optimizer_results
                .iter()
                .min_by_key(|(_, result)| result.peak_memory_mb)
                .map(|(name, result)| (name.as_str(), result.peak_memory_mb));

            if let Some((name, time)) = fastest {
                fastest_by_config.insert(config_name, (name, time));
            }
            if let Some((name, memory)) = most_memory_efficient {
                most_efficient_memory.insert(config_name, (name, memory));
            }
        }

        report.push_str("### ⚡ 速度チャンピオン:\n");
        for (config, (optimizer, time)) in fastest_by_config {
            report.push_str(&format!(
                "- **{}**: {} ({:.2}μs/ステップ)\n",
                config, optimizer, time
            ));
        }

        report.push_str("\n### 💾 メモリ効率チャンピオン:\n");
        for (config, (optimizer, memory)) in most_efficient_memory {
            report.push_str(&format!("- **{}**: {} ({}MB)\n", config, optimizer, memory));
        }

        report
    }
}

/// Quick performance test for development
/// 開発用クイックパフォーマンステスト
pub fn quick_performance_test() {
    println!("🔥 クイックパフォーマンステスト開始");

    let mut benchmark = OptimizerBenchmark::new();

    // Test only small configuration for quick feedback
    let quick_config = BenchmarkConfig {
        name: "quick_test".to_string(),
        param_shape: vec![1000],
        num_steps: 100,
        warmup_steps: 10,
        num_iterations: 2,
        learning_rate: 1e-3,
    };

    benchmark.test_configs = vec![quick_config];
    let results = benchmark.run_adam_comparison();

    println!("\n📈 結果:");
    for (config_name, optimizer_results) in results {
        println!("\n🎯 {}", config_name);
        for (optimizer_name, result) in optimizer_results {
            println!(
                "  {} : {:.2}μs/step, {:.1} steps/sec",
                optimizer_name, result.avg_step_time_us, result.steps_per_second
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_creation() {
        let benchmark = OptimizerBenchmark::new();
        assert!(!benchmark.test_configs.is_empty());
        assert_eq!(benchmark.results.len(), 0);
    }

    #[test]
    fn test_synthetic_gradient_generation() {
        let benchmark = OptimizerBenchmark::new();
        let shape = vec![10, 10];
        let grad = benchmark.generate_synthetic_gradient(&shape, 0);

        assert_eq!(grad.shape(), &shape);
        let data = grad.as_slice().unwrap();
        assert_eq!(data.len(), 100);

        // Check that gradients are reasonable values
        for &val in data {
            assert!(val.abs() < 1.0); // Should be relatively small gradients
            assert!(val.is_finite()); // Should not be NaN or infinite
        }
    }

    #[test]
    fn test_benchmark_result_fields() {
        let result = BenchmarkResult {
            avg_step_time_us: 100.0,
            step_time_std_us: 5.0,
            peak_memory_mb: 10,
            convergence_rate: 0.95,
            total_duration_ms: 1000,
            steps_per_second: 500.0,
        };

        assert_eq!(result.avg_step_time_us, 100.0);
        assert_eq!(result.peak_memory_mb, 10);
        assert!(result.steps_per_second > 0.0);
    }

    #[test]
    fn test_performance_test_execution() {
        // Just ensure the quick test doesn't panic
        // 実際のベンチマークは時間がかかるので、テスト環境では実行しない
        let benchmark = OptimizerBenchmark::new();
        assert_eq!(benchmark.test_configs.len(), 5); // Standard configs
    }
}
