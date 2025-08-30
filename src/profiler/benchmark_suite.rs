//! Advanced Benchmarking Framework
//! 高度ベンチマークフレームワーク

use crate::error::{RusTorchError, RusTorchResult};
use crate::profiler::metrics_collector::{CustomMetric, MetricType, MetricsCollector};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark category for organization
/// 組織化のためのベンチマークカテゴリ
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchmarkCategory {
    /// Tensor operations
    /// テンソル操作
    TensorOps,
    /// Memory operations
    /// メモリ操作
    Memory,
    /// GPU operations
    /// GPU操作
    Gpu,
    /// Neural network operations
    /// ニューラルネットワーク操作
    NeuralNetwork,
    /// Linear algebra operations
    /// 線形代数操作
    LinearAlgebra,
    /// System performance
    /// システムパフォーマンス
    System,
    /// Custom category
    /// カスタムカテゴリ
    Custom(String),
}

/// Benchmark configuration
/// ベンチマーク設定
#[derive(Debug, Clone)]
pub struct BenchmarkConfiguration {
    /// Number of warmup iterations
    /// ウォームアップ反復回数
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    /// 測定反復回数  
    pub measurement_iterations: usize,
    /// Minimum benchmark duration (milliseconds)
    /// 最小ベンチマーク期間（ミリ秒）
    pub min_duration_ms: u64,
    /// Maximum benchmark duration (milliseconds)
    /// 最大ベンチマーク期間（ミリ秒）
    pub max_duration_ms: u64,
    /// Statistical confidence level (0.0 to 1.0)
    /// 統計的信頼水準（0.0から1.0）
    pub confidence_level: f64,
    /// Acceptable measurement variance threshold
    /// 許容測定分散閾値
    pub variance_threshold: f64,
    /// Enable detailed memory profiling
    /// 詳細メモリプロファイリングを有効化
    pub enable_memory_profiling: bool,
    /// Enable GPU profiling
    /// GPUプロファイリングを有効化
    pub enable_gpu_profiling: bool,
    /// Enable system metrics collection
    /// システムメトリクス収集を有効化
    pub enable_system_metrics: bool,
    /// Collect GC statistics
    /// GC統計を収集
    pub collect_gc_stats: bool,
}

impl Default for BenchmarkConfiguration {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            min_duration_ms: 1000,
            max_duration_ms: 60000,
            confidence_level: 0.95,
            variance_threshold: 0.1,
            enable_memory_profiling: true,
            enable_gpu_profiling: true,
            enable_system_metrics: true,
            collect_gc_stats: false,
        }
    }
}

/// Benchmark result with statistical analysis
/// 統計分析付きベンチマーク結果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    /// ベンチマーク名
    pub name: String,
    /// Category
    /// カテゴリ
    pub category: BenchmarkCategory,
    /// Configuration used
    /// 使用した設定
    pub config: BenchmarkConfiguration,
    /// All measured timings (ms)
    /// 全測定時間（ミリ秒）
    pub timings_ms: Vec<f64>,
    /// Statistical summary
    /// 統計サマリー
    pub statistics: BenchmarkStatistics,
    /// Memory metrics if enabled
    /// 有効な場合のメモリメトリクス
    pub memory_metrics: Option<MemoryBenchmarkMetrics>,
    /// GPU metrics if enabled
    /// 有効な場合のGPUメトリクス
    pub gpu_metrics: Option<GpuBenchmarkMetrics>,
    /// System metrics during benchmark
    /// ベンチマーク中のシステムメトリクス
    pub system_metrics: Option<SystemBenchmarkMetrics>,
    /// Error message if benchmark failed
    /// ベンチマーク失敗時のエラーメッセージ
    pub error: Option<String>,
    /// Benchmark execution timestamp
    /// ベンチマーク実行タイムスタンプ
    pub timestamp: Instant,
}

/// Statistical analysis of benchmark results
/// ベンチマーク結果の統計分析
#[derive(Debug, Clone)]
pub struct BenchmarkStatistics {
    /// Number of samples
    /// サンプル数
    pub sample_count: usize,
    /// Mean execution time (ms)
    /// 平均実行時間（ミリ秒）
    pub mean_ms: f64,
    /// Median execution time (ms)
    /// 中央値実行時間（ミリ秒）
    pub median_ms: f64,
    /// Standard deviation (ms)
    /// 標準偏差（ミリ秒）
    pub std_dev_ms: f64,
    /// Minimum time (ms)
    /// 最小時間（ミリ秒）
    pub min_ms: f64,
    /// Maximum time (ms)
    /// 最大時間（ミリ秒）
    pub max_ms: f64,
    /// 95th percentile (ms)
    /// 95パーセンタイル（ミリ秒）
    pub p95_ms: f64,
    /// 99th percentile (ms)
    /// 99パーセンタイル（ミリ秒）
    pub p99_ms: f64,
    /// Coefficient of variation
    /// 変動係数
    pub coefficient_of_variation: f64,
    /// Throughput (operations per second)
    /// スループット（秒間操作数）
    pub throughput_ops_per_sec: f64,
    /// Confidence interval (95%)
    /// 信頼区間（95%）
    pub confidence_interval_ms: (f64, f64),
    /// Whether results are statistically stable
    /// 結果が統計的に安定しているか
    pub is_stable: bool,
}

/// Memory benchmark metrics
/// メモリベンチマークメトリクス
#[derive(Debug, Clone)]
pub struct MemoryBenchmarkMetrics {
    /// Peak memory usage (bytes)
    /// ピークメモリ使用量（バイト）
    pub peak_memory_bytes: u64,
    /// Average memory usage (bytes)
    /// 平均メモリ使用量（バイト）
    pub avg_memory_bytes: u64,
    /// Memory allocations count
    /// メモリ割り当て数
    pub allocations: usize,
    /// Memory deallocations count
    /// メモリ解放数
    pub deallocations: usize,
    /// Total bytes allocated
    /// 総割り当てバイト数
    pub total_allocated_bytes: u64,
    /// Total bytes deallocated
    /// 総解放バイト数
    pub total_deallocated_bytes: u64,
    /// Memory fragmentation score (0.0 to 1.0)
    /// メモリ断片化スコア（0.0から1.0）
    pub fragmentation_score: f64,
}

/// GPU benchmark metrics
/// GPUベンチマークメトリクス
#[derive(Debug, Clone)]
pub struct GpuBenchmarkMetrics {
    /// GPU utilization percentage
    /// GPU使用率
    pub gpu_utilization_percent: f64,
    /// Memory utilization percentage
    /// メモリ使用率
    pub memory_utilization_percent: f64,
    /// GPU memory used (bytes)
    /// GPU使用メモリ（バイト）
    pub gpu_memory_used_bytes: u64,
    /// Number of kernel launches
    /// カーネル起動数
    pub kernel_launches: usize,
    /// Total kernel execution time (ms)
    /// 総カーネル実行時間（ミリ秒）
    pub total_kernel_time_ms: f64,
    /// Memory transfer time (ms)
    /// メモリ転送時間（ミリ秒）
    pub memory_transfer_time_ms: f64,
    /// GPU temperature (Celsius)
    /// GPU温度（摂氏）
    pub gpu_temperature_celsius: Option<f32>,
    /// Power consumption (watts)
    /// 消費電力（ワット）
    pub power_consumption_watts: Option<f32>,
}

/// System benchmark metrics
/// システムベンチマークメトリクス
#[derive(Debug, Clone)]
pub struct SystemBenchmarkMetrics {
    /// CPU utilization percentage
    /// CPU使用率
    pub cpu_utilization_percent: f64,
    /// System memory usage (bytes)
    /// システムメモリ使用量（バイト）
    pub system_memory_bytes: u64,
    /// Disk I/O operations
    /// ディスクI/O操作
    pub disk_io_operations: usize,
    /// Network I/O bytes
    /// ネットワークI/Oバイト
    pub network_io_bytes: u64,
    /// System load average
    /// システム負荷平均
    pub load_average: f64,
    /// Context switches
    /// コンテキストスイッチ
    pub context_switches: usize,
}

/// Advanced benchmark suite
/// 高度ベンチマークスイート
#[derive(Debug)]
pub struct AdvancedBenchmarkSuite {
    /// Suite name
    /// スイート名
    pub name: String,
    /// Default configuration
    /// デフォルト設定
    pub default_config: BenchmarkConfiguration,
    /// Metrics collector
    /// メトリクス収集器
    metrics_collector: MetricsCollector,
    /// Benchmark results
    /// ベンチマーク結果
    results: HashMap<String, BenchmarkResult>,
    /// Suite execution metadata
    /// スイート実行メタデータ
    pub suite_metadata: SuiteMetadata,
}

/// Suite execution metadata
/// スイート実行メタデータ
#[derive(Debug, Clone)]
pub struct SuiteMetadata {
    /// Total execution time
    /// 総実行時間
    pub total_execution_time: Duration,
    /// Number of benchmarks run
    /// 実行したベンチマーク数
    pub benchmarks_run: usize,
    /// Number of benchmarks failed
    /// 失敗したベンチマーク数
    pub benchmarks_failed: usize,
    /// System information at start
    /// 開始時システム情報
    pub system_info: SystemInfo,
}

/// System information
/// システム情報
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// CPU model
    /// CPUモデル
    pub cpu_model: String,
    /// Number of CPU cores
    /// CPUコア数
    pub cpu_cores: usize,
    /// Total system memory (bytes)
    /// 総システムメモリ（バイト）
    pub total_memory_bytes: u64,
    /// Operating system
    /// オペレーティングシステム
    pub os_version: String,
    /// Rust version
    /// Rustバージョン
    pub rust_version: String,
    /// GPU information if available
    /// GPU情報（利用可能な場合）
    pub gpu_info: Option<String>,
}

impl AdvancedBenchmarkSuite {
    /// Create new benchmark suite
    /// 新しいベンチマークスイートを作成
    pub fn new(name: String) -> Self {
        Self {
            name,
            default_config: BenchmarkConfiguration::default(),
            metrics_collector: MetricsCollector::new(),
            results: HashMap::new(),
            suite_metadata: SuiteMetadata {
                total_execution_time: Duration::ZERO,
                benchmarks_run: 0,
                benchmarks_failed: 0,
                system_info: Self::collect_system_info(),
            },
        }
    }

    /// Set default benchmark configuration
    /// デフォルトベンチマーク設定を設定
    pub fn with_config(mut self, config: BenchmarkConfiguration) -> Self {
        self.default_config = config;
        self
    }

    /// Run a benchmark with custom configuration
    /// カスタム設定でベンチマークを実行
    pub fn benchmark<F, T>(&mut self, name: &str, category: BenchmarkCategory, config: Option<BenchmarkConfiguration>, mut operation: F) -> RusTorchResult<()>
    where
        F: FnMut() -> RusTorchResult<T>,
    {
        let config = config.unwrap_or_else(|| self.default_config.clone());
        let start_time = Instant::now();

        println!("🏁 Running benchmark: {}", name);

        // Initialize metrics collection
        let timing_metric = CustomMetric::new(format!("{}_timing", name), MetricType::TimingMs);
        self.metrics_collector.register_metric(timing_metric)?;

        if config.enable_memory_profiling {
            let memory_metric = CustomMetric::new(format!("{}_memory", name), MetricType::MemoryBytes);
            self.metrics_collector.register_metric(memory_metric)?;
        }

        // Warmup phase
        println!("  🔥 Warmup phase ({} iterations)...", config.warmup_iterations);
        for _ in 0..config.warmup_iterations {
            let _ = operation(); // Ignore warmup results
        }

        // Measurement phase
        println!("  📊 Measurement phase ({} iterations)...", config.measurement_iterations);
        let mut timings = Vec::with_capacity(config.measurement_iterations);
        let mut failed_iterations = 0;

        for i in 0..config.measurement_iterations {
            let iteration_start = Instant::now();
            
            match operation() {
                Ok(_) => {
                    let elapsed = iteration_start.elapsed();
                    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
                    timings.push(elapsed_ms);

                    // Record timing metric
                    self.metrics_collector.record_timing(&format!("{}_timing", name), elapsed)?;

                    // Collect memory metrics if enabled
                    if config.enable_memory_profiling {
                        // In a real implementation, this would collect actual memory usage
                        self.metrics_collector.update_metric(&format!("{}_memory", name), 0.0)?;
                    }
                }
                Err(e) => {
                    failed_iterations += 1;
                    println!("    ❌ Iteration {} failed: {}", i + 1, e);
                }
            }

            // Progress indicator
            if (i + 1) % (config.measurement_iterations / 10).max(1) == 0 {
                let progress = ((i + 1) as f64 / config.measurement_iterations as f64) * 100.0;
                println!("    Progress: {:.1}%", progress);
            }
        }

        // Check if we have enough successful samples
        if timings.len() < config.measurement_iterations / 2 {
            let error_msg = format!("Too many failed iterations: {}/{}", failed_iterations, config.measurement_iterations);
            self.results.insert(name.to_string(), BenchmarkResult {
                name: name.to_string(),
                category,
                config,
                timings_ms: Vec::new(),
                statistics: BenchmarkStatistics::default(),
                memory_metrics: None,
                gpu_metrics: None,
                system_metrics: None,
                error: Some(error_msg.clone()),
                timestamp: start_time,
            });
            self.suite_metadata.benchmarks_failed += 1;
            return Err(RusTorchError::Profiling { message: error_msg });
        }

        // Calculate statistics
        let statistics = Self::calculate_statistics(&timings, &config);
        
        // Collect additional metrics
        let memory_metrics = if config.enable_memory_profiling {
            Some(self.collect_memory_metrics(name)?)
        } else {
            None
        };

        let gpu_metrics = if config.enable_gpu_profiling {
            Some(self.collect_gpu_metrics(name)?)
        } else {
            None
        };

        let system_metrics = if config.enable_system_metrics {
            Some(self.collect_system_metrics()?)
        } else {
            None
        };

        // Create benchmark result
        let result = BenchmarkResult {
            name: name.to_string(),
            category,
            config,
            timings_ms: timings,
            statistics,
            memory_metrics,
            gpu_metrics,
            system_metrics,
            error: None,
            timestamp: start_time,
        };

        // Store result
        self.results.insert(name.to_string(), result);
        self.suite_metadata.benchmarks_run += 1;

        let total_time = start_time.elapsed();
        println!("  ✅ Benchmark completed in {:.2}s", total_time.as_secs_f64());
        println!("     Mean: {:.3}ms, Median: {:.3}ms, StdDev: {:.3}ms", 
                 self.results[name].statistics.mean_ms,
                 self.results[name].statistics.median_ms,
                 self.results[name].statistics.std_dev_ms);

        Ok(())
    }

    /// Run benchmark with default configuration
    /// デフォルト設定でベンチマークを実行
    pub fn benchmark_default<F, T>(&mut self, name: &str, category: BenchmarkCategory, operation: F) -> RusTorchResult<()>
    where
        F: FnMut() -> RusTorchResult<T>,
    {
        self.benchmark(name, category, None, operation)
    }

    /// Get benchmark result
    /// ベンチマーク結果を取得
    pub fn get_result(&self, name: &str) -> Option<&BenchmarkResult> {
        self.results.get(name)
    }

    /// Get all results
    /// 全結果を取得
    pub fn get_all_results(&self) -> &HashMap<String, BenchmarkResult> {
        &self.results
    }

    /// Get results by category
    /// カテゴリ別結果を取得
    pub fn get_results_by_category(&self, category: &BenchmarkCategory) -> Vec<&BenchmarkResult> {
        self.results.values()
            .filter(|result| &result.category == category)
            .collect()
    }

    /// Generate comprehensive report
    /// 包括的レポートを生成
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("📊 Benchmark Suite Report: {}\n", self.name));
        report.push_str(&format!("{}", "=".repeat(50)));
        report.push_str("\n\n");

        // Suite metadata
        report.push_str("🏆 Suite Summary:\n");
        report.push_str(&format!("  Benchmarks Run: {}\n", self.suite_metadata.benchmarks_run));
        report.push_str(&format!("  Benchmarks Failed: {}\n", self.suite_metadata.benchmarks_failed));
        report.push_str(&format!("  Success Rate: {:.1}%\n", 
            if self.suite_metadata.benchmarks_run > 0 {
                (self.suite_metadata.benchmarks_run - self.suite_metadata.benchmarks_failed) as f64 
                / self.suite_metadata.benchmarks_run as f64 * 100.0
            } else {
                0.0
            }));
        report.push_str(&format!("  Total Execution Time: {:.2}s\n\n", 
                                self.suite_metadata.total_execution_time.as_secs_f64()));

        // System information
        report.push_str("💻 System Information:\n");
        report.push_str(&format!("  CPU: {}\n", self.suite_metadata.system_info.cpu_model));
        report.push_str(&format!("  Cores: {}\n", self.suite_metadata.system_info.cpu_cores));
        report.push_str(&format!("  Memory: {:.2} GB\n", 
                                self.suite_metadata.system_info.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)));
        report.push_str(&format!("  OS: {}\n", self.suite_metadata.system_info.os_version));
        if let Some(ref gpu_info) = self.suite_metadata.system_info.gpu_info {
            report.push_str(&format!("  GPU: {}\n", gpu_info));
        }
        report.push_str("\n");

        // Results by category
        let categories: std::collections::HashSet<_> = self.results.values().map(|r| &r.category).collect();
        for category in categories {
            let category_results = self.get_results_by_category(category);
            if !category_results.is_empty() {
                report.push_str(&format!("📈 {:?} Results:\n", category));
                report.push_str(&format!("{:<30} {:>10} {:>10} {:>10} {:>10} {:>15}\n", 
                                "Benchmark", "Mean(ms)", "Median(ms)", "StdDev(ms)", "P99(ms)", "Throughput(ops/s)"));
                report.push_str(&"-".repeat(100));
                report.push_str("\n");

                for result in category_results {
                    if result.error.is_none() {
                        report.push_str(&format!("{:<30} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>15.2}\n",
                            if result.name.len() > 29 { &result.name[..29] } else { &result.name },
                            result.statistics.mean_ms,
                            result.statistics.median_ms,
                            result.statistics.std_dev_ms,
                            result.statistics.p99_ms,
                            result.statistics.throughput_ops_per_sec));
                    } else {
                        report.push_str(&format!("{:<30} {:>50}\n", result.name, "❌ FAILED"));
                    }
                }
                report.push_str("\n");
            }
        }

        // Performance insights
        report.push_str("💡 Performance Insights:\n");
        self.generate_insights(&mut report);

        report
    }

    /// Export results to JSON
    /// 結果をJSONにエクスポート
    pub fn export_json(&self) -> RusTorchResult<String> {
        // In a real implementation, this would use serde_json
        // For now, return a placeholder
        Ok("{}".to_string())
    }

    /// Clear all results
    /// 全結果をクリア
    pub fn clear_results(&mut self) {
        self.results.clear();
        self.suite_metadata.benchmarks_run = 0;
        self.suite_metadata.benchmarks_failed = 0;
        let _ = self.metrics_collector.clear_metrics();
    }

    // Private helper methods

    fn calculate_statistics(timings: &[f64], config: &BenchmarkConfiguration) -> BenchmarkStatistics {
        if timings.is_empty() {
            return BenchmarkStatistics::default();
        }

        let mut sorted_timings = timings.to_vec();
        sorted_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sample_count = timings.len();
        let sum: f64 = timings.iter().sum();
        let mean_ms = sum / sample_count as f64;
        
        let median_ms = if sample_count % 2 == 0 {
            (sorted_timings[sample_count / 2 - 1] + sorted_timings[sample_count / 2]) / 2.0
        } else {
            sorted_timings[sample_count / 2]
        };

        let variance = timings.iter()
            .map(|&t| (t - mean_ms).powi(2))
            .sum::<f64>() / sample_count as f64;
        let std_dev_ms = variance.sqrt();

        let min_ms = sorted_timings[0];
        let max_ms = sorted_timings[sample_count - 1];

        let p95_index = ((sample_count as f64) * 0.95) as usize;
        let p95_ms = sorted_timings[p95_index.min(sample_count - 1)];

        let p99_index = ((sample_count as f64) * 0.99) as usize;
        let p99_ms = sorted_timings[p99_index.min(sample_count - 1)];

        let coefficient_of_variation = if mean_ms > 0.0 { std_dev_ms / mean_ms } else { 0.0 };
        let throughput_ops_per_sec = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };

        // Confidence interval (95% using t-distribution approximation)
        let t_value = 1.96; // Approximate for large samples
        let margin_of_error = t_value * std_dev_ms / (sample_count as f64).sqrt();
        let confidence_interval_ms = (mean_ms - margin_of_error, mean_ms + margin_of_error);

        let is_stable = coefficient_of_variation <= config.variance_threshold;

        BenchmarkStatistics {
            sample_count,
            mean_ms,
            median_ms,
            std_dev_ms,
            min_ms,
            max_ms,
            p95_ms,
            p99_ms,
            coefficient_of_variation,
            throughput_ops_per_sec,
            confidence_interval_ms,
            is_stable,
        }
    }

    fn collect_memory_metrics(&self, _name: &str) -> RusTorchResult<MemoryBenchmarkMetrics> {
        // Placeholder implementation - in production would collect real memory metrics
        Ok(MemoryBenchmarkMetrics {
            peak_memory_bytes: 0,
            avg_memory_bytes: 0,
            allocations: 0,
            deallocations: 0,
            total_allocated_bytes: 0,
            total_deallocated_bytes: 0,
            fragmentation_score: 0.0,
        })
    }

    fn collect_gpu_metrics(&self, _name: &str) -> RusTorchResult<GpuBenchmarkMetrics> {
        // Placeholder implementation - in production would collect real GPU metrics
        Ok(GpuBenchmarkMetrics {
            gpu_utilization_percent: 0.0,
            memory_utilization_percent: 0.0,
            gpu_memory_used_bytes: 0,
            kernel_launches: 0,
            total_kernel_time_ms: 0.0,
            memory_transfer_time_ms: 0.0,
            gpu_temperature_celsius: None,
            power_consumption_watts: None,
        })
    }

    fn collect_system_metrics(&self) -> RusTorchResult<SystemBenchmarkMetrics> {
        // Placeholder implementation - in production would collect real system metrics
        Ok(SystemBenchmarkMetrics {
            cpu_utilization_percent: 0.0,
            system_memory_bytes: 0,
            disk_io_operations: 0,
            network_io_bytes: 0,
            load_average: 0.0,
            context_switches: 0,
        })
    }

    fn collect_system_info() -> SystemInfo {
        SystemInfo {
            cpu_model: "Unknown CPU".to_string(),
            cpu_cores: num_cpus::get(),
            total_memory_bytes: 0, // Would need system APIs to get real value
            os_version: std::env::consts::OS.to_string(),
            rust_version: "Rust 1.70+".to_string(), // Static version for compatibility
            gpu_info: None,
        }
    }

    fn generate_insights(&self, report: &mut String) {
        let successful_results: Vec<_> = self.results.values()
            .filter(|r| r.error.is_none())
            .collect();

        if successful_results.is_empty() {
            report.push_str("  No successful benchmarks to analyze.\n\n");
            return;
        }

        // Find fastest and slowest benchmarks
        if let (Some(fastest), Some(slowest)) = (
            successful_results.iter().min_by(|a, b| a.statistics.mean_ms.partial_cmp(&b.statistics.mean_ms).unwrap()),
            successful_results.iter().max_by(|a, b| a.statistics.mean_ms.partial_cmp(&b.statistics.mean_ms).unwrap()),
        ) {
            report.push_str(&format!("  🚀 Fastest: {} ({:.3}ms)\n", fastest.name, fastest.statistics.mean_ms));
            report.push_str(&format!("  🐌 Slowest: {} ({:.3}ms)\n", slowest.name, slowest.statistics.mean_ms));
            
            if fastest.statistics.mean_ms > 0.0 {
                let speedup = slowest.statistics.mean_ms / fastest.statistics.mean_ms;
                report.push_str(&format!("  📊 Performance Range: {:.1}x difference\n", speedup));
            }
        }

        // Stability analysis
        let unstable_count = successful_results.iter()
            .filter(|r| !r.statistics.is_stable)
            .count();
        
        if unstable_count > 0 {
            report.push_str(&format!("  ⚠️  {} benchmarks show high variance (>{}%)\n", 
                                   unstable_count, 
                                   self.default_config.variance_threshold * 100.0));
        }

        report.push_str("\n");
    }
}

impl Default for BenchmarkStatistics {
    fn default() -> Self {
        Self {
            sample_count: 0,
            mean_ms: 0.0,
            median_ms: 0.0,
            std_dev_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            coefficient_of_variation: 0.0,
            throughput_ops_per_sec: 0.0,
            confidence_interval_ms: (0.0, 0.0),
            is_stable: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = AdvancedBenchmarkSuite::new("test_suite".to_string());
        assert_eq!(suite.name, "test_suite");
        assert_eq!(suite.results.len(), 0);
    }

    #[test]
    fn test_simple_benchmark() {
        let mut suite = AdvancedBenchmarkSuite::new("test".to_string());
        
        let config = BenchmarkConfiguration {
            warmup_iterations: 2,
            measurement_iterations: 5,
            ..Default::default()
        };

        let result = suite.benchmark(
            "sleep_test",
            BenchmarkCategory::System,
            Some(config),
            || -> RusTorchResult<()> {
                thread::sleep(Duration::from_millis(10));
                Ok(())
            }
        );

        assert!(result.is_ok());
        
        let benchmark_result = suite.get_result("sleep_test").unwrap();
        assert_eq!(benchmark_result.name, "sleep_test");
        assert!(benchmark_result.error.is_none());
        assert!(benchmark_result.statistics.mean_ms >= 10.0);
        assert_eq!(benchmark_result.statistics.sample_count, 5);
    }

    #[test]
    fn test_benchmark_statistics() {
        let timings = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5];
        let config = BenchmarkConfiguration::default();
        let stats = AdvancedBenchmarkSuite::calculate_statistics(&timings, &config);

        assert_eq!(stats.sample_count, 7);
        assert!((stats.mean_ms - 11.5).abs() < 0.1);
        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.max_ms, 13.0);
        assert!(stats.std_dev_ms > 0.0);
    }

    #[test]
    fn test_benchmark_categories() {
        let mut suite = AdvancedBenchmarkSuite::new("category_test".to_string());
        
        let config = BenchmarkConfiguration {
            warmup_iterations: 1,
            measurement_iterations: 2,
            ..Default::default()
        };

        // Add benchmarks in different categories
        suite.benchmark("tensor_op", BenchmarkCategory::TensorOps, Some(config.clone()), || Ok(())).unwrap();
        suite.benchmark("memory_op", BenchmarkCategory::Memory, Some(config), || Ok(())).unwrap();

        let tensor_results = suite.get_results_by_category(&BenchmarkCategory::TensorOps);
        let memory_results = suite.get_results_by_category(&BenchmarkCategory::Memory);

        assert_eq!(tensor_results.len(), 1);
        assert_eq!(memory_results.len(), 1);
        assert_eq!(tensor_results[0].name, "tensor_op");
        assert_eq!(memory_results[0].name, "memory_op");
    }

    #[test]
    fn test_failed_benchmark() {
        let mut suite = AdvancedBenchmarkSuite::new("fail_test".to_string());
        
        let config = BenchmarkConfiguration {
            warmup_iterations: 1,
            measurement_iterations: 3,
            ..Default::default()
        };

        let result = suite.benchmark(
            "failing_test",
            BenchmarkCategory::System,
            Some(config),
            || -> RusTorchResult<()> {
                Err(RusTorchError::Profiling { message: "Intentional failure".to_string() })
            }
        );

        assert!(result.is_err());
        
        let benchmark_result = suite.get_result("failing_test").unwrap();
        assert!(benchmark_result.error.is_some());
        assert_eq!(suite.suite_metadata.benchmarks_failed, 1);
    }

    #[test]
    fn test_report_generation() {
        let mut suite = AdvancedBenchmarkSuite::new("report_test".to_string());
        
        let config = BenchmarkConfiguration {
            warmup_iterations: 1,
            measurement_iterations: 2,
            ..Default::default()
        };

        suite.benchmark("test1", BenchmarkCategory::System, Some(config), || Ok(())).unwrap();
        
        let report = suite.generate_report();
        assert!(report.contains("Benchmark Suite Report"));
        assert!(report.contains("test1"));
        assert!(report.contains("System Information"));
    }
}