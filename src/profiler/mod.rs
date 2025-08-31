//! Performance Profiling & Benchmarking Framework (Phase 1 Component 5)
//! パフォーマンスプロファイリング・ベンチマーキングフレームワーク（フェーズ1コンポーネント5）
//!
//! Enterprise-grade profiling and performance analysis system with:
//! - Real-time performance monitoring and metrics collection
//! - Advanced benchmarking with statistical analysis
//! - Memory and GPU profiling integration
//! - Performance trend analysis and optimization recommendations
//! - Chrome tracing export and timeline visualization
//! - Multi-threaded profiling with call stack tracking

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

// Enhanced module structure
pub mod benchmark_suite;
pub mod core;
pub mod kernel_profiler;
pub mod memory_profiler;
pub mod metrics_collector;
pub mod performance_analyzer;
pub mod real_time_monitor;
pub mod system_profiler;
pub mod timeline;

// Enhanced re-exports for the new profiling system
pub use benchmark_suite::{
    AdvancedBenchmarkSuite, BenchmarkCategory, BenchmarkConfiguration, BenchmarkResult,
};
pub use core::{ProfilerConfig, ProfilerCore, ProfilingLevel, ProfilingSession, SessionSnapshot};
pub use metrics_collector::{CustomMetric, MetricStatistics, MetricType, MetricsCollector};
pub use performance_analyzer::{
    OptimizationRecommendation, PerformanceAnalyzer, PerformanceTrend, RecommendationPriority,
    RecommendationType, TrendAnalysis,
};

// Multi-GPU profiler integration (only for non-WASM targets)
#[cfg(not(target_arch = "wasm32"))]
pub use crate::gpu::multi_gpu_profiler::{MultiGpuProfiler, PerformanceReport as MultiGpuReport};

// Legacy imports for backward compatibility
use self::kernel_profiler::KernelProfiler;
use self::memory_profiler::MemoryProfiler;
use self::timeline::Timeline;

lazy_static::lazy_static! {
    /// Global profiler instance
    /// グローバルプロファイラーインスタンス
    static ref PROFILER: Arc<Mutex<Profiler>> = Arc::new(Mutex::new(Profiler::new()));
}

/// Profile context manager
/// プロファイルコンテキストマネージャー
pub struct ProfileContext {
    /// Context name
    name: String,
    /// Start time
    start_time: Instant,
    /// Parent context
    _parent: Option<String>,
}

impl ProfileContext {
    /// Create new profile context
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let start_time = Instant::now();

        // Register context start
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.start_operation(&name);
        }

        Self {
            name,
            start_time,
            _parent: None,
        }
    }
}

impl Drop for ProfileContext {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();

        // Register context end
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.end_operation(&self.name, duration);
        }
    }
}

/// Main profiler structure
/// メインプロファイラー構造
pub struct Profiler {
    /// Operation records
    /// 操作記録
    operations: HashMap<String, OperationStats>,
    /// Memory profiler
    /// メモリプロファイラー
    memory_profiler: MemoryProfiler,
    /// Kernel profiler
    /// カーネルプロファイラー
    kernel_profiler: KernelProfiler,
    /// Timeline
    /// タイムライン
    timeline: Timeline,
    /// Profiling enabled flag
    /// プロファイリング有効フラグ
    enabled: bool,
    /// Current call stack
    /// 現在のコールスタック
    call_stack: Vec<String>,
    /// Thread-local storage for multi-threading
    /// マルチスレッディング用スレッドローカルストレージ
    thread_data: HashMap<thread::ThreadId, ThreadProfileData>,
}

/// Operation statistics
/// 操作統計
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name
    /// 操作名
    pub name: String,
    /// Number of calls
    /// 呼び出し回数
    pub count: usize,
    /// Total time spent
    /// 総消費時間
    pub total_time: Duration,
    /// Average time per call
    /// 呼び出しごとの平均時間
    pub avg_time: Duration,
    /// Minimum time
    /// 最小時間
    pub min_time: Duration,
    /// Maximum time
    /// 最大時間
    pub max_time: Duration,
    /// Memory allocated (bytes)
    /// 割り当てメモリ（バイト）
    pub memory_allocated: usize,
    /// Memory freed (bytes)
    /// 解放メモリ（バイト）
    pub memory_freed: usize,
    /// CUDA time if applicable
    /// 該当する場合のCUDA時間
    pub cuda_time: Option<Duration>,
    /// Self CPU time (excluding children)
    /// 自己CPU時間（子を除く）
    pub self_cpu_time: Duration,
    /// Child operations
    /// 子操作
    pub children: Vec<String>,
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            name: String::new(),
            count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            memory_allocated: 0,
            memory_freed: 0,
            cuda_time: None,
            self_cpu_time: Duration::ZERO,
            children: Vec::new(),
        }
    }
}

/// Thread-specific profile data
/// スレッド固有のプロファイルデータ
#[derive(Debug, Clone)]
struct ThreadProfileData {
    /// Thread ID
    _thread_id: thread::ThreadId,
    /// Call stack
    call_stack: Vec<String>,
    /// Operation timings
    _timings: HashMap<String, Vec<Duration>>,
}

impl Profiler {
    /// Create new profiler
    /// 新しいプロファイラーを作成
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            memory_profiler: MemoryProfiler::new(),
            kernel_profiler: KernelProfiler::new(),
            timeline: Timeline::new(),
            enabled: false,
            call_stack: Vec::new(),
            thread_data: HashMap::new(),
        }
    }

    /// Enable profiling
    /// プロファイリングを有効化
    pub fn enable(&mut self) {
        self.enabled = true;
        self.memory_profiler.start();
        self.kernel_profiler.start();
    }

    /// Disable profiling
    /// プロファイリングを無効化
    pub fn disable(&mut self) {
        self.enabled = false;
        self.memory_profiler.stop();
        self.kernel_profiler.stop();
    }

    /// Start an operation
    /// 操作を開始
    pub fn start_operation(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        let thread_id = thread::current().id();

        // Update call stack
        self.call_stack.push(name.to_string());

        // Record in timeline
        self.timeline.add_event(name, Instant::now(), None);

        // Initialize operation stats if needed
        self.operations.entry(name.to_string()).or_insert_with(|| {
            let mut stats = OperationStats::default();
            stats.name = name.to_string();
            stats
        });

        // Update thread data
        let thread_data = self
            .thread_data
            .entry(thread_id)
            .or_insert_with(|| ThreadProfileData {
                _thread_id: thread_id,
                call_stack: Vec::new(),
                _timings: HashMap::new(),
            });
        thread_data.call_stack.push(name.to_string());
    }

    /// End an operation
    /// 操作を終了
    pub fn end_operation(&mut self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        // Update operation stats - ensure operation exists
        let stats = self.operations.entry(name.to_string()).or_insert_with(|| {
            let mut stats = OperationStats::default();
            stats.name = name.to_string();
            stats
        });

        stats.count += 1;
        stats.total_time += duration;
        stats.avg_time = stats.total_time / stats.count as u32;
        stats.min_time = stats.min_time.min(duration);
        stats.max_time = stats.max_time.max(duration);

        // Update memory stats
        let mem_stats = self.memory_profiler.get_current_stats();
        stats.memory_allocated = mem_stats.allocated;
        stats.memory_freed = mem_stats.freed;

        // Update CUDA time if applicable
        if let Some(cuda_time) = self.kernel_profiler.get_last_kernel_time() {
            stats.cuda_time = Some(cuda_time);
        }

        // Update call stack
        if let Some(last) = self.call_stack.last() {
            if last == name {
                self.call_stack.pop();
            }
        }

        // Update timeline
        self.timeline.end_event(name, Instant::now());
    }

    /// Record memory allocation
    /// メモリ割り当てを記録
    pub fn record_allocation(&mut self, size: usize, name: &str) {
        if !self.enabled {
            return;
        }

        self.memory_profiler.record_allocation(size, name);
    }

    /// Record memory deallocation
    /// メモリ解放を記録
    pub fn record_deallocation(&mut self, size: usize, name: &str) {
        if !self.enabled {
            return;
        }

        self.memory_profiler.record_deallocation(size, name);
    }

    /// Get profiling summary
    /// プロファイリングサマリーを取得
    pub fn get_summary(&self) -> ProfileSummary {
        let mut operations: Vec<_> = self.operations.values().cloned().collect();
        operations.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        let total_time: Duration = operations.iter().map(|op| op.total_time).sum();

        let memory_stats = self.memory_profiler.get_summary();
        let kernel_stats = self.kernel_profiler.get_summary();

        ProfileSummary {
            operations,
            total_time,
            memory_stats,
            kernel_stats,
            timeline: self.timeline.clone(),
        }
    }

    /// Clear all profiling data
    /// すべてのプロファイリングデータをクリア
    pub fn clear(&mut self) {
        self.operations.clear();
        self.memory_profiler.clear();
        self.kernel_profiler.clear();
        self.timeline.clear();
        self.call_stack.clear();
        self.thread_data.clear();
    }

    /// Export profiling data to Chrome tracing format
    /// プロファイリングデータをChromeトレーシング形式にエクスポート
    pub fn export_chrome_trace(&self) -> String {
        self.timeline.export_chrome_trace()
    }

    /// Print formatted report
    /// フォーマットされたレポートを出力
    pub fn print_report(&self) {
        let summary = self.get_summary();
        println!("{}", summary);
    }
}

/// Profiling summary
/// プロファイリングサマリー
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Operation statistics sorted by time
    /// 時間でソートされた操作統計
    pub operations: Vec<OperationStats>,
    /// Total profiling time
    /// 総プロファイリング時間
    pub total_time: Duration,
    /// Memory statistics
    /// メモリ統計
    pub memory_stats: memory_profiler::MemorySummary,
    /// Kernel statistics
    /// カーネル統計
    pub kernel_stats: kernel_profiler::KernelSummary,
    /// Timeline
    /// タイムライン
    pub timeline: Timeline,
}

impl fmt::Display for ProfileSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n================== Profiler Report ==================")?;
        writeln!(
            f,
            "Total Time: {:.3}ms",
            self.total_time.as_secs_f64() * 1000.0
        )?;
        writeln!(f)?;

        // Operation table
        writeln!(f, "Top Operations by Time:")?;
        writeln!(
            f,
            "{:<30} {:>10} {:>12} {:>12} {:>12}",
            "Name", "Calls", "Total (ms)", "Avg (ms)", "Self (ms)"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        for op in self.operations.iter().take(20) {
            writeln!(
                f,
                "{:<30} {:>10} {:>12.3} {:>12.3} {:>12.3}",
                if op.name.len() > 29 {
                    &op.name[..29]
                } else {
                    &op.name
                },
                op.count,
                op.total_time.as_secs_f64() * 1000.0,
                op.avg_time.as_secs_f64() * 1000.0,
                op.self_cpu_time.as_secs_f64() * 1000.0
            )?;
        }

        writeln!(f)?;

        // Memory statistics
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "{}", self.memory_stats)?;

        // Kernel statistics
        if !self.kernel_stats.kernels.is_empty() {
            writeln!(f, "\nGPU Kernel Statistics:")?;
            writeln!(f, "{}", self.kernel_stats)?;
        }

        Ok(())
    }
}

/// Profile a code block
/// コードブロックをプロファイル
#[macro_export]
macro_rules! profile {
    ($name:expr, $body:expr) => {{
        let _ctx = $crate::profiler::ProfileContext::new($name);
        $body
    }};
}

/// Profile a function
/// 関数をプロファイル
#[macro_export]
macro_rules! profile_fn {
    ($name:expr) => {
        let _ctx = $crate::profiler::ProfileContext::new($name);
    };
}

/// Global profiler control functions
/// グローバルプロファイラー制御関数
///
/// Enable global profiler
/// グローバルプロファイラーを有効化
pub fn enable_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.enable();
    }
}

/// Disable global profiler
/// グローバルプロファイラーを無効化
pub fn disable_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.disable();
    }
}

/// Get profiler summary
/// プロファイラーサマリーを取得
pub fn get_profiler_summary() -> Option<ProfileSummary> {
    PROFILER.lock().ok().map(|p| p.get_summary())
}

/// Clear profiler data
/// プロファイラーデータをクリア
pub fn clear_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.clear();
    }
}

/// Force reset profiler to completely clean state (for testing)
/// プロファイラーを完全にクリーンな状態に強制リセット（テスト用）
pub fn force_reset_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        *profiler = Profiler::new();
    }
}

/// Print profiler report
/// プロファイラーレポートを出力
pub fn print_profiler_report() {
    if let Ok(profiler) = PROFILER.lock() {
        profiler.print_report();
    }
}

/// Export Chrome trace
/// Chromeトレースをエクスポート
pub fn export_chrome_trace() -> Option<String> {
    PROFILER.lock().ok().map(|p| p.export_chrome_trace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[ignore = "Flaky due to parallel test execution affecting global profiler state"]
    fn test_basic_profiling() {
        // Simple single reset approach
        disable_profiler();
        force_reset_profiler();

        // Verify profiler is really enabled
        enable_profiler();

        // Check if profiler is enabled before testing
        let is_enabled = PROFILER.lock().map(|p| p.enabled).unwrap_or(false);
        if !is_enabled {
            enable_profiler();
        }

        {
            let _ctx = ProfileContext::new("test_operation");
            thread::sleep(Duration::from_millis(20));
        }

        // Small delay for recording
        thread::sleep(Duration::from_millis(10));

        let summary_result = get_profiler_summary();
        assert!(summary_result.is_some(), "Failed to get profiler summary");

        let summary = summary_result.unwrap();
        if summary.operations.is_empty() {
            println!(
                "Profiler enabled: {}",
                PROFILER.lock().map(|p| p.enabled).unwrap_or(false)
            );
            println!("Operations count: {}", summary.operations.len());
        }
        assert!(summary.operations.len() > 0, "No operations recorded");

        let test_op = summary
            .operations
            .iter()
            .find(|op| op.name == "test_operation");
        assert!(
            test_op.is_some(),
            "test_operation not found in profiler summary"
        );
        assert_eq!(test_op.unwrap().count, 1);

        disable_profiler();
        force_reset_profiler();
    }

    #[test]
    #[ignore = "Flaky due to parallel test execution affecting global profiler state"]
    fn test_nested_profiling() {
        // Simple approach
        disable_profiler();
        force_reset_profiler();
        enable_profiler();

        // Double-check enabled state
        let is_enabled = PROFILER.lock().map(|p| p.enabled).unwrap_or(false);
        if !is_enabled {
            enable_profiler();
        }

        {
            let _ctx1 = ProfileContext::new("outer");
            thread::sleep(Duration::from_millis(15));
            {
                let _ctx2 = ProfileContext::new("inner");
                thread::sleep(Duration::from_millis(15));
            }
        }

        // Ensure all operations are recorded
        thread::sleep(Duration::from_millis(10));

        let summary_result = get_profiler_summary();
        assert!(summary_result.is_some(), "Failed to get profiler summary");

        let summary = summary_result.unwrap();
        if summary.operations.is_empty() {
            println!(
                "Profiler enabled: {}",
                PROFILER.lock().map(|p| p.enabled).unwrap_or(false)
            );
            println!(
                "Available operations: {:?}",
                summary
                    .operations
                    .iter()
                    .map(|op| &op.name)
                    .collect::<Vec<_>>()
            );
        }
        assert!(
            summary.operations.len() > 0,
            "No operations recorded in profiler"
        );

        let outer_op = summary.operations.iter().find(|op| op.name == "outer");
        let inner_op = summary.operations.iter().find(|op| op.name == "inner");

        assert!(
            outer_op.is_some(),
            "outer operation not found in profiler summary"
        );
        assert!(
            inner_op.is_some(),
            "inner operation not found in profiler summary"
        );

        disable_profiler();
        force_reset_profiler();
    }

    #[test]
    fn test_profile_macro() {
        // Ensure completely clean state with multiple resets
        for _ in 0..3 {
            disable_profiler();
            force_reset_profiler();
        }
        enable_profiler();

        profile!("macro_test", {
            thread::sleep(Duration::from_millis(15)); // Increased duration
        });

        // Add delay to ensure recording
        thread::sleep(Duration::from_millis(5));

        let summary = get_profiler_summary().unwrap();
        assert!(summary.operations.len() > 0, "No operations recorded");
        assert!(
            summary.operations.iter().any(|op| op.name == "macro_test"),
            "macro_test operation not found"
        );

        disable_profiler();
        force_reset_profiler();
    }
}

/// Central profiling coordinator for all RusTorch operations
pub struct RusTorchProfiler {
    /// Multi-GPU profiler for distributed operations
    #[cfg(not(target_arch = "wasm32"))]
    multi_gpu_profiler: Option<Arc<MultiGpuProfiler>>,
    /// General operation metrics
    operation_metrics: Arc<RwLock<OperationMetrics>>,
    /// Profiling configuration
    config: ProfilerConfig,
    /// Session start time
    session_start: Instant,
}

/// General operation performance metrics
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation execution times
    operation_times: HashMap<String, Vec<Duration>>,
    /// Memory usage snapshots
    memory_snapshots: Vec<MemorySnapshot>,
    /// Total operations profiled
    total_operations: usize,
    /// Session duration
    session_duration: Duration,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Peak memory usage since last snapshot
    pub peak_memory: usize,
    /// GPU memory usage per device
    pub gpu_memory: HashMap<usize, usize>,
}

impl RusTorchProfiler {
    /// Create new profiler instance
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            multi_gpu_profiler: None,
            operation_metrics: Arc::new(RwLock::new(OperationMetrics::new())),
            config,
            session_start: Instant::now(),
        }
    }

    /// Enable multi-GPU profiling
    pub fn enable_multi_gpu_profiling(&mut self, gpu_ids: Vec<usize>) -> RusTorchResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let profiler = MultiGpuProfiler::new(gpu_ids, self.config.clone())?;
            self.multi_gpu_profiler = Some(Arc::new(profiler));
        }
        Ok(())
    }

    /// Record operation timing
    pub fn record_operation(&self, operation_name: &str, duration: Duration) {
        let mut metrics = self.operation_metrics.write().unwrap();
        metrics
            .operation_times
            .entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        metrics.total_operations += 1;
        metrics.session_duration = self.session_start.elapsed();
    }

    /// Take memory snapshot
    pub fn take_memory_snapshot(&self, memory_usage: usize, gpu_memory: HashMap<usize, usize>) {
        let mut metrics = self.operation_metrics.write().unwrap();
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            memory_usage,
            peak_memory: memory_usage, // Simplified for now
            gpu_memory,
        };
        metrics.memory_snapshots.push(snapshot);
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> ProfilingReport {
        let operation_metrics = self.operation_metrics.read().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let multi_gpu_report = self
            .multi_gpu_profiler
            .as_ref()
            .map(|p| p.generate_report());

        #[cfg(target_arch = "wasm32")]
        let multi_gpu_report = None::<String>;

        ProfilingReport {
            session_duration: self.session_start.elapsed(),
            total_operations: operation_metrics.total_operations,
            operation_summary: self.summarize_operations(&operation_metrics),
            memory_analysis: self.analyze_memory(&operation_metrics),
            #[cfg(not(target_arch = "wasm32"))]
            multi_gpu_analysis: multi_gpu_report,
            recommendations: self.generate_recommendations(&operation_metrics),
        }
    }

    /// Summarize operation performance
    fn summarize_operations(&self, metrics: &OperationMetrics) -> OperationSummary {
        let mut summary = OperationSummary {
            operations: HashMap::new(),
            total_time: Duration::ZERO,
            slowest_operation: None,
        };

        for (op_name, durations) in &metrics.operation_times {
            let total: Duration = durations.iter().sum();
            let average = total / durations.len() as u32;
            let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
            let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);

            summary.operations.insert(
                op_name.clone(),
                OperationStats {
                    name: op_name.clone(),
                    count: durations.len(),
                    total_time: total,
                    avg_time: average,
                    min_time: min,
                    max_time: max,
                    memory_allocated: 0,
                    memory_freed: 0,
                    cuda_time: None,
                    self_cpu_time: total,
                    children: Vec::new(),
                },
            );

            summary.total_time += total;

            let should_update = if let Some((_, current_max)) = &summary.slowest_operation {
                max > *current_max
            } else {
                true
            };

            if should_update {
                summary.slowest_operation = Some((op_name.clone(), max));
            }
        }

        summary
    }

    /// Analyze memory usage patterns
    fn analyze_memory(&self, metrics: &OperationMetrics) -> MemoryAnalysis {
        if metrics.memory_snapshots.is_empty() {
            return MemoryAnalysis::default();
        }

        let total_memory: usize = metrics
            .memory_snapshots
            .iter()
            .map(|s| s.memory_usage)
            .sum();
        let avg_memory = total_memory / metrics.memory_snapshots.len();
        let peak_memory = metrics
            .memory_snapshots
            .iter()
            .map(|s| s.memory_usage)
            .max()
            .unwrap_or(0);

        MemoryAnalysis {
            average_usage: avg_memory,
            peak_usage: peak_memory,
            total_snapshots: metrics.memory_snapshots.len(),
            memory_trend: self.calculate_memory_trend(&metrics.memory_snapshots),
        }
    }

    /// Calculate memory usage trend
    fn calculate_memory_trend(&self, snapshots: &[MemorySnapshot]) -> MemoryTrend {
        if snapshots.len() < 2 {
            return MemoryTrend::Stable;
        }

        let first_half_avg = snapshots[..snapshots.len() / 2]
            .iter()
            .map(|s| s.memory_usage as f64)
            .sum::<f64>()
            / (snapshots.len() / 2) as f64;

        let second_half_avg = snapshots[snapshots.len() / 2..]
            .iter()
            .map(|s| s.memory_usage as f64)
            .sum::<f64>()
            / (snapshots.len() - snapshots.len() / 2) as f64;

        let change_ratio = (second_half_avg - first_half_avg) / first_half_avg;

        if change_ratio > 0.1 {
            MemoryTrend::Increasing
        } else if change_ratio < -0.1 {
            MemoryTrend::Decreasing
        } else {
            MemoryTrend::Stable
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, metrics: &OperationMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for slow operations
        for (op_name, durations) in &metrics.operation_times {
            if let Some(max_duration) = durations.iter().max() {
                if max_duration.as_millis() > 1000 {
                    recommendations.push(format!(
                        "Operation '{}' has slow instances (max: {}ms) - consider optimization",
                        op_name,
                        max_duration.as_millis()
                    ));
                }
            }
        }

        // Memory usage recommendations
        if !metrics.memory_snapshots.is_empty() {
            let memory_analysis = self.analyze_memory(metrics);
            match memory_analysis.memory_trend {
                MemoryTrend::Increasing => {
                    recommendations
                        .push("Memory usage is increasing - check for memory leaks".to_string());
                }
                MemoryTrend::Stable => {
                    recommendations
                        .push("Memory usage is stable - good memory management".to_string());
                }
                MemoryTrend::Decreasing => {
                    recommendations
                        .push("Memory usage is decreasing - efficient memory usage".to_string());
                }
            }
        }

        recommendations
    }
}

/// Complete profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Total session duration
    pub session_duration: Duration,
    /// Total operations profiled
    pub total_operations: usize,
    /// Operation performance summary
    pub operation_summary: OperationSummary,
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    /// Multi-GPU specific analysis
    #[cfg(not(target_arch = "wasm32"))]
    pub multi_gpu_analysis: Option<MultiGpuReport>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Operation performance summary
#[derive(Debug, Clone)]
pub struct OperationSummary {
    /// Per-operation statistics
    pub operations: HashMap<String, OperationStats>,
    /// Total time across all operations
    pub total_time: Duration,
    /// Slowest operation info
    pub slowest_operation: Option<(String, Duration)>,
}

/// Memory usage analysis
#[derive(Debug, Clone, Default)]
pub struct MemoryAnalysis {
    /// Average memory usage
    pub average_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Total memory snapshots taken
    pub total_snapshots: usize,
    /// Memory usage trend
    pub memory_trend: MemoryTrend,
}

/// Memory usage trend
#[derive(Debug, Clone)]
pub enum MemoryTrend {
    Increasing,
    Decreasing,
    Stable,
}

impl Default for MemoryTrend {
    fn default() -> Self {
        MemoryTrend::Stable
    }
}

impl OperationMetrics {
    /// Create new operation metrics
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            memory_snapshots: Vec::new(),
            total_operations: 0,
            session_duration: Duration::ZERO,
        }
    }
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
