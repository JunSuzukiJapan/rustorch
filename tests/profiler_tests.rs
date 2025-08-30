//! Comprehensive Tests for Performance Profiling & Benchmarking System
//! パフォーマンスプロファイリング・ベンチマーキングシステムの包括的テスト

use rustorch::error::RusTorchResult;
use rustorch::profiler::core::*;
use rustorch::profiler::metrics_collector::*;
use rustorch::profiler::benchmark_suite::*;
use rustorch::profiler::performance_analyzer::*;
use rustorch::profiler::system_profiler::*;
use rustorch::profiler::real_time_monitor::*;
use std::time::{Duration, Instant};
use std::thread;

#[cfg(test)]
mod core_profiler_tests {
    use super::*;

    #[test]
    fn test_profiler_core_creation() {
        let profiler = ProfilerCore::new();
        assert!(profiler.get_active_sessions().is_empty());
    }

    #[test]
    fn test_session_lifecycle() -> RusTorchResult<()> {
        let mut profiler = ProfilerCore::new();
        
        // Start session
        let session_id = profiler.start_session("test_session".to_string(), None)?;
        assert_eq!(profiler.get_active_sessions().len(), 1);
        
        // Stop session
        let snapshot = profiler.stop_session(&session_id)?;
        assert!(profiler.get_active_sessions().is_empty());
        assert_eq!(snapshot.session_id, session_id);
        
        Ok(())
    }

    #[test]
    fn test_timing_operations() -> RusTorchResult<()> {
        let mut profiler = ProfilerCore::new();
        let session_id = profiler.start_session("timing_test".to_string(), None)?;
        
        // Start and stop timer
        let timer_id = profiler.start_timer(&session_id, "test_operation".to_string())?;
        thread::sleep(Duration::from_millis(10));
        let metrics = profiler.stop_timer(&session_id, &timer_id)?;
        
        assert!(metrics.execution_time > Duration::from_millis(5));
        assert_eq!(metrics.operation_name, "test_operation");
        
        profiler.stop_session(&session_id)?;
        Ok(())
    }

    #[test]
    fn test_nested_operations() -> RusTorchResult<()> {
        let mut profiler = ProfilerCore::new();
        let session_id = profiler.start_session("nested_test".to_string(), None)?;
        
        // Start outer operation
        let outer_timer = profiler.start_timer(&session_id, "outer".to_string())?;
        thread::sleep(Duration::from_millis(5));
        
        // Start inner operation
        let inner_timer = profiler.start_timer(&session_id, "inner".to_string())?;
        thread::sleep(Duration::from_millis(10));
        
        // Stop operations
        let inner_metrics = profiler.stop_timer(&session_id, &inner_timer)?;
        let outer_metrics = profiler.stop_timer(&session_id, &outer_timer)?;
        
        assert!(outer_metrics.execution_time > inner_metrics.execution_time);
        
        let snapshot = profiler.stop_session(&session_id)?;
        assert_eq!(snapshot.operations.len(), 2);
        
        Ok(())
    }
}

#[cfg(test)]
mod metrics_collector_tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.get_metric_count(), 0);
    }

    #[test]
    fn test_counter_metrics() -> RusTorchResult<()> {
        let mut collector = MetricsCollector::new();
        
        // Update counter
        collector.update_metric("test_counter".to_string(), 5.0)?;
        collector.update_metric("test_counter".to_string(), 3.0)?;
        
        let stats = collector.get_metric_stats("test_counter")?;
        assert_eq!(stats.count, 2);
        assert_eq!(stats.sum, 8.0);
        assert_eq!(stats.mean, 4.0);
        
        Ok(())
    }

    #[test]
    fn test_gauge_metrics() -> RusTorchResult<()> {
        let mut collector = MetricsCollector::new();
        
        collector.set_gauge("memory_usage".to_string(), 1024.0)?;
        collector.set_gauge("memory_usage".to_string(), 2048.0)?;
        
        let stats = collector.get_metric_stats("memory_usage")?;
        assert_eq!(stats.current, 2048.0);
        assert_eq!(stats.count, 2);
        
        Ok(())
    }

    #[test]
    fn test_histogram_metrics() -> RusTorchResult<()> {
        let mut collector = MetricsCollector::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        
        for value in values {
            collector.record_histogram("response_time".to_string(), value)?;
        }
        
        let stats = collector.get_metric_stats("response_time")?;
        assert_eq!(stats.count, 8);
        assert!(stats.mean > 5.0);
        assert!(stats.percentiles.contains_key(&95.0));
        
        Ok(())
    }

    #[test]
    fn test_timing_metrics() -> RusTorchResult<()> {
        let mut collector = MetricsCollector::new();
        
        let start = Instant::now();
        thread::sleep(Duration::from_millis(10));
        let duration = start.elapsed();
        
        collector.record_timing("operation_duration".to_string(), duration)?;
        
        let stats = collector.get_metric_stats("operation_duration")?;
        assert!(stats.mean >= 10.0); // At least 10ms
        assert_eq!(stats.count, 1);
        
        Ok(())
    }

    #[test]
    fn test_system_metrics_collection() -> RusTorchResult<()> {
        let mut collector = MetricsCollector::new();
        
        collector.collect_system_metrics()?;
        
        // Verify system metrics were collected
        assert!(collector.get_metric_count() > 0);
        
        // Check for expected system metrics
        let cpu_result = collector.get_metric_stats("system.cpu_usage");
        let memory_result = collector.get_metric_stats("system.memory_usage");
        
        assert!(cpu_result.is_ok() || memory_result.is_ok());
        
        Ok(())
    }
}

#[cfg(test)]
mod benchmark_suite_tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = AdvancedBenchmarkSuite::new();
        assert_eq!(suite.get_benchmark_count(), 0);
    }

    #[test]
    fn test_simple_benchmark() -> RusTorchResult<()> {
        let mut suite = AdvancedBenchmarkSuite::new();
        
        let benchmark_fn = || {
            thread::sleep(Duration::from_millis(1));
            Ok(())
        };
        
        let result = suite.run_benchmark("sleep_test", 5, benchmark_fn)?;
        
        assert_eq!(result.name, "sleep_test");
        assert_eq!(result.iterations, 5);
        assert!(result.total_time > Duration::from_millis(4));
        assert!(result.statistics.mean >= 1.0); // At least 1ms mean
        
        Ok(())
    }

    #[test]
    fn test_benchmark_with_warmup() -> RusTorchResult<()> {
        let mut suite = AdvancedBenchmarkSuite::new();
        
        let config = BenchmarkConfig {
            iterations: 10,
            warmup_iterations: 3,
            confidence_level: 0.95,
        };
        
        let benchmark_fn = || {
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            Ok(sum)
        };
        
        let result = suite.run_benchmark_with_config("computation_test", config, benchmark_fn)?;
        
        assert_eq!(result.iterations, 10);
        assert!(result.statistics.confidence_interval.0 < result.statistics.confidence_interval.1);
        
        Ok(())
    }

    #[test]
    fn test_statistical_analysis() -> RusTorchResult<()> {
        let mut suite = AdvancedBenchmarkSuite::new();
        
        // Create predictable benchmark with known variance
        let benchmark_fn = || {
            let sleep_time = (rand::random::<u64>() % 5) + 1; // 1-5ms
            thread::sleep(Duration::from_millis(sleep_time));
            Ok(())
        };
        
        let result = suite.run_benchmark("variable_sleep", 20, benchmark_fn)?;
        
        assert!(result.statistics.std_deviation > 0.0);
        assert!(result.statistics.variance > 0.0);
        assert!(result.statistics.percentiles.contains_key(&50.0)); // Median
        assert!(result.statistics.percentiles.contains_key(&95.0)); // 95th percentile
        
        Ok(())
    }

    #[test]
    fn test_benchmark_comparison() -> RusTorchResult<()> {
        let mut suite = AdvancedBenchmarkSuite::new();
        
        // Fast benchmark
        let fast_fn = || {
            thread::sleep(Duration::from_millis(1));
            Ok(())
        };
        
        // Slow benchmark
        let slow_fn = || {
            thread::sleep(Duration::from_millis(5));
            Ok(())
        };
        
        let fast_result = suite.run_benchmark("fast_op", 10, fast_fn)?;
        let slow_result = suite.run_benchmark("slow_op", 10, slow_fn)?;
        
        let comparison = suite.compare_benchmarks(&fast_result, &slow_result)?;
        
        assert!(comparison.performance_ratio > 1.0); // Slow should be slower
        assert!(comparison.significance_level > 0.0);
        
        Ok(())
    }
}

#[cfg(test)]
mod performance_analyzer_tests {
    use super::*;

    #[test]
    fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.get_trend_count(), 0);
    }

    #[test]
    fn test_trend_analysis() -> RusTorchResult<()> {
        let mut analyzer = PerformanceAnalyzer::new();
        
        // Add sample data points
        let mut data_points = Vec::new();
        for i in 1..=10 {
            data_points.push((i as f64, (i as f64) * 10.0)); // Linear trend: y = 10x
        }
        
        let trend = analyzer.analyze_trend("linear_metric", data_points)?;
        
        assert_eq!(trend.metric_name, "linear_metric");
        assert!(trend.slope > 8.0 && trend.slope < 12.0); // Should be close to 10
        assert!(trend.correlation > 0.9); // Strong positive correlation
        assert_eq!(trend.trend_type, TrendType::Increasing);
        
        Ok(())
    }

    #[test]
    fn test_optimization_recommendations() -> RusTorchResult<()> {
        let mut analyzer = PerformanceAnalyzer::new();
        
        // Simulate high CPU usage trend
        let cpu_data: Vec<(f64, f64)> = (1..=10)
            .map(|i| (i as f64, 80.0 + (i as f64) * 2.0)) // Increasing CPU usage
            .collect();
        
        analyzer.analyze_trend("cpu_usage", cpu_data)?;
        
        let recommendations = analyzer.get_optimization_recommendations()?;
        assert!(!recommendations.is_empty());
        
        let cpu_rec = recommendations.iter()
            .find(|r| r.category == OptimizationCategory::CPU);
        assert!(cpu_rec.is_some());
        
        Ok(())
    }

    #[test]
    fn test_bottleneck_detection() -> RusTorchResult<()> {
        let mut analyzer = PerformanceAnalyzer::new();
        
        // Add metrics with different patterns
        let slow_operation = vec![(1.0, 100.0), (2.0, 150.0), (3.0, 200.0)];
        let fast_operation = vec![(1.0, 10.0), (2.0, 12.0), (3.0, 11.0)];
        
        analyzer.analyze_trend("slow_op", slow_operation)?;
        analyzer.analyze_trend("fast_op", fast_operation)?;
        
        let bottlenecks = analyzer.detect_bottlenecks()?;
        assert!(!bottlenecks.is_empty());
        
        let slow_bottleneck = bottlenecks.iter()
            .find(|b| b.operation_name == "slow_op");
        assert!(slow_bottleneck.is_some());
        
        Ok(())
    }
}

#[cfg(test)]
mod system_profiler_tests {
    use super::*;

    #[test]
    fn test_system_profiler_creation() {
        let profiler = SystemProfiler::new();
        assert_eq!(profiler.get_history().len(), 0);
    }

    #[test]
    fn test_metrics_collection() -> RusTorchResult<()> {
        let mut profiler = SystemProfiler::new();
        
        let metrics = profiler.collect_metrics()?;
        assert!(metrics.timestamp.elapsed() < Duration::from_secs(1));
        assert_eq!(profiler.get_history().len(), 1);
        
        Ok(())
    }

    #[test]
    fn test_system_summary() -> RusTorchResult<()> {
        let mut profiler = SystemProfiler::new();
        
        // Collect multiple metrics
        for _ in 0..5 {
            profiler.collect_metrics()?;
            thread::sleep(Duration::from_millis(10));
        }
        
        let summary = profiler.get_system_summary();
        assert_eq!(summary.sample_count, 5);
        assert!(summary.avg_cpu_usage >= 0.0);
        assert!(summary.avg_memory_usage_percent >= 0.0);
        
        Ok(())
    }

    #[test]
    fn test_history_limit() -> RusTorchResult<()> {
        let mut profiler = SystemProfiler::new();
        
        // Collect more metrics than the default limit
        for _ in 0..1005 { // Default limit is 1000
            profiler.collect_metrics()?;
        }
        
        assert_eq!(profiler.get_history().len(), 1000);
        
        Ok(())
    }

    #[test]
    fn test_clear_history() -> RusTorchResult<()> {
        let mut profiler = SystemProfiler::new();
        
        profiler.collect_metrics()?;
        assert_eq!(profiler.get_history().len(), 1);
        
        profiler.clear_history();
        assert_eq!(profiler.get_history().len(), 0);
        
        Ok(())
    }
}

#[cfg(test)]
mod real_time_monitor_tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = RealTimeMonitor::new(MonitorConfig::default());
        assert!(monitor.get_alerts().is_ok());
    }

    #[test]
    fn test_monitor_lifecycle() -> RusTorchResult<()> {
        let monitor = RealTimeMonitor::new(MonitorConfig::default());
        
        // Start monitoring
        monitor.start()?;
        
        // Stop monitoring
        monitor.stop()?;
        
        Ok(())
    }

    #[test]
    fn test_alert_management() -> RusTorchResult<()> {
        let monitor = RealTimeMonitor::new(MonitorConfig::default());
        
        // Initially no alerts
        let alerts = monitor.get_alerts()?;
        assert!(alerts.is_empty());
        
        // Clear alerts (should succeed even when empty)
        monitor.clear_alerts()?;
        
        let alerts_after_clear = monitor.get_alerts()?;
        assert!(alerts_after_clear.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds {
            cpu_threshold: 80.0,
            memory_threshold: 90.0,
            gpu_threshold: 95.0,
        };
        
        let config = MonitorConfig {
            sampling_interval: Duration::from_millis(50),
            alert_thresholds: thresholds.clone(),
            enable_system_monitoring: true,
        };
        
        let monitor = RealTimeMonitor::new(config);
        assert!(monitor.get_alerts().is_ok());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_profiler_with_metrics_collector() -> RusTorchResult<()> {
        let mut profiler = ProfilerCore::new();
        let mut collector = MetricsCollector::new();
        
        // Start profiling session
        let session_id = profiler.start_session("integration_test".to_string(), None)?;
        
        // Perform some operations
        for i in 0..5 {
            let timer_id = profiler.start_timer(&session_id, format!("operation_{}", i))?;
            thread::sleep(Duration::from_millis(10));
            let metrics = profiler.stop_timer(&session_id, &timer_id)?;
            
            // Record metrics in collector
            collector.record_timing(format!("op_{}_duration", i), metrics.execution_time)?;
        }
        
        // Stop session and verify
        let snapshot = profiler.stop_session(&session_id)?;
        assert_eq!(snapshot.operations.len(), 5);
        assert_eq!(collector.get_metric_count(), 5);
        
        Ok(())
    }

    #[test]
    fn test_end_to_end_profiling_workflow() -> RusTorchResult<()> {
        let mut profiler = ProfilerCore::new();
        let mut collector = MetricsCollector::new();
        let mut analyzer = PerformanceAnalyzer::new();
        let mut benchmark_suite = AdvancedBenchmarkSuite::new();
        
        // 1. Run benchmark to generate data
        let benchmark_fn = || {
            thread::sleep(Duration::from_millis(5));
            Ok(())
        };
        
        let benchmark_result = benchmark_suite.run_benchmark("test_operation", 10, benchmark_fn)?;
        
        // 2. Record benchmark data in metrics collector
        for duration in &benchmark_result.raw_times {
            collector.record_timing("benchmark_times".to_string(), *duration)?;
        }
        
        // 3. Analyze trends
        let trend_data: Vec<(f64, f64)> = benchmark_result.raw_times
            .iter()
            .enumerate()
            .map(|(i, d)| (i as f64, d.as_secs_f64() * 1000.0))
            .collect();
        
        let trend = analyzer.analyze_trend("benchmark_trend", trend_data)?;
        assert_eq!(trend.metric_name, "benchmark_trend");
        
        // 4. Get recommendations
        let recommendations = analyzer.get_optimization_recommendations()?;
        
        // Verify the workflow completed successfully
        assert!(benchmark_result.iterations > 0);
        assert!(collector.get_metric_count() > 0);
        assert_eq!(trend.metric_name, "benchmark_trend");
        
        Ok(())
    }

    #[test]
    fn test_profiler_performance_overhead() -> RusTorchResult<()> {
        let iterations = 1000;
        
        // Measure baseline performance (no profiling)
        let start_baseline = Instant::now();
        for _ in 0..iterations {
            let _ = (0..100).map(|x| x * x).sum::<i32>();
        }
        let baseline_duration = start_baseline.elapsed();
        
        // Measure with profiling enabled
        let mut profiler = ProfilerCore::new();
        let session_id = profiler.start_session("overhead_test".to_string(), None)?;
        
        let start_profiled = Instant::now();
        for i in 0..iterations {
            let timer_id = profiler.start_timer(&session_id, format!("computation_{}", i))?;
            let _ = (0..100).map(|x| x * x).sum::<i32>();
            profiler.stop_timer(&session_id, &timer_id)?;
        }
        let profiled_duration = start_profiled.elapsed();
        
        profiler.stop_session(&session_id)?;
        
        // Calculate overhead
        let overhead_ratio = profiled_duration.as_secs_f64() / baseline_duration.as_secs_f64();
        
        // Overhead should be reasonable (less than 10x)
        assert!(overhead_ratio < 10.0, "Profiling overhead too high: {}x", overhead_ratio);
        
        Ok(())
    }
}