//! Performance Profiling System
//!
//! Advanced profiling system for performance monitoring, bottleneck detection,
//! and operation timing analysis with statistical reporting.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::fmt;

use crate::error::{RusTorchError, RusTorchResult};

/// Performance profile entry
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub operation_name: String,
    pub duration: Duration,
    pub timestamp: Instant,
    pub metadata: HashMap<String, String>,
    pub call_count: usize,
}

impl ProfileEntry {
    /// Create new profile entry
    pub fn new(operation_name: String, duration: Duration) -> Self {
        Self {
            operation_name,
            duration,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
            call_count: 1,
        }
    }
    
    /// Create with metadata
    pub fn with_metadata(operation_name: String, duration: Duration, metadata: HashMap<String, String>) -> Self {
        Self {
            operation_name,
            duration,
            timestamp: Instant::now(),
            metadata,
            call_count: 1,
        }
    }
    
    /// Duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }
    
    /// Duration in microseconds
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }
}

/// Aggregated performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_operations: usize,
    pub average_duration_ms: f64,
    pub median_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub std_deviation_ms: f64,
    pub percentile_95_ms: f64,
    pub percentile_99_ms: f64,
    pub slowest_operation: String,
    pub fastest_operation: String,
    pub operations_per_second: f64,
    pub slow_operations_count: usize, // Operations > 100ms
    pub bottlenecks: Vec<String>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_duration_ms: 0.0,
            median_duration_ms: 0.0,
            min_duration_ms: 0.0,
            max_duration_ms: 0.0,
            std_deviation_ms: 0.0,
            percentile_95_ms: 0.0,
            percentile_99_ms: 0.0,
            slowest_operation: String::new(),
            fastest_operation: String::new(),
            operations_per_second: 0.0,
            slow_operations_count: 0,
            bottlenecks: Vec::new(),
        }
    }
}

/// Performance profiler for operation timing
pub struct DebugProfiler {
    enabled: bool,
    max_entries: usize,
    entries: Vec<ProfileEntry>,
    operation_stats: HashMap<String, OperationStats>,
    total_operations: usize,
    session_start: Instant,
}

/// Statistics for a specific operation type
#[derive(Debug, Clone)]
struct OperationStats {
    count: usize,
    total_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    durations: Vec<Duration>, // For percentile calculations
}

impl OperationStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            durations: Vec::new(),
        }
    }
    
    fn update(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.durations.push(duration);
        
        // Keep only recent durations to prevent memory growth
        if self.durations.len() > 1000 {
            self.durations.drain(0..100);
        }
    }
    
    fn average_duration(&self) -> Duration {
        if self.count > 0 {
            self.total_duration / self.count as u32
        } else {
            Duration::ZERO
        }
    }
    
    fn median_duration(&self) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        
        let mut sorted = self.durations.clone();
        sorted.sort();
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            sorted[mid]
        }
    }
    
    fn percentile(&self, p: f64) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        
        let mut sorted = self.durations.clone();
        sorted.sort();
        
        let index = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl fmt::Debug for DebugProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugProfiler")
            .field("enabled", &self.enabled)
            .field("max_entries", &self.max_entries)
            .field("total_operations", &self.total_operations)
            .field("operation_types", &self.operation_stats.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl DebugProfiler {
    /// Create new profiler
    pub fn new(enabled: bool, max_entries: usize) -> Self {
        Self {
            enabled,
            max_entries,
            entries: Vec::new(),
            operation_stats: HashMap::new(),
            total_operations: 0,
            session_start: Instant::now(),
        }
    }
    
    /// Record operation timing
    pub fn record_operation(&mut self, operation_name: &str, duration: Duration) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        // Create profile entry
        let entry = ProfileEntry::new(operation_name.to_string(), duration);
        
        // Update operation statistics
        let stats = self.operation_stats
            .entry(operation_name.to_string())
            .or_insert_with(OperationStats::new);
        stats.update(duration);
        
        // Add to entries
        self.entries.push(entry);
        self.total_operations += 1;
        
        // Maintain max entries limit
        if self.entries.len() > self.max_entries {
            self.entries.drain(0..self.max_entries / 10);
        }
        
        Ok(())
    }
    
    /// Record operation with metadata
    pub fn record_operation_with_metadata(
        &mut self, 
        operation_name: &str, 
        duration: Duration,
        metadata: HashMap<String, String>
    ) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        // Create profile entry with metadata
        let entry = ProfileEntry::with_metadata(operation_name.to_string(), duration, metadata);
        
        // Update operation statistics
        let stats = self.operation_stats
            .entry(operation_name.to_string())
            .or_insert_with(OperationStats::new);
        stats.update(duration);
        
        // Add to entries
        self.entries.push(entry);
        self.total_operations += 1;
        
        // Maintain max entries limit
        if self.entries.len() > self.max_entries {
            self.entries.drain(0..self.max_entries / 10);
        }
        
        Ok(())
    }
    
    /// Get comprehensive performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        if self.entries.is_empty() {
            return PerformanceMetrics::default();
        }
        
        // Collect all durations
        let durations: Vec<Duration> = self.entries.iter()
            .map(|e| e.duration)
            .collect();
        
        let mut sorted_durations = durations.clone();
        sorted_durations.sort();
        
        // Calculate basic statistics
        let total_duration: Duration = durations.iter().sum();
        let average_duration = total_duration / durations.len() as u32;
        
        let min_duration = sorted_durations.first().cloned().unwrap_or(Duration::ZERO);
        let max_duration = sorted_durations.last().cloned().unwrap_or(Duration::ZERO);
        
        // Calculate median
        let median_duration = if sorted_durations.len() % 2 == 0 {
            let mid = sorted_durations.len() / 2;
            (sorted_durations[mid - 1] + sorted_durations[mid]) / 2
        } else {
            sorted_durations[sorted_durations.len() / 2]
        };
        
        // Calculate standard deviation
        let mean_ms = average_duration.as_secs_f64() * 1000.0;
        let variance = durations.iter()
            .map(|d| {
                let diff = (d.as_secs_f64() * 1000.0) - mean_ms;
                diff * diff
            })
            .sum::<f64>() / durations.len() as f64;
        let std_deviation_ms = variance.sqrt();
        
        // Calculate percentiles
        let p95_index = ((sorted_durations.len() as f64 - 1.0) * 0.95).round() as usize;
        let p99_index = ((sorted_durations.len() as f64 - 1.0) * 0.99).round() as usize;
        
        let percentile_95_ms = sorted_durations[p95_index.min(sorted_durations.len() - 1)]
            .as_secs_f64() * 1000.0;
        let percentile_99_ms = sorted_durations[p99_index.min(sorted_durations.len() - 1)]
            .as_secs_f64() * 1000.0;
        
        // Find slowest and fastest operations
        let slowest_entry = self.entries.iter()
            .max_by_key(|e| e.duration)
            .map(|e| e.operation_name.clone())
            .unwrap_or_default();
        
        let fastest_entry = self.entries.iter()
            .min_by_key(|e| e.duration)
            .map(|e| e.operation_name.clone())
            .unwrap_or_default();
        
        // Calculate operations per second
        let session_duration = self.session_start.elapsed().as_secs_f64();
        let operations_per_second = if session_duration > 0.0 {
            self.total_operations as f64 / session_duration
        } else {
            0.0
        };
        
        // Count slow operations (> 100ms)
        let slow_operations_count = durations.iter()
            .filter(|d| d.as_millis() > 100)
            .count();
        
        // Identify bottlenecks (operations with high average duration)
        let mut bottlenecks: Vec<(String, f64)> = self.operation_stats.iter()
            .map(|(name, stats)| (name.clone(), stats.average_duration().as_secs_f64() * 1000.0))
            .filter(|(_, avg_ms)| *avg_ms > 50.0) // Bottleneck threshold: 50ms
            .collect();
        
        bottlenecks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let bottlenecks: Vec<String> = bottlenecks.into_iter()
            .take(5) // Top 5 bottlenecks
            .map(|(name, _)| name)
            .collect();
        
        PerformanceMetrics {
            total_operations: self.total_operations,
            average_duration_ms: mean_ms,
            median_duration_ms: median_duration.as_secs_f64() * 1000.0,
            min_duration_ms: min_duration.as_secs_f64() * 1000.0,
            max_duration_ms: max_duration.as_secs_f64() * 1000.0,
            std_deviation_ms,
            percentile_95_ms,
            percentile_99_ms,
            slowest_operation: slowest_entry,
            fastest_operation: fastest_entry,
            operations_per_second,
            slow_operations_count,
            bottlenecks,
        }
    }
    
    /// Get statistics for specific operation
    pub fn get_operation_stats(&self, operation_name: &str) -> Option<OperationStats> {
        self.operation_stats.get(operation_name).cloned()
    }
    
    /// Get all operation names
    pub fn get_operation_names(&self) -> Vec<String> {
        self.operation_stats.keys().cloned().collect()
    }
    
    /// Get recent entries (last n entries)
    pub fn get_recent_entries(&self, n: usize) -> Vec<&ProfileEntry> {
        let start = if self.entries.len() > n {
            self.entries.len() - n
        } else {
            0
        };
        
        self.entries[start..].iter().collect()
    }
    
    /// Get entries for specific operation
    pub fn get_entries_for_operation(&self, operation_name: &str) -> Vec<&ProfileEntry> {
        self.entries.iter()
            .filter(|e| e.operation_name == operation_name)
            .collect()
    }
    
    /// Get total number of recorded operations
    pub fn get_total_profiles(&self) -> usize {
        self.total_operations
    }
    
    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.entries.clear();
        self.operation_stats.clear();
        self.total_operations = 0;
        self.session_start = Instant::now();
    }
    
    /// Enable/disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        let metrics = self.get_performance_metrics();
        
        let mut report = String::new();
        report.push_str("🚀 Performance Profile Report\n");
        report.push_str("==============================\n\n");
        
        report.push_str(&format!("📊 Overall Statistics:\n"));
        report.push_str(&format!("  Total Operations: {}\n", metrics.total_operations));
        report.push_str(&format!("  Operations/Second: {:.2}\n", metrics.operations_per_second));
        report.push_str(&format!("  Average Duration: {:.3}ms\n", metrics.average_duration_ms));
        report.push_str(&format!("  Median Duration: {:.3}ms\n", metrics.median_duration_ms));
        report.push_str(&format!("  Std Deviation: {:.3}ms\n\n", metrics.std_deviation_ms));
        
        report.push_str(&format!("⚡ Performance Range:\n"));
        report.push_str(&format!("  Fastest: {:.3}ms ({})\n", metrics.min_duration_ms, metrics.fastest_operation));
        report.push_str(&format!("  Slowest: {:.3}ms ({})\n", metrics.max_duration_ms, metrics.slowest_operation));
        report.push_str(&format!("  95th Percentile: {:.3}ms\n", metrics.percentile_95_ms));
        report.push_str(&format!("  99th Percentile: {:.3}ms\n\n", metrics.percentile_99_ms));
        
        report.push_str(&format!("🐌 Performance Issues:\n"));
        report.push_str(&format!("  Slow Operations (>100ms): {}\n", metrics.slow_operations_count));
        
        if !metrics.bottlenecks.is_empty() {
            report.push_str("  Top Bottlenecks:\n");
            for (i, bottleneck) in metrics.bottlenecks.iter().enumerate() {
                report.push_str(&format!("    {}. {}\n", i + 1, bottleneck));
            }
        } else {
            report.push_str("  No significant bottlenecks detected\n");
        }
        
        report
    }
}

/// RAII profiling scope guard
pub struct ProfileScope {
    operation_name: String,
    start_time: Instant,
    profiler: Arc<Mutex<DebugProfiler>>,
    metadata: HashMap<String, String>,
}

impl ProfileScope {
    /// Create new profiling scope
    pub fn new(operation_name: String, profiler: Arc<Mutex<DebugProfiler>>) -> Self {
        Self {
            operation_name,
            start_time: Instant::now(),
            profiler,
            metadata: HashMap::new(),
        }
    }
    
    /// Create profiling scope with metadata
    pub fn with_metadata(
        operation_name: String, 
        profiler: Arc<Mutex<DebugProfiler>>,
        metadata: HashMap<String, String>
    ) -> Self {
        Self {
            operation_name,
            start_time: Instant::now(),
            profiler,
            metadata,
        }
    }
    
    /// Add metadata to the profile
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        
        if let Ok(mut profiler) = self.profiler.lock() {
            if self.metadata.is_empty() {
                let _ = profiler.record_operation(&self.operation_name, duration);
            } else {
                let _ = profiler.record_operation_with_metadata(
                    &self.operation_name, 
                    duration, 
                    self.metadata.clone()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;
    
    #[test]
    fn test_profile_entry_creation() {
        let duration = Duration::from_millis(100);
        let entry = ProfileEntry::new("test_operation".to_string(), duration);
        
        assert_eq!(entry.operation_name, "test_operation");
        assert_eq!(entry.duration, duration);
        assert_eq!(entry.duration_ms(), 100.0);
        assert_eq!(entry.call_count, 1);
    }
    
    #[test]
    fn test_profiler_creation() {
        let profiler = DebugProfiler::new(true, 1000);
        assert!(profiler.is_enabled());
        assert_eq!(profiler.get_total_profiles(), 0);
    }
    
    #[test]
    fn test_profiler_recording() {
        let mut profiler = DebugProfiler::new(true, 1000);
        
        let duration = Duration::from_millis(50);
        assert!(profiler.record_operation("test_op", duration).is_ok());
        
        assert_eq!(profiler.get_total_profiles(), 1);
        
        let metrics = profiler.get_performance_metrics();
        assert_eq!(metrics.total_operations, 1);
        assert_eq!(metrics.average_duration_ms, 50.0);
    }
    
    #[test]
    fn test_profiler_disabled() {
        let mut profiler = DebugProfiler::new(false, 1000);
        
        let duration = Duration::from_millis(50);
        assert!(profiler.record_operation("test_op", duration).is_ok());
        
        assert_eq!(profiler.get_total_profiles(), 0);
    }
    
    #[test]
    fn test_performance_metrics_calculation() {
        let mut profiler = DebugProfiler::new(true, 1000);
        
        // Record various operations
        let durations = [10, 20, 30, 40, 50, 100, 200, 500];
        for (i, &duration_ms) in durations.iter().enumerate() {
            let duration = Duration::from_millis(duration_ms);
            profiler.record_operation(&format!("op_{}", i), duration).unwrap();
        }
        
        let metrics = profiler.get_performance_metrics();
        
        assert_eq!(metrics.total_operations, 8);
        assert!(metrics.average_duration_ms > 0.0);
        assert!(metrics.median_duration_ms > 0.0);
        assert_eq!(metrics.min_duration_ms, 10.0);
        assert_eq!(metrics.max_duration_ms, 500.0);
        assert!(metrics.std_deviation_ms > 0.0);
        assert!(metrics.slow_operations_count > 0); // 200ms and 500ms are > 100ms
    }
    
    #[test]
    fn test_operation_stats() {
        let mut profiler = DebugProfiler::new(true, 1000);
        
        // Record multiple instances of same operation
        for i in 0..5 {
            let duration = Duration::from_millis(100 + i * 10);
            profiler.record_operation("repeated_op", duration).unwrap();
        }
        
        let stats = profiler.get_operation_stats("repeated_op").unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(140));
    }
    
    #[test]
    fn test_profile_scope() {
        let profiler = Arc::new(Mutex::new(DebugProfiler::new(true, 1000)));
        
        {
            let _scope = ProfileScope::new("scoped_operation".to_string(), Arc::clone(&profiler));
            thread::sleep(StdDuration::from_millis(10));
        } // Scope drops here, recording the profile
        
        let prof = profiler.lock().unwrap();
        assert_eq!(prof.get_total_profiles(), 1);
        
        let entries = prof.get_entries_for_operation("scoped_operation");
        assert_eq!(entries.len(), 1);
        assert!(entries[0].duration_ms() >= 10.0);
    }
    
    #[test]
    fn test_profile_scope_with_metadata() {
        let profiler = Arc::new(Mutex::new(DebugProfiler::new(true, 1000)));
        
        {
            let mut metadata = HashMap::new();
            metadata.insert("component".to_string(), "tensor".to_string());
            metadata.insert("operation_type".to_string(), "multiplication".to_string());
            
            let _scope = ProfileScope::with_metadata(
                "tensor_multiply".to_string(), 
                Arc::clone(&profiler),
                metadata
            );
            thread::sleep(StdDuration::from_millis(5));
        }
        
        let prof = profiler.lock().unwrap();
        let entries = prof.get_entries_for_operation("tensor_multiply");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].metadata.get("component"), Some(&"tensor".to_string()));
        assert_eq!(entries[0].metadata.get("operation_type"), Some(&"multiplication".to_string()));
    }
    
    #[test]
    fn test_bottleneck_detection() {
        let mut profiler = DebugProfiler::new(true, 1000);
        
        // Create a clear bottleneck
        for _ in 0..10 {
            profiler.record_operation("fast_op", Duration::from_millis(5)).unwrap();
            profiler.record_operation("slow_op", Duration::from_millis(200)).unwrap();
        }
        
        let metrics = profiler.get_performance_metrics();
        assert!(!metrics.bottlenecks.is_empty());
        assert!(metrics.bottlenecks.contains(&"slow_op".to_string()));
    }
    
    #[test]
    fn test_profiler_clear() {
        let mut profiler = DebugProfiler::new(true, 1000);
        
        profiler.record_operation("test_op", Duration::from_millis(50)).unwrap();
        assert_eq!(profiler.get_total_profiles(), 1);
        
        profiler.clear();
        assert_eq!(profiler.get_total_profiles(), 0);
        
        let metrics = profiler.get_performance_metrics();
        assert_eq!(metrics.total_operations, 0);
    }
}