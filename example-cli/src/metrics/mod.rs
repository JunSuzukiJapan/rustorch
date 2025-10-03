// Performance metrics collection system for RusTorch CLI

pub mod reporter;
pub mod timing;

use std::collections::HashMap;

/// Metrics collector for inference performance
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Time to first token (ms)
    pub ttft: Option<f64>,
    /// Tokens per second
    pub tokens_per_sec: Option<f64>,
    /// Total inference time (ms)
    pub total_time: Option<f64>,
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
    /// GPU memory usage (bytes)
    pub gpu_memory_usage: Option<usize>,
    /// Model size (bytes)
    pub model_size: Option<usize>,
    /// Backend used
    pub backend: Option<String>,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            ttft: None,
            tokens_per_sec: None,
            total_time: None,
            memory_usage: None,
            gpu_memory_usage: None,
            model_size: None,
            backend: None,
            custom: HashMap::new(),
        }
    }

    /// Set time to first token
    pub fn set_ttft(&mut self, ttft_ms: f64) {
        self.ttft = Some(ttft_ms);
    }

    /// Set tokens per second
    pub fn set_tokens_per_sec(&mut self, tps: f64) {
        self.tokens_per_sec = Some(tps);
    }

    /// Set total inference time
    pub fn set_total_time(&mut self, time_ms: f64) {
        self.total_time = Some(time_ms);
    }

    /// Set memory usage
    pub fn set_memory_usage(&mut self, bytes: usize) {
        self.memory_usage = Some(bytes);
    }

    /// Set GPU memory usage
    pub fn set_gpu_memory_usage(&mut self, bytes: usize) {
        self.gpu_memory_usage = Some(bytes);
    }

    /// Set model size
    pub fn set_model_size(&mut self, bytes: usize) {
        self.model_size = Some(bytes);
    }

    /// Set backend name
    pub fn set_backend(&mut self, backend: String) {
        self.backend = Some(backend);
    }

    /// Add custom metric
    pub fn add_custom(&mut self, key: String, value: f64) {
        self.custom.insert(key, value);
    }

    /// Calculate memory efficiency (model_size / memory_usage)
    pub fn memory_efficiency(&self) -> Option<f64> {
        match (self.model_size, self.memory_usage) {
            (Some(model), Some(mem)) if mem > 0 => Some(model as f64 / mem as f64),
            _ => None,
        }
    }

    /// Check if TTFT meets target (<200ms for 7B model)
    pub fn meets_ttft_target(&self) -> bool {
        self.ttft.map(|t| t < 200.0).unwrap_or(false)
    }

    /// Check if tokens/sec meets target (>20 for 7B model)
    pub fn meets_tps_target(&self) -> bool {
        self.tokens_per_sec.map(|t| t > 20.0).unwrap_or(false)
    }

    /// Check if memory usage is within target (<1.5x model size)
    pub fn meets_memory_target(&self) -> bool {
        match (self.model_size, self.memory_usage) {
            (Some(model), Some(mem)) => mem < (model as f64 * 1.5) as usize,
            _ => false,
        }
    }

    /// Check if all performance targets are met
    pub fn meets_all_targets(&self) -> bool {
        self.meets_ttft_target() && self.meets_tps_target() && self.meets_memory_target()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let metrics = MetricsCollector::new();
        assert!(metrics.ttft.is_none());
        assert!(metrics.tokens_per_sec.is_none());
    }

    #[test]
    fn test_set_metrics() {
        let mut metrics = MetricsCollector::new();
        metrics.set_ttft(150.0);
        metrics.set_tokens_per_sec(25.0);
        metrics.set_memory_usage(1_000_000);
        metrics.set_model_size(500_000);

        assert_eq!(metrics.ttft, Some(150.0));
        assert_eq!(metrics.tokens_per_sec, Some(25.0));
        assert_eq!(metrics.memory_usage, Some(1_000_000));
    }

    #[test]
    fn test_memory_efficiency() {
        let mut metrics = MetricsCollector::new();
        metrics.set_model_size(1_000_000);
        metrics.set_memory_usage(1_200_000);

        let efficiency = metrics.memory_efficiency().unwrap();
        assert!((efficiency - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_performance_targets() {
        let mut metrics = MetricsCollector::new();
        metrics.set_ttft(150.0); // <200ms ✓
        metrics.set_tokens_per_sec(25.0); // >20 ✓
        metrics.set_model_size(1_000_000);
        metrics.set_memory_usage(1_200_000); // <1.5x ✓

        assert!(metrics.meets_ttft_target());
        assert!(metrics.meets_tps_target());
        assert!(metrics.meets_memory_target());
        assert!(metrics.meets_all_targets());
    }

    #[test]
    fn test_custom_metrics() {
        let mut metrics = MetricsCollector::new();
        metrics.add_custom("latency_p50".to_string(), 100.0);
        metrics.add_custom("latency_p95".to_string(), 180.0);

        assert_eq!(metrics.custom.get("latency_p50"), Some(&100.0));
        assert_eq!(metrics.custom.get("latency_p95"), Some(&180.0));
    }
}
