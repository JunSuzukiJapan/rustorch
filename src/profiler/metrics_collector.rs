//! Advanced Metrics Collection System
//! 高度メトリクス収集システム

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Metric types
/// メトリクスタイプ
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricType {
    /// Counter that only increases
    /// 増加のみのカウンター
    Counter,
    /// Gauge that can increase or decrease
    /// 増減可能なゲージ
    Gauge,
    /// Histogram for value distributions
    /// 値分布のヒストグラム
    Histogram,
    /// Timer for measuring durations
    /// 期間測定用タイマー
    TimingMs,
    /// Throughput metric (operations per second)
    /// スループットメトリクス（秒間操作数）
    Throughput,
    /// Memory usage in bytes
    /// メモリ使用量（バイト）
    MemoryBytes,
    /// CPU utilization percentage
    /// CPU使用率
    CpuPercent,
    /// GPU utilization percentage
    /// GPU使用率
    GpuPercent,
    /// Custom user-defined metric
    /// カスタムユーザー定義メトリクス
    Custom(String),
}

/// Custom metric definition
/// カスタムメトリクス定義
#[derive(Debug, Clone)]
pub struct CustomMetric {
    /// Metric name
    /// メトリクス名
    pub name: String,
    /// Metric type
    /// メトリクスタイプ
    pub metric_type: MetricType,
    /// Current value
    /// 現在値
    pub value: f64,
    /// Timestamp of last update
    /// 最終更新タイムスタンプ
    pub timestamp: Instant,
    /// Value history for trend analysis
    /// 傾向分析用値履歴
    pub history: VecDeque<(Instant, f64)>,
    /// Description
    /// 説明
    pub description: Option<String>,
    /// Tags for categorization
    /// カテゴライゼーション用タグ
    pub tags: HashMap<String, String>,
    /// Unit of measurement
    /// 測定単位
    pub unit: Option<String>,
}

impl CustomMetric {
    /// Create new custom metric
    /// 新しいカスタムメトリクスを作成
    pub fn new(name: String, metric_type: MetricType) -> Self {
        Self {
            name,
            metric_type,
            value: 0.0,
            timestamp: Instant::now(),
            history: VecDeque::new(),
            description: None,
            tags: HashMap::new(),
            unit: None,
        }
    }

    /// Update metric value
    /// メトリクス値を更新
    pub fn update(&mut self, value: f64) {
        let now = Instant::now();
        
        match self.metric_type {
            MetricType::Counter => {
                if value >= self.value {
                    self.value = value;
                }
            }
            MetricType::Gauge | MetricType::TimingMs | MetricType::Throughput 
            | MetricType::MemoryBytes | MetricType::CpuPercent | MetricType::GpuPercent
            | MetricType::Custom(_) => {
                self.value = value;
            }
            MetricType::Histogram => {
                // For histograms, we store the cumulative count
                self.value += 1.0;
            }
        }

        self.timestamp = now;
        
        // Store in history (keep last 1000 points)
        self.history.push_back((now, self.value));
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    /// Increment counter
    /// カウンターをインクリメント
    pub fn increment(&mut self, delta: f64) {
        if matches!(self.metric_type, MetricType::Counter) {
            self.update(self.value + delta);
        }
    }

    /// Get rate of change (per second)
    /// 変化率を取得（秒間）
    pub fn get_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let recent = self.history.back()?;
        let older = self.history.get(self.history.len() - 2)?;
        
        let time_diff = recent.0.duration_since(older.0).as_secs_f64();
        if time_diff > 0.0 {
            Some((recent.1 - older.1) / time_diff)
        } else {
            None
        }
    }

    /// Get statistics over time window
    /// 時間窓での統計を取得
    pub fn get_statistics(&self, window: Duration) -> MetricStatistics {
        let cutoff_time = self.timestamp - window;
        let relevant_points: Vec<_> = self.history
            .iter()
            .filter(|(time, _)| *time >= cutoff_time)
            .map(|(_, value)| *value)
            .collect();

        if relevant_points.is_empty() {
            return MetricStatistics {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                sum: 0.0,
                rate: None,
            };
        }

        let count = relevant_points.len();
        let sum: f64 = relevant_points.iter().sum();
        let mean = sum / count as f64;
        let min = relevant_points.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = relevant_points.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let variance = relevant_points
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        MetricStatistics {
            count,
            min,
            max,
            mean,
            std_dev,
            sum,
            rate: self.get_rate(),
        }
    }

    /// Set description
    /// 説明を設定
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add tag
    /// タグを追加
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Set unit
    /// 単位を設定
    pub fn with_unit(mut self, unit: String) -> Self {
        self.unit = Some(unit);
        self
    }
}

/// Metric statistics
/// メトリクス統計
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Number of data points
    /// データポイント数
    pub count: usize,
    /// Minimum value
    /// 最小値
    pub min: f64,
    /// Maximum value
    /// 最大値
    pub max: f64,
    /// Mean value
    /// 平均値
    pub mean: f64,
    /// Standard deviation
    /// 標準偏差
    pub std_dev: f64,
    /// Sum of all values
    /// 全値の合計
    pub sum: f64,
    /// Rate of change (per second)
    /// 変化率（秒間）
    pub rate: Option<f64>,
}

/// Histogram bucket
/// ヒストグラムバケット
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound (inclusive)
    /// 下限（包含）
    pub lower_bound: f64,
    /// Upper bound (exclusive)
    /// 上限（排他）
    pub upper_bound: f64,
    /// Count of values in this bucket
    /// このバケット内の値の数
    pub count: usize,
}

/// Histogram data
/// ヒストグラムデータ
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram buckets
    /// ヒストグラムバケット
    pub buckets: Vec<HistogramBucket>,
    /// Total count
    /// 総数
    pub total_count: usize,
    /// Sum of all values
    /// 全値の合計
    pub sum: f64,
}

impl Histogram {
    /// Create new histogram with predefined buckets
    /// 事前定義バケットで新しいヒストグラムを作成
    pub fn new(bucket_bounds: Vec<f64>) -> Self {
        let mut buckets = Vec::new();
        for i in 0..bucket_bounds.len() {
            let lower = if i == 0 { f64::NEG_INFINITY } else { bucket_bounds[i - 1] };
            let upper = bucket_bounds[i];
            buckets.push(HistogramBucket {
                lower_bound: lower,
                upper_bound: upper,
                count: 0,
            });
        }

        // Add overflow bucket
        if let Some(&last_bound) = bucket_bounds.last() {
            buckets.push(HistogramBucket {
                lower_bound: last_bound,
                upper_bound: f64::INFINITY,
                count: 0,
            });
        }

        Self {
            buckets,
            total_count: 0,
            sum: 0.0,
        }
    }

    /// Add value to histogram
    /// ヒストグラムに値を追加
    pub fn add_value(&mut self, value: f64) {
        self.total_count += 1;
        self.sum += value;

        for bucket in &mut self.buckets {
            if value >= bucket.lower_bound && value < bucket.upper_bound {
                bucket.count += 1;
                break;
            }
        }
    }

    /// Get percentile value
    /// パーセンタイル値を取得
    pub fn get_percentile(&self, percentile: f64) -> Option<f64> {
        if self.total_count == 0 || percentile < 0.0 || percentile > 100.0 {
            return None;
        }

        let target_count = (self.total_count as f64 * percentile / 100.0) as usize;
        let mut cumulative_count = 0;

        for bucket in &self.buckets {
            cumulative_count += bucket.count;
            if cumulative_count >= target_count {
                // Linear interpolation within bucket
                let ratio = if bucket.count > 0 {
                    (target_count - (cumulative_count - bucket.count)) as f64 / bucket.count as f64
                } else {
                    0.0
                };
                
                let lower = if bucket.lower_bound.is_infinite() { 0.0 } else { bucket.lower_bound };
                let upper = if bucket.upper_bound.is_infinite() { 
                    bucket.lower_bound + 1.0 
                } else { 
                    bucket.upper_bound 
                };
                
                return Some(lower + ratio * (upper - lower));
            }
        }

        None
    }
}

/// Advanced metrics collector
/// 高度メトリクス収集器
#[derive(Debug)]
pub struct MetricsCollector {
    /// Custom metrics storage
    /// カスタムメトリクス保存場所
    metrics: Arc<Mutex<HashMap<String, CustomMetric>>>,
    /// Histograms storage
    /// ヒストグラム保存場所
    histograms: Arc<Mutex<HashMap<String, Histogram>>>,
    /// System metrics enabled
    /// システムメトリクス有効
    system_metrics_enabled: bool,
    /// Collection interval
    /// 収集間隔
    collection_interval: Duration,
    /// Last collection time
    /// 最終収集時間
    last_collection: Instant,
}

impl MetricsCollector {
    /// Create new metrics collector
    /// 新しいメトリクス収集器を作成
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            histograms: Arc::new(Mutex::new(HashMap::new())),
            system_metrics_enabled: true,
            collection_interval: Duration::from_secs(1),
            last_collection: Instant::now(),
        }
    }

    /// Enable system metrics collection
    /// システムメトリクス収集を有効化
    pub fn enable_system_metrics(&mut self, enabled: bool) {
        self.system_metrics_enabled = enabled;
    }

    /// Set collection interval
    /// 収集間隔を設定
    pub fn set_collection_interval(&mut self, interval: Duration) {
        self.collection_interval = interval;
    }

    /// Register custom metric
    /// カスタムメトリクスを登録
    pub fn register_metric(&self, metric: CustomMetric) -> RusTorchResult<()> {
        let mut metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        metrics.insert(metric.name.clone(), metric);
        Ok(())
    }

    /// Update metric value
    /// メトリクス値を更新
    pub fn update_metric(&self, name: &str, value: f64) -> RusTorchResult<()> {
        let mut metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        if let Some(metric) = metrics.get_mut(name) {
            metric.update(value);
            Ok(())
        } else {
            Err(RusTorchError::profiling(&format!("Metric '{}' not found", name)))
        }
    }

    /// Increment counter metric
    /// カウンターメトリクスをインクリメント
    pub fn increment_counter(&self, name: &str, delta: f64) -> RusTorchResult<()> {
        let mut metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        if let Some(metric) = metrics.get_mut(name) {
            metric.increment(delta);
            Ok(())
        } else {
            Err(RusTorchError::profiling(&format!("Counter '{}' not found", name)))
        }
    }

    /// Record timing measurement
    /// タイミング測定を記録
    pub fn record_timing(&self, name: &str, duration: Duration) -> RusTorchResult<()> {
        let timing_ms = duration.as_secs_f64() * 1000.0;
        self.update_metric(name, timing_ms)
    }

    /// Create histogram
    /// ヒストグラムを作成
    pub fn create_histogram(&self, name: String, bucket_bounds: Vec<f64>) -> RusTorchResult<()> {
        let mut histograms = self.histograms.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire histograms lock"))?;
        
        let histogram = Histogram::new(bucket_bounds);
        histograms.insert(name, histogram);
        Ok(())
    }

    /// Add value to histogram
    /// ヒストグラムに値を追加
    pub fn add_histogram_value(&self, name: &str, value: f64) -> RusTorchResult<()> {
        let mut histograms = self.histograms.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire histograms lock"))?;
        
        if let Some(histogram) = histograms.get_mut(name) {
            histogram.add_value(value);
            Ok(())
        } else {
            Err(RusTorchError::profiling(&format!("Histogram '{}' not found", name)))
        }
    }

    /// Get metric statistics
    /// メトリクス統計を取得
    pub fn get_metric_statistics(&self, name: &str, window: Duration) -> RusTorchResult<MetricStatistics> {
        let metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        if let Some(metric) = metrics.get(name) {
            Ok(metric.get_statistics(window))
        } else {
            Err(RusTorchError::profiling(&format!("Metric '{}' not found", name)))
        }
    }

    /// Get histogram percentile
    /// ヒストグラムパーセンタイルを取得
    pub fn get_histogram_percentile(&self, name: &str, percentile: f64) -> RusTorchResult<Option<f64>> {
        let histograms = self.histograms.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire histograms lock"))?;
        
        if let Some(histogram) = histograms.get(name) {
            Ok(histogram.get_percentile(percentile))
        } else {
            Err(RusTorchError::profiling(&format!("Histogram '{}' not found", name)))
        }
    }

    /// Collect system metrics
    /// システムメトリクスを収集
    pub fn collect_system_metrics(&self) -> RusTorchResult<()> {
        if !self.system_metrics_enabled {
            return Ok(());
        }

        let now = Instant::now();
        if now.duration_since(self.last_collection) < self.collection_interval {
            return Ok(());
        }

        // Collect CPU usage
        if let Ok(cpu_usage) = self.get_cpu_usage() {
            self.update_metric("system.cpu_percent", cpu_usage)?;
        }

        // Collect memory usage
        if let Ok(memory_usage) = self.get_memory_usage() {
            self.update_metric("system.memory_bytes", memory_usage)?;
        }

        // Collect GPU metrics if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(gpu_usage) = self.get_gpu_usage() {
                self.update_metric("system.gpu_percent", gpu_usage)?;
            }
        }

        Ok(())
    }

    /// Get all metrics snapshot
    /// 全メトリクススナップショットを取得
    pub fn get_all_metrics(&self) -> RusTorchResult<HashMap<String, CustomMetric>> {
        let metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        Ok(metrics.clone())
    }

    /// Clear all metrics
    /// 全メトリクスをクリア
    pub fn clear_metrics(&self) -> RusTorchResult<()> {
        let mut metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        let mut histograms = self.histograms.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire histograms lock"))?;
        
        metrics.clear();
        histograms.clear();
        Ok(())
    }

    /// Export metrics in Prometheus format
    /// Prometheus形式でメトリクスをエクスポート
    pub fn export_prometheus(&self) -> RusTorchResult<String> {
        let metrics = self.metrics.lock()
            .map_err(|_| RusTorchError::profiling("Failed to acquire metrics lock"))?;
        
        let mut output = String::new();
        
        for (name, metric) in metrics.iter() {
            // Add metric help
            if let Some(ref description) = metric.description {
                output.push_str(&format!("# HELP {} {}\n", name, description));
            }
            
            // Add metric type
            let metric_type_str = match metric.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
                _ => "gauge",
            };
            output.push_str(&format!("# TYPE {} {}\n", name, metric_type_str));
            
            // Add metric value
            if metric.tags.is_empty() {
                output.push_str(&format!("{} {}\n", name, metric.value));
            } else {
                let tags: Vec<String> = metric.tags.iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                output.push_str(&format!("{}{{{}}} {}\n", name, tags.join(","), metric.value));
            }
        }
        
        Ok(output)
    }

    // Private helper methods for system metrics collection
    fn get_cpu_usage(&self) -> RusTorchResult<f64> {
        // Simplified CPU usage - in production, use proper system monitoring
        Ok(0.0) // Placeholder
    }

    fn get_memory_usage(&self) -> RusTorchResult<f64> {
        // Simplified memory usage - in production, use proper system monitoring
        Ok(0.0) // Placeholder
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_usage(&self) -> RusTorchResult<f64> {
        // Simplified GPU usage - in production, use CUDA APIs
        Ok(0.0) // Placeholder
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
    fn test_custom_metric_creation() {
        let metric = CustomMetric::new("test_counter".to_string(), MetricType::Counter);
        assert_eq!(metric.name, "test_counter");
        assert_eq!(metric.value, 0.0);
    }

    #[test]
    fn test_counter_increment() {
        let mut metric = CustomMetric::new("counter".to_string(), MetricType::Counter);
        metric.increment(5.0);
        assert_eq!(metric.value, 5.0);
        
        metric.increment(3.0);
        assert_eq!(metric.value, 8.0);
    }

    #[test]
    fn test_gauge_update() {
        let mut metric = CustomMetric::new("gauge".to_string(), MetricType::Gauge);
        metric.update(10.0);
        assert_eq!(metric.value, 10.0);
        
        metric.update(5.0);
        assert_eq!(metric.value, 5.0);
    }

    #[test]
    fn test_histogram_creation() {
        let buckets = vec![1.0, 5.0, 10.0, 50.0, 100.0];
        let histogram = Histogram::new(buckets);
        assert_eq!(histogram.buckets.len(), 6); // 5 buckets + overflow
        assert_eq!(histogram.total_count, 0);
    }

    #[test]
    fn test_histogram_values() {
        let buckets = vec![1.0, 5.0, 10.0];
        let mut histogram = Histogram::new(buckets);
        
        histogram.add_value(0.5);
        histogram.add_value(3.0);
        histogram.add_value(7.0);
        histogram.add_value(15.0);
        
        assert_eq!(histogram.total_count, 4);
        assert_eq!(histogram.buckets[0].count, 1); // 0.5
        assert_eq!(histogram.buckets[1].count, 1); // 3.0
        assert_eq!(histogram.buckets[2].count, 1); // 7.0
        assert_eq!(histogram.buckets[3].count, 1); // 15.0
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        let counter = CustomMetric::new("test_counter".to_string(), MetricType::Counter);
        assert!(collector.register_metric(counter).is_ok());
        
        assert!(collector.increment_counter("test_counter", 1.0).is_ok());
        
        let stats = collector.get_metric_statistics("test_counter", Duration::from_secs(60));
        assert!(stats.is_ok());
    }

    #[test]
    fn test_metric_statistics() {
        let mut metric = CustomMetric::new("test".to_string(), MetricType::Gauge);
        
        metric.update(10.0);
        std::thread::sleep(Duration::from_millis(1));
        metric.update(20.0);
        std::thread::sleep(Duration::from_millis(1));
        metric.update(15.0);
        
        let stats = metric.get_statistics(Duration::from_secs(1));
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 20.0);
        assert!((stats.mean - 15.0).abs() < 0.1);
    }
}