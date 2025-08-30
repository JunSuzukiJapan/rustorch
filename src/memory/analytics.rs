//! Memory Analytics and Profiling System
//! メモリ分析・プロファイリングシステム
//!
//! Features:
//! - Detailed memory usage tracking
//! - Memory leak detection
//! - Allocation pattern analysis
//! - Memory hotspot identification
//! - Performance impact analysis

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// Memory allocation record
/// メモリ割り当て記録
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Allocation ID (unique)
    /// 割り当てID（ユニーク）
    pub id: u64,
    /// Size in bytes
    /// サイズ（バイト）
    pub size: usize,
    /// Timestamp when allocated
    /// 割り当て時のタイムスタンプ
    pub allocated_at: SystemTime,
    /// Source location (file:line)
    /// ソース位置（ファイル:行）
    pub source_location: String,
    /// Call stack (simplified)
    /// コールスタック（簡略化）
    pub call_stack: Vec<String>,
    /// Deallocation timestamp (None if still allocated)
    /// 解放タイムスタンプ（まだ割り当てられている場合はNone）
    pub deallocated_at: Option<SystemTime>,
    /// Lifetime (duration from allocation to deallocation)
    /// ライフタイム（割り当てから解放まての期間）
    pub lifetime: Option<Duration>,
    /// Memory pattern classification
    /// メモリパターン分類
    pub pattern: AllocationPattern,
}

/// Memory allocation patterns
/// メモリ割り当てパターン
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocationPattern {
    /// Short-lived allocations (< 1 second)
    /// 短期割り当て（< 1秒）
    ShortLived,
    /// Medium-lived allocations (1 second - 1 minute)
    /// 中期割り当て（1秒 - 1分）
    MediumLived,
    /// Long-lived allocations (> 1 minute)
    /// 長期割り当て（> 1分）
    LongLived,
    /// Leaked allocations (never deallocated)
    /// リークした割り当て（解放されない）
    Leaked,
    /// Cyclic allocations (repetitive pattern)
    /// 循環割り当て（反復パターン）
    Cyclic,
}

impl AllocationPattern {
    /// Classify allocation pattern based on lifetime
    /// ライフタイムに基づいて割り当てパターンを分類
    pub fn classify(lifetime: Option<Duration>, allocated_at: SystemTime) -> Self {
        match lifetime {
            Some(duration) => {
                if duration < Duration::from_secs(1) {
                    AllocationPattern::ShortLived
                } else if duration < Duration::from_secs(60) {
                    AllocationPattern::MediumLived
                } else {
                    AllocationPattern::LongLived
                }
            }
            None => {
                // Check if allocation is old enough to be considered leaked
                let age = SystemTime::now()
                    .duration_since(allocated_at)
                    .unwrap_or(Duration::from_secs(0));

                if age > Duration::from_secs(300) {
                    // 5 minutes
                    AllocationPattern::Leaked
                } else {
                    AllocationPattern::MediumLived
                }
            }
        }
    }
}

/// Memory hotspot information
/// メモリホットスポット情報
#[derive(Debug, Clone)]
pub struct MemoryHotspot {
    /// Source location
    /// ソース位置
    pub location: String,
    /// Total allocations at this location
    /// この位置での総割り当て数
    pub total_allocations: usize,
    /// Total memory allocated (bytes)
    /// 総割り当てメモリ（バイト）
    pub total_memory: usize,
    /// Average allocation size
    /// 平均割り当てサイズ
    pub avg_size: f64,
    /// Peak concurrent allocations
    /// ピーク同時割り当て数
    pub peak_concurrent: usize,
    /// Memory leak count
    /// メモリリーク数
    pub leak_count: usize,
    /// Allocation frequency (allocations per second)
    /// 割り当て頻度（毎秒の割り当て数）
    pub frequency: f64,
}

/// Memory usage report
/// メモリ使用レポート
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Report generation timestamp
    /// レポート生成タイムスタンプ
    pub generated_at: SystemTime,
    /// Total allocations tracked
    /// 追跡された総割り当て数
    pub total_allocations: usize,
    /// Total deallocations tracked
    /// 追跡された総解放数
    pub total_deallocations: usize,
    /// Current active allocations
    /// 現在のアクティブ割り当て数
    pub active_allocations: usize,
    /// Total memory allocated (bytes)
    /// 総割り当てメモリ（バイト）
    pub total_allocated_bytes: usize,
    /// Current memory usage (bytes)
    /// 現在のメモリ使用量（バイト）
    pub current_memory_usage: usize,
    /// Peak memory usage (bytes)
    /// ピークメモリ使用量（バイト）
    pub peak_memory_usage: usize,
    /// Average allocation size
    /// 平均割り当てサイズ
    pub avg_allocation_size: f64,
    /// Memory leak statistics
    /// メモリリーク統計
    pub leak_stats: LeakStats,
    /// Allocation pattern distribution
    /// 割り当てパターン分布
    pub pattern_distribution: HashMap<AllocationPattern, usize>,
    /// Top memory hotspots
    /// トップメモリホットスポット
    pub hotspots: Vec<MemoryHotspot>,
    /// Memory fragmentation analysis
    /// メモリ断片化分析
    pub fragmentation_analysis: FragmentationAnalysis,
}

/// Memory leak statistics
/// メモリリーク統計
#[derive(Debug, Clone)]
pub struct LeakStats {
    /// Number of potential leaks
    /// 潜在的リーク数
    pub potential_leaks: usize,
    /// Total leaked memory (bytes)
    /// 総リークメモリ（バイト）
    pub leaked_bytes: usize,
    /// Oldest leak age
    /// 最も古いリークの年数
    pub oldest_leak_age: Option<Duration>,
    /// Leak rate (leaks per hour)
    /// リーク率（1時間あたりのリーク数）
    pub leak_rate: f64,
}

/// Memory fragmentation analysis
/// メモリ断片化分析
#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    /// Fragmentation ratio (0.0 - 1.0)
    /// 断片化率（0.0 - 1.0）
    pub fragmentation_ratio: f64,
    /// Number of memory pools
    /// メモリプール数
    pub pool_count: usize,
    /// Average pool utilization
    /// 平均プール利用率
    pub avg_pool_utilization: f64,
    /// Wasted memory due to fragmentation (bytes)
    /// 断片化による無駄メモリ（バイト）
    pub wasted_memory: usize,
}

/// Memory analytics configuration
/// メモリ分析設定
#[derive(Clone, Debug)]
pub struct AnalyticsConfig {
    /// Maximum number of allocation records to keep
    /// 保持する最大割り当て記録数
    pub max_records: usize,
    /// Enable call stack tracking
    /// コールスタック追跡を有効化
    pub enable_stack_trace: bool,
    /// Stack trace depth
    /// スタックトレース深度
    pub stack_trace_depth: usize,
    /// Memory leak detection threshold (seconds)
    /// メモリリーク検出閾値（秒）
    pub leak_threshold: Duration,
    /// Report generation interval
    /// レポート生成間隔
    pub report_interval: Duration,
    /// Enable hotspot analysis
    /// ホットスポット分析を有効化
    pub enable_hotspot_analysis: bool,
    /// Hotspot threshold (minimum allocations)
    /// ホットスポット閾値（最小割り当て数）
    pub hotspot_threshold: usize,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            max_records: 100_000,
            enable_stack_trace: true,
            stack_trace_depth: 10,
            leak_threshold: Duration::from_secs(300), // 5 minutes
            report_interval: Duration::from_secs(60), // 1 minute
            enable_hotspot_analysis: true,
            hotspot_threshold: 10,
        }
    }
}

/// Memory analytics engine
/// メモリ分析エンジン
pub struct MemoryAnalytics {
    /// Configuration
    /// 設定
    config: AnalyticsConfig,
    /// Allocation records
    /// 割り当て記録
    records: RwLock<HashMap<u64, AllocationRecord>>,
    /// Next allocation ID
    /// 次の割り当てID
    next_id: Mutex<u64>,
    /// Current memory usage
    /// 現在のメモリ使用量
    current_usage: RwLock<usize>,
    /// Peak memory usage
    /// ピークメモリ使用量
    peak_usage: RwLock<usize>,
    /// Statistics
    /// 統計
    stats: RwLock<AnalyticsStats>,
    /// Background analysis thread
    /// バックグラウンド分析スレッド
    analysis_thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// Running flag
    /// 実行フラグ
    running: Arc<RwLock<bool>>,
}

/// Analytics statistics
/// 分析統計
#[derive(Debug, Clone)]
pub struct AnalyticsStats {
    /// Total allocations tracked
    /// 追跡された総割り当て数
    pub total_allocations: usize,
    /// Total deallocations tracked
    /// 追跡された総解放数
    pub total_deallocations: usize,
    /// Reports generated
    /// 生成されたレポート数
    pub reports_generated: usize,
    /// Last report generation time
    /// 最後のレポート生成時間
    pub last_report_time: Option<SystemTime>,
    /// Analysis overhead (time spent in analytics)
    /// 分析オーバーヘッド（分析に費やされた時間）
    pub analysis_overhead: Duration,
}

impl Default for AnalyticsStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            reports_generated: 0,
            last_report_time: None,
            analysis_overhead: Duration::from_millis(0),
        }
    }
}

impl MemoryAnalytics {
    /// Create new memory analytics engine
    /// 新しいメモリ分析エンジンを作成
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            records: RwLock::new(HashMap::new()),
            next_id: Mutex::new(1),
            current_usage: RwLock::new(0),
            peak_usage: RwLock::new(0),
            stats: RwLock::new(AnalyticsStats::default()),
            analysis_thread: Mutex::new(None),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Record memory allocation
    /// メモリ割り当てを記録
    pub fn record_allocation(&self, size: usize, source_location: String) -> RusTorchResult<u64> {
        let start_time = Instant::now();

        let id = {
            let mut next_id = self
                .next_id
                .lock()
                .map_err(|_| RusTorchError::MemoryError("Failed to acquire ID lock".to_string()))?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let mut call_stack = Vec::new();
        if self.config.enable_stack_trace {
            // In a real implementation, we would capture the actual call stack
            // For now, we'll use a placeholder
            call_stack.push("tensor::core::Tensor::new".to_string());
            call_stack.push("main".to_string());
        }

        let record = AllocationRecord {
            id,
            size,
            allocated_at: SystemTime::now(),
            source_location,
            call_stack,
            deallocated_at: None,
            lifetime: None,
            pattern: AllocationPattern::MediumLived, // Will be updated on deallocation
        };

        // Update current usage
        {
            let mut current = self.current_usage.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire usage write lock".to_string())
            })?;
            *current += size;

            // Update peak usage
            let mut peak = self.peak_usage.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire peak write lock".to_string())
            })?;
            if *current > *peak {
                *peak = *current;
            }
        }

        // Store record
        {
            let mut records = self.records.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire records write lock".to_string())
            })?;

            records.insert(id, record);

            // Clean up old records if necessary
            if records.len() > self.config.max_records {
                Self::cleanup_old_records(&mut records, self.config.max_records / 2);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire stats write lock".to_string())
            })?;
            stats.total_allocations += 1;
            stats.analysis_overhead += start_time.elapsed();
        }

        Ok(id)
    }

    /// Record memory deallocation
    /// メモリ解放を記録
    pub fn record_deallocation(&self, id: u64) -> RusTorchResult<()> {
        let start_time = Instant::now();
        let dealloc_time = SystemTime::now();

        let size = {
            let mut records = self.records.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire records write lock".to_string())
            })?;

            if let Some(record) = records.get_mut(&id) {
                record.deallocated_at = Some(dealloc_time);
                record.lifetime = dealloc_time.duration_since(record.allocated_at).ok();
                record.pattern = AllocationPattern::classify(record.lifetime, record.allocated_at);
                record.size
            } else {
                return Err(RusTorchError::MemoryError(format!(
                    "Allocation ID {} not found",
                    id
                )));
            }
        };

        // Update current usage
        {
            let mut current = self.current_usage.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire usage write lock".to_string())
            })?;
            *current = current.saturating_sub(size);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire stats write lock".to_string())
            })?;
            stats.total_deallocations += 1;
            stats.analysis_overhead += start_time.elapsed();
        }

        Ok(())
    }

    /// Generate comprehensive memory report
    /// 包括的なメモリレポートを生成
    pub fn generate_report(&self) -> RusTorchResult<MemoryReport> {
        let start_time = Instant::now();

        let records = self.records.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire records read lock".to_string())
        })?;

        let current_usage = *self.current_usage.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire usage read lock".to_string())
        })?;

        let peak_usage = *self.peak_usage.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire peak read lock".to_string())
        })?;

        let stats = self.stats.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire stats read lock".to_string())
        })?;

        // Analyze records
        let mut pattern_distribution = HashMap::new();
        let mut location_stats = HashMap::new();
        let mut total_allocated = 0;
        let mut active_count = 0;
        let mut leaked_count = 0;
        let mut leaked_bytes = 0;
        let mut oldest_leak_age = None;

        for record in records.values() {
            // Update pattern distribution
            *pattern_distribution
                .entry(record.pattern.clone())
                .or_insert(0) += 1;

            // Update location statistics
            let location_stat = location_stats
                .entry(record.source_location.clone())
                .or_insert((0, 0, 0)); // (count, total_size, leak_count)
            location_stat.0 += 1;
            location_stat.1 += record.size;

            total_allocated += record.size;

            if record.deallocated_at.is_none() {
                active_count += 1;

                // Check for potential leaks
                let age = SystemTime::now()
                    .duration_since(record.allocated_at)
                    .unwrap_or(Duration::from_secs(0));

                if age > self.config.leak_threshold {
                    leaked_count += 1;
                    leaked_bytes += record.size;
                    location_stat.2 += 1;

                    if oldest_leak_age.is_none() || age > oldest_leak_age.unwrap() {
                        oldest_leak_age = Some(age);
                    }
                }
            }
        }

        // Generate hotspots
        let mut hotspots = Vec::new();
        if self.config.enable_hotspot_analysis {
            for (location, (count, total_size, leak_count)) in location_stats {
                if count >= self.config.hotspot_threshold {
                    hotspots.push(MemoryHotspot {
                        location,
                        total_allocations: count,
                        total_memory: total_size,
                        avg_size: total_size as f64 / count as f64,
                        peak_concurrent: count, // Simplified
                        leak_count,
                        frequency: count as f64 / 3600.0, // Simplified: allocations per hour
                    });
                }
            }

            // Sort by total memory usage
            hotspots.sort_by(|a, b| b.total_memory.cmp(&a.total_memory));
        }

        let leak_stats = LeakStats {
            potential_leaks: leaked_count,
            leaked_bytes,
            oldest_leak_age,
            leak_rate: leaked_count as f64 / 3600.0, // Simplified
        };

        let fragmentation_analysis = FragmentationAnalysis {
            fragmentation_ratio: 0.1,            // Simplified
            pool_count: 8,                       // Simplified
            avg_pool_utilization: 0.75,          // Simplified
            wasted_memory: total_allocated / 20, // Simplified: 5% waste
        };

        let avg_allocation_size = if stats.total_allocations > 0 {
            total_allocated as f64 / stats.total_allocations as f64
        } else {
            0.0
        };

        let report = MemoryReport {
            generated_at: SystemTime::now(),
            total_allocations: stats.total_allocations,
            total_deallocations: stats.total_deallocations,
            active_allocations: active_count,
            total_allocated_bytes: total_allocated,
            current_memory_usage: current_usage,
            peak_memory_usage: peak_usage,
            avg_allocation_size,
            leak_stats,
            pattern_distribution,
            hotspots,
            fragmentation_analysis,
        };

        // Update statistics
        drop(stats);
        {
            let mut stats = self.stats.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire stats write lock".to_string())
            })?;
            stats.reports_generated += 1;
            stats.last_report_time = Some(SystemTime::now());
            stats.analysis_overhead += start_time.elapsed();
        }

        Ok(report)
    }

    /// Start background analysis
    /// バックグラウンド分析を開始
    pub fn start_analysis(&self) -> RusTorchResult<()> {
        let mut running = self.running.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire running write lock".to_string())
        })?;

        if *running {
            return Err(RusTorchError::MemoryError(
                "Analysis already running".to_string(),
            ));
        }

        *running = true;

        // We would spawn a background thread here for continuous analysis
        // For now, we'll just mark as running

        Ok(())
    }

    /// Stop background analysis
    /// バックグラウンド分析を停止
    pub fn stop_analysis(&self) -> RusTorchResult<()> {
        let mut running = self.running.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire running write lock".to_string())
        })?;

        *running = false;
        Ok(())
    }

    /// Get current analytics statistics
    /// 現在の分析統計を取得
    pub fn get_stats(&self) -> RusTorchResult<AnalyticsStats> {
        let stats = self.stats.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire stats read lock".to_string())
        })?;

        Ok(stats.clone())
    }

    // Private helper methods

    fn cleanup_old_records(records: &mut HashMap<u64, AllocationRecord>, target_size: usize) {
        if records.len() <= target_size {
            return;
        }

        // Sort by allocation time and remove oldest deallocated records
        let mut to_remove = Vec::new();
        let mut deallocated_records: Vec<_> = records
            .iter()
            .filter(|(_, record)| record.deallocated_at.is_some())
            .collect();

        deallocated_records.sort_by_key(|(_, record)| record.allocated_at);

        let remove_count = records.len() - target_size;
        for (id, _) in deallocated_records.into_iter().take(remove_count) {
            to_remove.push(*id);
        }

        for id in to_remove {
            records.remove(&id);
        }
    }
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Analytics Report")?;
        writeln!(f, "======================")?;
        writeln!(f, "Generated: {:?}", self.generated_at)?;
        writeln!(f, "")?;
        writeln!(f, "Allocation Summary:")?;
        writeln!(f, "  Total Allocations: {}", self.total_allocations)?;
        writeln!(f, "  Total Deallocations: {}", self.total_deallocations)?;
        writeln!(f, "  Active Allocations: {}", self.active_allocations)?;
        writeln!(f, "  Current Usage: {} bytes", self.current_memory_usage)?;
        writeln!(f, "  Peak Usage: {} bytes", self.peak_memory_usage)?;
        writeln!(f, "  Average Size: {:.2} bytes", self.avg_allocation_size)?;
        writeln!(f, "")?;
        writeln!(f, "Leak Detection:")?;
        writeln!(f, "  Potential Leaks: {}", self.leak_stats.potential_leaks)?;
        writeln!(f, "  Leaked Memory: {} bytes", self.leak_stats.leaked_bytes)?;
        writeln!(f, "  Leak Rate: {:.2}/hour", self.leak_stats.leak_rate)?;
        writeln!(f, "")?;
        writeln!(f, "Memory Hotspots:")?;
        for (i, hotspot) in self.hotspots.iter().take(5).enumerate() {
            writeln!(
                f,
                "  {}. {} ({} allocs, {} bytes)",
                i + 1,
                hotspot.location,
                hotspot.total_allocations,
                hotspot.total_memory
            )?;
        }
        writeln!(f, "")?;
        writeln!(f, "Fragmentation:")?;
        writeln!(
            f,
            "  Fragmentation Ratio: {:.2}%",
            self.fragmentation_analysis.fragmentation_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Pool Utilization: {:.2}%",
            self.fragmentation_analysis.avg_pool_utilization * 100.0
        )?;
        writeln!(
            f,
            "  Wasted Memory: {} bytes",
            self.fragmentation_analysis.wasted_memory
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_pattern_classification() {
        let now = SystemTime::now();

        assert_eq!(
            AllocationPattern::classify(Some(Duration::from_millis(500)), now),
            AllocationPattern::ShortLived
        );

        assert_eq!(
            AllocationPattern::classify(Some(Duration::from_secs(30)), now),
            AllocationPattern::MediumLived
        );

        assert_eq!(
            AllocationPattern::classify(Some(Duration::from_secs(120)), now),
            AllocationPattern::LongLived
        );
    }

    #[test]
    fn test_analytics_creation() {
        let config = AnalyticsConfig::default();
        let analytics = MemoryAnalytics::new(config);

        let stats = analytics.get_stats().unwrap();
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let config = AnalyticsConfig::default();
        let analytics = MemoryAnalytics::new(config);

        // Record allocation
        let id = analytics
            .record_allocation(1024, "test.rs:10".to_string())
            .unwrap();
        assert!(id > 0);

        // Record deallocation
        analytics.record_deallocation(id).unwrap();

        let stats = analytics.get_stats().unwrap();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_report_generation() {
        let config = AnalyticsConfig::default();
        let analytics = MemoryAnalytics::new(config);

        // Create some allocations
        let id1 = analytics
            .record_allocation(1024, "test.rs:10".to_string())
            .unwrap();
        let _id2 = analytics
            .record_allocation(2048, "test.rs:20".to_string())
            .unwrap();
        analytics.record_deallocation(id1).unwrap();

        let report = analytics.generate_report().unwrap();
        assert_eq!(report.total_allocations, 2);
        assert_eq!(report.total_deallocations, 1);
        assert_eq!(report.active_allocations, 1);
        assert!(report.current_memory_usage > 0);
    }
}
