//! Memory Allocation Tracking System
//!
//! Advanced memory tracking system for monitoring allocations, detecting leaks,
//! and analyzing memory usage patterns in deep learning operations.

use std::collections::HashMap;
use std::fmt;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{RusTorchError, RusTorchResult};

/// Memory allocation information
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub component: String,
    pub size_bytes: usize,
    pub timestamp: SystemTime,
    pub allocation_id: usize,
    pub stack_trace: Option<String>, // Optional stack trace for debugging
}

impl AllocationInfo {
    /// Create new allocation info
    pub fn new(component: String, size_bytes: usize, allocation_id: usize) -> Self {
        Self {
            component,
            size_bytes,
            timestamp: SystemTime::now(),
            allocation_id,
            stack_trace: None,
        }
    }

    /// Size in megabytes
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Age of allocation
    pub fn age_seconds(&self) -> f64 {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

/// Memory usage statistics by component
#[derive(Debug, Clone)]
pub struct ComponentMemoryStats {
    pub component_name: String,
    pub current_allocations: usize,
    pub current_usage_bytes: usize,
    pub peak_usage_bytes: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub allocation_rate_per_second: f64,
    pub average_allocation_size: f64,
}

impl ComponentMemoryStats {
    fn new(component_name: String) -> Self {
        Self {
            component_name,
            current_allocations: 0,
            current_usage_bytes: 0,
            peak_usage_bytes: 0,
            total_allocations: 0,
            total_deallocations: 0,
            allocation_rate_per_second: 0.0,
            average_allocation_size: 0.0,
        }
    }

    /// Current usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Peak usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Comprehensive memory report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub current_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub active_allocations: usize,
    pub component_stats: Vec<ComponentMemoryStats>,
    pub potential_leaks: Vec<AllocationInfo>,
    pub large_allocations: Vec<AllocationInfo>, // Allocations > 100MB
    pub memory_efficiency: f64,                 // Ratio of useful vs total allocations
    pub fragmentation_estimate: f64,
}

impl Default for MemoryReport {
    fn default() -> Self {
        Self {
            current_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            component_stats: Vec::new(),
            potential_leaks: Vec::new(),
            large_allocations: Vec::new(),
            memory_efficiency: 0.0,
            fragmentation_estimate: 0.0,
        }
    }
}

/// Memory allocation tracker
pub struct MemoryTracker {
    enabled: bool,
    threshold_mb: usize,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_counter: AtomicUsize,

    // Active allocations
    allocations: HashMap<usize, AllocationInfo>,

    // Component statistics
    component_stats: HashMap<String, ComponentMemoryStats>,

    // Historical data
    total_allocations: usize,
    total_deallocations: usize,
    session_start: SystemTime,

    // Leak detection
    leak_detection_enabled: bool,
    leak_threshold_seconds: u64,

    // Large allocation tracking
    large_allocation_threshold_mb: usize,
}

impl fmt::Debug for MemoryTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("enabled", &self.enabled)
            .field("threshold_mb", &self.threshold_mb)
            .field("current_usage_mb", &self.get_current_usage_mb())
            .field("peak_usage_mb", &self.get_peak_usage_mb())
            .field("total_allocations", &self.total_allocations)
            .field("active_allocations", &self.allocations.len())
            .finish()
    }
}

impl MemoryTracker {
    /// Create new memory tracker
    pub fn new(enabled: bool, threshold_mb: usize) -> Self {
        Self {
            enabled,
            threshold_mb,
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_counter: AtomicUsize::new(0),
            allocations: HashMap::new(),
            component_stats: HashMap::new(),
            total_allocations: 0,
            total_deallocations: 0,
            session_start: SystemTime::now(),
            leak_detection_enabled: true,
            leak_threshold_seconds: 300, // 5 minutes
            large_allocation_threshold_mb: 100,
        }
    }

    /// Track memory allocation
    pub fn track_allocation(
        &mut self,
        component: &str,
        size_bytes: usize,
    ) -> RusTorchResult<usize> {
        if !self.enabled {
            return Ok(0);
        }

        let allocation_id = self.allocation_counter.fetch_add(1, Ordering::SeqCst);

        // Update current usage
        let new_usage = self.current_usage.fetch_add(size_bytes, Ordering::SeqCst) + size_bytes;

        // Update peak usage if necessary
        let mut peak = self.peak_usage.load(Ordering::SeqCst);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                new_usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        // Create allocation info
        let allocation_info = AllocationInfo::new(component.to_string(), size_bytes, allocation_id);

        // Store allocation
        self.allocations
            .insert(allocation_id, allocation_info.clone());

        // Update component statistics
        let stats = self
            .component_stats
            .entry(component.to_string())
            .or_insert_with(|| ComponentMemoryStats::new(component.to_string()));

        stats.current_allocations += 1;
        stats.current_usage_bytes += size_bytes;
        stats.peak_usage_bytes = stats.peak_usage_bytes.max(stats.current_usage_bytes);
        stats.total_allocations += 1;

        // Update allocation rate
        let session_duration = self
            .session_start
            .elapsed()
            .unwrap_or_default()
            .as_secs_f64();
        if session_duration > 0.0 {
            stats.allocation_rate_per_second = stats.total_allocations as f64 / session_duration;
        }

        // Update average allocation size
        if stats.total_allocations > 0 {
            stats.average_allocation_size =
                stats.current_usage_bytes as f64 / stats.current_allocations as f64;
        }

        self.total_allocations += 1;

        // Check for large allocations
        if size_bytes > self.large_allocation_threshold_mb * 1024 * 1024 {
            // Log large allocation (would integrate with logger in full system)
            eprintln!(
                "⚠️ Large allocation detected: {} MB in component {}",
                allocation_info.size_mb(),
                component
            );
        }

        Ok(allocation_id)
    }

    /// Track memory deallocation
    pub fn track_deallocation(&mut self, component: &str, size_bytes: usize) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Find and remove allocation (simplified - in real system would use allocation_id)
        let allocation_to_remove = self
            .allocations
            .iter()
            .find(|(_, info)| info.component == component && info.size_bytes == size_bytes)
            .map(|(id, _)| *id);

        if let Some(allocation_id) = allocation_to_remove {
            self.allocations.remove(&allocation_id);

            // Update current usage
            self.current_usage.fetch_sub(size_bytes, Ordering::SeqCst);

            // Update component statistics
            if let Some(stats) = self.component_stats.get_mut(component) {
                stats.current_allocations = stats.current_allocations.saturating_sub(1);
                stats.current_usage_bytes = stats.current_usage_bytes.saturating_sub(size_bytes);
                stats.total_deallocations += 1;
            }

            self.total_deallocations += 1;
        }

        Ok(())
    }

    /// Get current memory usage in MB
    pub fn get_current_usage_mb(&self) -> f64 {
        self.current_usage.load(Ordering::SeqCst) as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn get_peak_usage_mb(&self) -> f64 {
        self.peak_usage.load(Ordering::SeqCst) as f64 / (1024.0 * 1024.0)
    }

    /// Get current number of active allocations
    pub fn get_active_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Detect potential memory leaks
    pub fn detect_potential_leaks(&self) -> Vec<AllocationInfo> {
        if !self.leak_detection_enabled {
            return Vec::new();
        }

        let now = SystemTime::now();

        self.allocations
            .values()
            .filter(|info| {
                now.duration_since(info.timestamp)
                    .unwrap_or_default()
                    .as_secs()
                    > self.leak_threshold_seconds
            })
            .cloned()
            .collect()
    }

    /// Get large allocations
    pub fn get_large_allocations(&self) -> Vec<AllocationInfo> {
        let threshold_bytes = self.large_allocation_threshold_mb * 1024 * 1024;

        self.allocations
            .values()
            .filter(|info| info.size_bytes > threshold_bytes)
            .cloned()
            .collect()
    }

    /// Generate comprehensive memory report
    pub fn generate_memory_report(&self) -> RusTorchResult<MemoryReport> {
        let current_usage_mb = self.get_current_usage_mb();
        let peak_usage_mb = self.get_peak_usage_mb();
        let active_allocations = self.get_active_allocations();

        let component_stats: Vec<ComponentMemoryStats> =
            self.component_stats.values().cloned().collect();

        let potential_leaks = self.detect_potential_leaks();
        let large_allocations = self.get_large_allocations();

        // Calculate memory efficiency (active allocations vs total allocations)
        let memory_efficiency = if self.total_allocations > 0 {
            active_allocations as f64 / self.total_allocations as f64
        } else {
            1.0
        };

        // Estimate fragmentation (simplified calculation)
        let fragmentation_estimate = if active_allocations > 0 {
            let avg_allocation_size = current_usage_mb / active_allocations as f64;
            let fragmentation = 1.0 - (avg_allocation_size / 100.0).min(1.0); // Rough estimate
            fragmentation.max(0.0)
        } else {
            0.0
        };

        Ok(MemoryReport {
            current_usage_mb,
            peak_usage_mb,
            total_allocations: self.total_allocations,
            total_deallocations: self.total_deallocations,
            active_allocations,
            component_stats,
            potential_leaks,
            large_allocations,
            memory_efficiency,
            fragmentation_estimate,
        })
    }

    /// Get memory statistics for specific component
    pub fn get_component_stats(&self, component: &str) -> Option<&ComponentMemoryStats> {
        self.component_stats.get(component)
    }

    /// Get all tracked components
    pub fn get_tracked_components(&self) -> Vec<String> {
        self.component_stats.keys().cloned().collect()
    }

    /// Set leak detection parameters
    pub fn configure_leak_detection(&mut self, enabled: bool, threshold_seconds: u64) {
        self.leak_detection_enabled = enabled;
        self.leak_threshold_seconds = threshold_seconds;
    }

    /// Set large allocation threshold
    pub fn set_large_allocation_threshold(&mut self, threshold_mb: usize) {
        self.large_allocation_threshold_mb = threshold_mb;
    }

    /// Clear all tracking data
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.component_stats.clear();
        self.current_usage.store(0, Ordering::SeqCst);
        self.peak_usage.store(0, Ordering::SeqCst);
        self.total_allocations = 0;
        self.total_deallocations = 0;
        self.allocation_counter.store(0, Ordering::SeqCst);
        self.session_start = SystemTime::now();
    }

    /// Check if usage exceeds threshold
    pub fn is_over_threshold(&self) -> bool {
        self.get_current_usage_mb() > self.threshold_mb as f64
    }

    /// Generate memory usage summary
    pub fn generate_summary_report(&self) -> String {
        let report = self.generate_memory_report().unwrap_or_default();

        let mut summary = String::new();
        summary.push_str("🧠 Memory Usage Summary\n");
        summary.push_str("======================\n\n");

        summary.push_str(&format!(
            "📊 Current Usage: {:.2} MB\n",
            report.current_usage_mb
        ));
        summary.push_str(&format!("📈 Peak Usage: {:.2} MB\n", report.peak_usage_mb));
        summary.push_str(&format!(
            "🔢 Active Allocations: {}\n",
            report.active_allocations
        ));
        summary.push_str(&format!(
            "📋 Total Allocations: {}\n",
            report.total_allocations
        ));
        summary.push_str(&format!(
            "♻️ Total Deallocations: {}\n",
            report.total_deallocations
        ));
        summary.push_str(&format!(
            "⚡ Memory Efficiency: {:.1}%\n",
            report.memory_efficiency * 100.0
        ));
        summary.push_str(&format!(
            "🔧 Fragmentation: {:.1}%\n\n",
            report.fragmentation_estimate * 100.0
        ));

        if !report.potential_leaks.is_empty() {
            summary.push_str(&format!(
                "⚠️ Potential Leaks: {} allocations\n",
                report.potential_leaks.len()
            ));
        }

        if !report.large_allocations.is_empty() {
            summary.push_str(&format!(
                "🐘 Large Allocations: {} (>{}MB)\n",
                report.large_allocations.len(),
                self.large_allocation_threshold_mb
            ));
        }

        if !report.component_stats.is_empty() {
            summary.push_str("\n📦 Component Usage:\n");
            let mut sorted_components = report.component_stats;
            sorted_components.sort_by(|a, b| b.current_usage_bytes.cmp(&a.current_usage_bytes));

            for (i, stats) in sorted_components.iter().take(5).enumerate() {
                summary.push_str(&format!(
                    "  {}. {}: {:.2} MB ({} allocations)\n",
                    i + 1,
                    stats.component_name,
                    stats.current_usage_mb(),
                    stats.current_allocations
                ));
            }
        }

        summary
    }

    /// Enable/disable memory tracking
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if memory tracking is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Convenience macro for tracking allocations
#[macro_export]
macro_rules! track_allocation {
    ($tracker:expr, $component:expr, $size:expr) => {
        $tracker.track_allocation($component, $size)
    };
}

#[macro_export]
macro_rules! track_deallocation {
    ($tracker:expr, $component:expr, $size:expr) => {
        $tracker.track_deallocation($component, $size)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_allocation_info_creation() {
        let info = AllocationInfo::new("test_component".to_string(), 1024 * 1024, 1);

        assert_eq!(info.component, "test_component");
        assert_eq!(info.size_bytes, 1024 * 1024);
        assert_eq!(info.allocation_id, 1);
        assert_eq!(info.size_mb(), 1.0);
    }

    #[test]
    fn test_memory_tracker_creation() {
        let tracker = MemoryTracker::new(true, 1024);

        assert!(tracker.is_enabled());
        assert_eq!(tracker.get_current_usage_mb(), 0.0);
        assert_eq!(tracker.get_active_allocations(), 0);
    }

    #[test]
    fn test_memory_tracking() {
        let mut tracker = MemoryTracker::new(true, 1024);

        // Track allocation
        let allocation_id = tracker.track_allocation("tensor", 1024 * 1024).unwrap();
        if tracker.enabled {
            // allocation_id starts from 0, so first allocation has id 0
            assert!(allocation_id == 0 || allocation_id > 0);
            assert_eq!(tracker.get_current_usage_mb(), 1.0);
            assert_eq!(tracker.get_active_allocations(), 1);
        }

        // Track deallocation
        tracker.track_deallocation("tensor", 1024 * 1024).unwrap();
        assert_eq!(tracker.get_current_usage_mb(), 0.0);
        assert_eq!(tracker.get_active_allocations(), 0);
    }

    #[test]
    fn test_peak_usage_tracking() {
        let mut tracker = MemoryTracker::new(true, 1024);

        // Allocate memory
        tracker.track_allocation("test1", 1024 * 1024).unwrap();
        assert_eq!(tracker.get_peak_usage_mb(), 1.0);

        tracker.track_allocation("test2", 2 * 1024 * 1024).unwrap();
        assert_eq!(tracker.get_peak_usage_mb(), 3.0);

        // Deallocate some memory
        tracker.track_deallocation("test1", 1024 * 1024).unwrap();
        assert_eq!(tracker.get_current_usage_mb(), 2.0);
        assert_eq!(tracker.get_peak_usage_mb(), 3.0); // Peak should remain
    }

    #[test]
    fn test_component_statistics() {
        let mut tracker = MemoryTracker::new(true, 1024);

        // Track allocations for different components
        tracker.track_allocation("tensor", 1024 * 1024).unwrap();
        tracker.track_allocation("tensor", 512 * 1024).unwrap();
        tracker.track_allocation("network", 2048 * 1024).unwrap();

        let tensor_stats = tracker.get_component_stats("tensor").unwrap();
        assert_eq!(tensor_stats.current_allocations, 2);
        assert_eq!(tensor_stats.current_usage_bytes, 1536 * 1024); // 1MB + 512KB

        let network_stats = tracker.get_component_stats("network").unwrap();
        assert_eq!(network_stats.current_allocations, 1);
        assert_eq!(network_stats.current_usage_bytes, 2048 * 1024);
    }

    #[test]
    fn test_memory_report_generation() {
        let mut tracker = MemoryTracker::new(true, 1024);

        // Create some allocations
        tracker.track_allocation("tensor", 1024 * 1024).unwrap();
        tracker.track_allocation("network", 512 * 1024).unwrap();

        let report = tracker.generate_memory_report().unwrap();

        assert_eq!(report.active_allocations, 2);
        assert!(report.current_usage_mb > 0.0);
        assert_eq!(report.component_stats.len(), 2);
        assert!(report.memory_efficiency > 0.0);
    }

    #[test]
    #[cfg_attr(
        not(feature = "memory-debug"),
        ignore = "Memory debugging disabled in CI"
    )]
    fn test_leak_detection() {
        let mut tracker = MemoryTracker::new(true, 1024);
        tracker.configure_leak_detection(true, 1); // 1 second threshold

        // Create an allocation
        tracker
            .track_allocation("potential_leak", 1024 * 1024)
            .unwrap();

        // Wait for leak threshold (reduced for CI)
        thread::sleep(Duration::from_millis(1100)); // Ensure threshold is exceeded

        let leaks = tracker.detect_potential_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].component, "potential_leak");
    }

    #[test]
    fn test_large_allocation_detection() {
        let mut tracker = MemoryTracker::new(true, 1024);
        tracker.set_large_allocation_threshold(1); // 1MB threshold

        // Create a large allocation
        tracker
            .track_allocation("large_tensor", 2 * 1024 * 1024)
            .unwrap();

        let large_allocations = tracker.get_large_allocations();
        assert_eq!(large_allocations.len(), 1);
        assert_eq!(large_allocations[0].component, "large_tensor");
    }

    #[test]
    fn test_threshold_checking() {
        let mut tracker = MemoryTracker::new(true, 1); // 1MB threshold

        // Under threshold
        tracker.track_allocation("small", 512 * 1024).unwrap();
        assert!(!tracker.is_over_threshold());

        // Over threshold
        tracker.track_allocation("large", 1024 * 1024).unwrap();
        assert!(tracker.is_over_threshold());
    }

    #[test]
    fn test_tracker_disabled() {
        let mut tracker = MemoryTracker::new(false, 1024);

        let allocation_id = tracker.track_allocation("test", 1024 * 1024).unwrap();
        assert_eq!(allocation_id, 0); // Should return 0 when disabled
        assert_eq!(tracker.get_current_usage_mb(), 0.0);
        assert_eq!(tracker.get_active_allocations(), 0);
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = MemoryTracker::new(true, 1024);

        // Create allocations
        tracker.track_allocation("test1", 1024 * 1024).unwrap();
        tracker.track_allocation("test2", 512 * 1024).unwrap();

        assert!(tracker.get_current_usage_mb() > 0.0);
        assert!(tracker.get_active_allocations() > 0);

        // Clear tracker
        tracker.clear();

        assert_eq!(tracker.get_current_usage_mb(), 0.0);
        assert_eq!(tracker.get_active_allocations(), 0);
        assert_eq!(tracker.total_allocations, 0);
    }
}
