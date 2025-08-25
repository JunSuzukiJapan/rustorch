//! Memory profiling for tensor operations
//! テンソル操作のメモリプロファイリング

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Memory profiler for tracking allocations
/// 割り当て追跡用メモリプロファイラー
#[derive(Clone)]
pub struct MemoryProfiler {
    /// Current allocated memory
    /// 現在割り当てられているメモリ
    current_allocated: Arc<AtomicUsize>,
    /// Peak allocated memory
    /// ピーク割り当てメモリ
    peak_allocated: Arc<AtomicUsize>,
    /// Total allocations
    /// 総割り当て
    total_allocations: Arc<AtomicUsize>,
    /// Total deallocations
    /// 総解放
    total_deallocations: Arc<AtomicUsize>,
    /// Allocation history by operation
    /// 操作ごとの割り当て履歴
    allocation_history: Arc<parking_lot::RwLock<HashMap<String, AllocationStats>>>,
    /// Active flag
    /// アクティブフラグ
    active: Arc<AtomicUsize>,
}

/// Allocation statistics for an operation
/// 操作の割り当て統計
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Operation name
    pub name: String,
    /// Number of allocations
    pub allocation_count: usize,
    /// Total bytes allocated
    pub bytes_allocated: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Total bytes deallocated
    pub bytes_deallocated: usize,
    /// Current live allocations
    pub live_allocations: isize,
    /// Peak memory usage
    pub peak_memory: usize,
}

/// Current memory statistics
/// 現在のメモリ統計
#[derive(Debug, Clone)]
pub struct CurrentMemoryStats {
    /// Currently allocated bytes
    pub allocated: usize,
    /// Currently freed bytes
    pub freed: usize,
    /// Net memory usage
    pub net_usage: isize,
}

/// Memory profiling summary
/// メモリプロファイリングサマリー
#[derive(Debug, Clone)]
pub struct MemorySummary {
    /// Current allocated memory
    pub current_allocated: usize,
    /// Peak allocated memory
    pub peak_allocated: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Total deallocations
    pub total_deallocations: usize,
    /// Top memory consumers
    pub top_consumers: Vec<AllocationStats>,
    /// Memory fragmentation estimate
    pub fragmentation_ratio: f64,
}

impl MemoryProfiler {
    /// Create new memory profiler
    pub fn new() -> Self {
        Self {
            current_allocated: Arc::new(AtomicUsize::new(0)),
            peak_allocated: Arc::new(AtomicUsize::new(0)),
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
            allocation_history: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Start memory profiling
    pub fn start(&self) {
        self.active.store(1, Ordering::SeqCst);
    }

    /// Stop memory profiling
    pub fn stop(&self) {
        self.active.store(0, Ordering::SeqCst);
    }

    /// Record memory allocation
    pub fn record_allocation(&self, size: usize, operation: &str) {
        if self.active.load(Ordering::SeqCst) == 0 {
            return;
        }

        // Update global counters
        let current = self.current_allocated.fetch_add(size, Ordering::SeqCst) + size;
        self.total_allocations.fetch_add(1, Ordering::SeqCst);

        // Update peak if necessary
        let mut peak = self.peak_allocated.load(Ordering::SeqCst);
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        // Update operation-specific stats
        let mut history = self.allocation_history.write();
        let stats = history.entry(operation.to_string()).or_insert_with(|| {
            let mut s = AllocationStats::default();
            s.name = operation.to_string();
            s
        });

        stats.allocation_count += 1;
        stats.bytes_allocated += size;
        stats.live_allocations += 1;
        stats.peak_memory = stats
            .peak_memory
            .max(stats.bytes_allocated - stats.bytes_deallocated);
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, size: usize, operation: &str) {
        if self.active.load(Ordering::SeqCst) == 0 {
            return;
        }

        // Update global counters
        self.current_allocated.fetch_sub(size, Ordering::SeqCst);
        self.total_deallocations.fetch_add(1, Ordering::SeqCst);

        // Update operation-specific stats
        let mut history = self.allocation_history.write();
        if let Some(stats) = history.get_mut(operation) {
            stats.deallocation_count += 1;
            stats.bytes_deallocated += size;
            stats.live_allocations -= 1;
        }
    }

    /// Get current memory statistics
    pub fn get_current_stats(&self) -> CurrentMemoryStats {
        let allocated = self.current_allocated.load(Ordering::SeqCst);
        let total_dealloc = self.total_deallocations.load(Ordering::SeqCst);

        CurrentMemoryStats {
            allocated,
            freed: if total_dealloc > 0 {
                allocated / total_dealloc
            } else {
                0
            },
            net_usage: allocated as isize,
        }
    }

    /// Get memory profiling summary
    pub fn get_summary(&self) -> MemorySummary {
        let history = self.allocation_history.read();
        let mut top_consumers: Vec<_> = history.values().cloned().collect();
        top_consumers.sort_by_key(|s| std::cmp::Reverse(s.bytes_allocated));
        top_consumers.truncate(10);

        let current = self.current_allocated.load(Ordering::SeqCst);
        let peak = self.peak_allocated.load(Ordering::SeqCst);

        // Estimate fragmentation
        let fragmentation_ratio = if peak > 0 {
            current as f64 / peak as f64
        } else {
            0.0
        };

        MemorySummary {
            current_allocated: current,
            peak_allocated: peak,
            total_allocations: self.total_allocations.load(Ordering::SeqCst),
            total_deallocations: self.total_deallocations.load(Ordering::SeqCst),
            top_consumers,
            fragmentation_ratio,
        }
    }

    /// Clear all memory profiling data
    pub fn clear(&self) {
        self.current_allocated.store(0, Ordering::SeqCst);
        self.peak_allocated.store(0, Ordering::SeqCst);
        self.total_allocations.store(0, Ordering::SeqCst);
        self.total_deallocations.store(0, Ordering::SeqCst);
        self.allocation_history.write().clear();
    }

    /// Get memory usage for specific operation
    pub fn get_operation_memory(&self, operation: &str) -> Option<AllocationStats> {
        self.allocation_history.read().get(operation).cloned()
    }
}

impl fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  Current Allocated: {} MB",
            self.current_allocated / 1_048_576
        )?;
        writeln!(
            f,
            "  Peak Allocated: {} MB",
            self.peak_allocated / 1_048_576
        )?;
        writeln!(f, "  Total Allocations: {}", self.total_allocations)?;
        writeln!(f, "  Total Deallocations: {}", self.total_deallocations)?;
        writeln!(
            f,
            "  Fragmentation Ratio: {:.2}%",
            self.fragmentation_ratio * 100.0
        )?;

        if !self.top_consumers.is_empty() {
            writeln!(f, "\n  Top Memory Consumers:")?;
            writeln!(
                f,
                "  {:<30} {:>15} {:>15} {:>10}",
                "Operation", "Allocated (MB)", "Deallocated (MB)", "Live"
            )?;
            writeln!(f, "  {}", "-".repeat(75))?;

            for consumer in &self.top_consumers {
                writeln!(
                    f,
                    "  {:<30} {:>15.2} {:>15.2} {:>10}",
                    if consumer.name.len() > 29 {
                        &consumer.name[..29]
                    } else {
                        &consumer.name
                    },
                    consumer.bytes_allocated as f64 / 1_048_576.0,
                    consumer.bytes_deallocated as f64 / 1_048_576.0,
                    consumer.live_allocations
                )?;
            }
        }

        Ok(())
    }
}

/// Memory snapshot for detailed analysis
/// 詳細分析用メモリスナップショット
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Allocated memory at this point
    pub allocated: usize,
    /// Operation that triggered this snapshot
    pub operation: String,
    /// Call stack at this point
    pub call_stack: Vec<String>,
}

impl MemoryProfiler {
    /// Take a memory snapshot
    pub fn take_snapshot(&self, operation: &str, call_stack: Vec<String>) -> MemorySnapshot {
        MemorySnapshot {
            timestamp: std::time::Instant::now(),
            allocated: self.current_allocated.load(Ordering::SeqCst),
            operation: operation.to_string(),
            call_stack,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();
        profiler.start();

        // Simulate allocations
        profiler.record_allocation(1024, "tensor_create");
        profiler.record_allocation(2048, "tensor_create");
        profiler.record_allocation(4096, "matmul");

        // Check current stats
        let stats = profiler.get_current_stats();
        assert_eq!(stats.allocated, 7168);

        // Simulate deallocation
        profiler.record_deallocation(1024, "tensor_create");

        let stats = profiler.get_current_stats();
        assert_eq!(stats.allocated, 6144);

        // Check summary
        let summary = profiler.get_summary();
        assert_eq!(summary.peak_allocated, 7168);
        assert_eq!(summary.current_allocated, 6144);
        assert_eq!(summary.total_allocations, 3);
        assert_eq!(summary.total_deallocations, 1);

        profiler.stop();
    }

    #[test]
    fn test_operation_tracking() {
        let profiler = MemoryProfiler::new();
        profiler.start();

        profiler.record_allocation(1024, "conv2d");
        profiler.record_allocation(2048, "conv2d");
        profiler.record_deallocation(1024, "conv2d");

        let stats = profiler.get_operation_memory("conv2d").unwrap();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.bytes_allocated, 3072);
        assert_eq!(stats.deallocation_count, 1);
        assert_eq!(stats.bytes_deallocated, 1024);
        assert_eq!(stats.live_allocations, 1);

        profiler.stop();
    }
}
