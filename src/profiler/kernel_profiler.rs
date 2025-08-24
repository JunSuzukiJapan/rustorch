//! GPU kernel profiling support
//! GPUカーネルプロファイリングサポート

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::fmt;

/// GPU kernel profiler
/// GPUカーネルプロファイラー
#[derive(Clone)]
pub struct KernelProfiler {
    /// Kernel execution records
    /// カーネル実行記録
    kernel_records: Arc<parking_lot::RwLock<Vec<KernelRecord>>>,
    /// Kernel statistics by name
    /// 名前ごとのカーネル統計
    kernel_stats: Arc<parking_lot::RwLock<HashMap<String, KernelStats>>>,
    /// Active flag
    /// アクティブフラグ
    active: Arc<std::sync::atomic::AtomicBool>,
}

/// Individual kernel execution record
/// 個別カーネル実行記録
#[derive(Debug, Clone)]
pub struct KernelRecord {
    /// Kernel name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// Duration
    pub duration: Duration,
    /// Grid size
    pub grid_size: (u32, u32, u32),
    /// Block size
    pub block_size: (u32, u32, u32),
    /// Shared memory used
    pub shared_memory: usize,
    /// Registers per thread
    pub registers: u32,
    /// Memory throughput (GB/s)
    pub memory_throughput: f64,
    /// Compute utilization (%)
    pub compute_utilization: f64,
}

/// Aggregated kernel statistics
/// 集約されたカーネル統計
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Kernel name
    pub name: String,
    /// Number of launches
    pub launch_count: usize,
    /// Total time
    pub total_time: Duration,
    /// Average time
    pub avg_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Average memory throughput
    pub avg_memory_throughput: f64,
    /// Average compute utilization
    pub avg_compute_utilization: f64,
    /// Total memory transferred
    pub total_memory_transferred: usize,
}

/// Kernel profiling summary
/// カーネルプロファイリングサマリー
#[derive(Debug, Clone)]
pub struct KernelSummary {
    /// Total kernel time
    pub total_kernel_time: Duration,
    /// Number of kernel launches
    pub total_launches: usize,
    /// Top kernels by time
    pub kernels: Vec<KernelStats>,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

impl Default for KernelStats {
    fn default() -> Self {
        Self {
            name: String::new(),
            launch_count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_memory_throughput: 0.0,
            avg_compute_utilization: 0.0,
            total_memory_transferred: 0,
        }
    }
}

impl KernelProfiler {
    /// Create new kernel profiler
    pub fn new() -> Self {
        Self {
            kernel_records: Arc::new(parking_lot::RwLock::new(Vec::new())),
            kernel_stats: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start kernel profiling
    pub fn start(&self) {
        self.active.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Stop kernel profiling
    pub fn stop(&self) {
        self.active.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record kernel execution
    pub fn record_kernel(&self, record: KernelRecord) {
        if !self.active.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        // Store record
        self.kernel_records.write().push(record.clone());

        // Update statistics
        let mut stats_map = self.kernel_stats.write();
        let stats = stats_map.entry(record.name.clone())
            .or_insert_with(|| {
                let mut s = KernelStats::default();
                s.name = record.name.clone();
                s
            });

        stats.launch_count += 1;
        stats.total_time += record.duration;
        stats.avg_time = stats.total_time / stats.launch_count as u32;
        stats.min_time = stats.min_time.min(record.duration);
        stats.max_time = stats.max_time.max(record.duration);
        
        // Update throughput and utilization
        let alpha = 1.0 / stats.launch_count as f64;
        stats.avg_memory_throughput = 
            stats.avg_memory_throughput * (1.0 - alpha) + record.memory_throughput * alpha;
        stats.avg_compute_utilization = 
            stats.avg_compute_utilization * (1.0 - alpha) + record.compute_utilization * alpha;
    }

    /// Get last kernel execution time
    pub fn get_last_kernel_time(&self) -> Option<Duration> {
        self.kernel_records.read()
            .last()
            .map(|r| r.duration)
    }

    /// Get kernel profiling summary
    pub fn get_summary(&self) -> KernelSummary {
        let stats_map = self.kernel_stats.read();
        let mut kernels: Vec<_> = stats_map.values().cloned().collect();
        kernels.sort_by_key(|k| std::cmp::Reverse(k.total_time));

        let total_kernel_time: Duration = kernels.iter()
            .map(|k| k.total_time)
            .sum();
        
        let total_launches: usize = kernels.iter()
            .map(|k| k.launch_count)
            .sum();

        // Calculate utilization
        let gpu_utilization = if !kernels.is_empty() {
            kernels.iter()
                .map(|k| k.avg_compute_utilization * k.launch_count as f64)
                .sum::<f64>() / total_launches as f64
        } else {
            0.0
        };

        let memory_bandwidth_utilization = if !kernels.is_empty() {
            kernels.iter()
                .map(|k| k.avg_memory_throughput * k.launch_count as f64)
                .sum::<f64>() / total_launches as f64 / 1000.0 // Normalize to percentage
        } else {
            0.0
        };

        KernelSummary {
            total_kernel_time,
            total_launches,
            kernels,
            gpu_utilization,
            memory_bandwidth_utilization,
        }
    }

    /// Clear all kernel profiling data
    pub fn clear(&self) {
        self.kernel_records.write().clear();
        self.kernel_stats.write().clear();
    }

    /// Simulate kernel execution (for demonstration)
    pub fn simulate_kernel_execution(
        &self,
        name: &str,
        duration_ms: u64,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
    ) {
        let record = KernelRecord {
            name: name.to_string(),
            start_time: Instant::now(),
            duration: Duration::from_millis(duration_ms),
            grid_size,
            block_size,
            shared_memory: 4096,
            registers: 32,
            memory_throughput: 250.0, // GB/s
            compute_utilization: 85.0, // %
        };
        
        self.record_kernel(record);
    }
}

impl fmt::Display for KernelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  Total Kernel Time: {:.3}ms", 
            self.total_kernel_time.as_secs_f64() * 1000.0)?;
        writeln!(f, "  Total Kernel Launches: {}", self.total_launches)?;
        writeln!(f, "  GPU Utilization: {:.1}%", self.gpu_utilization)?;
        writeln!(f, "  Memory Bandwidth Utilization: {:.1}%", 
            self.memory_bandwidth_utilization)?;
        
        if !self.kernels.is_empty() {
            writeln!(f, "\n  Top GPU Kernels:")?;
            writeln!(f, "  {:<30} {:>10} {:>12} {:>12} {:>10} {:>10}", 
                "Kernel", "Launches", "Total (ms)", "Avg (ms)", "Mem (GB/s)", "Compute %")?;
            writeln!(f, "  {}", "-".repeat(90))?;
            
            for kernel in self.kernels.iter().take(10) {
                writeln!(f, "  {:<30} {:>10} {:>12.3} {:>12.3} {:>10.1} {:>10.1}",
                    if kernel.name.len() > 29 { &kernel.name[..29] } else { &kernel.name },
                    kernel.launch_count,
                    kernel.total_time.as_secs_f64() * 1000.0,
                    kernel.avg_time.as_secs_f64() * 1000.0,
                    kernel.avg_memory_throughput,
                    kernel.avg_compute_utilization)?;
            }
        }
        
        Ok(())
    }
}

/// CUDA event for timing
/// タイミング用CUDAイベント
pub struct CudaEvent {
    /// Event ID
    pub id: usize,
    /// Timestamp
    pub timestamp: Instant,
}

impl CudaEvent {
    /// Create new CUDA event
    pub fn new() -> Self {
        Self {
            id: 0,
            timestamp: Instant::now(),
        }
    }

    /// Record event
    pub fn record(&mut self) {
        self.timestamp = Instant::now();
    }

    /// Elapsed time since another event
    pub fn elapsed_time(&self, other: &CudaEvent) -> Duration {
        self.timestamp.duration_since(other.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_profiler() {
        let profiler = KernelProfiler::new();
        profiler.start();
        
        // Simulate kernel executions
        profiler.simulate_kernel_execution("matmul", 10, (256, 1, 1), (16, 16, 1));
        profiler.simulate_kernel_execution("matmul", 12, (256, 1, 1), (16, 16, 1));
        profiler.simulate_kernel_execution("conv2d", 15, (128, 128, 1), (8, 8, 1));
        
        let summary = profiler.get_summary();
        assert_eq!(summary.total_launches, 3);
        assert!(summary.kernels.len() > 0);
        
        // Check matmul stats
        let matmul_stats = summary.kernels.iter()
            .find(|k| k.name == "matmul")
            .unwrap();
        assert_eq!(matmul_stats.launch_count, 2);
        
        profiler.stop();
    }

    #[test]
    fn test_cuda_events() {
        let mut start_event = CudaEvent::new();
        start_event.record();
        
        std::thread::sleep(Duration::from_millis(10));
        
        let mut end_event = CudaEvent::new();
        end_event.record();
        
        let elapsed = end_event.elapsed_time(&start_event);
        assert!(elapsed >= Duration::from_millis(10));
    }
}