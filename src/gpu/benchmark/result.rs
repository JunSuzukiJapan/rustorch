//! Benchmark result data structures and utilities
//! ベンチマーク結果データ構造とユーティリティ

/// Benchmark result for a single operation
/// 単一演算のベンチマーク結果
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub device_name: String,
    pub problem_size: String,
    pub cpu_time_ms: Option<f64>,
    pub gpu_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub cpu_throughput_gops: Option<f64>,
    pub gpu_throughput_gops: Option<f64>,
    pub memory_bandwidth_gb_s: Option<f64>,
    pub iterations: usize,
    pub total_flops: Option<u64>,
    pub total_memory_bytes: Option<u64>,
    pub error_message: Option<String>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    /// 新しいベンチマーク結果を作成
    pub fn new(operation_name: String, device_name: String, problem_size: String) -> Self {
        Self {
            operation_name,
            device_name,
            problem_size,
            cpu_time_ms: None,
            gpu_time_ms: None,
            speedup: None,
            cpu_throughput_gops: None,
            gpu_throughput_gops: None,
            memory_bandwidth_gb_s: None,
            iterations: 0,
            total_flops: None,
            total_memory_bytes: None,
            error_message: None,
        }
    }

    /// Add CPU timing information to the benchmark result
    /// ベンチマーク結果にCPUタイミング情報を追加
    pub fn with_cpu_timing(mut self, time_ms: f64, iterations: usize) -> Self {
        self.cpu_time_ms = Some(time_ms);
        self.iterations = iterations;
        if let Some(flops) = self.total_flops {
            self.cpu_throughput_gops = Some((flops as f64 * iterations as f64) / (time_ms * 1e6));
        }
        self
    }

    /// Add GPU timing information to the benchmark result
    /// ベンチマーク結果にGPUタイミング情報を追加
    pub fn with_gpu_timing(mut self, time_ms: f64, iterations: usize) -> Self {
        self.gpu_time_ms = Some(time_ms);
        self.iterations = iterations;
        if let Some(flops) = self.total_flops {
            self.gpu_throughput_gops = Some((flops as f64 * iterations as f64) / (time_ms * 1e6));
        }
        if let Some(bytes) = self.total_memory_bytes {
            self.memory_bandwidth_gb_s = Some((bytes as f64 * iterations as f64) / (time_ms * 1e6));
        }
        if let Some(cpu_time) = self.cpu_time_ms {
            self.speedup = Some(cpu_time / time_ms);
        }
        self
    }

    /// Add FLOPS count to the benchmark result
    /// ベンチマーク結果にFLOPS数を追加
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.total_flops = Some(flops);
        self
    }

    /// Add memory bytes to the benchmark result
    /// ベンチマーク結果にメモリバイト数を追加
    pub fn with_memory_bytes(mut self, bytes: u64) -> Self {
        self.total_memory_bytes = Some(bytes);
        self
    }

    /// Add error message to the benchmark result
    /// ベンチマーク結果にエラーメッセージを追加
    pub fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self
    }
}
