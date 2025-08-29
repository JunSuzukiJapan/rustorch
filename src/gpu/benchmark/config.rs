//! Benchmark configuration for GPU performance testing
//! GPU性能テスト用ベンチマーク設定

/// Benchmark configuration
/// ベンチマーク設定
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    /// ウォームアップ反復回数
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    /// 測定反復回数
    pub measurement_iterations: usize,
    /// Minimum benchmark duration in milliseconds
    /// 最小ベンチマーク時間（ミリ秒）
    pub min_duration_ms: u64,
    /// Maximum benchmark duration in milliseconds
    /// 最大ベンチマーク時間（ミリ秒）
    pub max_duration_ms: u64,
    /// Enable memory bandwidth measurements
    /// メモリ帯域幅測定を有効化
    pub measure_memory_bandwidth: bool,
    /// Enable FLOPS measurements
    /// FLOPS測定を有効化
    pub measure_flops: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 100,
            min_duration_ms: 1000,
            max_duration_ms: 30000,
            measure_memory_bandwidth: true,
            measure_flops: true,
        }
    }
}
