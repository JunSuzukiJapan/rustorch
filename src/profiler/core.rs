//! Core Profiling Engine
//! コアプロファイリングエンジン

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Profiling level configuration
/// プロファイリングレベル設定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingLevel {
    /// Disabled profiling
    /// プロファイリング無効
    Disabled,
    /// Basic timing only
    /// 基本的なタイミングのみ
    Basic,
    /// Standard profiling with memory tracking
    /// メモリトラッキング付き標準プロファイリング
    Standard,
    /// Comprehensive profiling with GPU and system metrics
    /// GPU・システムメトリクス付き包括的プロファイリング
    Comprehensive,
    /// Verbose profiling with detailed analysis
    /// 詳細分析付き詳細プロファイリング
    Verbose,
}

impl Default for ProfilingLevel {
    fn default() -> Self {
        ProfilingLevel::Standard
    }
}

/// Profiler configuration
/// プロファイラー設定
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Profiling level
    /// プロファイリングレベル
    pub level: ProfilingLevel,
    /// Enable memory profiling
    /// メモリプロファイリングを有効化
    pub enable_memory_profiling: bool,
    /// Enable GPU profiling
    /// GPUプロファイリングを有効化
    pub enable_gpu_profiling: bool,
    /// Enable system metrics
    /// システムメトリクスを有効化
    pub enable_system_metrics: bool,
    /// Enable call stack tracking
    /// コールスタックトラッキングを有効化
    pub enable_call_stack: bool,
    /// Maximum profiling session duration (seconds)
    /// 最大プロファイリングセッション時間（秒）
    pub max_session_duration: Option<u64>,
    /// Buffer size for metrics collection
    /// メトリクス収集用バッファサイズ
    pub metrics_buffer_size: usize,
    /// Sampling rate for continuous monitoring (Hz)
    /// 連続監視のサンプリングレート（Hz）
    pub sampling_rate: f64,
    /// Export format options
    /// エクスポート形式オプション
    pub export_chrome_trace: bool,
    pub export_tensorboard: bool,
    pub export_json: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            level: ProfilingLevel::Standard,
            enable_memory_profiling: true,
            enable_gpu_profiling: true,
            enable_system_metrics: true,
            enable_call_stack: true,
            max_session_duration: Some(3600), // 1 hour
            metrics_buffer_size: 10000,
            sampling_rate: 10.0, // 10 Hz
            export_chrome_trace: true,
            export_tensorboard: false,
            export_json: true,
        }
    }
}

/// Profiling session state
/// プロファイリングセッション状態
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    /// Not started
    /// 未開始
    NotStarted,
    /// Running
    /// 実行中
    Running,
    /// Paused
    /// 一時停止
    Paused,
    /// Completed
    /// 完了
    Completed,
    /// Error occurred
    /// エラー発生
    Error,
}

/// Profiling session
/// プロファイリングセッション
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    /// Session ID
    /// セッションID
    pub session_id: String,
    /// Session name
    /// セッション名
    pub session_name: String,
    /// Session state
    /// セッション状態
    pub state: SessionState,
    /// Start time
    /// 開始時間
    pub start_time: Instant,
    /// End time
    /// 終了時間
    pub end_time: Option<Instant>,
    /// Configuration
    /// 設定
    pub config: ProfilerConfig,
    /// Operations recorded
    /// 記録された操作
    pub operations: HashMap<String, OperationMetrics>,
    /// Call stack depth
    /// コールスタック深度
    pub max_call_depth: usize,
    /// Total operations
    /// 総操作数
    pub total_operations: usize,
    /// Error message if any
    /// エラーメッセージ（ある場合）
    pub error_message: Option<String>,
}

impl ProfilingSession {
    /// Create new profiling session
    /// 新しいプロファイリングセッションを作成
    pub fn new(name: String, config: ProfilerConfig) -> Self {
        let session_id = generate_session_id();
        Self {
            session_id,
            session_name: name,
            state: SessionState::NotStarted,
            start_time: Instant::now(),
            end_time: None,
            config,
            operations: HashMap::new(),
            max_call_depth: 0,
            total_operations: 0,
            error_message: None,
        }
    }

    /// Start the session
    /// セッションを開始
    pub fn start(&mut self) -> RusTorchResult<()> {
        if self.state != SessionState::NotStarted {
            return Err(RusTorchError::profiling("Session already started"));
        }
        
        self.state = SessionState::Running;
        self.start_time = Instant::now();
        Ok(())
    }

    /// Stop the session
    /// セッションを停止
    pub fn stop(&mut self) -> RusTorchResult<SessionSnapshot> {
        if self.state != SessionState::Running {
            return Err(RusTorchError::profiling("Session not running"));
        }
        
        self.state = SessionState::Completed;
        self.end_time = Some(Instant::now());
        
        Ok(self.create_snapshot())
    }

    /// Pause the session
    /// セッションを一時停止
    pub fn pause(&mut self) -> RusTorchResult<()> {
        if self.state != SessionState::Running {
            return Err(RusTorchError::profiling("Session not running"));
        }
        
        self.state = SessionState::Paused;
        Ok(())
    }

    /// Resume the session
    /// セッションを再開
    pub fn resume(&mut self) -> RusTorchResult<()> {
        if self.state != SessionState::Paused {
            return Err(RusTorchError::profiling("Session not paused"));
        }
        
        self.state = SessionState::Running;
        Ok(())
    }

    /// Record operation
    /// 操作を記録
    pub fn record_operation(&mut self, name: &str, duration: Duration, call_depth: usize) {
        if self.state != SessionState::Running {
            return;
        }

        let metrics = self.operations.entry(name.to_string()).or_insert_with(|| {
            OperationMetrics::new(name.to_string())
        });

        metrics.record_timing(duration);
        self.max_call_depth = self.max_call_depth.max(call_depth);
        self.total_operations += 1;
    }

    /// Get session duration
    /// セッション期間を取得
    pub fn duration(&self) -> Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Create session snapshot
    /// セッションスナップショットを作成
    pub fn create_snapshot(&self) -> SessionSnapshot {
        let operations: Vec<_> = self.operations.values().cloned().collect();
        
        SessionSnapshot {
            session_id: self.session_id.clone(),
            session_name: self.session_name.clone(),
            start_time: self.start_time,
            duration: self.duration(),
            operations,
            total_operations: self.total_operations,
            max_call_depth: self.max_call_depth,
            config: self.config.clone(),
        }
    }
}

/// Operation performance metrics
/// 操作パフォーマンスメトリクス
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name
    /// 操作名
    pub name: String,
    /// Call count
    /// 呼び出し回数
    pub call_count: usize,
    /// Total time
    /// 総時間
    pub total_time: Duration,
    /// Average time
    /// 平均時間
    pub avg_time: Duration,
    /// Minimum time
    /// 最小時間
    pub min_time: Duration,
    /// Maximum time
    /// 最大時間
    pub max_time: Duration,
    /// Standard deviation
    /// 標準偏差
    pub std_dev: f64,
    /// Timing samples
    /// タイミングサンプル
    pub timing_samples: Vec<Duration>,
    /// CPU percentage
    /// CPU使用率
    pub cpu_percentage: Option<f64>,
    /// Memory usage (bytes)
    /// メモリ使用量（バイト）
    pub memory_usage: Option<u64>,
    /// GPU time (if applicable)
    /// GPU時間（該当する場合）
    pub gpu_time: Option<Duration>,
}

impl OperationMetrics {
    /// Create new operation metrics
    /// 新しい操作メトリクスを作成
    pub fn new(name: String) -> Self {
        Self {
            name,
            call_count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            std_dev: 0.0,
            timing_samples: Vec::new(),
            cpu_percentage: None,
            memory_usage: None,
            gpu_time: None,
        }
    }

    /// Record timing measurement
    /// タイミング測定を記録
    pub fn record_timing(&mut self, duration: Duration) {
        self.call_count += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        
        // Update average
        self.avg_time = self.total_time / self.call_count as u32;
        
        // Store sample for statistics
        self.timing_samples.push(duration);
        if self.timing_samples.len() > 1000 {
            // Keep only recent samples
            self.timing_samples.drain(0..500);
        }
        
        // Update standard deviation
        self.update_std_dev();
    }

    /// Update standard deviation
    /// 標準偏差を更新
    fn update_std_dev(&mut self) {
        if self.timing_samples.len() < 2 {
            return;
        }

        let avg_secs = self.avg_time.as_secs_f64();
        let variance: f64 = self.timing_samples
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - avg_secs;
                diff * diff
            })
            .sum::<f64>() / (self.timing_samples.len() - 1) as f64;

        self.std_dev = variance.sqrt();
    }

    /// Get performance statistics
    /// パフォーマンス統計を取得
    pub fn get_statistics(&self) -> PerformanceStatistics {
        PerformanceStatistics {
            operation_name: self.name.clone(),
            call_count: self.call_count,
            total_time_ms: self.total_time.as_secs_f64() * 1000.0,
            avg_time_ms: self.avg_time.as_secs_f64() * 1000.0,
            min_time_ms: self.min_time.as_secs_f64() * 1000.0,
            max_time_ms: self.max_time.as_secs_f64() * 1000.0,
            std_dev_ms: self.std_dev * 1000.0,
            throughput_ops_per_sec: if self.avg_time.as_secs_f64() > 0.0 {
                1.0 / self.avg_time.as_secs_f64()
            } else {
                0.0
            },
            cpu_percentage: self.cpu_percentage.unwrap_or(0.0),
            memory_usage_mb: self.memory_usage.unwrap_or(0) as f64 / (1024.0 * 1024.0),
            gpu_time_ms: self.gpu_time.map(|d| d.as_secs_f64() * 1000.0),
        }
    }
}

/// Session snapshot for reporting
/// レポート用セッションスナップショット
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    /// Session ID
    /// セッションID
    pub session_id: String,
    /// Session name
    /// セッション名
    pub session_name: String,
    /// Start time
    /// 開始時間
    pub start_time: Instant,
    /// Total duration
    /// 総継続時間
    pub duration: Duration,
    /// Operation metrics
    /// 操作メトリクス
    pub operations: Vec<OperationMetrics>,
    /// Total operations count
    /// 総操作数
    pub total_operations: usize,
    /// Maximum call depth observed
    /// 観測された最大コール深度
    pub max_call_depth: usize,
    /// Configuration used
    /// 使用された設定
    pub config: ProfilerConfig,
}

/// Performance statistics summary
/// パフォーマンス統計サマリー
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Operation name
    /// 操作名
    pub operation_name: String,
    /// Number of calls
    /// 呼び出し回数
    pub call_count: usize,
    /// Total time (ms)
    /// 総時間（ミリ秒）
    pub total_time_ms: f64,
    /// Average time per call (ms)
    /// 呼び出しごとの平均時間（ミリ秒）
    pub avg_time_ms: f64,
    /// Minimum time (ms)
    /// 最小時間（ミリ秒）
    pub min_time_ms: f64,
    /// Maximum time (ms)
    /// 最大時間（ミリ秒）
    pub max_time_ms: f64,
    /// Standard deviation (ms)
    /// 標準偏差（ミリ秒）
    pub std_dev_ms: f64,
    /// Throughput (operations per second)
    /// スループット（秒間操作数）
    pub throughput_ops_per_sec: f64,
    /// CPU usage percentage
    /// CPU使用率
    pub cpu_percentage: f64,
    /// Memory usage (MB)
    /// メモリ使用量（MB）
    pub memory_usage_mb: f64,
    /// GPU time if applicable (ms)
    /// GPU時間（該当する場合、ミリ秒）
    pub gpu_time_ms: Option<f64>,
}

/// Core profiler engine
/// コアプロファイラーエンジン
#[derive(Debug)]
pub struct ProfilerCore {
    /// Current session
    /// 現在のセッション
    current_session: Option<ProfilingSession>,
    /// Configuration
    /// 設定
    config: ProfilerConfig,
    /// Active timers
    /// アクティブタイマー
    active_timers: HashMap<String, Instant>,
    /// Call stack
    /// コールスタック
    call_stack: Vec<String>,
    /// Session history
    /// セッション履歴
    session_history: Vec<SessionSnapshot>,
}

impl ProfilerCore {
    /// Create new profiler core
    /// 新しいプロファイラーコアを作成
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            current_session: None,
            config,
            active_timers: HashMap::new(),
            call_stack: Vec::new(),
            session_history: Vec::new(),
        }
    }

    /// Start profiling session
    /// プロファイリングセッションを開始
    pub fn start_session(&mut self, name: String) -> RusTorchResult<()> {
        if self.current_session.is_some() {
            return Err(RusTorchError::profiling("Session already active"));
        }

        let mut session = ProfilingSession::new(name, self.config.clone());
        session.start()?;
        self.current_session = Some(session);
        
        Ok(())
    }

    /// Stop current session
    /// 現在のセッションを停止
    pub fn stop_session(&mut self) -> RusTorchResult<SessionSnapshot> {
        let session = self.current_session.as_mut()
            .ok_or_else(|| RusTorchError::profiling("No active session"))?;
        
        let snapshot = session.stop()?;
        self.session_history.push(snapshot.clone());
        self.current_session = None;
        self.call_stack.clear();
        self.active_timers.clear();
        
        Ok(snapshot)
    }

    /// Start timing operation
    /// 操作タイミング開始
    pub fn start_timer(&mut self, name: String) -> RusTorchResult<()> {
        if let Some(session) = &self.current_session {
            if session.state != SessionState::Running {
                return Err(RusTorchError::profiling("Session not running"));
            }
        } else {
            return Err(RusTorchError::profiling("No active session"));
        }

        self.active_timers.insert(name.clone(), Instant::now());
        self.call_stack.push(name);
        
        Ok(())
    }

    /// Stop timing operation
    /// 操作タイミング停止
    pub fn stop_timer(&mut self, name: &str) -> RusTorchResult<f64> {
        let start_time = self.active_timers.remove(name)
            .ok_or_else(|| RusTorchError::profiling("Timer not found"))?;

        let duration = start_time.elapsed();
        
        if let Some(session) = &mut self.current_session {
            session.record_operation(name, duration, self.call_stack.len());
        }

        // Remove from call stack
        if let Some(pos) = self.call_stack.iter().rposition(|x| x == name) {
            self.call_stack.remove(pos);
        }

        Ok(duration.as_secs_f64() * 1000.0) // Return milliseconds
    }

    /// Record custom metric
    /// カスタムメトリクスを記録
    pub fn record_custom_metric(&mut self, name: &str, value: f64, metric_type: super::metrics_collector::MetricType) -> RusTorchResult<()> {
        if self.current_session.is_none() {
            return Err(RusTorchError::profiling("No active session"));
        }

        // For now, store as operation with the value as duration in nanoseconds
        let duration = Duration::from_nanos((value * 1_000_000.0) as u64);
        
        if let Some(session) = &mut self.current_session {
            session.record_operation(name, duration, self.call_stack.len());
        }

        Ok(())
    }

    /// Get current session statistics
    /// 現在のセッション統計を取得
    pub fn get_current_statistics(&self) -> Option<Vec<PerformanceStatistics>> {
        self.current_session.as_ref().map(|session| {
            session.operations.values()
                .map(|op| op.get_statistics())
                .collect()
        })
    }

    /// Get session history
    /// セッション履歴を取得
    pub fn get_session_history(&self) -> &[SessionSnapshot] {
        &self.session_history
    }

    /// Clear session history
    /// セッション履歴をクリア
    pub fn clear_history(&mut self) {
        self.session_history.clear();
    }

    /// Get current configuration
    /// 現在の設定を取得
    pub fn get_config(&self) -> &ProfilerConfig {
        &self.config
    }

    /// Update configuration
    /// 設定を更新
    pub fn update_config(&mut self, config: ProfilerConfig) -> RusTorchResult<()> {
        if self.current_session.is_some() {
            return Err(RusTorchError::profiling("Cannot update config during active session"));
        }

        self.config = config;
        Ok(())
    }
}

/// Generate unique session ID
/// 一意のセッションIDを生成
fn generate_session_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("session_{}", timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_core_creation() {
        let config = ProfilerConfig::default();
        let profiler = ProfilerCore::new(config);
        assert!(profiler.current_session.is_none());
    }

    #[test]
    fn test_session_lifecycle() {
        let config = ProfilerConfig::default();
        let mut profiler = ProfilerCore::new(config);

        // Start session
        assert!(profiler.start_session("test".to_string()).is_ok());
        assert!(profiler.current_session.is_some());

        // Stop session
        let snapshot = profiler.stop_session();
        assert!(snapshot.is_ok());
        assert!(profiler.current_session.is_none());
        assert_eq!(profiler.session_history.len(), 1);
    }

    #[test]
    fn test_timer_operations() {
        let config = ProfilerConfig::default();
        let mut profiler = ProfilerCore::new(config);
        
        profiler.start_session("test".to_string()).unwrap();
        
        // Start and stop timer
        assert!(profiler.start_timer("test_op".to_string()).is_ok());
        std::thread::sleep(Duration::from_millis(10));
        
        let elapsed = profiler.stop_timer("test_op");
        assert!(elapsed.is_ok());
        assert!(elapsed.unwrap() >= 10.0); // Should be at least 10ms
        
        // Check statistics
        let stats = profiler.get_current_statistics().unwrap();
        assert!(!stats.is_empty());
        assert_eq!(stats[0].operation_name, "test_op");
        assert_eq!(stats[0].call_count, 1);
    }

    #[test]
    fn test_operation_metrics() {
        let mut metrics = OperationMetrics::new("test".to_string());
        
        metrics.record_timing(Duration::from_millis(100));
        metrics.record_timing(Duration::from_millis(200));
        metrics.record_timing(Duration::from_millis(150));
        
        assert_eq!(metrics.call_count, 3);
        assert_eq!(metrics.min_time, Duration::from_millis(100));
        assert_eq!(metrics.max_time, Duration::from_millis(200));
        assert_eq!(metrics.avg_time, Duration::from_millis(150));
        
        let stats = metrics.get_statistics();
        assert_eq!(stats.call_count, 3);
        assert!((stats.avg_time_ms - 150.0).abs() < 0.1);
    }
}