//! Real-time Performance Monitoring
//! リアルタイムパフォーマンス監視

use crate::error::{RusTorchError, RusTorchResult};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Monitoring configuration
/// 監視設定
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Sampling interval
    /// サンプリング間隔
    pub sampling_interval: Duration,
    /// Alert thresholds
    /// アラート閾値
    pub alert_thresholds: AlertThresholds,
    /// Enable system monitoring
    /// システム監視を有効化
    pub enable_system_monitoring: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_millis(100),
            alert_thresholds: AlertThresholds::default(),
            enable_system_monitoring: true,
        }
    }
}

/// Alert thresholds
/// アラート閾値
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// CPU usage threshold (%)
    /// CPU使用量閾値（%）
    pub cpu_threshold: f64,
    /// Memory usage threshold (%)
    /// メモリ使用量閾値（%）
    pub memory_threshold: f64,
    /// GPU usage threshold (%)
    /// GPU使用量閾値（%）
    pub gpu_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 90.0,
            memory_threshold: 85.0,
            gpu_threshold: 95.0,
        }
    }
}

/// System alert
/// システムアラート
#[derive(Debug, Clone)]
pub struct SystemAlert {
    /// Alert type
    /// アラートタイプ
    pub alert_type: AlertType,
    /// Alert message
    /// アラートメッセージ
    pub message: String,
    /// Current value
    /// 現在値
    pub current_value: f64,
    /// Threshold value
    /// 閾値
    pub threshold: f64,
    /// Timestamp
    /// タイムスタンプ
    pub timestamp: Instant,
}

/// Alert type
/// アラートタイプ
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertType {
    /// High CPU usage
    /// 高CPU使用量
    HighCpuUsage,
    /// High memory usage
    /// 高メモリ使用量
    HighMemoryUsage,
    /// High GPU usage
    /// 高GPU使用量
    HighGpuUsage,
    /// Performance degradation
    /// パフォーマンス劣化
    PerformanceDegradation,
    /// System overload
    /// システム過負荷
    SystemOverload,
}

/// Real-time monitor
/// リアルタイム監視
pub struct RealTimeMonitor {
    /// Configuration
    /// 設定
    config: MonitorConfig,
    /// Running state
    /// 実行状態
    is_running: Arc<Mutex<bool>>,
    /// Collected alerts
    /// 収集されたアラート
    alerts: Arc<Mutex<Vec<SystemAlert>>>,
}

impl RealTimeMonitor {
    /// Create new real-time monitor
    /// 新しいリアルタイム監視を作成
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            is_running: Arc::new(Mutex::new(false)),
            alerts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start monitoring
    /// 監視開始
    pub fn start(&self) -> RusTorchResult<()> {
        let mut running = self
            .is_running
            .lock()
            .map_err(|_| RusTorchError::Profiling {
                message: "Failed to acquire running lock".to_string(),
            })?;

        if *running {
            return Err(RusTorchError::Profiling {
                message: "Monitor already running".to_string(),
            });
        }

        *running = true;
        println!("📡 Real-time monitoring started");
        Ok(())
    }

    /// Stop monitoring
    /// 監視停止
    pub fn stop(&self) -> RusTorchResult<()> {
        let mut running = self
            .is_running
            .lock()
            .map_err(|_| RusTorchError::Profiling {
                message: "Failed to acquire running lock".to_string(),
            })?;

        *running = false;
        println!("📡 Real-time monitoring stopped");
        Ok(())
    }

    /// Get current alerts
    /// 現在のアラートを取得
    pub fn get_alerts(&self) -> RusTorchResult<Vec<SystemAlert>> {
        let alerts = self.alerts.lock().map_err(|_| RusTorchError::Profiling {
            message: "Failed to acquire alerts lock".to_string(),
        })?;

        Ok(alerts.clone())
    }

    /// Clear alerts
    /// アラートをクリア
    pub fn clear_alerts(&self) -> RusTorchResult<()> {
        let mut alerts = self.alerts.lock().map_err(|_| RusTorchError::Profiling {
            message: "Failed to acquire alerts lock".to_string(),
        })?;

        alerts.clear();
        Ok(())
    }
}

impl Default for RealTimeMonitor {
    fn default() -> Self {
        Self::new(MonitorConfig::default())
    }
}
