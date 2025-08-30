//! RusTorch Debug & Logging Framework
//! 
//! Comprehensive logging and debugging system with structured logging,
//! performance profiling, memory tracking, and advanced debug utilities.
//!
//! # Architecture
//! - Core Logger: Structured logging with multiple levels and outputs
//! - Debug Profiler: Performance monitoring and bottleneck detection  
//! - Memory Tracker: Allocation monitoring and leak detection
//! - Log Analyzer: Pattern recognition and alert generation
//!
//! # Usage Examples
//! ```rust
//! use rustorch::debug::{Logger, DebugFramework, LogLevel};
//!
//! // Initialize framework
//! let mut debug_framework = DebugFramework::new();
//! 
//! // Structured logging
//! debug_framework.log(LogLevel::Info, "operation_complete", 
//!     &[("duration_ms", "150"), ("tensors_processed", "1000")]);
//!
//! // Performance profiling
//! let _guard = debug_framework.profile("matrix_multiply");
//! // ... operation code ...
//! // Profile automatically captured on drop
//!
//! // Memory tracking
//! debug_framework.track_allocation("tensor_data", 1024 * 1024);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::sync::{Arc, Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::PathBuf;
use std::thread;
use std::fmt;
use serde_json::{json, Value};

use crate::error::{RusTorchError, RusTorchResult};

pub mod logger;
pub mod profiler;
pub mod memory_tracker;
pub mod log_analyzer;
pub mod debug_utils;

pub use logger::{Logger, LogLevel, LogEntry, LogOutput};
pub use profiler::{DebugProfiler, ProfileEntry, PerformanceMetrics};
pub use memory_tracker::{MemoryTracker, AllocationInfo, MemoryReport};
pub use log_analyzer::{LogAnalyzer, LogPattern, AlertRule};
pub use debug_utils::{DebugUtils, SystemInfo, StackTrace};

/// Unified Debug Framework
/// 
/// Central coordinator for all debugging and logging operations.
/// Integrates logging, profiling, memory tracking, and analysis.
#[derive(Debug)]
pub struct DebugFramework {
    /// Core logging system
    logger: Arc<Mutex<Logger>>,
    /// Performance profiler
    profiler: Arc<Mutex<DebugProfiler>>,
    /// Memory allocation tracker
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Log pattern analyzer
    log_analyzer: Arc<Mutex<LogAnalyzer>>,
    /// Framework configuration
    config: FrameworkConfig,
    /// Session metadata
    session_id: String,
    /// Framework start time
    start_time: Instant,
}

/// Framework Configuration
#[derive(Debug, Clone)]
pub struct FrameworkConfig {
    /// Logging configuration
    pub log_level: LogLevel,
    pub log_to_console: bool,
    pub log_to_file: bool,
    pub log_file_path: Option<PathBuf>,
    
    /// Profiling configuration
    pub enable_profiling: bool,
    pub profile_threshold_ms: u64,
    pub max_profile_entries: usize,
    
    /// Memory tracking configuration
    pub enable_memory_tracking: bool,
    pub memory_threshold_mb: usize,
    pub track_allocations: bool,
    
    /// Analysis configuration
    pub enable_log_analysis: bool,
    pub analysis_window_size: usize,
    pub alert_threshold: usize,
}

/// Debug Session Information
#[derive(Debug, Clone)]
pub struct DebugSession {
    pub session_id: String,
    pub start_time: SystemTime,
    pub duration: Duration,
    pub total_logs: usize,
    pub total_profiles: usize,
    pub memory_peak_mb: f64,
    pub alerts_triggered: usize,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            log_to_console: true,
            log_to_file: true,
            log_file_path: Some(PathBuf::from("rustorch_debug.log")),
            
            enable_profiling: true,
            profile_threshold_ms: 1,
            max_profile_entries: 10000,
            
            enable_memory_tracking: true,
            memory_threshold_mb: 1024,
            track_allocations: true,
            
            enable_log_analysis: true,
            analysis_window_size: 1000,
            alert_threshold: 10,
        }
    }
}

impl DebugFramework {
    /// Create new debug framework with default configuration
    pub fn new() -> Self {
        Self::with_config(FrameworkConfig::default())
    }
    
    /// Create debug framework with custom configuration
    pub fn with_config(config: FrameworkConfig) -> Self {
        let session_id = format!("debug_session_{}", 
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs());
        
        let logger = Arc::new(Mutex::new(
            Logger::new(config.log_level, config.log_to_console, config.log_to_file)
        ));
        
        let profiler = Arc::new(Mutex::new(
            DebugProfiler::new(config.enable_profiling, config.max_profile_entries)
        ));
        
        let memory_tracker = Arc::new(Mutex::new(
            MemoryTracker::new(config.enable_memory_tracking, config.memory_threshold_mb)
        ));
        
        let log_analyzer = Arc::new(Mutex::new(
            LogAnalyzer::new(config.enable_log_analysis, config.analysis_window_size)
        ));
        
        Self {
            logger,
            profiler,
            memory_tracker,
            log_analyzer,
            config,
            session_id,
            start_time: Instant::now(),
        }
    }
    
    /// Log structured message with metadata
    pub fn log(&self, level: LogLevel, message: &str, metadata: &[(&str, &str)]) -> RusTorchResult<()> {
        let mut logger = self.logger.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire logger lock".to_string() 
            })?;
        
        let mut meta_map = HashMap::new();
        for (key, value) in metadata {
            meta_map.insert(key.to_string(), value.to_string());
        }
        meta_map.insert("session_id".to_string(), self.session_id.clone());
        
        logger.log(level, message, meta_map.clone())?;
        
        // Feed to log analyzer if enabled
        if self.config.enable_log_analysis {
            if let Ok(mut analyzer) = self.log_analyzer.lock() {
                let _ = analyzer.analyze_log_entry(level, message, &meta_map);
            }
        }
        
        Ok(())
    }
    
    /// Create profiling guard for automatic timing
    pub fn profile(&self, operation_name: &str) -> RusTorchResult<ProfileGuard> {
        if !self.config.enable_profiling {
            return Ok(ProfileGuard::disabled());
        }
        
        Ok(ProfileGuard::new(
            operation_name.to_string(),
            Arc::clone(&self.profiler),
            self.config.profile_threshold_ms,
        ))
    }
    
    /// Track memory allocation
    pub fn track_allocation(&self, component: &str, size_bytes: usize) -> RusTorchResult<()> {
        if !self.config.enable_memory_tracking {
            return Ok(());
        }
        
        let mut tracker = self.memory_tracker.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire memory tracker lock".to_string() 
            })?;
        
        tracker.track_allocation(component, size_bytes)?;
        
        // Check for memory threshold alerts
        let current_usage = tracker.get_current_usage_mb();
        if current_usage > self.config.memory_threshold_mb as f64 {
            self.log(LogLevel::Warning, "Memory threshold exceeded", &[
                ("current_mb", &format!("{:.2}", current_usage)),
                ("threshold_mb", &self.config.memory_threshold_mb.to_string()),
                ("component", component),
            ])?;
        }
        
        Ok(())
    }
    
    /// Track memory deallocation
    pub fn track_deallocation(&self, component: &str, size_bytes: usize) -> RusTorchResult<()> {
        if !self.config.enable_memory_tracking {
            return Ok(());
        }
        
        let mut tracker = self.memory_tracker.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire memory tracker lock".to_string() 
            })?;
        
        tracker.track_deallocation(component, size_bytes)
    }
    
    /// Get comprehensive debug report
    pub fn generate_debug_report(&self) -> RusTorchResult<DebugReport> {
        let session = self.get_session_info()?;
        
        let log_summary = self.logger.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire logger lock".to_string() 
            })?
            .get_summary();
        
        let performance_metrics = self.profiler.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire profiler lock".to_string() 
            })?
            .get_performance_metrics();
        
        let memory_report = self.memory_tracker.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire memory tracker lock".to_string() 
            })?
            .generate_memory_report()?;
        
        let analysis_summary = self.log_analyzer.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire log analyzer lock".to_string() 
            })?
            .get_analysis_summary();
        
        Ok(DebugReport {
            session,
            log_summary,
            performance_metrics,
            memory_report,
            analysis_summary,
            config: self.config.clone(),
        })
    }
    
    /// Get current session information
    pub fn get_session_info(&self) -> RusTorchResult<DebugSession> {
        let duration = self.start_time.elapsed();
        
        let total_logs = self.logger.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire logger lock".to_string() 
            })?
            .get_total_logs();
        
        let total_profiles = self.profiler.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire profiler lock".to_string() 
            })?
            .get_total_profiles();
        
        let memory_peak_mb = self.memory_tracker.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire memory tracker lock".to_string() 
            })?
            .get_peak_usage_mb();
        
        let alerts_triggered = self.log_analyzer.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire log analyzer lock".to_string() 
            })?
            .get_total_alerts();
        
        Ok(DebugSession {
            session_id: self.session_id.clone(),
            start_time: SystemTime::now() - duration,
            duration,
            total_logs,
            total_profiles,
            memory_peak_mb,
            alerts_triggered,
        })
    }
    
    /// Flush all pending data
    pub fn flush(&self) -> RusTorchResult<()> {
        // Flush logger
        self.logger.lock()
            .map_err(|_| RusTorchError::Debug { 
                message: "Failed to acquire logger lock".to_string() 
            })?
            .flush()?;
        
        Ok(())
    }
}

/// RAII Profile Guard for automatic timing
pub struct ProfileGuard {
    operation_name: Option<String>,
    profiler: Option<Arc<Mutex<DebugProfiler>>>,
    start_time: Instant,
    threshold_ms: u64,
}

impl ProfileGuard {
    fn new(operation_name: String, profiler: Arc<Mutex<DebugProfiler>>, threshold_ms: u64) -> Self {
        Self {
            operation_name: Some(operation_name),
            profiler: Some(profiler),
            start_time: Instant::now(),
            threshold_ms,
        }
    }
    
    fn disabled() -> Self {
        Self {
            operation_name: None,
            profiler: None,
            start_time: Instant::now(),
            threshold_ms: 0,
        }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        if let (Some(operation_name), Some(profiler)) = (&self.operation_name, &self.profiler) {
            let duration = self.start_time.elapsed();
            
            if duration.as_millis() as u64 >= self.threshold_ms {
                if let Ok(mut prof) = profiler.lock() {
                    let _ = prof.record_operation(operation_name, duration);
                }
            }
        }
    }
}

/// Comprehensive Debug Report
#[derive(Debug)]
pub struct DebugReport {
    pub session: DebugSession,
    pub log_summary: LogSummary,
    pub performance_metrics: PerformanceMetrics,
    pub memory_report: MemoryReport,
    pub analysis_summary: AnalysisSummary,
    pub config: FrameworkConfig,
}

/// Log Summary Statistics
#[derive(Debug, Clone)]
pub struct LogSummary {
    pub total_logs: usize,
    pub logs_by_level: HashMap<String, usize>,
    pub recent_errors: Vec<String>,
    pub log_rate_per_second: f64,
}

/// Analysis Summary
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    pub patterns_detected: usize,
    pub alerts_triggered: usize,
    pub most_common_errors: Vec<(String, usize)>,
    pub performance_bottlenecks: Vec<String>,
}

impl fmt::Display for DebugReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔧 RusTorch Debug Report")?;
        writeln!(f, "========================")?;
        writeln!(f, "Session ID: {}", self.session.session_id)?;
        writeln!(f, "Duration: {:.2}s", self.session.duration.as_secs_f64())?;
        writeln!(f, "Total Logs: {}", self.session.total_logs)?;
        writeln!(f, "Total Profiles: {}", self.session.total_profiles)?;
        writeln!(f, "Memory Peak: {:.2} MB", self.session.memory_peak_mb)?;
        writeln!(f, "Alerts: {}", self.session.alerts_triggered)?;
        writeln!(f)?;
        writeln!(f, "📊 Performance Summary:")?;
        writeln!(f, "  Average Operation Time: {:.2}ms", self.performance_metrics.average_duration_ms)?;
        writeln!(f, "  Slowest Operation: {}", self.performance_metrics.slowest_operation)?;
        writeln!(f, "  Operations > 100ms: {}", self.performance_metrics.slow_operations_count)?;
        writeln!(f)?;
        writeln!(f, "🧠 Memory Summary:")?;
        writeln!(f, "  Current Usage: {:.2} MB", self.memory_report.current_usage_mb)?;
        writeln!(f, "  Peak Usage: {:.2} MB", self.memory_report.peak_usage_mb)?;
        writeln!(f, "  Total Allocations: {}", self.memory_report.total_allocations)?;
        Ok(())
    }
}

// Convenience functions for quick debugging
impl DebugFramework {
    /// Quick error log
    pub fn error(&self, message: &str) -> RusTorchResult<()> {
        self.log(LogLevel::Error, message, &[])
    }
    
    /// Quick warning log
    pub fn warn(&self, message: &str) -> RusTorchResult<()> {
        self.log(LogLevel::Warning, message, &[])
    }
    
    /// Quick info log
    pub fn info(&self, message: &str) -> RusTorchResult<()> {
        self.log(LogLevel::Info, message, &[])
    }
    
    /// Quick debug log
    pub fn debug(&self, message: &str) -> RusTorchResult<()> {
        self.log(LogLevel::Debug, message, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;
    
    #[test]
    fn test_debug_framework_creation() {
        let framework = DebugFramework::new();
        assert!(framework.session_id.contains("debug_session_"));
    }
    
    #[test]
    fn test_logging_functionality() {
        let framework = DebugFramework::new();
        
        // Test different log levels
        assert!(framework.info("Test info message").is_ok());
        assert!(framework.warn("Test warning message").is_ok());
        assert!(framework.error("Test error message").is_ok());
        assert!(framework.debug("Test debug message").is_ok());
        
        // Test structured logging
        assert!(framework.log(LogLevel::Info, "Structured log", &[
            ("key1", "value1"),
            ("key2", "value2"),
        ]).is_ok());
    }
    
    #[test]
    fn test_profiling_functionality() {
        let framework = DebugFramework::new();
        
        // Test profiling guard
        {
            let _guard = framework.profile("test_operation").unwrap();
            thread::sleep(StdDuration::from_millis(10));
        } // Guard drops here, recording the profile
        
        let report = framework.generate_debug_report().unwrap();
        assert!(report.performance_metrics.total_operations > 0);
    }
    
    #[test]
    fn test_memory_tracking() {
        let framework = DebugFramework::new();
        
        // Track allocations
        assert!(framework.track_allocation("test_component", 1024).is_ok());
        assert!(framework.track_allocation("test_component", 2048).is_ok());
        
        // Track deallocation
        assert!(framework.track_deallocation("test_component", 1024).is_ok());
        
        let report = framework.generate_debug_report().unwrap();
        assert!(report.memory_report.total_allocations > 0);
    }
    
    #[test]
    fn test_debug_report_generation() {
        let framework = DebugFramework::new();
        
        // Generate some activity
        let _ = framework.info("Test message");
        let _ = framework.track_allocation("test", 100);
        
        let report = framework.generate_debug_report();
        assert!(report.is_ok());
        
        let report = report.unwrap();
        assert!(!report.session.session_id.is_empty());
        assert!(report.session.total_logs > 0);
    }
}