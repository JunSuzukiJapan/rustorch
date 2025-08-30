//! Structured Logging System
//!
//! Advanced logging system with multiple levels, structured metadata,
//! and configurable output targets (console, file, network).

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter, stdout};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::{json, Value};

use crate::error::{RusTorchError, RusTorchResult};
use super::LogSummary;

/// Log severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Critical = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level_str = match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRIT",
        };
        write!(f, "{}", level_str)
    }
}

impl LogLevel {
    /// Convert from string representation
    pub fn from_str(s: &str) -> Option<LogLevel> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARNING" | "WARN" => Some(LogLevel::Warning),
            "ERROR" => Some(LogLevel::Error),
            "CRITICAL" | "CRIT" => Some(LogLevel::Critical),
            _ => None,
        }
    }
    
    /// Get ANSI color code for console output
    pub fn color_code(&self) -> &'static str {
        match self {
            LogLevel::Trace => "\x1b[37m",      // White
            LogLevel::Debug => "\x1b[36m",      // Cyan
            LogLevel::Info => "\x1b[32m",       // Green
            LogLevel::Warning => "\x1b[33m",    // Yellow
            LogLevel::Error => "\x1b[31m",      // Red
            LogLevel::Critical => "\x1b[35m",   // Magenta
        }
    }
    
    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            LogLevel::Trace => "🔍",
            LogLevel::Debug => "🐛",
            LogLevel::Info => "ℹ️",
            LogLevel::Warning => "⚠️",
            LogLevel::Error => "❌",
            LogLevel::Critical => "🚨",
        }
    }
}

/// Log entry with structured metadata
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub metadata: HashMap<String, String>,
    pub thread_id: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

impl LogEntry {
    /// Create new log entry
    pub fn new(level: LogLevel, message: String, metadata: HashMap<String, String>) -> Self {
        Self {
            timestamp: SystemTime::now(),
            level,
            message,
            metadata,
            thread_id: format!("{:?}", std::thread::current().id()),
            file: None,
            line: None,
        }
    }
    
    /// Format as human-readable string
    pub fn format_human(&self) -> String {
        let timestamp = self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let metadata_str = if self.metadata.is_empty() {
            String::new()
        } else {
            format!(" [{}]", 
                self.metadata.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        
        format!("{} {} [{}] {}{}", 
                timestamp,
                self.level.emoji(),
                self.level,
                self.message,
                metadata_str)
    }
    
    /// Format as colored console output
    pub fn format_console(&self) -> String {
        let reset = "\x1b[0m";
        let color = self.level.color_code();
        let timestamp = self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let metadata_str = if self.metadata.is_empty() {
            String::new()
        } else {
            format!(" [{}]", 
                self.metadata.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        
        format!("{}{} {} [{}] {}{}{}", 
                color,
                timestamp,
                self.level.emoji(),
                self.level,
                self.message,
                metadata_str,
                reset)
    }
    
    /// Format as JSON
    pub fn format_json(&self) -> RusTorchResult<String> {
        let timestamp_ms = self.timestamp
            .duration_since(UNIX_EPOCH)
            .map_err(|_| RusTorchError::Debug { 
                message: "Invalid timestamp".to_string() 
            })?
            .as_millis() as u64;
        
        let mut json_obj = json!({
            "timestamp": timestamp_ms,
            "level": self.level.to_string(),
            "message": self.message,
            "thread_id": self.thread_id,
        });
        
        // Add metadata
        for (key, value) in &self.metadata {
            json_obj[key] = Value::String(value.clone());
        }
        
        // Add file/line if available
        if let Some(file) = &self.file {
            json_obj["file"] = Value::String(file.clone());
        }
        if let Some(line) = self.line {
            json_obj["line"] = Value::Number(serde_json::Number::from(line));
        }
        
        serde_json::to_string(&json_obj)
            .map_err(|e| RusTorchError::Debug { 
                message: format!("JSON serialization failed: {}", e) 
            })
    }
}

/// Log output configuration
#[derive(Debug, Clone)]
pub enum LogOutput {
    Console,
    File(PathBuf),
    Both(PathBuf),
    Network(String), // URL endpoint
}

/// Core logging system
pub struct Logger {
    min_level: LogLevel,
    output: LogOutput,
    file_writer: Option<Arc<Mutex<BufWriter<File>>>>,
    entries: Vec<LogEntry>,
    total_logs: usize,
    logs_by_level: HashMap<String, usize>,
}

impl fmt::Debug for Logger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger")
            .field("min_level", &self.min_level)
            .field("output", &self.output)
            .field("total_logs", &self.total_logs)
            .field("logs_by_level", &self.logs_by_level)
            .finish()
    }
}

impl Logger {
    /// Create new logger
    pub fn new(min_level: LogLevel, to_console: bool, to_file: bool) -> Self {
        let output = match (to_console, to_file) {
            (true, true) => LogOutput::Both(PathBuf::from("rustorch_debug.log")),
            (true, false) => LogOutput::Console,
            (false, true) => LogOutput::File(PathBuf::from("rustorch_debug.log")),
            (false, false) => LogOutput::Console, // Default fallback
        };
        
        Self::with_output(min_level, output)
    }
    
    /// Create logger with specific output configuration
    pub fn with_output(min_level: LogLevel, output: LogOutput) -> Self {
        let file_writer = match &output {
            LogOutput::File(path) | LogOutput::Both(path) => {
                Self::create_file_writer(path).ok()
            },
            _ => None,
        };
        
        Self {
            min_level,
            output,
            file_writer,
            entries: Vec::new(),
            total_logs: 0,
            logs_by_level: HashMap::new(),
        }
    }
    
    /// Create buffered file writer
    fn create_file_writer(path: &PathBuf) -> RusTorchResult<Arc<Mutex<BufWriter<File>>>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)
            .map_err(|e| RusTorchError::Debug { 
                message: format!("Failed to open log file {}: {}", path.display(), e) 
            })?;
        
        Ok(Arc::new(Mutex::new(BufWriter::new(file))))
    }
    
    /// Log structured entry
    pub fn log(&mut self, level: LogLevel, message: &str, metadata: HashMap<String, String>) -> RusTorchResult<()> {
        if level < self.min_level {
            return Ok(());
        }
        
        let entry = LogEntry::new(level, message.to_string(), metadata);
        
        // Write to outputs
        self.write_entry(&entry)?;
        
        // Update statistics
        self.entries.push(entry.clone());
        self.total_logs += 1;
        *self.logs_by_level.entry(level.to_string()).or_insert(0) += 1;
        
        // Keep only recent entries to prevent memory growth
        if self.entries.len() > 10000 {
            self.entries.drain(0..1000);
        }
        
        Ok(())
    }
    
    /// Write entry to configured outputs
    fn write_entry(&self, entry: &LogEntry) -> RusTorchResult<()> {
        match &self.output {
            LogOutput::Console => {
                println!("{}", entry.format_console());
            },
            LogOutput::File(_) => {
                self.write_to_file(entry)?;
            },
            LogOutput::Both(_) => {
                println!("{}", entry.format_console());
                self.write_to_file(entry)?;
            },
            LogOutput::Network(_url) => {
                // Network logging not implemented in this version
                println!("{}", entry.format_console());
            },
        }
        
        Ok(())
    }
    
    /// Write entry to file
    fn write_to_file(&self, entry: &LogEntry) -> RusTorchResult<()> {
        if let Some(writer) = &self.file_writer {
            let json_entry = entry.format_json()?;
            let mut writer = writer.lock()
                .map_err(|_| RusTorchError::Debug { 
                    message: "Failed to acquire file writer lock".to_string() 
                })?;
            
            writeln!(writer, "{}", json_entry)
                .map_err(|e| RusTorchError::Debug { 
                    message: format!("Failed to write to log file: {}", e) 
                })?;
        }
        
        Ok(())
    }
    
    /// Get log summary statistics
    pub fn get_summary(&self) -> LogSummary {
        let recent_errors: Vec<String> = self.entries
            .iter()
            .rev()
            .take(10)
            .filter(|entry| entry.level >= LogLevel::Error)
            .map(|entry| entry.message.clone())
            .collect();
        
        let log_rate_per_second = if !self.entries.is_empty() {
            let duration = self.entries.last().unwrap().timestamp
                .duration_since(self.entries.first().unwrap().timestamp)
                .unwrap_or_default()
                .as_secs_f64();
            
            if duration > 0.0 {
                self.total_logs as f64 / duration
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        LogSummary {
            total_logs: self.total_logs,
            logs_by_level: self.logs_by_level.clone(),
            recent_errors,
            log_rate_per_second,
        }
    }
    
    /// Get total log count
    pub fn get_total_logs(&self) -> usize {
        self.total_logs
    }
    
    /// Flush all pending writes
    pub fn flush(&mut self) -> RusTorchResult<()> {
        if let Some(writer) = &self.file_writer {
            let mut writer = writer.lock()
                .map_err(|_| RusTorchError::Debug { 
                    message: "Failed to acquire file writer lock".to_string() 
                })?;
            
            writer.flush()
                .map_err(|e| RusTorchError::Debug { 
                    message: format!("Failed to flush log file: {}", e) 
                })?;
        }
        
        Ok(())
    }
    
    /// Filter entries by level
    pub fn filter_by_level(&self, min_level: LogLevel) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.level >= min_level)
            .collect()
    }
    
    /// Filter entries by metadata key-value pair
    pub fn filter_by_metadata(&self, key: &str, value: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.metadata.get(key)
                    .map(|v| v == value)
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Search entries by message content
    pub fn search_messages(&self, query: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.message.contains(query))
            .collect()
    }
}

/// Convenience macro for structured logging
#[macro_export]
macro_rules! log_structured {
    ($logger:expr, $level:expr, $message:expr, $($key:expr => $value:expr),*) => {{
        let mut metadata = std::collections::HashMap::new();
        $(
            metadata.insert($key.to_string(), $value.to_string());
        )*
        $logger.log($level, $message, metadata)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Critical);
    }
    
    #[test]
    fn test_log_level_from_string() {
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("ERROR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warning));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }
    
    #[test]
    fn test_log_entry_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string(), metadata.clone());
        
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.metadata, metadata);
        assert!(!entry.thread_id.is_empty());
    }
    
    #[test]
    fn test_log_entry_formatting() {
        let metadata = HashMap::new();
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string(), metadata);
        
        let human = entry.format_human();
        assert!(human.contains("Test message"));
        assert!(human.contains("INFO"));
        
        let console = entry.format_console();
        assert!(console.contains("Test message"));
        assert!(console.contains("ℹ️"));
        
        let json = entry.format_json().unwrap();
        assert!(json.contains("Test message"));
        assert!(json.contains("INFO"));
    }
    
    #[test]
    fn test_logger_creation() {
        let logger = Logger::new(LogLevel::Info, true, false);
        assert_eq!(logger.min_level, LogLevel::Info);
        assert_eq!(logger.total_logs, 0);
    }
    
    #[test]
    fn test_logger_filtering() {
        let mut logger = Logger::new(LogLevel::Warning, true, false);
        
        let metadata = HashMap::new();
        
        // This should be filtered out
        assert!(logger.log(LogLevel::Info, "Info message", metadata.clone()).is_ok());
        assert_eq!(logger.total_logs, 0);
        
        // This should be logged
        assert!(logger.log(LogLevel::Error, "Error message", metadata).is_ok());
        assert_eq!(logger.total_logs, 1);
    }
    
    #[test]
    fn test_logger_statistics() {
        let mut logger = Logger::new(LogLevel::Debug, true, false);
        let metadata = HashMap::new();
        
        // Log various levels
        logger.log(LogLevel::Info, "Info 1", metadata.clone()).unwrap();
        logger.log(LogLevel::Info, "Info 2", metadata.clone()).unwrap();
        logger.log(LogLevel::Error, "Error 1", metadata.clone()).unwrap();
        
        let summary = logger.get_summary();
        assert_eq!(summary.total_logs, 3);
        assert_eq!(*summary.logs_by_level.get("INFO").unwrap_or(&0), 2);
        assert_eq!(*summary.logs_by_level.get("ERROR").unwrap_or(&0), 1);
    }
    
    #[test]
    fn test_logger_search_and_filter() {
        let mut logger = Logger::new(LogLevel::Debug, true, false);
        
        let mut metadata1 = HashMap::new();
        metadata1.insert("component".to_string(), "tensor".to_string());
        
        let mut metadata2 = HashMap::new();
        metadata2.insert("component".to_string(), "network".to_string());
        
        logger.log(LogLevel::Info, "Tensor operation completed", metadata1).unwrap();
        logger.log(LogLevel::Error, "Network error occurred", metadata2).unwrap();
        logger.log(LogLevel::Warning, "Tensor validation failed", HashMap::new()).unwrap();
        
        // Test search
        let tensor_logs = logger.search_messages("Tensor");
        assert_eq!(tensor_logs.len(), 2);
        
        // Test metadata filter
        let tensor_component_logs = logger.filter_by_metadata("component", "tensor");
        assert_eq!(tensor_component_logs.len(), 1);
        
        // Test level filter
        let error_logs = logger.filter_by_level(LogLevel::Error);
        assert_eq!(error_logs.len(), 1);
    }
}