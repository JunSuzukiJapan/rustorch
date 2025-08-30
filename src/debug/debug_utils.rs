//! Debug Utilities and System Information
//!
//! Collection of debugging utilities including system information gathering,
//! stack trace capture, and diagnostic helpers for deep learning operations.

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{RusTorchError, RusTorchResult};

/// System information for debugging
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub architecture: String,
    pub cpu_count: usize,
    pub available_memory_mb: usize,
    pub rust_version: String,
    pub debug_build: bool,
    pub environment_vars: HashMap<String, String>,
    pub timestamp: SystemTime,
}

impl SystemInfo {
    /// Collect current system information
    pub fn collect() -> Self {
        let cpu_count = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Collect relevant environment variables
        let mut env_vars = HashMap::new();
        for (key, value) in env::vars() {
            if key.starts_with("RUST")
                || key.starts_with("CARGO")
                || key.contains("CUDA")
                || key.contains("GPU")
                || key.contains("OPENCL")
            {
                env_vars.insert(key, value);
            }
        }

        Self {
            os: env::consts::OS.to_string(),
            architecture: env::consts::ARCH.to_string(),
            cpu_count,
            available_memory_mb: Self::estimate_available_memory(),
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            debug_build: cfg!(debug_assertions),
            environment_vars: env_vars,
            timestamp: SystemTime::now(),
        }
    }

    /// Estimate available system memory (simplified)
    fn estimate_available_memory() -> usize {
        // This is a rough estimate - in a real system you'd use platform-specific APIs
        match std::env::consts::OS {
            "linux" | "macos" => 8192, // Assume 8GB default
            "windows" => 16384,        // Assume 16GB default
            _ => 4096,                 // Conservative default
        }
    }

    /// Format system info as string
    pub fn format_summary(&self) -> String {
        format!(
            "System: {} {}, {} CPUs, ~{}MB RAM, Rust {}, Debug: {}",
            self.os,
            self.architecture,
            self.cpu_count,
            self.available_memory_mb,
            self.rust_version,
            self.debug_build
        )
    }
}

/// Stack trace information (simplified)
#[derive(Debug, Clone)]
pub struct StackTrace {
    pub frames: Vec<String>,
    pub timestamp: SystemTime,
    pub thread_name: String,
}

impl StackTrace {
    /// Capture current stack trace (simplified implementation)
    pub fn capture() -> Self {
        // Note: This is a simplified implementation
        // A real implementation would use backtrace crate or similar
        let frames = vec![
            "rustorch::debug::capture_stack".to_string(),
            "rustorch::tensor::operation".to_string(),
            "rustorch::main".to_string(),
        ];

        let thread_name = thread::current().name().unwrap_or("unnamed").to_string();

        Self {
            frames,
            timestamp: SystemTime::now(),
            thread_name,
        }
    }

    /// Format stack trace as string
    pub fn format_trace(&self) -> String {
        let mut trace = format!("Stack trace (thread: {}):\n", self.thread_name);
        for (i, frame) in self.frames.iter().enumerate() {
            trace.push_str(&format!("  {}: {}\n", i, frame));
        }
        trace
    }
}

/// Performance measurement helper
#[derive(Debug, Clone)]
pub struct PerfTimer {
    name: String,
    start_time: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl PerfTimer {
    /// Start new performance timer
    pub fn start(name: String) -> Self {
        Self {
            name,
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Add checkpoint
    pub fn checkpoint(&mut self, label: &str) {
        let elapsed = self.start_time.elapsed();
        self.checkpoints.push((label.to_string(), elapsed));
    }

    /// Finish timing and get total duration
    pub fn finish(mut self) -> (Duration, Vec<(String, Duration)>) {
        let total_duration = self.start_time.elapsed();
        self.checkpoints.push(("TOTAL".to_string(), total_duration));
        (total_duration, self.checkpoints)
    }

    /// Get current elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate timing report
    pub fn generate_report(&self) -> String {
        let mut report = format!("Performance Report: {}\n", self.name);
        report.push_str("====================\n");

        for (label, duration) in &self.checkpoints {
            report.push_str(&format!(
                "{}: {:.3}ms\n",
                label,
                duration.as_secs_f64() * 1000.0
            ));
        }

        let total_elapsed = self.elapsed();
        report.push_str(&format!(
            "Current Elapsed: {:.3}ms\n",
            total_elapsed.as_secs_f64() * 1000.0
        ));

        report
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: SystemTime,
    pub component_usage: HashMap<String, usize>,
    pub total_allocated: usize,
    pub estimated_available: usize,
}

impl MemorySnapshot {
    /// Take memory snapshot (simplified)
    pub fn take(component_usage: HashMap<String, usize>) -> Self {
        let total_allocated: usize = component_usage.values().sum();

        Self {
            timestamp: SystemTime::now(),
            component_usage,
            total_allocated,
            estimated_available: SystemInfo::estimate_available_memory() * 1024 * 1024, // Convert to bytes
        }
    }

    /// Calculate memory utilization percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.estimated_available > 0 {
            (self.total_allocated as f64 / self.estimated_available as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format memory snapshot
    pub fn format_summary(&self) -> String {
        let total_mb = self.total_allocated as f64 / (1024.0 * 1024.0);
        let available_mb = self.estimated_available as f64 / (1024.0 * 1024.0);

        format!(
            "Memory: {:.1}MB / {:.1}MB ({:.1}% used)",
            total_mb,
            available_mb,
            self.utilization_percent()
        )
    }
}

/// Diagnostic context for debugging
#[derive(Debug, Clone)]
pub struct DiagnosticContext {
    pub operation_name: String,
    pub parameters: HashMap<String, String>,
    pub system_info: SystemInfo,
    pub memory_snapshot: Option<MemorySnapshot>,
    pub stack_trace: Option<StackTrace>,
    pub timestamp: SystemTime,
}

impl DiagnosticContext {
    /// Create new diagnostic context
    pub fn new(operation_name: String) -> Self {
        Self {
            operation_name,
            parameters: HashMap::new(),
            system_info: SystemInfo::collect(),
            memory_snapshot: None,
            stack_trace: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Add parameter
    pub fn add_parameter(&mut self, key: &str, value: &str) {
        self.parameters.insert(key.to_string(), value.to_string());
    }

    /// Set memory snapshot
    pub fn set_memory_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.memory_snapshot = Some(snapshot);
    }

    /// Capture stack trace
    pub fn capture_stack_trace(&mut self) {
        self.stack_trace = Some(StackTrace::capture());
    }

    /// Generate comprehensive diagnostic report
    pub fn generate_diagnostic_report(&self) -> String {
        let mut report = String::new();

        report.push_str("🔧 Diagnostic Report\n");
        report.push_str("===================\n\n");

        report.push_str(&format!("Operation: {}\n", self.operation_name));
        report.push_str(&format!("Timestamp: {:?}\n\n", self.timestamp));

        // System information
        report.push_str("🖥️ System Information:\n");
        report.push_str(&format!("  {}\n\n", self.system_info.format_summary()));

        // Parameters
        if !self.parameters.is_empty() {
            report.push_str("📋 Parameters:\n");
            for (key, value) in &self.parameters {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
            report.push('\n');
        }

        // Memory snapshot
        if let Some(snapshot) = &self.memory_snapshot {
            report.push_str("🧠 Memory Status:\n");
            report.push_str(&format!("  {}\n", snapshot.format_summary()));

            if !snapshot.component_usage.is_empty() {
                report.push_str("  Component Usage:\n");
                let mut components: Vec<_> = snapshot.component_usage.iter().collect();
                components.sort_by(|a, b| b.1.cmp(a.1));

                for (component, usage) in components.iter().take(5) {
                    let mb = **usage as f64 / (1024.0 * 1024.0);
                    report.push_str(&format!("    {}: {:.1}MB\n", component, mb));
                }
            }
            report.push('\n');
        }

        // Stack trace
        if let Some(trace) = &self.stack_trace {
            report.push_str("📚 Stack Trace:\n");
            report.push_str(&trace.format_trace());
            report.push('\n');
        }

        // Environment variables
        if !self.system_info.environment_vars.is_empty() {
            report.push_str("🌍 Relevant Environment:\n");
            for (key, value) in &self.system_info.environment_vars {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        report
    }
}

/// Debug utilities collection
pub struct DebugUtils;

impl DebugUtils {
    /// Create diagnostic context for operation
    pub fn create_diagnostic_context(operation_name: &str) -> DiagnosticContext {
        DiagnosticContext::new(operation_name.to_string())
    }

    /// Start performance measurement
    pub fn start_perf_timer(name: &str) -> PerfTimer {
        PerfTimer::start(name.to_string())
    }

    /// Collect system information
    pub fn get_system_info() -> SystemInfo {
        SystemInfo::collect()
    }

    /// Capture stack trace
    pub fn capture_stack_trace() -> StackTrace {
        StackTrace::capture()
    }

    /// Take memory snapshot
    pub fn take_memory_snapshot(component_usage: HashMap<String, usize>) -> MemorySnapshot {
        MemorySnapshot::take(component_usage)
    }

    /// Format duration for human reading
    pub fn format_duration(duration: Duration) -> String {
        let total_ms = duration.as_secs_f64() * 1000.0;

        if total_ms < 1.0 {
            format!("{:.3}μs", total_ms * 1000.0)
        } else if total_ms < 1000.0 {
            format!("{:.3}ms", total_ms)
        } else if total_ms < 60000.0 {
            format!("{:.2}s", total_ms / 1000.0)
        } else {
            let minutes = (total_ms / 60000.0) as u32;
            let seconds = (total_ms % 60000.0) / 1000.0;
            format!("{}m {:.1}s", minutes, seconds)
        }
    }

    /// Format bytes for human reading
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{}B", bytes)
        } else {
            format!("{:.2}{}", size, UNITS[unit_index])
        }
    }

    /// Check if running in debug mode
    pub fn is_debug_build() -> bool {
        cfg!(debug_assertions)
    }

    /// Get current thread information
    pub fn get_thread_info() -> (String, String) {
        let current = thread::current();
        let name = current.name().unwrap_or("unnamed").to_string();
        let id = format!("{:?}", current.id());
        (name, id)
    }

    /// Generate environment report
    pub fn generate_environment_report() -> String {
        let system_info = Self::get_system_info();

        let mut report = String::new();
        report.push_str("🌍 Environment Report\n");
        report.push_str("====================\n\n");

        report.push_str(&format!("System: {}\n", system_info.format_summary()));

        let (thread_name, thread_id) = Self::get_thread_info();
        report.push_str(&format!(
            "Current Thread: {} ({})\n",
            thread_name, thread_id
        ));

        report.push_str(&format!("Debug Build: {}\n", Self::is_debug_build()));

        if !system_info.environment_vars.is_empty() {
            report.push_str("\nRelevant Environment Variables:\n");
            for (key, value) in &system_info.environment_vars {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        report
    }

    /// Simple assertion with diagnostic context
    pub fn debug_assert_with_context<F>(condition: bool, context_fn: F, message: &str)
    where
        F: FnOnce() -> DiagnosticContext,
    {
        if cfg!(debug_assertions) && !condition {
            let context = context_fn();
            eprintln!("🚨 Debug Assertion Failed: {}", message);
            eprintln!("{}", context.generate_diagnostic_report());
            panic!("Debug assertion failed: {}", message);
        }
    }

    /// Conditional debugging output
    pub fn debug_print<T: fmt::Display>(value: &T, enabled: bool) {
        if enabled && cfg!(debug_assertions) {
            eprintln!("🐛 DEBUG: {}", value);
        }
    }

    /// Time a code block and return result with timing
    pub fn time_block<T, F>(name: &str, block: F) -> (T, Duration)
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = block();
        let duration = start.elapsed();

        if cfg!(debug_assertions) {
            eprintln!("⏱️ {}: {}", name, Self::format_duration(duration));
        }

        (result, duration)
    }
}

/// Macro for easy diagnostic context creation
#[macro_export]
macro_rules! diagnostic_context {
    ($operation:expr) => {
        $crate::debug::DebugUtils::create_diagnostic_context($operation)
    };
    ($operation:expr, $($key:expr => $value:expr),*) => {{
        let mut context = $crate::debug::DebugUtils::create_diagnostic_context($operation);
        $(
            context.add_parameter($key, &$value.to_string());
        )*
        context
    }};
}

/// Macro for performance timing
#[macro_export]
macro_rules! perf_timer {
    ($name:expr) => {
        $crate::debug::DebugUtils::start_perf_timer($name)
    };
}

/// Macro for timed code blocks
#[macro_export]
macro_rules! time_block {
    ($name:expr, $block:block) => {
        $crate::debug::DebugUtils::time_block($name, || $block)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;

    #[test]
    fn test_system_info_collection() {
        let info = SystemInfo::collect();

        assert!(!info.os.is_empty());
        assert!(!info.architecture.is_empty());
        assert!(info.cpu_count > 0);
        assert!(info.available_memory_mb > 0);
        assert!(!info.rust_version.is_empty());

        let summary = info.format_summary();
        assert!(summary.contains(&info.os));
        assert!(summary.contains("CPUs"));
    }

    #[test]
    fn test_stack_trace_capture() {
        let trace = StackTrace::capture();

        assert!(!trace.frames.is_empty());
        assert!(!trace.thread_name.is_empty());

        let formatted = trace.format_trace();
        assert!(formatted.contains("Stack trace"));
        assert!(formatted.contains(&trace.thread_name));
    }

    #[test]
    fn test_perf_timer() {
        let mut timer = PerfTimer::start("test_operation".to_string());

        thread::sleep(StdDuration::from_millis(10));
        timer.checkpoint("checkpoint_1");

        thread::sleep(StdDuration::from_millis(10));
        timer.checkpoint("checkpoint_2");

        let (total_duration, checkpoints) = timer.finish();

        assert!(total_duration.as_millis() >= 20);
        assert_eq!(checkpoints.len(), 3); // 2 checkpoints + TOTAL

        let last_checkpoint = &checkpoints[checkpoints.len() - 1];
        assert_eq!(last_checkpoint.0, "TOTAL");
        assert_eq!(last_checkpoint.1, total_duration);
    }

    #[test]
    fn test_memory_snapshot() {
        let mut usage = HashMap::new();
        usage.insert("tensor".to_string(), 1024 * 1024); // 1MB
        usage.insert("network".to_string(), 2048 * 1024); // 2MB

        let snapshot = MemorySnapshot::take(usage);

        assert_eq!(snapshot.total_allocated, 3 * 1024 * 1024); // 3MB
        assert!(snapshot.utilization_percent() >= 0.0);

        let summary = snapshot.format_summary();
        assert!(summary.contains("Memory:"));
        assert!(summary.contains("MB"));
    }

    #[test]
    fn test_diagnostic_context() {
        let mut context = DiagnosticContext::new("test_operation".to_string());

        context.add_parameter("param1", "value1");
        context.add_parameter("param2", "value2");

        let mut usage = HashMap::new();
        usage.insert("component1".to_string(), 1024 * 1024);
        let snapshot = MemorySnapshot::take(usage);
        context.set_memory_snapshot(snapshot);

        context.capture_stack_trace();

        let report = context.generate_diagnostic_report();

        assert!(report.contains("Diagnostic Report"));
        assert!(report.contains("test_operation"));
        assert!(report.contains("param1: value1"));
        assert!(report.contains("param2: value2"));
        assert!(report.contains("Memory Status"));
        assert!(report.contains("Stack Trace"));
    }

    #[test]
    fn test_debug_utils_formatting() {
        // Test duration formatting
        let duration = Duration::from_millis(1500);
        let formatted = DebugUtils::format_duration(duration);
        assert!(formatted.contains("1.50s"));

        let micro_duration = Duration::from_nanos(500);
        let micro_formatted = DebugUtils::format_duration(micro_duration);
        assert!(micro_formatted.contains("μs"));

        // Test bytes formatting
        let bytes = DebugUtils::format_bytes(1536); // 1.5KB
        assert!(bytes.contains("1.50KB"));

        let mb_bytes = DebugUtils::format_bytes(2 * 1024 * 1024); // 2MB
        assert!(mb_bytes.contains("2.00MB"));
    }

    #[test]
    fn test_thread_info() {
        let (name, id) = DebugUtils::get_thread_info();

        // Thread name might be empty, but ID should exist
        assert!(!id.is_empty());
    }

    #[test]
    fn test_environment_report() {
        let report = DebugUtils::generate_environment_report();

        assert!(report.contains("Environment Report"));
        assert!(report.contains("System:"));
        assert!(report.contains("Current Thread:"));
        assert!(report.contains("Debug Build:"));
    }

    #[test]
    fn test_time_block_macro() {
        let (result, duration) = time_block!("test_block", {
            thread::sleep(StdDuration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_debug_print() {
        // This test just ensures the function doesn't panic
        DebugUtils::debug_print(&"test message", true);
        DebugUtils::debug_print(&"test message", false);
    }
}
