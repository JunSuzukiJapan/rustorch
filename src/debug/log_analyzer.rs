//! Log Analysis and Pattern Detection System
//!
//! Advanced log analysis system for pattern recognition, anomaly detection,
//! and automated alert generation based on log content analysis.

use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, SystemTime};

use super::{AnalysisSummary, LogLevel};
use crate::error::{RusTorchError, RusTorchResult};

/// Log pattern for detection
#[derive(Debug, Clone)]
pub struct LogPattern {
    pub name: String,
    pub pattern: Regex,
    pub severity: LogLevel,
    pub description: String,
    pub action_required: bool,
}

impl LogPattern {
    /// Create new log pattern
    pub fn new(
        name: String,
        pattern_str: &str,
        severity: LogLevel,
        description: String,
        action_required: bool,
    ) -> RusTorchResult<Self> {
        let pattern = Regex::new(pattern_str).map_err(|e| RusTorchError::Debug {
            message: format!("Invalid regex pattern '{}': {}", pattern_str, e),
        })?;

        Ok(Self {
            name,
            pattern,
            severity,
            description,
            action_required,
        })
    }

    /// Check if log entry matches this pattern
    pub fn matches(&self, log_message: &str) -> bool {
        self.pattern.is_match(log_message)
    }

    /// Extract capture groups from log message
    pub fn extract_captures(&self, log_message: &str) -> Option<Vec<String>> {
        self.pattern.captures(log_message).map(|caps| {
            caps.iter()
                .skip(1) // Skip the full match
                .filter_map(|m| m.map(|m| m.as_str().to_string()))
                .collect()
        })
    }
}

/// Alert rule for automated notifications
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub pattern_name: String,
    pub threshold_count: usize,
    pub time_window_seconds: u64,
    pub enabled: bool,
    pub last_triggered: Option<SystemTime>,
    pub cooldown_seconds: u64,
}

impl AlertRule {
    /// Create new alert rule
    pub fn new(
        name: String,
        pattern_name: String,
        threshold_count: usize,
        time_window_seconds: u64,
        cooldown_seconds: u64,
    ) -> Self {
        Self {
            name,
            pattern_name,
            threshold_count,
            time_window_seconds,
            enabled: true,
            last_triggered: None,
            cooldown_seconds,
        }
    }

    /// Check if rule can be triggered (not in cooldown)
    pub fn can_trigger(&self) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(last_triggered) = self.last_triggered {
            let elapsed = SystemTime::now()
                .duration_since(last_triggered)
                .unwrap_or_default();
            elapsed.as_secs() >= self.cooldown_seconds
        } else {
            true
        }
    }

    /// Mark rule as triggered
    pub fn mark_triggered(&mut self) {
        self.last_triggered = Some(SystemTime::now());
    }
}

/// Pattern detection result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub message: String,
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub captures: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Alert notification
#[derive(Debug, Clone)]
pub struct AlertNotification {
    pub rule_name: String,
    pub pattern_name: String,
    pub message: String,
    pub severity: LogLevel,
    pub match_count: usize,
    pub time_window: Duration,
    pub timestamp: SystemTime,
    pub recent_matches: Vec<String>,
}

impl fmt::Display for AlertNotification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "🚨 ALERT: {} - {} matches of pattern '{}' in {:.1}s",
            self.rule_name,
            self.match_count,
            self.pattern_name,
            self.time_window.as_secs_f64()
        )
    }
}

/// Log analyzer with pattern detection and alerting
pub struct LogAnalyzer {
    enabled: bool,
    window_size: usize,

    // Pattern recognition
    patterns: HashMap<String, LogPattern>,
    pattern_matches: VecDeque<PatternMatch>,

    // Alert system
    alert_rules: HashMap<String, AlertRule>,
    triggered_alerts: Vec<AlertNotification>,

    // Statistics
    total_logs_analyzed: usize,
    patterns_detected: usize,
    alerts_triggered: usize,

    // Error tracking
    error_patterns: HashMap<String, usize>,
    recent_errors: VecDeque<String>,

    // Performance bottleneck detection
    performance_patterns: Vec<String>,
    bottleneck_keywords: Vec<String>,
}

impl fmt::Debug for LogAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogAnalyzer")
            .field("enabled", &self.enabled)
            .field("window_size", &self.window_size)
            .field("patterns_count", &self.patterns.len())
            .field("alert_rules_count", &self.alert_rules.len())
            .field("total_logs_analyzed", &self.total_logs_analyzed)
            .field("patterns_detected", &self.patterns_detected)
            .field("alerts_triggered", &self.alerts_triggered)
            .finish()
    }
}

impl LogAnalyzer {
    /// Create new log analyzer
    pub fn new(enabled: bool, window_size: usize) -> Self {
        let mut analyzer = Self {
            enabled,
            window_size,
            patterns: HashMap::new(),
            pattern_matches: VecDeque::new(),
            alert_rules: HashMap::new(),
            triggered_alerts: Vec::new(),
            total_logs_analyzed: 0,
            patterns_detected: 0,
            alerts_triggered: 0,
            error_patterns: HashMap::new(),
            recent_errors: VecDeque::new(),
            performance_patterns: Vec::new(),
            bottleneck_keywords: vec![
                "slow".to_string(),
                "timeout".to_string(),
                "bottleneck".to_string(),
                "performance".to_string(),
                "latency".to_string(),
                "memory".to_string(),
                "allocation".to_string(),
            ],
        };

        // Add default patterns
        analyzer.add_default_patterns().ok();
        analyzer.add_default_alert_rules();

        analyzer
    }

    /// Add default log patterns
    fn add_default_patterns(&mut self) -> RusTorchResult<()> {
        // Error patterns
        self.add_pattern(LogPattern::new(
            "out_of_memory".to_string(),
            r"(?i)(out of memory|oom|memory allocation failed)",
            LogLevel::Critical,
            "Memory allocation failure detected".to_string(),
            true,
        )?)?;

        self.add_pattern(LogPattern::new(
            "null_pointer".to_string(),
            r"(?i)(null pointer|segmentation fault|segfault)",
            LogLevel::Critical,
            "Memory access violation detected".to_string(),
            true,
        )?)?;

        self.add_pattern(LogPattern::new(
            "tensor_shape_mismatch".to_string(),
            r"(?i)(shape mismatch|dimension error|incompatible shapes)",
            LogLevel::Error,
            "Tensor shape compatibility issue".to_string(),
            true,
        )?)?;

        // Performance patterns
        self.add_pattern(LogPattern::new(
            "slow_operation".to_string(),
            r"(?i)(slow|timeout|took (\d+)ms|(\d+\.\d+)s)",
            LogLevel::Warning,
            "Performance issue detected".to_string(),
            false,
        )?)?;

        self.add_pattern(LogPattern::new(
            "high_memory_usage".to_string(),
            r"(?i)(memory usage|(\d+)MB|(\d+\.\d+)GB)",
            LogLevel::Info,
            "Memory usage information".to_string(),
            false,
        )?)?;

        // CUDA/GPU patterns
        self.add_pattern(LogPattern::new(
            "cuda_error".to_string(),
            r"(?i)(cuda error|gpu error|device error)",
            LogLevel::Error,
            "GPU/CUDA runtime error".to_string(),
            true,
        )?)?;

        // Network patterns
        self.add_pattern(LogPattern::new(
            "connection_error".to_string(),
            r"(?i)(connection (failed|refused|timeout)|network error)",
            LogLevel::Error,
            "Network connectivity issue".to_string(),
            true,
        )?)?;

        Ok(())
    }

    /// Add default alert rules
    fn add_default_alert_rules(&mut self) {
        self.add_alert_rule(AlertRule::new(
            "critical_errors".to_string(),
            "out_of_memory".to_string(),
            1,   // Any OOM error triggers alert
            60,  // Within 1 minute
            300, // 5 minute cooldown
        ));

        self.add_alert_rule(AlertRule::new(
            "frequent_shape_errors".to_string(),
            "tensor_shape_mismatch".to_string(),
            5,   // 5 shape errors
            300, // Within 5 minutes
            600, // 10 minute cooldown
        ));

        self.add_alert_rule(AlertRule::new(
            "performance_degradation".to_string(),
            "slow_operation".to_string(),
            10,  // 10 slow operations
            120, // Within 2 minutes
            300, // 5 minute cooldown
        ));

        self.add_alert_rule(AlertRule::new(
            "gpu_issues".to_string(),
            "cuda_error".to_string(),
            3,   // 3 CUDA errors
            180, // Within 3 minutes
            900, // 15 minute cooldown
        ));
    }

    /// Add log pattern
    pub fn add_pattern(&mut self, pattern: LogPattern) -> RusTorchResult<()> {
        self.patterns.insert(pattern.name.clone(), pattern);
        Ok(())
    }

    /// Add alert rule
    pub fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alert_rules.insert(rule.name.clone(), rule);
    }

    /// Analyze log entry for patterns
    pub fn analyze_log_entry(
        &mut self,
        level: LogLevel,
        message: &str,
        metadata: &HashMap<String, String>,
    ) -> RusTorchResult<Vec<PatternMatch>> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        self.total_logs_analyzed += 1;
        let mut matches = Vec::new();

        // Check against all patterns
        for pattern in self.patterns.values() {
            if pattern.matches(message) {
                let captures = pattern.extract_captures(message).unwrap_or_default();

                let pattern_match = PatternMatch {
                    pattern_name: pattern.name.clone(),
                    message: message.to_string(),
                    timestamp: SystemTime::now(),
                    level,
                    captures,
                    metadata: metadata.clone(),
                };

                matches.push(pattern_match.clone());
                self.pattern_matches.push_back(pattern_match);
                self.patterns_detected += 1;

                // Track error patterns
                if level >= LogLevel::Error {
                    *self.error_patterns.entry(pattern.name.clone()).or_insert(0) += 1;
                    self.recent_errors.push_back(message.to_string());

                    // Maintain recent errors window
                    if self.recent_errors.len() > 50 {
                        self.recent_errors.pop_front();
                    }
                }

                // Track performance patterns
                if self
                    .bottleneck_keywords
                    .iter()
                    .any(|keyword| message.to_lowercase().contains(keyword))
                {
                    self.performance_patterns.push(pattern.name.clone());
                }
            }
        }

        // Maintain pattern matches window
        while self.pattern_matches.len() > self.window_size {
            self.pattern_matches.pop_front();
        }

        // Check for triggered alerts
        self.check_alert_rules(&matches)?;

        Ok(matches)
    }

    /// Check alert rules and generate notifications
    fn check_alert_rules(&mut self, new_matches: &[PatternMatch]) -> RusTorchResult<()> {
        let now = SystemTime::now();

        for rule in self.alert_rules.values_mut() {
            if !rule.can_trigger() {
                continue;
            }

            // Count matching patterns within time window
            let window_start = now - Duration::from_secs(rule.time_window_seconds);
            let matching_patterns: Vec<&PatternMatch> = self
                .pattern_matches
                .iter()
                .filter(|m| m.pattern_name == rule.pattern_name && m.timestamp >= window_start)
                .collect();

            if matching_patterns.len() >= rule.threshold_count {
                // Create alert notification
                let recent_messages: Vec<String> = matching_patterns
                    .iter()
                    .rev()
                    .take(5)
                    .map(|m| m.message.clone())
                    .collect();

                let severity = matching_patterns
                    .iter()
                    .map(|m| m.level)
                    .max()
                    .unwrap_or(LogLevel::Warning);

                let alert = AlertNotification {
                    rule_name: rule.name.clone(),
                    pattern_name: rule.pattern_name.clone(),
                    message: format!("Alert triggered for pattern '{}'", rule.pattern_name),
                    severity,
                    match_count: matching_patterns.len(),
                    time_window: Duration::from_secs(rule.time_window_seconds),
                    timestamp: now,
                    recent_matches: recent_messages,
                };

                self.triggered_alerts.push(alert);
                rule.mark_triggered();
                self.alerts_triggered += 1;

                // In a real system, this would send notifications
                eprintln!("🚨 {}", self.triggered_alerts.last().unwrap());
            }
        }

        Ok(())
    }

    /// Get analysis summary
    pub fn get_analysis_summary(&self) -> AnalysisSummary {
        let most_common_errors: Vec<(String, usize)> = {
            let mut errors: Vec<(String, usize)> = self
                .error_patterns
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            errors.sort_by(|a, b| b.1.cmp(&a.1));
            errors.into_iter().take(5).collect()
        };

        let performance_bottlenecks: Vec<String> = {
            let mut bottlenecks = self.performance_patterns.clone();
            bottlenecks.sort();
            bottlenecks.dedup();
            bottlenecks
        };

        AnalysisSummary {
            patterns_detected: self.patterns_detected,
            alerts_triggered: self.alerts_triggered,
            most_common_errors,
            performance_bottlenecks,
        }
    }

    /// Get recent pattern matches
    pub fn get_recent_matches(&self, n: usize) -> Vec<&PatternMatch> {
        self.pattern_matches.iter().rev().take(n).collect()
    }

    /// Get matches for specific pattern
    pub fn get_pattern_matches(&self, pattern_name: &str) -> Vec<&PatternMatch> {
        self.pattern_matches
            .iter()
            .filter(|m| m.pattern_name == pattern_name)
            .collect()
    }

    /// Get all triggered alerts
    pub fn get_triggered_alerts(&self) -> &[AlertNotification] {
        &self.triggered_alerts
    }

    /// Get total alerts count
    pub fn get_total_alerts(&self) -> usize {
        self.alerts_triggered
    }

    /// Enable/disable pattern
    pub fn enable_pattern(&mut self, pattern_name: &str, enabled: bool) -> RusTorchResult<()> {
        if enabled {
            // Pattern enabling is implicit in this implementation
            Ok(())
        } else {
            self.patterns.remove(pattern_name);
            Ok(())
        }
    }

    /// Enable/disable alert rule
    pub fn enable_alert_rule(&mut self, rule_name: &str, enabled: bool) -> RusTorchResult<()> {
        if let Some(rule) = self.alert_rules.get_mut(rule_name) {
            rule.enabled = enabled;
            Ok(())
        } else {
            Err(RusTorchError::Debug {
                message: format!("Alert rule '{}' not found", rule_name),
            })
        }
    }

    /// Clear analysis data
    pub fn clear(&mut self) {
        self.pattern_matches.clear();
        self.triggered_alerts.clear();
        self.error_patterns.clear();
        self.recent_errors.clear();
        self.performance_patterns.clear();
        self.total_logs_analyzed = 0;
        self.patterns_detected = 0;
        self.alerts_triggered = 0;

        // Reset alert rule triggers
        for rule in self.alert_rules.values_mut() {
            rule.last_triggered = None;
        }
    }

    /// Generate analysis report
    pub fn generate_analysis_report(&self) -> String {
        let summary = self.get_analysis_summary();

        let mut report = String::new();
        report.push_str("🔍 Log Analysis Report\n");
        report.push_str("=====================\n\n");

        report.push_str(&format!("📊 Analysis Statistics:\n"));
        report.push_str(&format!("  Logs Analyzed: {}\n", self.total_logs_analyzed));
        report.push_str(&format!(
            "  Patterns Detected: {}\n",
            summary.patterns_detected
        ));
        report.push_str(&format!(
            "  Alerts Triggered: {}\n",
            summary.alerts_triggered
        ));
        report.push_str(&format!("  Active Patterns: {}\n", self.patterns.len()));
        report.push_str(&format!("  Alert Rules: {}\n\n", self.alert_rules.len()));

        if !summary.most_common_errors.is_empty() {
            report.push_str("🚨 Most Common Error Patterns:\n");
            for (i, (pattern, count)) in summary.most_common_errors.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. {}: {} occurrences\n",
                    i + 1,
                    pattern,
                    count
                ));
            }
            report.push('\n');
        }

        if !summary.performance_bottlenecks.is_empty() {
            report.push_str("⚡ Performance Issues Detected:\n");
            for (i, bottleneck) in summary.performance_bottlenecks.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, bottleneck));
            }
            report.push('\n');
        }

        if !self.triggered_alerts.is_empty() {
            report.push_str("🔔 Recent Alerts:\n");
            for (i, alert) in self.triggered_alerts.iter().rev().take(5).enumerate() {
                report.push_str(&format!(
                    "  {}. {} ({})\n",
                    i + 1,
                    alert.rule_name,
                    alert.severity
                ));
            }
        }

        report
    }

    /// Enable/disable analyzer
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if analyzer is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_pattern_creation() {
        let pattern = LogPattern::new(
            "test_pattern".to_string(),
            r"error: (.+)",
            LogLevel::Error,
            "Test pattern".to_string(),
            true,
        )
        .unwrap();

        assert_eq!(pattern.name, "test_pattern");
        assert_eq!(pattern.severity, LogLevel::Error);
        assert!(pattern.action_required);
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = LogPattern::new(
            "error_pattern".to_string(),
            r"(?i)error: (.+)",
            LogLevel::Error,
            "Error pattern".to_string(),
            true,
        )
        .unwrap();

        assert!(pattern.matches("Error: something went wrong"));
        assert!(pattern.matches("ERROR: critical failure"));
        assert!(!pattern.matches("Warning: minor issue"));

        let captures = pattern.extract_captures("Error: file not found");
        assert_eq!(captures, Some(vec!["file not found".to_string()]));
    }

    #[test]
    fn test_alert_rule_creation() {
        let mut rule = AlertRule::new(
            "test_rule".to_string(),
            "error_pattern".to_string(),
            5,   // threshold
            300, // 5 minute window
            60,  // 1 minute cooldown
        );

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.threshold_count, 5);
        assert!(rule.can_trigger());

        rule.mark_triggered();
        assert!(!rule.can_trigger()); // Should be in cooldown
    }

    #[test]
    fn test_log_analyzer_creation() {
        let analyzer = LogAnalyzer::new(true, 1000);

        assert!(analyzer.is_enabled());
        assert!(analyzer.patterns.len() > 0); // Should have default patterns
        assert!(analyzer.alert_rules.len() > 0); // Should have default rules
    }

    #[test]
    fn test_log_analysis() {
        let mut analyzer = LogAnalyzer::new(true, 1000);

        let metadata = HashMap::new();

        // Test error pattern matching
        let matches = analyzer
            .analyze_log_entry(LogLevel::Error, "Out of memory error occurred", &metadata)
            .unwrap();

        assert!(!matches.is_empty());
        assert_eq!(matches[0].pattern_name, "out_of_memory");

        // Test non-matching message
        let matches = analyzer
            .analyze_log_entry(LogLevel::Info, "Normal operation completed", &metadata)
            .unwrap();

        assert!(matches.is_empty());
    }

    #[test]
    fn test_pattern_statistics() {
        let mut analyzer = LogAnalyzer::new(true, 1000);
        let metadata = HashMap::new();

        // Generate multiple error events
        for _ in 0..5 {
            analyzer
                .analyze_log_entry(
                    LogLevel::Error,
                    "Shape mismatch error in tensor operation",
                    &metadata,
                )
                .unwrap();
        }

        let summary = analyzer.get_analysis_summary();
        assert_eq!(summary.patterns_detected, 5);

        // Should have tensor_shape_mismatch in most common errors
        let has_shape_error = summary
            .most_common_errors
            .iter()
            .any(|(pattern, count)| pattern == "tensor_shape_mismatch" && *count == 5);
        assert!(has_shape_error);
    }

    #[test]
    fn test_performance_bottleneck_detection() {
        let mut analyzer = LogAnalyzer::new(true, 1000);
        let metadata = HashMap::new();

        // Log performance issues
        analyzer
            .analyze_log_entry(
                LogLevel::Warning,
                "Operation took 500ms - performance bottleneck detected",
                &metadata,
            )
            .unwrap();

        let summary = analyzer.get_analysis_summary();
        assert!(!summary.performance_bottlenecks.is_empty());
    }

    #[test]
    fn test_analyzer_disabled() {
        let mut analyzer = LogAnalyzer::new(false, 1000);
        let metadata = HashMap::new();

        let matches = analyzer
            .analyze_log_entry(LogLevel::Error, "Out of memory error occurred", &metadata)
            .unwrap();

        assert!(matches.is_empty());
        assert_eq!(analyzer.total_logs_analyzed, 0);
    }

    #[test]
    fn test_pattern_enable_disable() {
        let mut analyzer = LogAnalyzer::new(true, 1000);

        // Disable a pattern
        let initial_count = analyzer.patterns.len();
        analyzer.enable_pattern("out_of_memory", false).unwrap();
        assert_eq!(analyzer.patterns.len(), initial_count - 1);
    }

    #[test]
    fn test_alert_rule_enable_disable() {
        let mut analyzer = LogAnalyzer::new(true, 1000);

        // Disable an alert rule
        analyzer
            .enable_alert_rule("critical_errors", false)
            .unwrap();

        let rule = analyzer.alert_rules.get("critical_errors").unwrap();
        assert!(!rule.enabled);

        // Test invalid rule name
        let result = analyzer.enable_alert_rule("nonexistent_rule", false);
        assert!(result.is_err());
    }
}
