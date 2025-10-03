// Performance metrics reporting and formatting

use super::MetricsCollector;
use std::fmt;

/// Report format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Human-readable text
    Text,
    /// JSON format
    Json,
    /// Markdown table
    Markdown,
}

/// Performance report generator
pub struct PerformanceReporter {
    metrics: MetricsCollector,
    format: ReportFormat,
}

impl PerformanceReporter {
    /// Create new performance reporter
    pub fn new(metrics: MetricsCollector, format: ReportFormat) -> Self {
        Self { metrics, format }
    }

    /// Generate report
    pub fn generate(&self) -> String {
        match self.format {
            ReportFormat::Text => self.generate_text(),
            ReportFormat::Json => self.generate_json(),
            ReportFormat::Markdown => self.generate_markdown(),
        }
    }

    /// Generate text report
    fn generate_text(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Performance Metrics Report ===\n\n");

        if let Some(backend) = &self.metrics.backend {
            report.push_str(&format!("Backend: {}\n", backend));
        }

        if let Some(ttft) = self.metrics.ttft {
            let status = if self.metrics.meets_ttft_target() {
                "✓"
            } else {
                "✗"
            };
            report.push_str(&format!("Time to First Token: {:.2} ms {}\n", ttft, status));
        }

        if let Some(tps) = self.metrics.tokens_per_sec {
            let status = if self.metrics.meets_tps_target() {
                "✓"
            } else {
                "✗"
            };
            report.push_str(&format!("Tokens/sec: {:.2} {}\n", tps, status));
        }

        if let Some(total) = self.metrics.total_time {
            report.push_str(&format!("Total Time: {:.2} ms\n", total));
        }

        if let Some(mem) = self.metrics.memory_usage {
            let mb = mem as f64 / 1_048_576.0;
            report.push_str(&format!("Memory Usage: {:.2} MB\n", mb));
        }

        if let Some(gpu_mem) = self.metrics.gpu_memory_usage {
            let mb = gpu_mem as f64 / 1_048_576.0;
            report.push_str(&format!("GPU Memory: {:.2} MB\n", mb));
        }

        if let Some(model) = self.metrics.model_size {
            let mb = model as f64 / 1_048_576.0;
            report.push_str(&format!("Model Size: {:.2} MB\n", mb));
        }

        if let Some(efficiency) = self.metrics.memory_efficiency() {
            report.push_str(&format!("Memory Efficiency: {:.2}x\n", efficiency));
        }

        if !self.metrics.custom.is_empty() {
            report.push_str("\nCustom Metrics:\n");
            for (key, value) in &self.metrics.custom {
                report.push_str(&format!("  {}: {:.2}\n", key, value));
            }
        }

        report.push_str("\n");
        if self.metrics.meets_all_targets() {
            report.push_str("✓ All performance targets met!\n");
        } else {
            report.push_str("✗ Some performance targets not met\n");
        }

        report
    }

    /// Generate JSON report
    fn generate_json(&self) -> String {
        let mut json = String::from("{\n");

        if let Some(ref backend) = self.metrics.backend {
            json.push_str(&format!("  \"backend\": \"{}\",\n", backend));
        }

        if let Some(ttft) = self.metrics.ttft {
            json.push_str(&format!("  \"ttft_ms\": {:.2},\n", ttft));
        }

        if let Some(tps) = self.metrics.tokens_per_sec {
            json.push_str(&format!("  \"tokens_per_sec\": {:.2},\n", tps));
        }

        if let Some(total) = self.metrics.total_time {
            json.push_str(&format!("  \"total_time_ms\": {:.2},\n", total));
        }

        if let Some(mem) = self.metrics.memory_usage {
            json.push_str(&format!("  \"memory_usage_bytes\": {},\n", mem));
        }

        if let Some(gpu_mem) = self.metrics.gpu_memory_usage {
            json.push_str(&format!("  \"gpu_memory_bytes\": {},\n", gpu_mem));
        }

        if let Some(model) = self.metrics.model_size {
            json.push_str(&format!("  \"model_size_bytes\": {},\n", model));
        }

        json.push_str(&format!(
            "  \"meets_targets\": {}\n",
            self.metrics.meets_all_targets()
        ));
        json.push('}');

        json
    }

    /// Generate Markdown report
    fn generate_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Performance Metrics Report\n\n");

        md.push_str("| Metric | Value | Target | Status |\n");
        md.push_str("|--------|-------|--------|--------|\n");

        if let Some(backend) = &self.metrics.backend {
            md.push_str(&format!("| Backend | {} | - | - |\n", backend));
        }

        if let Some(ttft) = self.metrics.ttft {
            let status = if self.metrics.meets_ttft_target() {
                "✓"
            } else {
                "✗"
            };
            md.push_str(&format!(
                "| TTFT | {:.2} ms | <200 ms | {} |\n",
                ttft, status
            ));
        }

        if let Some(tps) = self.metrics.tokens_per_sec {
            let status = if self.metrics.meets_tps_target() {
                "✓"
            } else {
                "✗"
            };
            md.push_str(&format!("| Tokens/sec | {:.2} | >20 | {} |\n", tps, status));
        }

        if let Some(total) = self.metrics.total_time {
            md.push_str(&format!("| Total Time | {:.2} ms | - | - |\n", total));
        }

        if let Some(mem) = self.metrics.memory_usage {
            let mb = mem as f64 / 1_048_576.0;
            let status = if self.metrics.meets_memory_target() {
                "✓"
            } else {
                "✗"
            };
            md.push_str(&format!(
                "| Memory | {:.2} MB | <1.5x model | {} |\n",
                mb, status
            ));
        }

        if let Some(gpu_mem) = self.metrics.gpu_memory_usage {
            let mb = gpu_mem as f64 / 1_048_576.0;
            md.push_str(&format!("| GPU Memory | {:.2} MB | - | - |\n", mb));
        }

        md.push_str("\n");
        if self.metrics.meets_all_targets() {
            md.push_str("**✓ All performance targets met!**\n");
        } else {
            md.push_str("**✗ Some performance targets not met**\n");
        }

        md
    }
}

impl fmt::Display for PerformanceReporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.generate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics() -> MetricsCollector {
        let mut metrics = MetricsCollector::new();
        metrics.set_backend("metal".to_string());
        metrics.set_ttft(150.0);
        metrics.set_tokens_per_sec(25.0);
        metrics.set_total_time(5000.0);
        metrics.set_memory_usage(1_200_000_000);
        metrics.set_model_size(1_000_000_000);
        metrics
    }

    #[test]
    fn test_text_report() {
        let metrics = create_test_metrics();
        let reporter = PerformanceReporter::new(metrics, ReportFormat::Text);
        let report = reporter.generate();

        assert!(report.contains("Backend: metal"));
        assert!(report.contains("Time to First Token: 150.00 ms"));
        assert!(report.contains("Tokens/sec: 25.00"));
        assert!(report.contains("All performance targets met"));
    }

    #[test]
    fn test_json_report() {
        let metrics = create_test_metrics();
        let reporter = PerformanceReporter::new(metrics, ReportFormat::Json);
        let report = reporter.generate();

        assert!(report.contains("\"backend\": \"metal\""));
        assert!(report.contains("\"ttft_ms\": 150.00"));
        assert!(report.contains("\"tokens_per_sec\": 25.00"));
        assert!(report.contains("\"meets_targets\": true"));
    }

    #[test]
    fn test_markdown_report() {
        let metrics = create_test_metrics();
        let reporter = PerformanceReporter::new(metrics, ReportFormat::Markdown);
        let report = reporter.generate();

        assert!(report.contains("# Performance Metrics Report"));
        assert!(report.contains("| Backend | metal |"));
        assert!(report.contains("| TTFT | 150.00 ms | <200 ms | ✓ |"));
        assert!(report.contains("All performance targets met"));
    }

    #[test]
    fn test_display_trait() {
        let metrics = create_test_metrics();
        let reporter = PerformanceReporter::new(metrics, ReportFormat::Text);
        let display_output = format!("{}", reporter);
        let generate_output = reporter.generate();

        assert_eq!(display_output, generate_output);
    }
}
