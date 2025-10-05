// High-precision timing utilities for performance measurement

use std::time::{Duration, Instant};

/// Timer for measuring execution time
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    /// Create and start a new timer
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: label.into(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed time in seconds
    pub fn elapsed_sec(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    /// Get label
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Multi-stage timer for complex workflows
#[derive(Debug)]
pub struct MultiStageTimer {
    stages: Vec<(String, Duration)>,
    current_stage: Option<Timer>,
}

impl MultiStageTimer {
    /// Create new multi-stage timer
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            current_stage: None,
        }
    }

    /// Start a new stage
    pub fn start_stage(&mut self, label: impl Into<String>) {
        // End previous stage if exists
        if let Some(timer) = self.current_stage.take() {
            let elapsed = timer.elapsed();
            self.stages.push((timer.label, elapsed));
        }
        // Start new stage
        self.current_stage = Some(Timer::new(label));
    }

    /// End current stage
    pub fn end_stage(&mut self) {
        if let Some(timer) = self.current_stage.take() {
            let elapsed = timer.elapsed();
            self.stages.push((timer.label, elapsed));
        }
    }

    /// Get all stage timings
    pub fn stages(&self) -> &[(String, Duration)] {
        &self.stages
    }

    /// Get total time across all stages
    pub fn total_time(&self) -> Duration {
        self.stages.iter().map(|(_, d)| *d).sum()
    }

    /// Get total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_time().as_secs_f64() * 1000.0
    }

    /// Get stage time in milliseconds
    pub fn stage_ms(&self, label: &str) -> Option<f64> {
        self.stages
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, d)| d.as_secs_f64() * 1000.0)
    }
}

impl Default for MultiStageTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Inference-specific timer
pub struct InferenceTimer {
    /// Time to first token
    pub ttft: Option<Duration>,
    /// Per-token generation times
    pub token_times: Vec<Duration>,
    /// Total inference time
    pub total_time: Option<Duration>,
    /// Internal timer
    timer: Option<Timer>,
    /// First token generated flag
    first_token_done: bool,
}

impl InferenceTimer {
    /// Create new inference timer
    pub fn new() -> Self {
        Self {
            ttft: None,
            token_times: Vec::new(),
            total_time: None,
            timer: None,
            first_token_done: false,
        }
    }

    /// Start inference timing
    pub fn start(&mut self) {
        self.timer = Some(Timer::new("inference"));
        self.first_token_done = false;
    }

    /// Mark first token generation
    pub fn mark_first_token(&mut self) {
        if let Some(ref timer) = self.timer {
            if !self.first_token_done {
                self.ttft = Some(timer.elapsed());
                self.first_token_done = true;
            }
        }
    }

    /// Mark token generation
    pub fn mark_token(&mut self) {
        if let Some(ref timer) = self.timer {
            self.token_times.push(timer.elapsed());
        }
    }

    /// End inference timing
    pub fn end(&mut self) {
        if let Some(timer) = self.timer.take() {
            self.total_time = Some(timer.elapsed());
        }
    }

    /// Get TTFT in milliseconds
    pub fn ttft_ms(&self) -> Option<f64> {
        self.ttft.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get total time in milliseconds
    pub fn total_ms(&self) -> Option<f64> {
        self.total_time.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Calculate tokens per second
    pub fn tokens_per_sec(&self) -> Option<f64> {
        match self.total_time {
            Some(total) if total.as_secs_f64() > 0.0 => {
                let count = self.token_times.len() as f64;
                Some(count / total.as_secs_f64())
            }
            _ => None,
        }
    }

    /// Get average token generation time in milliseconds
    pub fn avg_token_time_ms(&self) -> Option<f64> {
        if self.token_times.is_empty() {
            return None;
        }
        let sum: Duration = self.token_times.iter().sum();
        Some(sum.as_secs_f64() * 1000.0 / self.token_times.len() as f64)
    }
}

impl Default for InferenceTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);
        assert_eq!(timer.label(), "test");
    }

    #[test]
    fn test_multi_stage_timer() {
        let mut timer = MultiStageTimer::new();

        timer.start_stage("stage1");
        thread::sleep(Duration::from_millis(10));

        timer.start_stage("stage2");
        thread::sleep(Duration::from_millis(10));

        timer.end_stage();

        assert_eq!(timer.stages().len(), 2);
        assert!(timer.total_ms() >= 20.0);
    }

    #[test]
    fn test_inference_timer() {
        let mut timer = InferenceTimer::new();
        timer.start();

        thread::sleep(Duration::from_millis(10));
        timer.mark_first_token();

        thread::sleep(Duration::from_millis(10));
        timer.mark_token();

        thread::sleep(Duration::from_millis(10));
        timer.mark_token();

        timer.end();

        assert!(timer.ttft_ms().unwrap() >= 10.0);
        assert_eq!(timer.token_times.len(), 2);
        assert!(timer.total_ms().unwrap() >= 30.0);
    }
}
