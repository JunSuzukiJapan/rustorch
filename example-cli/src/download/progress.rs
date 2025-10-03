/// Download progress tracking and display
use std::time::{Duration, Instant};

/// Progress bar for downloads
pub struct ProgressBar {
    pub total: u64,
    pub downloaded: u64,
    pub start_time: Instant,
    pub last_update: Instant,
    pub filename: String,
}

impl ProgressBar {
    pub fn new(filename: String, total: u64) -> Self {
        let now = Instant::now();
        Self {
            total,
            downloaded: 0,
            start_time: now,
            last_update: now,
            filename,
        }
    }

    pub fn update(&mut self, downloaded: u64) {
        self.downloaded = downloaded;
        self.last_update = Instant::now();
    }

    pub fn finish(&self) {
        println!("\n✓ Download complete: {}", self.filename);
    }

    pub fn render(&self) -> String {
        let percentage = if self.total > 0 {
            (self.downloaded as f64 / self.total as f64 * 100.0) as u32
        } else {
            0
        };

        let elapsed = self.start_time.elapsed();
        let speed = if elapsed.as_secs() > 0 {
            self.downloaded / elapsed.as_secs()
        } else {
            0
        };

        let eta = if speed > 0 && self.total > self.downloaded {
            let remaining = self.total - self.downloaded;
            Duration::from_secs(remaining / speed)
        } else {
            Duration::from_secs(0)
        };

        format!(
            "\r{}: {} / {} ({}%) - {} - ETA: {}",
            self.filename,
            Self::format_bytes(self.downloaded),
            Self::format_bytes(self.total),
            percentage,
            Self::format_speed(speed),
            Self::format_duration(eta)
        )
    }

    pub fn display(&self) {
        print!("{}", self.render());
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    fn format_speed(bytes_per_sec: u64) -> String {
        format!("{}/s", Self::format_bytes(bytes_per_sec))
    }

    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();

        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }
}

/// Simple spinner for indeterminate progress
pub struct Spinner {
    pub frames: Vec<&'static str>,
    pub current_frame: usize,
    pub message: String,
    pub last_update: Instant,
}

impl Spinner {
    pub fn new(message: String) -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            current_frame: 0,
            message,
            last_update: Instant::now(),
        }
    }

    pub fn tick(&mut self) {
        if self.last_update.elapsed() > Duration::from_millis(80) {
            self.current_frame = (self.current_frame + 1) % self.frames.len();
            self.last_update = Instant::now();
        }
    }

    pub fn display(&self) {
        print!("\r{} {}", self.frames[self.current_frame], self.message);
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    pub fn finish(&self, final_message: &str) {
        println!("\r✓ {}", final_message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_creation() {
        let pb = ProgressBar::new("test.bin".to_string(), 1000);
        assert_eq!(pb.total, 1000);
        assert_eq!(pb.downloaded, 0);
        assert_eq!(pb.filename, "test.bin");
    }

    #[test]
    fn test_progress_bar_update() {
        let mut pb = ProgressBar::new("test.bin".to_string(), 1000);
        pb.update(500);
        assert_eq!(pb.downloaded, 500);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(ProgressBar::format_bytes(512), "512.00 B");
        assert_eq!(ProgressBar::format_bytes(1024), "1.00 KB");
        assert_eq!(ProgressBar::format_bytes(1_048_576), "1.00 MB");
        assert_eq!(ProgressBar::format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(ProgressBar::format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(
            ProgressBar::format_duration(Duration::from_secs(90)),
            "1m 30s"
        );
        assert_eq!(
            ProgressBar::format_duration(Duration::from_secs(3661)),
            "1h 1m"
        );
    }

    #[test]
    fn test_spinner_creation() {
        let spinner = Spinner::new("Loading...".to_string());
        assert_eq!(spinner.message, "Loading...");
        assert_eq!(spinner.current_frame, 0);
    }

    #[test]
    fn test_spinner_tick() {
        let mut spinner = Spinner::new("Loading...".to_string());
        let initial_frame = spinner.current_frame;

        // Force tick by setting last_update to past
        spinner.last_update = Instant::now() - Duration::from_millis(100);
        spinner.tick();

        // Frame should have advanced (with wrapping)
        assert_ne!(spinner.current_frame, initial_frame);
    }
}
