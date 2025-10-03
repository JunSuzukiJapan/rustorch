use indicatif::{ProgressBar, ProgressStyle};

pub struct ProgressIndicator {
    pb: ProgressBar,
}

impl ProgressIndicator {
    pub fn new(message: &str) -> Self {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .expect("Failed to set progress style"),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        Self { pb }
    }

    pub fn update_message(&self, message: &str) {
        self.pb.set_message(message.to_string());
    }

    pub fn finish(&self) {
        self.pb.finish_and_clear();
    }

    pub fn finish_with_message(&self, message: &str) {
        self.pb.finish_with_message(message.to_string());
    }
}

impl Drop for ProgressIndicator {
    fn drop(&mut self) {
        self.pb.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_indicator_creation() {
        let progress = ProgressIndicator::new("Testing...");
        progress.finish();
    }

    #[test]
    fn test_progress_indicator_update() {
        let progress = ProgressIndicator::new("Initial");
        progress.update_message("Updated");
        progress.finish();
    }

    #[test]
    fn test_progress_indicator_finish_with_message() {
        let progress = ProgressIndicator::new("Working...");
        progress.finish_with_message("Done!");
    }
}
