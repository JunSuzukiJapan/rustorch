use tracing_subscriber::{fmt, EnvFilter};
use crate::cli::LogLevel;

pub fn init_logger(log_level: LogLevel) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level.as_str()));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .init();
}
