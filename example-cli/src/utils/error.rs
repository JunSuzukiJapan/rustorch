use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Session error: {0}")]
    SessionError(String),
}

pub type Result<T> = std::result::Result<T, CliError>;
