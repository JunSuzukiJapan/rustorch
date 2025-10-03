pub mod backend;
pub mod cli;
pub mod download;
pub mod metrics;
pub mod model;
pub mod session;
pub mod tokenizer;
pub mod utils;

// Re-exports
pub use backend::Backend as ComputeBackend;
pub use cli::{Backend, CliArgs, Commands, LogLevel, REPL};
pub use download::{DownloadOptions, ModelDownloadManager, ModelIdentifier};
pub use model::{InferenceEngine, KVCache, ModelLoader, TransformerConfig, TransformerModel};
pub use session::{GenerationConfig, SessionManager};
pub use tokenizer::{Tokenizer, TokenizerWrapper};
pub use utils::{init_logger, CliError, Config, ProgressIndicator};
