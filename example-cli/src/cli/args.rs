use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "rustorch-cli",
    version,
    about = "Interactive CLI for local LLM inference using RusTorch",
    long_about = "RusTorch CLI provides an interactive REPL interface for running local \
                  large language models with support for multiple backends (CPU/GPU/Neural Engine)."
)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path to the model file
    #[arg(short, long, value_name = "FILE", global = true)]
    pub model: Option<PathBuf>,

    /// Computation backend
    #[arg(short, long, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// Path to configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Log level
    #[arg(long, value_enum, default_value = "info")]
    pub log_level: LogLevel,

    /// Maximum tokens to generate
    #[arg(long, default_value = "2048")]
    pub max_tokens: usize,

    /// Sampling temperature (0.0-2.0)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p sampling threshold
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling threshold
    #[arg(long, default_value = "40")]
    pub top_k: u32,

    /// Save conversation history to file
    #[arg(long, value_name = "FILE")]
    pub save_history: Option<PathBuf>,

    /// Load conversation history from file
    #[arg(long, value_name = "FILE")]
    pub load_history: Option<PathBuf>,

    /// Path to tokenizer file
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,

    /// Disable progress bar
    #[arg(long)]
    pub no_progress: bool,

    /// System prompt
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Use TUI mode (Terminal User Interface) with status bar
    #[arg(long)]
    pub tui: bool,

    /// Input token IDs directly (comma-separated, bypasses tokenizer)
    /// Example: --tokens "15043,29892,2787"
    #[arg(long, value_name = "IDS")]
    pub tokens: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Backend {
    /// CPU computation
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Apple Metal GPU
    Metal,
    /// OpenCL GPU
    Opencl,
    /// Hybrid mode (automatic backend selection)
    Hybrid,
    /// Hybrid mode with F32 precision
    HybridF32,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Opencl => "opencl",
            Backend::Hybrid => "hybrid",
            Backend::HybridF32 => "hybrid-f32",
        }
    }

    pub fn is_available(&self) -> bool {
        match self {
            Backend::Cpu => true,
            #[cfg(feature = "cuda")]
            Backend::Cuda => true,
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => false,
            #[cfg(feature = "metal")]
            Backend::Metal => true,
            #[cfg(not(feature = "metal"))]
            Backend::Metal => false,
            #[cfg(feature = "opencl")]
            Backend::Opencl => true,
            #[cfg(not(feature = "opencl"))]
            Backend::Opencl => false,
            #[cfg(feature = "mac-hybrid")]
            Backend::Hybrid => true,
            #[cfg(not(feature = "mac-hybrid"))]
            Backend::Hybrid => false,
            #[cfg(feature = "hybrid-f32")]
            Backend::HybridF32 => true,
            #[cfg(not(feature = "hybrid-f32"))]
            Backend::HybridF32 => false,
        }
    }

    pub fn auto_detect() -> Self {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return Backend::Metal;
        }

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            return Backend::Cuda;
        }

        #[cfg(feature = "opencl")]
        {
            return Backend::Opencl;
        }

        Backend::Cpu
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

impl CliArgs {
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate temperature
        if self.temperature < 0.0 || self.temperature > 2.0 {
            anyhow::bail!("Temperature must be between 0.0 and 2.0");
        }

        // Validate top_p
        if self.top_p < 0.0 || self.top_p > 1.0 {
            anyhow::bail!("Top-p must be between 0.0 and 1.0");
        }

        // Validate max_tokens
        if self.max_tokens == 0 {
            anyhow::bail!("Max tokens must be greater than 0");
        }

        // Check backend availability
        if !self.backend.is_available() {
            anyhow::bail!(
                "Backend '{}' is not available. Please recompile with the appropriate feature flag.",
                self.backend.as_str()
            );
        }

        Ok(())
    }

    pub fn get_default_config_path() -> PathBuf {
        if let Some(config_dir) = directories::BaseDirs::new() {
            config_dir.config_dir().join("rustorch").join("config.toml")
        } else {
            PathBuf::from("config.toml")
        }
    }

    pub fn get_default_history_path() -> PathBuf {
        if let Some(data_dir) = directories::BaseDirs::new() {
            data_dir
                .data_dir()
                .join("rustorch")
                .join("history")
                .join("latest.json")
        } else {
            PathBuf::from("history.json")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_string_conversion() {
        assert_eq!(Backend::Cpu.as_str(), "cpu");
        assert_eq!(Backend::Cuda.as_str(), "cuda");
        assert_eq!(Backend::Metal.as_str(), "metal");
    }

    #[test]
    fn test_log_level_string_conversion() {
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Debug.as_str(), "debug");
    }

    #[test]
    fn test_cpu_backend_always_available() {
        assert!(Backend::Cpu.is_available());
    }

    #[test]
    fn test_validate_temperature() {
        let mut args = CliArgs {
            command: None,
            model: None,
            backend: Backend::Cpu,
            config: None,
            log_level: LogLevel::Info,
            max_tokens: 2048,
            temperature: 2.5, // Invalid
            top_p: 0.9,
            top_k: 40,
            save_history: None,
            load_history: None,
            tokenizer: None,
            no_progress: false,
            system_prompt: None,
        };

        assert!(args.validate().is_err());

        args.temperature = 0.7;
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_top_p() {
        let mut args = CliArgs {
            command: None,
            model: None,
            backend: Backend::Cpu,
            config: None,
            log_level: LogLevel::Info,
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 1.5, // Invalid
            top_k: 40,
            save_history: None,
            load_history: None,
            tokenizer: None,
            no_progress: false,
            system_prompt: None,
        };

        assert!(args.validate().is_err());

        args.top_p = 0.9;
        assert!(args.validate().is_ok());
    }
}

/// Subcommands for RusTorch CLI
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Download models from HuggingFace, Ollama, or other sources
    Download {
        /// Model identifier (e.g., hf:TheBloke/Llama-2-7B-GGUF, ollama:llama2:7b)
        #[arg(value_name = "MODEL_ID")]
        model_id: String,

        /// Output directory for downloaded models
        #[arg(short, long, value_name = "DIR")]
        output_dir: Option<PathBuf>,

        /// Model format preference (gguf, safetensors, pytorch)
        #[arg(short, long)]
        format: Option<String>,

        /// Quantization level (q4_0, q4_k_m, q8_0, etc.)
        #[arg(short, long)]
        quantization: Option<String>,

        /// Force re-download even if file exists
        #[arg(long)]
        force: bool,

        /// HuggingFace token for private models
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },

    /// List available models from a source
    List {
        /// Model source (ollama, hf, ms)
        #[arg(value_name = "SOURCE")]
        source: String,
    },
}
