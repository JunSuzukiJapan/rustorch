/// Model download functionality for various platforms
pub mod huggingface;
pub mod ollama;
pub mod manager;
pub mod progress;

pub use manager::ModelDownloadManager;
pub use huggingface::HuggingFaceClient;
pub use ollama::OllamaClient;

use anyhow::Result;
use std::path::PathBuf;

/// Supported model sources
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    HuggingFace,
    Ollama,
    ModelScope,
}

impl ModelSource {
    pub fn from_prefix(prefix: &str) -> Result<Self> {
        match prefix.to_lowercase().as_str() {
            "hf" | "huggingface" => Ok(ModelSource::HuggingFace),
            "ollama" => Ok(ModelSource::Ollama),
            "ms" | "modelscope" => Ok(ModelSource::ModelScope),
            _ => anyhow::bail!("Unknown model source: {}. Supported: hf, ollama, ms", prefix),
        }
    }
}

/// Model identifier with source
#[derive(Debug, Clone)]
pub struct ModelIdentifier {
    pub source: ModelSource,
    pub repo_id: String,
    pub filename: Option<String>,
    pub revision: Option<String>,
}

impl ModelIdentifier {
    /// Parse model identifier from string like "hf:TheBloke/Llama-2-7B-GGUF"
    pub fn parse(input: &str) -> Result<Self> {
        let parts: Vec<&str> = input.splitn(2, ':').collect();

        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid model identifier format. Expected: <source>:<model-id>\n\
                 Examples:\n\
                 - hf:TheBloke/Llama-2-7B-GGUF\n\
                 - ollama:llama2:7b\n\
                 - ms:qwen/Qwen-7B"
            );
        }

        let source = ModelSource::from_prefix(parts[0])?;
        let repo_id = parts[1].to_string();

        Ok(ModelIdentifier {
            source,
            repo_id,
            filename: None,
            revision: None,
        })
    }

    pub fn with_filename(mut self, filename: String) -> Self {
        self.filename = Some(filename);
        self
    }

    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = Some(revision);
        self
    }
}

/// Download options
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub output_dir: PathBuf,
    pub format: Option<String>,
    pub quantization: Option<String>,
    pub force: bool,
    pub token: Option<String>,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            output_dir: Self::default_model_dir(),
            format: None,
            quantization: None,
            force: false,
            token: None,
        }
    }
}

impl DownloadOptions {
    /// Get default model directory (~/.rustorch/models)
    pub fn default_model_dir() -> PathBuf {
        if let Some(custom_dir) = std::env::var("RUSTORCH_MODEL_DIR").ok() {
            PathBuf::from(custom_dir)
        } else if let Some(home_dir) = dirs::home_dir() {
            home_dir.join(".rustorch").join("models")
        } else {
            PathBuf::from("models")
        }
    }

    /// Get HuggingFace token from environment
    pub fn get_hf_token() -> Option<String> {
        std::env::var("HF_TOKEN").ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_source_from_prefix() {
        assert_eq!(ModelSource::from_prefix("hf").unwrap(), ModelSource::HuggingFace);
        assert_eq!(ModelSource::from_prefix("huggingface").unwrap(), ModelSource::HuggingFace);
        assert_eq!(ModelSource::from_prefix("ollama").unwrap(), ModelSource::Ollama);
        assert_eq!(ModelSource::from_prefix("ms").unwrap(), ModelSource::ModelScope);
        assert!(ModelSource::from_prefix("unknown").is_err());
    }

    #[test]
    fn test_model_identifier_parse() {
        let id = ModelIdentifier::parse("hf:TheBloke/Llama-2-7B-GGUF").unwrap();
        assert_eq!(id.source, ModelSource::HuggingFace);
        assert_eq!(id.repo_id, "TheBloke/Llama-2-7B-GGUF");

        let id = ModelIdentifier::parse("ollama:llama2:7b").unwrap();
        assert_eq!(id.source, ModelSource::Ollama);
        assert_eq!(id.repo_id, "llama2:7b");

        assert!(ModelIdentifier::parse("invalid").is_err());
    }

    #[test]
    fn test_download_options_default() {
        let opts = DownloadOptions::default();
        assert!(opts.output_dir.to_string_lossy().contains(".rustorch"));
        assert_eq!(opts.format, None);
        assert_eq!(opts.force, false);
    }
}
