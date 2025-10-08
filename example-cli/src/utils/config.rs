/// Configuration management module
///
/// Handles loading and saving TOML configuration files for RusTorch CLI.
/// Configuration priority: CLI args > Environment variables > Config file > Defaults
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::session::GenerationConfig;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub generation: GenerationConfig,
    #[serde(default)]
    pub backend: BackendConfig,
    #[serde(default)]
    pub session: SessionConfig,
    #[serde(default)]
    pub ui: UiConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model path
    pub default: Option<String>,
    /// Model cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Default backend to use
    #[serde(default = "default_backend")]
    pub default: String,
    /// Fallback backend if default fails
    #[serde(default = "default_fallback_backend")]
    pub fallback: String,
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Auto-save sessions on exit
    #[serde(default = "default_auto_save")]
    pub auto_save: bool,
    /// History file path
    #[serde(default = "default_history_file")]
    pub history_file: String,
    /// Maximum number of history entries
    #[serde(default = "default_max_history")]
    pub max_history: usize,
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Enable colored output
    #[serde(default = "default_color")]
    pub color: bool,
    /// Enable streaming output
    #[serde(default = "default_stream")]
    pub stream: bool,
    /// Show performance metrics
    #[serde(default)]
    pub show_metrics: bool,
}

// Default functions for serde
fn default_cache_dir() -> String {
    "~/.rustorch/models".to_string()
}


fn default_backend() -> String {
    "cpu".to_string()
}

fn default_fallback_backend() -> String {
    "cpu".to_string()
}

fn default_auto_save() -> bool {
    true
}

fn default_history_file() -> String {
    "~/.rustorch/history".to_string()
}

fn default_max_history() -> usize {
    1000
}

fn default_color() -> bool {
    true
}

fn default_stream() -> bool {
    true
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default: None,
            cache_dir: default_cache_dir(),
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            default: default_backend(),
            fallback: default_fallback_backend(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            auto_save: default_auto_save(),
            history_file: default_history_file(),
            max_history: default_max_history(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            color: default_color(),
            stream: default_stream(),
            show_metrics: false,
        }
    }
}

impl Config {
    /// Load configuration from the default location (~/.rustorch/config.toml)
    pub fn load_default() -> Result<Self> {
        let config_path = Self::default_config_path()?;

        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            tracing::info!("No config file found, using defaults");
            Ok(Self::default())
        }
    }

    /// Load configuration from a specific file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        tracing::info!("Loaded configuration from {}", path.display());
        Ok(config)
    }

    /// Save configuration to the default location
    pub fn save_default(&self) -> Result<()> {
        let config_path = Self::default_config_path()?;
        self.save_to_file(&config_path)
    }

    /// Save configuration to a specific file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize configuration")?;

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        tracing::info!("Saved configuration to {}", path.display());
        Ok(())
    }

    /// Get the default config file path (~/.rustorch/config.toml)
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = directories::ProjectDirs::from("", "", "rustorch")
            .context("Failed to determine config directory")?
            .config_dir()
            .to_path_buf();

        Ok(config_dir.join("config.toml"))
    }

    /// Expand tilde (~) in path strings
    pub fn expand_path(path: &str) -> String {
        if path.starts_with('~') {
            if let Some(home) = directories::BaseDirs::new() {
                return path.replacen('~', home.home_dir().to_str().unwrap_or("~"), 1);
            }
        }
        path.to_string()
    }

    /// Merge with another config (self takes precedence)
    pub fn merge(&mut self, other: &Config) {
        // Model config
        if self.model.default.is_none() && other.model.default.is_some() {
            self.model.default.clone_from(&other.model.default);
        }

        // Generation config - only merge if using defaults
        let defaults = GenerationConfig::default();
        if self.generation.max_tokens == defaults.max_tokens {
            self.generation.max_tokens = other.generation.max_tokens;
        }
        if self.generation.temperature == defaults.temperature {
            self.generation.temperature = other.generation.temperature;
        }
        if self.generation.top_p == defaults.top_p {
            self.generation.top_p = other.generation.top_p;
        }
        if self.generation.top_k == defaults.top_k {
            self.generation.top_k = other.generation.top_k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.generation.max_tokens, 2048);
        assert_eq!(config.generation.temperature, 0.7);
        assert_eq!(config.backend.default, "cpu");
        assert!(config.session.auto_save);
        assert!(config.ui.color);
        assert!(config.ui.stream);
    }

    #[test]
    fn test_save_and_load() -> Result<()> {
        let mut config = Config::default();
        config.generation.max_tokens = 1024;
        config.generation.temperature = 0.8;
        config.backend.default = "metal".to_string();

        let temp_file = NamedTempFile::new()?;
        config.save_to_file(temp_file.path())?;

        let loaded = Config::load_from_file(temp_file.path())?;
        assert_eq!(loaded.generation.max_tokens, 1024);
        assert_eq!(loaded.generation.temperature, 0.8);
        assert_eq!(loaded.backend.default, "metal");

        Ok(())
    }

    #[test]
    fn test_expand_path() {
        let path = "~/.rustorch/models";
        let expanded = Config::expand_path(path);
        assert!(!expanded.starts_with('~'));

        let path = "/absolute/path";
        let expanded = Config::expand_path(path);
        assert_eq!(expanded, "/absolute/path");
    }

    #[test]
    fn test_toml_serialization() -> Result<()> {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config)?;

        // Should be able to deserialize back
        let _parsed: Config = toml::from_str(&toml_str)?;

        Ok(())
    }

    #[test]
    fn test_partial_config() -> Result<()> {
        let toml_str = r#"
            [model]
            default = "models/llama-7b.gguf"

            [generation]
            max_tokens = 256
        "#;

        let config: Config = toml::from_str(toml_str)?;
        assert_eq!(
            config.model.default,
            Some("models/llama-7b.gguf".to_string())
        );
        assert_eq!(config.generation.max_tokens, 256);
        // Other fields should use defaults
        assert_eq!(config.generation.temperature, 0.7);
        assert_eq!(config.backend.default, "cpu");

        Ok(())
    }

    #[test]
    fn test_merge_configs() {
        let mut config1 = Config::default();
        config1.generation.max_tokens = 256;

        let mut config2 = Config::default();
        config2.model.default = Some("model.gguf".to_string());
        config2.generation.max_tokens = 1024;

        config1.merge(&config2);

        // config1's max_tokens should be preserved (non-default)
        assert_eq!(config1.generation.max_tokens, 256);
        // config2's model should be merged (config1 had None)
        assert_eq!(config1.model.default, Some("model.gguf".to_string()));
    }
}
