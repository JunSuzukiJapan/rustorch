use serde::{Deserialize, Serialize};
use crate::CliArgs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = deterministic, higher = more random)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling threshold
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

fn default_max_tokens() -> usize {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_top_k() -> u32 {
    40
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        }
    }
}

impl GenerationConfig {
    /// Merge CLI arguments with file config
    /// CLI arguments take precedence over file config
    pub fn merge_from_cli(file_config: &Self, args: &CliArgs) -> Self {
        Self {
            max_tokens: Self::override_value(
                args.max_tokens,
                512, // CLI default from CliArgs
                file_config.max_tokens,
            ),
            temperature: Self::override_f32(
                args.temperature,
                0.7, // CLI default from CliArgs
                file_config.temperature,
            ),
            top_p: Self::override_f32(
                args.top_p,
                0.9, // CLI default from CliArgs
                file_config.top_p,
            ),
            top_k: Self::override_value(
                args.top_k,
                40, // CLI default from CliArgs
                file_config.top_k,
            ),
        }
    }

    /// Override a value: use CLI if different from default, otherwise use file config
    fn override_value<T: PartialEq + Copy>(cli_value: T, cli_default: T, file_value: T) -> T {
        if cli_value != cli_default {
            cli_value
        } else {
            file_value
        }
    }

    /// Override an f32 value with epsilon comparison
    fn override_f32(cli_value: f32, cli_default: f32, file_value: f32) -> f32 {
        if (cli_value - cli_default).abs() > f32::EPSILON {
            cli_value
        } else {
            file_value
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.temperature < 0.0 || self.temperature > 2.0 {
            anyhow::bail!("Temperature must be between 0.0 and 2.0");
        }

        if self.top_p < 0.0 || self.top_p > 1.0 {
            anyhow::bail!("Top-p must be between 0.0 and 1.0");
        }

        if self.max_tokens == 0 {
            anyhow::bail!("Max tokens must be greater than 0");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 40);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = GenerationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_temperature() {
        let mut config = GenerationConfig::default();
        config.temperature = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_top_p() {
        let mut config = GenerationConfig::default();
        config.top_p = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_max_tokens() {
        let mut config = GenerationConfig::default();
        config.max_tokens = 0;
        assert!(config.validate().is_err());
    }
}
