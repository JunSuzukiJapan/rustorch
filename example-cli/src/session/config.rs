use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,

    /// Top-k sampling threshold
    pub top_k: u32,
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
