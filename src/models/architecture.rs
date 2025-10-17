/// Model architecture types supported by RusTorch
/// RusTorchがサポートするモデルアーキテクチャの種類

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// LLaMA family (Meta)
    LLaMA,
    /// Mistral family (Mistral AI)
    Mistral,
    /// Phi family (Microsoft)
    Phi,
    /// Gemma family (Google)
    Gemma,
    /// Qwen family (Alibaba)
    Qwen,
}

impl ModelArchitecture {
    /// Create from GGUF architecture string
    /// GGUF architectureメタデータ文字列から作成
    ///
    /// # Arguments
    /// * `arch_str` - Architecture string from GGUF metadata (e.g., "llama", "mistral")
    ///
    /// # Example
    /// ```
    /// use rustorch::models::ModelArchitecture;
    ///
    /// let arch = ModelArchitecture::from_gguf_string("llama");
    /// assert_eq!(arch, Some(ModelArchitecture::LLaMA));
    /// ```
    pub fn from_gguf_string(arch_str: &str) -> Option<Self> {
        match arch_str.to_lowercase().as_str() {
            "llama" => Some(ModelArchitecture::LLaMA),
            "mistral" => Some(ModelArchitecture::Mistral),
            "phi" | "phi2" | "phi3" => Some(ModelArchitecture::Phi),
            "gemma" | "gemma2" => Some(ModelArchitecture::Gemma),
            "qwen" | "qwen2" => Some(ModelArchitecture::Qwen),
            _ => None,
        }
    }

    /// Get architecture name as string
    /// アーキテクチャ名を文字列として取得
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelArchitecture::LLaMA => "llama",
            ModelArchitecture::Mistral => "mistral",
            ModelArchitecture::Phi => "phi",
            ModelArchitecture::Gemma => "gemma",
            ModelArchitecture::Qwen => "qwen",
        }
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_gguf_string() {
        assert_eq!(
            ModelArchitecture::from_gguf_string("llama"),
            Some(ModelArchitecture::LLaMA)
        );
        assert_eq!(
            ModelArchitecture::from_gguf_string("mistral"),
            Some(ModelArchitecture::Mistral)
        );
        assert_eq!(
            ModelArchitecture::from_gguf_string("phi2"),
            Some(ModelArchitecture::Phi)
        );
        assert_eq!(
            ModelArchitecture::from_gguf_string("unknown"),
            None
        );
    }

    #[test]
    fn test_as_str() {
        assert_eq!(ModelArchitecture::LLaMA.as_str(), "llama");
        assert_eq!(ModelArchitecture::Mistral.as_str(), "mistral");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ModelArchitecture::LLaMA), "llama");
        assert_eq!(format!("{}", ModelArchitecture::Mistral), "mistral");
    }
}
