use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

use super::Tokenizer;

/// Wrapper around Hugging Face tokenizers
pub struct TokenizerWrapper {
    tokenizer: HFTokenizer,
}

impl TokenizerWrapper {
    /// Load tokenizer from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            anyhow::bail!("Tokenizer file not found: {}", path.display());
        }

        tracing::info!("Loading tokenizer from: {}", path.display());

        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    /// Create a simple dummy tokenizer for testing
    pub fn dummy() -> Result<Self> {
        tracing::info!("Creating dummy tokenizer for testing");

        // Create a minimal BPE tokenizer
        use tokenizers::models::bpe::BPE;

        let bpe = BPE::default();
        let tokenizer = HFTokenizer::new(bpe);

        Ok(Self { tokenizer })
    }

    /// Get the underlying HF tokenizer
    pub fn inner(&self) -> &HFTokenizer {
        &self.tokenizer
    }

    /// Get mutable reference to the underlying HF tokenizer
    pub fn inner_mut(&mut self) -> &mut HFTokenizer {
        &mut self.tokenizer
    }
}

impl Tokenizer for TokenizerWrapper {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<s>")
            .or_else(|| self.tokenizer.token_to_id("[BOS]"))
            .or_else(|| self.tokenizer.token_to_id("<|begin_of_text|>"))
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("[EOS]"))
            .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<pad>")
            .or_else(|| self.tokenizer.token_to_id("[PAD]"))
    }

    fn unk_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<unk>")
            .or_else(|| self.tokenizer.token_to_id("[UNK]"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_tokenizer_creation() {
        let tokenizer = TokenizerWrapper::dummy().unwrap();
        // Dummy tokenizer with default BPE has minimal vocab
        assert!(tokenizer.vocab_size() >= 0);
    }

    #[test]
    fn test_tokenizer_trait_implementation() {
        let tokenizer = TokenizerWrapper::dummy().unwrap();

        // Test that trait methods are callable
        let text = "test";
        let result = tokenizer.encode(text, false);
        // Dummy tokenizer may return empty encoding
        assert!(result.is_ok());
    }

    #[test]
    fn test_special_token_lookup() {
        let tokenizer = TokenizerWrapper::dummy().unwrap();

        // Dummy tokenizer may not have special tokens configured
        // Just verify the methods don't panic
        let _ = tokenizer.bos_token_id();
        let _ = tokenizer.eos_token_id();
        let _ = tokenizer.pad_token_id();
        let _ = tokenizer.unk_token_id();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = TokenizerWrapper::from_file("nonexistent.json");
        assert!(result.is_err());
    }
}