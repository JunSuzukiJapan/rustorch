pub mod wrapper;
pub mod gguf_tokenizer;
pub mod llama_spm;

use anyhow::Result;
pub use wrapper::TokenizerWrapper;
pub use gguf_tokenizer::GGUFTokenizer;
pub use llama_spm::LlamaSpmTokenizer;

/// Tokenizer trait for encoding and decoding text
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special token IDs
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
    fn unk_token_id(&self) -> Option<u32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_trait_exists() {
        // Trait compilation test
        fn _assert_tokenizer<T: Tokenizer>() {}
    }
}
