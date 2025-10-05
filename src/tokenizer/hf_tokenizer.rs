//! Hugging Face tokenizer wrapper
//! Hugging Faceトークナイザーラッパー

use super::Tokenizer;
use crate::error::{RusTorchError, RusTorchResult};
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizerImpl;

/// Wrapper around Hugging Face tokenizers
/// Hugging Faceトークナイザーのラッパー
pub struct HFTokenizer {
    tokenizer: HFTokenizerImpl,
}

impl HFTokenizer {
    /// Load tokenizer from file
    /// ファイルからトークナイザーを読み込み
    pub fn from_file<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(RusTorchError::IoError(
                format!("Tokenizer file not found: {}", path.display()),
            ));
        }

        let tokenizer = HFTokenizerImpl::from_file(path)
            .map_err(|e| RusTorchError::ConfigError(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { tokenizer })
    }

    /// Create a simple BPE tokenizer for testing
    /// テスト用の簡易BPEトークナイザーを作成
    pub fn dummy() -> RusTorchResult<Self> {
        use tokenizers::models::bpe::BPE;

        let bpe = BPE::default();
        let tokenizer = HFTokenizerImpl::new(bpe);

        Ok(Self { tokenizer })
    }

    /// Get the underlying HF tokenizer
    /// 内部のHFトークナイザーを取得
    pub fn inner(&self) -> &HFTokenizerImpl {
        &self.tokenizer
    }

    /// Get mutable reference to the underlying HF tokenizer
    /// 内部のHFトークナイザーへの可変参照を取得
    pub fn inner_mut(&mut self) -> &mut HFTokenizerImpl {
        &mut self.tokenizer
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> RusTorchResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| RusTorchError::ConfigError(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> RusTorchResult<String> {
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| RusTorchError::ConfigError(format!("Decoding failed: {}", e)))?;

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
        let _tokenizer = HFTokenizer::dummy().unwrap();
    }

    #[test]
    fn test_tokenizer_trait_implementation() {
        let tokenizer = HFTokenizer::dummy().unwrap();

        let text = "test";
        let result = tokenizer.encode(text, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_special_token_lookup() {
        let tokenizer = HFTokenizer::dummy().unwrap();

        // Dummy tokenizer may not have special tokens configured
        let _ = tokenizer.bos_token_id();
        let _ = tokenizer.eos_token_id();
        let _ = tokenizer.pad_token_id();
        let _ = tokenizer.unk_token_id();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = HFTokenizer::from_file("nonexistent.json");
        assert!(result.is_err());
    }
}
