//! Tokenizer module for text encoding and decoding
//! テキストのエンコード・デコード用トークナイザーモジュール

#[cfg(feature = "tokenizer")]
pub mod hf_tokenizer;

#[cfg(feature = "tokenizer")]
pub use hf_tokenizer::HFTokenizer;

use crate::error::RusTorchResult;

/// Trait for tokenizers
/// トークナイザーのトレイト
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    /// テキストをトークンIDにエンコード
    fn encode(&self, text: &str, add_special_tokens: bool) -> RusTorchResult<Vec<u32>>;

    /// Decode token IDs to text
    /// トークンIDをテキストにデコード
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> RusTorchResult<String>;

    /// Get vocabulary size
    /// 語彙サイズを取得
    fn vocab_size(&self) -> usize;

    /// Get BOS (Beginning of Sequence) token ID
    /// BOS（文頭）トークンIDを取得
    fn bos_token_id(&self) -> Option<u32>;

    /// Get EOS (End of Sequence) token ID
    /// EOS（文末）トークンIDを取得
    fn eos_token_id(&self) -> Option<u32>;

    /// Get PAD (Padding) token ID
    /// PAD（パディング）トークンIDを取得
    fn pad_token_id(&self) -> Option<u32>;

    /// Get UNK (Unknown) token ID
    /// UNK（未知語）トークンIDを取得
    fn unk_token_id(&self) -> Option<u32>;
}
