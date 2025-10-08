//! GGUF-embedded tokenizer using HuggingFace tokenizers crate
//! GGUFファイルに埋め込まれたトークナイザー（HuggingFace実装）

use anyhow::Result;
use tokenizers::{Tokenizer as HFTokenizer, AddedToken};
use std::collections::HashMap;

/// GGUF tokenizer using HuggingFace tokenizers crate with GGUF-extracted vocabulary
/// Properly implements SentencePiece-style BPE tokenization
pub struct GGUFTokenizer {
    tokenizer: HFTokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl GGUFTokenizer {
    /// Create a new GGUF tokenizer from vocabulary and merges
    /// Uses HuggingFace tokenizers crate for proper BPE implementation with SentencePiece compatibility
    pub fn from_gguf(vocab: Vec<String>, merges: Vec<(String, String)>) -> Result<Self> {
        use tokenizers::models::bpe::BPE;
        use tokenizers::{normalizers, decoders, NormalizerWrapper, DecoderWrapper};

        // Build vocabulary HashMap for BPE model
        let mut vocab_map = HashMap::new();
        for (id, token) in vocab.iter().enumerate() {
            vocab_map.insert(token.clone(), id as u32);
        }

        // Build merges in HuggingFace format: Vec<(String, String)>
        let merges_vec: Vec<(String, String)> = merges;

        // Create BPE model
        let bpe = BPE::builder()
            .vocab_and_merges(vocab_map.clone(), merges_vec)
            .unk_token("<unk>".to_string())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BPE model: {:?}", e))?;

        // Create tokenizer with BPE model
        let mut tokenizer = HFTokenizer::new(bpe);

        // Add SentencePiece-compatible normalizer
        // 1. Prepend ▁ to the beginning
        // 2. Replace spaces with ▁
        let normalizer = normalizers::Sequence::new(vec![
            NormalizerWrapper::Prepend(normalizers::Prepend::new("▁".to_string())),
            NormalizerWrapper::Replace(normalizers::Replace::new(" ".to_string(), "▁".to_string()).map_err(|e| anyhow::anyhow!("Failed to create Replace normalizer: {:?}", e))?),
        ]);
        tokenizer.with_normalizer(normalizer);

        // Add decoder to convert ▁ back to spaces
        use tokenizers::pre_tokenizers::metaspace::PrependScheme;
        tokenizer.with_decoder(DecoderWrapper::Metaspace(
            decoders::metaspace::Metaspace::new('▁', PrependScheme::Always, false)
        ));

        // Add special tokens
        let special_tokens = vec![
            AddedToken::from("<unk>", true),
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ];
        tokenizer.add_special_tokens(&special_tokens);

        Ok(Self {
            tokenizer,
            bos_token_id: 1,  // <s>
            eos_token_id: 2,  // </s>
        })
    }

    /// Create tokenizer from vocabulary only (no merges)
    /// Falls back to simple word-based tokenization
    pub fn new(vocab: Vec<String>) -> Self {
        use tokenizers::models::bpe::BPE;

        // Build vocabulary HashMap
        let mut vocab_map = HashMap::new();
        for (id, token) in vocab.iter().enumerate() {
            vocab_map.insert(token.clone(), id as u32);
        }

        // Create BPE model with empty merges
        let bpe = BPE::builder()
            .vocab_and_merges(vocab_map, Vec::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build basic BPE model");

        let mut tokenizer = HFTokenizer::new(bpe);

        // Add special tokens
        let special_tokens = vec![
            AddedToken::from("<unk>", true),
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ];
        tokenizer.add_special_tokens(&special_tokens);

        Self {
            tokenizer,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

impl super::Tokenizer for GGUFTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode token IDs: {:?}", e))?;

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(self.bos_token_id)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(0)
    }

    fn unk_token_id(&self) -> Option<u32> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Tokenizer;

    #[test]
    fn test_basic_tokenizer() {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁hello".to_string(),
            "▁world".to_string(),
        ];

        let tokenizer = GGUFTokenizer::new(vocab);
        assert_eq!(tokenizer.bos_token_id(), Some(1));
        assert_eq!(tokenizer.eos_token_id(), Some(2));
    }

    #[test]
    fn test_from_gguf_with_merges() {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "▁w".to_string(),
            "r".to_string(),
            "d".to_string(),
        ];

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("o".to_string(), "r".to_string()),
        ];

        let tokenizer = GGUFTokenizer::from_gguf(vocab, merges).unwrap();
        assert_eq!(tokenizer.bos_token_id(), Some(1));
        assert_eq!(tokenizer.eos_token_id(), Some(2));
    }
}
