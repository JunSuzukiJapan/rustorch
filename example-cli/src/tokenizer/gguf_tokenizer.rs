//! GGUF-embedded tokenizer support
//! GGUFファイルに埋め込まれたトークナイザーのサポート

use anyhow::Result;

/// Simple tokenizer using vocabulary extracted from GGUF
pub struct GGUFTokenizer {
    vocab: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
}

impl GGUFTokenizer {
    /// Create a new GGUF tokenizer from vocabulary
    pub fn new(vocab: Vec<String>) -> Self {
        let mut token_to_id = std::collections::HashMap::new();
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }
        
        Self { vocab, token_to_id }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl super::Tokenizer for GGUFTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Simple word-based tokenization for demonstration
        // In production, you would use BPE or SentencePiece
        let mut tokens = Vec::new();
        
        // Add BOS token (typically token 1 for Llama models)
        tokens.push(1);
        
        // Simple splitting by whitespace and special characters
        for word in text.split_whitespace() {
            // Try to find exact match first
            if let Some(&id) = self.token_to_id.get(word) {
                tokens.push(id);
            } else {
                // Fallback: character-level tokenization
                for ch in word.chars() {
                    let char_str = ch.to_string();
                    if let Some(&id) = self.token_to_id.get(&char_str) {
                        tokens.push(id);
                    } else {
                        // Unknown token (UNK)
                        tokens.push(0);
                    }
                }
            }
        }
        
        // Add EOS token (typically token 2 for Llama models)
        tokens.push(2);
        
        Ok(tokens)
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();
        
        for &id in ids {
            if id as usize >= self.vocab.len() {
                continue;
            }
            
            let token = &self.vocab[id as usize];
            
            // Skip special tokens (BOS, EOS, PAD)
            if id <= 2 {
                continue;
            }
            
            // Add space before token if needed
            if !result.is_empty() && !token.starts_with("▁") {
                result.push(' ');
            }
            
            // Remove SentencePiece underscore prefix if present
            let clean_token = if token.starts_with("▁") {
                &token[3..] // "▁" is 3 bytes in UTF-8
            } else {
                token
            };
            
            result.push_str(clean_token);
        }
        
        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(1) // Llama models use 1 for BOS
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(2) // Llama models use 2 for EOS
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(0) // Typically 0 for PAD
    }

    fn unk_token_id(&self) -> Option<u32> {
        Some(0) // Unknown token
    }
}
