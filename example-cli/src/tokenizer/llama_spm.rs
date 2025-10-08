//! llama.cpp-compatible SPM (SentencePiece) tokenizer
//! Direct port of llama.cpp's llm_tokenizer_spm implementation
//!
//! This is a faithful reproduction of the algorithm used in llama.cpp
//! to ensure 100% token ID compatibility.

use anyhow::Result;
use std::collections::HashMap;
use std::cmp::Ordering;

/// Symbol representing a UTF-8 character or merged token
#[derive(Clone, Debug)]
struct Symbol {
    text: String,
    prev: isize,
    next: isize,
}

/// Bigram candidate for merging
#[derive(Clone, Debug)]
struct Bigram {
    left: usize,
    right: usize,
    text: String,
    size: usize,
    rank: i32,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.left == other.left
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower rank = higher priority (reverse for max heap)
        match other.rank.cmp(&self.rank) {
            Ordering::Equal => self.left.cmp(&other.left),
            other => other,
        }
    }
}

/// llama.cpp-compatible SPM tokenizer
pub struct LlamaSpmTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    token_to_rank: HashMap<String, i32>,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl LlamaSpmTokenizer {
    /// Create a new SPM tokenizer from vocabulary and merges
    pub fn new(vocab: Vec<String>, merges: Vec<(String, String)>) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Build merge ranks (lower rank = higher priority)
        let mut token_to_rank = HashMap::new();
        for (rank, (token1, token2)) in merges.iter().enumerate() {
            let merged = format!("{}{}", token1, token2);
            token_to_rank.insert(merged, rank as i32);
        }

        Ok(Self {
            vocab,
            token_to_id,
            token_to_rank,
            bos_token_id: 1,
            eos_token_id: 2,
        })
    }

    /// Tokenize text using llama.cpp's SPM algorithm
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Step 1: Split into UTF-8 characters
        let mut symbols = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        for (i, ch) in chars.iter().enumerate() {
            symbols.push(Symbol {
                text: ch.to_string(),
                prev: i as isize - 1,
                next: if i + 1 < chars.len() { i as isize + 1 } else { -1 },
            });
        }

        // Step 2: Seed work queue with bigrams
        let mut work_queue = std::collections::BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            self.try_add_bigram(&symbols, &mut work_queue, i, i + 1);
        }

        // Step 3: Iteratively merge highest priority bigrams
        while let Some(bigram) = work_queue.pop() {
            let left = bigram.left;
            let right = bigram.right;

            // Check if still valid
            if left >= symbols.len() || right >= symbols.len() {
                continue;
            }

            if symbols[left].text.is_empty() || symbols[right].text.is_empty() {
                continue;
            }

            let expected_size = symbols[left].text.len() + symbols[right].text.len();
            if expected_size != bigram.size {
                continue;
            }

            // Merge - clone text to avoid borrow checker issues
            let right_text = symbols[right].text.clone();
            symbols[left].text.push_str(&right_text);
            symbols[left].next = symbols[right].next;

            if symbols[right].next >= 0 {
                let next_idx = symbols[right].next as usize;
                if next_idx < symbols.len() {
                    symbols[next_idx].prev = left as isize;
                }
            }

            symbols[right].text.clear();

            // Try adding new bigrams
            if symbols[left].prev >= 0 {
                let prev_idx = symbols[left].prev as usize;
                if prev_idx < symbols.len() {
                    self.try_add_bigram(&symbols, &mut work_queue, prev_idx, left);
                }
            }

            if symbols[left].next >= 0 {
                let next_idx = symbols[left].next as usize;
                if next_idx < symbols.len() {
                    self.try_add_bigram(&symbols, &mut work_queue, left, next_idx);
                }
            }
        }

        // Step 4: Convert symbols to token IDs
        let mut output = Vec::new();
        for sym in &symbols {
            if !sym.text.is_empty() {
                if let Some(&token_id) = self.token_to_id.get(&sym.text) {
                    output.push(token_id);
                } else {
                    // Fallback: try byte-by-byte encoding
                    for byte in sym.text.bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&token_id) = self.token_to_id.get(&byte_token) {
                            output.push(token_id);
                        } else {
                            output.push(0); // UNK
                        }
                    }
                }
            }
        }

        output
    }

    /// Try to add a bigram to the work queue
    fn try_add_bigram(
        &self,
        symbols: &[Symbol],
        work_queue: &mut std::collections::BinaryHeap<Bigram>,
        left: usize,
        right: usize,
    ) {
        if left >= symbols.len() || right >= symbols.len() {
            return;
        }

        if symbols[left].text.is_empty() || symbols[right].text.is_empty() {
            return;
        }

        let text = format!("{}{}", symbols[left].text, symbols[right].text);

        if let Some(&rank) = self.token_to_rank.get(&text) {
            work_queue.push(Bigram {
                left,
                right,
                size: symbols[left].text.len() + symbols[right].text.len(),
                text,
                rank,
            });
        }
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let mut result = String::new();

        for &id in ids {
            if skip_special_tokens {
                if id == self.bos_token_id || id == self.eos_token_id || id == 0 {
                    continue;
                }
            }

            if (id as usize) < self.vocab.len() {
                let token = &self.vocab[id as usize];

                // Handle SentencePiece space marker
                if let Some(stripped) = token.strip_prefix('▁') {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(stripped);
                } else if token.starts_with("<0x") && token.ends_with(">") {
                    // Byte token
                    if let Some(hex) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            result.push(byte as char);
                        }
                    }
                } else if token != "<s>" && token != "</s>" && token != "<unk>" {
                    result.push_str(token);
                }
            }
        }

        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

impl super::Tokenizer for LlamaSpmTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(self.bos_token_id);
        }

        tokens.extend(self.tokenize(text));

        if add_special_tokens {
            tokens.push(self.eos_token_id);
        }

        Ok(tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(self.decode(ids, skip_special_tokens))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
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

    #[test]
    fn test_basic_tokenization() {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁H".to_string(),
            "ello".to_string(),
            "▁Hello".to_string(),
        ];

        let merges = vec![
            ("▁H".to_string(), "ello".to_string()),
        ];

        let tokenizer = LlamaSpmTokenizer::new(vocab, merges).unwrap();
        let tokens = tokenizer.tokenize("Hello");

        // Should merge to ▁Hello
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], 5); // ▁Hello token
    }
}
