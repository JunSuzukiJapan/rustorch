/// Compare tokenization output with llama.cpp
/// Dumps token IDs for manual verification
use rustorch::formats::gguf::GGUFLoader;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            std::env::var("HOME").unwrap() +
            "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        });

    println!("📂 Model: {}", model_path);

    let loader = GGUFLoader::from_file(&model_path)?;
    let vocab = loader.extract_tokenizer_vocab()?;

    println!("✅ Extracted {} tokens", vocab.len());

    // Build token map
    let mut token_to_id = HashMap::new();
    for (id, token) in vocab.iter().enumerate() {
        token_to_id.insert(token.as_str(), id as u32);
    }

    // Test prompt
    let text = "What is the capital of France?";
    println!("\n📝 Text: \"{}\"", text);

    // Simple tokenization (longest-match without BPE)
    let mut tokens = Vec::new();
    tokens.push(1); // BOS

    // Split by spaces and tokenize each word
    for word in text.split_whitespace() {
        let word_with_space = format!("▁{}", word);

        // Try to find token for word
        if let Some(&id) = token_to_id.get(word_with_space.as_str()) {
            tokens.push(id);
        } else {
            // Try character by character with space prefix
            for (i, ch) in word.chars().enumerate() {
                let ch_str = if i == 0 {
                    format!("▁{}", ch)
                } else {
                    ch.to_string()
                };

                if let Some(&id) = token_to_id.get(ch_str.as_str()) {
                    tokens.push(id);
                } else {
                    // Fallback to UNK
                    tokens.push(0);
                }
            }
        }
    }

    println!("\n🔢 Token IDs (simple longest-match):");
    println!("{:?}", tokens);

    println!("\n📋 Token details:");
    for (i, &token_id) in tokens.iter().enumerate() {
        if token_id < vocab.len() as u32 {
            println!("  [{}] {} -> '{}'", i, token_id, vocab[token_id as usize]);
        }
    }

    println!("\n💡 For llama.cpp comparison, run:");
    println!("  echo \"{}\" | /opt/homebrew/bin/llama-tokenize -m {}", text, model_path);

    Ok(())
}
