//! Test LlamaSpmTokenizer to identify tokenization issues

use rustorch::formats::gguf::GGUFLoader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("Loading GGUF model...");
    let gguf = GGUFLoader::from_file(model_path)?;

    let vocab = gguf.extract_tokenizer_vocab()?;
    let merges = gguf.extract_bpe_merges()?;
    println!("✅ Extracted vocab ({} tokens) and merges ({} rules)\n", vocab.len(), merges.len());

    // Import the tokenizer - need to make it public first
    // For now, let's just check the vocab structure

    // Check if "▁Hello" and "Hello" exist in vocab
    println!("🔍 Looking for Hello variants in vocab:");
    for variant in &["Hello", "▁Hello", "H", "▁H", "ello", "▁hell", "▁he"] {
        if let Some(id) = vocab.iter().position(|t| t == variant) {
            println!("  {:?} -> ID {}", variant, id);
        } else {
            println!("  {:?} -> NOT FOUND", variant);
        }
    }

    // Check for "What" variants
    println!("\n🔍 Looking for What variants in vocab:");
    for variant in &["What", "▁What", "W", "▁W", "hat"] {
        if let Some(id) = vocab.iter().position(|t| t == variant) {
            println!("  {:?} -> ID {}", variant, id);
        } else {
            println!("  {:?} -> NOT FOUND", variant);
        }
    }

    // Check first few merge rules to understand the pattern
    println!("\n🔄 First 20 merge rules:");
    for (i, (token1, token2)) in merges.iter().take(20).enumerate() {
        let merged = format!("{}{}", token1, token2);
        if let Some(id) = vocab.iter().position(|t| t == &merged) {
            println!("  [{}] {:?} + {:?} = {:?} (ID {})", i, token1, token2, merged, id);
        } else {
            println!("  [{}] {:?} + {:?} = {:?} (NOT IN VOCAB!)", i, token1, token2, merged);
        }
    }

    Ok(())
}
