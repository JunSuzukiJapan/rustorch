//! Compare Layer 0 outputs between llama.cpp and RusTorch to isolate where divergence begins

use rustorch::formats::gguf::GGUFLoader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("Loading GGUF model...");
    let gguf = GGUFLoader::from_file(model_path)?;

    let vocab = gguf.extract_tokenizer_vocab()?;
    let merges = gguf.extract_bpe_merges()?;
    
    println!("✅ Extracted vocab ({} tokens) and merges ({} rules)\n", vocab.len(), merges.len());

    // Import the actual tokenizers used by CLI
    // We need to test them directly to see which one works

    // Test simple input: "Hello"
    let test_inputs = vec![
        "Hello",
        "What",
        "What is",
    ];

    for input in test_inputs {
        println!("🔍 Test input: {:?}", input);
        
        // Expected from llama.cpp for "Hello": Should start with BOS (1), then actual tokens
        // We need to find what "Hello" tokenizes to
        
        // Manually check vocab for "Hello" variants
        let candidates = vec!["Hello", "▁Hello", "H", "▁H", "e", "l", "o"];
        println!("  Checking candidates in vocab:");
        for cand in &candidates {
            if let Some(id) = vocab.iter().position(|t| t == cand) {
                println!("    {:?} -> ID {}", cand, id);
            }
        }
        println!();
    }

    Ok(())
}
