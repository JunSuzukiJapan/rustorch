//! Test the fixed LlamaSpmTokenizer

use rustorch::formats::gguf::GGUFLoader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("Loading GGUF model...");
    let gguf = GGUFLoader::from_file(model_path)?;

    let vocab = gguf.extract_tokenizer_vocab()?;
    let merges = gguf.extract_bpe_merges()?;
    println!("✅ Loaded vocab ({} tokens) and merges ({} rules)\n", vocab.len(), merges.len());

    // Import tokenizer
    use example_cli::tokenizer::llama_spm::LlamaSpmTokenizer;

    let tokenizer = LlamaSpmTokenizer::new(vocab.clone(), merges)?;

    // Test cases
    let test_cases = vec![
        ("Hello", "Should tokenize to ▁Hello"),
        ("What", "Should tokenize to ▁What"),
        ("Hello world", "Should have two tokens with ▁ prefix"),
    ];

    for (input, desc) in test_cases {
        println!("🔍 Test: {}", desc);
        println!("   Input: {:?}", input);
        
        let tokens = tokenizer.tokenize(input);
        println!("   Token IDs: {:?}", tokens);
        
        for &token_id in &tokens {
            if let Some(token) = vocab.get(token_id as usize) {
                println!("     {} -> {:?}", token_id, token);
            }
        }

        let decoded = tokenizer.decode(&tokens, false);
        println!("   Decoded: {:?}", decoded);
        println!();
    }

    Ok(())
}
