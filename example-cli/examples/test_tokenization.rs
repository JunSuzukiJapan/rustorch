use rustorch_cli::tokenizer::{LlamaSpmTokenizer, Tokenizer};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Load TinyLlama model
    let model_path = PathBuf::from(
        std::env::var("HOME")?
    ).join(".rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    println!("Loading model: {}", model_path.display());

    let gguf = rustorch::formats::gguf::GGUFLoader::from_file(&model_path)?;

    // Extract vocabulary
    let vocab_raw = gguf.get_string_array("tokenizer.ggml.tokens")?;
    let tokens: Vec<String> = vocab_raw.iter().map(|s| s.to_string()).collect();

    // Extract merge rules
    let merges_raw = gguf.get_string_array("tokenizer.ggml.merges")?;
    let merge_rules: Vec<String> = merges_raw.iter().map(|s| s.to_string()).collect();

    println!("Loaded {} tokens and {} merge rules", tokens.len(), merge_rules.len());

    let tokenizer = LlamaSpmTokenizer::new(tokens, merge_rules);

    // Test cases
    let test_cases = vec![
        ("1", "Should tokenize '1' without prefix (token 29896)"),
        (" 1", "Should tokenize ' 1' with space prefix"),
        ("Hello", "Should tokenize 'Hello'"),
        ("Hello world", "Should tokenize 'Hello world'"),
    ];

    for (input, description) in test_cases {
        println!("\n=== {} ===", description);
        println!("Input: {:?}", input);

        let token_ids = tokenizer.encode(input, false)?;
        println!("Tokens: {:?}", token_ids);

        let decoded = tokenizer.decode(&token_ids, false)?;
        println!("Decoded: {:?}", decoded);
    }

    Ok(())
}
