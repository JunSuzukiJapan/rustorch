/// Decode specific token IDs to see what they represent
use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 Loading GGUF file...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    // Get tokenizer data from metadata
    println!("\n🔍 Extracting tokenizer data from GGUF metadata...");

    // Tokens RusTorch predicts (top 5 from test_generation_loop)
    let tokens_to_check = vec![1247, 13487, 6243, 4305, 24101];

    println!("\n🔍 Tokens RusTorch predicts (top 5 logits):");
    for token_id in &tokens_to_check {
        // For now, just show token ID - we'll need to implement vocab extraction
        println!("  Token {}: <need vocab extraction>", token_id);
    }

    // Note: GGUFLoader doesn't expose tokenizer vocab directly
    // We need to read it from the GGUF metadata "tokenizer.ggml.tokens"
    println!("\n💡 Note: Full token decoding requires reading tokenizer.ggml.tokens from GGUF metadata");
    println!("For now, let's check what llama.cpp says about these tokens:");
    println!("  Token 1247, 13487, 6243, 4305, 24101");

    Ok(())
}
