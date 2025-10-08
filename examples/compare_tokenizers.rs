use rustorch::formats::gguf::GGUFLoader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("Loading GGUF model...");
    let gguf = GGUFLoader::from_file(model_path)?;

    let vocab = gguf.extract_tokenizer_vocab()?;
    let merges = gguf.extract_bpe_merges()?;
    println!("✅ Extracted vocab ({} tokens) and merges ({} rules)", vocab.len(), merges.len());

    // Test text from IMPLEMENTATION_VERIFICATION.md
    let test_text = "What is the capital of France?";
    
    // Expected from llama.cpp (from docs):
    // [1, 529, 29989, 1792, 29989, 29958, 13, 5618, 338, 278, 7483, 310, 3444, 29973, ...]
    let expected_tokens = vec![5618, 338, 278, 7483, 310, 3444, 29973];
    
    println!("\n📝 Test text: {:?}", test_text);
    println!("Expected tokens (llama.cpp): {:?}", expected_tokens);
    
    // Decode expected tokens to verify vocab
    println!("\n🔍 Decoding expected tokens:");
    for (i, &token_id) in expected_tokens.iter().enumerate() {
        if let Some(token) = vocab.get(token_id as usize) {
            println!("  Token {} (ID {}): {:?}", i, token_id, token);
        } else {
            println!("  Token {} (ID {}): OUT OF RANGE", i, token_id);
        }
    }

    Ok(())
}
