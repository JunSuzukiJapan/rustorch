/// Test tokenization and compare with expected token IDs
use rustorch::formats::gguf::GGUFLoader;
use tokenizers::Tokenizer;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("HOME")? +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 Model: {}", model_path);

    // Extract vocabulary and merges from GGUF
    let loader = GGUFLoader::from_file(&model_path)?;
    let vocab = loader.extract_tokenizer_vocab()?;
    let merges = loader.extract_bpe_merges()?;

    println!("✅ Extracted {} tokens", vocab.len());
    println!("✅ Extracted {} BPE merge rules", merges.len());

    // Build HuggingFace BPE tokenizer
    use tokenizers::models::bpe::BPE;

    let mut vocab_map = HashMap::new();
    for (id, token) in vocab.iter().enumerate() {
        vocab_map.insert(token.clone(), id as u32);
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab_map, merges)
        .unk_token("<unk>".to_string())
        .build()?;

    let tokenizer = Tokenizer::new(bpe);

    // Test tokenization
    let test_text = "Hello, how are you?";
    println!("\n📝 Test text: \"{}\"", test_text);

    let encoding = tokenizer.encode(test_text, false)?;
    let token_ids = encoding.get_ids();

    println!("\n🔢 RusTorch token IDs:");
    println!("{:?}", token_ids);

    println!("\n📋 Token details:");
    for (i, &token_id) in token_ids.iter().enumerate() {
        if (token_id as usize) < vocab.len() {
            println!("  [{}] {} -> '{}'", i, token_id, vocab[token_id as usize]);
        }
    }

    // Expected IDs from llama.cpp
    let expected = vec![15043, 29892, 920, 526, 366, 29973];
    println!("\n✅ Expected token IDs (from llama.cpp):");
    println!("{:?}", expected);

    // Compare
    println!("\n🔍 Comparison:");
    let match_count = token_ids.iter().zip(&expected).filter(|(a, b)| a == b).count();
    println!("  Matches: {}/{}", match_count, expected.len());

    if token_ids == expected.as_slice() {
        println!("\n✅ PERFECT MATCH! Tokenizer is working correctly.");
    } else {
        println!("\n❌ MISMATCH! Token IDs don't match llama.cpp output.");
    }

    Ok(())
}
