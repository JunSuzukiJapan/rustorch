use rustorch::formats::gguf::GGUFLoader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("Loading GGUF model: {}", model_path.display());
    let gguf = GGUFLoader::from_file(model_path)?;

    // Extract vocab and merges
    let vocab = gguf.extract_tokenizer_vocab()?;
    println!("✅ Extracted {} tokens from GGUF", vocab.len());

    let merges = gguf.extract_bpe_merges()?;
    println!("✅ Extracted {} BPE merge rules", merges.len());

    // Print first 10 vocab entries
    println!("\n📝 First 10 tokens:");
    for (i, token) in vocab.iter().take(10).enumerate() {
        println!("  [{}] {:?}", i, token);
    }

    // Print first 10 merges
    println!("\n🔄 First 10 merge rules:");
    for (i, (token1, token2)) in merges.iter().take(10).enumerate() {
        println!("  [{}] {:?} + {:?}", i, token1, token2);
    }

    // Test specific tokens
    println!("\n🔍 Looking for specific tokens:");
    for test_token in &["Hello", "▁Hello", "▁What", "▁is", "▁the", "▁capital"] {
        if let Some(id) = vocab.iter().position(|t| t == test_token) {
            println!("  {:?} -> ID {}", test_token, id);
        } else {
            println!("  {:?} -> NOT FOUND", test_token);
        }
    }

    Ok(())
}
