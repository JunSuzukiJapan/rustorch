/// Dump Token 1 embedding vector from RusTorch
///
/// Compare with llama.cpp to verify weight extraction

use rustorch::formats::gguf::GGUFLoader;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Dumping Token 1 Embedding\n");

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            std::env::var("HOME").unwrap() +
            "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        });

    println!("📂 Model: {}", model_path.split('/').last().unwrap_or(&model_path));

    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| format!("GGUF load failed: {}", e))?;

    // First, load output.weight to check its type
    println!("🔍 Checking output.weight type first...");
    let _output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| format!("Output tensor load failed: {}", e))?;

    println!("\n🔍 Now loading token_embd.weight...");
    let embd_tensor = loader.load_tensor("token_embd.weight")
        .map_err(|e| format!("Tensor load failed: {}", e))?;

    println!("✅ token_embd.weight loaded");
    println!("   Shape: {:?}", embd_tensor.shape());

    let data: Vec<f64> = embd_tensor.data.iter().cloned().collect();

    // Token 1 (BOS) embedding
    // Shape is [vocab_size, hidden_size] = [32000, 2048]
    // Token 1 starts at index 1 * 2048 = 2048

    let token_id = 1;
    let hidden_size = 2048;
    let start_idx = token_id * hidden_size;
    let end_idx = start_idx + hidden_size;

    let token_1_embd = &data[start_idx..end_idx];

    println!("\n📝 Writing to /tmp/rustorch_token1_embedding.txt");
    let mut file = std::fs::File::create("/tmp/rustorch_token1_embedding.txt")?;
    for (i, val) in token_1_embd.iter().enumerate() {
        writeln!(file, "{} {:.10}", i, val)?;
    }

    println!("✅ Done! {} values written", token_1_embd.len());

    // Statistics
    let mean: f64 = token_1_embd.iter().sum::<f64>() / hidden_size as f64;
    let abs_mean: f64 = token_1_embd.iter().map(|x| x.abs()).sum::<f64>() / hidden_size as f64;
    let max = token_1_embd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = token_1_embd.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("\n📊 Statistics:");
    println!("   Mean: {:.10}", mean);
    println!("   Abs Mean: {:.10}", abs_mean);
    println!("   Max: {:.10}", max);
    println!("   Min: {:.10}", min);

    println!("\nFirst 10 values:");
    for i in 0..10 {
        println!("  [{}]: {:.10}", i, token_1_embd[i]);
    }

    Ok(())
}
