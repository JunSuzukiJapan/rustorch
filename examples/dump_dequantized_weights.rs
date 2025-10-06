/// Dump dequantized weight values for comparison with llama.cpp
///
/// This allows us to verify if dequantization produces correct values
/// independent of the forward pass.

use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::{F32Result, F32Error};

fn main() -> F32Result<()> {
    println!("🔍 Dumping Dequantized Weights\n");

    // Load Q4_K_M model
    let q4km_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 Loading GGUF file: Q4_K_M");
    let loader = GGUFLoader::from_file(&q4km_path)
        .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

    // Get token embedding weight (should be column-major: [2048, 32000])
    let embd_tensor = loader.load_tensor("token_embd.weight")
        .map_err(|e| F32Error::device_error(format!("Failed to load tensor: {}", e)))?;

    println!("✅ Loaded token_embd.weight");
    println!("   Shape: {:?}", embd_tensor.shape());
    println!("   Expected: [2048, 32000]");

    // Access the raw data vector
    let embd_data_vec: Vec<f64> = embd_tensor.data.iter().cloned().collect();

    // Dump first few values of BOS token embedding (token ID 1)
    // Column-major: embedding[dim] = data[dim * vocab_size + token_id]
    println!("\n📊 BOS Token (ID=1) Embedding (first 10 dims):");
    for dim in 0..10 {
        let idx = dim * 32000 + 1;
        if idx < embd_data_vec.len() {
            println!("   dim[{}]: {:.8}", dim, embd_data_vec[idx]);
        }
    }

    // Get Layer 0 attention Q weight
    let q_tensor = loader.load_tensor("blk.0.attn_q.weight")
        .map_err(|e| F32Error::device_error(format!("Failed to load tensor: {}", e)))?;
    let q_data_vec: Vec<f64> = q_tensor.data.iter().cloned().collect();

    println!("\n📊 Layer 0 Q Weight (first 10 values):");
    for i in 0..10 {
        println!("   [{}]: {:.8}", i, q_data_vec[i]);
    }

    // Get Layer 0 FFN gate weight (should be: [5632, 2048])
    let gate_tensor = loader.load_tensor("blk.0.ffn_gate.weight")
        .map_err(|e| F32Error::device_error(format!("Failed to load tensor: {}", e)))?;
    let gate_data_vec: Vec<f64> = gate_tensor.data.iter().cloned().collect();

    println!("\n📊 Layer 0 FFN Gate Weight (first 10 values):");
    for i in 0..10 {
        println!("   [{}]: {:.8}", i, gate_data_vec[i]);
    }

    // Test Q4_0 as well
    let q40_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

    println!("\n═══════════════════════════════════════");
    println!("📂 Loading GGUF file: Q4_0");
    let loader_0 = GGUFLoader::from_file(&q40_path)
        .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

    let embd_tensor_0 = loader_0.load_tensor("token_embd.weight")
        .map_err(|e| F32Error::device_error(format!("Failed to load tensor: {}", e)))?;
    let embd_data_vec_0: Vec<f64> = embd_tensor_0.data.iter().cloned().collect();

    println!("\n📊 Q4_0 - BOS Token (ID=1) Embedding (first 10 dims):");
    for dim in 0..10 {
        let idx = dim * 32000 + 1;
        println!("   dim[{}]: {:.8}", dim, embd_data_vec_0[idx]);
    }

    // Compare the two
    println!("\n═══════════════════════════════════════");
    println!("🔍 Comparison: Q4_K_M vs Q4_0");
    println!("═══════════════════════════════════════");

    let mut max_diff = 0.0f64;
    let mut total_diff = 0.0f64;

    for dim in 0..2048 {
        let idx = dim * 32000 + 1;
        let diff = (embd_data_vec[idx] - embd_data_vec_0[idx]).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
    }

    let avg_diff = total_diff / 2048.0;

    println!("\nBOS Token Embedding Differences:");
    println!("   Max difference: {:.8}", max_diff);
    println!("   Avg difference: {:.8}", avg_diff);
    println!();

    if max_diff > 0.1 {
        println!("⚠️  LARGE DIFFERENCE detected!");
        println!("   This suggests quantization significantly affects weights");
    } else if max_diff > 0.01 {
        println!("✅ Moderate differences (expected for different quantization)");
    } else {
        println!("✅ Very small differences");
    }

    println!("\n💡 Next Steps:");
    println!("   1. Compare these values with llama.cpp's weight dump");
    println!("   2. Verify dequantization formulas match exactly");
    println!("   3. Check if weight differences explain output differences");

    Ok(())
}
