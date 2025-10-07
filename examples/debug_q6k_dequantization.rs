/// Debug Q6_K dequantization to see if scale factors are correct
///
/// Compare first few weights with llama.cpp expected values

use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Q6_K Dequantization Debug\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    // Load both Q6_K (output.weight) and Q4_K (token_embd.weight)
    println!("📂 Loading output.weight (Q6_K)...");
    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ output.weight loaded");
    println!("   Shape: {:?}", output_tensor.shape());

    let data: Vec<f64> = output_tensor.data.iter().cloned().collect();

    println!("\n📊 First 20 weights (should start near Token 0, dim 0):");
    for i in 0..20 {
        println!("  [{:2}] = {:.6}", i, data[i]);
    }

    println!("\n🔍 Checking specific positions:");
    let vocab_size = 32000;

    // Token 0, first few dimensions
    println!("\nToken 0 (row-major [hidden, vocab]):");
    for h in 0..5 {
        let idx = h * vocab_size + 0;
        println!("  dim {}: {:.6}", h, data[idx]);
    }

    // Token 450, first few dimensions
    println!("\nToken 450 (row-major [hidden, vocab]):");
    for h in 0..5 {
        let idx = h * vocab_size + 450;
        println!("  dim {}: {:.6}", h, data[idx]);
    }

    println!("\n💡 Expected from llama.cpp (token 450, dim 0): ~10.154");
    println!("   RusTorch actual (token 450, dim 0): {:.6}", data[0 * vocab_size + 450]);

    // Check if values are systematically scaled
    let expected_450_dim0 = 10.154;
    let actual_450_dim0 = data[0 * vocab_size + 450];
    let scale_factor = expected_450_dim0 / actual_450_dim0;

    println!("\n📏 Implied scale factor: {:.2}x", scale_factor);

    if scale_factor > 100.0 {
        println!("❌ Scale factor >100x suggests missing scale multiplication!");
    } else if scale_factor > 10.0 {
        println!("⚠️  Scale factor >10x suggests partial scale issue");
    } else {
        println!("✅ Scale factor reasonable");
    }

    // Also check token_embd.weight (Q4_K) for comparison
    println!("\n📂 Loading token_embd.weight (Q4_K) for comparison...");
    let embd_tensor = loader.load_tensor("token_embd.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ token_embd.weight loaded");
    println!("   Shape: {:?}", embd_tensor.shape());

    let embd_data: Vec<f64> = embd_tensor.data.iter().cloned().collect();

    println!("\n📊 token_embd first 10 values:");
    for i in 0..10 {
        println!("  [{:2}] = {:.6}", i, embd_data[i]);
    }

    // Check magnitude
    let embd_mean: f64 = embd_data.iter().take(1000).map(|x| x.abs()).sum::<f64>() / 1000.0;
    let output_mean: f64 = data.iter().take(1000).map(|x| x.abs()).sum::<f64>() / 1000.0;

    println!("\n📏 Magnitude comparison (first 1000 elements):");
    println!("   token_embd (Q4_K) mean abs: {:.6}", embd_mean);
    println!("   output (Q6_K) mean abs: {:.6}", output_mean);
    println!("   Ratio (output/embd): {:.2}x", output_mean / embd_mean);

    if (output_mean / embd_mean - 1.0).abs() > 10.0 {
        println!("\n❌ Q6_K and Q4_K have very different magnitudes!");
        println!("   This suggests quantization-specific dequantization errors.");
    }

    Ok(())
}
