/// Test if output.weight layout is correct by checking specific known values
///
/// Strategy: Extract a specific weight column and compare with llama.cpp

use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Weight Layout Verification\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ output.weight loaded");
    println!("   Shape: {:?}", output_tensor.shape());

    let data: Vec<f64> = output_tensor.data.iter().cloned().collect();
    let hidden_size = 2048;
    let vocab_size = 32000;

    println!("\n📊 Testing two possible layouts:\n");

    // Layout 1: Row-major [hidden_size, vocab_size]
    // Access: data[h * vocab_size + v]
    println!("Layout 1 (Row-major [2048, 32000]):");
    println!("  Token 450, dim 0: {:.6}", data[0 * vocab_size + 450]);
    println!("  Token 450, dim 1: {:.6}", data[1 * vocab_size + 450]);
    println!("  Token 450, dim 2: {:.6}", data[2 * vocab_size + 450]);
    println!();

    // Layout 2: Column-major [vocab_size, hidden_size]
    // Access: data[v * hidden_size + h]
    println!("Layout 2 (Column-major or transposed [32000, 2048]):");
    println!("  Token 450, dim 0: {:.6}", data[450 * hidden_size + 0]);
    println!("  Token 450, dim 1: {:.6}", data[450 * hidden_size + 1]);
    println!("  Token 450, dim 2: {:.6}", data[450 * hidden_size + 2]);
    println!();

    println!("🔍 Manual logit calculation test:");
    println!("  Using a simple hidden state: [1, 0, 0, ..., 0]");
    println!("  This should give logit[v] = weight[0, v]");
    println!();

    // Create simple hidden state
    let mut hidden = vec![0.0f32; hidden_size];
    hidden[0] = 1.0;  // Only first dimension is 1

    // Calculate logits with Layout 1 (current implementation)
    let mut logits_layout1 = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let mut sum = 0.0f32;
        for h in 0..hidden_size {
            let idx = h * vocab_size + v;
            sum += hidden[h] * data[idx] as f32;
        }
        logits_layout1[v] = sum;
    }

    println!("Layout 1 results:");
    println!("  Logit[450]: {:.6}", logits_layout1[450]);
    println!("  Logit[1247]: {:.6}", logits_layout1[1247]);
    println!("  Logit[12711]: {:.6}", logits_layout1[12711]);
    println!();

    // Calculate logits with Layout 2
    let mut logits_layout2 = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let mut sum = 0.0f32;
        for h in 0..hidden_size {
            let idx = v * hidden_size + h;
            sum += hidden[h] * data[idx] as f32;
        }
        logits_layout2[v] = sum;
    }

    println!("Layout 2 results:");
    println!("  Logit[450]: {:.6}", logits_layout2[450]);
    println!("  Logit[1247]: {:.6}", logits_layout2[1247]);
    println!("  Logit[12711]: {:.6}", logits_layout2[12711]);
    println!();

    println!("Expected from llama.cpp (for reference):");
    println!("  Logit[450]: 10.154105");
    println!("  Logit[1247]: -5.834903");
    println!("  Logit[12711]: -0.640889");
    println!();

    println!("💡 Note: With hidden=[1,0,0,...], logit[v] equals weight[0,v]");
    println!("   So the first values printed above should match these expected values.");

    Ok(())
}
