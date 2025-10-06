use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing output.weight layout\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Get output.weight [2048, 32000]
    let output_weight = model.get_weight("output.weight").expect("output.weight not found");
    let weight_data = output_weight.as_slice();

    let hidden_size = 2048;
    let vocab_size = 32000;

    println!("📊 Output Weight Shape: {:?}", output_weight.shape());
    println!("   Expected: [2048, 32000] (row-major: 2048 rows, each 32000 elements)\n");

    // Test: Extract row 0 (first 10 elements of dimension 0 across all tokens)
    println!("🔹 Row 0 (dim 0 for all tokens, first 10):");
    let row_0: Vec<f32> = (0..10).map(|i| weight_data[0 * vocab_size + i]).collect();
    println!("   {:?}", row_0);

    // Test: Extract column for token 450
    println!("\n🔸 Column 450 (all dims for token 450, first 10):");
    let col_450: Vec<f32> = (0..10).map(|dim| weight_data[dim * vocab_size + 450]).collect();
    println!("   {:?}", col_450);

    // CRITICAL TEST: Manual dot product
    // Hidden state from debug: [1.1820991, 1.5812036, -0.38069266, 1.3746278, ...]
    let hidden = vec![1.1820991, 1.5812036, -0.38069266, 1.3746278];

    println!("\n🔬 Manual dot product test (first 4 dimensions only):");
    println!("   Hidden[0..4]: {:?}", hidden);

    // For token 450: sum of hidden[i] * weight[i * vocab_size + 450]
    let logit_450_partial: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 450])
        .sum();
    println!("   Partial logit for token 450 (4 dims): {:.6}", logit_450_partial);

    // For token 20780
    let logit_20780_partial: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 20780])
        .sum();
    println!("   Partial logit for token 20780 (4 dims): {:.6}", logit_20780_partial);

    // Now test if maybe the weight matrix is TRANSPOSED
    println!("\n⚠️  Testing TRANSPOSED layout hypothesis:");
    println!("   If weight is actually [32000, 2048] stored as rows...\n");

    // Transpose hypothesis: weight[token_id * hidden_size + dim]
    let logit_450_trans_partial: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[450 * hidden_size + i])
        .sum();
    println!("   TRANSPOSED partial logit for token 450 (4 dims): {:.6}", logit_450_trans_partial);

    let logit_20780_trans_partial: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[20780 * hidden_size + i])
        .sum();
    println!("   TRANSPOSED partial logit for token 20780 (4 dims): {:.6}", logit_20780_trans_partial);

    println!("\n❓ Which layout gives sensible results?");

    Ok(())
}
