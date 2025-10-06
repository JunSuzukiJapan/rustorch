use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Comparing CPU vs Metal for LM head matmul\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Create the exact hidden state from debug output
    // We'll use a simplified version - just use first 10 values
    let hidden_vals = vec![
        1.1820991, 1.5812036, -0.38069266, 1.3746278, -0.24576735,
        -1.3941354, -2.4821181, 1.7011507, 0.54846644, -0.20032796
    ];

    // Pad to 2048 with zeros for simplicity
    let mut full_hidden = hidden_vals.clone();
    full_hidden.resize(2048, 0.0);

    let hidden = F32Tensor::from_vec(full_hidden.clone(), &[1, 2048])?;
    let lm_weight = model.get_weight("output.weight").expect("output.weight not found");

    println!("🔹 Hidden state (first 10 of 2048): {:?}", &full_hidden[0..10]);
    println!("🔹 LM weight shape: {:?}\n", lm_weight.shape());

    // Perform matmul
    println!("🧮 Performing matmul [1, 2048] @ [2048, 32000]...");
    let logits = hidden.matmul(lm_weight)?;
    let logits_data = logits.as_slice();

    println!("   Result shape: {:?}", logits.shape());
    println!("   Logits[0..10]: {:?}", &logits_data[0..10]);
    println!("   Logits[450]: {:.6}", logits_data[450]);
    println!("   Logits[20780]: {:.6}", logits_data[20780]);

    // Manual calculation for verification
    let weight_data = lm_weight.as_slice();
    let vocab_size = 32000;

    let manual_450: f32 = full_hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 450])
        .sum();

    let manual_20780: f32 = full_hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 20780])
        .sum();

    println!("\n🔢 Manual calculation:");
    println!("   Token 450: {:.6}", manual_450);
    println!("   Token 20780: {:.6}", manual_20780);

    println!("\n❓ Do they match?");
    println!("   450 match: {}", (logits_data[450] - manual_450).abs() < 0.001);
    println!("   20780 match: {}", (logits_data[20780] - manual_20780).abs() < 0.001);

    Ok(())
}
