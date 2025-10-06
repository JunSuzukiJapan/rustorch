use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing final LM head matmul\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Run forward pass to get the actual hidden state after all layers
    println!("🔄 Running forward pass with BOS token...");
    let _logits = model.forward(&[1])?;  // BOS token

    // Get the last hidden state that the model computed
    // We need to extract it from the model's internal state
    // For now, let's run the forward again and capture intermediate values

    println!("\n🧪 Testing matmul with simple vector:");

    // Create a simple test vector [1, 2048] with known values
    let test_hidden = F32Tensor::from_vec(vec![1.0; 2048], &[1, 2048])?;

    // Get output.weight
    let lm_head = model.get_weight("output.weight").expect("output.weight not found");

    println!("  test_hidden shape: {:?}", test_hidden.shape());
    println!("  lm_head shape: {:?}", lm_head.shape());

    // Matmul: [1, 2048] @ [2048, 32000] = [1, 32000]
    let logits = test_hidden.matmul(lm_head)?;
    let logits_data = logits.as_slice();

    println!("  logits shape: {:?}", logits.shape());
    println!("  logits[0..10]: {:?}", &logits_data[0..10]);
    println!("  logits[450]: {:.6}", logits_data[450]);
    println!("  logits[20780]: {:.6}", logits_data[20780]);

    // Manual calculation: sum of lm_head column
    let lm_data = lm_head.as_slice();
    let hidden_size = 2048;
    let vocab_size = 32000;

    let manual_450: f32 = (0..hidden_size)
        .map(|dim| lm_data[dim * vocab_size + 450])
        .sum();
    let manual_20780: f32 = (0..hidden_size)
        .map(|dim| lm_data[dim * vocab_size + 20780])
        .sum();

    println!("\n🔢 Manual calculation (sum of columns):");
    println!("  Token 450: {:.6}", manual_450);
    println!("  Token 20780: {:.6}", manual_20780);

    println!("\n✅ If matmul is correct, these should match!");
    println!("  Match 450: {}", (logits_data[450] - manual_450).abs() < 0.001);
    println!("  Match 20780: {}", (logits_data[20780] - manual_20780).abs() < 0.001);

    Ok(())
}
