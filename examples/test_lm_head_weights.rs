use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Verifying LM head weight extraction\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Get output.weight [2048, 32000]
    let lm_head = model.get_weight("output.weight").expect("output.weight not found");
    let lm_data = lm_head.as_slice();

    println!("📊 LM Head Weight Info:");
    println!("  Shape: {:?}", lm_head.shape());
    println!("  Total elements: {}", lm_data.len());
    println!("  Expected: 2048 * 32000 = {}", 2048 * 32000);

    // Test: Extract column for token 450 (" The")
    let hidden_size = 2048;
    let vocab_size = 32000;
    let token_450_col: Vec<f32> = (0..hidden_size)
        .map(|dim| lm_data[dim * vocab_size + 450])
        .collect();

    println!("\n🔹 Token 450 column (should give high logit after BOS):");
    println!("  First 10 values: {:?}", &token_450_col[0..10]);
    println!("  Sum: {:.6}", token_450_col.iter().sum::<f32>());
    println!("  Non-zero: {}/{}", token_450_col.iter().filter(|&&x| x != 0.0).count(), hidden_size);

    // Test: Extract column for token 20780 (our wrong prediction)
    let token_20780_col: Vec<f32> = (0..hidden_size)
        .map(|dim| lm_data[dim * vocab_size + 20780])
        .collect();

    println!("\n🔸 Token 20780 column (our wrong top prediction):");
    println!("  First 10 values: {:?}", &token_20780_col[0..10]);
    println!("  Sum: {:.6}", token_20780_col.iter().sum::<f32>());
    println!("  Non-zero: {}/{}", token_20780_col.iter().filter(|&&x| x != 0.0).count(), hidden_size);

    // Compute dot product manually with BOS embedding
    let token_embd = model.get_weight("token_embd.weight").expect("token_embd not found");
    let embd_data = token_embd.as_slice();

    // Extract BOS embedding (token 1)
    let bos_emb: Vec<f32> = (0..hidden_size)
        .map(|dim| embd_data[dim * vocab_size + 1])
        .collect();

    println!("\n🔢 Manual logit calculation:");
    println!("  BOS embedding[0..10]: {:?}", &bos_emb[0..10]);

    // Dot product for token 450
    let logit_450: f32 = bos_emb.iter()
        .zip(token_450_col.iter())
        .map(|(a, b)| a * b)
        .sum();
    println!("  Logit for token 450: {:.6}", logit_450);

    // Dot product for token 20780
    let logit_20780: f32 = bos_emb.iter()
        .zip(token_20780_col.iter())
        .map(|(a, b)| a * b)
        .sum();
    println!("  Logit for token 20780: {:.6}", logit_20780);

    println!("\n❓ Question: Why is token 20780 higher than token 450?");

    Ok(())
}
