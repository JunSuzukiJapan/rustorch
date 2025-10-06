use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 Testing single token forward pass\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Test with BOS token only
    let input_ids = vec![1];
    println!("Input: token 1 (BOS)");

    let logits = model.forward(&input_ids)?;
    let logits_data = logits.as_slice();

    println!("\nLogits shape: {:?}", logits.shape());
    println!("Vocab size: {}", logits_data.len());

    // Find top 10
    let mut indexed: Vec<(usize, f32)> = logits_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 predictions after BOS:");
    for (rank, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        println!("  {}. Token {}: {:.6}", rank + 1, token_id, logit);
    }

    // Statistics
    let max_logit = indexed[0].1;
    let min_logit = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let sum: f32 = logits_data.iter().sum();
    let mean = sum / logits_data.len() as f32;

    println!("\nLogit statistics:");
    println!("  Max: {:.6}", max_logit);
    println!("  Min: {:.6}", min_logit);
    println!("  Mean: {:.6}", mean);

    println!("\n✅ Single token test completed");

    Ok(())
}
