use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing with Q4_0 Quantization Model\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

    println!("Loading Q4_0 model (simpler quantization)...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Test with BOS token
    let input_tokens = vec![1]; // BOS

    println!("🔄 Running forward pass with BOS token...");
    let output = model.forward(&input_tokens)?;
    let logits = output.as_slice();

    println!("\n📊 Logits shape: [1, {}]", logits.len());
    println!("   Vocab size: {}", model.config().vocab_size);

    // Find top predictions
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10 predictions after BOS:");
    for (rank, &(token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        println!("  {}. Token {}: {:.6}", rank + 1, token_id, logit);
    }

    // Check specific tokens
    println!("\n🔍 Specific token logits:");
    println!("  Token 450 (\" The\"): {:.6}", logits[450]);
    println!("  Token 20780: {:.6}", logits[20780]);

    // Compare with Q4_K_M results
    println!("\n📊 Comparison with Q4_K_M:");
    println!("  Q4_K_M predicted: Token 20780 with logit 9.579");
    println!("  Q4_K_M Token 450 logit: 0.063");
    println!("\n  Q4_0 predicted: Token {} with logit {:.6}", indexed_logits[0].0, indexed_logits[0].1);
    println!("  Q4_0 Token 450 logit: {:.6}", logits[450]);

    if indexed_logits[0].0 == 450 || indexed_logits[0].0 == 20780 {
        if indexed_logits[0].0 == 450 {
            println!("\n✅ SUCCESS: Q4_0 correctly predicts token 450!");
            println!("   This confirms Q4_K_M dequantization is the problem");
        } else {
            println!("\n❌ SAME ISSUE: Q4_0 also predicts token 20780");
            println!("   Problem may be in common code, not Q4_K_M specific");
        }
    } else {
        println!("\n⚠️  Q4_0 predicts different token: {}", indexed_logits[0].0);
        println!("   Need to verify which is correct");
    }

    Ok(())
}
