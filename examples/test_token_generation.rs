use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("🚀 Loading Llama model...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded");

    // Test tokens: BOS + " What" + " is" (with leading spaces from BPE)
    // Verified with llama.cpp tokenizer: 1 -> '<s>', 1724 -> ' What', 338 -> ' is'
    let mut input_ids = vec![1, 1724, 338];

    println!("\n🧪 Testing token generation (5 tokens)");
    println!("📥 Input tokens: {:?}", input_ids);

    // Generate 5 tokens
    for step in 0..5 {
        // Forward pass with KV cache
        let input_for_forward = if step == 0 {
            &input_ids[..]  // All tokens on first step
        } else {
            &input_ids[input_ids.len() - 1..]  // Only last token
        };

        println!("\n🔄 Step {}: Forward with {} token(s)", step, input_for_forward.len());
        let logits = model.forward(input_for_forward)?;
        let logits_data = logits.as_slice();

        // Find top-5 tokens
        let mut indexed: Vec<(usize, f32)> = logits_data.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("📊 Top-5 logits:");
        for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
            println!("   {}. Token {}: {:.3}", i + 1, token_id, logit);
        }

        // Logit statistics
        let max_logit = indexed[0].1;
        let min_logit = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let sum: f32 = logits_data.iter().sum();
        let mean = sum / logits_data.len() as f32;
        println!("   Stats: max={:.3}, min={:.3}, mean={:.3}", max_logit, min_logit, mean);

        // Sample top token (greedy)
        let next_token = indexed[0].0;
        println!("✅ Sampled token: {}", next_token);

        input_ids.push(next_token);
    }

    println!("\n📤 Output tokens: {:?}", &input_ids[3..]);
    println!("\n✅ Test completed!");

    Ok(())
}
