use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("🚀 Loading Llama model...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    println!("✅ Model loaded with num_kv_heads={}", model.config().num_kv_heads);

    // Simple generation: Start with BOS token and generate a few tokens
    let mut tokens = vec![1]; // BOS token

    println!("\n🎯 Generating tokens (greedy sampling):");
    print!("Tokens: [1");

    for step in 0..5 {
        let logits = model.forward(&tokens)?;
        let logits_data = logits.as_slice();

        // Greedy: Pick token with highest logit
        let next_token = logits_data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        print!(", {}", next_token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        tokens.push(next_token);

        // Show some diagnostic info
        if step == 0 {
            println!("]");
            println!("\n📊 Step {} diagnostics:", step);
            println!("   Top token: {} (logit: {:.4})", next_token, logits_data[next_token]);

            // Show top 5 tokens
            let mut top_tokens: Vec<(usize, f32)> = logits_data.iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("   Top 5 tokens:");
            for (idx, &(token, logit)) in top_tokens.iter().take(5).enumerate() {
                println!("     {}. Token {} (logit: {:.4})", idx + 1, token, logit);
            }
        }
    }

    println!("\n\n✅ Generation completed!");
    println!("Final token sequence: {:?}", tokens);

    Ok(())
}
