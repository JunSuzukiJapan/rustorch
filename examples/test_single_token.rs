use rustorch::hybrid_f32::models::llama::F32LlamaModel;
use rustorch::hybrid_f32::models::llama::DeviceType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing with single token (BOS token = 1)");

    let model_path = std::path::Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    let mut model = F32LlamaModel::from_gguf_with_device(model_path, DeviceType::Metal)?;

    // Test with BOS token only
    let input_ids = vec![1];
    println!("📥 Input: {:?}", input_ids);

    let logits = model.forward(&input_ids)?;
    println!("📊 Logits shape: {:?}", logits.shape());

    // Find top 5 tokens
    let logits_data = logits.as_slice();
    let vocab_size = 32000;
    let mut indexed: Vec<(usize, f32)> = logits_data[..vocab_size]
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🎯 Top 5 predicted tokens:");
    for (rank, (token, logit)) in indexed.iter().take(5).enumerate() {
        println!("  #{}: token={} logit={:.4}", rank+1, token, logit);
    }

    // Check if "diplom" (13487) is in top 10
    let diplom_pos = indexed.iter().position(|(t, _)| *t == 13487);
    if let Some(pos) = diplom_pos {
        let diplom_logit = indexed[pos].1;
        println!("\n⚠️  Token 13487 (diplom): rank={}, logit={:.4}", pos+1, diplom_logit);
    } else {
        println!("\n✅ Token 13487 (diplom) not in top predictions");
    }

    Ok(())
}
