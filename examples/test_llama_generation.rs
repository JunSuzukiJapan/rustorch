use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("🚀 Loading Llama model...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    println!("\n✅ Model loaded successfully!");
    println!("   Config: vocab={}, hidden={}, layers={}, heads={}, kv_heads={}",
        model.config().vocab_size,
        model.config().hidden_size,
        model.config().num_layers,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    // Test tokens: "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n"
    // Approximation with basic tokens
    let test_tokens = vec![1, 1724, 3681]; // BOS, "What", "is"

    println!("\n🧪 Testing forward pass with tokens: {:?}", test_tokens);

    let logits = model.forward(&test_tokens)?;
    let logits_data = logits.as_slice();

    println!("   Logits shape: {:?}", logits.shape());
    println!("   Logits sample[0..10]: {:?}", &logits_data[0..10.min(logits_data.len())]);

    // Find top token
    let top_idx = logits_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let top_logit = logits_data[top_idx];

    println!("\n📊 Prediction:");
    println!("   Top token ID: {}", top_idx);
    println!("   Top logit: {:.6}", top_logit);

    // Check specific tokens
    let paris_token = 3681; // Approximate token for "Paris"
    let the_token = 278;    // Token for "the"

    if paris_token < logits_data.len() {
        println!("   Logit for 'Paris' ({}): {:.6}", paris_token, logits_data[paris_token]);
    }
    if the_token < logits_data.len() {
        println!("   Logit for 'the' ({}): {:.6}", the_token, logits_data[the_token]);
    }

    println!("\n✅ Test completed successfully!");

    Ok(())
}
