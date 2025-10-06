use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Comparing embeddings with llama.cpp expectations\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Get token embeddings
    let token_embd = model.get_weight("token_embd.weight")
        .expect("token_embd.weight not found");

    println!("Token embedding shape: {:?}", token_embd.shape());
    println!("Expected: [hidden_size, vocab_size] = [2048, 32000]\n");

    // Check embeddings for our test tokens
    let test_tokens = vec![
        (1, "<s>"),
        (259, "  "),
        (5618, "What"),
        (338, " is"),
        (1552, " the"),
    ];

    let embd_data = token_embd.as_slice();
    let hidden_size = 2048;
    let vocab_size = 32000;

    for (token_id, token_text) in test_tokens {
        println!("Token {} ('{}'):", token_id, token_text);

        // Extract embedding for this token
        // token_embd.weight is [hidden_size, vocab_size] in row-major
        // So token_id's embedding starts at column token_id
        let mut embedding: Vec<f32> = Vec::new();
        for row in 0..hidden_size {
            let idx = row * vocab_size + token_id;
            embedding.push(embd_data[idx]);
        }

        // Show first 10 values
        println!("  Embedding[0..10]: {:?}", &embedding[0..10]);

        // Calculate statistics
        let sum: f32 = embedding.iter().sum();
        let mean = sum / hidden_size as f32;
        let variance: f32 = embedding.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;
        let std_dev = variance.sqrt();
        let non_zero = embedding.iter().filter(|&&x| x.abs() > 1e-6).count();

        println!("  Stats: mean={:.6}, std={:.6}, non_zero={}/{}", mean, std_dev, non_zero, hidden_size);
        println!();
    }

    // Compare with what we logged during forward pass
    println!("📊 Comparison with forward pass logs:");
    println!("  Our logged embedding for token 1:");
    println!("    [-0.0013000965, 0.0019042492, -0.0019409657, 0.0038268566, ...]");
    println!("\n  If embeddings match, the issue is in:");
    println!("    - Transformer layers");
    println!("    - RMSNorm");
    println!("    - Attention mechanism");
    println!("\n  If embeddings DON'T match, the issue is in:");
    println!("    - Weight loading");
    println!("    - Quantization dequantization");

    Ok(())
}
