/// Verify that token embeddings are extracted correctly from GGUF
///
/// Compare embeddings for specific tokens between RusTorch and expected values

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;
use rustorch::formats::gguf::GGUFLoader;

fn main() -> F32Result<()> {
    println!("🔍 Embedding Extraction Verification\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    // Load token embeddings directly from GGUF
    println!("📂 Loading token.embd.weight from GGUF...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    let token_embd = loader.load_tensor("token_embd.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ token_embd.weight loaded");
    println!("   Shape: {:?}", token_embd.shape());
    println!("   Expected: [32000, 2048] (vocab_size, hidden_size)");

    let embd_data: Vec<f64> = token_embd.data.iter().cloned().collect();

    // GGUF stores embeddings as [vocab_size, hidden_size]
    // For token T, embedding is embd_data[T*hidden_size .. (T+1)*hidden_size]
    let hidden_size = 2048;
    let vocab_size = 32000;

    // Check embeddings for input tokens
    let test_tokens = vec![
        (1, "BOS"),
        (529, "Token 529"),
        (29989, "Token 29989"),
        (1247, "Token 1247 (ragment)"),
        (12711, "Token 12711 ( there)"),
    ];

    println!("\n═══════════════════════════════════════");
    println!("📊 Token Embedding Statistics");
    println!("═══════════════════════════════════════\n");

    for (token_id, name) in &test_tokens {
        let start = token_id * hidden_size;
        let end = start + hidden_size;
        let embedding: Vec<f64> = embd_data[start..end].to_vec();

        let sum: f64 = embedding.iter().sum();
        let mean = sum / hidden_size as f64;
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        let min = embedding.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = embedding.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Token {} ({}):", token_id, name);
        println!("   Mean: {:.8}", mean);
        println!("   L2 Norm: {:.8}", norm);
        println!("   Min: {:.8}, Max: {:.8}", min, max);
        println!("   First 5 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                 embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);
        println!();
    }

    // Now compare with model's embedding lookup
    println!("═══════════════════════════════════════");
    println!("🔬 Compare with Model Embedding Lookup");
    println!("═══════════════════════════════════════\n");

    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // Forward with single token to get its embedding
    for (token_id, name) in &test_tokens[0..2] {  // Just BOS and token 529
        let input = vec![*token_id];
        let output = model.forward(&input)?;
        let logits = output.as_slice();

        println!("Token {} ({}):", token_id, name);
        println!("   Logit[450]: {:.8}", logits[450]);
        println!("   Logit[1247]: {:.8}", logits[1247]);
        println!("   Logit[12711]: {:.8}", logits[12711]);
        println!();
    }

    // Check if embedding extraction is consistent
    println!("═══════════════════════════════════════");
    println!("🧪 Embedding Consistency Check");
    println!("═══════════════════════════════════════\n");

    // Token 1 (BOS) should have same embedding whether alone or in sequence
    let input_single = vec![1];
    let input_sequence = vec![1, 529, 29989];

    let out1 = model.forward(&input_single)?;
    model.clear_cache();
    let out2 = model.forward(&input_sequence)?;

    let logits1 = out1.as_slice();
    let logits2 = out2.as_slice();

    println!("Logits for token 1 (BOS):");
    println!("   Single: logit[450]={:.8}, logit[1247]={:.8}", logits1[450], logits1[1247]);
    println!("   In sequence (last pos): logit[450]={:.8}, logit[1247]={:.8}", logits2[450], logits2[1247]);
    println!();

    let diff_450 = (logits1[450] - logits2[450]).abs();
    let diff_1247 = (logits1[1247] - logits2[1247]).abs();

    if diff_450 < 0.001 && diff_1247 < 0.001 {
        println!("✅ Embeddings are consistent!");
    } else {
        println!("❌ Embeddings differ! (diff_450={:.8}, diff_1247={:.8})", diff_450, diff_1247);
        println!("   This suggests embedding extraction or position encoding issues.");
    }

    Ok(())
}
