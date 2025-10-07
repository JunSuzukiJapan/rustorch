/// Verify LM head weight columns for tokens 1247 vs 12711
///
/// Token 1247 always gets high logit (~9.9) while token 12711 (what llama.cpp generates)
/// gets low logit (~1.4). Check if the weight columns are problematic.

use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 LM Head Weight Verification\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 Loading output.weight from GGUF...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ output.weight loaded");
    println!("   Shape: {:?}", output_tensor.shape());
    println!("   Expected: [2048, 32000] (hidden_size, vocab_size)");

    let output_data: Vec<f64> = output_tensor.data.iter().cloned().collect();
    let hidden_size = 2048;
    let vocab_size = 32000;

    // Extract weight columns for specific tokens
    let test_tokens = vec![
        (1247, "Token 1247 (ragment) - RusTorch predicts"),
        (12711, "Token 12711 ( there) - llama.cpp predicts"),
        (450, "Token 450 ( The) - reference"),
    ];

    println!("\n═══════════════════════════════════════");
    println!("📊 Weight Column Statistics");
    println!("═══════════════════════════════════════\n");

    for (token_id, name) in &test_tokens {
        let mut weights = Vec::new();
        for dim in 0..hidden_size {
            let idx = dim * vocab_size + token_id;
            weights.push(output_data[idx]);
        }

        let sum: f64 = weights.iter().sum();
        let mean = sum / hidden_size as f64;
        let l2_norm: f64 = weights.iter().map(|x| x * x).sum::<f64>().sqrt();
        let max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let abs_mean = weights.iter().map(|x| x.abs()).sum::<f64>() / hidden_size as f64;

        println!("{} ({}):", token_id, name);
        println!("   Mean: {:.8}", mean);
        println!("   Abs Mean: {:.8}", abs_mean);
        println!("   L2 Norm: {:.8}", l2_norm);
        println!("   Min: {:.8}, Max: {:.8}", min, max);
        println!("   First 5 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                 weights[0], weights[1], weights[2], weights[3], weights[4]);
        println!();
    }

    // Check if token 1247 has abnormally large weights
    println!("═══════════════════════════════════════");
    println!("🔬 Weight Anomaly Analysis");
    println!("═══════════════════════════════════════\n");

    let mut norm_1247 = 0.0;
    let mut norm_12711 = 0.0;
    let mut norm_450 = 0.0;

    for dim in 0..hidden_size {
        let idx_1247 = dim * vocab_size + 1247;
        let idx_12711 = dim * vocab_size + 12711;
        let idx_450 = dim * vocab_size + 450;

        norm_1247 += output_data[idx_1247] * output_data[idx_1247];
        norm_12711 += output_data[idx_12711] * output_data[idx_12711];
        norm_450 += output_data[idx_450] * output_data[idx_450];
    }

    norm_1247 = norm_1247.sqrt();
    norm_12711 = norm_12711.sqrt();
    norm_450 = norm_450.sqrt();

    println!("L2 Norm comparison:");
    println!("   Token 1247: {:.8}", norm_1247);
    println!("   Token 12711: {:.8}", norm_12711);
    println!("   Token 450 (reference): {:.8}", norm_450);
    println!();

    let ratio_1247_vs_12711 = norm_1247 / norm_12711;
    let ratio_1247_vs_450 = norm_1247 / norm_450;

    println!("Norm ratios:");
    println!("   Token 1247 / Token 12711 = {:.4}x", ratio_1247_vs_12711);
    println!("   Token 1247 / Token 450 = {:.4}x", ratio_1247_vs_450);
    println!();

    if ratio_1247_vs_12711 > 2.0 || ratio_1247_vs_450 > 2.0 {
        println!("❌ Token 1247 has ABNORMALLY LARGE weights!");
        println!("   This explains why it always gets high logit.");
        println!("   Possible causes:");
        println!("   - Incorrect weight extraction from GGUF");
        println!("   - Wrong memory layout (row-major vs column-major)");
        println!("   - Dequantization error for this specific token");
    } else {
        println!("✅ Weight magnitudes are within normal range.");
        println!("   The problem is likely elsewhere (embeddings, layer computation, etc.)");
    }

    // Manual logit calculation
    println!("\n═══════════════════════════════════════");
    println!("🧮 Manual Logit Calculation");
    println!("═══════════════════════════════════════\n");

    // Load token embeddings
    let token_embd = loader.load_tensor("token_embd.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;
    let embd_data: Vec<f64> = token_embd.data.iter().cloned().collect();

    // Get embedding for token 1 (BOS)
    let token_1_emb_start = 1 * hidden_size;
    let token_1_emb_end = token_1_emb_start + hidden_size;
    let token_1_emb: Vec<f64> = embd_data[token_1_emb_start..token_1_emb_end].to_vec();

    // Calculate logits manually: logit[t] = embedding · weight_column[t]
    let mut manual_logit_1247 = 0.0;
    let mut manual_logit_12711 = 0.0;
    let mut manual_logit_450 = 0.0;

    for dim in 0..hidden_size {
        let idx_1247 = dim * vocab_size + 1247;
        let idx_12711 = dim * vocab_size + 12711;
        let idx_450 = dim * vocab_size + 450;

        manual_logit_1247 += token_1_emb[dim] * output_data[idx_1247];
        manual_logit_12711 += token_1_emb[dim] * output_data[idx_12711];
        manual_logit_450 += token_1_emb[dim] * output_data[idx_450];
    }

    println!("Manual logits (BOS token embedding · weight columns):");
    println!("   Logit[450]: {:.8}", manual_logit_450);
    println!("   Logit[1247]: {:.8}", manual_logit_1247);
    println!("   Logit[12711]: {:.8}", manual_logit_12711);
    println!();
    println!("Note: These are BEFORE passing through 22 transformer layers.");
    println!("      Actual model logits will be different due to layer processing.");

    Ok(())
}
