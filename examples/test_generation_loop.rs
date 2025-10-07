/// Generation loopをシミュレートして、2回目以降のforwardでlogitsが異常になるかテスト
///
/// CLIと同じパターンで実行：
/// 1. clear_cache()
/// 2. 全トークンでforward (step 0)
/// 3. 最後の1トークンだけでforward (step 1) - KV cacheを使用
/// 4. 最後の1トークンだけでforward (step 2) - KV cacheを使用

use rustorch::hybrid_f32::error::F32Result;
use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};

fn main() -> F32Result<()> {
    println!("🔄 Generation Loop Simulation Test\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 モデル読み込み...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Metal)?;

    // Input: "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n"
    let mut generated: Vec<usize> = vec![1, 529, 29989, 1792, 29989, 29958, 13, 5618, 338, 278, 7483, 310, 3444, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13];
    println!("Initial tokens ({}): {:?}\n", generated.len(), &generated[0..10.min(generated.len())]);

    // Clear cache like CLI does
    println!("🧹 Clearing KV cache...");
    model.clear_cache();

    // Step 0: Forward with all tokens (like CLI first step)
    println!("\n═══════════════════════════════════════");
    println!("📍 Step 0: Forward with ALL {} tokens", generated.len());
    println!("═══════════════════════════════════════");

    let logits_0 = model.forward(&generated)?;
    let logits_0_data = logits_0.as_slice();

    println!("✅ Step 0 logits:");
    println!("   Token 450: {:.8}", logits_0_data[450]);
    println!("   Token 20780: {:.8}", logits_0_data[20780]);
    println!("   Token 12517: {:.8}", logits_0_data[12517]);
    println!("   Token 12711 (' there'): {:.8}", logits_0_data[12711]);
    println!("   Max: {:.8}", logits_0_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("   Min: {:.8}", logits_0_data.iter().cloned().fold(f32::INFINITY, f32::min));

    // Find top-5 tokens
    let mut indexed: Vec<(usize, f32)> = logits_0_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("   Top 5:");
    for (rank, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("      {}. token={} logit={:.8}", rank + 1, token_id, logit);
    }

    // Simulate token generation: pick top token
    let next_token = indexed[0].0;
    generated.push(next_token);
    println!("\n   ➡️  Generated token: {}", next_token);

    // Step 1: Forward with ONLY last token (incremental, using KV cache)
    println!("\n═══════════════════════════════════════");
    println!("📍 Step 1: Forward with LAST token only (incremental)");
    println!("═══════════════════════════════════════");

    let last_token = vec![next_token];
    let logits_1 = model.forward(&last_token)?;
    let logits_1_data = logits_1.as_slice();

    println!("✅ Step 1 logits:");
    println!("   Token 450: {:.8}", logits_1_data[450]);
    println!("   Token 20780: {:.8}", logits_1_data[20780]);
    println!("   Token 12517: {:.8}", logits_1_data[12517]);
    println!("   Max: {:.8}", logits_1_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("   Min: {:.8}", logits_1_data.iter().cloned().fold(f32::INFINITY, f32::min));

    // Find top-5 tokens
    let mut indexed_1: Vec<(usize, f32)> = logits_1_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("   Top 5:");
    for (rank, (token_id, logit)) in indexed_1.iter().take(5).enumerate() {
        println!("      {}. token={} logit={:.8}", rank + 1, token_id, logit);
    }

    let next_token_1 = indexed_1[0].0;
    generated.push(next_token_1);
    println!("\n   ➡️  Generated token: {}", next_token_1);

    // Step 2: Forward with ONLY last token (incremental, using KV cache)
    println!("\n═══════════════════════════════════════");
    println!("📍 Step 2: Forward with LAST token only (incremental)");
    println!("═══════════════════════════════════════");

    let last_token_2 = vec![next_token_1];
    let logits_2 = model.forward(&last_token_2)?;
    let logits_2_data = logits_2.as_slice();

    println!("✅ Step 2 logits:");
    println!("   Token 450: {:.8}", logits_2_data[450]);
    println!("   Token 20780: {:.8}", logits_2_data[20780]);
    println!("   Token 12517: {:.8}", logits_2_data[12517]);
    println!("   Max: {:.8}", logits_2_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("   Min: {:.8}", logits_2_data.iter().cloned().fold(f32::INFINITY, f32::min));

    // Find top-5 tokens
    let mut indexed_2: Vec<(usize, f32)> = logits_2_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("   Top 5:");
    for (rank, (token_id, logit)) in indexed_2.iter().take(5).enumerate() {
        println!("      {}. token={} logit={:.8}", rank + 1, token_id, logit);
    }

    // Final summary
    println!("\n═══════════════════════════════════════");
    println!("📊 Summary");
    println!("═══════════════════════════════════════");
    println!("Step 0 (all tokens): max={:.2}, normal range",
             logits_0_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("Step 1 (incremental): max={:.2}, {} range",
             logits_1_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
             if logits_1_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) > 5.0 { "ABNORMAL" } else { "normal" });
    println!("Step 2 (incremental): max={:.2}, {} range",
             logits_2_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
             if logits_2_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) > 5.0 { "ABNORMAL" } else { "normal" });

    println!("\nGenerated sequence: {:?}", generated);

    Ok(())
}
