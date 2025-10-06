/// Direct comparison between RusTorch and llama.cpp output
///
/// This test verifies if the discrepancy is due to:
/// 1. Quantization differences (Q4_0 vs Q4_K_M)
/// 2. Chat template application in llama.cpp
/// 3. Actual model inference differences

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔬 Comparison: RusTorch vs llama.cpp\n");

    // Test with Q4_K_M
    println!("═══════════════════════════════════════");
    println!("📊 Q4_K_M Quantization");
    println!("═══════════════════════════════════════\n");

    let q4km_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading Q4_K_M model...");
    let mut model_km = F32LlamaModel::from_gguf_with_device(&q4km_path, DeviceType::Cpu)?;

    let input = vec![1]; // BOS only
    let logits_km = model_km.forward(&input)?;
    let logits_km_data = logits_km.as_slice();

    let mut indexed_km: Vec<(usize, f32)> = logits_km_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_km.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 RusTorch Q4_K_M - Top 5:");
    for (i, (token, logit)) in indexed_km.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    println!("\nllama.cpp Q4_K_M output: 'Air Force Rec'");
    println!("   → Suggests tokens like [Air, Force, Rec, ...]");
    println!("   → But llama.cpp likely applied chat template!");

    // Test with Q4_0
    println!("\n═══════════════════════════════════════");
    println!("📊 Q4_0 Quantization");
    println!("═══════════════════════════════════════\n");

    let q40_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

    println!("Loading Q4_0 model...");
    let mut model_0 = F32LlamaModel::from_gguf_with_device(&q40_path, DeviceType::Cpu)?;

    let logits_0 = model_0.forward(&input)?;
    let logits_0_data = logits_0.as_slice();

    let mut indexed_0: Vec<(usize, f32)> = logits_0_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_0.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 RusTorch Q4_0 - Top 5:");
    for (i, (token, logit)) in indexed_0.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    println!("\nllama.cpp Q4_0 output: 'The book's'");
    println!("   → Suggests tokens like [The, book, 's, ...]");
    println!("   → But llama.cpp likely applied chat template!");

    // Analysis
    println!("\n═══════════════════════════════════════");
    println!("🔍 Analysis");
    println!("═══════════════════════════════════════\n");

    println!("KEY FINDING:");
    println!("  llama.cpp produces DIFFERENT outputs for Q4_0 vs Q4_K_M");
    println!("  This is UNEXPECTED if both use same input!");
    println!();
    println!("HYPOTHESIS:");
    println!("  1. llama.cpp applies chat template → adds tokens beyond BOS");
    println!("  2. OR llama.cpp uses temperature/sampling → non-deterministic");
    println!("  3. OR quantization legitimately affects output");
    println!();
    println!("NEXT STEPS:");
    println!("  1. Run llama.cpp with --temp 0 for deterministic output");
    println!("  2. Use llama.cpp's verbose mode to see actual input tokens");
    println!("  3. Compare raw logits from llama.cpp vs RusTorch");

    Ok(())
}
