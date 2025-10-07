/// Compare logit outputs between CPU and Metal backends
///
/// If Metal produces different logits than CPU, that's the root cause.

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 CPU vs Metal Comparison\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    // Test input
    let input = vec![1, 529, 29989, 1792, 29989, 29958, 13, 5618, 338, 278, 7483, 310, 3444, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13];

    println!("Input: {} tokens", input.len());
    println!("Input tokens: {:?}\n", input);

    // Test with CPU
    println!("═══════════════════════════════════════");
    println!("🖥️  CPU Backend");
    println!("═══════════════════════════════════════\n");

    let mut model_cpu = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    let logits_cpu = model_cpu.forward(&input)?;
    let logits_cpu_data = logits_cpu.as_slice();

    println!("CPU Logits:");
    println!("   Token 450: {:.8}", logits_cpu_data[450]);
    println!("   Token 1247: {:.8}", logits_cpu_data[1247]);
    println!("   Token 12711: {:.8}", logits_cpu_data[12711]);

    // Find top 5
    let mut indexed_cpu: Vec<(usize, f32)> = logits_cpu_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_cpu.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n   Top 5:");
    for (rank, (token_id, logit)) in indexed_cpu[..5].iter().enumerate() {
        println!("      {}. token={} logit={:.4}", rank + 1, token_id, logit);
    }

    // Test with Metal
    println!("\n═══════════════════════════════════════");
    println!("🎨 Metal Backend");
    println!("═══════════════════════════════════════\n");

    let mut model_metal = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Metal)?;
    let logits_metal = model_metal.forward(&input)?;
    let logits_metal_data = logits_metal.as_slice();

    println!("Metal Logits:");
    println!("   Token 450: {:.8}", logits_metal_data[450]);
    println!("   Token 1247: {:.8}", logits_metal_data[1247]);
    println!("   Token 12711: {:.8}", logits_metal_data[12711]);

    // Find top 5
    let mut indexed_metal: Vec<(usize, f32)> = logits_metal_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_metal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n   Top 5:");
    for (rank, (token_id, logit)) in indexed_metal[..5].iter().enumerate() {
        println!("      {}. token={} logit={:.4}", rank + 1, token_id, logit);
    }

    // Compare
    println!("\n═══════════════════════════════════════");
    println!("📊 Comparison");
    println!("═══════════════════════════════════════\n");

    let diff_450 = (logits_cpu_data[450] - logits_metal_data[450]).abs();
    let diff_1247 = (logits_cpu_data[1247] - logits_metal_data[1247]).abs();
    let diff_12711 = (logits_cpu_data[12711] - logits_metal_data[12711]).abs();

    println!("Absolute differences:");
    println!("   Token 450: {:.8}", diff_450);
    println!("   Token 1247: {:.8}", diff_1247);
    println!("   Token 12711: {:.8}", diff_12711);
    println!();

    let max_diff = diff_450.max(diff_1247).max(diff_12711);

    if max_diff < 0.01 {
        println!("✅ CPU and Metal produce nearly identical logits!");
        println!("   The problem is NOT device-specific.");
    } else if max_diff < 0.1 {
        println!("⚠️  Small differences between CPU and Metal (< 0.1)");
        println!("   Likely acceptable floating-point variance.");
    } else {
        println!("❌ SIGNIFICANT differences between CPU and Metal!");
        println!("   Metal backend has a bug or numerical instability.");
    }

    // Check if top predictions differ
    if indexed_cpu[0].0 != indexed_metal[0].0 {
        println!("\n❌ CRITICAL: Different top predictions!");
        println!("   CPU predicts: Token {}", indexed_cpu[0].0);
        println!("   Metal predicts: Token {}", indexed_metal[0].0);
    } else {
        println!("\n✅ Both backends predict the same top token: {}", indexed_cpu[0].0);
    }

    Ok(())
}
