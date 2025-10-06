use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing Q4_K_M Dequantization\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Test a specific weight tensor dequantization
    // Let's check the Q weight for Layer 0
    let q_weight = model.get_weight("blk.0.attn_q.weight")
        .expect("Q weight not found");

    println!("📊 blk.0.attn_q.weight dequantized values:");
    println!("   Shape: {:?}", q_weight.shape());

    let data = q_weight.as_slice();

    // Show first 20 values
    println!("\n   First 20 values:");
    for (i, &val) in data.iter().take(20).enumerate() {
        println!("   [{}] = {:.9}", i, val);
    }

    // Show statistics
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let non_zero = data.iter().filter(|&&x| x.abs() > 1e-8).count();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n   Statistics:");
    println!("   Total elements: {}", data.len());
    println!("   Sum: {:.6}", sum);
    println!("   Mean: {:.9}", mean);
    println!("   Min: {:.9}", min);
    println!("   Max: {:.9}", max);
    println!("   Non-zero: {}/{}", non_zero, data.len());

    // Check a specific pattern: values at positions that correspond to
    // the first super-block boundary (256 elements)
    println!("\n   Values at super-block boundaries:");
    for i in &[0, 255, 256, 511, 512, 767, 768, 1023] {
        if *i < data.len() {
            println!("   [{}] = {:.9}", i, data[*i]);
        }
    }

    // Save first 1000 values to file for comparison with llama.cpp
    use std::io::Write;
    if let Ok(mut file) = std::fs::File::create("/tmp/rustorch_q_weight.txt") {
        for (i, &val) in data.iter().take(1000).enumerate() {
            writeln!(file, "{} {:.9}", i, val).ok();
        }
        println!("\n💾 First 1000 values saved to /tmp/rustorch_q_weight.txt");
        println!("   Compare with llama.cpp output using same model");
    }

    Ok(())
}
