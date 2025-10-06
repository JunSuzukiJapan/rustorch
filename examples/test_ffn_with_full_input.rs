use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> F32Result<()> {
    println!("🔍 Testing FFN with FULL Input\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Load full Layer 0 after_attn_residual
    let input_file = File::open("/tmp/layer0_after_attn_residual.txt")
        .expect("Layer 0 after_attn_residual file not found");

    let reader = BufReader::new(input_file);
    let full_input: Vec<f32> = reader.lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();

    println!("📊 Loaded {} values", full_input.len());
    println!("   First 10: {:?}", &full_input[0..10]);

    // Apply RMSNorm manually
    let ffn_norm_weight = model.get_weight("blk.0.ffn_norm.weight").expect("FFN norm weight not found");
    let weight_data = ffn_norm_weight.as_slice();

    // Calculate RMS
    let sum_sq: f32 = full_input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / 2048.0 + 1e-5).sqrt();

    println!("\n📊 RMSNorm Calculation:");
    println!("   sum_sq: {:.6}", sum_sq);
    println!("   rms: {:.6}", rms);
    println!("   Expected rms from debug: 0.017489");

    // Apply normalization
    let mut normed_data = Vec::with_capacity(2048);
    for (i, &x) in full_input.iter().enumerate() {
        normed_data.push((x / rms) * weight_data[i]);
    }

    println!("\n📊 After RMSNorm (first 10): {:?}", &normed_data[0..10]);
    println!("   Expected: [0.008640736, -0.04344028, -0.024030644, 0.0045052688, -0.0030050415, -0.036962792, 0.0032823079, -0.0002949513, -0.026730722, 0.035248507]");

    // Compare
    let expected = vec![
        0.008640736, -0.04344028, -0.024030644, 0.0045052688, -0.0030050415,
        -0.036962792, 0.0032823079, -0.0002949513, -0.026730722, 0.035248507
    ];

    println!("\n🔍 Comparison:");
    for (i, (&computed, &exp)) in normed_data.iter().zip(expected.iter()).enumerate() {
        let diff = (computed - exp).abs();
        let matches = diff < 1e-6;
        println!("   [{}]: computed={:.9}, expected={:.9}, diff={:.9}, match={}",
            i, computed, exp, diff, matches);
    }

    Ok(())
}
