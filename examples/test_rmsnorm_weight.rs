use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing RMSNorm Weight Values\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Check Layer 0 normalization weights
    let attn_norm = model.get_weight("blk.0.attn_norm.weight").expect("Attn norm not found");
    let ffn_norm = model.get_weight("blk.0.ffn_norm.weight").expect("FFN norm not found");

    println!("📊 Layer 0 Attention Norm Weight:");
    println!("   First 10: {:?}", &attn_norm.as_slice()[0..10]);
    let attn_mean: f32 = attn_norm.as_slice().iter().sum::<f32>() / 2048.0;
    println!("   Mean: {:.6}", attn_mean);

    println!("\n📊 Layer 0 FFN Norm Weight:");
    println!("   First 10: {:?}", &ffn_norm.as_slice()[0..10]);
    let ffn_mean: f32 = ffn_norm.as_slice().iter().sum::<f32>() / 2048.0;
    println!("   Mean: {:.6}", ffn_mean);

    // Check if they're around 1.0 (typical for RMSNorm weights)
    println!("\n🔍 Analysis:");
    println!("   Attn norm weight close to 1.0: {}", (attn_mean - 1.0).abs() < 0.5);
    println!("   FFN norm weight close to 1.0: {}", (ffn_mean - 1.0).abs() < 0.5);

    // Test RMSNorm calculation with known input
    let test_input = vec![0.0023096392, -0.011357173, -0.0048085107, 0.0003897791, -0.000791425,
                          -0.00894543, 0.00081642263, -6.950322e-5, -0.006838815, 0.008416813];
    let mut full_input = test_input.clone();
    full_input.resize(2048, 0.0);

    // Manual RMSNorm
    let sum_sq: f32 = full_input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / 2048.0 + 1e-5).sqrt();
    println!("\n📊 RMSNorm Test:");
    println!("   Input sum_sq: {:.9}", sum_sq);
    println!("   RMS: {:.9}", rms);
    println!("   Expected RMS for debug output: ~0.0052 (based on ratio)");

    // Apply with FFN norm weight
    let weight_data = ffn_norm.as_slice();
    let normed_0 = (full_input[0] / rms) * weight_data[0];
    let normed_1 = (full_input[1] / rms) * weight_data[1];

    println!("\n📊 RMSNorm Output:");
    println!("   normed[0] = ({:.9} / {:.9}) * {:.6} = {:.9}",
             full_input[0], rms, weight_data[0], normed_0);
    println!("   normed[1] = ({:.9} / {:.9}) * {:.6} = {:.9}",
             full_input[1], rms, weight_data[1], normed_1);
    println!("\n   Expected normed[0]: 0.008640736");
    println!("   Expected normed[1]: -0.04344028");

    Ok(())
}
