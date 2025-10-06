use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Checking output_norm.weight\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Get output_norm.weight
    let output_norm = model.get_weight("output_norm.weight").expect("output_norm.weight not found");
    let norm_data = output_norm.as_slice();

    println!("📊 Output Norm Weight:");
    println!("  Shape: {:?}", output_norm.shape());
    println!("  First 10 values: {:?}", &norm_data[0..10]);
    println!("  Values in range [100-110]: {:?}", &norm_data[100..110]);

    // Statistics
    let min = norm_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = norm_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = norm_data.iter().sum::<f32>() / norm_data.len() as f32;

    println!("\n📈 Statistics:");
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);
    println!("  Mean: {:.6}", mean);
    println!("  Non-zero: {}/{}", norm_data.iter().filter(|&&x| x != 0.0).count(), norm_data.len());

    // Check if weights are around 1.0 (expected for norm weights)
    let around_one = norm_data.iter().filter(|&&x| (x - 1.0).abs() < 0.1).count();
    println!("  Values close to 1.0 (±0.1): {}/{}", around_one, norm_data.len());

    // RMSNorm formula check
    println!("\n🔬 RMSNorm Test:");
    let test_input = vec![0.70855033, 1.0006536, -0.22543797, 0.7980008];
    println!("  Input (from L21): {:?}", test_input);

    // Calculate RMS
    let sum_sq: f32 = test_input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / test_input.len() as f32;
    let rms = (mean_sq + 1e-5).sqrt();
    println!("  RMS = sqrt(mean(x^2) + 1e-5) = {:.6}", rms);

    // Apply norm weights
    let normed: Vec<f32> = test_input.iter()
        .zip(norm_data.iter())
        .map(|(x, w)| (x / rms) * w)
        .collect();

    println!("  Normalized (first 4): {:?}", normed);
    println!("  Expected (from debug): [1.1820991, 1.5812036, -0.38069266, 1.3746278]");

    // Check ratio
    let ratio_0 = 1.1820991 / (0.70855033 / rms);
    let ratio_1 = 1.5812036 / (1.0006536 / rms);
    println!("\n📐 Weight ratios:");
    println!("  ratio[0] = {:.6} (weight[0] = {:.6})", ratio_0, norm_data[0]);
    println!("  ratio[1] = {:.6} (weight[1] = {:.6})", ratio_1, norm_data[1]);

    Ok(())
}
