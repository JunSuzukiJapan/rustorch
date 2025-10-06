use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> F32Result<()> {
    println!("🔍 Testing Q projection with FULL Layer 0 input\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Load full Layer 0 input from saved file
    let input_file = File::open("/tmp/layer0_input.txt")
        .expect("Layer 0 input file not found. Run test_single_token first.");

    let reader = BufReader::new(input_file);
    let full_input: Vec<f32> = reader.lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .collect();

    println!("📊 Loaded {} values from /tmp/layer0_input.txt", full_input.len());
    println!("   First 10: {:?}", &full_input[0..10]);
    println!("   Sum: {:.6}", full_input.iter().sum::<f32>());
    println!("   Non-zero: {}/{}", full_input.iter().filter(|&&x| x != 0.0).count(), full_input.len());

    if full_input.len() != 2048 {
        panic!("Expected 2048 values, got {}", full_input.len());
    }

    // Create tensor and perform Q projection
    let input_tensor = F32Tensor::from_vec(full_input.clone(), &[1, 2048])?;
    let q_weight = model.get_weight("blk.0.attn_q.weight").expect("Q weight not found");

    println!("\n🧮 Performing Q projection: [1, 2048] @ [2048, 2048] = [1, 2048]");
    let q = input_tensor.matmul(q_weight)?;
    let q_data = q.as_slice();

    println!("\n✅ Q projection result (first 10): {:?}", &q_data[0..10]);
    println!("   Expected from debug: [-0.0049681785, 0.08529683, -0.06258121, 0.042906996, -0.04466039, 0.00035404388, -0.020965828, -0.034250896, -0.014786486, 0.03756856]");

    // Compare each value
    let expected = vec![
        -0.0049681785, 0.08529683, -0.06258121, 0.042906996, -0.04466039,
        0.00035404388, -0.020965828, -0.034250896, -0.014786486, 0.03756856
    ];

    println!("\n📊 Comparison:");
    let mut all_match = true;
    for (i, (&computed, &exp)) in q_data.iter().zip(expected.iter()).enumerate() {
        let diff = (computed - exp).abs();
        let matches = diff < 1e-6;
        println!("   Q[{}]: computed={:.9}, expected={:.9}, diff={:.9}, match={}",
            i, computed, exp, diff, matches);
        if !matches {
            all_match = false;
        }
    }

    println!("\n{} Overall match: {}", if all_match { "✅" } else { "❌" }, all_match);

    // Manual calculation for first element to verify
    let q_weight_data = q_weight.as_slice();
    let manual_q0: f32 = full_input.iter()
        .enumerate()
        .map(|(i, &x)| x * q_weight_data[i * 2048 + 0])
        .sum();

    println!("\n🔢 Manual verification:");
    println!("   Q[0] from matmul: {:.9}", q_data[0]);
    println!("   Q[0] from manual: {:.9}", manual_q0);
    println!("   Match: {}", (q_data[0] - manual_q0).abs() < 1e-6);

    Ok(())
}
