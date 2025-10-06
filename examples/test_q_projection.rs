use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing Q projection calculation\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Get BOS embedding (from Layer 0 input after RMSNorm)
    // From debug: [-1.5296754e-6, -0.0058243065, -0.020103388, 0.0, ...]
    let layer0_input = vec![
        -1.5296754e-6, -0.0058243065, -0.020103388, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002951049
    ];
    let mut full_input = layer0_input.clone();
    full_input.resize(2048, 0.0);

    let input_tensor = F32Tensor::from_vec(full_input.clone(), &[1, 2048])?;
    let q_weight = model.get_weight("blk.0.attn_q.weight").expect("Q weight not found");

    println!("📊 Input (Layer 0 after attn_norm, first 10): {:?}", &full_input[0..10]);
    println!("📊 Q weight shape: {:?}\n", q_weight.shape());

    // Perform Q projection: [1, 2048] @ [2048, 2048] = [1, 2048]
    let q = input_tensor.matmul(q_weight)?;
    let q_data = q.as_slice();

    println!("🧮 Q projection result (first 10): {:?}", &q_data[0..10]);
    println!("   Expected from debug: [-0.0049681785, 0.08529683, -0.06258121, 0.042906996, -0.04466039, 0.00035404388, -0.020965828, -0.034250896, -0.014786486, 0.03756856]\n");

    // Manual calculation for first element
    let q_weight_data = q_weight.as_slice();
    let manual_q0: f32 = full_input.iter()
        .enumerate()
        .map(|(i, &x)| x * q_weight_data[i * 2048 + 0])
        .sum();

    println!("🔢 Manual calculation:");
    println!("   Q[0] from matmul: {:.9}", q_data[0]);
    println!("   Q[0] from manual: {:.9}", manual_q0);
    println!("   Match: {}", (q_data[0] - manual_q0).abs() < 1e-6);

    Ok(())
}
