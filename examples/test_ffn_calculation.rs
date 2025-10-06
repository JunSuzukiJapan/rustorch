use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> F32Result<()> {
    println!("🔍 Testing FFN Calculation Step-by-Step\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Load Layer 0 FFN input (after attention + residual)
    // From Layer 0 debug: after_attn_residual[0..10]=[0.0023096392, -0.011357173, ...]
    let ffn_input_vals = vec![
        0.0023096392, -0.011357173, -0.0048085107, 0.0003897791, -0.000791425,
        -0.00894543, 0.00081642263, -6.950322e-5, -0.006838815, 0.008416813
    ];

    // Pad to 2048
    let mut full_input = ffn_input_vals.clone();
    full_input.resize(2048, 0.0);

    println!("📊 FFN Input (first 10): {:?}", &full_input[0..10]);

    // Apply RMSNorm manually
    let ffn_norm_weight = model.get_weight("blk.0.ffn_norm.weight").expect("FFN norm weight not found");
    let weight_data = ffn_norm_weight.as_slice();

    // Calculate RMS
    let sum_sq: f32 = full_input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / 2048.0 + 1e-5).sqrt();

    // Apply normalization
    let mut normed_data = Vec::with_capacity(2048);
    for (i, &x) in full_input.iter().enumerate() {
        normed_data.push((x / rms) * weight_data[i]);
    }

    println!("📊 After FFN RMSNorm (first 10): {:?}", &normed_data[0..10]);
    println!("   Expected from debug: [0.008640736, -0.04344028, -0.024030644, 0.0045052688, -0.0030050415, -0.036962792, 0.0032823079, -0.0002949513, -0.026730722, 0.035248507]");

    // Gate projection
    let normed_tensor = F32Tensor::from_vec(normed_data.clone(), &[1, 2048])?;
    let gate_weight = model.get_weight("blk.0.ffn_gate.weight").expect("Gate weight not found");
    let gate = normed_tensor.matmul(gate_weight)?;
    let gate_data = gate.as_slice();

    println!("\n📊 Gate projection (first 10): {:?}", &gate_data[0..10]);

    // Up projection
    let up_weight = model.get_weight("blk.0.ffn_up.weight").expect("Up weight not found");
    let up = normed_tensor.matmul(up_weight)?;
    let up_data = up.as_slice();

    println!("📊 Up projection (first 10): {:?}", &up_data[0..10]);

    // SwiGLU
    let mut swiglu_vals = Vec::with_capacity(5632);
    for (&g, &u) in gate_data.iter().zip(up_data.iter()) {
        let silu = g / (1.0 + (-g).exp());
        swiglu_vals.push(silu * u);
    }

    println!("\n📊 After SwiGLU (first 10): {:?}", &swiglu_vals[0..10]);

    // Down projection
    let down_weight = model.get_weight("blk.0.ffn_down.weight").expect("Down weight not found");
    let swiglu_tensor = F32Tensor::from_vec(swiglu_vals, &[1, 5632])?;
    let final_out = swiglu_tensor.matmul(down_weight)?;
    let final_data = final_out.as_slice();

    println!("\n📊 FFN Final Output (first 10): {:?}", &final_data[0..10]);
    println!("   Expected from debug: [0.0022328205, -0.0020529367, -0.0043444955, -0.00095260947, -0.0011416259, -0.00064007053, 0.0022228716, -0.0009704352, 0.0003653685, 0.00049026456]");

    // Compare
    let expected = vec![
        0.0022328205, -0.0020529367, -0.0043444955, -0.00095260947, -0.0011416259,
        -0.00064007053, 0.0022228716, -0.0009704352, 0.0003653685, 0.00049026456
    ];

    println!("\n🔍 Comparison:");
    for (i, (&computed, &exp)) in final_data.iter().zip(expected.iter()).enumerate() {
        let diff = (computed - exp).abs();
        let matches = diff < 1e-6;
        println!("   [{}]: computed={:.9}, expected={:.9}, diff={:.9}, match={}",
            i, computed, exp, diff, matches);
    }

    Ok(())
}
