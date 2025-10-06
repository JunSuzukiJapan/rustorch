use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Testing FFN Weight Dequantization\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Check FFN weights for Layer 0
    let gate_weight = model.get_weight("blk.0.ffn_gate.weight").expect("Gate weight not found");
    let up_weight = model.get_weight("blk.0.ffn_up.weight").expect("Up weight not found");
    let down_weight = model.get_weight("blk.0.ffn_down.weight").expect("Down weight not found");

    println!("📊 FFN Weight Shapes:");
    println!("   Gate: {:?}", gate_weight.shape());
    println!("   Up: {:?}", up_weight.shape());
    println!("   Down: {:?}", down_weight.shape());

    // Check if shapes are transposed correctly for matmul
    // Expected: gate [2048, 5632], up [2048, 5632], down [5632, 2048]
    println!("\n✅ Expected for matmul:");
    println!("   x [1, 2048] @ gate [2048, 5632] = [1, 5632]");
    println!("   x [1, 2048] @ up [2048, 5632] = [1, 5632]");
    println!("   swiglu [1, 5632] @ down [5632, 2048] = [1, 2048]");

    // Show first 10 values of each
    println!("\n📊 Gate weight first 10 values:");
    for (i, &val) in gate_weight.as_slice().iter().take(10).enumerate() {
        println!("   [{}] = {:.9}", i, val);
    }

    println!("\n📊 Up weight first 10 values:");
    for (i, &val) in up_weight.as_slice().iter().take(10).enumerate() {
        println!("   [{}] = {:.9}", i, val);
    }

    println!("\n📊 Down weight first 10 values:");
    for (i, &val) in down_weight.as_slice().iter().take(10).enumerate() {
        println!("   [{}] = {:.9}", i, val);
    }

    // Check statistics
    let gate_data = gate_weight.as_slice();
    let up_data = up_weight.as_slice();
    let down_data = down_weight.as_slice();

    println!("\n📈 Statistics:");
    println!("   Gate: mean={:.6}, min={:.6}, max={:.6}",
        gate_data.iter().sum::<f32>() / gate_data.len() as f32,
        gate_data.iter().cloned().fold(f32::INFINITY, f32::min),
        gate_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    println!("   Up:   mean={:.6}, min={:.6}, max={:.6}",
        up_data.iter().sum::<f32>() / up_data.len() as f32,
        up_data.iter().cloned().fold(f32::INFINITY, f32::min),
        up_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    println!("   Down: mean={:.6}, min={:.6}, max={:.6}",
        down_data.iter().sum::<f32>() / down_data.len() as f32,
        down_data.iter().cloned().fold(f32::INFINITY, f32::min),
        down_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    Ok(())
}
