use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> F32Result<()> {
    println!("🔍 Testing exact hidden state from forward pass\n");

    // Load hidden state from file
    let file = File::open("/tmp/hidden_state.txt").expect("Hidden state file not found");
    let reader = BufReader::new(file);
    let hidden: Vec<f32> = reader.lines()
        .map(|line| line.unwrap().parse::<f32>().unwrap())
        .collect();

    println!("📊 Loaded hidden state: {} values", hidden.len());
    println!("   First 10: {:?}", &hidden[0..10]);
    println!("   Sum: {:.6}\n", hidden.iter().sum::<f32>());

    // Load model
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    let hidden_tensor = F32Tensor::from_vec(hidden.clone(), &[1, 2048])?;
    let lm_weight = model.get_weight("output.weight").expect("output.weight not found");

    println!("🧮 Performing matmul...");
    let logits = hidden_tensor.matmul(lm_weight)?;
    let logits_data = logits.as_slice();

    println!("   Result shape: {:?}", logits.shape());
    println!("   Logits[0..10]: {:?}", &logits_data[0..10]);
    println!("   Logits[450]: {:.6}", logits_data[450]);
    println!("   Logits[20780]: {:.6}\n", logits_data[20780]);

    // Manual calculation
    let weight_data = lm_weight.as_slice();
    let vocab_size = 32000;

    let manual_450: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 450])
        .sum();

    let manual_20780: f32 = hidden.iter()
        .enumerate()
        .map(|(i, &h)| h * weight_data[i * vocab_size + 20780])
        .sum();

    println!("🔢 Manual calculation:");
    println!("   Token 450: {:.6}", manual_450);
    println!("   Token 20780: {:.6}\n", manual_20780);

    println!("❓ Do they match?");
    println!("   450: matmul={:.6}, manual={:.6}, match={}",
        logits_data[450], manual_450, (logits_data[450] - manual_450).abs() < 0.001);
    println!("   20780: matmul={:.6}, manual={:.6}, match={}",
        logits_data[20780], manual_20780, (logits_data[20780] - manual_20780).abs() < 0.001);

    // Top predictions
    let mut indexed: Vec<(usize, f32)> = logits_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n📊 Top 10 tokens:");
    for (i, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token_id, logit);
    }

    Ok(())
}
