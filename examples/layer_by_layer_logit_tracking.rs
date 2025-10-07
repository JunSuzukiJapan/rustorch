/// Track how logit predictions change layer by layer
///
/// For each transformer layer, calculate what the model would predict
/// if we stopped at that layer and went directly to LM head.
/// This helps identify which layers cause Token 1247 to dominate.

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;
use rustorch::formats::gguf::GGUFLoader;

fn main() -> F32Result<()> {
    println!("🔍 Layer-by-Layer Logit Tracking\n");
    println!("Goal: Find which layers cause Token 1247 to get high logit\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    // Load LM head weights
    println!("📂 Loading output.weight...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    let output_data: Vec<f64> = output_tensor.data.iter().cloned().collect();
    let hidden_size = 2048;
    let vocab_size = 32000;

    // Extract weight columns for tokens of interest
    let mut weights_450 = Vec::new();
    let mut weights_1247 = Vec::new();
    let mut weights_12711 = Vec::new();

    for dim in 0..hidden_size {
        weights_450.push(output_data[dim * vocab_size + 450] as f32);
        weights_1247.push(output_data[dim * vocab_size + 1247] as f32);
        weights_12711.push(output_data[dim * vocab_size + 12711] as f32);
    }

    println!("✅ LM head weights loaded\n");

    // Create model and run forward pass, saving intermediate states
    println!("🚀 Running forward pass...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // Input: BOS token (simplest case)
    let input = vec![1];
    let _output = model.forward(&input)?;

    println!("\n📊 Reading saved hidden states from /tmp/hidden_state_*.txt\n");
    println!("═══════════════════════════════════════");
    println!("Layer | Logit[450] | Logit[1247] | Logit[12711] | Winner");
    println!("═══════════════════════════════════════");

    // Layer 0: After embedding
    if let Ok(hidden_0) = read_hidden_state("/tmp/hidden_state_layer_0.txt", hidden_size) {
        let logit_450 = dot_product(&hidden_0, &weights_450);
        let logit_1247 = dot_product(&hidden_0, &weights_1247);
        let logit_12711 = dot_product(&hidden_0, &weights_12711);
        let winner = find_winner(logit_450, logit_1247, logit_12711);
        println!("{:5} | {:10.4} | {:11.4} | {:12.4} | {}",
                 0, logit_450, logit_1247, logit_12711, winner);
    }

    // Layers 1-21
    for layer in 1..=21 {
        let filename = format!("/tmp/hidden_state_layer_{}.txt", layer);
        if let Ok(hidden) = read_hidden_state(&filename, hidden_size) {
            let logit_450 = dot_product(&hidden, &weights_450);
            let logit_1247 = dot_product(&hidden, &weights_1247);
            let logit_12711 = dot_product(&hidden, &weights_12711);
            let winner = find_winner(logit_450, logit_1247, logit_12711);
            println!("{:5} | {:10.4} | {:11.4} | {:12.4} | {}",
                     layer, logit_450, logit_1247, logit_12711, winner);
        }
    }

    println!("═══════════════════════════════════════\n");
    println!("Note: These hidden states are saved by llama.rs debug logging.");
    println!("      Make sure RUST_LOG=debug is set when running the model.");

    Ok(())
}

fn read_hidden_state(filename: &str, expected_size: usize) -> Result<Vec<f32>, std::io::Error> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(val) = line.trim().parse::<f32>() {
            values.push(val);
        }
    }

    if values.len() != expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected {} values, got {}", expected_size, values.len())
        ));
    }

    Ok(values)
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn find_winner(logit_450: f32, logit_1247: f32, logit_12711: f32) -> &'static str {
    let max = logit_450.max(logit_1247).max(logit_12711);
    if (logit_450 - max).abs() < 1e-6 {
        "Token 450"
    } else if (logit_1247 - max).abs() < 1e-6 {
        "Token 1247 ❌"
    } else {
        "Token 12711 ✅"
    }
}
