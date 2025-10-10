/// Demonstration of batch inference with Llama model
///
/// This example shows how to use the forward_batch API to process
/// multiple sequences in parallel (currently implemented as sequential
/// processing until Metal kernels support batch dimension).
///
/// Usage:
///   cargo run --example batch_inference_demo --features metal

use rustorch::models::llama::LlamaModel;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦙 Llama Batch Inference Demo\n");

    // Model path (adjust as needed)
    let model_path = std::env::var("RUSTORCH_MODEL_PATH")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            format!("{home}/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
        });

    println!("📁 Loading model from: {}", model_path);

    if !Path::new(&model_path).exists() {
        eprintln!("❌ Model file not found: {}", model_path);
        eprintln!("   Set RUSTORCH_MODEL_PATH environment variable or place model in default location");
        return Err("Model file not found".into());
    }

    // Load model with Metal backend (if available)
    #[cfg(feature = "metal")]
    let mut model = {
        println!("🔧 Loading model with Metal backend...");
        LlamaModel::from_gguf_with_backend(&model_path, rustorch::backends::DeviceType::Metal)?
    };

    #[cfg(not(feature = "metal"))]
    let mut model = {
        println!("🔧 Loading model with CPU backend...");
        LlamaModel::from_gguf(&model_path)?
    };

    println!("✅ Model loaded successfully!");
    println!("📊 Config: batch_size={}, hidden={}, layers={}, heads={}",
             model.config.batch_size, model.config.hidden_size,
             model.config.num_layers, model.config.num_heads);
    println!();

    // Prepare batch of input sequences
    // These are token IDs - in real usage, you would tokenize text first
    let input_batch: Vec<&[usize]> = vec![
        &[1],      // Single token: BOS
        &[1, 2],   // Two tokens
        &[1, 2, 3], // Three tokens (note: will be processed but may not work correctly due to current limitations)
    ];

    println!("📝 Processing batch of {} sequences:", input_batch.len());
    for (i, seq) in input_batch.iter().enumerate() {
        println!("   Sequence {}: {:?}", i, seq);
    }

    // Perform batch inference
    println!("\n🚀 Running batch inference...");
    match model.forward_batch(&input_batch) {
        Ok(outputs) => {
            println!("✅ Batch inference completed!");
            println!("   Generated {} output tensors", outputs.len());

            for (i, output) in outputs.iter().enumerate() {
                println!("\n   Output {}:", i);
                println!("     Shape: {:?}", output.shape());
                println!("     Total elements: {}", output.shape().iter().product::<usize>());
            }
        }
        Err(e) => {
            eprintln!("❌ Batch inference failed: {}", e);
            return Err(e.into());
        }
    }

    println!("\n📊 Performance Notes:");
    println!("   - Current implementation processes sequences individually");
    println!("   - True parallel batch processing requires Metal kernel updates");
    println!("   - Batch size is configurable via config.batch_size");
    println!("   - KVCache is allocated per batch item");

    println!("\n✅ Demo completed successfully!");

    Ok(())
}
