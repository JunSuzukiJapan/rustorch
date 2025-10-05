//! Example demonstrating GPT model loading from various formats
//! 各種形式からGPTモデルを読み込むデモ
//!
//! Supported formats:
//! - GGUF (automatically extracts config)
//! - Safetensors (requires manual config)
//! - MLX (requires manual config)

use rustorch::error::RusTorchResult;
use rustorch::models::{GPTConfig, GPTModel};

fn main() -> RusTorchResult<()> {
    // Check if model path is provided
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --example gpt_model_loading_demo <model_file>");
            eprintln!("\nSupported formats:");
            eprintln!("  - .gguf: cargo run --example gpt_model_loading_demo model.gguf");
            eprintln!("  - .safetensors: cargo run --example gpt_model_loading_demo model.safetensors");
            eprintln!("  - .mlx: cargo run --example gpt_model_loading_demo model.mlx");
            eprintln!("\nExample:");
            eprintln!("  cargo run --example gpt_model_loading_demo ~/.rustorch/models/TheBloke_Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf");
            std::process::exit(1);
        });

    println!("Loading GPT model from: {}", model_path);

    // Detect format from file extension
    let model = if model_path.ends_with(".gguf") {
        println!("Format: GGUF (auto-config)");
        GPTModel::from_gguf(&model_path)?
    } else if model_path.ends_with(".safetensors") {
        println!("Format: Safetensors (manual config)");
        // Create default config for Llama-2-7B
        let config = GPTConfig {
            vocab_size: 32000,
            d_model: 4096,
            num_layers: 32,
            num_heads: 32,
            d_ff: 11008,
            max_seq_len: 4096,
            dropout: 0.0,
        };
        GPTModel::from_safetensors(&model_path, config)?
    } else if model_path.ends_with(".mlx") {
        println!("Format: MLX (manual config)");
        // Create default config for Llama-2-7B
        let config = GPTConfig {
            vocab_size: 32000,
            d_model: 4096,
            num_layers: 32,
            num_heads: 32,
            d_ff: 11008,
            max_seq_len: 4096,
            dropout: 0.0,
        };
        GPTModel::from_mlx(&model_path, config)?
    } else {
        eprintln!("Error: Unsupported file format. Supported: .gguf, .safetensors, .mlx");
        std::process::exit(1);
    };

    // Display model configuration
    let config = model.config();
    println!("\n=== Model Configuration ===");
    println!("Vocabulary size: {}", config.vocab_size);
    println!("Model dimension: {}", config.d_model);
    println!("Number of layers: {}", config.num_layers);
    println!("Number of heads: {}", config.num_heads);
    println!("Feed-forward dimension: {}", config.d_ff);
    println!("Max sequence length: {}", config.max_seq_len);
    println!("Dropout: {}", config.dropout);

    // Display weight information
    let weight_names = model.weight_names();
    println!("\n=== Model Weights ===");
    println!("Total tensors loaded: {}", weight_names.len());

    if !weight_names.is_empty() {
        println!("\nFirst 10 tensors:");
        for (i, name) in weight_names.iter().take(10).enumerate() {
            if let Some(tensor) = model.get_weight(name) {
                println!("  {}: {} (shape: {:?})", i + 1, name, tensor.shape());
            }
        }

        if weight_names.len() > 10 {
            println!("  ... and {} more tensors", weight_names.len() - 10);
        }
    }

    // Test forward pass with dummy input
    println!("\n=== Testing Forward Pass ===");
    let input_ids = vec![1, 2, 3, 4, 5]; // Dummy token IDs
    let output = model.forward(&input_ids)?;
    println!("Input: {} tokens", input_ids.len());
    println!("Output shape: {:?}", output.shape());
    println!("Note: This is a placeholder forward pass. Full transformer implementation coming soon.");

    println!("\n✅ Model loaded successfully!");

    Ok(())
}
