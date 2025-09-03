//! Model formats demonstration
//! モデルフォーマットのデモンストレーション

use rustorch::formats::pytorch::PyTorchModel;
use rustorch::tensor::Tensor;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Model Formats Demo");
    println!("================================");

    // 1. PyTorch Compatible Format
    pytorch_format_demo()?;

    // 2. Safetensors Format (if feature enabled)
    #[cfg(feature = "safetensors")]
    safetensors_format_demo()?;

    // 3. ONNX Format (if feature enabled)
    #[cfg(feature = "onnx")]
    onnx_format_demo()?;

    Ok(())
}

fn pytorch_format_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 PyTorch Compatible Format Demo");
    println!("-----------------------------------");

    // Create a simple model
    let mut model = PyTorchModel::new();

    // Add layers (simulating a simple neural network)
    let linear1_weights = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]);
    let linear1_bias = Tensor::<f32>::from_vec(vec![0.1, 0.2], vec![2]);

    let linear2_weights = Tensor::<f32>::from_vec(vec![0.7, 0.8, 0.9, 1.0], vec![2, 2]);
    let linear2_bias = Tensor::<f32>::from_vec(vec![0.3], vec![1]);

    // Set model layers
    model.set_layer_weights("linear1", &linear1_weights);
    model.set_layer_bias("linear1", &linear1_bias);
    model.set_layer_weights("linear2", &linear2_weights);
    model.set_layer_bias("output", &linear2_bias);

    model.set_architecture("Simple 2-layer Neural Network".to_string());

    println!("✅ Created model with {} layers", model.layer_names().len());

    // Display model statistics
    let stats = rustorch::formats::pytorch::utils::model_statistics(&model);
    println!("📊 Model Statistics:");
    println!("   - Total parameters: {}", stats["total_parameters"]);
    println!("   - Layer count: {}", stats["layer_count"]);

    // Validate model
    match rustorch::formats::pytorch::utils::validate_model(&model) {
        Ok(()) => println!("✅ Model validation passed"),
        Err(e) => println!("❌ Model validation failed: {}", e),
    }

    // Save model
    let temp_path = "temp_pytorch_model.json";
    model.save(temp_path)?;
    println!("💾 Saved model to {}", temp_path);

    // Load model back
    let loaded_model = PyTorchModel::load(temp_path)?;
    println!(
        "📂 Loaded model with {} layers",
        loaded_model.layer_names().len()
    );

    // Compare original and loaded weights
    let original_weights: Tensor<f32> = model.get_layer_weights("linear1").unwrap();
    let loaded_weights: Tensor<f32> = loaded_model.get_layer_weights("linear1").unwrap();

    if original_weights.data == loaded_weights.data {
        println!("✅ Weights match after save/load cycle");
    } else {
        println!("❌ Weights don't match after save/load cycle");
    }

    // Clean up
    std::fs::remove_file(temp_path).ok();

    Ok(())
}

#[cfg(feature = "safetensors")]
fn safetensors_format_demo() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::formats::safetensors::{SafetensorsLoader, SafetensorsSaver};

    println!("\n🔒 Safetensors Format Demo");
    println!("--------------------------");

    // Create tensors for safetensors demo
    let mut tensors = HashMap::new();

    let weights = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::<f32>::from_vec(vec![0.5, 1.0], vec![2]);

    tensors.insert("model.linear.weight".to_string(), weights);
    tensors.insert("model.linear.bias".to_string(), bias);

    println!("✅ Created {} tensors for safetensors demo", tensors.len());

    // Save to safetensors format
    let safetensors_path = "temp_model.safetensors";
    SafetensorsSaver::save_to_file(&tensors, safetensors_path)?;
    println!(
        "💾 Saved tensors to safetensors format: {}",
        safetensors_path
    );

    // Load from safetensors
    let loader = SafetensorsLoader::from_file(safetensors_path)?;
    let tensor_names = loader.tensor_names();
    println!(
        "📂 Found {} tensors in safetensors file",
        tensor_names.len()
    );

    for name in &tensor_names {
        println!("   - {}", name);
    }

    // Load specific tensor
    let loaded_weights: Tensor<f32> = loader.load_tensor("model.linear.weight")?;
    println!(
        "✅ Loaded tensor 'model.linear.weight' with shape: {:?}",
        loaded_weights.shape()
    );

    // Load all tensors
    let all_loaded: HashMap<String, Tensor<f32>> = loader.load_all_tensors()?;
    println!("✅ Loaded all {} tensors successfully", all_loaded.len());

    // Verify data integrity
    if let Some(original) = tensors.get("model.linear.weight") {
        if let Some(loaded) = all_loaded.get("model.linear.weight") {
            if original.data == loaded.data {
                println!("✅ Data integrity verified");
            } else {
                println!("❌ Data integrity check failed");
            }
        }
    }

    // Clean up
    std::fs::remove_file(safetensors_path).ok();

    Ok(())
}

#[cfg(feature = "onnx")]
fn onnx_format_demo() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::formats::onnx::utils;

    println!("\n🔄 ONNX Format Demo");
    println!("-------------------");

    // Show available ONNX execution providers
    let providers = utils::get_available_providers();
    println!("🔧 Available ONNX Execution Providers:");
    for (i, provider) in providers.iter().enumerate() {
        println!("   {}. {}", i + 1, provider);
    }

    println!("\n📝 ONNX functionality available:");
    println!("   ✅ Model loading from .onnx files");
    println!("   ✅ Inference with multiple execution providers");
    println!("   ✅ Batch processing support");
    println!("   ✅ Performance benchmarking");
    println!("   ⏳ Model export (planned for future version)");

    // Note: Actual ONNX model loading and inference would require
    // an ONNX model file. This demo shows the available functionality.

    println!("\n💡 To use ONNX inference:");
    println!("   1. Enable the 'onnx' feature: cargo run --features onnx");
    println!("   2. Place your .onnx model file in the project directory");
    println!("   3. Use OnnxModel::from_file() to load the model");
    println!("   4. Use model.run() or model.run_single() for inference");

    Ok(())
}

#[cfg(not(feature = "safetensors"))]
#[allow(dead_code)]
fn safetensors_format_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔒 Safetensors Format Demo");
    println!("--------------------------");
    println!("⚠️  Safetensors feature not enabled");
    println!("💡 To enable: cargo run --features safetensors --example model_formats_demo");
    Ok(())
}

#[cfg(not(feature = "onnx"))]
#[allow(dead_code)]
fn onnx_format_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 ONNX Format Demo");
    println!("-------------------");
    println!("⚠️  ONNX feature not enabled");
    println!("💡 To enable: cargo run --features onnx --example model_formats_demo");
    Ok(())
}
