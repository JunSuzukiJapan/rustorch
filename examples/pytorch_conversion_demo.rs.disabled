//! PyTorch to RusTorch automatic conversion demonstration
//! PyTorchからRusTorch自動変換のデモンストレーション

use rustorch::convert::{ModelParser, SimplePyTorchConverter};
use rustorch::formats::pytorch::{PyTorchModel, StateDict, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 PyTorch → RusTorch Automatic Conversion Demo");
    println!("==============================================");

    // 1. Create a sample PyTorch model
    let pytorch_model = create_sample_pytorch_model();

    // 2. Parse model architecture
    model_parsing_demo(&pytorch_model)?;

    // 3. Convert to RusTorch
    pytorch_conversion_demo(&pytorch_model)?;

    // 4. Test inference
    inference_demo(&pytorch_model)?;

    Ok(())
}

/// Create a sample PyTorch model for demonstration
/// デモ用のサンプルPyTorchモデルを作成
fn create_sample_pytorch_model() -> PyTorchModel {
    println!("📦 Creating Sample PyTorch Model");
    println!("---------------------------------");

    let mut state_dict = StateDict::new();

    // Convolutional layer (Conv2d)
    // Input channels: 3 (RGB), Output channels: 32, Kernel: 3x3
    state_dict.tensors.insert(
        "features.0.weight".to_string(),
        TensorData {
            shape: vec![32, 3, 3, 3],
            data: generate_random_data(32 * 3 * 3 * 3),
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "features.0.bias".to_string(),
        TensorData {
            shape: vec![32],
            data: generate_random_data(32),
            dtype: "f32".to_string(),
        },
    );

    // Batch Normalization
    state_dict.tensors.insert(
        "features.1.weight".to_string(),
        TensorData {
            shape: vec![32],
            data: vec![1.0; 32], // gamma
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "features.1.bias".to_string(),
        TensorData {
            shape: vec![32],
            data: vec![0.0; 32], // beta
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "features.1.running_mean".to_string(),
        TensorData {
            shape: vec![32],
            data: vec![0.0; 32],
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "features.1.running_var".to_string(),
        TensorData {
            shape: vec![32],
            data: vec![1.0; 32],
            dtype: "f32".to_string(),
        },
    );

    // Second Convolutional layer
    state_dict.tensors.insert(
        "features.3.weight".to_string(),
        TensorData {
            shape: vec![64, 32, 3, 3],
            data: generate_random_data(64 * 32 * 3 * 3),
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "features.3.bias".to_string(),
        TensorData {
            shape: vec![64],
            data: generate_random_data(64),
            dtype: "f32".to_string(),
        },
    );

    // Classifier (Linear layer)
    // Assuming input is flattened to 64 * 7 * 7 = 3136 features
    state_dict.tensors.insert(
        "classifier.weight".to_string(),
        TensorData {
            shape: vec![10, 3136],
            data: generate_random_data(10 * 3136),
            dtype: "f32".to_string(),
        },
    );
    state_dict.tensors.insert(
        "classifier.bias".to_string(),
        TensorData {
            shape: vec![10],
            data: generate_random_data(10),
            dtype: "f32".to_string(),
        },
    );

    let mut model = PyTorchModel::from_state_dict(state_dict);
    model.set_architecture(
        "CNN for CIFAR-10: Conv2d -> BatchNorm -> ReLU -> Conv2d -> AdaptiveAvgPool -> Linear"
            .to_string(),
    );

    println!(
        "✅ Created PyTorch model with {} layers",
        model.layer_names().len()
    );

    model
}

/// Generate random data for tensors
/// テンソル用のランダムデータを生成
fn generate_random_data(size: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Demonstrate model parsing capabilities
/// モデル解析機能をデモンストレーション
fn model_parsing_demo(pytorch_model: &PyTorchModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Model Architecture Parsing");
    println!("------------------------------");

    let parser = ModelParser::new();
    let graph = parser.parse_model(pytorch_model)?;

    println!("📊 Model Statistics:");
    println!("   - Total layers: {}", graph.layers.len());
    println!("   - Input layers: {:?}", graph.input_layers);
    println!("   - Output layers: {:?}", graph.output_layers);

    println!("\n🏗️ Layer Details:");
    for (layer_name, layer_info) in &graph.layers {
        println!("   Layer '{}': {:?}", layer_name, layer_info.layer_type);
        println!("     Parameters: {}", layer_info.num_parameters);
        if let Some(input_shape) = &layer_info.input_shape {
            println!("     Input shape: {:?}", input_shape);
        }
        if let Some(output_shape) = &layer_info.output_shape {
            println!("     Output shape: {:?}", output_shape);
        }
    }

    println!("\n🔗 Layer Connections:");
    for (from_layer, to_layers) in &graph.connections {
        for to_layer in to_layers {
            println!("   {} → {}", from_layer, to_layer);
        }
    }

    Ok(())
}

/// Demonstrate PyTorch to RusTorch conversion
/// PyTorchからRusTorch変換をデモンストレーション
fn pytorch_conversion_demo(pytorch_model: &PyTorchModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 PyTorch → RusTorch Conversion");
    println!("---------------------------------");

    let converted_model = SimplePyTorchConverter::convert(pytorch_model)?;

    println!("✅ Successfully converted PyTorch model to RusTorch");
    converted_model.print_summary();

    Ok(())
}

/// Demonstrate inference with converted model
/// 変換されたモデルでの推論をデモンストレーション
fn inference_demo(pytorch_model: &PyTorchModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Inference Simulation Demo");
    println!("----------------------------");

    // Convert model
    let converted_model = SimplePyTorchConverter::convert(pytorch_model)?;

    // Simulate inference with shape propagation
    let input_shape = vec![1, 3, 32, 32]; // Batch size 1, RGB, 32x32
    println!("📥 Input shape: {:?}", input_shape);

    match converted_model.simulate_forward(input_shape) {
        Ok(output_shape) => {
            println!("✅ Shape propagation successful!");
            println!("📤 Final output shape: {:?}", output_shape);
        }
        Err(e) => {
            println!("❌ Shape propagation failed: {}", e);
        }
    }

    // Show some converted tensors
    println!("\n🔍 Converted Tensor Samples:");
    for layer_name in converted_model.layer_names().into_iter().take(2) {
        if let Some(layer) = converted_model.get_layer(layer_name) {
            if let Some(weight_tensor) = layer.tensors.get("weight") {
                let sample_size = weight_tensor.data.len().min(5);
                let sample_data: Vec<f32> = weight_tensor
                    .data
                    .iter()
                    .take(sample_size)
                    .cloned()
                    .collect();
                println!(
                    "   {}.weight (first {} values): {:?}",
                    layer_name, sample_size, sample_data
                );
            }
        }
    }

    Ok(())
}

/// Demonstrate advanced conversion features
/// 高度な変換機能をデモンストレーション
fn _advanced_features_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Advanced Conversion Features");
    println!("--------------------------------");

    // Model optimization suggestions
    println!("🔧 Available Optimizations:");
    println!("   ✓ Layer fusion (Conv + BatchNorm + ReLU)");
    println!("   ✓ Memory layout optimization");
    println!("   ✓ SIMD vectorization for supported operations");
    println!("   ✓ GPU acceleration (when available)");

    // Compatibility notes
    println!("\n📋 Compatibility Status:");
    println!("   ✅ Linear layers");
    println!("   ✅ Conv2d layers");
    println!("   ✅ BatchNorm2d layers");
    println!("   ✅ ReLU activation");
    println!("   ✅ MaxPool2d/AvgPool2d layers");
    println!("   ✅ Dropout layers");
    println!("   ⏳ LSTM/GRU layers (planned)");
    println!("   ⏳ Transformer layers (planned)");
    println!("   ⏳ Custom layers (via trait implementation)");

    // Usage recommendations
    println!("\n💡 Usage Recommendations:");
    println!("   1. Test converted models thoroughly");
    println!("   2. Compare outputs with original PyTorch model");
    println!("   3. Use model validation utilities");
    println!("   4. Profile performance for production use");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_model_creation() {
        let model = create_sample_pytorch_model();
        assert!(!model.state_dict.tensors.is_empty());
        assert!(model.architecture.is_some());
    }

    #[test]
    fn test_model_parsing() {
        let model = create_sample_pytorch_model();
        let parser = ModelParser::new();
        let result = parser.parse_model(&model);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert!(!graph.layers.is_empty());
        assert!(!graph.execution_order.is_empty());
    }

    #[test]
    fn test_model_conversion() {
        let model = create_sample_pytorch_model();
        let result = SimplePyTorchConverter::convert(&model);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert!(!converted.layers.is_empty());
        assert!(!converted.execution_order.is_empty());
    }
}
