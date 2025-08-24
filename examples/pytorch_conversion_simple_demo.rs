//! Simple PyTorch to RusTorch conversion demonstration (working version)
//! PyTorchからRusTorch変換のシンプルデモンストレーション（動作版）

use rustorch::formats::pytorch::{PyTorchModel, StateDict, TensorData};
use rustorch::convert::{SimplePyTorchConverter, ModelParser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 PyTorch → RusTorch Simple Conversion Demo");
    println!("============================================");
    
    // 1. Create a sample PyTorch model
    let pytorch_model = create_sample_pytorch_model();
    
    // 2. Parse model architecture
    model_parsing_demo(&pytorch_model)?;
    
    // 3. Convert to simplified RusTorch representation
    conversion_demo(&pytorch_model)?;
    
    Ok(())
}

/// Create a sample PyTorch model for demonstration
/// デモ用のサンプルPyTorchモデルを作成
fn create_sample_pytorch_model() -> PyTorchModel {
    println!("📦 Creating Sample PyTorch Model");
    println!("---------------------------------");
    
    let mut state_dict = StateDict::new();
    
    // First linear layer (features -> hidden)
    state_dict.tensors.insert("features.0.weight".to_string(), TensorData {
        shape: vec![128, 784], // 28*28 MNIST input -> 128 hidden
        data: generate_random_data(128 * 784),
        dtype: "f32".to_string(),
    });
    state_dict.tensors.insert("features.0.bias".to_string(), TensorData {
        shape: vec![128],
        data: generate_random_data(128),
        dtype: "f32".to_string(),
    });
    
    // Batch normalization
    state_dict.tensors.insert("features.1.weight".to_string(), TensorData {
        shape: vec![128],
        data: vec![1.0; 128], // gamma (scale)
        dtype: "f32".to_string(),
    });
    state_dict.tensors.insert("features.1.bias".to_string(), TensorData {
        shape: vec![128],
        data: vec![0.0; 128], // beta (shift)
        dtype: "f32".to_string(),
    });
    state_dict.tensors.insert("features.1.running_mean".to_string(), TensorData {
        shape: vec![128],
        data: vec![0.0; 128],
        dtype: "f32".to_string(),
    });
    state_dict.tensors.insert("features.1.running_var".to_string(), TensorData {
        shape: vec![128],
        data: vec![1.0; 128],
        dtype: "f32".to_string(),
    });
    
    // Second linear layer (hidden -> output)
    state_dict.tensors.insert("classifier.weight".to_string(), TensorData {
        shape: vec![10, 128], // 128 hidden -> 10 classes
        data: generate_random_data(10 * 128),
        dtype: "f32".to_string(),
    });
    state_dict.tensors.insert("classifier.bias".to_string(), TensorData {
        shape: vec![10],
        data: generate_random_data(10),
        dtype: "f32".to_string(),
    });
    
    let mut model = PyTorchModel::from_state_dict(state_dict);
    model.set_architecture("Simple MLP: Linear(784->128) -> BatchNorm -> ReLU -> Linear(128->10)".to_string());
    
    println!("✅ Created PyTorch model with {} parameter tensors", model.layer_names().len());
    
    model
}

/// Generate random data for tensors
/// テンソル用のランダムデータを生成
fn generate_random_data(size: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
}

/// Demonstrate model parsing capabilities
/// モデル解析機能をデモンストレーション
fn model_parsing_demo(pytorch_model: &PyTorchModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n🔍 Model Architecture Analysis");
    println!("-------------------------------");
    
    let parser = ModelParser::new();
    match parser.parse_model(pytorch_model) {
        Ok(graph) => {
            println!("📊 Model Statistics:");
            println!("   - Total layers: {}", graph.layers.len());
            println!("   - Input layers: {:?}", graph.input_layers);
            println!("   - Output layers: {:?}", graph.output_layers);
            
            println!("\\n🏗️ Layer Analysis:");
            for (layer_name, layer_info) in &graph.layers {
                println!("   📦 {}: {:?}", layer_name, layer_info.layer_type);
                println!("      Parameters: {}", layer_info.num_parameters);
                if let Some(input_shape) = &layer_info.input_shape {
                    println!("      Input: {:?}", input_shape);
                }
                if let Some(output_shape) = &layer_info.output_shape {
                    println!("      Output: {:?}", output_shape);
                }
                println!();
            }
        },
        Err(e) => {
            println!("⚠️  Model parsing failed: {}", e);
            println!("💡 Using simplified conversion approach instead");
        }
    }
    
    Ok(())
}

/// Demonstrate PyTorch to RusTorch conversion
/// PyTorchからRusTorch変換をデモンストレーション
fn conversion_demo(pytorch_model: &PyTorchModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n🔄 PyTorch → RusTorch Conversion");
    println!("---------------------------------");
    
    let converted_model = SimplePyTorchConverter::convert(pytorch_model)?;
    
    println!("✅ Conversion successful!");
    converted_model.print_summary();
    
    // Demonstrate shape inference
    println!("\\n🧠 Shape Inference Demo");
    println!("-------------------------");
    
    let input_shape = vec![1, 784]; // Batch size 1, flattened MNIST image
    println!("📥 Input shape: {:?}", input_shape);
    
    match converted_model.simulate_forward(input_shape) {
        Ok(output_shape) => {
            println!("✅ Shape propagation successful!");
            println!("📤 Final output shape: {:?}", output_shape);
        },
        Err(e) => {
            println!("❌ Shape propagation failed: {}", e);
        }
    }
    
    // Show tensor samples
    println!("\\n🔍 Converted Tensor Samples:");
    for layer_name in converted_model.layer_names().into_iter().take(3) {
        if let Some(layer) = converted_model.get_layer(layer_name) {
            println!("   📊 Layer: {}", layer_name);
            
            for (param_name, tensor) in &layer.tensors {
                let sample_size = tensor.data.len().min(5);
                let mut sample_data = Vec::new();
                for i in 0..sample_size {
                    sample_data.push(tensor.data[i]);
                }
                println!("      {} {:?}: {:?}...", param_name, tensor.shape(), sample_data);
            }
            println!();
        }
    }
    
    // Demonstrate usage information
    println!("\\n💡 Usage Notes:");
    println!("================");
    println!("✅ Successfully converted {} layers", converted_model.layers.len());
    println!("✅ Total parameters: {}", converted_model.total_parameters);
    println!("✅ All tensors converted to RusTorch format");
    println!("✅ Layer topology preserved");
    println!("\\n📝 Next Steps:");
    println!("   1. Use converted tensors for RusTorch neural network construction");
    println!("   2. Implement actual forward pass with RusTorch layers");
    println!("   3. Add support for more complex architectures");
    println!("   4. Validate outputs against original PyTorch model");
    
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
        assert!(converted.total_parameters > 0);
    }
    
    #[test] 
    fn test_shape_inference() {
        let model = create_sample_pytorch_model();
        let converted = SimplePyTorchConverter::convert(&model).unwrap();
        
        let input_shape = vec![1, 784];
        let result = converted.simulate_forward(input_shape);
        assert!(result.is_ok());
        
        let output_shape = result.unwrap();
        assert_eq!(output_shape, vec![1, 10]); // Should end up as [batch, 10_classes]
    }
}