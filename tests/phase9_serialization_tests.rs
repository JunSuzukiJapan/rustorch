//! Comprehensive tests for Phase 9 serialization system
//! フェーズ9シリアライゼーションシステム包括テスト

use rustorch::{
    autograd::Variable,
    error::RusTorchError,
    serialization::{core::*, formats::*, jit::*, model_io::*},
    tensor::Tensor,
};
use std::collections::HashMap;
use std::path::Path;

/// Test tensor serialization and deserialization
/// テンソルのシリアライゼーション・デシリアライゼーションをテスト
#[test]
fn test_tensor_save_load_roundtrip() {
    let original_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let temp_path = std::env::temp_dir().join("test_tensor.rustorch");

    // Save tensor
    save(&original_tensor, &temp_path).expect("Failed to save tensor");

    // Load tensor
    let loaded_tensor: Tensor<f32> = load(&temp_path).expect("Failed to load tensor");

    // Verify data integrity
    assert_eq!(original_tensor.shape(), loaded_tensor.shape());
    assert_eq!(
        original_tensor.data.as_slice(),
        loaded_tensor.data.as_slice()
    );

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test state dict functionality
/// StateDict機能をテスト
#[test]
fn test_state_dict_operations() {
    let mut state_dict = StateDict::<f32>::new();

    // Add parameters
    let weight = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    let bias = Tensor::from_vec(vec![0.1, 0.2], vec![2]);

    state_dict.add_parameter("layer.weight".to_string(), weight);
    state_dict.add_parameter("layer.bias".to_string(), bias);

    // Add buffers
    let running_mean = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
    state_dict.add_buffer("layer.running_mean".to_string(), running_mean);

    // Test retrieval
    assert!(state_dict.get_parameter("layer.weight").is_some());
    assert!(state_dict.get_parameter("layer.bias").is_some());
    assert!(state_dict.get_buffer("layer.running_mean").is_some());
    assert!(state_dict.get_parameter("nonexistent").is_none());

    // Test training state
    assert!(!state_dict.is_training());
    state_dict.set_training(true);
    assert!(state_dict.is_training());
}

/// Test model checkpoint system
/// モデルチェックポイントシステムをテスト
#[test]
fn test_model_checkpoint_system() {
    let mut state_dict = StateDict::<f32>::new();
    let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    state_dict.add_parameter("fc.weight".to_string(), weight);

    let mut checkpoint = ModelCheckpoint::new(10, 1000, state_dict);
    checkpoint.add_metric("loss".to_string(), 0.5);
    checkpoint.add_metric("accuracy".to_string(), 0.95);

    let temp_path = std::env::temp_dir().join("test_checkpoint.rustorch");

    // Save checkpoint
    checkpoint
        .save_checkpoint(&temp_path)
        .expect("Failed to save checkpoint");

    // Load checkpoint
    let loaded_checkpoint =
        ModelCheckpoint::<f32>::load_checkpoint(&temp_path).expect("Failed to load checkpoint");

    // Verify checkpoint data
    assert_eq!(loaded_checkpoint.epoch, 10);
    assert_eq!(loaded_checkpoint.step, 1000);
    assert!(loaded_checkpoint
        .model_state
        .get_parameter("fc.weight")
        .is_some());
    assert_eq!(loaded_checkpoint.metrics.get("loss").unwrap(), &0.5);

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test JIT script module functionality
/// JITスクリプトモジュール機能をテスト
#[test]
fn test_jit_script_module() {
    // Create a simple script module
    let module = script(|inputs: &[Tensor<f32>]| {
        if inputs.is_empty() {
            vec![]
        } else {
            let mut result = inputs[0].clone();
            result.data.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
            vec![result]
        }
    })
    .expect("Failed to create script module");

    // Test basic properties
    assert_eq!(module.type_id(), "script_module");
    assert!(!module.graph.nodes.is_empty());

    // Test save/load
    let temp_path = std::env::temp_dir().join("test_script_module.rustorch");
    save(&module, &temp_path).expect("Failed to save script module");

    let loaded_module: ScriptModule<f32> = load(&temp_path).expect("Failed to load script module");

    assert_eq!(loaded_module.graph.nodes.len(), module.graph.nodes.len());

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test JIT trace functionality
/// JITトレース機能をテスト
#[test]
fn test_jit_trace_functionality() {
    let example_input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], vec![4]);
    let inputs = vec![example_input];

    // Trace a simple function
    let traced_module = trace(
        |inputs: &[Tensor<f32>]| {
            if inputs.is_empty() {
                vec![]
            } else {
                let mut result = inputs[0].clone();
                result.data.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                vec![result]
            }
        },
        &inputs,
    )
    .expect("Failed to trace function");

    // Verify trace results
    assert_eq!(traced_module.type_id(), "script_module");
    assert!(!traced_module.graph.inputs.is_empty());
    assert!(!traced_module.graph.outputs.is_empty());

    // Test execution
    let output = traced_module
        .forward(&inputs)
        .expect("Failed to execute traced module");
    assert!(!output.is_empty());
}

/// Test SafeTensors format
/// SafeTensors形式をテスト
#[test]
fn test_safetensors_format() {
    let mut tensors = HashMap::new();

    let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let bias = Tensor::from_vec(vec![0.1f32, 0.2], vec![2]);

    tensors.insert("model.weight".to_string(), weight);
    tensors.insert("model.bias".to_string(), bias);

    let temp_path = std::env::temp_dir().join("test_model.safetensors");

    // Save in SafeTensors format
    SafeTensorsFormat::save_safetensors(&tensors, &temp_path).expect("Failed to save SafeTensors");

    // Load SafeTensors format
    let loaded_tensors = SafeTensorsFormat::load_safetensors::<_, f32>(&temp_path)
        .expect("Failed to load SafeTensors");

    // Verify data
    assert!(loaded_tensors.contains_key("model.weight"));
    assert!(loaded_tensors.contains_key("model.bias"));

    let loaded_weight = loaded_tensors.get("model.weight").unwrap();
    assert_eq!(loaded_weight.shape(), &[2, 2]);

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test HuggingFace format
/// HuggingFace形式をテスト
#[test]
fn test_huggingface_format() {
    let mut tensors = HashMap::new();
    let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    tensors.insert("embeddings.weight".to_string(), weight);

    let mut config = HashMap::new();
    config.insert("model_type".to_string(), "bert".to_string());
    config.insert("vocab_size".to_string(), "30000".to_string());
    config.insert("hidden_size".to_string(), "768".to_string());

    let temp_dir = std::env::temp_dir().join("test_hf_model");

    // Save in HuggingFace format
    HuggingFaceFormat::save_hf(&tensors, &config, &temp_dir)
        .expect("Failed to save HuggingFace format");

    // Load HuggingFace format
    let (loaded_tensors, loaded_config) =
        HuggingFaceFormat::load_hf::<_, f32>(&temp_dir).expect("Failed to load HuggingFace format");

    // Verify data
    assert!(loaded_tensors.contains_key("embeddings.weight"));
    assert_eq!(loaded_config.get("model_type").unwrap(), "bert");
    assert_eq!(loaded_config.get("vocab_size").unwrap(), "30000");

    // Cleanup
    std::fs::remove_dir_all(temp_dir).ok();
}

/// Test format detection
/// 形式検出をテスト
#[test]
fn test_format_detection() {
    // Create test files with different formats
    let temp_dir = std::env::temp_dir().join("format_test");
    std::fs::create_dir_all(&temp_dir).ok();

    // Create RusTorch format file
    let rustorch_path = temp_dir.join("test.rustorch");
    let tensor = Tensor::from_vec(vec![1.0f32], vec![1]);
    save(&tensor, &rustorch_path).expect("Failed to save RusTorch format");

    // Test format detection
    let detected_format = detect_format(&rustorch_path).expect("Failed to detect format");
    assert_eq!(detected_format, "rustorch");

    // Cleanup
    std::fs::remove_dir_all(temp_dir).ok();
}

/// Test JIT cache system
/// JITキャッシュシステムをテスト
#[test]
fn test_jit_cache_system() {
    let cache_dir = std::env::temp_dir().join("jit_cache_test");
    let mut cache = JitCache::<f32>::new(&cache_dir).expect("Failed to create JIT cache");

    // Create a module
    let mut module = ScriptModule::<f32>::new();
    let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let variable = Variable::new(tensor, false);
    module.add_parameter("test_param".to_string(), variable);

    // Store in cache
    cache
        .store("test_module".to_string(), module)
        .expect("Failed to store module in cache");

    // Retrieve from cache
    let cached_module = cache
        .get("test_module")
        .expect("Failed to retrieve module from cache");
    assert!(cached_module.parameters.contains_key("test_param"));

    // Test cache persistence
    let mut new_cache =
        JitCache::<f32>::new(&cache_dir).expect("Failed to create new cache instance");
    let persisted_module = new_cache
        .get("test_module")
        .expect("Failed to retrieve persisted module");
    assert!(persisted_module.parameters.contains_key("test_param"));

    // Cleanup
    cache.clear().ok();
    std::fs::remove_dir_all(cache_dir).ok();
}

/// Test serialization error handling
/// シリアライゼーションエラーハンドリングをテスト
#[test]
fn test_serialization_error_handling() {
    // Test loading non-existent file
    let result = load::<_, Tensor<f32>>(Path::new("nonexistent.rustorch"));
    assert!(result.is_err());

    // Test type mismatch
    let tensor = Tensor::from_vec(vec![1.0f32], vec![1]);
    let temp_path = std::env::temp_dir().join("type_test.rustorch");
    save(&tensor, &temp_path).expect("Failed to save tensor");

    // Try to load as different type (this should work with our current implementation,
    // but in a stricter implementation it might fail)
    let result = load::<_, StateDict<f32>>(&temp_path);
    assert!(result.is_err());

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test computation graph validation
/// 計算グラフ検証をテスト
#[test]
fn test_computation_graph_validation() {
    let mut graph = ComputationGraph::<f32>::new();

    // Add valid nodes
    let node1 = GraphNode {
        id: 0,
        op_type: "add".to_string(),
        inputs: vec![],
        outputs: vec![0],
        attributes: HashMap::new(),
    };

    let node2 = GraphNode {
        id: 1,
        op_type: "relu".to_string(),
        inputs: vec![0],
        outputs: vec![1],
        attributes: HashMap::new(),
    };

    graph.add_node(node1);
    graph.add_node(node2);

    // Valid graph should pass validation
    assert!(graph.validate().is_ok());

    // Add invalid node with out-of-bounds input
    let invalid_node = GraphNode {
        id: 2,
        op_type: "mul".to_string(),
        inputs: vec![99], // Invalid input ID
        outputs: vec![2],
        attributes: HashMap::new(),
    };

    graph.add_node(invalid_node);

    // Invalid graph should fail validation
    assert!(graph.validate().is_err());
}

/// Test precision conversion
/// 精度変換をテスト
#[test]
fn test_precision_conversion() {
    let mut tensors_f32 = HashMap::new();

    let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.5, 3.7, -1.2], vec![2, 2]);
    tensors_f32.insert("test_tensor".to_string(), tensor_f32);

    // Convert f32 to f64
    let tensors_f64: HashMap<String, Tensor<f64>> = conversion::convert_precision(&tensors_f32);

    assert!(tensors_f64.contains_key("test_tensor"));

    let converted_tensor = tensors_f64.get("test_tensor").unwrap();
    assert_eq!(converted_tensor.shape(), &[2, 2]);

    // Verify data conversion
    if let Some(data) = converted_tensor.data.as_slice() {
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 2.5).abs() < 1e-9);
        assert!((data[2] - 3.7).abs() < 1e-9);
        assert!((data[3] - (-1.2)).abs() < 1e-9);
    }
}

/// Test safe tensor format with large tensors
/// 大きなテンソルでセーフテンソル形式をテスト
#[test]
fn test_large_tensor_serialization() {
    let mut safe_format = SafeTensorFormat::<f32>::new();

    // Create a larger tensor
    let size = 1000;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
    let large_tensor = Tensor::from_vec(data, vec![size]);

    safe_format.add_tensor("large_tensor".to_string(), large_tensor);
    safe_format
        .metadata
        .insert("description".to_string(), "test large tensor".to_string());

    let temp_path = std::env::temp_dir().join("large_tensor.safetensors");

    // Save and verify
    safe_format
        .save_safetensors(&temp_path)
        .expect("Failed to save large tensor");

    // Verify file exists and has reasonable size
    let metadata = std::fs::metadata(&temp_path).expect("Failed to get file metadata");
    assert!(metadata.len() > 4000); // Should be at least 4KB for 1000 floats

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

/// Test graph execution with basic operations
/// 基本操作での計算グラフ実行をテスト
#[test]
fn test_graph_execution() {
    let mut module = ScriptModule::<f32>::new();

    // Create a simple computation: input -> relu -> output
    let node = GraphNode {
        id: 0,
        op_type: "relu".to_string(),
        inputs: vec![0],
        outputs: vec![1],
        attributes: HashMap::new(),
    };

    module.graph.add_node(node);
    module.graph.inputs.push("input_0".to_string());
    module.graph.outputs.push("1".to_string());

    // Test execution with negative input
    let input_tensor = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, -2.0], vec![4]);
    let outputs = module
        .forward(&[input_tensor])
        .expect("Failed to execute graph");

    assert!(!outputs.is_empty());

    // Verify ReLU operation (negative values should become 0)
    if let Some(data) = outputs[0].data.as_slice() {
        assert_eq!(data[0], 0.0); // -1.0 -> 0.0
        assert_eq!(data[1], 0.0); // 0.0 -> 0.0
        assert_eq!(data[2], 1.0); // 1.0 -> 1.0
        assert_eq!(data[3], 0.0); // -2.0 -> 0.0
    }
}

/// Test checksum validation
/// チェックサム検証をテスト
#[test]
fn test_checksum_validation() {
    let data1 = b"test data for checksum";
    let data2 = b"different test data";

    let checksum1 = compute_checksum(data1);
    let checksum2 = compute_checksum(data1);
    let checksum3 = compute_checksum(data2);

    // Same data should produce same checksum
    assert_eq!(checksum1, checksum2);

    // Different data should produce different checksum
    assert_ne!(checksum1, checksum3);

    // Empty data should produce consistent checksum
    let empty_checksum1 = compute_checksum(&[]);
    let empty_checksum2 = compute_checksum(&[]);
    assert_eq!(empty_checksum1, empty_checksum2);
}

/// Test error conversion and compatibility
/// エラー変換と互換性をテスト
#[test]
fn test_error_conversion() {
    // Test SerializationError to RusTorchError conversion
    let ser_error = SerializationError::IoError("test error".to_string());
    let rust_error: RusTorchError = ser_error.into();

    match rust_error {
        RusTorchError::SerializationError { operation, message } => {
            assert_eq!(operation, "serialization");
            assert!(message.contains("test error"));
        }
        _ => panic!("Expected SerializationError"),
    }

    // Test std::io::Error to SerializationError conversion
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let ser_error: SerializationError = io_error.into();

    match ser_error {
        SerializationError::IoError(msg) => {
            assert!(msg.contains("file not found"));
        }
        _ => panic!("Expected IoError"),
    }
}
