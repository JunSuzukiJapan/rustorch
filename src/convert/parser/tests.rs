//! Tests for model parser functionality
//! モデルパーサー機能のテスト

#[cfg(test)]
mod tests {
    use super::super::core::ModelParser;
    use super::super::formats::*;
    use super::super::types::*;
    use crate::formats::pytorch::{PyTorchModel, StateDict, TensorData};
    use std::collections::HashMap;

    fn create_test_model() -> PyTorchModel {
        let mut state_dict = StateDict::new();

        // Conv layer
        state_dict.tensors.insert(
            "features.0.weight".to_string(),
            TensorData {
                shape: vec![32, 3, 3, 3],
                data: vec![0.1; 864],
                dtype: "f32".to_string(),
            },
        );
        state_dict.tensors.insert(
            "features.0.bias".to_string(),
            TensorData {
                shape: vec![32],
                data: vec![0.1; 32],
                dtype: "f32".to_string(),
            },
        );

        // Linear layer - compatible dimension with conv output (32*32*32 = 32768)
        state_dict.tensors.insert(
            "classifier.weight".to_string(),
            TensorData {
                shape: vec![10, 32768],
                data: vec![0.1; 327680],
                dtype: "f32".to_string(),
            },
        );
        state_dict.tensors.insert(
            "classifier.bias".to_string(),
            TensorData {
                shape: vec![10],
                data: vec![0.1; 10],
                dtype: "f32".to_string(),
            },
        );

        crate::formats::pytorch::PyTorchModel::from_state_dict(state_dict)
    }

    #[test]
    fn test_layer_extraction() {
        let parser = ModelParser::new();
        let model = create_test_model();

        let layers = parser.extract_layers(&model.state_dict).unwrap();

        assert_eq!(layers.len(), 2);
        assert!(layers.contains_key("features.0"));
        assert!(layers.contains_key("classifier"));

        // Check conv layer
        let conv_layer = &layers["features.0"];
        assert!(matches!(conv_layer.layer_type, LayerType::Conv2d { .. }));

        // Check linear layer
        let linear_layer = &layers["classifier"];
        assert!(matches!(linear_layer.layer_type, LayerType::Linear { .. }));
    }

    #[test]
    fn test_model_parsing() {
        let parser = ModelParser::new();
        let model = create_test_model();

        // Test that we can extract layers correctly
        let layers = parser.extract_layers(&model.state_dict).unwrap();
        assert_eq!(layers.len(), 2);
        assert!(layers.contains_key("features.0"));
        assert!(layers.contains_key("classifier"));

        // Test layer types
        assert!(matches!(
            layers["features.0"].layer_type,
            LayerType::Conv2d { .. }
        ));
        assert!(matches!(
            layers["classifier"].layer_type,
            LayerType::Linear { .. }
        ));
    }

    #[test]
    fn test_parameter_name_parsing() {
        let parser = ModelParser::new();

        let (layer_name, param_type) = parser.parse_parameter_name("features.0.weight").unwrap();
        assert_eq!(layer_name, "features.0");
        assert_eq!(param_type, "weight");

        let (layer_name, param_type) = parser.parse_parameter_name("classifier.bias").unwrap();
        assert_eq!(layer_name, "classifier");
        assert_eq!(param_type, "bias");
    }

    #[test]
    fn test_layer_type_inference() {
        let parser = ModelParser::new();
        let mut params: HashMap<String, &TensorData> = HashMap::new();

        // Test Conv2d inference
        let conv_weight = TensorData {
            shape: vec![32, 3, 3, 3],
            data: vec![0.1; 864],
            dtype: "f32".to_string(),
        };
        params.insert("weight".to_string(), &conv_weight);

        let layer_type = parser.infer_layer_type("conv1", &params).unwrap();
        assert!(matches!(layer_type, LayerType::Conv2d { .. }));

        // Test Conv3d inference
        let conv3d_weight = TensorData {
            shape: vec![16, 8, 3, 3, 3],
            data: vec![0.1; 3456],
            dtype: "f32".to_string(),
        };
        params.clear();
        params.insert("weight".to_string(), &conv3d_weight);

        let layer_type = parser.infer_layer_type("conv3d", &params).unwrap();
        assert!(matches!(layer_type, LayerType::Conv3d { .. }));

        // Test Linear inference
        let linear_weight = TensorData {
            shape: vec![10, 512],
            data: vec![0.1; 5120],
            dtype: "f32".to_string(),
        };
        params.clear();
        params.insert("weight".to_string(), &linear_weight);

        let layer_type = parser.infer_layer_type("fc", &params).unwrap();
        assert!(matches!(layer_type, LayerType::Linear { .. }));
    }

    #[test]
    fn test_simple_architecture_parsing() {
        let parser = ModelParser::new();
        let architecture = "conv2d -> relu -> maxpool -> flatten -> linear";

        let desc = parser.parse_simple_format(architecture).unwrap();

        assert_eq!(desc.layers.len(), 5);
        assert_eq!(desc.connections.len(), 4);
        assert_eq!(desc.layers[0].layer_type, "conv2d");
        assert_eq!(desc.layers[4].layer_type, "linear");

        // Check connections
        assert_eq!(desc.connections[0].from, "layer_0");
        assert_eq!(desc.connections[0].to, "layer_1");
    }

    #[test]
    fn test_json_architecture_parsing() {
        let parser = ModelParser::new();
        let json_arch = r#"
        {
            "metadata": {
                "name": "test_model",
                "framework": "pytorch"
            },
            "layers": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "params": {"in_channels": 3, "out_channels": 32, "kernel_size": [3, 3]}
                },
                {
                    "name": "relu1", 
                    "type": "ReLU"
                }
            ],
            "connections": [
                {"from": "conv1", "to": "relu1"}
            ]
        }"#;

        let desc = parser.parse_architecture_string(json_arch).unwrap();
        assert_eq!(desc.layers.len(), 2);
        assert_eq!(desc.connections.len(), 1);
        assert_eq!(desc.metadata.name, "test_model");
    }

    #[test]
    fn test_yaml_architecture_parsing() {
        let parser = ModelParser::new();
        let yaml_arch = r#"
        metadata:
          name: test_model
          framework: pytorch
        layers:
          - name: conv3d1
            type: Conv3d
            params:
              in_channels: 3
              out_channels: 16
              kernel_size: [3, 3, 3]
          - name: relu1
            type: ReLU
        connections:
          - from: conv3d1
            to: relu1
        "#;

        let desc = parser.parse_architecture_string(yaml_arch).unwrap();
        assert_eq!(desc.layers.len(), 2);
        assert_eq!(desc.connections.len(), 1);
        assert_eq!(desc.layers[0].layer_type, "Conv3d");
    }

    #[test]
    fn test_execution_order_computation() {
        let parser = ModelParser::new();

        let desc = ArchitectureDescription {
            metadata: ModelMetadata {
                name: "test".to_string(),
                version: None,
                framework: None,
                description: None,
            },
            layers: vec![
                LayerDefinition {
                    name: "input".to_string(),
                    layer_type: "Conv2d".to_string(),
                    params: None,
                    input_shape: None,
                    output_shape: None,
                },
                LayerDefinition {
                    name: "hidden".to_string(),
                    layer_type: "ReLU".to_string(),
                    params: None,
                    input_shape: None,
                    output_shape: None,
                },
                LayerDefinition {
                    name: "output".to_string(),
                    layer_type: "Linear".to_string(),
                    params: None,
                    input_shape: None,
                    output_shape: None,
                },
            ],
            connections: vec![
                ConnectionDefinition {
                    from: "input".to_string(),
                    to: "hidden".to_string(),
                    connection_type: None,
                },
                ConnectionDefinition {
                    from: "hidden".to_string(),
                    to: "output".to_string(),
                    connection_type: None,
                },
            ],
        };

        let execution_order = parser.compute_execution_order(&desc).unwrap();
        assert_eq!(execution_order, vec!["input", "hidden", "output"]);
    }
}