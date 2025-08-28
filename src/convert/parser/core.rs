//! Core model parsing logic
//! コアモデル解析ロジック

use crate::formats::pytorch::{PyTorchModel, StateDict, TensorData};
use std::collections::{HashMap, HashSet};

use super::errors::{ParsingError, ParsingResult};
use super::formats::ArchitectureDescription;
use super::types::{LayerInfo, LayerType, ModelGraph};
use super::validation::ModelValidator;

/// PyTorch model parser
/// PyTorchモデルパーサー
pub struct ModelParser {
    /// Layer naming patterns
    /// レイヤー命名パターン
    layer_patterns: HashMap<String, LayerType>,
}

impl ModelParser {
    /// Create new model parser
    /// 新しいモデルパーサーを作成
    pub fn new() -> Self {
        let mut parser = Self {
            layer_patterns: HashMap::new(),
        };

        // Initialize common layer patterns
        parser.init_layer_patterns();
        parser
    }

    /// Initialize common PyTorch layer naming patterns
    /// 一般的なPyTorchレイヤー命名パターンを初期化
    fn init_layer_patterns(&mut self) {
        // Common patterns for different layer types
        let patterns = vec![
            (
                "linear",
                LayerType::Linear {
                    in_features: 0,
                    out_features: 0,
                },
            ),
            (
                "fc",
                LayerType::Linear {
                    in_features: 0,
                    out_features: 0,
                },
            ),
            (
                "classifier",
                LayerType::Linear {
                    in_features: 0,
                    out_features: 0,
                },
            ),
            (
                "conv3d",
                LayerType::Conv3d {
                    in_channels: 0,
                    out_channels: 0,
                    kernel_size: (0, 0, 0),
                    stride: (1, 1, 1),
                    padding: (0, 0, 0),
                },
            ),
            (
                "conv1d",
                LayerType::Conv1d {
                    in_channels: 0,
                    out_channels: 0,
                    kernel_size: 0,
                    stride: 1,
                    padding: 0,
                },
            ),
            (
                "conv",
                LayerType::Conv2d {
                    in_channels: 0,
                    out_channels: 0,
                    kernel_size: (0, 0),
                    stride: (1, 1),
                    padding: (0, 0),
                },
            ),
            ("bn", LayerType::BatchNorm2d { num_features: 0 }),
            ("batch_norm", LayerType::BatchNorm2d { num_features: 0 }),
            ("relu", LayerType::ReLU),
            ("dropout", LayerType::Dropout { p: 0.5 }),
            (
                "maxpool",
                LayerType::MaxPool2d {
                    kernel_size: (2, 2),
                    stride: (2, 2),
                },
            ),
            (
                "avgpool",
                LayerType::AvgPool2d {
                    kernel_size: (2, 2),
                    stride: (2, 2),
                },
            ),
            ("flatten", LayerType::Flatten),
        ];

        for (pattern, layer_type) in patterns {
            self.layer_patterns.insert(pattern.to_string(), layer_type);
        }
    }

    /// Parse PyTorch model into model graph
    /// PyTorchモデルをモデルグラフに解析
    pub fn parse_model(&self, pytorch_model: &PyTorchModel) -> Result<ModelGraph, ParsingError> {
        // Extract layer information from state dict
        let layers = self.extract_layers(&pytorch_model.state_dict)?;

        // Infer model architecture
        let (execution_order, connections) = if let Some(architecture) = &pytorch_model.architecture
        {
            self.parse_explicit_architecture(architecture, &layers)?
        } else {
            self.infer_architecture(&layers)?
        };

        // Determine input and output layers
        let input_layers = self.find_input_layers(&execution_order, &connections);
        let output_layers = self.find_output_layers(&execution_order, &connections);

        // Validate the model graph
        let graph = ModelGraph {
            layers,
            execution_order,
            connections,
            input_layers,
            output_layers,
        };

        ModelValidator::validate_graph(&graph)?;
        Ok(graph)
    }

    /// Extract layer information from state dictionary
    /// ステートディクショナリからレイヤー情報を抽出
    pub fn extract_layers(
        &self,
        state_dict: &StateDict,
    ) -> Result<HashMap<String, LayerInfo>, ParsingError> {
        let mut layers = HashMap::new();

        // Group parameters by layer name
        let mut layer_params: HashMap<String, HashMap<String, &TensorData>> = HashMap::new();

        for (param_name, tensor_data) in &state_dict.tensors {
            let (layer_name, param_type) = self.parse_parameter_name(param_name)?;

            layer_params
                .entry(layer_name.clone())
                .or_default()
                .insert(param_type, tensor_data);
        }

        // Convert each layer's parameters to LayerInfo
        for (layer_name, params) in layer_params {
            let layer_info = self.create_layer_info(&layer_name, &params)?;
            layers.insert(layer_name, layer_info);
        }

        Ok(layers)
    }

    /// Parse parameter name to extract layer name and parameter type
    /// パラメータ名を解析してレイヤー名とパラメータタイプを抽出
    pub fn parse_parameter_name(&self, param_name: &str) -> Result<(String, String), ParsingError> {
        let parts: Vec<&str> = param_name.split('.').collect();

        if parts.len() < 2 {
            return Err(ParsingError::InvalidArchitecture(format!(
                "Invalid parameter name format: {}",
                param_name
            )));
        }

        let param_type = parts.last().unwrap().to_string();
        let layer_name = parts[..parts.len() - 1].join(".");

        Ok((layer_name, param_type))
    }

    /// Create layer information from parameters
    /// パラメータからレイヤー情報を作成
    fn create_layer_info(
        &self,
        layer_name: &str,
        params: &HashMap<String, &TensorData>,
    ) -> Result<LayerInfo, ParsingError> {
        let layer_type = self.infer_layer_type(layer_name, params)?;

        // Calculate total parameters
        let num_parameters: usize = params
            .values()
            .map(|tensor_data| tensor_data.shape.iter().product::<usize>())
            .sum();

        // Extract parameter shapes
        let mut parameters = HashMap::new();
        for (param_name, tensor_data) in params {
            parameters.insert(param_name.clone(), tensor_data.shape.clone());
        }

        // Infer input/output shapes based on layer type
        let (input_shape, output_shape) = self.infer_shapes(&layer_type, params);

        Ok(LayerInfo {
            name: layer_name.to_string(),
            layer_type,
            input_shape,
            output_shape,
            parameters,
            num_parameters,
        })
    }

    /// Infer layer type from name and parameters
    /// 名前とパラメータからレイヤータイプを推論
    pub fn infer_layer_type(
        &self,
        layer_name: &str,
        params: &HashMap<String, &TensorData>,
    ) -> Result<LayerType, ParsingError> {
        // Check layer name patterns first - prioritize exact matches
        let layer_lower = layer_name.to_lowercase();

        // First pass: exact matches
        for (pattern, base_type) in &self.layer_patterns {
            if layer_lower == *pattern {
                return self.refine_layer_type(base_type.clone(), params);
            }
        }

        // Second pass: substring matches, but prioritize longer patterns
        let mut matches: Vec<(&String, &LayerType)> = self
            .layer_patterns
            .iter()
            .filter(|(pattern, _)| layer_lower.contains(pattern.as_str()))
            .collect();

        // Sort by pattern length (longest first) to prefer more specific matches
        matches.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        if let Some((_, base_type)) = matches.first() {
            return self.refine_layer_type((*base_type).clone(), params);
        }

        // Infer from parameter shapes if no pattern match
        if let Some(weight) = params.get("weight") {
            match weight.shape.len() {
                2 => {
                    // Linear layer
                    let out_features = weight.shape[0];
                    let in_features = weight.shape[1];
                    Ok(LayerType::Linear {
                        in_features,
                        out_features,
                    })
                }
                3 => {
                    // Conv1d layer
                    let out_channels = weight.shape[0];
                    let in_channels = weight.shape[1];
                    let kernel_size = weight.shape[2];
                    Ok(LayerType::Conv1d {
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride: 1,  // Default
                        padding: 0, // Default
                    })
                }
                4 => {
                    // Conv2d layer
                    let out_channels = weight.shape[0];
                    let in_channels = weight.shape[1];
                    let kernel_h = weight.shape[2];
                    let kernel_w = weight.shape[3];
                    Ok(LayerType::Conv2d {
                        in_channels,
                        out_channels,
                        kernel_size: (kernel_h, kernel_w),
                        stride: (1, 1),  // Default
                        padding: (0, 0), // Default
                    })
                }
                5 => {
                    // Conv3d layer
                    let out_channels = weight.shape[0];
                    let in_channels = weight.shape[1];
                    let kernel_d = weight.shape[2];
                    let kernel_h = weight.shape[3];
                    let kernel_w = weight.shape[4];
                    Ok(LayerType::Conv3d {
                        in_channels,
                        out_channels,
                        kernel_size: (kernel_d, kernel_h, kernel_w),
                        stride: (1, 1, 1),  // Default
                        padding: (0, 0, 0), // Default
                    })
                }
                1 => {
                    // BatchNorm layer
                    let num_features = weight.shape[0];
                    Ok(LayerType::BatchNorm2d { num_features })
                }
                _ => Ok(LayerType::Unknown(format!(
                    "weight_shape_{:?}",
                    weight.shape
                ))),
            }
        } else {
            // No weight parameter - might be activation or pooling
            Ok(LayerType::Unknown(layer_name.to_string()))
        }
    }

    /// Refine base layer type with actual parameters
    /// 実際のパラメータでベースレイヤータイプを詳細化
    fn refine_layer_type(
        &self,
        mut base_type: LayerType,
        params: &HashMap<String, &TensorData>,
    ) -> Result<LayerType, ParsingError> {
        match &mut base_type {
            LayerType::Linear {
                in_features,
                out_features,
            } => {
                if let Some(weight) = params.get("weight") {
                    if weight.shape.len() == 2 {
                        *out_features = weight.shape[0];
                        *in_features = weight.shape[1];
                    }
                }
            }
            LayerType::Conv1d {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                if let Some(weight) = params.get("weight") {
                    if weight.shape.len() == 3 {
                        *out_channels = weight.shape[0];
                        *in_channels = weight.shape[1];
                        *kernel_size = weight.shape[2];
                    }
                }
            }
            LayerType::Conv2d {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                if let Some(weight) = params.get("weight") {
                    if weight.shape.len() == 4 {
                        *out_channels = weight.shape[0];
                        *in_channels = weight.shape[1];
                        *kernel_size = (weight.shape[2], weight.shape[3]);
                    }
                }
            }
            LayerType::Conv3d {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                if let Some(weight) = params.get("weight") {
                    if weight.shape.len() == 5 {
                        *out_channels = weight.shape[0];
                        *in_channels = weight.shape[1];
                        *kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4]);
                    }
                }
            }
            LayerType::BatchNorm2d { num_features } => {
                if let Some(weight) = params.get("weight") {
                    if weight.shape.len() == 1 {
                        *num_features = weight.shape[0];
                    }
                }
            }
            _ => {} // Other types don't need refinement
        }

        Ok(base_type)
    }

    /// Infer input and output shapes for a layer
    /// レイヤーの入力と出力形状を推論
    fn infer_shapes(
        &self,
        layer_type: &LayerType,
        _params: &HashMap<String, &TensorData>,
    ) -> (Option<Vec<usize>>, Option<Vec<usize>>) {
        match layer_type {
            LayerType::Linear {
                in_features,
                out_features,
            } => (Some(vec![*in_features]), Some(vec![*out_features])),
            LayerType::Conv2d {
                in_channels,
                out_channels,
                ..
            } => {
                // Input: [batch, in_channels, H, W] - we can't know H, W from weights alone
                // Output: [batch, out_channels, H', W'] - depends on input size
                (Some(vec![*in_channels]), Some(vec![*out_channels]))
            }
            LayerType::BatchNorm2d { num_features } => {
                (Some(vec![*num_features]), Some(vec![*num_features]))
            }
            _ => (None, None), // Cannot infer shapes for other layer types
        }
    }

    /// Infer model architecture from layers
    /// レイヤーからモデルアーキテクチャを推論
    fn infer_architecture(&self, layers: &HashMap<String, LayerInfo>) -> ParsingResult {
        // Simple sequential ordering based on layer names
        let mut execution_order: Vec<String> = layers.keys().cloned().collect();
        execution_order.sort();

        // Create sequential connections
        let mut connections: HashMap<String, Vec<String>> = HashMap::new();
        for i in 0..execution_order.len().saturating_sub(1) {
            let current_layer = &execution_order[i];
            let next_layer = &execution_order[i + 1];

            connections
                .entry(current_layer.clone())
                .or_default()
                .push(next_layer.clone());
        }

        Ok((execution_order, connections))
    }

    /// Parse explicit architecture description
    /// 明示的なアーキテクチャ記述を解析
    fn parse_explicit_architecture(
        &self,
        architecture: &str,
        layers: &HashMap<String, LayerInfo>,
    ) -> ParsingResult {
        // Try to parse as JSON first, then YAML
        let arch_desc = self.parse_architecture_string(architecture)?;

        // Convert architecture description to execution order and connections
        let execution_order = self.compute_execution_order(&arch_desc)?;
        let connections = self.build_connections_map(&arch_desc);

        // Validate that all referenced layers exist
        ModelValidator::validate_layer_references(&arch_desc, layers)?;

        Ok((execution_order, connections))
    }

    /// Parse architecture string as JSON or YAML
    /// アーキテクチャ文字列をJSONまたはYAMLとして解析
    pub fn parse_architecture_string(
        &self,
        architecture: &str,
    ) -> Result<ArchitectureDescription, ParsingError> {
        // Try JSON parsing first
        if let Ok(desc) = serde_json::from_str::<ArchitectureDescription>(architecture) {
            return Ok(desc);
        }

        // Try YAML parsing
        if let Ok(desc) = serde_yaml::from_str::<ArchitectureDescription>(architecture) {
            return Ok(desc);
        }

        // If both fail, try simple format parsing
        self.parse_simple_format(architecture)
    }

    /// Parse simple architecture format (e.g., "conv2d -> relu -> pool -> linear")
    /// シンプルなアーキテクチャ形式を解析
    pub fn parse_simple_format(
        &self,
        architecture: &str,
    ) -> Result<ArchitectureDescription, ParsingError> {
        use super::formats::*;

        let layer_names: Vec<&str> = architecture.split("->").map(|s| s.trim()).collect();

        if layer_names.is_empty() {
            return Err(ParsingError::InvalidArchitecture(
                "Empty architecture description".to_string(),
            ));
        }

        let mut layers = Vec::new();
        let mut connections = Vec::new();

        // Create layer definitions
        for (i, layer_name) in layer_names.iter().enumerate() {
            layers.push(LayerDefinition {
                name: format!("layer_{}", i),
                layer_type: layer_name.to_string(),
                params: None,
                input_shape: None,
                output_shape: None,
            });

            // Create connections (except for last layer)
            if i < layer_names.len() - 1 {
                connections.push(ConnectionDefinition {
                    from: format!("layer_{}", i),
                    to: format!("layer_{}", i + 1),
                    connection_type: None,
                });
            }
        }

        Ok(ArchitectureDescription {
            metadata: ModelMetadata {
                name: "parsed_model".to_string(),
                version: None,
                framework: Some("unknown".to_string()),
                description: Some("Parsed from simple format".to_string()),
            },
            layers,
            connections,
        })
    }

    /// Compute execution order from architecture description using topological sort
    /// アーキテクチャ記述からトポロジカルソートを使って実行順序を計算
    pub fn compute_execution_order(
        &self,
        desc: &ArchitectureDescription,
    ) -> Result<Vec<String>, ParsingError> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Initialize in_degree for all layers
        for layer in &desc.layers {
            in_degree.insert(layer.name.clone(), 0);
            graph.insert(layer.name.clone(), Vec::new());
        }

        // Build graph and calculate in-degrees
        for connection in &desc.connections {
            graph
                .entry(connection.from.clone())
                .or_default()
                .push(connection.to.clone());

            *in_degree.entry(connection.to.clone()).or_insert(0) += 1;
        }

        // Topological sort using Kahn's algorithm
        let mut queue = Vec::new();
        let mut execution_order = Vec::new();

        // Add all layers with no incoming edges
        for (layer_name, degree) in &in_degree {
            if *degree == 0 {
                queue.push(layer_name.clone());
            }
        }

        while let Some(current) = queue.pop() {
            execution_order.push(current.clone());

            // Update in-degrees for neighbors
            if let Some(neighbors) = graph.get(&current) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        // Check for cycles
        if execution_order.len() != desc.layers.len() {
            return Err(ParsingError::CircularDependency(
                "Circular dependency detected in architecture description".to_string(),
            ));
        }

        Ok(execution_order)
    }

    /// Build connections map from architecture description
    /// アーキテクチャ記述から接続マップを構築
    fn build_connections_map(
        &self,
        desc: &ArchitectureDescription,
    ) -> HashMap<String, Vec<String>> {
        let mut connections: HashMap<String, Vec<String>> = HashMap::new();

        for connection in &desc.connections {
            connections
                .entry(connection.from.clone())
                .or_default()
                .push(connection.to.clone());
        }

        connections
    }

    /// Find input layers (layers with no incoming connections)
    /// 入力レイヤーを見つける（入力接続のないレイヤー）
    fn find_input_layers(
        &self,
        execution_order: &[String],
        connections: &HashMap<String, Vec<String>>,
    ) -> Vec<String> {
        let mut has_incoming: HashSet<String> = HashSet::new();

        // Mark all layers that have incoming connections
        for targets in connections.values() {
            for target in targets {
                has_incoming.insert(target.clone());
            }
        }

        // Input layers are those without incoming connections
        execution_order
            .iter()
            .filter(|layer| !has_incoming.contains(*layer))
            .cloned()
            .collect()
    }

    /// Find output layers (layers with no outgoing connections)
    /// 出力レイヤーを見つける（出力接続のないレイヤー）
    fn find_output_layers(
        &self,
        execution_order: &[String],
        connections: &HashMap<String, Vec<String>>,
    ) -> Vec<String> {
        execution_order
            .iter()
            .filter(|layer| !connections.contains_key(*layer) || connections[*layer].is_empty())
            .cloned()
            .collect()
    }
}

impl Default for ModelParser {
    fn default() -> Self {
        Self::new()
    }
}