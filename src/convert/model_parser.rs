//! PyTorch model architecture parsing and analysis
//! PyTorchモデルアーキテクチャの解析と分析

use crate::formats::pytorch::{PyTorchModel, StateDict, TensorData};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Model parsing errors
/// モデル解析エラー
#[derive(Debug)]
pub enum ParsingError {
    /// Invalid architecture format
    /// 無効なアーキテクチャ形式
    InvalidArchitecture(String),
    /// Circular dependency detected
    /// 循環依存を検出
    CircularDependency(String),
    /// Missing layer connection
    /// レイヤー接続が見つからない
    MissingConnection(String),
    /// Incompatible layer dimensions
    /// 互換性のないレイヤー次元
    IncompatibleDimensions {
        /// First layer name
        layer1: String,
        /// Second layer name
        layer2: String,
    },
}

impl fmt::Display for ParsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParsingError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {}", msg),
            ParsingError::CircularDependency(msg) => write!(f, "Circular dependency: {}", msg),
            ParsingError::MissingConnection(msg) => write!(f, "Missing connection: {}", msg),
            ParsingError::IncompatibleDimensions { layer1, layer2 } => write!(
                f,
                "Incompatible dimensions between {} and {}",
                layer1, layer2
            ),
        }
    }
}

impl Error for ParsingError {}

/// Layer information extracted from PyTorch model
/// PyTorchモデルから抽出されたレイヤー情報
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Layer type (Linear, Conv2d, etc.)
    /// レイヤータイプ（Linear、Conv2dなど）
    pub layer_type: LayerType,
    /// Input shape
    /// 入力形状
    pub input_shape: Option<Vec<usize>>,
    /// Output shape
    /// 出力形状
    pub output_shape: Option<Vec<usize>>,
    /// Layer parameters
    /// レイヤーパラメータ
    pub parameters: HashMap<String, Vec<usize>>, // parameter name -> shape
    /// Number of parameters
    /// パラメータ数
    pub num_parameters: usize,
}

/// Supported layer types
/// サポートされているレイヤータイプ
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    /// Linear/Dense layer
    /// Linear/Denseレイヤー
    Linear {
        /// Input features
        in_features: usize,
        /// Output features
        out_features: usize,
    },
    /// 2D Convolution layer
    /// 2D畳み込みレイヤー
    Conv2d {
        /// Input channels
        in_channels: usize,
        /// Output channels
        out_channels: usize,
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
        /// Padding
        padding: (usize, usize),
    },
    /// 2D Batch Normalization
    /// 2Dバッチ正規化
    BatchNorm2d {
        /// Number of features
        num_features: usize,
    },
    /// ReLU activation
    /// ReLU活性化
    ReLU,
    /// Dropout layer
    /// Dropoutレイヤー
    Dropout {
        /// Dropout probability
        p: f64,
    },
    /// 2D Max Pooling
    /// 2D最大プーリング
    MaxPool2d {
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
    },
    /// 2D Average Pooling
    /// 2D平均プーリング
    AvgPool2d {
        /// Kernel size
        kernel_size: (usize, usize),
        /// Stride
        stride: (usize, usize),
    },
    /// Flatten layer
    /// Flattenレイヤー
    Flatten,
    /// Unknown layer type
    /// 不明なレイヤータイプ
    Unknown(String),
}

/// Model architecture graph
/// モデルアーキテクチャグラフ
#[derive(Debug)]
pub struct ModelGraph {
    /// Layers in the model
    /// モデル内のレイヤー
    pub layers: HashMap<String, LayerInfo>,
    /// Layer execution order
    /// レイヤー実行順序
    pub execution_order: Vec<String>,
    /// Layer connections (from -> to)
    /// レイヤー接続（from -> to）
    pub connections: HashMap<String, Vec<String>>,
    /// Model input layers
    /// モデル入力レイヤー
    pub input_layers: Vec<String>,
    /// Model output layers
    /// モデル出力レイヤー
    pub output_layers: Vec<String>,
}

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

        self.validate_graph(&graph)?;
        Ok(graph)
    }

    /// Extract layer information from state dictionary
    /// ステートディクショナリからレイヤー情報を抽出
    fn extract_layers(
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
                .or_insert_with(HashMap::new)
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
    fn parse_parameter_name(&self, param_name: &str) -> Result<(String, String), ParsingError> {
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
    fn infer_layer_type(
        &self,
        layer_name: &str,
        params: &HashMap<String, &TensorData>,
    ) -> Result<LayerType, ParsingError> {
        // Check layer name patterns first
        for (pattern, base_type) in &self.layer_patterns {
            if layer_name.to_lowercase().contains(pattern) {
                return self.refine_layer_type(base_type.clone(), params);
            }
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
    fn infer_architecture(
        &self,
        layers: &HashMap<String, LayerInfo>,
    ) -> Result<(Vec<String>, HashMap<String, Vec<String>>), ParsingError> {
        // Simple sequential ordering based on layer names
        let mut execution_order: Vec<String> = layers.keys().cloned().collect();
        execution_order.sort();

        // Create sequential connections
        let mut connections = HashMap::new();
        for i in 0..execution_order.len().saturating_sub(1) {
            let current_layer = &execution_order[i];
            let next_layer = &execution_order[i + 1];

            connections
                .entry(current_layer.clone())
                .or_insert_with(Vec::new)
                .push(next_layer.clone());
        }

        Ok((execution_order, connections))
    }

    /// Parse explicit architecture description
    /// 明示的なアーキテクチャ記述を解析
    fn parse_explicit_architecture(
        &self,
        _architecture: &str,
        layers: &HashMap<String, LayerInfo>,
    ) -> Result<(Vec<String>, HashMap<String, Vec<String>>), ParsingError> {
        // TODO: Implement parsing of explicit architecture descriptions
        // For now, fall back to inference
        self.infer_architecture(layers)
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

    /// Validate model graph for consistency
    /// モデルグラフの一貫性を検証
    fn validate_graph(&self, graph: &ModelGraph) -> Result<(), ParsingError> {
        // Check for cycles
        self.check_cycles(graph)?;

        // Check layer compatibility
        self.check_layer_compatibility(graph)?;

        // Ensure all referenced layers exist
        for (from_layer, to_layers) in &graph.connections {
            if !graph.layers.contains_key(from_layer) {
                return Err(ParsingError::MissingConnection(format!(
                    "Source layer '{}' not found",
                    from_layer
                )));
            }

            for to_layer in to_layers {
                if !graph.layers.contains_key(to_layer) {
                    return Err(ParsingError::MissingConnection(format!(
                        "Target layer '{}' not found",
                        to_layer
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check for cycles in the model graph
    /// モデルグラフの循環をチェック
    fn check_cycles(&self, graph: &ModelGraph) -> Result<(), ParsingError> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn dfs_cycle_check(
            node: &str,
            connections: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            rec_stack: &mut HashSet<String>,
        ) -> Result<(), String> {
            visited.insert(node.to_string());
            rec_stack.insert(node.to_string());

            if let Some(neighbors) = connections.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        dfs_cycle_check(neighbor, connections, visited, rec_stack)?;
                    } else if rec_stack.contains(neighbor) {
                        return Err(format!("Cycle detected: {} -> {}", node, neighbor));
                    }
                }
            }

            rec_stack.remove(node);
            Ok(())
        }

        for layer in graph.layers.keys() {
            if !visited.contains(layer) {
                if let Err(cycle_info) =
                    dfs_cycle_check(layer, &graph.connections, &mut visited, &mut rec_stack)
                {
                    return Err(ParsingError::CircularDependency(cycle_info));
                }
            }
        }

        Ok(())
    }

    /// Check layer dimension compatibility
    /// レイヤー次元の互換性をチェック
    fn check_layer_compatibility(&self, graph: &ModelGraph) -> Result<(), ParsingError> {
        for (from_layer, to_layers) in &graph.connections {
            let from_info = &graph.layers[from_layer];

            for to_layer in to_layers {
                let to_info = &graph.layers[to_layer];

                // Check if output shape of from_layer matches input shape of to_layer
                if let (Some(output_shape), Some(input_shape)) =
                    (&from_info.output_shape, &to_info.input_shape)
                {
                    if !self.shapes_compatible(output_shape, input_shape) {
                        return Err(ParsingError::IncompatibleDimensions {
                            layer1: from_layer.clone(),
                            layer2: to_layer.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if two shapes are compatible for layer connection
    /// 2つの形状がレイヤー接続に対して互換性があるかチェック
    fn shapes_compatible(&self, output_shape: &[usize], input_shape: &[usize]) -> bool {
        // Simple compatibility check - in practice, this would be more sophisticated
        // considering transformations like flatten, reshape, etc.

        if output_shape.len() == 1 && input_shape.len() == 1 {
            // Both are 1D - must match exactly
            output_shape[0] == input_shape[0]
        } else if output_shape.len() > 1 && input_shape.len() == 1 {
            // Output is multi-dimensional, input expects 1D (e.g., after flatten)
            let output_size: usize = output_shape.iter().product();
            output_size == input_shape[0]
        } else {
            // For now, assume compatible if we can't determine
            true
        }
    }
}

impl Default for ModelParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::pytorch::{StateDict, TensorData};

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

        // Test Linear inference
        let linear_weight = TensorData {
            shape: vec![10, 512],
            data: vec![0.1; 5120],
            dtype: "f32".to_string(),
        };
        params.insert("weight".to_string(), &linear_weight);

        let layer_type = parser.infer_layer_type("fc", &params).unwrap();
        assert!(matches!(layer_type, LayerType::Linear { .. }));
    }
}
