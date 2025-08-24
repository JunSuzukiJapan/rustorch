//! Integration with actual RusTorch layers
//! 実際のRusTorchレイヤーとの統合

use crate::tensor::Tensor;
use crate::nn::{Linear, Conv2d, BatchNorm2d};
use crate::convert::{SimpleLayerInfo, SimpleConversionError};
use std::collections::HashMap;

/// Layer factory for creating actual RusTorch layers
/// 実際のRusTorchレイヤー作成のためのレイヤーファクトリ
pub struct LayerFactory;

impl LayerFactory {
    /// Create Linear layer with converted parameters
    /// 変換されたパラメータでLinearレイヤーを作成
    pub fn create_linear<T>(
        layer_info: &SimpleLayerInfo
    ) -> Result<LinearLayerWrapper<T>, SimpleConversionError>
    where
        T: num_traits::Float + 'static + std::fmt::Debug + Default + 
           num_traits::FromPrimitive + num_traits::ToPrimitive + 
           num_traits::Zero + num_traits::One + Send + Sync + Copy + 
           ndarray::ScalarOperand + std::iter::Sum + std::fmt::Display,
    {
        // Extract weight tensor
        let weight_tensor = layer_info.tensors.get("weight")
            .ok_or_else(|| SimpleConversionError::MissingParameter("weight".to_string()))?;

        let shape = weight_tensor.shape();
        if shape.len() != 2 {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Expected 2D weight tensor, got {:?}", shape)
            ));
        }

        let out_features = shape[0];
        let in_features = shape[1];
        let has_bias = layer_info.tensors.contains_key("bias");

        // Create RusTorch Linear layer
        let linear_layer = if has_bias {
            Linear::new(in_features, out_features)
        } else {
            Linear::new_no_bias(in_features, out_features)
        };

        // Convert parameter data
        let weight_data: Vec<T> = weight_tensor.data.iter()
            .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
            .collect();

        let bias_data = if has_bias {
            let bias_tensor = layer_info.tensors.get("bias").unwrap();
            Some(bias_tensor.data.iter()
                .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
                .collect())
        } else {
            None
        };

        Ok(LinearLayerWrapper {
            layer: linear_layer,
            name: layer_info.name.clone(),
            weight_data,
            bias_data,
            in_features,
            out_features,
        })
    }

    /// Create Conv2d layer with converted parameters
    /// 変換されたパラメータでConv2dレイヤーを作成
    pub fn create_conv2d<T>(
        layer_info: &SimpleLayerInfo
    ) -> Result<Conv2dLayerWrapper<T>, SimpleConversionError>
    where
        T: num_traits::Float + Send + Sync + ndarray::ScalarOperand + 
           num_traits::FromPrimitive + 'static,
    {
        let weight_tensor = layer_info.tensors.get("weight")
            .ok_or_else(|| SimpleConversionError::MissingParameter("weight".to_string()))?;

        let shape = weight_tensor.shape();
        if shape.len() != 4 {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Expected 4D weight tensor for Conv2d, got {:?}", shape)
            ));
        }

        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_h = shape[2];
        let kernel_w = shape[3];
        let has_bias = layer_info.tensors.contains_key("bias");

        // Create RusTorch Conv2d layer with default parameters
        let conv_layer = Conv2d::new(
            in_channels,
            out_channels,
            (kernel_h, kernel_w),
            Some((1, 1)), // Default stride
            Some((0, 0)), // Default padding
            Some(has_bias), // Bias option
        );

        // Convert parameter data
        let weight_data: Vec<T> = weight_tensor.data.iter()
            .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
            .collect();

        let bias_data = if has_bias {
            let bias_tensor = layer_info.tensors.get("bias").unwrap();
            Some(bias_tensor.data.iter()
                .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
                .collect())
        } else {
            None
        };

        Ok(Conv2dLayerWrapper {
            layer: conv_layer,
            name: layer_info.name.clone(),
            weight_data,
            bias_data,
            in_channels,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
        })
    }

    /// Create BatchNorm2d layer with converted parameters
    /// 変換されたパラメータでBatchNorm2dレイヤーを作成
    pub fn create_batch_norm2d<T>(
        layer_info: &SimpleLayerInfo
    ) -> Result<BatchNorm2dLayerWrapper<T>, SimpleConversionError>
    where
        T: num_traits::Float + Send + Sync + Copy + 'static,
    {
        let weight_tensor = layer_info.tensors.get("weight")
            .ok_or_else(|| SimpleConversionError::MissingParameter("weight".to_string()))?;

        let shape = weight_tensor.shape();
        if shape.len() != 1 {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Expected 1D weight tensor for BatchNorm2d, got {:?}", shape)
            ));
        }

        let num_features = shape[0];
        let batch_norm = BatchNorm2d::new(num_features);

        // Convert all BatchNorm parameters
        let weight_data: Vec<T> = weight_tensor.data.iter()
            .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
            .collect();

        let bias_data: Vec<T> = if let Some(bias_tensor) = layer_info.tensors.get("bias") {
            bias_tensor.data.iter()
                .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
                .collect()
        } else {
            vec![T::zero(); num_features]
        };

        let running_mean_data: Vec<T> = if let Some(mean_tensor) = layer_info.tensors.get("running_mean") {
            mean_tensor.data.iter()
                .map(|&x| T::from(x as f64).unwrap_or_else(|| T::zero()))
                .collect()
        } else {
            vec![T::zero(); num_features]
        };

        let running_var_data: Vec<T> = if let Some(var_tensor) = layer_info.tensors.get("running_var") {
            var_tensor.data.iter()
                .map(|&x| T::from(x as f64).unwrap_or_else(|| T::one()))
                .collect()
        } else {
            vec![T::one(); num_features]
        };

        Ok(BatchNorm2dLayerWrapper {
            layer: batch_norm,
            name: layer_info.name.clone(),
            weight_data,
            bias_data,
            running_mean_data,
            running_var_data,
            num_features,
        })
    }
}

/// Wrapper for Linear layer with converted parameters
/// 変換されたパラメータ付きLinearレイヤーのラッパー
pub struct LinearLayerWrapper<T> {
    /// The actual RusTorch Linear layer
    /// 実際のRusTorch Linearレイヤー
    pub layer: Linear<T>,
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Converted weight data
    /// 変換された重みデータ
    pub weight_data: Vec<T>,
    /// Converted bias data
    /// 変換されたバイアスデータ
    pub bias_data: Option<Vec<T>>,
    /// Input features
    /// 入力特徴
    pub in_features: usize,
    /// Output features
    /// 出力特徴
    pub out_features: usize,
}

impl<T> LinearLayerWrapper<T>
where
    T: num_traits::Float + 'static + std::fmt::Debug + Default + 
       num_traits::FromPrimitive + num_traits::ToPrimitive + 
       num_traits::Zero + num_traits::One + Send + Sync + Copy + 
       ndarray::ScalarOperand + std::iter::Sum + std::fmt::Display,
{
    /// Get input shape
    /// 入力形状を取得
    pub fn input_shape(&self) -> Vec<usize> {
        vec![self.in_features]
    }

    /// Get output shape
    /// 出力形状を取得
    pub fn output_shape(&self) -> Vec<usize> {
        vec![self.out_features]
    }

    /// Get parameter count
    /// パラメータ数を取得
    pub fn parameter_count(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias_data.is_some() { self.out_features } else { 0 };
        weight_params + bias_params
    }

    /// Perform forward pass simulation
    /// 順伝播シミュレーションを実行
    pub fn simulate_forward(&self, input_shape: &[usize]) -> Result<Vec<usize>, SimpleConversionError> {
        if input_shape.is_empty() {
            return Err(SimpleConversionError::InvalidParameter("Empty input shape".to_string()));
        }

        let last_dim = input_shape[input_shape.len() - 1];
        if last_dim != self.in_features {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Input dimension mismatch: expected {}, got {}", self.in_features, last_dim)
            ));
        }

        let mut output_shape = input_shape.to_vec();
        output_shape[output_shape.len() - 1] = self.out_features;
        Ok(output_shape)
    }
}

/// Wrapper for Conv2d layer with converted parameters
/// 変換されたパラメータ付きConv2dレイヤーのラッパー
pub struct Conv2dLayerWrapper<T> {
    /// The actual RusTorch Conv2d layer
    /// 実際のRusTorch Conv2dレイヤー
    pub layer: Conv2d<T>,
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Converted weight data
    /// 変換された重みデータ
    pub weight_data: Vec<T>,
    /// Converted bias data
    /// 変換されたバイアスデータ
    pub bias_data: Option<Vec<T>>,
    /// Input channels
    /// 入力チャンネル
    pub in_channels: usize,
    /// Output channels
    /// 出力チャンネル
    pub out_channels: usize,
    /// Kernel size
    /// カーネルサイズ
    pub kernel_size: (usize, usize),
}

impl<T> Conv2dLayerWrapper<T>
where
    T: num_traits::Float + Send + Sync + ndarray::ScalarOperand + 
       num_traits::FromPrimitive + 'static,
{
    /// Get parameter count
    /// パラメータ数を取得
    pub fn parameter_count(&self) -> usize {
        let weight_params = self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let bias_params = if self.bias_data.is_some() { self.out_channels } else { 0 };
        weight_params + bias_params
    }

    /// Simulate forward pass
    /// 順伝播をシミュレーション
    pub fn simulate_forward(&self, input_shape: &[usize]) -> Result<Vec<usize>, SimpleConversionError> {
        if input_shape.len() != 4 {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Expected 4D input for Conv2d, got {}D", input_shape.len())
            ));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        if in_channels != self.in_channels {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Input channel mismatch: expected {}, got {}", self.in_channels, in_channels)
            ));
        }

        // Simplified calculation (assumes stride=1, padding=0)
        let out_height = in_height.saturating_sub(self.kernel_size.0) + 1;
        let out_width = in_width.saturating_sub(self.kernel_size.1) + 1;

        Ok(vec![batch_size, self.out_channels, out_height, out_width])
    }
}

/// Wrapper for BatchNorm2d layer with converted parameters
/// 変換されたパラメータ付きBatchNorm2dレイヤーのラッパー
pub struct BatchNorm2dLayerWrapper<T> {
    /// The actual RusTorch BatchNorm2d layer
    /// 実際のRusTorch BatchNorm2dレイヤー
    pub layer: BatchNorm2d,
    /// Layer name
    /// レイヤー名
    pub name: String,
    /// Converted weight data (gamma)
    /// 変換された重みデータ（ガンマ）
    pub weight_data: Vec<T>,
    /// Converted bias data (beta)
    /// 変換されたバイアスデータ（ベータ）
    pub bias_data: Vec<T>,
    /// Running mean
    /// 実行中平均
    pub running_mean_data: Vec<T>,
    /// Running variance
    /// 実行中分散
    pub running_var_data: Vec<T>,
    /// Number of features
    /// 特徴数
    pub num_features: usize,
}

impl<T> BatchNorm2dLayerWrapper<T>
where
    T: num_traits::Float + Send + Sync + Copy + 'static,
{
    /// Get parameter count
    /// パラメータ数を取得
    pub fn parameter_count(&self) -> usize {
        // weight + bias + running_mean + running_var
        self.num_features * 4
    }

    /// Simulate forward pass
    /// 順伝播をシミュレーション
    pub fn simulate_forward(&self, input_shape: &[usize]) -> Result<Vec<usize>, SimpleConversionError> {
        if input_shape.len() != 4 {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Expected 4D input for BatchNorm2d, got {}D", input_shape.len())
            ));
        }

        let channels = input_shape[1];
        if channels != self.num_features {
            return Err(SimpleConversionError::InvalidParameter(
                format!("Channel mismatch: expected {}, got {}", self.num_features, channels)
            ));
        }

        // BatchNorm doesn't change the shape
        Ok(input_shape.to_vec())
    }
}

/// Integrated model runner that uses actual RusTorch layers
/// 実際のRusTorchレイヤーを使用する統合モデルランナー
pub struct IntegratedModelRunner<T> {
    /// Linear layers
    /// Linearレイヤー
    pub linear_layers: HashMap<String, LinearLayerWrapper<T>>,
    /// Conv2d layers
    /// Conv2dレイヤー
    pub conv_layers: HashMap<String, Conv2dLayerWrapper<T>>,
    /// BatchNorm2d layers
    /// BatchNorm2dレイヤー
    pub batch_norm_layers: HashMap<String, BatchNorm2dLayerWrapper<T>>,
    /// Execution order
    /// 実行順序
    pub execution_order: Vec<String>,
}

impl<T> IntegratedModelRunner<T>
where
    T: num_traits::Float + 'static + std::fmt::Debug + Default + 
       num_traits::FromPrimitive + num_traits::ToPrimitive + 
       num_traits::Zero + num_traits::One + Send + Sync + Copy + 
       ndarray::ScalarOperand + std::iter::Sum + std::fmt::Display,
{
    /// Create new integrated model runner
    /// 新しい統合モデルランナーを作成
    pub fn new() -> Self {
        Self {
            linear_layers: HashMap::new(),
            conv_layers: HashMap::new(),
            batch_norm_layers: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add Linear layer
    /// Linearレイヤーを追加
    pub fn add_linear_layer(&mut self, wrapper: LinearLayerWrapper<T>) {
        let name = wrapper.name.clone();
        self.linear_layers.insert(name.clone(), wrapper);
        if !self.execution_order.contains(&name) {
            self.execution_order.push(name);
        }
    }

    /// Add Conv2d layer
    /// Conv2dレイヤーを追加
    pub fn add_conv_layer(&mut self, wrapper: Conv2dLayerWrapper<T>) {
        let name = wrapper.name.clone();
        self.conv_layers.insert(name.clone(), wrapper);
        if !self.execution_order.contains(&name) {
            self.execution_order.push(name);
        }
    }

    /// Add BatchNorm2d layer
    /// BatchNorm2dレイヤーを追加
    pub fn add_batch_norm_layer(&mut self, wrapper: BatchNorm2dLayerWrapper<T>) {
        let name = wrapper.name.clone();
        self.batch_norm_layers.insert(name.clone(), wrapper);
        if !self.execution_order.contains(&name) {
            self.execution_order.push(name);
        }
    }

    /// Get total parameter count
    /// 総パラメータ数を取得
    pub fn total_parameters(&self) -> usize {
        let linear_params: usize = self.linear_layers.values()
            .map(|layer| layer.parameter_count())
            .sum();
        
        let conv_params: usize = self.conv_layers.values()
            .map(|layer| layer.parameter_count())
            .sum();
        
        let bn_params: usize = self.batch_norm_layers.values()
            .map(|layer| layer.parameter_count())
            .sum();

        linear_params + conv_params + bn_params
    }

    /// Simulate forward pass through the integrated model
    /// 統合モデルを通じた順伝播をシミュレーション
    pub fn simulate_forward(&self, input_shape: Vec<usize>) -> Result<Vec<usize>, SimpleConversionError> {
        let mut current_shape = input_shape;

        for layer_name in &self.execution_order {
            current_shape = if let Some(layer) = self.linear_layers.get(layer_name) {
                layer.simulate_forward(&current_shape)?
            } else if let Some(layer) = self.conv_layers.get(layer_name) {
                layer.simulate_forward(&current_shape)?
            } else if let Some(layer) = self.batch_norm_layers.get(layer_name) {
                layer.simulate_forward(&current_shape)?
            } else {
                return Err(SimpleConversionError::InvalidParameter(
                    format!("Unknown layer: {}", layer_name)
                ));
            };
        }

        Ok(current_shape)
    }

    /// Get model summary
    /// モデル要約を取得
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Integrated Model Summary\n"));
        summary.push_str(&format!("========================\n"));
        summary.push_str(&format!("Total parameters: {}\n", self.total_parameters()));
        summary.push_str(&format!("Linear layers: {}\n", self.linear_layers.len()));
        summary.push_str(&format!("Conv2d layers: {}\n", self.conv_layers.len()));
        summary.push_str(&format!("BatchNorm2d layers: {}\n", self.batch_norm_layers.len()));
        summary.push_str(&format!("Execution order: {:?}\n", self.execution_order));
        summary
    }
}

impl<T> Default for IntegratedModelRunner<T>
where
    T: num_traits::Float + 'static + std::fmt::Debug + Default + 
       num_traits::FromPrimitive + num_traits::ToPrimitive + 
       num_traits::Zero + num_traits::One + Send + Sync + Copy + 
       ndarray::ScalarOperand + std::iter::Sum + std::fmt::Display,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Convert simplified model to integrated model with actual RusTorch layers
/// 簡略モデルを実際のRusTorchレイヤーを使った統合モデルに変換
pub fn convert_to_integrated_model<T>(
    layers: &HashMap<String, SimpleLayerInfo>,
    execution_order: &[String],
) -> Result<IntegratedModelRunner<T>, SimpleConversionError>
where
    T: num_traits::Float + 'static + std::fmt::Debug + Default + 
       num_traits::FromPrimitive + num_traits::ToPrimitive + 
       num_traits::Zero + num_traits::One + Send + Sync + Copy + 
       ndarray::ScalarOperand + std::iter::Sum + std::fmt::Display,
{
    let mut runner = IntegratedModelRunner::new();
    runner.execution_order = execution_order.to_vec();

    for layer_name in execution_order {
        if let Some(layer_info) = layers.get(layer_name) {
            match layer_info.layer_type.as_str() {
                "Linear" => {
                    let wrapper = LayerFactory::create_linear::<T>(layer_info)?;
                    runner.add_linear_layer(wrapper);
                },
                "Conv2d" => {
                    let wrapper = LayerFactory::create_conv2d::<T>(layer_info)?;
                    runner.add_conv_layer(wrapper);
                },
                "BatchNorm2d" => {
                    let wrapper = LayerFactory::create_batch_norm2d::<T>(layer_info)?;
                    runner.add_batch_norm_layer(wrapper);
                },
                _ => {
                    // Skip unsupported layer types for now
                    continue;
                }
            }
        }
    }

    Ok(runner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_linear_layer_info() -> SimpleLayerInfo {
        let mut tensors = HashMap::new();
        
        // Create weight tensor [2, 3]
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let weight_tensor = Tensor::from_vec(weight_data, vec![2, 3]);
        tensors.insert("weight".to_string(), weight_tensor);
        
        // Create bias tensor [2]
        let bias_data = vec![0.1, 0.2];
        let bias_tensor = Tensor::from_vec(bias_data, vec![2]);
        tensors.insert("bias".to_string(), bias_tensor);

        let mut parameter_shapes = HashMap::new();
        parameter_shapes.insert("weight".to_string(), vec![2, 3]);
        parameter_shapes.insert("bias".to_string(), vec![2]);

        SimpleLayerInfo {
            name: "test_linear".to_string(),
            layer_type: "Linear".to_string(),
            parameter_shapes,
            num_parameters: 8,
            tensors,
        }
    }

    #[test]
    fn test_linear_layer_creation() {
        let layer_info = create_test_linear_layer_info();
        let wrapper: Result<LinearLayerWrapper<f32>, _> = LayerFactory::create_linear(&layer_info);
        
        assert!(wrapper.is_ok());
        let wrapper = wrapper.unwrap();
        assert_eq!(wrapper.in_features, 3);
        assert_eq!(wrapper.out_features, 2);
        assert_eq!(wrapper.parameter_count(), 8);
    }

    #[test]
    fn test_linear_shape_simulation() {
        let layer_info = create_test_linear_layer_info();
        let wrapper: LinearLayerWrapper<f32> = LayerFactory::create_linear(&layer_info).unwrap();
        
        let input_shape = vec![1, 3];
        let output_shape = wrapper.simulate_forward(&input_shape).unwrap();
        assert_eq!(output_shape, vec![1, 2]);
    }

    #[test]
    fn test_integrated_model_runner() {
        let layer_info = create_test_linear_layer_info();
        let wrapper: LinearLayerWrapper<f32> = LayerFactory::create_linear(&layer_info).unwrap();
        
        let mut runner = IntegratedModelRunner::new();
        runner.add_linear_layer(wrapper);
        
        assert_eq!(runner.total_parameters(), 8);
        
        let input_shape = vec![1, 3];
        let output_shape = runner.simulate_forward(input_shape).unwrap();
        assert_eq!(output_shape, vec![1, 2]);
    }
}