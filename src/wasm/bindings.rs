//! WASM neural network bindings
//! WASMニューラルネットワークバインディング

#[cfg(feature = "wasm")]
use super::tensor::WasmTensor;
#[cfg(feature = "wasm")]
use std::f32::consts::PI;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Complete linear layer for WASM neural networks
/// WASM用完全な線形レイヤー
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLinear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    has_bias: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLinear {
    /// Create new linear layer with Xavier/Glorot initialization
    /// Xavier/Glorot初期化による新しい線形レイヤーを作成
    #[wasm_bindgen(constructor)]
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier/Glorot uniform initialization
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..(in_features * out_features))
            .map(|_| (js_sys::Math::random() as f32 * 2.0 - 1.0) * bound)
            .collect();

        let bias_data = if bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        WasmLinear {
            in_features,
            out_features,
            weight: weight_data,
            bias: bias_data,
            has_bias: bias,
        }
    }

    /// Create linear layer with custom initialization
    /// カスタム初期化による線形レイヤーを作成
    #[wasm_bindgen]
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<WasmLinear, JsValue> {
        if weights.len() != in_features * out_features {
            return Err(JsValue::from_str("Weight size mismatch"));
        }

        if let Some(ref b) = bias {
            if b.len() != out_features {
                return Err(JsValue::from_str("Bias size mismatch"));
            }
        }

        let has_bias = bias.is_some();
        Ok(WasmLinear {
            in_features,
            out_features,
            weight: weights,
            bias,
            has_bias,
        })
    }

    /// Forward pass through linear layer
    /// 線形レイヤーの順伝播
    #[wasm_bindgen]
    pub fn forward(&self, input: Vec<f32>, batch_size: usize) -> Result<Vec<f32>, JsValue> {
        let input_size = input.len();

        // Check input dimensions
        if input_size != batch_size * self.in_features {
            return Err(JsValue::from_str(&format!(
                "Input size {} doesn't match expected {} (batch_size {} * in_features {})",
                input_size,
                batch_size * self.in_features,
                batch_size,
                self.in_features
            )));
        }

        let mut output = vec![0.0; batch_size * self.out_features];

        // Matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
        for batch in 0..batch_size {
            for out_idx in 0..self.out_features {
                let mut sum = 0.0;

                for in_idx in 0..self.in_features {
                    let input_val = input[batch * self.in_features + in_idx];
                    let weight_val = self.weight[out_idx * self.in_features + in_idx];
                    sum += input_val * weight_val;
                }

                // Add bias if present
                if let Some(ref bias) = self.bias {
                    sum += bias[out_idx];
                }

                output[batch * self.out_features + out_idx] = sum;
            }
        }

        Ok(output)
    }

    /// Get layer parameters for training
    /// 訓練用のレイヤーパラメータを取得
    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f32> {
        self.weight.clone()
    }

    /// Get bias parameters
    /// バイアスパラメータを取得
    #[wasm_bindgen]
    pub fn get_bias(&self) -> Option<Vec<f32>> {
        self.bias.clone()
    }

    /// Update weights with new values
    /// 新しい値で重みを更新
    #[wasm_bindgen]
    pub fn update_weights(&mut self, new_weights: Vec<f32>) -> Result<(), JsValue> {
        if new_weights.len() != self.weight.len() {
            return Err(JsValue::from_str("Weight size mismatch"));
        }
        self.weight = new_weights;
        Ok(())
    }

    /// Update bias with new values
    /// 新しい値でバイアスを更新
    #[wasm_bindgen]
    pub fn update_bias(&mut self, new_bias: Vec<f32>) -> Result<(), JsValue> {
        if !self.has_bias {
            return Err(JsValue::from_str("This layer has no bias"));
        }
        if new_bias.len() != self.out_features {
            return Err(JsValue::from_str("Bias size mismatch"));
        }
        self.bias = Some(new_bias);
        Ok(())
    }

    /// Get input features count
    #[wasm_bindgen]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features count  
    #[wasm_bindgen]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if layer has bias
    #[wasm_bindgen]
    pub fn has_bias(&self) -> bool {
        self.has_bias
    }
}

/// 2D Convolutional layer for WASM
/// WASM用2次元畳み込みレイヤー
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmConv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight: Vec<f32>, // Shape: (out_channels, in_channels, kernel_size, kernel_size)
    bias: Option<Vec<f32>>, // Shape: (out_channels,)
    has_bias: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmConv2d {
    /// Create new 2D convolutional layer
    /// 新しい2次元畳み込みレイヤーを作成
    #[wasm_bindgen(constructor)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        // He initialization for conv layers
        let fan_in = in_channels * kernel_size * kernel_size;
        let std_dev = (2.0 / fan_in as f32).sqrt();

        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let weight_data: Vec<f32> = (0..weight_count)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1 = js_sys::Math::random() as f32;
                let u2 = js_sys::Math::random() as f32;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                z * std_dev
            })
            .collect();

        let bias_data = if bias {
            Some(vec![0.0; out_channels])
        } else {
            None
        };

        WasmConv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight: weight_data,
            bias: bias_data,
            has_bias: bias,
        }
    }

    /// Forward pass through convolution layer
    /// 畳み込みレイヤーの順伝播
    #[wasm_bindgen]
    pub fn forward(
        &self,
        input: Vec<f32>,
        batch_size: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<Vec<f32>, JsValue> {
        // Check input dimensions
        let expected_input_size = batch_size * self.in_channels * input_height * input_width;
        if input.len() != expected_input_size {
            return Err(JsValue::from_str("Input size mismatch"));
        }

        // Calculate output dimensions
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_size = batch_size * self.out_channels * output_height * output_width;

        let mut output = vec![0.0; output_size];

        // Perform convolution
        for batch in 0..batch_size {
            for out_ch in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut conv_sum = 0.0;

                        // Apply convolution kernel
                        for in_ch in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    // Calculate input coordinates with padding
                                    let input_h = out_h * self.stride + kh;
                                    let input_w = out_w * self.stride + kw;

                                    // Check bounds and apply padding
                                    if input_h >= self.padding
                                        && input_w >= self.padding
                                        && input_h < input_height + self.padding
                                        && input_w < input_width + self.padding
                                    {
                                        let actual_input_h = input_h - self.padding;
                                        let actual_input_w = input_w - self.padding;

                                        if actual_input_h < input_height
                                            && actual_input_w < input_width
                                        {
                                            // Get input value
                                            let input_idx = batch
                                                * self.in_channels
                                                * input_height
                                                * input_width
                                                + in_ch * input_height * input_width
                                                + actual_input_h * input_width
                                                + actual_input_w;
                                            let input_val = input[input_idx];

                                            // Get weight value
                                            let weight_idx = out_ch
                                                * self.in_channels
                                                * self.kernel_size
                                                * self.kernel_size
                                                + in_ch * self.kernel_size * self.kernel_size
                                                + kh * self.kernel_size
                                                + kw;
                                            let weight_val = self.weight[weight_idx];

                                            conv_sum += input_val * weight_val;
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(ref bias) = self.bias {
                            conv_sum += bias[out_ch];
                        }

                        // Store result
                        let output_idx = batch * self.out_channels * output_height * output_width
                            + out_ch * output_height * output_width
                            + out_h * output_width
                            + out_w;
                        output[output_idx] = conv_sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Calculate output dimensions for given input
    /// 入力に対する出力次元を計算
    #[wasm_bindgen]
    pub fn output_shape(&self, input_height: usize, input_width: usize) -> Vec<usize> {
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        vec![self.out_channels, output_height, output_width]
    }

    /// Get layer weights
    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f32> {
        self.weight.clone()
    }

    /// Get layer bias
    #[wasm_bindgen]
    pub fn get_bias(&self) -> Option<Vec<f32>> {
        self.bias.clone()
    }

    /// Update weights
    #[wasm_bindgen]
    pub fn update_weights(&mut self, new_weights: Vec<f32>) -> Result<(), JsValue> {
        if new_weights.len() != self.weight.len() {
            return Err(JsValue::from_str("Weight size mismatch"));
        }
        self.weight = new_weights;
        Ok(())
    }

    /// Get layer configuration
    #[wasm_bindgen]
    pub fn get_config(&self) -> js_sys::Object {
        let config = js_sys::Object::new();
        js_sys::Reflect::set(
            &config,
            &"in_channels".into(),
            &JsValue::from_f64(self.in_channels as f64),
        )
        .unwrap();
        js_sys::Reflect::set(
            &config,
            &"out_channels".into(),
            &JsValue::from_f64(self.out_channels as f64),
        )
        .unwrap();
        js_sys::Reflect::set(
            &config,
            &"kernel_size".into(),
            &JsValue::from_f64(self.kernel_size as f64),
        )
        .unwrap();
        js_sys::Reflect::set(
            &config,
            &"stride".into(),
            &JsValue::from_f64(self.stride as f64),
        )
        .unwrap();
        js_sys::Reflect::set(
            &config,
            &"padding".into(),
            &JsValue::from_f64(self.padding as f64),
        )
        .unwrap();
        js_sys::Reflect::set(
            &config,
            &"has_bias".into(),
            &JsValue::from_bool(self.has_bias),
        )
        .unwrap();
        config
    }
}

/// Simple ReLU activation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmReLU;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmReLU {
    /// Create new ReLU activation layer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmReLU
    }

    /// Apply ReLU activation function
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        input.relu()
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod linear_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_linear_layer_creation() {
        let layer = WasmLinear::new(10, 5, true);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.has_bias());

        let weights = layer.get_weights();
        assert_eq!(weights.len(), 50); // 10 * 5

        let bias = layer.get_bias();
        assert!(bias.is_some());
        assert_eq!(bias.unwrap().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_linear_forward_pass() {
        let layer = WasmLinear::new(3, 2, false);
        let input = vec![1.0, 2.0, 3.0]; // Single sample
        let output = layer.forward(input, 1).unwrap();

        assert_eq!(output.len(), 2); // 2 output features
    }

    #[wasm_bindgen_test]
    fn test_linear_batch_processing() {
        let layer = WasmLinear::new(2, 3, true);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples of 2 features each
        let output = layer.forward(input, 2).unwrap();

        assert_eq!(output.len(), 6); // 2 samples * 3 output features
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod conv_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_conv2d_creation() {
        let conv = WasmConv2d::new(3, 64, 3, 1, 1, true);
        let config = conv.get_config();

        // Verify configuration
        let in_channels = js_sys::Reflect::get(&config, &"in_channels".into()).unwrap();
        assert_eq!(in_channels.as_f64().unwrap() as usize, 3);

        let out_channels = js_sys::Reflect::get(&config, &"out_channels".into()).unwrap();
        assert_eq!(out_channels.as_f64().unwrap() as usize, 64);
    }

    #[wasm_bindgen_test]
    fn test_conv2d_output_shape() {
        let conv = WasmConv2d::new(3, 16, 3, 1, 1, false);
        let output_shape = conv.output_shape(32, 32);

        // With padding=1, stride=1, kernel=3: output = input
        assert_eq!(output_shape, vec![16, 32, 32]);
    }

    #[wasm_bindgen_test]
    fn test_conv2d_forward_pass() {
        let conv = WasmConv2d::new(1, 1, 3, 1, 0, false);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3 single channel image

        let output = conv.forward(input, 1, 3, 3).unwrap();
        assert_eq!(output.len(), 1); // 1x1 output (no padding, 3x3 kernel on 3x3 input)
    }
}

/// Simple neural network model for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmModel {
    layers: Vec<String>, // Simple layer tracking
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmModel {
    /// Create new neural network model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmModel { layers: Vec::new() }
    }

    /// Add linear layer
    #[wasm_bindgen]
    pub fn add_linear(&mut self, in_features: usize, out_features: usize, _bias: bool) {
        self.layers
            .push(format!("linear_{}_{}", in_features, out_features));
    }

    /// Add ReLU activation
    #[wasm_bindgen]
    pub fn add_relu(&mut self) {
        self.layers.push("relu".to_string());
    }

    /// Get number of layers
    #[wasm_bindgen]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Simple forward pass (placeholder)
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        // Simplified forward pass - just return a processed version
        input.relu()
    }
}
