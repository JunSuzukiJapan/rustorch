//! WASM neural network bindings
//! WASMニューラルネットワークバインディング

#[cfg(feature = "wasm")]
use super::tensor::WasmTensor;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Simple linear layer for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLinear {
    weight: WasmTensor,
    bias: Option<WasmTensor>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLinear {
    /// Create new linear layer
    #[wasm_bindgen(constructor)]
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Simple random initialization
        let weight_data: Vec<f32> = (0..(in_features * out_features))
            .map(|_| (js_sys::Math::random() as f32 - 0.5) * 0.2)
            .collect();
        let weight = WasmTensor::new(weight_data, vec![out_features, in_features]);

        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_features).map(|_| 0.0).collect();
            Some(WasmTensor::new(bias_data, vec![out_features]))
        } else {
            None
        };

        WasmLinear { weight, bias }
    }

    /// Forward pass
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let output = input.matmul(&self.weight)?;

        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
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
