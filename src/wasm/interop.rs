//! JavaScript/TypeScript interoperability utilities
//! JavaScript/TypeScript相互運用ユーティリティ

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use super::tensor::WasmTensor;

/// JavaScript interop utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct JsInterop;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl JsInterop {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsInterop
    }
    
    /// Create tensor filled with ones
    #[wasm_bindgen]
    pub fn ones(&self, shape: js_sys::Array) -> WasmTensor {
        let shape_vec: Vec<usize> = shape.iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();
        
        let total_elements: usize = shape_vec.iter().product();
        let data = vec![1.0f32; total_elements];
        
        WasmTensor::new(data, shape_vec)
    }
    
    /// Create tensor filled with zeros
    #[wasm_bindgen]
    pub fn zeros(&self, shape: js_sys::Array) -> WasmTensor {
        let shape_vec: Vec<usize> = shape.iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();
        
        let total_elements: usize = shape_vec.iter().product();
        let data = vec![0.0f32; total_elements];
        
        WasmTensor::new(data, shape_vec)
    }
    
    /// Create random tensor
    #[wasm_bindgen]
    pub fn random_tensor(&self, shape: js_sys::Array, min: f32, max: f32) -> WasmTensor {
        let shape_vec: Vec<usize> = shape.iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();
        
        let total_elements: usize = shape_vec.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| {
                let random = js_sys::Math::random() as f32;
                min + random * (max - min)
            })
            .collect();
        
        WasmTensor::new(data, shape_vec)
    }
    
    /// Log tensor information to console
    #[wasm_bindgen]
    pub fn log_tensor(&self, tensor: &WasmTensor, name: &str) {
        web_sys::console::log_2(
            &format!("Tensor {}: shape {:?}", name, tensor.shape()).into(),
            &format!("data: {:?}", &tensor.data()[..5.min(tensor.data().len())]).into()
        );
    }
}

/// Initialize WASM module
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init_wasm() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    web_sys::console::log_1(&"RusTorch WASM initialized successfully!".into());
}