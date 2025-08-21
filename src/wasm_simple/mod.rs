//! Simple WebAssembly bindings for basic tensor operations
//! 基本的なテンソル操作のためのシンプルなWebAssemblyバインディング

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;


#[cfg(feature = "wasm")]
/// Simple WASM tensor wrapper
#[wasm_bindgen]
pub struct SimpleTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl SimpleTensor {
    /// Create a new tensor
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        SimpleTensor { data, shape }
    }
    
    /// Get tensor data
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
    
    /// Get tensor shape
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    
    /// Add two tensors
    #[wasm_bindgen]
    pub fn add(&self, other: &SimpleTensor) -> Result<SimpleTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(SimpleTensor::new(result, self.shape.clone()))
    }
    
    /// Apply ReLU activation
    #[wasm_bindgen]
    pub fn relu(&self) -> SimpleTensor {
        let result: Vec<f32> = self.data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        
        SimpleTensor::new(result, self.shape.clone())
    }
}

