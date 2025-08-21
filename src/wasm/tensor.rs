//! WASM tensor operations
//! WASMテンソル操作

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible tensor wrapper
/// WASM互換テンソルラッパー
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensor {
    /// Create a new WASM tensor
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        WasmTensor { data, shape }
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
    
    /// Element-wise addition
    #[wasm_bindgen]
    pub fn add(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(WasmTensor::new(result, self.shape.clone()))
    }
    
    /// Element-wise multiplication
    #[wasm_bindgen]
    pub fn multiply(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        Ok(WasmTensor::new(result, self.shape.clone()))
    }
    
    /// ReLU activation
    #[wasm_bindgen]
    pub fn relu(&self) -> WasmTensor {
        let result: Vec<f32> = self.data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        
        WasmTensor::new(result, self.shape.clone())
    }
    
    /// Sigmoid activation
    #[wasm_bindgen]
    pub fn sigmoid(&self) -> WasmTensor {
        let result: Vec<f32> = self.data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        WasmTensor::new(result, self.shape.clone())
    }
    
    /// Matrix multiplication (2D only)
    #[wasm_bindgen]
    pub fn matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(JsValue::from_str("Only 2D matrices supported"));
        }
        
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k != k2 {
            return Err(JsValue::from_str("Matrix dimensions don't match"));
        }
        
        let mut result = vec![0.0f32; m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        
        Ok(WasmTensor::new(result, vec![m, n]))
    }
}