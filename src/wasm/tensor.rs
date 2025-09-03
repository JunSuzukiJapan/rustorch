//! WASM tensor operations
//! WASMテンソル操作

#[cfg(feature = "wasm")]
use js_sys;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use crate::tensor::shared_ops::{math_ops, activation_ops, math_funcs, stats_ops, shape_ops, CommonTensorOps};

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
        if !shape_ops::shapes_compatible(&self.shape, &other.shape) {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let result = math_ops::element_wise_add(&self.data, &other.data)
            .map_err(|e| JsValue::from_str(e))?;

        Ok(WasmTensor::new(result, self.shape.clone()))
    }

    /// Element-wise multiplication
    #[wasm_bindgen]
    pub fn multiply(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if !shape_ops::shapes_compatible(&self.shape, &other.shape) {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let result = math_ops::element_wise_mul(&self.data, &other.data)
            .map_err(|e| JsValue::from_str(e))?;

        Ok(WasmTensor::new(result, self.shape.clone()))
    }

    /// ReLU activation
    #[wasm_bindgen]
    pub fn relu(&self) -> WasmTensor {
        let result = activation_ops::relu(&self.data);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Sigmoid activation
    #[wasm_bindgen]
    pub fn sigmoid(&self) -> WasmTensor {
        let result = activation_ops::sigmoid(&self.data);
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

    /// Create tensor filled with zeros
    #[wasm_bindgen]
    pub fn zeros(shape: Vec<usize>) -> WasmTensor {
        let size: usize = shape.iter().product();
        WasmTensor::new(vec![0.0; size], shape)
    }

    /// Create tensor filled with ones
    #[wasm_bindgen]
    pub fn ones(shape: Vec<usize>) -> WasmTensor {
        let size: usize = shape.iter().product();
        WasmTensor::new(vec![1.0; size], shape)
    }

    /// Create tensor with random values
    #[wasm_bindgen]
    pub fn random(shape: Vec<usize>) -> WasmTensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| js_sys::Math::random() as f32).collect();
        WasmTensor::new(data, shape)
    }

    /// Reshape tensor
    #[wasm_bindgen]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        if !shape_ops::can_reshape(&self.shape, &new_shape) {
            return Err(JsValue::from_str("Cannot reshape: size mismatch"));
        }

        Ok(WasmTensor::new(self.data.clone(), new_shape))
    }

    /// Get tensor size (total number of elements)
    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        shape_ops::total_elements(&self.shape)
    }

    /// Get tensor dimensions (number of axes)
    #[wasm_bindgen]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Transpose 2D tensor
    #[wasm_bindgen]
    pub fn transpose(&self) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("Only 2D tensors supported"));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(WasmTensor::new(result, vec![cols, rows]))
    }

    /// Element-wise subtraction
    #[wasm_bindgen]
    pub fn subtract(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if !shape_ops::shapes_compatible(&self.shape, &other.shape) {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let result = math_ops::element_wise_sub(&self.data, &other.data)
            .map_err(|e| JsValue::from_str(e))?;

        Ok(WasmTensor::new(result, self.shape.clone()))
    }

    /// Element-wise division
    #[wasm_bindgen]
    pub fn divide(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if !shape_ops::shapes_compatible(&self.shape, &other.shape) {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let result = math_ops::element_wise_div(&self.data, &other.data)
            .map_err(|e| JsValue::from_str(e))?;

        Ok(WasmTensor::new(result, self.shape.clone()))
    }

    /// Scalar addition
    #[wasm_bindgen]
    pub fn add_scalar(&self, scalar: f32) -> WasmTensor {
        let result = math_ops::scalar_add(&self.data, scalar);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Scalar multiplication
    #[wasm_bindgen]
    pub fn mul_scalar(&self, scalar: f32) -> WasmTensor {
        let result = math_ops::scalar_mul(&self.data, scalar);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Power function
    #[wasm_bindgen]
    pub fn pow(&self, exponent: f32) -> WasmTensor {
        let result = math_funcs::pow(&self.data, exponent);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Square root
    #[wasm_bindgen]
    pub fn sqrt(&self) -> WasmTensor {
        let result = math_funcs::sqrt(&self.data);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Exponential function
    #[wasm_bindgen]
    pub fn exp(&self) -> WasmTensor {
        let result = math_funcs::exp(&self.data);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Natural logarithm
    #[wasm_bindgen]
    pub fn log(&self) -> WasmTensor {
        let result = math_funcs::log(&self.data);
        WasmTensor::new(result, self.shape.clone())
    }

    /// Sum all elements
    #[wasm_bindgen]
    pub fn sum(&self) -> f32 {
        stats_ops::sum(&self.data)
    }

    /// Mean of all elements
    #[wasm_bindgen]
    pub fn mean(&self) -> f32 {
        stats_ops::mean(&self.data)
    }

    /// Maximum element
    #[wasm_bindgen]
    pub fn max(&self) -> f32 {
        stats_ops::max(&self.data)
    }

    /// Minimum element
    #[wasm_bindgen]
    pub fn min(&self) -> f32 {
        stats_ops::min(&self.data)
    }

    /// Tanh activation
    #[wasm_bindgen]
    pub fn tanh(&self) -> WasmTensor {
        let result = activation_ops::tanh(&self.data);
        WasmTensor::new(result, self.shape.clone())
    }
}

/// Implement common tensor operations trait for WasmTensor
/// WasmTensorに共通テンソル操作トレイトを実装
#[cfg(feature = "wasm")]
impl CommonTensorOps<f32> for WasmTensor {
    type Error = JsValue;
    
    fn add_elements(&self, other: &Self) -> Result<Self, Self::Error> {
        self.add(other)
    }
    
    fn sub_elements(&self, other: &Self) -> Result<Self, Self::Error> {
        self.subtract(other)
    }
    
    fn relu_activation(&self) -> Self {
        self.relu()
    }
    
    fn sigmoid_activation(&self) -> Self {
        self.sigmoid()
    }
}
