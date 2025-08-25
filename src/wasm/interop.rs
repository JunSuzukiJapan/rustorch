//! JavaScript/TypeScript interoperability utilities
//! JavaScript/TypeScript相互運用ユーティリティ

#[cfg(feature = "wasm")]
use super::tensor::WasmTensor;
#[cfg(feature = "wasm")]
use js_sys::{Array, Float32Array};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// JavaScript interop utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct JsInterop;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl JsInterop {
    /// Create new JavaScript interop utility
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsInterop
    }

    /// Create tensor filled with ones
    #[wasm_bindgen]
    pub fn ones(&self, shape: js_sys::Array) -> WasmTensor {
        let shape_vec: Vec<usize> = shape
            .iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();

        let total_elements: usize = shape_vec.iter().product();
        let data = vec![1.0f32; total_elements];

        WasmTensor::new(data, shape_vec)
    }

    /// Create tensor filled with zeros
    #[wasm_bindgen]
    pub fn zeros(&self, shape: js_sys::Array) -> WasmTensor {
        let shape_vec: Vec<usize> = shape
            .iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();

        let total_elements: usize = shape_vec.iter().product();
        let data = vec![0.0f32; total_elements];

        WasmTensor::new(data, shape_vec)
    }

    /// Create random tensor
    #[wasm_bindgen]
    pub fn random_tensor(&self, shape: js_sys::Array, min: f32, max: f32) -> WasmTensor {
        let shape_vec: Vec<usize> = shape
            .iter()
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
            &format!("data: {:?}", &tensor.data()[..5.min(tensor.data().len())]).into(),
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

/// Create tensor from Float32Array
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tensor_from_float32_array(
    data: &Float32Array,
    shape: &Array,
) -> Result<WasmTensor, JsValue> {
    let data_vec: Vec<f32> = data.to_vec();
    let shape_vec: Vec<usize> = shape
        .iter()
        .map(|val| val.as_f64().unwrap_or(0.0) as usize)
        .collect();

    let expected_size: usize = shape_vec.iter().product();
    if data_vec.len() != expected_size {
        return Err(JsValue::from_str("Data size doesn't match shape"));
    }

    Ok(WasmTensor::new(data_vec, shape_vec))
}

/// Convert tensor to Float32Array
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tensor_to_float32_array(tensor: &WasmTensor) -> Float32Array {
    Float32Array::from(&tensor.data()[..])
}

/// Create tensor from nested JavaScript array
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tensor_from_nested_array(array: &JsValue) -> Result<WasmTensor, JsValue> {
    if !array.is_object() {
        return Err(JsValue::from_str("Expected an array"));
    }

    let array = Array::from(array);
    if array.length() == 0 {
        return Err(JsValue::from_str("Empty array"));
    }

    // Check if it's a 1D array
    if array.get(0).as_f64().is_some() {
        let data: Vec<f32> = array
            .iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let shape = vec![data.len()];
        return Ok(WasmTensor::new(data, shape));
    }

    // Handle 2D array
    let first_row = Array::from(&array.get(0));
    let rows = array.length() as usize;
    let cols = first_row.length() as usize;

    if cols == 0 {
        return Err(JsValue::from_str("Empty rows in array"));
    }

    let mut data = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row = Array::from(&array.get(i as u32));
        if row.length() as usize != cols {
            return Err(JsValue::from_str("Inconsistent row lengths"));
        }

        for j in 0..cols {
            let val = row.get(j as u32).as_f64().unwrap_or(0.0) as f32;
            data.push(val);
        }
    }

    Ok(WasmTensor::new(data, vec![rows, cols]))
}

/// Convert tensor to nested JavaScript array
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tensor_to_nested_array(tensor: &WasmTensor) -> Result<Array, JsValue> {
    let shape = tensor.shape();
    let data = tensor.data();

    match shape.len() {
        1 => {
            let result = Array::new();
            for value in data {
                result.push(&JsValue::from(value));
            }
            Ok(result)
        }
        2 => {
            let rows = shape[0];
            let cols = shape[1];
            let result = Array::new();

            for i in 0..rows {
                let row = Array::new();
                for j in 0..cols {
                    let idx = i * cols + j;
                    row.push(&JsValue::from(data[idx]));
                }
                result.push(&row);
            }
            Ok(result)
        }
        _ => Err(JsValue::from_str("Only 1D and 2D tensors supported")),
    }
}

/// Memory-efficient tensor slicing
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tensor_slice(tensor: &WasmTensor, start: usize, end: usize) -> Result<WasmTensor, JsValue> {
    let data = tensor.data();
    let shape = tensor.shape();

    if shape.len() != 1 {
        return Err(JsValue::from_str("Only 1D slicing supported"));
    }

    if start >= data.len() || end > data.len() || start >= end {
        return Err(JsValue::from_str("Invalid slice indices"));
    }

    let sliced_data = data[start..end].to_vec();
    Ok(WasmTensor::new(sliced_data, vec![end - start]))
}

/// Performance benchmarking utility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BenchmarkResult {
    operation: String,
    duration_ms: f64,
    throughput: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BenchmarkResult {
    /// Get operation name
    #[wasm_bindgen(getter)]
    pub fn operation(&self) -> String {
        self.operation.clone()
    }

    /// Get duration in milliseconds
    #[wasm_bindgen(getter)]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    /// Get throughput (operations per second)
    #[wasm_bindgen(getter)]
    pub fn throughput(&self) -> f64 {
        self.throughput
    }
}

/// Simple benchmark for tensor operations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn benchmark_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a = WasmTensor::random(vec![size, size]);
    let b = WasmTensor::random(vec![size, size]);

    let start = web_sys::window().unwrap().performance().unwrap().now();

    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }

    let end = web_sys::window().unwrap().performance().unwrap().now();

    let duration_ms = end - start;
    let ops_per_sec = (iterations as f64 * 1000.0) / duration_ms;
    let flops = (2.0 * size as f64 * size as f64 * size as f64) * ops_per_sec;

    BenchmarkResult {
        operation: "Matrix Multiplication".to_string(),
        duration_ms,
        throughput: flops,
    }
}
