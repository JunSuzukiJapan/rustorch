//! WebGPU-accelerated tensor operations for Chrome browser
//! Chrome ブラウザ用WebGPU加速テンソル演算

#[cfg(feature = "webgpu")]
use super::backend::WebGPUContext;
#[cfg(feature = "webgpu")]
use crate::tensor::Tensor;
#[cfg(feature = "webgpu")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "webgpu")]
use wgpu;

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub struct WebGPUTensorEngine {
    context: WebGPUContext,
    buffer_counter: u32,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
impl WebGPUTensorEngine {
    #[wasm_bindgen]
    impl WebGPUTensorEngine {
        #[wasm_bindgen]
        pub async fn create() -> Result<WebGPUTensorEngine, JsValue> {
            let mut context = WebGPUContext::new().await?;

            // Initialize all compute shaders
            context.create_compute_pipeline("tensor_add", super::backend::TENSOR_ADD_SHADER);
            context.create_compute_pipeline("tensor_mul", super::backend::TENSOR_MUL_SHADER);
            context.create_compute_pipeline("tensor_matmul", super::backend::TENSOR_MATMUL_SHADER);
            context.create_compute_pipeline("tensor_relu", super::backend::TENSOR_RELU_SHADER);
            context.create_compute_pipeline("tensor_sigmoid", super::backend::TENSOR_SIGMOID_SHADER);
            context.create_compute_pipeline("tensor_softmax", super::backend::TENSOR_SOFTMAX_SHADER);

            Ok(WebGPUTensorEngine {
                context,
                buffer_counter: 0,
            })
        }
    }

    #[wasm_bindgen]
    pub fn upload_tensor_f32(&mut self, data: Vec<f32>, shape: Vec<u32>) -> String {
        let buffer_label = format!("tensor_{}", self.buffer_counter);
        self.buffer_counter += 1;

        let size = data.len() as u64 * 4; // f32 = 4 bytes
        let usage = wgpu::BufferUsages::STORAGE.bits();

        // Create buffer
        self.context.create_buffer(&buffer_label, size, usage);

        // Upload data
        self.context.write_buffer_data(&buffer_label, &data);

        buffer_label
    }

    #[wasm_bindgen]
    pub async fn download_tensor_f32(&self, buffer_label: &str) -> Result<Vec<f32>, JsValue> {
        self.context.read_buffer_data(buffer_label).await
    }

    #[wasm_bindgen]
    pub async fn add_tensors(&mut self, a_label: &str, b_label: &str) -> Result<String, JsValue> {
        let output_label = format!("add_result_{}", self.buffer_counter);
        self.buffer_counter += 1;

        // Create output buffer with same size as input
        let a_data = self.context.read_buffer_data(a_label).await?;
        let size = a_data.len() as u64 * 4;
        let usage = wgpu::BufferUsages::STORAGE.bits();

        self.context.create_buffer(&output_label, size, usage);

        // Execute addition
        self.context
            .tensor_add(a_label, b_label, &output_label)
            .await?;

        Ok(output_label)
    }

    #[wasm_bindgen]
    pub async fn mul_tensors(&mut self, a_label: &str, b_label: &str) -> Result<String, JsValue> {
        let output_label = format!("mul_result_{}", self.buffer_counter);
        self.buffer_counter += 1;

        // Create output buffer with same size as input
        let a_data = self.context.read_buffer_data(a_label).await?;
        let size = a_data.len() as u64 * 4;
        let usage = wgpu::BufferUsages::STORAGE.bits();

        self.context.create_buffer(&output_label, size, usage);

        // Execute multiplication
        self.context
            .tensor_mul(a_label, b_label, &output_label)
            .await?;

        Ok(output_label)
    }

    #[wasm_bindgen]
    pub async fn matmul_tensors(
        &mut self,
        a_label: &str,
        b_label: &str,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<String, JsValue> {
        let output_label = format!("matmul_result_{}", self.buffer_counter);
        self.buffer_counter += 1;

        // Create output buffer for M×N result
        let size = (m * n) as u64 * 4; // f32 = 4 bytes
        let usage = wgpu::BufferUsages::STORAGE.bits();

        self.context.create_buffer(&output_label, size, usage);

        // Execute matrix multiplication
        self.context
            .tensor_matmul(a_label, b_label, &output_label, m, n, k)
            .await?;

        Ok(output_label)
    }

    #[wasm_bindgen]
    pub async fn relu_tensor(&mut self, input_label: &str) -> Result<String, JsValue> {
        let output_label = format!("relu_result_{}", self.buffer_counter);
        self.buffer_counter += 1;

        // Create output buffer with same size as input
        let input_data = self.context.read_buffer_data(input_label).await?;
        let size = input_data.len() as u64 * 4;
        let usage = wgpu::BufferUsages::STORAGE.bits();

        self.context.create_buffer(&output_label, size, usage);

        // Execute ReLU
        self.context.tensor_relu(input_label, &output_label).await?;

        Ok(output_label)
    }

    #[wasm_bindgen]
    pub async fn sigmoid_tensor(&mut self, input_label: &str) -> Result<String, JsValue> {
        let output_label = format!("sigmoid_result_{}", self.buffer_counter);
        self.buffer_counter += 1;

        // Create output buffer with same size as input
        let input_data = self.context.read_buffer_data(input_label).await?;
        let size = input_data.len() as u64 * 4;
        let usage = wgpu::BufferUsages::STORAGE.bits();

        self.context.create_buffer(&output_label, size, usage);

        // Execute Sigmoid
        self.context
            .tensor_sigmoid(input_label, &output_label)
            .await?;

        Ok(output_label)
    }

    #[wasm_bindgen]
    pub fn get_performance_estimate(&self, operation: &str, data_size: u32) -> f32 {
        // Conservative performance estimates for Chrome WebGPU vs CPU
        match operation {
            "tensor_add" | "tensor_mul" => {
                if data_size > 1000 {
                    2.0
                } else {
                    1.2
                }
            }
            "tensor_matmul" => {
                if data_size > 256 {
                    10.0
                } else if data_size > 64 {
                    4.0
                } else {
                    1.5
                }
            }
            "tensor_relu" | "tensor_sigmoid" => {
                if data_size > 500 {
                    3.0
                } else {
                    1.5
                }
            }
            "tensor_softmax" => {
                if data_size > 1000 {
                    5.0
                } else {
                    2.0
                }
            }
            _ => 1.0,
        }
    }

    #[wasm_bindgen]
    pub fn get_adapter_info(&self) -> String {
        format!(
            "Adapter: {} ({})",
            self.context.get_adapter_name(),
            self.context.get_backend_type()
        )
    }

    #[wasm_bindgen]
    pub fn cleanup(&mut self) {
        self.context.clear_cache();
        self.buffer_counter = 0;
    }
}

// Utility functions for integrating with RusTorch Tensor API
#[cfg(feature = "webgpu")]
impl WebGPUTensorEngine {
    /// Convert RusTorch Tensor<f32> to WebGPU buffer
    pub fn tensor_to_webgpu(&mut self, tensor: &Tensor<f32>) -> String {
        let data: Vec<f32> = tensor.data.iter().cloned().collect();
        let shape: Vec<u32> = tensor.data.shape().iter().map(|&x| x as u32).collect();
        self.upload_tensor_f32(data, shape)
    }

    /// Convert WebGPU buffer back to RusTorch Tensor<f32>
    pub async fn webgpu_to_tensor(
        &self,
        buffer_label: &str,
        shape: Vec<usize>,
    ) -> Result<Tensor<f32>, JsValue> {
        let data = self.download_tensor_f32(buffer_label).await?;

        match Tensor::try_from_vec(data, shape) {
            Ok(tensor) => Ok(tensor),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to create tensor: {}",
                e
            ))),
        }
    }
}

// High-level WebGPU tensor operations that integrate with RusTorch API
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn webgpu_tensor_add_f32(
    engine: &mut WebGPUTensorEngine,
    a_data: Vec<f32>,
    b_data: Vec<f32>,
    shape: Vec<u32>,
) -> Result<Vec<f32>, JsValue> {
    if a_data.len() != b_data.len() {
        return Err(JsValue::from_str("Tensor dimensions must match"));
    }

    let a_label = engine.upload_tensor_f32(a_data, shape.clone());
    let b_label = engine.upload_tensor_f32(b_data, shape);

    let result_label = engine.add_tensors(&a_label, &b_label).await?;
    let result = engine.download_tensor_f32(&result_label).await?;

    Ok(result)
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn webgpu_tensor_matmul_f32(
    engine: &mut WebGPUTensorEngine,
    a_data: Vec<f32>,
    b_data: Vec<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<f32>, JsValue> {
    let a_shape = vec![m, k];
    let b_shape = vec![k, n];

    let a_label = engine.upload_tensor_f32(a_data, a_shape);
    let b_label = engine.upload_tensor_f32(b_data, b_shape);

    let result_label = engine.matmul_tensors(&a_label, &b_label, m, n, k).await?;
    let result = engine.download_tensor_f32(&result_label).await?;

    Ok(result)
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn webgpu_check_browser_support() -> bool {
    super::backend::check_webgpu_support().await
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub fn webgpu_get_browser_info() -> String {
    super::backend::get_chrome_webgpu_info()
}
