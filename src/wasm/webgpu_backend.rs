//! WebGPU backend for Chrome browser GPU acceleration
//! Chrome ブラウザGPU加速用WebGPUバックエンド

#[cfg(feature = "webgpu")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "webgpu")]
use wgpu::*;
#[cfg(feature = "webgpu")]
use wgpu::util::{DeviceExt, BufferInitDescriptor};
#[cfg(feature = "webgpu")]
use std::collections::HashMap;
#[cfg(feature = "webgpu")]
use bytemuck;
#[cfg(feature = "webgpu")]
use js_sys;

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(feature = "webgpu")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub struct WebGPUContext {
    device: Device,
    queue: Queue,
    adapter_info: AdapterInfo,
    buffer_cache: HashMap<String, Buffer>,
    compute_pipeline_cache: HashMap<String, ComputePipeline>,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
impl WebGPUContext {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<WebGPUContext, JsValue> {
        console_error_panic_hook::set_once();
        
        // Request WebGPU adapter with Chrome optimization
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| JsValue::from_str("Failed to find WebGPU adapter"))?;

        let adapter_info = adapter.get_info();
        console_log!("WebGPU Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);

        // Request device with compute shader support
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("RusTorch WebGPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to request device: {:?}", e)))?;

        console_log!("WebGPU device initialized successfully");

        Ok(WebGPUContext {
            device,
            queue,
            adapter_info,
            buffer_cache: HashMap::new(),
            compute_pipeline_cache: HashMap::new(),
        })
    }

    #[wasm_bindgen]
    pub fn get_adapter_name(&self) -> String {
        self.adapter_info.name.clone()
    }

    #[wasm_bindgen]
    pub fn get_backend_type(&self) -> String {
        format!("{:?}", self.adapter_info.backend)
    }

    #[wasm_bindgen]
    pub fn create_buffer(&mut self, label: &str, size: u64, usage: u32) -> bool {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::from_bits_truncate(usage),
            mapped_at_creation: false,
        });

        self.buffer_cache.insert(label.to_string(), buffer);
        true
    }

    #[wasm_bindgen]
    pub fn write_buffer_data(&mut self, buffer_label: &str, data: &[f32]) -> bool {
        if let Some(buffer) = self.buffer_cache.get(buffer_label) {
            let data_bytes = bytemuck::cast_slice(data);
            self.queue.write_buffer(buffer, 0, data_bytes);
            true
        } else {
            false
        }
    }

    #[wasm_bindgen]
    pub async fn read_buffer_data(&self, buffer_label: &str) -> Result<Vec<f32>, JsValue> {
        let buffer = self.buffer_cache.get(buffer_label)
            .ok_or_else(|| JsValue::from_str("Buffer not found"))?;

        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |_| {});
        
        // Wait for the mapping to complete
        self.device.poll(Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        buffer.unmap();
        
        Ok(result)
    }

    #[wasm_bindgen]
    pub fn create_compute_pipeline(&mut self, label: &str, shader_source: &str) -> bool {
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label)),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let compute_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.compute_pipeline_cache.insert(label.to_string(), compute_pipeline);
        true
    }

    #[wasm_bindgen]
    pub fn dispatch_compute(&self, pipeline_label: &str, workgroup_x: u32, workgroup_y: u32, workgroup_z: u32) -> bool {
        if let Some(pipeline) = self.compute_pipeline_cache.get(pipeline_label) {
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
            }

            self.queue.submit([encoder.finish()]);
            true
        } else {
            false
        }
    }

    #[wasm_bindgen]
    pub fn get_buffer_count(&self) -> u32 {
        self.buffer_cache.len() as u32
    }

    #[wasm_bindgen]
    pub fn get_pipeline_count(&self) -> u32 {
        self.compute_pipeline_cache.len() as u32
    }

    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.buffer_cache.clear();
        self.compute_pipeline_cache.clear();
    }

    // Tensor operation execution functions with binding groups
    #[wasm_bindgen]
    pub async fn tensor_add(&mut self, a_label: &str, b_label: &str, output_label: &str) -> Result<bool, JsValue> {
        self.execute_binary_operation("tensor_add", a_label, b_label, output_label).await
    }

    #[wasm_bindgen]
    pub async fn tensor_mul(&mut self, a_label: &str, b_label: &str, output_label: &str) -> Result<bool, JsValue> {
        self.execute_binary_operation("tensor_mul", a_label, b_label, output_label).await
    }

    #[wasm_bindgen]
    pub async fn tensor_relu(&mut self, input_label: &str, output_label: &str) -> Result<bool, JsValue> {
        self.execute_unary_operation("tensor_relu", input_label, output_label).await
    }

    #[wasm_bindgen]
    pub async fn tensor_sigmoid(&mut self, input_label: &str, output_label: &str) -> Result<bool, JsValue> {
        self.execute_unary_operation("tensor_sigmoid", input_label, output_label).await
    }

    #[wasm_bindgen]
    pub async fn tensor_matmul(&mut self, a_label: &str, b_label: &str, output_label: &str, m: u32, n: u32, k: u32) -> Result<bool, JsValue> {
        let pipeline = self.compute_pipeline_cache.get("tensor_matmul")
            .ok_or_else(|| JsValue::from_str("Matrix multiplication pipeline not found"))?;

        let a_buffer = self.buffer_cache.get(a_label)
            .ok_or_else(|| JsValue::from_str("Buffer A not found"))?;
        let b_buffer = self.buffer_cache.get(b_label)
            .ok_or_else(|| JsValue::from_str("Buffer B not found"))?;
        let output_buffer = self.buffer_cache.get(output_label)
            .ok_or_else(|| JsValue::from_str("Output buffer not found"))?;

        // Create dimensions buffer
        let dims_data = [m, n, k];
        let dims_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: BufferUsages::STORAGE,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_x = (n + 7) / 8;
            let workgroup_y = (m + 7) / 8;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(true)
    }

    // Helper function for binary operations (add, mul)
    async fn execute_binary_operation(&mut self, operation: &str, a_label: &str, b_label: &str, output_label: &str) -> Result<bool, JsValue> {
        let pipeline = self.compute_pipeline_cache.get(operation)
            .ok_or_else(|| JsValue::from_str(&format!("Pipeline {} not found", operation)))?;

        let a_buffer = self.buffer_cache.get(a_label)
            .ok_or_else(|| JsValue::from_str("Buffer A not found"))?;
        let b_buffer = self.buffer_cache.get(b_label)
            .ok_or_else(|| JsValue::from_str("Buffer B not found"))?;
        let output_buffer = self.buffer_cache.get(output_label)
            .ok_or_else(|| JsValue::from_str("Output buffer not found"))?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", operation)),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", operation)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Pass", operation)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroups based on buffer size
            let workgroup_count = (output_buffer.size() as u32 / 4 + 63) / 64; // f32 = 4 bytes, workgroup_size = 64
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(true)
    }

    // Helper function for unary operations (relu, sigmoid)
    async fn execute_unary_operation(&mut self, operation: &str, input_label: &str, output_label: &str) -> Result<bool, JsValue> {
        let pipeline = self.compute_pipeline_cache.get(operation)
            .ok_or_else(|| JsValue::from_str(&format!("Pipeline {} not found", operation)))?;

        let input_buffer = self.buffer_cache.get(input_label)
            .ok_or_else(|| JsValue::from_str("Input buffer not found"))?;
        let output_buffer = self.buffer_cache.get(output_label)
            .ok_or_else(|| JsValue::from_str("Output buffer not found"))?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", operation)),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", operation)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Pass", operation)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroups based on buffer size
            let workgroup_count = (output_buffer.size() as u32 / 4 + 63) / 64; // f32 = 4 bytes, workgroup_size = 64
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(true)
    }
}

// WebGPU Tensor for Chrome GPU-accelerated operations
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub struct WebGPUTensor {
    data: Vec<f32>,
    shape: Vec<u32>,
    buffer_label: String,
    device_id: String,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
impl WebGPUTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<u32>, buffer_label: String) -> WebGPUTensor {
        WebGPUTensor {
            data,
            shape,
            buffer_label,
            device_id: "default".to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    #[wasm_bindgen]
    pub fn buffer_label(&self) -> String {
        self.buffer_label.clone()
    }

    #[wasm_bindgen]
    pub fn numel(&self) -> u32 {
        self.shape.iter().product()
    }

    #[wasm_bindgen]
    pub fn byte_size(&self) -> u32 {
        self.numel() * 4 // f32 = 4 bytes
    }
}

// WebGPU compute shader templates for tensor operations
#[cfg(feature = "webgpu")]
pub const TENSOR_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}
"#;

#[cfg(feature = "webgpu")]
pub const TENSOR_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] * input_b[index];
}
"#;

#[cfg(feature = "webgpu")]
pub const TENSOR_MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> dimensions: array<u32>; // [M, N, K]

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = dimensions[0];
    let N = dimensions[1];
    let K = dimensions[2];
    
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= M || col >= N) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let a_index = row * K + k;
        let b_index = k * N + col;
        sum += input_a[a_index] * input_b[b_index];
    }
    
    let output_index = row * N + col;
    output[output_index] = sum;
}
"#;

#[cfg(feature = "webgpu")]
pub const TENSOR_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = max(input[index], 0.0);
}
"#;

#[cfg(feature = "webgpu")]
pub const TENSOR_SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = 1.0 / (1.0 + exp(-input[index]));
}
"#;

#[cfg(feature = "webgpu")]
pub const TENSOR_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>; // [batch_size, features]

// Two-pass softmax: first pass finds max, second pass computes softmax
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = params[0];
    let features = params[1];
    let batch_idx = global_id.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    let start_idx = batch_idx * features;
    
    // Find max value in this batch
    var max_val: f32 = input[start_idx];
    for (var i: u32 = 1u; i < features; i++) {
        max_val = max(max_val, input[start_idx + i]);
    }
    
    // Compute sum of exponentials
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < features; i++) {
        sum_exp += exp(input[start_idx + i] - max_val);
    }
    
    // Compute softmax
    for (var i: u32 = 0u; i < features; i++) {
        let idx = start_idx + i;
        output[idx] = exp(input[idx] - max_val) / sum_exp;
    }
}
"#;

// Chrome WebGPU Performance Optimizer
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub struct ChromeWebGPUOptimizer {
    context: WebGPUContext,
    workgroup_sizes: HashMap<String, (u32, u32, u32)>,
    optimal_buffer_sizes: HashMap<String, u64>,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
impl ChromeWebGPUOptimizer {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<ChromeWebGPUOptimizer, JsValue> {
        let context = WebGPUContext::new().await?;
        
        let mut workgroup_sizes = HashMap::new();
        workgroup_sizes.insert("tensor_add".to_string(), (64, 1, 1));
        workgroup_sizes.insert("tensor_mul".to_string(), (64, 1, 1));
        workgroup_sizes.insert("tensor_matmul".to_string(), (8, 8, 1));
        workgroup_sizes.insert("tensor_relu".to_string(), (64, 1, 1));
        workgroup_sizes.insert("tensor_sigmoid".to_string(), (64, 1, 1));
        workgroup_sizes.insert("tensor_softmax".to_string(), (64, 1, 1));

        let mut optimal_buffer_sizes = HashMap::new();
        optimal_buffer_sizes.insert("small".to_string(), 1024 * 1024);      // 1MB
        optimal_buffer_sizes.insert("medium".to_string(), 16 * 1024 * 1024); // 16MB
        optimal_buffer_sizes.insert("large".to_string(), 64 * 1024 * 1024);  // 64MB

        Ok(ChromeWebGPUOptimizer {
            context,
            workgroup_sizes,
            optimal_buffer_sizes,
        })
    }

    #[wasm_bindgen]
    pub fn initialize_shaders(&mut self) -> bool {
        let shaders = [
            ("tensor_add", TENSOR_ADD_SHADER),
            ("tensor_mul", TENSOR_MUL_SHADER),
            ("tensor_matmul", TENSOR_MATMUL_SHADER),
            ("tensor_relu", TENSOR_RELU_SHADER),
            ("tensor_sigmoid", TENSOR_SIGMOID_SHADER),
            ("tensor_softmax", TENSOR_SOFTMAX_SHADER),
        ];

        for (name, source) in &shaders {
            if !self.context.create_compute_pipeline(name, source) {
                console_log!("Failed to create shader: {}", name);
                return false;
            }
        }

        console_log!("All compute shaders initialized successfully");
        true
    }

    #[wasm_bindgen]
    pub fn get_recommended_workgroup_size(&self, operation: &str, data_size: u32) -> Vec<u32> {
        if let Some(&(x, y, z)) = self.workgroup_sizes.get(operation) {
            // Adjust workgroup size based on data size for optimal performance
            match operation {
                "tensor_matmul" => {
                    let optimal_x = (data_size as f32).sqrt().ceil() as u32;
                    let optimal_y = optimal_x;
                    vec![optimal_x.min(16), optimal_y.min(16), z]
                }
                _ => {
                    let optimal_x = (data_size / 64).max(1).min(256);
                    vec![optimal_x, y, z]
                }
            }
        } else {
            vec![64, 1, 1] // Default workgroup size
        }
    }

    #[wasm_bindgen]
    pub fn get_optimal_buffer_size(&self, data_size_bytes: u64) -> u64 {
        if data_size_bytes <= self.optimal_buffer_sizes["small"] {
            self.optimal_buffer_sizes["small"]
        } else if data_size_bytes <= self.optimal_buffer_sizes["medium"] {
            self.optimal_buffer_sizes["medium"]
        } else {
            self.optimal_buffer_sizes["large"]
        }
    }

    #[wasm_bindgen]
    pub fn estimate_performance_gain(&self, operation: &str, data_size: u32) -> f32 {
        // Conservative performance estimates for Chrome WebGPU vs CPU
        match operation {
            "tensor_add" | "tensor_mul" => {
                if data_size > 1000 { 2.0 } else { 1.2 }
            }
            "tensor_matmul" => {
                if data_size > 256 { 10.0 } else if data_size > 64 { 4.0 } else { 1.5 }
            }
            "tensor_relu" | "tensor_sigmoid" => {
                if data_size > 500 { 3.0 } else { 1.5 }
            }
            "tensor_softmax" => {
                if data_size > 1000 { 5.0 } else { 2.0 }
            }
            _ => 1.0
        }
    }
}

// Error handling for WebGPU operations
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub struct WebGPUError {
    message: String,
    error_type: String,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
impl WebGPUError {
    #[wasm_bindgen(constructor)]
    pub fn new(message: String, error_type: String) -> WebGPUError {
        WebGPUError { message, error_type }
    }

    #[wasm_bindgen]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    #[wasm_bindgen]
    pub fn error_type(&self) -> String {
        self.error_type.clone()
    }
}

// WebGPU feature detection and compatibility
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn check_webgpu_support() -> bool {
    if let Some(window) = web_sys::window() {
        let navigator = window.navigator();
        // Use js_sys to access gpu property that might not be in web-sys yet
        let gpu_val = js_sys::Reflect::get(&navigator, &wasm_bindgen::JsValue::from_str("gpu"));
        if let Ok(gpu) = gpu_val {
            if !gpu.is_undefined() && !gpu.is_null() {
                console_log!("WebGPU API found in navigator");
                return true;
            }
        }
    }
    console_log!("WebGPU not supported in this browser");
    false
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub fn get_chrome_webgpu_info() -> String {
    let webgpu_available = if let Some(window) = web_sys::window() {
        let navigator = window.navigator();
        let gpu_val = js_sys::Reflect::get(&navigator, &wasm_bindgen::JsValue::from_str("gpu"));
        gpu_val.is_ok() && !gpu_val.unwrap().is_undefined()
    } else {
        false
    };

    format!(
        "Chrome WebGPU Support: {}\nRecommended for: Chrome 113+, Edge 113+\nOptimal performance: Chrome with hardware acceleration enabled",
        if webgpu_available { "Available" } else { "Not Available" }
    )
}

// Utility functions for Chrome WebGPU optimization
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub fn calculate_optimal_workgroups(total_elements: u32, workgroup_size: u32) -> u32 {
    (total_elements + workgroup_size - 1) / workgroup_size
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub fn estimate_gpu_memory_usage(tensor_count: u32, average_size_mb: f32) -> f32 {
    // Conservative estimate including buffer overhead
    tensor_count as f32 * average_size_mb * 1.5
}