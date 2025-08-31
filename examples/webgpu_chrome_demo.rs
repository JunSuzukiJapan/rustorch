//! WebGPU Chrome Browser Demo
//! Chrome ブラウザ用WebGPU デモ
//!
//! This example demonstrates WebGPU acceleration for tensor operations
//! specifically optimized for Google Chrome browser.
//!
//! To run this demo:
//! 1. Build with WebGPU features: cargo build --target wasm32-unknown-unknown --features webgpu
//! 2. Use wasm-pack: wasm-pack build --target web --features webgpu
//! 3. Serve with a local web server in Chrome 113+ with WebGPU enabled

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use rustorch::wasm::webgpu_backend::*;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use rustorch::wasm::webgpu_tensor::*;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn time(name: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn time_end(name: &str);
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub struct WebGPUChromeDemoRunner {
    engine: Option<WebGPUTensorEngine>,
    demo_results: Vec<String>,
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
impl WebGPUChromeDemoRunner {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebGPUChromeDemoRunner {
        console_error_panic_hook::set_once();
        console_log!("🚀 WebGPU Chrome Demo Runner initialized");

        WebGPUChromeDemoRunner {
            engine: None,
            demo_results: Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<String, JsValue> {
        console_log!("🔍 Checking WebGPU browser support...");

        // Check WebGPU support
        let webgpu_support = webgpu_check_browser_support().await;
        if !webgpu_support {
            let message = "❌ WebGPU not supported in this browser. Please use Chrome 113+ with WebGPU enabled.";
            console_log!("{}", message);
            return Ok(message.to_string());
        }

        console_log!("✅ WebGPU supported! Initializing engine...");

        // Initialize WebGPU engine
        match WebGPUTensorEngine::new().await {
            Ok(engine) => {
                let info = engine.get_adapter_info();
                console_log!("🎮 {}", info);

                self.engine = Some(engine);
                let browser_info = webgpu_get_browser_info();
                console_log!("🌐 {}", browser_info);

                Ok(format!(
                    "✅ WebGPU engine initialized successfully\n{}\n{}",
                    info, browser_info
                ))
            }
            Err(e) => {
                let message = format!("❌ Failed to initialize WebGPU engine: {:?}", e);
                console_log!("{}", message);
                Err(e)
            }
        }
    }

    #[wasm_bindgen]
    pub async fn run_tensor_add_demo(&mut self) -> Result<String, JsValue> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        console_log!("🧮 Running tensor addition demo...");
        time("tensor_add_demo");

        // Create test tensors
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
        let shape = vec![2, 3];

        // Execute WebGPU tensor addition
        let result = webgpu_tensor_add_f32(engine, a_data.clone(), b_data.clone(), shape).await?;

        time_end("tensor_add_demo");

        // Calculate expected result for verification
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let message = format!(
            "✅ Tensor Addition Demo Complete:\n  Input A: {:?}\n  Input B: {:?}\n  WebGPU Result: {:?}\n  Expected: {:?}\n  ✅ Results Match: {}",
            a_data, b_data, result, expected,
            result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
        );

        console_log!("{}", message);
        self.demo_results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub async fn run_matrix_multiplication_demo(&mut self) -> Result<String, JsValue> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        console_log!("🧮 Running matrix multiplication demo...");
        time("matrix_mul_demo");

        // Create 2x3 and 3x2 matrices for multiplication
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix
        let m = 2; // rows of A
        let n = 2; // cols of B
        let k = 3; // cols of A = rows of B

        // Execute WebGPU matrix multiplication
        let result =
            webgpu_tensor_matmul_f32(engine, a_data.clone(), b_data.clone(), m, n, k).await?;

        time_end("matrix_mul_demo");

        // Calculate expected result manually: C[i,j] = sum(A[i,k] * B[k,j])
        let expected = vec![
            1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,  // C[0,0] = 58
            1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0, // C[0,1] = 64
            4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,  // C[1,0] = 139
            4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0, // C[1,1] = 154
        ];

        let message = format!(
            "✅ Matrix Multiplication Demo Complete:\n  Matrix A (2x3): {:?}\n  Matrix B (3x2): {:?}\n  WebGPU Result (2x2): {:?}\n  Expected (2x2): {:?}\n  ✅ Results Match: {}",
            a_data, b_data, result, expected,
            result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
        );

        console_log!("{}", message);
        self.demo_results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub async fn run_activation_functions_demo(&mut self) -> Result<String, JsValue> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        console_log!("🧮 Running activation functions demo...");
        time("activation_demo");

        // Create test data with negative and positive values
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let shape = vec![2, 3];

        // Upload input tensor
        let input_label = engine.upload_tensor_f32(input_data.clone(), shape);

        // Test ReLU
        let relu_label = engine.relu_tensor(&input_label).await?;
        let relu_result = engine.download_tensor_f32(&relu_label).await?;

        // Test Sigmoid
        let sigmoid_label = engine.sigmoid_tensor(&input_label).await?;
        let sigmoid_result = engine.download_tensor_f32(&sigmoid_label).await?;

        time_end("activation_demo");

        // Calculate expected results
        let expected_relu: Vec<f32> = input_data.iter().map(|&x| x.max(0.0)).collect();
        let expected_sigmoid: Vec<f32> = input_data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let message = format!(
            "✅ Activation Functions Demo Complete:\n  Input: {:?}\n  ReLU Result: {:?}\n  ReLU Expected: {:?}\n  Sigmoid Result: {:?}\n  Sigmoid Expected: {:?}\n  ✅ ReLU Match: {}\n  ✅ Sigmoid Match: {}",
            input_data, relu_result, expected_relu, sigmoid_result, expected_sigmoid,
            relu_result.iter().zip(expected_relu.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            sigmoid_result.iter().zip(expected_sigmoid.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
        );

        console_log!("{}", message);
        self.demo_results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub async fn run_performance_benchmark(&mut self) -> Result<String, JsValue> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        console_log!("📊 Running performance benchmark...");

        let test_sizes = vec![100, 1000, 10000];
        let mut benchmark_results = Vec::new();

        for size in test_sizes {
            console_log!("🔬 Testing tensor size: {}", size);

            // Generate test data
            let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2 + 1.0).collect();
            let shape = vec![size as u32];

            // Benchmark tensor addition
            let perf_estimate = engine.get_performance_estimate("tensor_add", size as u32);

            time(&format!("add_benchmark_{}", size));
            let _result = webgpu_tensor_add_f32(engine, data_a, data_b, shape).await?;
            time_end(&format!("add_benchmark_{}", size));

            let result_msg = format!(
                "  Size {}: Estimated {}x speedup over CPU",
                size, perf_estimate
            );
            console_log!("{}", result_msg);
            benchmark_results.push(result_msg);
        }

        let message = format!(
            "📊 Performance Benchmark Complete:\n{}",
            benchmark_results.join("\n")
        );

        console_log!("{}", message);
        self.demo_results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub async fn run_comprehensive_demo(&mut self) -> Result<String, JsValue> {
        console_log!("🎯 Starting comprehensive WebGPU Chrome demo...");

        // Run all demo components
        let init_result = self.initialize().await?;
        let add_result = self.run_tensor_add_demo().await?;
        let matmul_result = self.run_matrix_multiplication_demo().await?;
        let activation_result = self.run_activation_functions_demo().await?;
        let benchmark_result = self.run_performance_benchmark().await?;

        let summary = format!(
            "🎉 WebGPU Chrome Demo Complete!\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
            init_result, add_result, matmul_result, activation_result, benchmark_result
        );

        console_log!("{}", summary);
        Ok(summary)
    }

    #[wasm_bindgen]
    pub fn get_demo_results(&self) -> Vec<String> {
        self.demo_results.clone()
    }

    #[wasm_bindgen]
    pub fn cleanup(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.cleanup();
        }
        self.demo_results.clear();
        console_log!("🧹 Demo cleanup completed");
    }
}

// Entry point for WASM demo
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(start)]
pub async fn main() -> Result<(), JsValue> {
    console_log!("🌟 RusTorch WebGPU Chrome Demo Starting...");

    let mut demo = WebGPUChromeDemoRunner::new();
    match demo.run_comprehensive_demo().await {
        Ok(summary) => {
            console_log!("✨ Demo completed successfully!");
            console_log!("{}", summary);
        }
        Err(e) => {
            console_log!("❌ Demo failed: {:?}", e);
        }
    }

    Ok(())
}

// Non-WASM fallback
#[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
fn main() {
    println!("❌ This demo requires WebGPU feature and WASM target.");
    println!("📝 To run this demo:");
    println!("   1. Install wasm-pack: cargo install wasm-pack");
    println!("   2. Build: wasm-pack build --target web --features webgpu");
    println!("   3. Serve with local web server in Chrome 113+");
    println!("   4. Open browser developer console to see demo output");
}
