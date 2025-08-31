//! Simple WebGPU Chrome Demo with CPU fallback
//! CPU フォールバック付きシンプルWebGPU Chrome デモ

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use rustorch::wasm::webgpu_simple::*;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log_demo(s: &str);
    
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_demo_main(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn time(name: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn time_end(name: &str);
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
macro_rules! demo_log {
    ($($t:tt)*) => (log_demo(&format_args!($($t)*).to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub struct WebGPUSimpleDemo {
    engine: Option<WebGPUSimple>,
    results: Vec<String>,
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
impl WebGPUSimpleDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebGPUSimpleDemo {
        demo_log!("🚀 WebGPU Simple Demo initialized");
        
        WebGPUSimpleDemo {
            engine: None,
            results: Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<String, JsValue> {
        let mut engine = WebGPUSimple::new();
        
        match engine.initialize().await {
            Ok(message) => {
                demo_log!("✅ Engine initialized: {}", message);
                self.engine = Some(engine);
                Ok(message)
            }
            Err(e) => {
                demo_log!("⚠️ WebGPU failed, using CPU fallback: {:?}", e);
                self.engine = Some(engine); // Still use engine for CPU operations
                Ok("⚠️ WebGPU unavailable, using CPU fallback".to_string())
            }
        }
    }

    #[wasm_bindgen]
    pub fn run_tensor_addition_demo(&mut self) -> Result<String, JsValue> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        demo_log!("🧮 Running tensor addition demo...");
        time("tensor_add_demo");

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        
        let result = engine.tensor_add_cpu(a.clone(), b.clone())?;
        let expected = vec![1.5, 3.5, 5.5, 7.5, 9.5];
        
        time_end("tensor_add_demo");

        let message = format!(
            "✅ Tensor Addition:\n  A: {:?}\n  B: {:?}\n  Result: {:?}\n  Expected: {:?}\n  ✅ Match: {}",
            a, b, result, expected,
            result.iter().zip(expected.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
        );
        
        demo_log!("{}", message);
        self.results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub fn run_matrix_multiplication_demo(&mut self) -> Result<String, JsValue> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        demo_log!("🧮 Running matrix multiplication demo...");
        time("matrix_mul_demo");

        // 2x3 × 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [[7,8], [9,10], [11,12]]
        
        let result = engine.matrix_multiply_cpu(a.clone(), b.clone(), 2, 2, 3)?;
        let expected = vec![58.0, 64.0, 139.0, 154.0]; // Manually calculated
        
        time_end("matrix_mul_demo");

        let message = format!(
            "✅ Matrix Multiplication (2x3 × 3x2):\n  A: {:?}\n  B: {:?}\n  Result (2x2): {:?}\n  Expected: {:?}\n  ✅ Match: {}",
            a, b, result, expected,
            result.iter().zip(expected.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
        );
        
        demo_log!("{}", message);
        self.results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub fn run_activation_functions_demo(&mut self) -> Result<String, JsValue> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        demo_log!("🧮 Running activation functions demo...");
        time("activation_demo");

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        
        let relu_result = engine.relu_cpu(input.clone());
        let sigmoid_result = engine.sigmoid_cpu(input.clone());
        
        let expected_relu = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        let expected_sigmoid: Vec<f32> = input.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        time_end("activation_demo");

        let message = format!(
            "✅ Activation Functions:\n  Input: {:?}\n  ReLU: {:?}\n  Expected ReLU: {:?}\n  Sigmoid: {:?}\n  ✅ ReLU Match: {}\n  ✅ Sigmoid Match: {}",
            input, relu_result, expected_relu, sigmoid_result,
            relu_result.iter().zip(expected_relu.iter()).all(|(x, y)| (x - y).abs() < 1e-6),
            sigmoid_result.iter().zip(expected_sigmoid.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
        );
        
        demo_log!("{}", message);
        self.results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub fn run_performance_benchmark(&mut self) -> Result<String, JsValue> {
        demo_log!("📊 Running performance benchmark...");
        
        let test_sizes = vec![100, 1000, 10000];
        let mut benchmark_results = Vec::new();

        for size in test_sizes {
            demo_log!("🔬 Testing size: {}", size);
            
            let estimate_add = calculate_performance_estimate("add", size);
            let estimate_matmul = calculate_performance_estimate("matmul", size);
            
            let result = format!(
                "  Size {}: Add {}x, MatMul {}x speedup estimate",
                size, estimate_add, estimate_matmul
            );
            demo_log!("{}", result);
            benchmark_results.push(result);
        }

        let message = format!(
            "📊 Performance Estimates (WebGPU vs CPU):\n{}",
            benchmark_results.join("\n")
        );
        
        demo_log!("{}", message);
        self.results.push(message.clone());
        Ok(message)
    }

    #[wasm_bindgen]
    pub async fn run_comprehensive_demo(&mut self) -> Result<String, JsValue> {
        demo_log!("🎯 Starting comprehensive WebGPU demo...");
        
        let init_result = self.initialize().await?;
        let add_result = self.run_tensor_addition_demo()?;
        let matmul_result = self.run_matrix_multiplication_demo()?;
        let activation_result = self.run_activation_functions_demo()?;
        let benchmark_result = self.run_performance_benchmark()?;

        let info = self.engine.as_ref().unwrap().get_chrome_info();
        let browser_info = get_browser_webgpu_info();

        let summary = format!(
            "🎉 WebGPU Chrome Demo Complete!\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
            info, browser_info, init_result, add_result, matmul_result, activation_result, benchmark_result
        );

        demo_log!("{}", summary);
        Ok(summary)
    }

    #[wasm_bindgen]
    pub fn get_all_results(&self) -> Vec<String> {
        self.results.clone()
    }

    #[wasm_bindgen]
    pub fn cleanup(&mut self) {
        self.results.clear();
        self.engine = None;
        demo_log!("🧹 Demo cleanup completed");
    }
}

// Simple entry point for WASM
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    demo_log!("🌟 RusTorch WebGPU Simple Demo Starting...");
    demo_log!("🎮 Demo runner available for JavaScript integration");
}

// Main function for all targets
fn main() {
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    {
        // WASM initialization is handled by wasm_main
        return;
    }
    
    #[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
    {
        println!("❌ This demo requires WebGPU feature and WASM target.");
        println!("📝 Build with: wasm-pack build --target web --features webgpu");
    }
}