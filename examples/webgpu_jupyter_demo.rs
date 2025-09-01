//! WebGPU Jupyter Integration Demo
//! WebGPU Jupyter統合デモ

use rustorch::tensor::Tensor;
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use web_sys::{console, window};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 RusTorch WebGPU Jupyter Demo");
    println!("===============================");

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("⚠️  This demo is designed for WASM/WebGPU targets");
        println!("Run with: wasm-pack build --features webgpu");

        // Still run CPU demo for testing
        cpu_demo_for_comparison()?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        webgpu_demo()?;
    }

    Ok(())
}

fn cpu_demo_for_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 CPU Performance Baseline:");

    let sizes = vec![128, 256, 512];

    for size in sizes {
        println!("Testing {}x{} matrix multiplication:", size, size);

        let a = Tensor::<f32>::ones(&[size, size]);
        let b = Tensor::<f32>::ones(&[size, size]);

        let start = Instant::now();
        let result = a.matmul(&b)?;
        let cpu_time = start.elapsed();

        // Calculate GFLOPS
        let flops = 2.0 * (size * size * size) as f64;
        let gflops = flops / (cpu_time.as_secs_f64() * 1e9);

        println!(
            "  CPU: {:.3}ms ({:.2} GFLOPS)",
            cpu_time.as_secs_f64() * 1000.0,
            gflops
        );

        assert_eq!(result.shape(), &[size, size]);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn webgpu_demo() -> Result<(), Box<dyn std::error::Error>> {
    console::log_1(&"🚀 Starting WebGPU demo...".into());

    // Check WebGPU availability
    let window = window().unwrap();
    let navigator = window.navigator();

    // This would be expanded with actual WebGPU calls
    console::log_1(&"📊 WebGPU tensor operations demo".into());

    // CPU fallback demo
    let sizes = vec![64, 128, 256];

    for size in sizes {
        console::log_1(&format!("Testing {}x{} matrices", size, size).into());

        let a = Tensor::<f32>::ones(&[size, size]);
        let b = Tensor::<f32>::ones(&[size, size]);

        let start = Instant::now();
        let result = a.matmul(&b)?;
        let time = start.elapsed();

        let flops = 2.0 * (size * size * size) as f64;
        let gflops = flops / (time.as_secs_f64() * 1e9);

        console::log_1(
            &format!(
                "  {}x{}: {:.3}ms ({:.2} GFLOPS)",
                size,
                size,
                time.as_secs_f64() * 1000.0,
                gflops
            )
            .into(),
        );

        assert_eq!(result.shape(), &[size, size]);
    }

    console::log_1(&"✅ WebGPU demo completed".into());
    Ok(())
}

// WASM bindings for Jupyter integration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn test_webgpu_support() -> bool {
    // This would check actual WebGPU support
    true
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn webgpu_matrix_multiply(size: usize) -> Result<f64, JsValue> {
    let a = Tensor::<f32>::ones(&[size, size]);
    let b = Tensor::<f32>::ones(&[size, size]);

    let start = Instant::now();
    let _result = a
        .matmul(&b)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let time = start.elapsed();

    Ok(time.as_secs_f64() * 1000.0) // Return time in milliseconds
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn create_tensor_f32(data: &[f32], shape: &[usize]) -> Result<String, JsValue> {
    let tensor = Tensor::<f32>::from_vec(data.to_vec(), shape.to_vec());
    Ok(format!("Tensor created with shape: {:?}", tensor.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_demo() {
        let result = cpu_demo_for_comparison();
        assert!(result.is_ok());
    }
}
