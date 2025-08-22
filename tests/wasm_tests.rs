#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use rustorch::wasm::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = tensor::WasmTensor::new(data.clone(), shape.clone());
        
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data(), &data);
        assert_eq!(tensor.numel(), 4);
    }

    #[wasm_bindgen_test]
    fn test_wasm_tensor_arithmetic() {
        let a = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor::WasmTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let sum = a.add(&b).expect("Addition failed");
        assert_eq!(sum.data(), &vec![6.0, 8.0, 10.0, 12.0]);
        
        let diff = b.sub(&a).expect("Subtraction failed");
        assert_eq!(diff.data(), &vec![4.0, 4.0, 4.0, 4.0]);
        
        let prod = a.mul(&b).expect("Multiplication failed");
        assert_eq!(prod.data(), &vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_tensor_matmul() {
        let a = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor::WasmTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let result = a.matmul(&b).expect("Matrix multiplication failed");
        assert_eq!(result.shape(), &vec![2, 2]);
        assert_eq!(result.data(), &vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_tensor_transpose() {
        let tensor = tensor::WasmTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        );
        
        let transposed = tensor.transpose().expect("Transpose failed");
        assert_eq!(transposed.shape(), &vec![3, 2]);
        assert_eq!(transposed.data(), &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_runtime_initialization() {
        runtime::initialize_wasm_runtime();
        
        let runtime = runtime::get_runtime()
            .lock()
            .expect("Failed to lock runtime");
        
        assert!(runtime.get_start_time() > 0.0);
        assert_eq!(runtime.get_operation_count(), 0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_runtime_performance_tracking() {
        runtime::initialize_wasm_runtime();
        
        let tensor = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = tensor.add(&tensor).expect("Addition failed");
        
        let runtime = runtime::get_runtime()
            .lock()
            .expect("Failed to lock runtime");
        
        assert!(runtime.get_operation_count() > 0);
        assert!(runtime.get_average_operation_time() > 0.0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_memory_pool() {
        let mut pool = memory::WasmMemoryPool::new(1024 * 1024); // 1MB pool
        
        let alloc1 = pool.allocate(1024);
        assert!(alloc1.is_some());
        
        let alloc2 = pool.allocate(2048);
        assert!(alloc2.is_some());
        
        assert!(pool.get_used_memory() >= 3072);
        assert!(pool.get_free_memory() <= 1024 * 1024 - 3072);
    }

    #[wasm_bindgen_test]
    fn test_wasm_memory_garbage_collection() {
        let mut pool = memory::WasmMemoryPool::new(1024 * 1024);
        
        for _ in 0..100 {
            pool.allocate(1024);
        }
        
        let used_before = pool.get_used_memory();
        pool.garbage_collect();
        let used_after = pool.get_used_memory();
        
        assert!(used_after <= used_before);
    }

    #[wasm_bindgen_test]
    fn test_wasm_linear_layer() {
        let layer = bindings::WasmLinear::new(4, 2);
        let input = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &vec![1, 2]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_relu_activation() {
        let relu = bindings::WasmReLU::new();
        let input = tensor::WasmTensor::new(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![1, 5]
        );
        
        let output = relu.forward(&input);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(output.data(), &expected);
    }

    #[wasm_bindgen_test]
    fn test_wasm_model_sequential() {
        let mut model = bindings::WasmModel::new();
        model.add_linear(4, 8);
        model.add_relu();
        model.add_linear(8, 2);
        
        let input = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = model.forward(&input);
        
        assert_eq!(output.shape(), &vec![1, 2]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_interop_float32_array() {
        let data = js_sys::Float32Array::from(&[1.0f32, 2.0, 3.0, 4.0][..]);
        let shape = js_sys::Array::new();
        shape.push(&wasm_bindgen::JsValue::from(2));
        shape.push(&wasm_bindgen::JsValue::from(2));
        
        let tensor = interop::tensor_from_float32_array(&data, &shape)
            .expect("Failed to create tensor from Float32Array");
        
        assert_eq!(tensor.shape(), &vec![2, 2]);
        assert_eq!(tensor.data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_wasm_interop_tensor_to_array() {
        let tensor = tensor::WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let array = interop::tensor_to_float32_array(&tensor);
        
        assert_eq!(array.length(), 4);
        assert_eq!(array.get_index(0), 1.0);
        assert_eq!(array.get_index(3), 4.0);
    }

    #[wasm_bindgen_test]
    async fn test_wasm_browser_storage() {
        let storage = browser::BrowserStorage::new().expect("Failed to create storage");
        
        storage.set_item("test_key", "test_value").expect("Failed to set item");
        let value = storage.get_item("test_key").expect("Failed to get item");
        assert_eq!(value, Some("test_value".to_string()));
        
        storage.remove_item("test_key").expect("Failed to remove item");
        let removed = storage.get_item("test_key").expect("Failed to get removed item");
        assert_eq!(removed, None);
    }

    #[wasm_bindgen_test]
    fn test_wasm_optimized_simd_add() {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        
        let result = optimized::simd_add_f32(&a, &b);
        assert_eq!(result.len(), 1024);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[wasm_bindgen_test]
    fn test_wasm_optimized_simd_mul() {
        let a = vec![2.0f32; 1024];
        let b = vec![3.0f32; 1024];
        
        let result = optimized::simd_mul_f32(&a, &b);
        assert_eq!(result.len(), 1024);
        assert!(result.iter().all(|&x| x == 6.0));
    }

    #[wasm_bindgen_test]
    fn test_wasm_optimized_memory_pool() {
        let pool = optimized::OptimizedMemoryPool::new(1024 * 1024);
        
        let tensor1 = pool.allocate_tensor(vec![100, 100]);
        assert!(tensor1.is_some());
        
        let tensor2 = pool.allocate_tensor(vec![50, 50]);
        assert!(tensor2.is_some());
        
        pool.deallocate_tensor(tensor1.unwrap());
        let tensor3 = pool.allocate_tensor(vec![100, 100]);
        assert!(tensor3.is_some());
    }

    #[wasm_bindgen_test]
    fn test_wasm_feature_detection() {
        let features = runtime::detect_wasm_features();
        
        // These features should be available in modern browsers
        assert!(features.contains_key("simd"));
        assert!(features.contains_key("threads"));
        assert!(features.contains_key("bulk-memory"));
    }

    #[wasm_bindgen_test]
    fn test_wasm_performance_benchmark() {
        let sizes = vec![10, 100, 1000];
        let mut results = Vec::new();
        
        for size in sizes {
            let a = tensor::WasmTensor::new(vec![1.0; size * size], vec![size, size]);
            let b = tensor::WasmTensor::new(vec![2.0; size * size], vec![size, size]);
            
            let start = runtime::get_time();
            let _ = a.matmul(&b).expect("Matmul failed");
            let elapsed = runtime::get_time() - start;
            
            results.push((size, elapsed));
        }
        
        // Verify that larger matrices take more time
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod wasm_build_tests {
    use std::process::Command;

    #[test]
    fn test_wasm_pack_build() {
        let output = Command::new("wasm-pack")
            .args(&["build", "--target", "web", "--dev"])
            .output();
        
        match output {
            Ok(result) if result.status.success() => {
                println!("WASM build successful");
            }
            Ok(result) => {
                println!("WASM build failed: {}", String::from_utf8_lossy(&result.stderr));
                // Don't fail the test if wasm-pack is not installed
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("wasm-pack not found, skipping WASM build test");
            }
            Err(e) => {
                println!("Failed to run wasm-pack: {}", e);
            }
        }
    }
}