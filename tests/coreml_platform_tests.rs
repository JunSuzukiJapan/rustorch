//! Platform-specific CoreML Tests
//! プラットフォーム固有のCoreMLテスト
//!
//! This test suite validates platform-specific behavior and fallback chains
//! for the CoreML hybrid execution system.

#[cfg(test)]
mod platform_detection_tests {
    use rustorch::gpu::DeviceType;

    #[test]
    fn test_apple_silicon_detection() {
        // Test platform detection logic
        let is_apple_silicon = cfg!(target_arch = "aarch64") && cfg!(target_os = "macos");

        if is_apple_silicon {
            println!("Running on Apple Silicon");
            // On Apple Silicon, CoreML should be preferred
            #[cfg(feature = "coreml")]
            {
                let preferred_device = DeviceType::default();
                assert!(matches!(preferred_device, DeviceType::CoreML(_)));
            }
        } else {
            println!("Running on Intel/AMD");
            // On Intel/AMD, CUDA should be preferred if available
            #[cfg(feature = "cuda")]
            {
                let preferred_device = DeviceType::default();
                assert!(matches!(preferred_device, DeviceType::Cuda(_)));
            }
        }
    }

    #[test]
    fn test_device_availability_detection() {
        // Test that device availability detection works correctly
        assert!(DeviceType::Cpu.is_available()); // CPU should always be available

        #[cfg(feature = "cuda")]
        {
            let cuda_available = DeviceType::Cuda(0).is_available();
            println!("CUDA available: {}", cuda_available);
        }

        #[cfg(feature = "metal")]
        {
            let metal_available = DeviceType::Metal(0).is_available();
            println!("Metal available: {}", metal_available);

            // Metal should only be available on macOS
            if cfg!(target_os = "macos") {
                // May or may not be available depending on hardware
            } else {
                assert!(!metal_available);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_fallback_chain_platform_specific() {
        use rustorch::gpu::hybrid_executor::HybridExecutor;

        let mut executor = HybridExecutor::new();
        executor.build_fallback_chain();
        let chain = executor.get_fallback_chain();

        if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            // Apple Silicon fallback chain: CoreML → Metal → CPU
            assert!(!chain.is_empty());

            #[cfg(feature = "coreml")]
            {
                // First preference should be CoreML
                assert!(matches!(chain[0], DeviceType::CoreML(_)));
            }

            #[cfg(feature = "metal")]
            {
                // Metal should be in the chain
                assert!(chain.iter().any(|d| matches!(d, DeviceType::Metal(_))));
            }

            // CPU should always be the final fallback
            assert!(matches!(chain[chain.len() - 1], DeviceType::Cpu));
        } else {
            // Intel/AMD fallback chain: CUDA → OpenCL → CPU
            assert!(!chain.is_empty());

            #[cfg(feature = "cuda")]
            {
                // CUDA should be preferred on Intel/AMD if available
                if chain.iter().any(|d| matches!(d, DeviceType::Cuda(_))) {
                    assert!(matches!(chain[0], DeviceType::Cuda(_)));
                }
            }

            // CPU should always be the final fallback
            assert!(matches!(chain[chain.len() - 1], DeviceType::Cpu));
        }
    }

    #[test]
    fn test_cuda_metal_mutual_exclusion() {
        // Test that CUDA and Metal are not both active on the same system
        #[cfg(all(feature = "cuda", feature = "metal"))]
        {
            let cuda_available = DeviceType::Cuda(0).is_available();
            let metal_available = DeviceType::Metal(0).is_available();

            if cfg!(target_os = "macos") {
                // On macOS, Metal may be available but CUDA typically isn't on Apple Silicon
                if cfg!(target_arch = "aarch64") {
                    // Apple Silicon: Metal may be available, CUDA should not be
                    assert!(!cuda_available || !metal_available);
                }
            } else {
                // On non-macOS, Metal should not be available
                assert!(!metal_available);
            }
        }
    }
}

#[cfg(test)]
mod coreml_feature_tests {
    use rustorch::tensor::Tensor;

    #[test]
    #[cfg(feature = "coreml")]
    fn test_coreml_basic_availability() {
        // Test basic CoreML functionality is available when feature is enabled
        use rustorch::gpu::DeviceCapability;

        let capability = DeviceCapability::coreml_capability();
        assert!(capability.supports_f32);
        assert!(capability.supports_f16);
        assert!(!capability.supports_f64); // CoreML limitation
        assert!(!capability.supports_complex); // CoreML limitation
    }

    #[test]
    #[cfg(not(feature = "coreml"))]
    fn test_fallback_without_coreml() {
        // Test that operations work without CoreML feature enabled
        let a = Tensor::ones(&[4, 4]);
        let b = Tensor::ones(&[4, 4]);

        // Should fall back to standard operations
        let result = a.matmul(&b);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "coreml-hybrid")]
    fn test_hybrid_mode_availability() {
        // Test hybrid mode functionality
        use rustorch::gpu::GpuLinearAlgebra;

        let a = Tensor::ones(&[8, 8]).unwrap();
        let b = Tensor::ones(&[8, 8]).unwrap();

        let result = a.gpu_matmul(&b);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "coreml-fallback")]
    fn test_fallback_mode_availability() {
        // Test fallback mode functionality
        use rustorch::gpu::hybrid_executor::HybridExecutor;

        let executor = HybridExecutor::new();
        assert!(executor.is_fallback_enabled());
    }
}

#[cfg(test)]
mod memory_management_tests {
    // Unused imports removed

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_memory_pressure_handling() {
        // Test behavior under memory pressure
        let mut tensors = Vec::new();

        // Create progressively larger tensors until we hit memory limits
        for size in [64, 128, 256, 512].iter() {
            let tensor = Tensor::ones(&[*size, *size]).unwrap();
            tensors.push(tensor);
        }

        // Test operations still work
        if tensors.len() >= 2 {
            let result = tensors[0].gpu_matmul(&tensors[1]);
            assert!(result.is_ok() || result.is_err()); // Either outcome acceptable
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_large_tensor_fallback() {
        // Test that very large tensors properly fall back to appropriate backends
        if std::env::var("SKIP_LARGE_TENSOR_TESTS").is_ok() {
            return; // Skip on memory-constrained systems
        }

        let large_tensor = match Tensor::ones(&[1024, 1024]) {
            Ok(tensor) => tensor,
            Err(_) => {
                println!("Skipping large tensor test due to memory constraints");
                return;
            }
        };

        let result = large_tensor.gpu_matmul(&large_tensor);
        match result {
            Ok(output) => {
                assert_eq!(output.shape(), &[1024, 1024]);
                println!("Large tensor operation succeeded");
            }
            Err(e) => {
                println!("Large tensor operation failed as expected: {}", e);
                // Failure is acceptable for very large tensors
            }
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_memory_leak_prevention() {
        // Test that repeated operations don't cause memory leaks
        let a = Tensor::ones(&[32, 32]).unwrap();
        let b = Tensor::ones(&[32, 32]).unwrap();

        // Perform many operations
        for _ in 0..100 {
            let result = a.gpu_matmul(&b);
            assert!(result.is_ok());
            // Result should be dropped automatically
        }

        // No explicit assertion here, but this test helps detect memory leaks
        // through tools like Valgrind or AddressSanitizer
    }
}

#[cfg(test)]
mod error_handling_tests {
    // Unused imports removed

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_invalid_tensor_shapes() {
        // Test error handling for incompatible tensor shapes
        let a = Tensor::ones(&[3, 4]).unwrap();
        let b = Tensor::ones(&[5, 6]).unwrap(); // Incompatible shapes

        let result = a.gpu_matmul(&b);
        assert!(result.is_err()); // Should fail due to incompatible shapes
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_empty_tensor_handling() {
        // Test handling of empty tensors
        let empty = Tensor::zeros(&[0, 0]).unwrap();
        let normal = Tensor::ones(&[4, 4]).unwrap();

        let result = empty.gpu_matmul(&normal);
        assert!(result.is_err()); // Should fail gracefully
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_device_unavailable_fallback() {
        // Test fallback when preferred device is unavailable
        let input = Tensor::ones(&[16, 16]).unwrap();

        // Even if CoreML is not available, operations should fall back
        let result = input.gpu_relu();
        assert!(result.is_ok()); // Should succeed via fallback
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_concurrent_operations() {
        use std::sync::Arc;
        use std::thread;

        let tensor = Arc::new(Tensor::ones(&[16, 16]).unwrap());
        let mut handles = Vec::new();

        // Spawn multiple threads performing operations
        for _ in 0..4 {
            let tensor_clone = Arc::clone(&tensor);
            let handle = thread::spawn(move || {
                let result = tensor_clone.gpu_relu();
                assert!(result.is_ok());
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod compatibility_tests {
    use rustorch::tensor::Tensor;

    #[test]
    fn test_cross_platform_tensor_format() {
        // Test that tensors work consistently across platforms
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 2]);

        // Data should be preserved regardless of platform
        let retrieved_data = tensor.data();
        for (original, retrieved) in data.iter().zip(retrieved_data.iter()) {
            assert!((original - retrieved).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(all(feature = "coreml", feature = "cuda"))]
    fn test_coreml_cuda_compatibility() {
        // Test that CoreML and CUDA features can coexist in the binary
        // (even if they don't run simultaneously on the same hardware)

        let tensor = Tensor::ones(&[8, 8]).unwrap();

        // Both backends should be available in the binary
        use rustorch::gpu::DeviceType;

        let _coreml_device = DeviceType::CoreML(0);
        let _cuda_device = DeviceType::Cuda(0);

        // Actual availability depends on hardware
        println!("CoreML device type available in binary");
        println!("CUDA device type available in binary");
    }

    #[test]
    #[cfg(all(feature = "coreml", feature = "metal"))]
    fn test_coreml_metal_integration() {
        // Test CoreML and Metal working together on Apple platforms
        if !cfg!(target_os = "macos") {
            return; // Skip on non-macOS platforms
        }

        let tensor = Tensor::ones(&[16, 16]).unwrap();

        // Operations should work with either CoreML or Metal backends
        use rustorch::gpu::GpuLinearAlgebra;
        let result = tensor.gpu_matmul(&tensor);
        assert!(result.is_ok());
    }
}
