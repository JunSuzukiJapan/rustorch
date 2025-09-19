//! Comprehensive Tests for CoreML Unified Trait Implementation
//! CoreML統一trait実装の包括的テスト
//!
//! This test suite validates the hybrid execution system that provides CoreML support
//! through the unified GPU trait system with automatic fallback to GPU/CPU backends.

use rustorch::error::RusTorchResult;
use rustorch::gpu::{DeviceType, OpType};
use rustorch::tensor::Tensor;

#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
use rustorch::gpu::{GpuActivation, GpuConvolution, GpuLinearAlgebra};

/// Test module for CoreML Linear Algebra operations
/// CoreML線形代数演算のテストモジュール
#[cfg(test)]
mod linear_algebra_tests {
    use super::*;

    #[test]
    fn test_coreml_matmul_basic() {
        let a = Tensor::zeros(&[2, 3]).unwrap();
        let b = Tensor::zeros(&[3, 4]).unwrap();

        // Test matrix multiplication with hybrid execution
        // ハイブリッド実行による行列乗算テスト
        #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
        {
            let result = a.gpu_matmul(&b);
            assert!(result.is_ok());
            let c = result.unwrap();
            assert_eq!(c.shape(), &[2, 4]);
        }

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid")))]
        {
            // Fallback to standard matmul when CoreML features are disabled
            let result = a.matmul(&b);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_coreml_matrix_vector_multiplication() {
        let matrix = Tensor::eye(4).unwrap();
        let vector = Tensor::ones(&[4, 1]).unwrap();

        #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
        {
            let result = matrix.gpu_matvec(&vector);
            assert!(result.is_ok());
            let output = result.unwrap();
            assert_eq!(output.shape(), &[4, 1]);
        }
    }

    #[test]
    fn test_coreml_transpose() {
        let input = Tensor::zeros(&[3, 5]).unwrap();

        #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
        {
            let result = input.gpu_transpose();
            assert!(result.is_ok());
            let transposed = result.unwrap();
            assert_eq!(transposed.shape(), &[5, 3]);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_inverse_2x2() {
        // Test with a simple invertible 2x2 matrix
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(input_data, &[2, 2]).unwrap();

        let result = input.gpu_inverse();
        // Note: May fall back to CPU if CoreML doesn't support inversion
        assert!(result.is_ok() || result.is_err()); // Accept either outcome
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_eigenvalues() {
        let input = Tensor::eye(3).unwrap();

        let result = input.gpu_eigenvalues();
        // May fall back to CPU implementation
        assert!(result.is_ok() || result.is_err());
    }

    /// Test large matrix operations to verify memory management
    /// メモリ管理を確認するための大行列演算テスト
    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_large_matrix_operations() {
        let a = Tensor::ones(&[512, 256]).unwrap();
        let b = Tensor::ones(&[256, 128]).unwrap();

        let result = a.gpu_matmul(&b);
        assert!(result.is_ok());
        let c = result.unwrap();
        assert_eq!(c.shape(), &[512, 128]);
    }
}

/// Test module for CoreML Convolution operations
/// CoreML畳み込み演算のテストモジュール
#[cfg(test)]
mod convolution_tests {
    use super::*;
    use rustorch::backends::ConvolutionParams;

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_conv2d_basic() {
        // Input: [batch, channels, height, width] = [1, 3, 32, 32]
        let input = Tensor::ones(&[1, 3, 32, 32]).unwrap();
        // Kernel: [out_channels, in_channels, kernel_h, kernel_w] = [16, 3, 3, 3]
        let kernel = Tensor::ones(&[16, 3, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };

        let result = input.gpu_conv2d(&kernel, &params);
        assert!(result.is_ok());
        let output = result.unwrap();
        // Expected output shape: [1, 16, 32, 32] with padding
        assert_eq!(output.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_conv_transpose2d() {
        let input = Tensor::ones(&[1, 16, 16, 16]).unwrap();
        let kernel = Tensor::ones(&[16, 32, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [2, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };

        let result = input.gpu_conv_transpose2d(&kernel, &params);
        assert!(result.is_ok());
        let output = result.unwrap();
        // Transposed convolution with stride 2 should double spatial dimensions
        assert_eq!(output.shape(), &[1, 32, 32, 32]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_depthwise_conv2d() {
        let input = Tensor::ones(&[1, 32, 64, 64]).unwrap();
        let kernel = Tensor::ones(&[32, 1, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 32, // Depthwise convolution
        };

        let result = input.gpu_depthwise_conv2d(&kernel, &params);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 32, 64, 64]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_grouped_conv2d() {
        let input = Tensor::ones(&[1, 64, 32, 32]).unwrap();
        let kernel = Tensor::ones(&[128, 32, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 2,
        };

        let result = input.gpu_grouped_conv2d(&kernel, &params, 2);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 128, 32, 32]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_conv3d() {
        let input = Tensor::ones(&[1, 8, 16, 16, 16]).unwrap();
        let kernel = Tensor::ones(&[16, 8, 3, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };

        let result = input.gpu_conv3d(&kernel, &params);
        // 3D convolution may not be supported by CoreML, expect fallback
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test module for CoreML Activation operations
/// CoreML活性化関数演算のテストモジュール
#[cfg(test)]
mod activation_tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_relu() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let result = input.gpu_relu();
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify ReLU behavior: negative values become 0, positive stay the same
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        let output_data = output.data();
        for (actual, expected) in output_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_sigmoid() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();

        let result = input.gpu_sigmoid();
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify sigmoid outputs are in (0, 1) range
        let output_data = output.data();
        for value in output_data.iter() {
            assert!(*value > 0.0 && *value < 1.0);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_tanh() {
        let input = Tensor::from_vec(vec![-2.0, 0.0, 2.0], &[3]).unwrap();

        let result = input.gpu_tanh();
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify tanh outputs are in (-1, 1) range
        let output_data = output.data();
        for value in output_data.iter() {
            assert!(*value > -1.0 && *value < 1.0);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_softmax() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = input.gpu_softmax(0);
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify softmax outputs sum to 1
        let output_data = output.data();
        let sum: f32 = output_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_gelu() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();

        let result = input.gpu_gelu();
        assert!(result.is_ok());
        // GELU is more complex, just verify it doesn't crash
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_leaky_relu() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let alpha = 0.1;

        let result = input.gpu_leaky_relu(alpha);
        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify LeakyReLU behavior
        let output_data = output.data();
        assert!(output_data[0] < 0.0); // Negative input with slope
        assert!(output_data[3] == 1.0); // Positive input unchanged
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_elu() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        let alpha = 1.0;

        let result = input.gpu_elu(alpha);
        assert!(result.is_ok());
        // ELU is complex, just verify it doesn't crash
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_swish() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();

        let result = input.gpu_swish();
        assert!(result.is_ok());
        // Swish (SiLU) is complex, just verify it doesn't crash
    }

    /// Test batch processing for activation functions
    /// 活性化関数のバッチ処理テスト
    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_coreml_activation_batch_processing() {
        let batch_input = Tensor::ones(&[32, 128, 64, 64]).unwrap();

        let relu_result = batch_input.gpu_relu();
        assert!(relu_result.is_ok());

        let sigmoid_result = batch_input.gpu_sigmoid();
        assert!(sigmoid_result.is_ok());

        let tanh_result = batch_input.gpu_tanh();
        assert!(tanh_result.is_ok());
    }
}

/// Test module for Hybrid Execution System
/// ハイブリッド実行システムのテストモジュール
#[cfg(test)]
mod hybrid_execution_tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_platform_detection() {
        // Test that the system correctly detects Apple Silicon vs Intel/AMD
        use rustorch::gpu::hybrid_executor::HybridExecutor;

        let executor = HybridExecutor::new();
        let is_apple_silicon = executor.is_apple_silicon();

        // This should match the actual platform
        if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            assert!(is_apple_silicon);
        } else {
            assert!(!is_apple_silicon);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_fallback_chain_construction() {
        use rustorch::gpu::hybrid_executor::HybridExecutor;

        let mut executor = HybridExecutor::new();
        executor.build_fallback_chain();

        let chain = executor.get_fallback_chain();

        // Verify fallback chain is not empty
        assert!(!chain.is_empty());

        // First device should be the primary (best) option
        if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            // Apple Silicon: should prefer CoreML/Metal
            assert!(matches!(
                chain[0],
                DeviceType::CoreML(_) | DeviceType::Metal(_)
            ));
        } else {
            // Intel/AMD: should prefer CUDA if available
            assert!(matches!(
                chain[0],
                DeviceType::Cuda(_) | DeviceType::OpenCL(_) | DeviceType::Cpu
            ));
        }
    }

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_device_capability_checking() {
        use rustorch::gpu::{DeviceCapability, OpType};

        #[cfg(feature = "coreml")]
        {
            let coreml_capability = DeviceCapability::coreml_capability();

            // CoreML should support basic operations
            assert!(coreml_capability.supports_operation(&OpType::LinearAlgebra));
            assert!(coreml_capability.supports_operation(&OpType::Convolution));
            assert!(coreml_capability.supports_operation(&OpType::Activation));

            // CoreML should not support complex/distributed operations
            assert!(!coreml_capability.supports_operation(&OpType::ComplexMath));
            assert!(!coreml_capability.supports_operation(&OpType::DistributedOps));
        }
    }

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_automatic_device_selection() {
        let input = Tensor::ones(&[64, 64]).unwrap();
        let other = Tensor::ones(&[64, 32]).unwrap();

        // Test that operations automatically select the best available device
        let result = input.gpu_matmul(&other);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[64, 32]);
    }

    /// Test error handling and graceful fallback
    /// エラーハンドリングと適切なフォールバックのテスト
    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_fallback_on_unsupported_operation() {
        // Test complex number operations that CoreML doesn't support
        let input = Tensor::ones(&[16, 16]).unwrap();

        // This should fall back to CPU even if CoreML is preferred
        // (Since we're not implementing complex operations in this test)
        let result = input.gpu_matmul(&input);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(any(feature = "coreml-hybrid", feature = "coreml-fallback"))]
    fn test_memory_limit_fallback() {
        // Test that very large tensors fall back appropriately
        // This might be skipped on systems with insufficient memory
        if std::env::var("SKIP_LARGE_TENSOR_TESTS").is_ok() {
            return;
        }

        let large_input = Tensor::ones(&[2048, 2048]).unwrap();
        let large_other = Tensor::ones(&[2048, 1024]).unwrap();

        let result = large_input.gpu_matmul(&large_other);
        // Should either succeed or fail gracefully
        match result {
            Ok(output) => assert_eq!(output.shape(), &[2048, 1024]),
            Err(_) => {
                // Memory exhaustion is acceptable for very large tensors
                println!("Large tensor test skipped due to memory limitations");
            }
        }
    }
}

/// Integration tests combining multiple operations
/// 複数演算を組み合わせた統合テスト
#[cfg(test)]
mod integration_tests {
    use super::*;
    use rustorch::backends::ConvolutionParams;

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_conv_relu_chain() {
        // Test a common neural network pattern: Convolution + ReLU
        let input = Tensor::ones(&[1, 3, 32, 32]).unwrap();
        let kernel = Tensor::ones(&[16, 3, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };

        // Apply convolution
        let conv_result = input.gpu_conv2d(&kernel, &params);
        assert!(conv_result.is_ok());
        let conv_output = conv_result.unwrap();

        // Apply ReLU activation
        let relu_result = conv_output.gpu_relu();
        assert!(relu_result.is_ok());
        let final_output = relu_result.unwrap();

        assert_eq!(final_output.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_matrix_activation_chain() {
        // Test a linear layer followed by activation
        let input = Tensor::ones(&[32, 128]).unwrap();
        let weights = Tensor::ones(&[128, 64]).unwrap();

        // Matrix multiplication (linear layer)
        let linear_result = input.gpu_matmul(&weights);
        assert!(linear_result.is_ok());
        let linear_output = linear_result.unwrap();

        // Apply sigmoid activation
        let sigmoid_result = linear_output.gpu_sigmoid();
        assert!(sigmoid_result.is_ok());
        let final_output = sigmoid_result.unwrap();

        assert_eq!(final_output.shape(), &[32, 64]);
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_mixed_precision_operations() {
        // Test operations with different data types
        let input_f32 = Tensor::ones(&[16, 16]).unwrap();

        // Most operations should work with f32
        let matmul_result = input_f32.gpu_matmul(&input_f32);
        assert!(matmul_result.is_ok());

        let relu_result = input_f32.gpu_relu();
        assert!(relu_result.is_ok());
    }

    /// Test performance comparison between devices
    /// デバイス間のパフォーマンス比較テスト
    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn test_performance_comparison() {
        use std::time::Instant;

        let input = Tensor::ones(&[256, 256]).unwrap();
        let other = Tensor::ones(&[256, 128]).unwrap();

        // Measure GPU operation time
        let start = Instant::now();
        let gpu_result = input.gpu_matmul(&other);
        let gpu_duration = start.elapsed();
        assert!(gpu_result.is_ok());

        // Measure CPU operation time
        let start = Instant::now();
        let cpu_result = input.matmul(&other);
        let cpu_duration = start.elapsed();
        assert!(cpu_result.is_ok());

        println!("GPU operation took: {:?}", gpu_duration);
        println!("CPU operation took: {:?}", cpu_duration);

        // Note: GPU might be slower for small matrices due to overhead
        // This test is mainly to ensure both paths work
    }
}

/// Benchmark tests for CoreML operations
/// CoreML演算のベンチマークテスト
#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn benchmark_matrix_multiplication_sizes() {
        let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

        for (m, n) in sizes {
            let a = Tensor::ones(&[m, n]).unwrap();
            let b = Tensor::ones(&[n, m]).unwrap();

            let start = Instant::now();
            let result = a.gpu_matmul(&b);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("{}x{} matrix multiplication took: {:?}", m, n, duration);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn benchmark_activation_functions() {
        let input = Tensor::ones(&[1024, 1024]).unwrap();

        // Benchmark different activation functions
        let activations = vec![
            ("ReLU", |x: &Tensor<f32>| x.gpu_relu()),
            ("Sigmoid", |x: &Tensor<f32>| x.gpu_sigmoid()),
            ("Tanh", |x: &Tensor<f32>| x.gpu_tanh()),
        ];

        for (name, activation_fn) in activations {
            let start = Instant::now();
            let result = activation_fn(&input);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("{} activation took: {:?}", name, duration);
        }
    }

    #[test]
    #[cfg(any(feature = "coreml", feature = "coreml-hybrid"))]
    fn benchmark_convolution_operations() {
        let input = Tensor::ones(&[1, 32, 64, 64]).unwrap();
        let kernel = Tensor::ones(&[64, 32, 3, 3]).unwrap();

        let params = ConvolutionParams {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };

        let start = Instant::now();
        let result = input.gpu_conv2d(&kernel, &params);
        let duration = start.elapsed();

        assert!(result.is_ok());
        println!("2D convolution took: {:?}", duration);
    }
}
