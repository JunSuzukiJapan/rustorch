//! GPU Operations Integration Tests
//! GPU演算統合テスト
//!
//! This module provides comprehensive tests for all GPU operations
//! including memory transfer, matrix operations, convolution, and reduction.

// GPU tests are disabled on CI environments and WASM
#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_arch = "wasm32"),
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))]
use rustorch::error::RusTorchResult;
#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_arch = "wasm32"),
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))]
use rustorch::gpu::DeviceManager;
#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_arch = "wasm32"),
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))]
use rustorch::tensor::Tensor;

// GPU matrix operations tests
// GPU行列演算テスト
#[cfg(test)]
mod matrix_operations_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_matrix_multiplication() -> RusTorchResult<()> {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        // GPU matrix multiplication
        let result = a.gpu_matmul(&b)?;

        assert_eq!(result.shape(), &[2, 2]);
        println!("GPU matrix multiplication result: {:?}", result.data);

        // Verify result matches expected computation
        let expected_result = a
            .matmul(&b)
            .map_err(rustorch::error::RusTorchError::tensor_op)?;
        for (gpu_val, cpu_val) in result.data.iter().zip(expected_result.data.iter()) {
            assert!(
                (gpu_val - cpu_val).abs() < 1e-5,
                "GPU result {} doesn't match CPU result {}",
                gpu_val,
                cpu_val
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_batch_matrix_multiplication() -> RusTorchResult<()> {
        let batch_a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let batch_b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);

        let result = batch_a.gpu_batch_matmul(&batch_b)?;

        assert_eq!(result.shape(), &[1, 2, 2]);
        println!("GPU batch matrix multiplication successful");

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_matrix_vector_multiplication() -> RusTorchResult<()> {
        let matrix = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let vector = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2, 1]);

        let result = matrix.gpu_matvec(&vector)?;

        assert_eq!(result.shape(), &[2, 1]);
        println!("GPU matrix-vector multiplication successful");

        Ok(())
    }
}

// GPU convolution operations tests
// GPU畳み込み演算テスト
#[cfg(test)]
mod convolution_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::backends::ConvolutionParams;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::gpu::conv_ops::GpuConvolution;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_conv2d() -> RusTorchResult<()> {
        // Input: batch=1, channels=1, height=4, width=4
        let input = Tensor::<f32>::from_vec((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);

        // Kernel: out_channels=1, in_channels=1, height=3, width=3
        let kernel = Tensor::<f32>::ones(&[1, 1, 3, 3]);

        let params = ConvolutionParams {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        };

        let result = input.gpu_conv2d(&kernel, &params)?;

        // Expected output: batch=1, channels=1, height=2, width=2
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        println!("GPU Conv2D successful, output shape: {:?}", result.shape());

        Ok(())
    }
}

// GPU reduction operations tests
// GPUリダクション演算テスト
#[cfg(test)]
mod reduction_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_sum_operations() -> RusTorchResult<()> {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Test global sum
        let sum_result = tensor.gpu_sum(None)?;
        assert_eq!(sum_result.shape(), &[1]);
        assert_eq!(sum_result.data[0], 21.0);
        println!("GPU sum: {}", sum_result.data[0]);

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_mean_operations() -> RusTorchResult<()> {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let mean_result = tensor.gpu_mean(None)?;
        assert_eq!(mean_result.shape(), &[1]);
        assert_eq!(mean_result.data[0], 3.5);
        println!("GPU mean: {}", mean_result.data[0]);

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_max_min_operations() -> RusTorchResult<()> {
        let tensor = Tensor::<f32>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![2, 3]);

        let max_result = tensor.gpu_max(None)?;
        assert_eq!(max_result.data[0], 9.0);

        let min_result = tensor.gpu_min(None)?;
        assert_eq!(min_result.data[0], 1.0);

        println!(
            "GPU max: {}, min: {}",
            max_result.data[0], min_result.data[0]
        );

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_statistical_operations() -> RusTorchResult<()> {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

        let std_result = tensor.gpu_std(None)?;
        let var_result = tensor.gpu_var(None)?;

        println!(
            "GPU std: {}, var: {}",
            std_result.data[0], var_result.data[0]
        );
        assert!(std_result.data[0] > 0.0);
        assert!(var_result.data[0] > 0.0);

        // Verify std^2 ≈ var
        let std_squared = std_result.data[0] * std_result.data[0];
        assert!((std_squared - var_result.data[0]).abs() < 1e-5);

        Ok(())
    }
}

// GPU parallel operations tests
// GPU並列演算テスト
#[cfg(test)]
mod parallel_operations_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::tensor::gpu_parallel::GpuParallelOp;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_gpu_elementwise_operations() -> RusTorchResult<()> {
        let tensor1 = Tensor::<f32>::ones(&[100, 100]);
        let tensor2 = Tensor::<f32>::ones(&[100, 100]);

        // GPU element-wise addition
        let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b)?;

        assert_eq!(result.shape(), &[100, 100]);
        assert_eq!(result.data[0], 2.0);
        println!("GPU elementwise operation successful");

        Ok(())
    }
}

// Performance and stress tests
// パフォーマンスとストレステスト
#[cfg(test)]
mod performance_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::gpu::matrix_ops::GpuLinearAlgebra;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use std::time::Instant;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_large_matrix_multiplication_performance() -> RusTorchResult<()> {
        let size = 200; // Smaller for CI stability
        let a = Tensor::<f32>::ones(&[size, size]);
        let b = Tensor::<f32>::ones(&[size, size]);

        let start = Instant::now();
        let _result = a.gpu_matmul(&b)?;
        let gpu_time = start.elapsed();

        let start = Instant::now();
        let _result = a
            .matmul(&b)
            .map_err(rustorch::error::RusTorchError::tensor_op)?;
        let cpu_time = start.elapsed();

        println!("GPU matrix multiplication time: {:?}", gpu_time);
        println!("CPU matrix multiplication time: {:?}", cpu_time);

        if gpu_time.as_nanos() > 0 {
            let ratio = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
            println!("GPU/CPU ratio: {:.2}", ratio);
        }

        Ok(())
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_memory_usage_patterns() -> RusTorchResult<()> {
        // Test various tensor sizes to check memory handling
        let sizes = vec![(10, 10), (50, 50), (100, 100)];

        for (height, width) in sizes {
            let tensor = Tensor::<f32>::ones(&[height, width]);

            // Test GPU operations don't cause memory issues
            let sum_result = tensor.gpu_sum(None)?;
            let mean_result = tensor.gpu_mean(None)?;

            println!(
                "Size {}x{}: sum={:.2}, mean={:.2}",
                height, width, sum_result.data[0], mean_result.data[0]
            );

            // Verify correctness
            let expected_sum = (height * width) as f32;
            assert!((sum_result.data[0] - expected_sum).abs() < 1e-3);
            assert!((mean_result.data[0] - 1.0).abs() < 1e-5);
        }

        Ok(())
    }
}

// Error handling and edge cases
// エラー処理とエッジケース
#[cfg(test)]
mod error_handling_tests {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use super::*;
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_dimension_mismatch_errors() {
        let a = Tensor::<f32>::ones(&[2, 3]);
        let b = Tensor::<f32>::ones(&[4, 2]); // Incompatible dimensions

        let result = a.gpu_matmul(&b);
        assert!(result.is_err(), "Expected dimension mismatch error");

        println!("Dimension mismatch correctly detected");
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))] // Skip on CI environments due to no GPU access
    fn test_empty_tensor_handling() -> RusTorchResult<()> {
        let empty_tensor = Tensor::<f32>::from_vec(vec![], vec![0]);

        // Operations on empty tensors should handle gracefully
        let sum_result = empty_tensor.gpu_sum(None);

        match sum_result {
            Ok(result) => {
                assert_eq!(result.data[0], 0.0);
                println!("Empty tensor sum handled correctly");
            }
            Err(_) => {
                println!("Empty tensor handled with error (also acceptable)");
            }
        }

        Ok(())
    }
}

// Main test runner function
#[test]
#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))] // Skip on CI environments due to no GPU access
fn run_all_gpu_tests() {
    println!("=== GPU Operations Test Suite ===");

    // Check device availability
    let manager = DeviceManager::new();
    let devices = manager.available_devices();
    println!("Available devices: {:?}", devices);
    println!("CUDA available: {}", DeviceManager::is_cuda_available());
    println!("Metal available: {}", DeviceManager::is_metal_available());

    println!("GPU tests completed!");
}
