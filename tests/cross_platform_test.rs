//! Cross-platform optimization tests
//! クロスプラットフォーム最適化テスト

use rustorch::optimization::{
    HardwareOptimizer, OptimizationLevel, PlatformOptimizer, SimdBackend, SimdOptimizer,
    VectorizedOperation,
};
use rustorch::tensor::Tensor;

#[test]
fn test_simd_detection() {
    let optimizer = SimdOptimizer::new();
    let backend = optimizer.backend();

    println!("Detected SIMD backend: {:?}", backend);
    println!("Vector width: {}", optimizer.vector_width());

    // Backend should be detected
    assert_ne!(backend, SimdBackend::Auto);

    // Vector width should be at least 1
    assert!(optimizer.vector_width() >= 1);
}

#[test]
fn test_simd_operations_correctness() {
    let optimizer = SimdOptimizer::new();

    // Test addition
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);

    let result = optimizer
        .apply_vectorized(VectorizedOperation::Add, &a, Some(&b))
        .unwrap();

    assert_eq!(result.shape(), &[4]);
    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);
    }

    // Test multiplication
    let result = optimizer
        .apply_vectorized(VectorizedOperation::Multiply, &a, Some(&b))
        .unwrap();

    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[5.0, 12.0, 21.0, 32.0]);
    }
}

#[test]
fn test_simd_matmul() {
    let optimizer = SimdOptimizer::new();

    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let result = optimizer
        .apply_vectorized(VectorizedOperation::MatMul, &a, Some(&b))
        .unwrap();

    assert_eq!(result.shape(), &[2, 2]);

    // Expected result:
    // [1, 2, 3] @ [[7, 8],    = [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    //              [9, 10],      = [58, 64]
    //              [11, 12]]
    // [4, 5, 6] @ [[7, 8],    = [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    //              [9, 10],      = [139, 154]
    //              [11, 12]]
    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }
}

#[test]
fn test_different_simd_backends() {
    // Test with different backends
    for backend in [SimdBackend::Scalar, SimdBackend::SSE2, SimdBackend::AVX2].iter() {
        let optimizer = SimdOptimizer::with_backend(*backend);

        // Only test if the backend is actually available
        if optimizer.backend() == *backend || *backend == SimdBackend::Scalar {
            let a = Tensor::<f32>::ones(&[100]);
            let b = Tensor::<f32>::ones(&[100]);

            let result = optimizer.apply_vectorized(VectorizedOperation::Add, &a, Some(&b));

            assert!(result.is_ok());
            let result = result.unwrap();

            if let Some(data) = result.as_slice() {
                for val in data {
                    assert!((val - 2.0).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
fn test_platform_detection() {
    let optimizer = PlatformOptimizer::new();
    let features = optimizer.features();

    println!("Platform features:");
    println!("  OS: {}", features.os);
    println!("  Architecture: {}", features.arch);
    println!("  CPU cores: {}", features.cpu_cores);
    println!("  Cache line size: {} bytes", features.cache_line_size);
    println!("  Page size: {} bytes", features.page_size);

    // Basic sanity checks
    assert!(features.cpu_cores > 0);
    assert!(features.cache_line_size > 0);
    assert!(features.page_size > 0);
    assert!(features.total_memory > 0);
}

#[test]
fn test_memory_alignment() {
    let optimizer = PlatformOptimizer::new();
    let cache_line = optimizer.features().cache_line_size;

    // Test alignment calculation
    let unaligned_sizes = [17, 33, 100, 1000];
    for size in unaligned_sizes.iter() {
        let aligned = optimizer.align_memory(*size);

        // Aligned size should be >= original
        assert!(aligned >= *size);

        // Should be aligned to cache line
        assert_eq!(aligned % cache_line, 0);
    }
}

#[test]
fn test_optimization_levels() {
    let mut optimizer = PlatformOptimizer::new();
    let cpu_cores = optimizer.features().cpu_cores;

    // Test different optimization levels
    optimizer.set_optimization_level(OptimizationLevel::None);
    assert_eq!(optimizer.thread_pool_size(), 1);

    optimizer.set_optimization_level(OptimizationLevel::Basic);
    assert!(optimizer.thread_pool_size() <= cpu_cores);

    optimizer.set_optimization_level(OptimizationLevel::Standard);
    assert!(optimizer.thread_pool_size() <= cpu_cores);

    optimizer.set_optimization_level(OptimizationLevel::Aggressive);
    assert_eq!(optimizer.thread_pool_size(), cpu_cores);
}

#[test]
fn test_hardware_detection() {
    let optimizer = HardwareOptimizer::new();
    let capabilities = optimizer.capabilities();

    println!("Hardware capabilities:");
    println!("  CPU vendor: {}", capabilities.cpu_info.vendor);
    println!("  CPU model: {}", capabilities.cpu_info.model);
    println!("  Physical cores: {}", capabilities.cpu_info.physical_cores);
    println!("  Logical cores: {}", capabilities.cpu_info.logical_cores);
    println!("  CPU extensions: {:?}", capabilities.cpu_info.extensions);

    // Basic sanity checks
    assert!(capabilities.cpu_info.physical_cores > 0);
    assert!(capabilities.cpu_info.logical_cores >= capabilities.cpu_info.physical_cores);
    assert!(capabilities.memory_hierarchy.l1_cache > 0);
    assert!(capabilities.memory_hierarchy.l2_cache > 0);
}

#[test]
fn test_optimal_tile_sizes() {
    let optimizer = HardwareOptimizer::new();

    // Test tile size calculation for different operations
    let (m1, n1) = optimizer.optimal_tile_size("matmul");
    assert!(m1 > 0 && n1 > 0);

    let (m2, n2) = optimizer.optimal_tile_size("conv2d");
    assert!(m2 > 0 && n2 > 0);

    let (m3, n3) = optimizer.optimal_tile_size("unknown");
    assert!(m3 > 0 && n3 > 0);
}

#[test]
fn test_data_layout_selection() {
    let optimizer = HardwareOptimizer::new();

    // Test layout selection for different tensor shapes
    let layout1 = optimizer.optimal_data_layout(&[1024, 1024]);
    let layout2 = optimizer.optimal_data_layout(&[100, 7]);
    let layout3 = optimizer.optimal_data_layout(&[32, 32, 3, 3]);

    // Layouts should be determined
    println!("Layout for [1024, 1024]: {:?}", layout1);
    println!("Layout for [100, 7]: {:?}", layout2);
    println!("Layout for [32, 32, 3, 3]: {:?}", layout3);
}

#[test]
fn test_cross_platform_consistency() {
    // Test that optimized operations produce same results as scalar
    let scalar_opt = SimdOptimizer::with_backend(SimdBackend::Scalar);
    let auto_opt = SimdOptimizer::new();

    let a = Tensor::<f32>::randn(&[100]);
    let b = Tensor::<f32>::randn(&[100]);

    // Addition
    let scalar_result = scalar_opt
        .apply_vectorized(VectorizedOperation::Add, &a, Some(&b))
        .unwrap();

    let auto_result = auto_opt
        .apply_vectorized(VectorizedOperation::Add, &a, Some(&b))
        .unwrap();

    // Results should be very close (allowing for floating point differences)
    if let (Some(scalar_data), Some(auto_data)) = (scalar_result.as_slice(), auto_result.as_slice())
    {
        for (s, a) in scalar_data.iter().zip(auto_data.iter()) {
            assert!((s - a).abs() < 1e-5);
        }
    }
}

#[test]
fn test_large_tensor_optimization() {
    let optimizer = SimdOptimizer::new();

    // Test with large tensors
    let size = 10000;
    let a = Tensor::<f32>::ones(&[size]);
    let b = Tensor::<f32>::ones(&[size]);

    let result = optimizer
        .apply_vectorized(VectorizedOperation::Add, &a, Some(&b))
        .unwrap();

    assert_eq!(result.shape(), &[size]);

    // Spot check some values
    if let Some(data) = result.as_slice() {
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[size / 2] - 2.0).abs() < 1e-6);
        assert!((data[size - 1] - 2.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_specific_optimizations() {
    // Test x86-specific features
    let optimizer = SimdOptimizer::new();

    if optimizer.backend() == SimdBackend::AVX2 || optimizer.backend() == SimdBackend::SSE2 {
        println!("x86 SIMD backend detected: {:?}", optimizer.backend());

        // Test vectorized operations
        let a = vec![1.0f32; 16];
        let b = vec![2.0f32; 16];
        let mut result = vec![0.0f32; 16];

        unsafe {
            SimdOptimizer::add_f32_avx2(&a, &b, &mut result);
        }

        for val in result.iter() {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_arm_specific_optimizations() {
    // Test ARM-specific features
    let optimizer = SimdOptimizer::new();

    if optimizer.backend() == SimdBackend::NEON {
        println!("ARM NEON backend detected");

        // ARM-specific tests would go here
        let a = Tensor::<f32>::ones(&[4]);
        let b = Tensor::<f32>::ones(&[4]);

        let result = optimizer
            .apply_vectorized(VectorizedOperation::Add, &a, Some(&b))
            .unwrap();

        if let Some(data) = result.as_slice() {
            for val in data {
                assert!((val - 2.0).abs() < 1e-6);
            }
        }
    }
}
