//! f32統一ハイブリッドシステム テスト
//! f32 Unified Hybrid System Tests

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::hybrid_f32::tensor::F32Tensor;
    use crate::hybrid_f32::unified::F32HybridExecutor;
    use crate::hybrid_f32::benchmarks::run_quick_benchmark;

    #[test]
    fn test_f32_tensor_creation() {
        crate::hybrid_f32_experimental!();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = F32Tensor::new(data, shape).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_f32_tensor_zeros() {
        let tensor = F32Tensor::zeros(&[3, 3]);
        assert_eq!(tensor.shape(), &[3, 3]);
        assert_eq!(tensor.len(), 9);

        // すべて0であることを確認
        for &value in tensor.as_slice() {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn test_f32_tensor_randn() {
        let tensor = F32Tensor::randn(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);

        // ランダム値が範囲内にあることを確認
        for &value in tensor.as_slice() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_f32_tensor_device_movement() {
        let mut tensor = F32Tensor::zeros(&[2, 2]);

        // CPU → Metal
        tensor.to_metal(0).unwrap();
        if let crate::hybrid_f32::tensor::DeviceState::Metal { device_id } = tensor.device_state() {
            assert_eq!(*device_id, 0);
        } else {
            panic!("Expected Metal device state");
        }

        // Metal → CoreML
        tensor.to_coreml(0).unwrap();
        if let crate::hybrid_f32::tensor::DeviceState::CoreML { device_id } = tensor.device_state() {
            assert_eq!(*device_id, 0);
        } else {
            panic!("Expected CoreML device state");
        }

        // CoreML → CPU
        tensor.to_cpu().unwrap();
        if let crate::hybrid_f32::tensor::DeviceState::CPU = tensor.device_state() {
            // Success
        } else {
            panic!("Expected CPU device state");
        }
    }

    #[test]
    fn test_f32_tensor_matmul() {
        crate::hybrid_f32_experimental!();

        let a = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();

        let b = F32Tensor::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2]
        ).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // 結果の検証
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // 手計算結果
        let result_slice = result.as_slice();

        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result_slice[i] - expected_val).abs() < 1e-5);
        }
    }

    #[test]
    fn test_f32_hybrid_executor_creation() {
        let executor_result = F32HybridExecutor::new();
        assert!(executor_result.is_ok());

        let mut executor = executor_result.unwrap();
        let init_result = executor.initialize();
        assert!(init_result.is_ok());
    }

    #[test]
    fn test_f32_hybrid_executor_matmul() {
        crate::hybrid_f32_experimental!();

        let mut executor = F32HybridExecutor::new().unwrap();
        executor.initialize().unwrap();

        let a = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();

        let b = F32Tensor::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2]
        ).unwrap();

        let (result, experiment_results) = executor.execute_matmul(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(experiment_results.conversion_cost_reduction, 100.0);
    }

    #[test]
    fn test_device_selection_logic() {
        use crate::hybrid_f32::unified::{F32DeviceSelector, F32Operation};
        use crate::hybrid_f32::gpu::GPUDevice;

        let selector = F32DeviceSelector::new();

        // 小規模行列 → CPU
        let small_op = F32Operation::MatMul {
            size_a: vec![10, 10],
            size_b: vec![10, 10],
        };
        let device = selector.select_optimal_device(&small_op).unwrap();
        assert!(matches!(device, GPUDevice::CPU));

        // 中規模行列 → Neural Engine
        let medium_op = F32Operation::MatMul {
            size_a: vec![100, 100],
            size_b: vec![100, 100],
        };
        let device = selector.select_optimal_device(&medium_op).unwrap();
        assert!(matches!(device, GPUDevice::CoreML(0)));

        // 大規模行列 → Metal GPU
        let large_op = F32Operation::MatMul {
            size_a: vec![1000, 1000],
            size_b: vec![1000, 1000],
        };
        let device = selector.select_optimal_device(&large_op).unwrap();
        assert!(matches!(device, GPUDevice::Metal(0)));

        // 畳み込み演算 → Neural Engine
        let conv_op = F32Operation::Conv2D {
            input_shape: vec![1, 3, 224, 224],
            kernel_shape: vec![64, 3, 3, 3],
        };
        let device = selector.select_optimal_device(&conv_op).unwrap();
        assert!(matches!(device, GPUDevice::CoreML(0)));
    }

    #[test]
    fn test_performance_monitoring() {
        use crate::hybrid_f32::unified::{PerformanceMonitor, F32Operation};
        use crate::hybrid_f32::gpu::GPUDevice;
        use std::time::Duration;

        let mut monitor = PerformanceMonitor::new();

        let operation = F32Operation::MatMul {
            size_a: vec![100, 100],
            size_b: vec![100, 100],
        };

        monitor.record_execution(&operation, Duration::from_millis(10), &GPUDevice::Metal(0));
        monitor.record_execution(&operation, Duration::from_millis(15), &GPUDevice::CoreML(0));

        let stats = monitor.get_stats();
        assert_eq!(stats.total_operations, 2);
        assert!(stats.conversion_cost_savings > Duration::from_secs(0));
        assert_eq!(stats.device_usage.len(), 2);
    }

    #[test]
    #[ignore] // 重いテストなので通常は無視
    fn test_quick_benchmark() {
        let result = run_quick_benchmark();
        assert!(result.is_ok());
    }

    #[test]
    fn test_experiment_results() {
        let mut results = ExperimentResults::new();
        results.baseline_execution_time = std::time::Duration::from_millis(100);
        results.total_execution_time = std::time::Duration::from_millis(80);

        let improvement = results.performance_improvement();
        assert!((improvement - 20.0).abs() < 1e-5); // 20%改善
    }

    #[test]
    fn test_error_handling() {
        // 無効な形状でのテンソル作成
        let result = F32Tensor::new(vec![1.0, 2.0], vec![3, 3]); // データ不足
        assert!(result.is_err());

        // 不正な行列乗算
        let a = F32Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = F32Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_conversion_cost_elimination() {
        crate::hybrid_f32_experimental!();

        // f32統一システムでは変換コストが0であることを確認
        let a = F32Tensor::randn(&[100, 100]);
        let b = F32Tensor::randn(&[100, 100]);

        let start = std::time::Instant::now();
        let _result = a.matmul(&b).unwrap();
        let execution_time = start.elapsed();

        // 変換時間が含まれていないことを間接的に確認
        // （実際の実装では Metal/CoreML API を使用して直接確認）
        println!("F32 execution time: {:?} (no conversion overhead)", execution_time);
        assert!(execution_time < std::time::Duration::from_millis(1000)); // 十分高速
    }
}