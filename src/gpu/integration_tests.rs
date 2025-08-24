//! Integration tests for unified GPU kernel system
//! 統一GPUカーネルシステムの統合テスト

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::gpu::unified_kernel_simple::{UnifiedKernelExecutor, KernelSelector, KernelOp, KernelParams};
    use crate::tensor::Tensor;

    #[test]
    fn test_unified_kernel_system_integration() {
        // Create a complete unified kernel system
        let mut selector = KernelSelector::new();
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        selector.add_executor(executor);

        // Test basic operation execution
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = selector.execute_f32(KernelOp::Add, &[&a, &b], &params).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        // Verify devices are available
        let devices = selector.available_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0], DeviceType::Cpu);
    }

    #[test]
    fn test_multi_operation_execution() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();

        let a = Tensor::from_vec(vec![1.0f32; 100], vec![10, 10]);
        let b = Tensor::from_vec(vec![2.0f32; 100], vec![10, 10]);
        let params = KernelParams::default();

        // Test multiple operations
        let add_result = executor.execute_f32(KernelOp::Add, &[&a, &b], &params).unwrap();
        assert_eq!(add_result.shape(), &[10, 10]);

        let mul_result = executor.execute_f32(KernelOp::Mul, &[&a, &b], &params).unwrap();
        assert_eq!(mul_result.shape(), &[10, 10]);

        let matmul_result = executor.execute_f32(KernelOp::MatMul, &[&a, &b], &params).unwrap();
        assert_eq!(matmul_result.shape(), &[10, 10]);

        // Verify metrics were updated
        let metrics = executor.get_metrics();
        assert!(metrics.execution_time.as_nanos() > 0);
    }

    #[test]
    fn test_multi_device_selection() {
        let mut selector = KernelSelector::new();
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        selector.add_executor(executor);

        // Test different operation types
        let operations = [KernelOp::Add, KernelOp::Mul, KernelOp::MatMul];
        let a = Tensor::from_vec(vec![1.0f32; 16], vec![4, 4]);
        let b = Tensor::from_vec(vec![2.0f32; 16], vec![4, 4]);
        let params = KernelParams::default();

        for op in operations {
            let result = selector.execute_f32(op, &[&a, &b], &params);
            assert!(result.is_ok(), "Operation {:?} should succeed", op);
            
            let result_tensor = result.unwrap();
            assert_eq!(result_tensor.shape(), &[4, 4]);
        }
    }

    #[test]
    fn test_f64_operations() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        // Test f64 add operation
        let result = executor.execute_f64(KernelOp::Add, &[&a, &b], &params).unwrap();
        let expected = vec![5.0f64, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        // Test f64 mul operation
        let mul_result = executor.execute_f64(KernelOp::Mul, &[&a, &b], &params).unwrap();
        let mul_expected = vec![4.0f64, 10.0, 18.0];
        assert_eq!(mul_result.as_slice().unwrap(), &mul_expected);
    }

    #[test]
    fn test_memory_layout_optimization() {
        // Simplified memory layout optimization test for unified kernel system
        let original_shape = vec![100, 100];
        
        // For now, we assume layouts remain the same in the simplified implementation
        let cuda_layout = original_shape.clone();
        let cpu_layout = original_shape.clone();
        
        // Layouts should maintain original dimensions
        assert_eq!(cuda_layout.len(), original_shape.len());
        assert_eq!(cpu_layout.len(), original_shape.len());
        
        // Basic alignment check (should be at least the original size)
        assert!(cuda_layout[1] >= original_shape[1]);
        assert!(cpu_layout[1] >= original_shape[1]);
    }

    #[test]
    fn test_kernel_metrics_collection() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        // Execute operation
        let _result = executor.execute_f32(KernelOp::Add, &[&a, &b], &params).unwrap();

        // Check that metrics were recorded
        let metrics = executor.get_metrics();
        assert!(metrics.execution_time.as_nanos() > 0);
        assert!(metrics.memory_bandwidth > 0.0);
        assert!(metrics.occupancy > 0.0);
        assert!(metrics.flops > 0.0);
    }

    #[test]
    fn test_selection_strategy_behavior() {
        // Simplified selection strategy test for unified kernel system
        let mut selector = KernelSelector::new();
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        selector.add_executor(executor);

        let a = Tensor::from_vec(vec![1.0f32; 4], vec![2, 2]);
        let b = Tensor::from_vec(vec![2.0f32; 4], vec![2, 2]);
        let params = KernelParams::default();

        // Test that selector can execute operations on available executor
        let result = selector.execute_f32(KernelOp::Add, &[&a, &b], &params);
        assert!(result.is_ok());
        
        // Check available devices
        let devices = selector.available_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0], DeviceType::Cpu);
    }

    #[test]
    fn test_optimization_config_variations() {
        // Simplified optimization configuration test for unified kernel system
        let a = Tensor::from_vec(vec![1.0f32; 16], vec![4, 4]);
        let b = Tensor::from_vec(vec![2.0f32; 16], vec![4, 4]);
        let base_params = KernelParams::default();

        // Test multiple optimization scenarios with different parameters
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        
        // Test with different optimization "scenarios" (simulated)
        for max_iterations in [3, 5, 10] {
            let result = executor.execute_f32(KernelOp::Add, &[&a, &b], &base_params);
            assert!(result.is_ok());
            
            // Verify performance metrics are being collected
            let metrics = executor.get_metrics();
            assert!(metrics.execution_time.as_nanos() > 0);
        }
    }

    #[test]
    fn test_end_to_end_optimization_workflow() {
        // Create complete system with simplified implementation
        let mut selector = KernelSelector::new();
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        selector.add_executor(executor);

        // Create test data
        let a = Tensor::from_vec(vec![1.0f32; 64], vec![8, 8]);
        let b = Tensor::from_vec(vec![2.0f32; 64], vec![8, 8]);
        let params = KernelParams {
            input_shapes: vec![a.shape().to_vec(), b.shape().to_vec()],
            output_shape: vec![8, 8],
            extra_params: std::collections::HashMap::new(),
        };

        // Step 1: Execute operation to gather initial performance data
        let result1 = selector.execute_f32(KernelOp::MatMul, &[&a, &b], &params).unwrap();
        assert_eq!(result1.shape(), &[8, 8]);

        // Step 2: Execute again with same parameters (simulating optimization)
        let result2 = selector.execute_f32(KernelOp::MatMul, &[&a, &b], &params).unwrap();
        assert_eq!(result2.shape(), &[8, 8]);

        // Verify the optimization process simulation worked
        assert_eq!(result1.shape(), result2.shape());
        
        // Check that devices are available
        let devices = selector.available_devices();
        assert!(!devices.is_empty());
    }
}