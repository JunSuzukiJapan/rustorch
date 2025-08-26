//! Integration tests for the unified backend system
//! 統一バックエンドシステムの統合テスト

use rustorch::backends::compute_backend::ComputeBackendGeneric;
use rustorch::backends::{
    initialize_backends, DeviceManager, Operation, SelectionStrategy, UnifiedComputeBackend,
    UnifiedCpuBackend, UnifiedDeviceType as DeviceType,
};
use rustorch::error::RusTorchResult;
use rustorch::tensor::Tensor;

#[test]
fn test_unified_cpu_backend_basic_operations() -> RusTorchResult<()> {
    let backend = UnifiedCpuBackend::new()?;

    // Test device info
    assert_eq!(backend.device_type(), DeviceType::Cpu);
    assert!(backend.is_available());

    // Test addition
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
    let add_op = Operation::Add {
        a: a.clone(),
        b: b.clone(),
    };

    let result = backend.execute_operation(&add_op)?;
    let expected = vec![5.0f32, 7.0, 9.0];
    assert_eq!(result.as_slice().unwrap(), &expected);

    // Test multiplication
    let mul_op = Operation::Multiply {
        a: a.clone(),
        b: b.clone(),
    };

    let result = backend.execute_operation(&mul_op)?;
    let expected = vec![4.0f32, 10.0, 18.0];
    assert_eq!(result.as_slice().unwrap(), &expected);

    // Verify metrics are recorded
    let metrics = backend.get_metrics();
    assert!(metrics.execution_time_ns > 0);

    Ok(())
}

#[test]
fn test_device_manager_registration() -> RusTorchResult<()> {
    let mut manager = DeviceManager::new(SelectionStrategy::Performance);

    // Register CPU backend
    let cpu_backend = UnifiedCpuBackend::new()?;
    manager.register_backend(Box::new(cpu_backend))?;

    // Verify registration
    let available_devices = manager.available_devices();
    assert!(available_devices.contains(&DeviceType::Cpu));

    Ok(())
}

#[test]
fn test_device_manager_operation_execution() -> RusTorchResult<()> {
    let mut manager = DeviceManager::new(SelectionStrategy::Balanced);

    // Register CPU backend
    let cpu_backend = UnifiedCpuBackend::new()?;
    manager.register_backend(Box::new(cpu_backend))?;

    // Execute operation through manager
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
    let operation = Operation::Add {
        a: a.clone(),
        b: b.clone(),
    };

    let result = manager.execute_operation(&operation)?;
    let expected = vec![5.0f32, 7.0, 9.0];
    assert_eq!(result.as_slice().unwrap(), &expected);

    // Verify performance stats are tracked
    let stats = manager.get_device_stats(DeviceType::Cpu);
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert!(stats.execution_time_ns > 0);

    Ok(())
}

#[test]
fn test_backend_selection_strategies() -> RusTorchResult<()> {
    // Test performance strategy
    let performance_manager = DeviceManager::new(SelectionStrategy::Performance);
    assert_eq!(performance_manager.strategy, SelectionStrategy::Performance);

    // Test memory strategy
    let memory_manager = DeviceManager::new(SelectionStrategy::Memory);
    assert_eq!(memory_manager.strategy, SelectionStrategy::Memory);

    // Test manual strategy
    let manual_priorities = vec![DeviceType::Cuda(0), DeviceType::Cpu];
    let manual_manager = DeviceManager::new(SelectionStrategy::Manual(manual_priorities.clone()));

    if let SelectionStrategy::Manual(ref priorities) = manual_manager.strategy {
        assert_eq!(priorities, &manual_priorities);
    } else {
        panic!("Strategy should be Manual");
    }

    Ok(())
}

#[test]
fn test_matrix_multiplication_through_unified_backend() -> RusTorchResult<()> {
    let backend = UnifiedCpuBackend::new()?;

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    let matmul_op = Operation::MatMul {
        a: a.clone(),
        b: b.clone(),
    };

    let result = backend.execute_operation(&matmul_op)?;
    assert_eq!(result.shape(), &[2, 2]);

    // Check that result has reasonable values
    let result_data = result.as_slice().unwrap();
    assert_eq!(result_data.len(), 4);

    // Verify metrics
    let metrics = backend.get_metrics();
    assert!(metrics.execution_time_ns > 0);
    assert!(metrics.memory_usage_bytes > 0);

    Ok(())
}

#[test]
fn test_reduction_operations() -> RusTorchResult<()> {
    use rustorch::backends::compute_backend::ReduceOp;

    let backend = UnifiedCpuBackend::new()?;
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);

    // Test sum reduction
    let sum_op = Operation::Reduce {
        input: input.clone(),
        operation: ReduceOp::Sum,
        axes: None,
    };
    let sum_result = backend.execute_operation(&sum_op)?;
    assert_eq!(sum_result.as_slice().unwrap(), &[10.0f32]);

    // Test mean reduction
    let mean_op = Operation::Reduce {
        input: input.clone(),
        operation: ReduceOp::Mean,
        axes: None,
    };
    let mean_result = backend.execute_operation(&mean_op)?;
    assert_eq!(mean_result.as_slice().unwrap(), &[2.5f32]);

    // Test max reduction
    let max_op = Operation::Reduce {
        input: input.clone(),
        operation: ReduceOp::Max,
        axes: None,
    };
    let max_result = backend.execute_operation(&max_op)?;
    assert_eq!(max_result.as_slice().unwrap(), &[4.0f32]);

    // Test min reduction
    let min_op = Operation::Reduce {
        input: input.clone(),
        operation: ReduceOp::Min,
        axes: None,
    };
    let min_result = backend.execute_operation(&min_op)?;
    assert_eq!(min_result.as_slice().unwrap(), &[1.0f32]);

    Ok(())
}

#[test]
fn test_memory_transfer_operations() -> RusTorchResult<()> {
    use rustorch::backends::TransferDirection;

    let backend = UnifiedCpuBackend::new()?;

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Test host to device transfer
    let transferred_data = backend.memory_transfer(&data, TransferDirection::HostToDevice)?;
    assert_eq!(transferred_data, data);

    // Test device to host transfer
    let transferred_back =
        backend.memory_transfer(&transferred_data, TransferDirection::DeviceToHost)?;
    assert_eq!(transferred_back, data);

    Ok(())
}

#[test]
fn test_backend_info_and_configuration() -> RusTorchResult<()> {
    let mut backend = UnifiedCpuBackend::new()?;

    // Get backend info
    let info = backend.get_info();
    assert!(info.contains_key("device_name"));
    assert!(info.contains_key("simd_features"));
    assert!(info.contains_key("cores"));

    // Test configuration
    let config_value = Box::new(42u32) as Box<dyn std::any::Any + Send + Sync>;
    backend.set_config("test_param", config_value)?;

    // Test synchronization
    backend.synchronize()?;

    // Test memory query
    let available_memory = backend.available_memory()?;
    assert!(available_memory > 0);

    Ok(())
}

#[test]
fn test_performance_metrics_tracking() -> RusTorchResult<()> {
    let backend = UnifiedCpuBackend::new()?;

    // Execute multiple operations to generate metrics
    let a = Tensor::from_vec(vec![1.0f32; 100], vec![100]);
    let b = Tensor::from_vec(vec![2.0f32; 100], vec![100]);

    for _ in 0..5 {
        let operation = Operation::Add {
            a: a.clone(),
            b: b.clone(),
        };
        let _result = backend.execute_operation(&operation)?;
    }

    let metrics = backend.get_metrics();

    // Verify metrics are meaningful
    assert!(metrics.execution_time_ns > 0);
    assert!(metrics.memory_bandwidth_gbps >= 0.0);
    assert_eq!(metrics.device_utilization, 100.0); // CPU should be 100%
    assert!(metrics.memory_usage_bytes > 0);

    Ok(())
}

#[test]
fn test_error_handling_in_unified_backend() {
    let backend = UnifiedCpuBackend::new().unwrap();

    // Test shape mismatch error
    let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);

    let operation = Operation::Add {
        a: a.clone(),
        b: b.clone(),
    };

    let result = backend.execute_operation(&operation);
    assert!(result.is_err());

    // Verify it's the correct error type
    match result {
        Err(rustorch::error::RusTorchError::ShapeMismatch { .. }) => {
            // Expected error
        }
        _ => panic!("Expected ShapeMismatch error"),
    }
}

#[test]
fn test_f64_operations() -> RusTorchResult<()> {
    let backend = UnifiedCpuBackend::new()?;

    // Test f64 addition
    let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], vec![3]);

    let operation = Operation::Add {
        a: a.clone(),
        b: b.clone(),
    };

    let result = backend.execute_operation(&operation)?;
    let expected = vec![5.0f64, 7.0, 9.0];
    assert_eq!(result.as_slice().unwrap(), &expected);

    // Test f64 matrix multiplication
    let a_matrix = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b_matrix = Tensor::from_vec(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]);

    let matmul_op = Operation::MatMul {
        a: a_matrix.clone(),
        b: b_matrix.clone(),
    };

    let result = backend.execute_operation(&matmul_op)?;
    assert_eq!(result.shape(), &[2, 2]);

    Ok(())
}

// Integration test with global device manager
#[test]
fn test_global_device_manager_integration() -> RusTorchResult<()> {
    // This test should be run carefully as it modifies global state
    initialize_backends()?;

    let manager = rustorch::backends::global_device_manager();
    let available_devices = {
        let manager_guard = manager.read().unwrap();
        manager_guard.available_devices()
    };

    // At minimum, CPU should be available
    assert!(available_devices.contains(&DeviceType::Cpu));

    Ok(())
}
