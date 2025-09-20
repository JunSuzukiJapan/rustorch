//! Comprehensive tests for Python bindings
//! Pythonバインディングの包括的テスト

#[cfg(all(test, feature = "python"))]
mod python_bindings_tests {
    use pyo3::exceptions::*;
    use rustorch::error::RusTorchError;
    use rustorch::python::common::{
        conversions, memory, to_py_err, validation, PyWrapper, ThreadSafePyWrapper,
    };
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_error_conversion() {
        // Test different error types are converted correctly
        let mem_err = RusTorchError::MemoryError("Out of memory".to_string());
        let py_err = to_py_err(mem_err);
        assert!(py_err.is_instance_of::<PyMemoryError>(pyo3::Python::acquire_gil().python()));

        let dim_err = RusTorchError::InvalidDimensions("Invalid shape".to_string());
        let py_err = to_py_err(dim_err);
        assert!(py_err.is_instance_of::<PyValueError>(pyo3::Python::acquire_gil().python()));

        let comp_err = RusTorchError::ComputationError("Computation failed".to_string());
        let py_err = to_py_err(comp_err);
        assert!(py_err.is_instance_of::<PyRuntimeError>(pyo3::Python::acquire_gil().python()));
    }

    #[test]
    fn test_dimension_validation() {
        // Valid dimensions
        assert!(validation::validate_dimensions(&[2, 3, 4]).is_ok());
        assert!(validation::validate_dimensions(&[1]).is_ok());
        assert!(validation::validate_dimensions(&[1000, 1000]).is_ok());

        // Invalid dimensions
        assert!(validation::validate_dimensions(&[]).is_err()); // Empty
        assert!(validation::validate_dimensions(&[0, 3]).is_err()); // Contains zero
        assert!(validation::validate_dimensions(&[2, 0]).is_err()); // Contains zero

        // Too large tensor
        assert!(validation::validate_dimensions(&[100_000, 100_000]).is_err());
    }

    #[test]
    fn test_learning_rate_validation() {
        // Valid learning rates
        assert!(validation::validate_learning_rate(0.001).is_ok());
        assert!(validation::validate_learning_rate(0.1).is_ok());
        assert!(validation::validate_learning_rate(1.0).is_ok());

        // Invalid learning rates
        assert!(validation::validate_learning_rate(0.0).is_err()); // Zero
        assert!(validation::validate_learning_rate(-0.1).is_err()); // Negative
        assert!(validation::validate_learning_rate(1.1).is_err()); // Too large
    }

    #[test]
    fn test_beta_validation() {
        // Valid beta values
        assert!(validation::validate_beta(0.0, "beta1").is_ok());
        assert!(validation::validate_beta(0.9, "beta1").is_ok());
        assert!(validation::validate_beta(0.999, "beta2").is_ok());

        // Invalid beta values
        assert!(validation::validate_beta(-0.1, "beta1").is_err()); // Negative
        assert!(validation::validate_beta(1.0, "beta2").is_err()); // Equal to 1
        assert!(validation::validate_beta(1.1, "beta1").is_err()); // Greater than 1
    }

    #[test]
    fn test_epsilon_validation() {
        // Valid epsilon values
        assert!(validation::validate_epsilon(1e-8).is_ok());
        assert!(validation::validate_epsilon(1e-6).is_ok());
        assert!(validation::validate_epsilon(0.1).is_ok());

        // Invalid epsilon values
        assert!(validation::validate_epsilon(0.0).is_err()); // Zero
        assert!(validation::validate_epsilon(-1e-8).is_err()); // Negative
    }

    #[test]
    fn test_safe_memory_access() {
        // Test safe read access
        let data = Arc::new(RwLock::new(42i32));
        let result = memory::safe_read(&data, |x| *x * 2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 84);

        // Test safe write access
        let result = memory::safe_write(&data, |x| {
            *x = 100;
            *x
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);
        assert_eq!(*data.read().unwrap(), 100);
    }

    #[test]
    fn test_conversions() {
        use pyo3::Python;

        Python::with_gil(|py| {
            // Test Vec to PyArray conversion
            let vec_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let py_array = conversions::vec_to_pyarray(vec_data.clone(), py);
            assert_eq!(py_array.len(), 4);

            // Test PyList to Vec<usize> conversion (create mock list)
            let py_list = pyo3::types::PyList::new(py, &[1, 2, 3, 4]);
            let result = conversions::pylist_to_vec_usize(&py_list.as_borrowed());
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), vec![1, 2, 3, 4]);

            // Test PyList to Vec<f32> conversion
            let py_list_f32 = pyo3::types::PyList::new(py, &[1.0f32, 2.0, 3.0]);
            let result = conversions::pylist_to_vec_f32(&py_list_f32.as_borrowed());
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);

            // Test shape conversion with validation
            let shape_list = pyo3::types::PyList::new(py, &[2, 3, 4]);
            let result = conversions::pylist_to_shape(&shape_list.as_borrowed());
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), vec![2, 3, 4]);

            // Test invalid shape conversion
            let invalid_shape = pyo3::types::PyList::new(py, &[0, 3]);
            let result = conversions::pylist_to_shape(&invalid_shape.as_borrowed());
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_conversion_error_handling() {
        use pyo3::Python;

        Python::with_gil(|py| {
            // Test type conversion errors
            let mixed_list = pyo3::types::PyList::new(py, &[1, "not_a_number", 3]);
            let result = conversions::pylist_to_vec_usize(&mixed_list.as_borrowed());
            assert!(result.is_err());

            let invalid_float_list = pyo3::types::PyList::new(py, &[1.0, "not_a_float"]);
            let result = conversions::pylist_to_vec_f32(&invalid_float_list.as_borrowed());
            assert!(result.is_err());
        });
    }

    // Mock struct for testing PyWrapper trait
    #[derive(Clone, Debug, PartialEq)]
    struct MockRustType {
        value: i32,
    }

    #[derive(Clone)]
    struct MockPyWrapper {
        inner: MockRustType,
    }

    impl PyWrapper<MockRustType> for MockPyWrapper {
        fn from_rust(value: MockRustType) -> Self {
            Self { inner: value }
        }

        fn to_rust(&self) -> &MockRustType {
            &self.inner
        }

        fn into_rust(self) -> MockRustType {
            self.inner
        }
    }

    #[test]
    fn test_py_wrapper_trait() {
        let rust_value = MockRustType { value: 42 };
        let py_wrapper = MockPyWrapper::from_rust(rust_value.clone());

        assert_eq!(py_wrapper.to_rust(), &rust_value);
        assert_eq!(py_wrapper.into_rust(), rust_value);
    }

    // Mock struct for testing ThreadSafePyWrapper trait
    struct MockThreadSafePyWrapper {
        inner: Arc<RwLock<MockRustType>>,
    }

    impl ThreadSafePyWrapper<MockRustType> for MockThreadSafePyWrapper {
        fn from_arc_rwlock(value: Arc<RwLock<MockRustType>>) -> Self {
            Self { inner: value }
        }

        fn as_arc_rwlock(&self) -> &Arc<RwLock<MockRustType>> {
            &self.inner
        }

        fn clone_arc_rwlock(&self) -> Arc<RwLock<MockRustType>> {
            Arc::clone(&self.inner)
        }
    }

    #[test]
    fn test_thread_safe_py_wrapper_trait() {
        let rust_value = MockRustType { value: 100 };
        let arc_value = Arc::new(RwLock::new(rust_value));
        let py_wrapper = MockThreadSafePyWrapper::from_arc_rwlock(arc_value.clone());

        assert_eq!(py_wrapper.as_arc_rwlock().read().unwrap().value, 100);

        let cloned_arc = py_wrapper.clone_arc_rwlock();
        assert_eq!(cloned_arc.read().unwrap().value, 100);
    }

    #[test]
    fn test_validation_edge_cases() {
        // Test dimension validation edge cases
        assert!(validation::validate_dimensions(&[1; 10]).is_ok()); // Many small dimensions
        assert!(validation::validate_dimensions(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).is_ok()); // Many dimensions

        // Test learning rate edge cases
        assert!(validation::validate_learning_rate(f64::EPSILON).is_ok()); // Very small positive
        assert!(validation::validate_learning_rate(1.0 - f64::EPSILON).is_ok()); // Just under 1.0

        // Test beta edge cases
        assert!(validation::validate_beta(f64::EPSILON, "beta").is_ok()); // Very small positive
        assert!(validation::validate_beta(1.0 - f64::EPSILON, "beta").is_ok()); // Just under 1.0

        // Test epsilon edge cases
        assert!(validation::validate_epsilon(f64::MIN_POSITIVE).is_ok()); // Smallest positive
    }
}

#[cfg(all(test, feature = "python"))]
mod integration_tests {
    use pyo3::Python;
    use rustorch::python::autograd::PyVariable;
    use rustorch::python::tensor::PyTensor;

    #[test]
    fn test_tensor_creation_and_operations() {
        Python::with_gil(|py| {
            // Test tensor creation
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];
            let tensor = PyTensor::new(data, shape);
            assert!(tensor.is_ok());

            let tensor = tensor.unwrap();
            assert_eq!(tensor.shape(), vec![2, 2]);
            assert_eq!(tensor.numel(), 4);
            assert_eq!(tensor.ndim(), 2);

            // Test tensor operations
            let zeros = PyTensor::zeros(vec![2, 2]);
            assert!(zeros.is_ok());

            let ones = PyTensor::ones(vec![2, 2]);
            assert!(ones.is_ok());

            let randn = PyTensor::randn(vec![2, 2]);
            assert!(randn.is_ok());

            // Test arithmetic operations
            let zeros = zeros.unwrap();
            let ones = ones.unwrap();

            let sum_result = tensor.__add__(&zeros);
            assert!(sum_result.is_ok());

            let mul_result = tensor.__mul__(&ones);
            assert!(mul_result.is_ok());
        });
    }

    #[test]
    fn test_tensor_validation() {
        // Test invalid tensor creation
        let invalid_data = vec![1.0, 2.0];
        let invalid_shape = vec![0, 2]; // Contains zero
        let result = PyTensor::new(invalid_data, invalid_shape);
        assert!(result.is_err());

        // Test empty shape
        let data = vec![1.0];
        let empty_shape = vec![];
        let result = PyTensor::new(data, empty_shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_creation_and_operations() {
        Python::with_gil(|py| {
            // Test variable creation
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];
            let var = PyVariable::from_data(data, shape, Some(true));
            assert!(var.is_ok());

            let var = var.unwrap();
            assert_eq!(var.shape(), vec![2, 2]);
            assert_eq!(var.numel(), 4);
            assert!(var.requires_grad());

            // Test mathematical operations
            let pow_result = var.pow(2.0);
            assert!(pow_result.is_ok());

            let exp_result = var.exp();
            assert!(exp_result.is_ok());

            let log_result = var.log();
            assert!(log_result.is_ok());

            let sin_result = var.sin();
            assert!(sin_result.is_ok());

            let cos_result = var.cos();
            assert!(cos_result.is_ok());

            let sqrt_result = var.sqrt();
            assert!(sqrt_result.is_ok());
        });
    }

    #[test]
    fn test_variable_gradient_operations() {
        Python::with_gil(|py| {
            // Test variable with gradients
            let var = PyVariable::from_data(vec![2.0, 3.0], vec![2], Some(true));
            assert!(var.is_ok());

            let mut var = var.unwrap();

            // Test backward pass (simplified)
            let backward_result = var.backward(None, None);
            assert!(backward_result.is_ok());

            // Test zero grad
            let zero_grad_result = var.zero_grad();
            assert!(zero_grad_result.is_ok());

            // Test detach
            let detached = var.detach();
            assert!(detached.is_ok());
            assert!(!detached.unwrap().requires_grad());
        });
    }
}

// Performance and stress tests
#[cfg(all(test, feature = "python"))]
mod performance_tests {
    use rustorch::python::common::validation;
    use rustorch::python::tensor::PyTensor;
    use std::time::Instant;

    #[test]
    fn test_large_tensor_creation() {
        let start = Instant::now();

        // Create moderately large tensors
        for _ in 0..100 {
            let size = 100;
            let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
            let tensor = PyTensor::new(data, vec![size, size]);
            assert!(tensor.is_ok());
        }

        let duration = start.elapsed();
        println!("Created 100 tensors of size 100x100 in {:?}", duration);

        // Should complete in reasonable time (adjust threshold as needed)
        assert!(duration.as_secs() < 5);
    }

    #[test]
    fn test_validation_performance() {
        let start = Instant::now();

        // Test validation performance
        for _ in 0..10000 {
            assert!(validation::validate_dimensions(&[32, 32, 3]).is_ok());
            assert!(validation::validate_learning_rate(0.001).is_ok());
            assert!(validation::validate_beta(0.9, "beta1").is_ok());
            assert!(validation::validate_epsilon(1e-8).is_ok());
        }

        let duration = start.elapsed();
        println!("Performed 40000 validations in {:?}", duration);

        // Validation should be very fast
        assert!(duration.as_millis() < 100);
    }

    #[test]
    fn test_memory_safety_stress() {
        use rustorch::python::common::memory;
        use std::sync::{Arc, RwLock};
        use std::thread;

        let data = Arc::new(RwLock::new(0i32));
        let mut handles = vec![];

        // Spawn multiple threads accessing the same data
        for i in 0..10 {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let result = memory::safe_write(&data_clone, |x| {
                        *x += 1;
                        *x
                    });
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final result
        let final_value = memory::safe_read(&data, |x| *x).unwrap();
        assert_eq!(final_value, 1000); // 10 threads * 100 increments
    }
}
