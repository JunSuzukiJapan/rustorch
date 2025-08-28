//! Tests for enhanced error handling in tensor operations
//! テンソル操作の拡張エラーハンドリングのテスト

use crate::tensor::Tensor;
use crate::error::RusTorchError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_from_vec_success() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        
        let result = Tensor::try_from_vec(data, shape);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test] 
    fn test_try_from_vec_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0]; // 3 elements
        let shape = vec![2, 2]; // expects 4 elements
        
        let result = Tensor::try_from_vec(data, shape);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            RusTorchError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, vec![4]); // expected total elements
                assert_eq!(actual, vec![3]);   // actual total elements
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_try_from_vec_empty_shape() {
        let data = vec![1.0f32];
        let shape = vec![];
        
        let result = Tensor::try_from_vec(data, shape);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            RusTorchError::TensorOp { message, .. } => {
                assert!(message.contains("Shape cannot be empty"));
            }
            _ => panic!("Expected TensorOp error"),
        }
    }

    #[test]
    fn test_try_from_vec_zero_dimension() {
        let data: Vec<f32> = vec![];
        let shape = vec![0, 2];
        
        let result = Tensor::try_from_vec(data, shape);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            RusTorchError::TensorOp { message, .. } => {
                assert!(message.contains("zero dimension"));
            }
            _ => panic!("Expected TensorOp error"),
        }
    }

    #[test]
    fn test_try_view_success() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        let result = tensor.try_view(&[4]);
        assert!(result.is_ok());
        
        let reshaped = result.unwrap();
        assert_eq!(reshaped.shape(), &[4]);
    }

    #[test]
    fn test_try_view_shape_mismatch() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        let result = tensor.try_view(&[3]); // 4 elements can't be viewed as [3]
        assert!(result.is_err());
        
        match result.unwrap_err() {
            RusTorchError::ShapeMismatch { .. } => {
                // Expected error
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_try_zeros_success() {
        let result = Tensor::<f32>::try_zeros(&[2, 3]);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        // All elements should be zero
        for &value in tensor.as_slice().unwrap() {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn test_try_zeros_empty_shape() {
        let result = Tensor::<f32>::try_zeros(&[]);
        assert!(result.is_err());
    }

    #[test] 
    fn test_try_zeros_zero_dimension() {
        let result = Tensor::<f32>::try_zeros(&[2, 0, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_ones_success() {
        let result = Tensor::<f32>::try_ones(&[2, 2]);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        // All elements should be one
        for &value in tensor.as_slice().unwrap() {
            assert_eq!(value, 1.0);
        }
    }

    #[test]
    #[should_panic] // This test should panic due to memory constraints
    fn test_try_zeros_too_large() {
        // This would require too much memory - should return an error
        let result = Tensor::<f32>::try_zeros(&[1_000_000, 1_000_000]);
        
        // In a real test, we'd check that it returns an error
        // But for safety, we use should_panic to catch any actual OOM
        result.unwrap();
    }

    #[test]
    fn test_try_zeros_reasonable_size() {
        // Test a reasonably large but manageable size
        let result = Tensor::<f32>::try_zeros(&[100, 100]);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[100, 100]);
    }
}