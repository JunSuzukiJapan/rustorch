//! Basic F32Tensor test - standalone verification
//! Ensures core tensor functionality works without dependencies

#[cfg(feature = "hybrid-f32")]
mod f32_tensor_basic {
    use rustorch::error::RusTorchResult;
    use rustorch::hybrid_f32::tensor::core::{DeviceState, F32Tensor};

    #[test]
    fn test_basic_tensor_creation() -> RusTorchResult<()> {
        // Basic tensor creation
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.ndim(), 1);
        assert!(!tensor.is_empty());
        Ok(())
    }

    #[test]
    fn test_zeros_and_ones() -> RusTorchResult<()> {
        // Zero tensor
        let zeros = F32Tensor::zeros(&[2, 2])?;
        assert_eq!(zeros.shape(), &[2, 2]);
        assert_eq!(zeros.numel(), 4);

        // One tensor
        let ones = F32Tensor::ones(&[2, 2])?;
        assert_eq!(ones.shape(), &[2, 2]);
        assert_eq!(ones.numel(), 4);

        Ok(())
    }

    #[test]
    fn test_basic_operations() -> RusTorchResult<()> {
        let a = F32Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let b = F32Tensor::from_vec(vec![3.0, 4.0], &[2])?;

        // Addition
        let sum = a.add(&b)?;
        assert_eq!(sum.shape(), &[2]);

        // Multiplication
        let mul = a.mul(&b)?;
        assert_eq!(mul.shape(), &[2]);

        Ok(())
    }

    #[test]
    fn test_scalar_operations() -> RusTorchResult<()> {
        let scalar = F32Tensor::from_vec(vec![42.0], &[1])?;
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value()?, 42.0);
        Ok(())
    }

    #[test]
    fn test_sum_operation() -> RusTorchResult<()> {
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let total = tensor.sum()?;
        assert_eq!(total, 6.0);
        Ok(())
    }

    #[test]
    fn test_comparison_operations() -> RusTorchResult<()> {
        let a = F32Tensor::from_vec(vec![1.0, 3.0], &[2])?;
        let b = F32Tensor::from_vec(vec![2.0, 2.0], &[2])?;

        let gt = a.gt(&b)?;
        assert_eq!(gt.shape(), &[2]);

        Ok(())
    }

    #[test]
    fn test_device_state() -> RusTorchResult<()> {
        let tensor = F32Tensor::from_vec(vec![1.0], &[1])?;
        assert!(matches!(tensor.device_state(), DeviceState::CPU));
        Ok(())
    }

    #[test]
    fn test_cloning() -> RusTorchResult<()> {
        let original = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let cloned = original.clone();

        assert_eq!(original.shape(), cloned.shape());
        assert_eq!(original.numel(), cloned.numel());
        Ok(())
    }
}
