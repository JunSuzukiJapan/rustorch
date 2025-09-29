//! hybrid_f32最小テスト
//! hybrid_f32 minimal test

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::core::{F32Tensor, DeviceState};

    #[test]
    fn test_f32_tensor_creation() {
        // 基本的なテンソル作成
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.ndim(), 1);
    }

    #[test]
    fn test_tensor_zeros_ones() {
        // ゼロテンソル
        let zeros = F32Tensor::zeros(&[2, 2]).unwrap();
        assert_eq!(zeros.shape(), &[2, 2]);
        assert_eq!(zeros.numel(), 4);
        for &val in zeros.data.as_slice().unwrap() {
            assert_eq!(val, 0.0);
        }

        // ワンテンソル
        let ones = F32Tensor::ones(&[2, 2]).unwrap();
        assert_eq!(ones.shape(), &[2, 2]);
        assert_eq!(ones.numel(), 4);
        for &val in ones.data.as_slice().unwrap() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        
        // 形状関連
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());

        // デバイス状態
        assert!(matches!(tensor.device_state(), DeviceState::CPU));

        // 勾配追跡
        assert!(!tensor.is_grad_enabled());
    }

    #[test]
    fn test_scalar_tensor() {
        let scalar = F32Tensor::from_vec(vec![42.0], &[1]).unwrap();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value().unwrap(), 42.0);
    }

    #[test]
    fn test_tensor_cloning() {
        let original = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let cloned = original.clone();

        assert_eq!(original.shape(), cloned.shape());
        assert_eq!(original.numel(), cloned.numel());
    }

    #[test]
    fn test_gradient_tracking() {
        let mut tensor = F32Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        assert!(!tensor.is_grad_enabled());

        tensor.requires_grad(true);
        assert!(tensor.is_grad_enabled());

        tensor.requires_grad(false);
        assert!(!tensor.is_grad_enabled());
    }

    #[test]
    fn test_error_cases() {
        // 無効な形状
        let invalid = F32Tensor::new(vec![1.0, 2.0], &[3, 3]);
        assert!(invalid.is_err());

        // 非スカラーからのスカラー値取得
        let non_scalar = F32Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(non_scalar.scalar_value().is_err());
    }
}