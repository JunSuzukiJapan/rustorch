//! hybrid_f32コアテスト
//! hybrid_f32 core tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::core::F32Tensor;
    use rustorch::hybrid_f32::error::{F32Error, F32Result};

    /// テストヘルパー関数
    /// Test helper functions
    fn assert_tensor_eq(a: &F32Tensor, b: &F32Tensor, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Shape mismatch");
        
        for i in 0..a.numel() {
            let val_a = a.data.as_slice().unwrap()[i];
            let val_b = b.data.as_slice().unwrap()[i];
            assert!(
                (val_a - val_b).abs() < tolerance,
                "Values differ: {} vs {} (tolerance: {})",
                val_a, val_b, tolerance
            );
        }
    }

    #[test]
    fn test_tensor_creation() {
        // ベクター作成
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.ndim(), 1);

        // 多次元作成
        let tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);

        // ゼロテンソル
        let zeros = F32Tensor::zeros(&[3, 3]).unwrap();
        assert_eq!(zeros.shape(), &[3, 3]);
        assert_eq!(zeros.numel(), 9);
        for &val in zeros.data.as_slice().unwrap() {
            assert_eq!(val, 0.0);
        }

        // ワンテンソル
        let ones = F32Tensor::ones(&[2, 3]).unwrap();
        assert_eq!(ones.shape(), &[2, 3]);
        assert_eq!(ones.numel(), 6);
        for &val in ones.data.as_slice().unwrap() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ).unwrap();

        // 基本プロパティ
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());

        // スカラーテンソル
        let scalar = F32Tensor::from_vec(vec![42.0], &[1]).unwrap();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value().unwrap(), 42.0);

        // 空テンソル
        let empty = F32Tensor::zeros(&[0]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_gradient_tracking() {
        let mut tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        
        // デフォルトでは勾配追跡なし
        assert!(!tensor.is_grad_enabled());

        // 勾配追跡を有効化
        tensor.requires_grad(true);
        assert!(tensor.is_grad_enabled());

        // 勾配追跡を無効化
        tensor.requires_grad(false);
        assert!(!tensor.is_grad_enabled());
    }

    #[test]
    fn test_device_state() {
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        
        // デフォルトはCPU
        assert!(matches!(tensor.device_state(), 
                         rustorch::hybrid_f32::tensor::core::DeviceState::CPU));
    }

    #[test]
    fn test_tensor_cloning() {
        let original = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ).unwrap();
        let cloned = original.clone();

        // 同じ形状とデータ
        assert_eq!(original.shape(), cloned.shape());
        assert_tensor_eq(&original, &cloned, 1e-10);
    }

    #[test]
    fn test_error_handling() {
        // 無効なテンソル作成
        let invalid = F32Tensor::new(vec![1.0, 2.0], &[3, 3]);
        assert!(invalid.is_err());

        // スカラー値取得エラー
        let non_scalar = F32Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(non_scalar.scalar_value().is_err());
    }

    #[test]
    fn test_edge_cases() {
        // 最小サイズテンソル
        let single = F32Tensor::from_vec(vec![42.0], &[1]).unwrap();
        assert!(single.is_scalar());
        assert_eq!(single.scalar_value().unwrap(), 42.0);

        // 大きなテンソル（メモリテスト）
        let large_size = 10000;
        let large_tensor = F32Tensor::zeros(&[large_size]).unwrap();
        assert_eq!(large_tensor.numel(), large_size);

        // 高次元テンソル
        let high_dim = F32Tensor::zeros(&[2, 3, 4, 5]).unwrap();
        assert_eq!(high_dim.ndim(), 4);
        assert_eq!(high_dim.numel(), 120);
    }

    #[test]
    fn test_numerical_precision() {
        // 浮動小数点精度テスト
        let a = F32Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        
        // f32の精度を考慮した比較
        assert!((a.data.as_slice().unwrap()[0] - 0.1).abs() < 1e-6);
        assert!((a.data.as_slice().unwrap()[1] - 0.2).abs() < 1e-6);

        // 特殊値のテスト
        let special = F32Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN], &[3]).unwrap();
        assert!(special.data.as_slice().unwrap()[0].is_infinite());
        assert!(special.data.as_slice().unwrap()[1].is_infinite());
        assert!(special.data.as_slice().unwrap()[2].is_nan());
    }
}