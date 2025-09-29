//! hybrid_f32統合テスト
//! hybrid_f32 integration tests

use rustorch::hybrid_f32::tensor::core::F32Tensor;
use rustorch::hybrid_f32::tensor::indexing::{Index2D, Index3D, SliceRange};
use rustorch::hybrid_f32::error::{F32Error, F32Result};

#[cfg(test)]
mod integration_tests {
    use super::*;

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

    fn create_test_tensor_1d() -> F32Tensor {
        F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()
    }

    fn create_test_tensor_2d() -> F32Tensor {
        F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap()
    }

    fn create_test_tensor_3d() -> F32Tensor {
        F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        ).unwrap()
    }

    #[test]
    fn test_tensor_creation() {
        // ベクター作成
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.ndim(), 1);

        // 多次元作成
        let tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);

        // ゼロテンソル
        let zeros = F32Tensor::zeros(vec![3, 3]).unwrap();
        assert_eq!(zeros.shape(), &[3, 3]);
        assert_eq!(zeros.numel(), 9);
        for &val in zeros.data.as_slice().unwrap() {
            assert_eq!(val, 0.0);
        }

        // ワンテンソル
        let ones = F32Tensor::ones(vec![2, 3]).unwrap();
        assert_eq!(ones.shape(), &[2, 3]);
        assert_eq!(ones.numel(), 6);
        for &val in ones.data.as_slice().unwrap() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = create_test_tensor_2d();

        // 基本プロパティ
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());

        // スカラーテンソル
        let scalar = F32Tensor::from_vec(vec![42.0]).unwrap();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value().unwrap(), 42.0);

        // 空テンソル
        let empty = F32Tensor::zeros(vec![0]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_arithmetic_operators() {
        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
        let b = F32Tensor::from_vec(vec![4.0, 5.0, 6.0]).unwrap();

        // テンソル同士の演算
        let sum = (&a + &b).unwrap();
        let expected_sum = F32Tensor::from_vec(vec![5.0, 7.0, 9.0]).unwrap();
        assert_tensor_eq(&sum, &expected_sum, 1e-6);

        let diff = (&a - &b).unwrap();
        let expected_diff = F32Tensor::from_vec(vec![-3.0, -3.0, -3.0]).unwrap();
        assert_tensor_eq(&diff, &expected_diff, 1e-6);

        let prod = (&a * &b).unwrap();
        let expected_prod = F32Tensor::from_vec(vec![4.0, 10.0, 18.0]).unwrap();
        assert_tensor_eq(&prod, &expected_prod, 1e-6);

        let quot = (&b / &a).unwrap();
        let expected_quot = F32Tensor::from_vec(vec![4.0, 2.5, 2.0]).unwrap();
        assert_tensor_eq(&quot, &expected_quot, 1e-6);

        // スカラー演算
        let scalar_sum = (&a + 10.0).unwrap();
        let expected_scalar_sum = F32Tensor::from_vec(vec![11.0, 12.0, 13.0]).unwrap();
        assert_tensor_eq(&scalar_sum, &expected_scalar_sum, 1e-6);

        let scalar_prod = (&a * 2.0).unwrap();
        let expected_scalar_prod = F32Tensor::from_vec(vec![2.0, 4.0, 6.0]).unwrap();
        assert_tensor_eq(&scalar_prod, &expected_scalar_prod, 1e-6);

        // 負号
        let neg = (-&a).unwrap();
        let expected_neg = F32Tensor::from_vec(vec![-1.0, -2.0, -3.0]).unwrap();
        assert_tensor_eq(&neg, &expected_neg, 1e-6);
    }

    #[test]
    fn test_indexing_operations() {
        // 1次元インデックス
        let tensor_1d = create_test_tensor_1d();
        assert_eq!(tensor_1d[0], 1.0);
        assert_eq!(tensor_1d[2], 3.0);
        assert_eq!(tensor_1d[4], 5.0);

        // 2次元インデックス
        let tensor_2d = create_test_tensor_2d();
        assert_eq!(tensor_2d[Index2D(0, 0)], 1.0);
        assert_eq!(tensor_2d[Index2D(0, 2)], 3.0);
        assert_eq!(tensor_2d[Index2D(1, 1)], 5.0);
        assert_eq!(tensor_2d[Index2D(1, 2)], 6.0);

        // 3次元インデックス
        let tensor_3d = create_test_tensor_3d();
        assert_eq!(tensor_3d[Index3D(0, 0, 0)], 1.0);
        assert_eq!(tensor_3d[Index3D(0, 1, 1)], 4.0);
        assert_eq!(tensor_3d[Index3D(1, 0, 1)], 6.0);
        assert_eq!(tensor_3d[Index3D(1, 1, 1)], 8.0);
    }

    #[test]
    fn test_mutable_indexing() {
        // 1次元
        let mut tensor_1d = create_test_tensor_1d();
        tensor_1d[1] = 20.0;
        assert_eq!(tensor_1d[1], 20.0);

        // 2次元
        let mut tensor_2d = create_test_tensor_2d();
        tensor_2d[Index2D(1, 0)] = 30.0;
        assert_eq!(tensor_2d[Index2D(1, 0)], 30.0);

        // 3次元
        let mut tensor_3d = create_test_tensor_3d();
        tensor_3d[Index3D(0, 1, 0)] = 40.0;
        assert_eq!(tensor_3d[Index3D(0, 1, 0)], 40.0);
    }

    #[test]
    fn test_safe_indexing() {
        let tensor = create_test_tensor_2d();

        // 正常ケース
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);

        // エラーケース
        assert!(tensor.get(&[2, 0]).is_err()); // 行が範囲外
        assert!(tensor.get(&[0, 3]).is_err()); // 列が範囲外
        assert!(tensor.get(&[0]).is_err());    // 次元数不一致
        assert!(tensor.get(&[0, 1, 2]).is_err()); // 次元数不一致

        // 安全な設定
        let mut tensor_mut = create_test_tensor_2d();
        assert!(tensor_mut.set(&[1, 1], 99.0).is_ok());
        assert_eq!(tensor_mut.get(&[1, 1]).unwrap(), 99.0);

        assert!(tensor_mut.set(&[2, 0], 99.0).is_err()); // 範囲外
    }

    #[test]
    fn test_slicing() {
        let tensor = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]

        // 全行、列1-3取得
        let slice1 = tensor.slice(&[
            SliceRange::all(),
            SliceRange::range(1, 3),
        ]).unwrap();
        
        assert_eq!(slice1.shape(), &[2, 2]);
        assert_eq!(slice1.get(&[0, 0]).unwrap(), 2.0); // tensor[0, 1]
        assert_eq!(slice1.get(&[0, 1]).unwrap(), 3.0); // tensor[0, 2]
        assert_eq!(slice1.get(&[1, 0]).unwrap(), 5.0); // tensor[1, 1]
        assert_eq!(slice1.get(&[1, 1]).unwrap(), 6.0); // tensor[1, 2]

        // 行1のみ、全列
        let slice2 = tensor.slice(&[
            SliceRange::range(1, 2),
            SliceRange::all(),
        ]).unwrap();
        
        assert_eq!(slice2.shape(), &[1, 3]);
        assert_eq!(slice2.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(slice2.get(&[0, 1]).unwrap(), 5.0);
        assert_eq!(slice2.get(&[0, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_error_handling() {
        // 形状不一致エラー
        let a = F32Tensor::from_vec(vec![1.0, 2.0]).unwrap();
        let b = F32Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
        
        let result = &a + &b;
        assert!(result.is_err());
        
        if let Err(err) = result {
            assert!(matches!(err, F32Error::ShapeMismatch { .. }));
        }

        // ゼロ除算エラー
        let zero_div = &a / 0.0;
        assert!(zero_div.is_err());

        // 無効なテンソル作成
        let invalid = F32Tensor::new(vec![1.0, 2.0], vec![3, 3]);
        assert!(invalid.is_err());

        // スカラー値取得エラー
        let non_scalar = F32Tensor::from_vec(vec![1.0, 2.0]).unwrap();
        assert!(non_scalar.scalar_value().is_err());
    }

    #[test]
    fn test_tensor_cloning() {
        let original = create_test_tensor_2d();
        let cloned = original.clone();

        // 同じ形状とデータ
        assert_eq!(original.shape(), cloned.shape());
        assert_tensor_eq(&original, &cloned, 1e-10);

        // 独立したデータ
        let mut modified = cloned;
        modified[Index2D(0, 0)] = 999.0;
        
        assert_ne!(original[Index2D(0, 0)], modified[Index2D(0, 0)]);
    }

    #[test]
    fn test_gradient_tracking() {
        let mut tensor = create_test_tensor_1d();
        
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
        let tensor = create_test_tensor_1d();
        
        // デフォルトはCPU
        assert!(matches!(tensor.device_state(), 
                         rustorch::hybrid_f32::tensor::core::DeviceState::CPU));
    }

    #[test]
    fn test_edge_cases() {
        // 最小サイズテンソル
        let single = F32Tensor::from_vec(vec![42.0]).unwrap();
        assert!(single.is_scalar());
        assert_eq!(single.scalar_value().unwrap(), 42.0);

        // 大きなテンソル（メモリテスト）
        let large_size = 10000;
        let large_tensor = F32Tensor::zeros(vec![large_size]).unwrap();
        assert_eq!(large_tensor.numel(), large_size);

        // 高次元テンソル
        let high_dim = F32Tensor::zeros(vec![2, 3, 4, 5]).unwrap();
        assert_eq!(high_dim.ndim(), 4);
        assert_eq!(high_dim.numel(), 120);
    }

    #[test]
    fn test_numerical_precision() {
        // 浮動小数点精度テスト
        let a = F32Tensor::from_vec(vec![0.1, 0.2]).unwrap();
        let b = F32Tensor::from_vec(vec![0.3, 0.4]).unwrap();
        
        let sum = (&a + &b).unwrap();
        
        // f32の精度を考慮した比較
        assert!((sum[0] - 0.4).abs() < 1e-6);
        assert!((sum[1] - 0.6).abs() < 1e-6);

        // 特殊値のテスト
        let special = F32Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).unwrap();
        assert!(special[0].is_infinite());
        assert!(special[1].is_infinite());
        assert!(special[2].is_nan());
    }

    #[test]
    fn test_performance_characteristics() {
        use std::time::Instant;

        // 大量のテンソル演算のパフォーマンステスト
        let size = 1000;
        let a = F32Tensor::ones(vec![size]).unwrap();
        let b = F32Tensor::ones(vec![size]).unwrap();

        let start = Instant::now();
        for _ in 0..100 {
            let _result = (&a + &b).unwrap();
        }
        let duration = start.elapsed();

        // パフォーマンスの妥当性確認（具体的な閾値は環境依存）
        assert!(duration.as_millis() < 1000, "Addition too slow: {:?}", duration);

        println!("100 additions of size {} took: {:?}", size, duration);
    }
}