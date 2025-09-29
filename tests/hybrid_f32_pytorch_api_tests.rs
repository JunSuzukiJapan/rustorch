//! hybrid_f32 PyTorchライクAPI テスト
//! hybrid_f32 PyTorch-like API tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::error::RusTorchResult;
    use rustorch::hybrid_f32::tensor::core::{Index2D, Index3D};
    use rustorch::hybrid_f32::tensor::F32Tensor;

    #[test]
    fn test_operator_overloading() -> RusTorchResult<()> {
        let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = F32Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

        // テンソル + テンソル
        let result = (&a + &b)?;
        assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

        // テンソル - テンソル
        let result = (&a - &b)?;
        assert_eq!(result.as_slice(), &[-4.0, -4.0, -4.0, -4.0]);

        // テンソル * テンソル（要素積）
        let result = (&a * &b)?;
        assert_eq!(result.as_slice(), &[5.0, 12.0, 21.0, 32.0]);

        // テンソル / テンソル
        let result = (&a / &b)?;
        assert_eq!(
            result.as_slice(),
            &[1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0, 4.0 / 8.0]
        );

        Ok(())
    }

    #[test]
    fn test_scalar_operations() -> RusTorchResult<()> {
        let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // テンソル + スカラー
        let result = (&a + 10.0)?;
        assert_eq!(result.as_slice(), &[11.0, 12.0, 13.0, 14.0]);

        // テンソル * スカラー
        let result = (&a * 2.0)?;
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);

        // テンソル / スカラー
        let result = (&a / 2.0)?;
        assert_eq!(result.as_slice(), &[0.5, 1.0, 1.5, 2.0]);

        // 負数
        let result = (-&a)?;
        assert_eq!(result.as_slice(), &[-1.0, -2.0, -3.0, -4.0]);

        Ok(())
    }

    #[test]
    fn test_1d_indexing() -> RusTorchResult<()> {
        let mut tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;

        // 読み取り
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[2], 3.0);
        assert_eq!(tensor[4], 5.0);

        // 書き込み
        tensor[1] = 10.0;
        assert_eq!(tensor[1], 10.0);
        assert_eq!(tensor.as_slice(), &[1.0, 10.0, 3.0, 4.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_2d_indexing() -> RusTorchResult<()> {
        let mut tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

        // 読み取り
        assert_eq!(tensor[Index2D(0, 0)], 1.0);
        assert_eq!(tensor[Index2D(0, 2)], 3.0);
        assert_eq!(tensor[Index2D(1, 1)], 5.0);

        // 書き込み
        tensor[Index2D(1, 2)] = 20.0;
        assert_eq!(tensor[Index2D(1, 2)], 20.0);

        Ok(())
    }

    #[test]
    fn test_3d_indexing() -> RusTorchResult<()> {
        let mut tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

        // 読み取り
        assert_eq!(tensor[Index3D(0, 0, 0)], 1.0);
        assert_eq!(tensor[Index3D(1, 1, 1)], 8.0);

        // 書き込み
        tensor[Index3D(0, 1, 0)] = 100.0;
        assert_eq!(tensor[Index3D(0, 1, 0)], 100.0);

        Ok(())
    }

    #[test]
    fn test_try_methods_success() -> RusTorchResult<()> {
        let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = F32Tensor::new(vec![2.0, 2.0, 2.0, 2.0], &[2, 2])?;

        // 成功ケース
        let result = a.try_add(&b)?;
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 6.0]);

        let result = a.try_mul_scalar(2.0)?;
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);

        let result = a.try_reshape(&[4, 1])?;
        assert_eq!(result.shape(), &[4, 1]);

        // 2D転置
        let result = a.try_transpose()?;
        assert_eq!(result.shape(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_try_methods_error_cases() {
        let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // 無効なスカラー値
        assert!(a.try_mul_scalar(f32::NAN).is_err());
        assert!(a.try_mul_scalar(f32::INFINITY).is_err());

        // 無効な形状変更
        assert!(a.try_reshape(&[3, 2]).is_err()); // 4要素 → 6要素

        // 1Dテンソルでの転置
        let b = F32Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(b.try_transpose().is_err());

        // 無効なスライス
        assert!(a.try_slice(&[(0, 1)]).is_err()); // 2Dテンソルに1つの範囲
        assert!(a.try_slice(&[(0, 3), (0, 2)]).is_err()); // 範囲外
    }

    #[test]
    fn test_safe_element_access() -> RusTorchResult<()> {
        let mut tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // 正常アクセス
        assert_eq!(tensor.try_get(&[0, 1])?, 2.0);
        assert_eq!(tensor.try_get(&[1, 0])?, 3.0);

        // 正常設定
        tensor.try_set(&[1, 1], 10.0)?;
        assert_eq!(tensor.try_get(&[1, 1])?, 10.0);

        // エラーケース
        assert!(tensor.try_get(&[2, 0]).is_err()); // 範囲外
        assert!(tensor.try_get(&[0]).is_err()); // 次元不一致
        assert!(tensor.try_set(&[0, 0], f32::NAN).is_err()); // 無効値

        Ok(())
    }

    #[test]
    fn test_division_by_zero() {
        let a = F32Tensor::new(vec![1.0, 2.0], &[2]).unwrap();

        // ゼロ除算エラー
        let result = &a / 0.0;
        assert!(result.is_err());
    }

    #[test]
    fn test_chained_operations() -> RusTorchResult<()> {
        let a = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = F32Tensor::new(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;

        // 連鎖演算
        let result = ((&a + &b)? * 2.0)? - 1.0;
        assert_eq!(result?.as_slice(), &[3.0, 5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_type_conversion() -> RusTorchResult<()> {
        let tensor = F32Tensor::new(vec![1.0, 2.5, 3.7, 4.2], &[4])?;

        // f64への変換（浮動小数点精度を考慮）
        let f64_vec: Vec<f64> = tensor.try_to_type()?;
        let expected = vec![1.0, 2.5, 3.7, 4.2];
        assert_eq!(f64_vec.len(), expected.len());
        for (actual, expected) in f64_vec.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Expected {}, got {}",
                expected,
                actual
            );
        }

        // 空テンソル
        let empty = F32Tensor::zeros(&[0])?;
        let empty_vec: Vec<f64> = empty.try_to_type()?;
        assert!(empty_vec.is_empty());

        Ok(())
    }

    #[test]
    fn test_feature_availability() {
        let mut tensor = F32Tensor::new(vec![1.0, 2.0], &[2]).unwrap();

        // Metal/CoreMLの利用可能性テスト
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Metalが利用可能な場合のテスト
            // 実際のハードウェアがない場合はエラーが予想される
            let _result = tensor.try_to_metal(0);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            // Metalが利用不可能な場合
            assert!(tensor.try_to_metal(0).is_err());
        }

        #[cfg(all(target_os = "macos", feature = "coreml"))]
        {
            // CoreMLが利用可能な場合のテスト
            let _result = tensor.try_to_coreml(0);
        }

        #[cfg(not(all(target_os = "macos", feature = "coreml")))]
        {
            // CoreMLが利用不可能な場合
            assert!(tensor.try_to_coreml(0).is_err());
        }
    }
}
