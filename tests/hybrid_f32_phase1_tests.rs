//! hybrid_f32 フェーズ1基本演算テスト
//! hybrid_f32 Phase 1 Basic Operations Tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::F32Tensor;
    use rustorch::error::RusTorchResult;

    #[test]
    fn test_tensor_creation_methods() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // zeros
        let zeros = F32Tensor::zeros(&[2, 3])?;
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.numel(), 6);
        assert!(zeros.as_slice().iter().all(|&x| x == 0.0));

        // ones
        let ones = F32Tensor::ones(&[2, 2])?;
        assert_eq!(ones.shape(), &[2, 2]);
        assert!(ones.as_slice().iter().all(|&x| x == 1.0));

        // from_vec
        let from_vec = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert_eq!(from_vec.shape(), &[2, 2]);
        assert_eq!(from_vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        // new (同じ機能)
        let tensor = F32Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.as_slice(), &[5.0, 6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_basic_tensor_properties() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

        // 基本プロパティ
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());

        // スカラーテンソル
        let scalar = F32Tensor::from_vec(vec![42.0], &[1])?;
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value()?, 42.0);

        // 空テンソル
        let empty = F32Tensor::zeros(&[0])?;
        assert!(empty.is_empty());

        Ok(())
    }

    #[test]
    fn test_tensor_arithmetic() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

        // 基本算術演算
        let sum = a.add(&b)?;
        assert_eq!(sum.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

        let diff = a.subtract(&b)?;
        assert_eq!(diff.as_slice(), &[-4.0, -4.0, -4.0, -4.0]);

        let prod = a.multiply(&b)?;
        assert_eq!(prod.as_slice(), &[5.0, 12.0, 21.0, 32.0]);

        let quot = b.divide(&a)?;
        assert_eq!(quot.as_slice(), &[5.0, 3.0, 7.0/3.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_scalar_ops() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // スカラー乗算
        let scaled = tensor.multiply_scalar(2.0)?;
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0, 8.0]);

        // スカラー加算
        let added = tensor.add_scalar(10.0)?;
        assert_eq!(added.as_slice(), &[11.0, 12.0, 13.0, 14.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_reshape_ops() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

        // reshape
        let reshaped = tensor.try_reshape(&[3, 2])?;
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);

        // flatten
        let flattened = tensor.try_reshape(&[6])?;
        assert_eq!(flattened.shape(), &[6]);

        // transpose (2D only)
        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let transposed = matrix.try_transpose()?;
        assert_eq!(transposed.shape(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_tensor_slicing() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

        // 安全なアクセス
        assert_eq!(tensor.try_get(&[0, 0])?, 1.0);
        assert_eq!(tensor.try_get(&[1, 2])?, 6.0);

        // 範囲外アクセスはエラー
        assert!(tensor.try_get(&[2, 0]).is_err());
        assert!(tensor.try_get(&[0, 3]).is_err());

        Ok(())
    }

    #[test]
    fn test_tensor_statistics() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // 基本統計 (実装されている場合)
        let sum = tensor.sum()?;
        assert_eq!(sum, 10.0);

        let mean = tensor.mean()?;
        assert_eq!(mean, 2.5);

        let min = tensor.min()?;
        assert_eq!(min, 1.0);

        let max = tensor.max()?;
        assert_eq!(max, 4.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 無効な形状でのテンソル作成
        let result = F32Tensor::new(vec![1.0, 2.0], &[3, 3]); // データ不足
        assert!(result.is_err());

        // 形状不一致での演算
        let a = F32Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let b = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert!(a.try_add(&b).is_err());

        // ゼロ除算
        let zero_tensor = F32Tensor::from_vec(vec![0.0, 0.0], &[2])?;
        assert!(a.try_div(&zero_tensor).is_err());

        // 無効なreshape
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert!(tensor.try_reshape(&[3, 2]).is_err()); // 4要素 → 6要素

        Ok(())
    }

    #[test]
    fn test_tensor_cloning_and_device() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        let original = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let cloned = original.clone();

        // 同じデータ
        assert_eq!(original.shape(), cloned.shape());
        assert_eq!(original.as_slice(), cloned.as_slice());

        // デバイス状態
        assert!(matches!(original.device_state(), rustorch::hybrid_f32::tensor::core::DeviceState::CPU));

        // 勾配追跡
        let mut tensor = original;
        assert!(!tensor.is_grad_enabled());
        tensor.requires_grad(true);
        assert!(tensor.is_grad_enabled());

        Ok(())
    }
}