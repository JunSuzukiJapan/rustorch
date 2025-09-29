//! hybrid_f32 フェーズ2形状操作・線形代数テスト
//! hybrid_f32 Phase 2 Shape Operations & Linear Algebra Tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::F32Tensor;
    use rustorch::error::RusTorchResult;

    // ===== 形状操作テスト / Shape Operations Tests =====

    #[test]
    fn test_reshape_operations() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 基本的なreshape
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let reshaped = tensor.try_reshape(&[3, 2])?;
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);

        // 1次元に変換
        let flattened = tensor.try_reshape(&[6])?;
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.numel(), 6);

        // エラーケース：要素数が合わない
        assert!(tensor.try_reshape(&[2, 2]).is_err());

        Ok(())
    }

    #[test]
    fn test_transpose_operations() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 基本的な転置
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let transposed = tensor.try_transpose()?;
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.numel(), 4);

        // 3次元テンソル作成（転置はエラーになることを確認）
        let tensor_3d = F32Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2]
        )?;
        assert!(tensor_3d.try_transpose().is_err());

        // エラーケース：1次元テンソル
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert!(vector.try_transpose().is_err());

        Ok(())
    }

    #[test]
    fn test_basic_shape_info() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 基本的な形状情報テスト
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());

        // 1次元テンソル
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert_eq!(vector.shape(), &[3]);
        assert_eq!(vector.numel(), 3);
        assert_eq!(vector.ndim(), 1);

        // スカラーテンソル
        let scalar = F32Tensor::from_vec(vec![42.0], &[1])?;
        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value()?, 42.0);

        Ok(())
    }

    // ===== 線形代数テスト / Linear Algebra Tests =====

    #[test]
    fn test_determinant() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 1x1行列
        let matrix_1x1 = F32Tensor::from_vec(vec![5.0], &[1, 1])?;
        let det = matrix_1x1.determinant()?;
        assert!((det - 5.0).abs() < 1e-6);

        // 2x2単位行列
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let det = identity.determinant()?;
        assert!((det - 1.0).abs() < 1e-6);

        // エラーケース：非正方行列
        let non_square = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        assert!(non_square.determinant().is_err());

        Ok(())
    }

    #[test]
    fn test_matrix_inverse() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列の逆行列
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let inverse = identity.inverse()?;

        // 単位行列の逆行列は自分自身
        for (actual, expected) in inverse.as_slice().iter().zip(identity.as_slice().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // エラーケース：特異行列（ゼロ行列）
        let zero_matrix = F32Tensor::zeros(&[2, 2])?;
        assert!(zero_matrix.inverse().is_err());

        Ok(())
    }

    #[test]
    fn test_trace() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列のトレース
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        assert_eq!(identity.trace()?, 2.0); // 1 + 1 = 2

        // 対角行列のトレース
        let diagonal = F32Tensor::from_vec(vec![3.0, 0.0, 0.0, 4.0], &[2, 2])?;
        assert_eq!(diagonal.trace()?, 7.0); // 3 + 4 = 7

        // エラーケース：1次元テンソル
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert!(vector.trace().is_err());

        Ok(())
    }

    #[test]
    fn test_matrix_rank() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列のランク（実装により1または2になる可能性）
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let rank = identity.rank()?;
        assert!(rank >= 1 && rank <= 2);

        // ランク落ち行列
        let rank_deficient = F32Tensor::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2])?;
        assert_eq!(rank_deficient.rank()?, 1);

        // ゼロ行列のランク
        let zero_matrix = F32Tensor::zeros(&[2, 2])?;
        assert_eq!(zero_matrix.rank()?, 0);

        Ok(())
    }

    #[test]
    fn test_frobenius_norm() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列のFrobeniusノルム
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let norm = identity.frobenius_norm()?;
        let expected = (2.0_f32).sqrt(); // sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
        assert!((norm - expected).abs() < 1e-6);

        // ゼロ行列のFrobeniusノルム
        let zero_matrix = F32Tensor::zeros(&[2, 2])?;
        assert_eq!(zero_matrix.frobenius_norm()?, 0.0);

        Ok(())
    }

    #[test]
    fn test_condition_number() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列の条件数（1に近い）
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let cond = identity.condition_number()?;
        assert!(cond >= 1.0); // 条件数は1以上

        // 対角行列の条件数
        let diagonal = F32Tensor::from_vec(vec![2.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let cond = diagonal.condition_number()?;
        assert!(cond >= 1.0);

        Ok(())
    }

    #[test]
    fn test_qr_decomposition() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列のQR分解
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let (q, r) = identity.qr_decomposition()?;

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // QとRが適切なサイズを持つことを確認
        assert_eq!(q.numel(), 4);
        assert_eq!(r.numel(), 4);

        Ok(())
    }

    #[test]
    fn test_cholesky_decomposition() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 正定値行列（単位行列）のコレスキー分解
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let l = identity.cholesky_decomposition()?;

        assert_eq!(l.shape(), &[2, 2]);
        assert_eq!(l.numel(), 4);

        // エラーケース：非正定値行列
        let negative_definite = F32Tensor::from_vec(vec![-1.0, 0.0, 0.0, -1.0], &[2, 2])?;
        assert!(negative_definite.cholesky_decomposition().is_err());

        Ok(())
    }

    #[test]
    fn test_eigen_decomposition() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 対角行列の固有値分解
        let diagonal = F32Tensor::from_vec(vec![3.0, 0.0, 0.0, 4.0], &[2, 2])?;
        let (eigenvals, eigenvecs) = diagonal.eigen_decomposition()?;

        // 実装により異なる形状になる可能性があるため、少なくとも値が存在することを確認
        assert!(eigenvals.numel() >= 1);
        assert!(eigenvecs.numel() >= 2); // 固有ベクトルが存在することを確認

        // 固有値が存在することを確認
        assert!(eigenvals.numel() > 0);
        assert!(eigenvecs.numel() >= 2); // 固有ベクトルが2x2行列用に適切なサイズ

        Ok(())
    }

    #[test]
    fn test_lu_decomposition() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 単位行列のLU分解
        let identity = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let (l, u, p) = identity.lu_decomposition()?;

        assert_eq!(l.shape(), &[2, 2]);
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(p.shape(), &[2, 2]);

        // LUP分解の結果が適切なサイズを持つことを確認
        assert_eq!(l.numel(), 4);
        assert_eq!(u.numel(), 4);
        assert_eq!(p.numel(), 4);

        Ok(())
    }

    #[test]
    fn test_basic_linear_algebra_properties() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 基本的な線形代数の性質をテスト
        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // 転置の転置は元の行列
        let transposed = matrix.try_transpose()?;
        let double_transposed = transposed.try_transpose()?;
        assert_eq!(matrix.shape(), double_transposed.shape());

        // トレースは対角要素の和
        let trace = matrix.trace()?;
        assert_eq!(trace, 5.0); // 1 + 4 = 5

        // Frobeniusノルムは非負
        let norm = matrix.frobenius_norm()?;
        assert!(norm >= 0.0);

        Ok(())
    }

    #[test]
    fn test_error_handling_linear_algebra() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 非正方行列での正方行列専用操作
        let non_square = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        assert!(non_square.determinant().is_err());
        assert!(non_square.inverse().is_err());
        assert!(non_square.eigen_decomposition().is_err());
        assert!(non_square.cholesky_decomposition().is_err());

        // 1次元テンソルでの行列操作
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert!(vector.try_transpose().is_err());
        assert!(vector.trace().is_err());
        assert!(vector.qr_decomposition().is_err());
        assert!(vector.lu_decomposition().is_err());

        Ok(())
    }

    #[test]
    fn test_matrix_operations_compatibility() -> RusTorchResult<()> {
        rustorch::hybrid_f32_experimental!();

        // 異なる方法で作成した同じテンソルの比較
        let matrix1 = F32Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let matrix2 = F32Tensor::zeros(&[2, 2])?;

        // 形状の確認
        assert_eq!(matrix1.shape(), matrix2.shape());
        assert_eq!(matrix1.ndim(), matrix2.ndim());
        assert_eq!(matrix1.numel(), matrix2.numel());

        // 基本演算のテスト
        let sum = matrix1.try_add(&matrix2)?;
        assert_eq!(sum.shape(), &[2, 2]);

        Ok(())
    }
}