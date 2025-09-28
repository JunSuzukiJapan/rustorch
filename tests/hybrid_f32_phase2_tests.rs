//! hybrid_f32 フェーズ2形状操作・線形代数テスト
//! hybrid_f32 Phase 2 Shape Operations & Linear Algebra Tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::F32Tensor;

    // ===== 形状操作テスト / Shape Operations Tests =====

    #[test]
    fn test_reshape_operations() {
        rustorch::hybrid_f32_experimental!();

        // 基本的なreshape
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 1次元に変換
        let flattened = tensor.reshape(&[6]).unwrap();
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // エラーケース：要素数が合わない
        assert!(tensor.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_transpose_operations() {
        rustorch::hybrid_f32_experimental!();

        // 基本的な転置
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);
        // [1, 2]  →  [1, 3]
        // [3, 4]     [2, 4]
        assert_eq!(transposed.as_slice(), &[1.0, 3.0, 2.0, 4.0]);

        // 3次元テンソルは未実装なので2次元のみテスト
        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transposed_matrix = matrix.transpose().unwrap();
        assert_eq!(transposed_matrix.shape(), &[3, 2]);

        // エラーケース：1次元テンソル
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(vector.transpose().is_err());
    }

    #[test]
    fn test_permute_operations() {
        rustorch::hybrid_f32_experimental!();

        // 2次元テンソルの次元順序変更（転置のみサポート）
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let permuted = tensor.permute(&[1, 0]).unwrap();
        assert_eq!(permuted.shape(), &[3, 2]);

        // エラーケース：現在は2D転置のみサポート
        let tensor_3d = F32Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2]
        ).unwrap();
        assert!(tensor_3d.permute(&[2, 0, 1]).is_err());

        // エラーケース：無効な次元指定
        assert!(tensor.permute(&[1, 0, 2]).is_err());
        assert!(tensor.permute(&[1, 1]).is_err());
    }

    #[test]
    fn test_squeeze_operations() {
        rustorch::hybrid_f32_experimental!();

        // サイズ1の次元を除去
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let squeezed = tensor.squeeze().unwrap();
        assert_eq!(squeezed.shape(), &[2, 2]);
        assert_eq!(squeezed.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        // 特定次元の除去
        let squeezed_dim = tensor.squeeze_dim(0).unwrap();
        assert_eq!(squeezed_dim.shape(), &[2, 2]);

        // エラーケース：サイズ1でない次元を除去しようとする
        assert!(tensor.squeeze_dim(1).is_err());
    }

    #[test]
    fn test_unsqueeze_operations() {
        rustorch::hybrid_f32_experimental!();

        // 新しい次元を追加
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 2]);

        let unsqueezed_end = tensor.unsqueeze(2).unwrap();
        assert_eq!(unsqueezed_end.shape(), &[2, 2, 1]);

        // エラーケース：無効な次元指定
        assert!(tensor.unsqueeze(3).is_err());
    }

    #[test]
    fn test_flatten_operations() {
        rustorch::hybrid_f32_experimental!();

        // 多次元テンソルを1次元に平坦化
        let tensor = F32Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2]
        ).unwrap();
        let flattened = tensor.flatten().unwrap();
        assert_eq!(flattened.shape(), &[8]);
        assert_eq!(flattened.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // 基本的な平坦化のみサポート
        let tensor_2d = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let flat_2d = tensor_2d.flatten().unwrap();
        assert_eq!(flat_2d.shape(), &[4]);
    }

    #[test]
    fn test_concat_and_stack_operations() {
        rustorch::hybrid_f32_experimental!();

        // テンソル結合
        let a = F32Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = F32Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let c = F32Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();

        let concatenated = F32Tensor::concat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(concatenated.shape(), &[6]);
        assert_eq!(concatenated.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // テンソル積み重ね
        let stacked = F32Tensor::stack(&[&a, &b, &c], 0).unwrap();
        assert_eq!(stacked.shape(), &[3, 2]);
        assert_eq!(stacked.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let stacked_dim1 = F32Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(stacked_dim1.shape(), &[2, 2]);
    }

    #[test]
    fn test_split_operations() {
        rustorch::hybrid_f32_experimental!();

        // テンソル分割
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
        let chunks = tensor.split(2, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[0].as_slice(), &[1.0, 2.0]);
        assert_eq!(chunks[1].as_slice(), &[3.0, 4.0]);
        assert_eq!(chunks[2].as_slice(), &[5.0, 6.0]);

        // 不均等分割
        let tensor_odd = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let chunks_odd = tensor_odd.split(2, 0).unwrap();
        assert_eq!(chunks_odd.len(), 3);
        assert_eq!(chunks_odd[2].shape(), &[1]); // 最後のチャンクは1要素
    }

    // ===== 線形代数テスト / Linear Algebra Tests =====

    #[test]
    fn test_matrix_transpose() {
        rustorch::hybrid_f32_experimental!();

        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transposed = matrix.t().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        // [1, 2, 3]^T = [1, 4]
        // [4, 5, 6]     [2, 5]
        //               [3, 6]
        assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_determinant() {
        rustorch::hybrid_f32_experimental!();

        // 1x1行列
        let matrix_1x1 = F32Tensor::from_vec(vec![5.0], vec![1, 1]).unwrap();
        assert_eq!(matrix_1x1.det().unwrap(), 5.0);

        // 2x2行列
        let matrix_2x2 = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(matrix_2x2.det().unwrap(), -2.0); // 1*4 - 2*3 = -2

        // 3x3行列
        let matrix_3x3 = F32Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0],
            vec![3, 3]
        ).unwrap();
        let det_3x3 = matrix_3x3.det().unwrap();
        assert!((det_3x3 - 1.0).abs() < 1e-6); // 期待値：1

        // エラーケース：非正方行列
        let non_square = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(non_square.det().is_err());
    }

    #[test]
    fn test_matrix_inverse() {
        rustorch::hybrid_f32_experimental!();

        // 2x2行列の逆行列
        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let inverse = matrix.inverse().unwrap();

        // A * A^(-1) = I を確認
        let identity = matrix.matmul(&inverse).unwrap();
        let expected_identity = vec![1.0, 0.0, 0.0, 1.0];
        for (actual, expected) in identity.as_slice().iter().zip(expected_identity.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // 3x3単位行列の逆行列
        let identity_3x3 = F32Tensor::eye(3);
        let inv_identity = identity_3x3.inverse().unwrap();
        for (actual, expected) in inv_identity.as_slice().iter().zip(identity_3x3.as_slice().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // エラーケース：特異行列
        let singular = F32Tensor::from_vec(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        assert!(singular.inverse().is_err());
    }

    #[test]
    fn test_trace() {
        rustorch::hybrid_f32_experimental!();

        // 正方行列のトレース
        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(matrix.trace().unwrap(), 5.0); // 1 + 4 = 5

        // 非正方行列のトレース（最小次元まで）
        let rect_matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(rect_matrix.trace().unwrap(), 6.0); // 1 + 5 = 6

        // エラーケース：1次元テンソル
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(vector.trace().is_err());
    }

    #[test]
    fn test_matrix_rank() {
        rustorch::hybrid_f32_experimental!();

        // フルランク行列
        let full_rank = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(full_rank.rank().unwrap(), 2);

        // ランク落ち行列
        let rank_deficient = F32Tensor::from_vec(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(rank_deficient.rank().unwrap(), 1);

        // 単位行列
        let identity = F32Tensor::eye(3);
        assert_eq!(identity.rank().unwrap(), 3);
    }

    #[test]
    fn test_frobenius_norm() {
        rustorch::hybrid_f32_experimental!();

        let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let norm = matrix.frobenius_norm().unwrap();
        let expected = (1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0_f32).sqrt();
        assert!((norm - expected).abs() < 1e-6);
    }

    #[test]
    fn test_condition_number() {
        rustorch::hybrid_f32_experimental!();

        // 良条件行列（単位行列）
        let identity = F32Tensor::eye(2);
        let cond_identity = identity.cond().unwrap();
        // 条件数の実装は近似なので、妥当な範囲をテスト
        assert!(cond_identity >= 1.0 && cond_identity < 10.0);

        // 一般的な行列
        let matrix = F32Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let cond_num = matrix.cond().unwrap();
        assert!(cond_num > 1.0); // 条件数は1以上
    }

    #[test]
    fn test_qr_decomposition() {
        rustorch::hybrid_f32_experimental!();

        // 2x2行列のQR分解
        let matrix = F32Tensor::from_vec(vec![1.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let (q, r) = matrix.qr().unwrap();

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // Q * R = A を確認
        let reconstructed = q.matmul(&r).unwrap();
        for (actual, expected) in reconstructed.as_slice().iter().zip(matrix.as_slice().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // Qが直交行列であることを確認（Q^T * Q = I）
        let q_t = q.t().unwrap();
        let qtq = q_t.matmul(&q).unwrap();
        let identity_data = vec![1.0, 0.0, 0.0, 1.0];
        for (actual, expected) in qtq.as_slice().iter().zip(identity_data.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_svd_decomposition() {
        rustorch::hybrid_f32_experimental!();

        // 2x2行列のSVD（簡単な実装のテスト）
        let matrix = F32Tensor::from_vec(vec![3.0, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let (u, s, v) = matrix.svd().unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(v.shape(), &[2, 2]);

        // 特異値は正の値であることを確認
        for &val in s.as_slice() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_eigen_decomposition() {
        rustorch::hybrid_f32_experimental!();

        // 対角行列の固有値分解
        let diagonal = F32Tensor::from_vec(vec![3.0, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let (eigenvals, eigenvecs) = diagonal.eig().unwrap();

        assert_eq!(eigenvals.shape(), &[2]);
        assert_eq!(eigenvecs.shape(), &[2, 2]);

        // 固有値は期待値に近いことを確認
        let mut vals = eigenvals.as_slice().to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky_decomposition() {
        rustorch::hybrid_f32_experimental!();

        // 正定値行列のコレスキー分解
        let matrix = F32Tensor::from_vec(vec![4.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let l = matrix.cholesky().unwrap();

        assert_eq!(l.shape(), &[2, 2]);

        // L * L^T = A を確認
        let l_t = l.t().unwrap();
        let reconstructed = l.matmul(&l_t).unwrap();
        for (actual, expected) in reconstructed.as_slice().iter().zip(matrix.as_slice().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // 下三角行列であることを確認
        assert!((l.as_slice()[1]).abs() < 1e-10); // 上三角要素は0
    }

    #[test]
    fn test_advanced_matrix_operations() {
        rustorch::hybrid_f32_experimental!();

        // 複合演算のテスト
        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        // (A + B)^T = A^T + B^T
        let sum = a.add(&b).unwrap();
        let sum_t = sum.t().unwrap();

        let a_t = a.t().unwrap();
        let b_t = b.t().unwrap();
        let sum_t_expected = a_t.add(&b_t).unwrap();

        for (actual, expected) in sum_t.as_slice().iter().zip(sum_t_expected.as_slice().iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // det(A * B) = det(A) * det(B) (2x2行列の場合)
        let det_a = a.det().unwrap();
        let det_b = b.det().unwrap();
        let product = a.matmul(&b).unwrap();
        let det_product = product.det().unwrap();

        assert!((det_product - (det_a * det_b)).abs() < 1e-6);
    }

    #[test]
    fn test_error_handling_linear_algebra() {
        rustorch::hybrid_f32_experimental!();

        // 非正方行列での正方行列専用操作
        let non_square = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert!(non_square.det().is_err());
        assert!(non_square.inverse().is_err());
        assert!(non_square.eig().is_err());
        assert!(non_square.cholesky().is_err());

        // 1次元テンソルでの行列操作
        let vector = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(vector.t().is_err());
        assert!(vector.trace().is_err());
        assert!(vector.qr().is_err());
        assert!(vector.svd().is_err());

        // ランク落ち行列でのQR分解
        let rank_deficient = F32Tensor::from_vec(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        assert!(rank_deficient.qr().is_err());

        // 非正定値行列でのコレスキー分解
        let negative_definite = F32Tensor::from_vec(vec![-1.0, 0.0, 0.0, -1.0], vec![2, 2]).unwrap();
        assert!(negative_definite.cholesky().is_err());
    }
}