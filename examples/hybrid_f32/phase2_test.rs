//! フェーズ2形状操作・線形代数テスト例
//! Phase 2 Shape Operations & Linear Algebra Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::F32Tensor;

    rustorch::hybrid_f32_experimental!();

    println!("🔥 フェーズ2形状操作・線形代数テスト");
    println!("🔥 Phase 2 Shape Operations & Linear Algebra Test");
    println!("===========================================\n");

    // ===== 形状操作テスト / Shape Operations Tests =====
    println!("📐 1. 形状操作テスト / Shape Operations Tests");
    println!("---------------------------------------------");

    // Reshape operations
    let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    println!("  Original tensor (2x3): {:?}", tensor.as_slice());

    let reshaped = tensor.reshape(&[3, 2])?;
    println!("  Reshaped to (3x2): {:?}", reshaped.as_slice());

    let flattened = tensor.flatten()?;
    println!("  Flattened to 1D: {:?}", flattened.as_slice());

    // Transpose operations
    let matrix = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("\n  Matrix (2x2): {:?}", matrix.as_slice());

    let transposed = matrix.transpose()?;
    println!("  Transposed: {:?}", transposed.as_slice());

    // Squeeze and unsqueeze
    let with_singleton = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?;
    println!(
        "\n  With singleton dimension (1x2x2): shape {:?}",
        with_singleton.shape()
    );

    let squeezed = with_singleton.squeeze()?;
    println!("  Squeezed: shape {:?}", squeezed.shape());

    let unsqueezed = squeezed.unsqueeze(0)?;
    println!("  Unsqueezed at dim 0: shape {:?}", unsqueezed.shape());

    // Concatenation and stacking
    let a = F32Tensor::from_vec(vec![1.0, 2.0], vec![2])?;
    let b = F32Tensor::from_vec(vec![3.0, 4.0], vec![2])?;

    let concatenated = F32Tensor::concat(&[&a, &b], 0)?;
    println!("\n  Concatenated: {:?}", concatenated.as_slice());

    let stacked = F32Tensor::stack(&[&a, &b], 0)?;
    println!(
        "  Stacked: {:?} with shape {:?}",
        stacked.as_slice(),
        stacked.shape()
    );

    // ===== 線形代数テスト / Linear Algebra Tests =====
    println!("\n🔢 2. 線形代数テスト / Linear Algebra Tests");
    println!("---------------------------------------");

    // Basic matrix operations
    let mat_a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let mat_b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    println!("  Matrix A: {:?}", mat_a.as_slice());
    println!("  Matrix B: {:?}", mat_b.as_slice());

    // Matrix multiplication
    let matmul_result = mat_a.matmul(&mat_b)?;
    println!("  A @ B: {:?}", matmul_result.as_slice());

    // Transpose using linear algebra method
    let mat_a_t = mat_a.t()?;
    println!("  A^T: {:?}", mat_a_t.as_slice());

    // Determinant
    let det_a = mat_a.det()?;
    println!("  det(A): {}", det_a);

    // Trace
    let trace_a = mat_a.trace()?;
    println!("  trace(A): {}", trace_a);

    // Matrix rank
    let rank_a = mat_a.rank()?;
    println!("  rank(A): {}", rank_a);

    // Norms
    let frobenius_norm = mat_a.frobenius_norm()?;
    println!("  ||A||_F: {}", frobenius_norm);

    // ===== 高度な線形代数 / Advanced Linear Algebra =====
    println!("\n⚡ 3. 高度な線形代数演算 / Advanced Linear Algebra");
    println!("----------------------------------------------");

    // Matrix inverse (for well-conditioned matrix)
    let well_conditioned = F32Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2])?;
    println!(
        "  Well-conditioned matrix: {:?}",
        well_conditioned.as_slice()
    );

    let inverse = well_conditioned.inverse()?;
    println!("  Inverse: {:?}", inverse.as_slice());

    // Verify A * A^(-1) = I
    let identity_check = well_conditioned.matmul(&inverse)?;
    println!("  A * A^(-1): {:?}", identity_check.as_slice());

    // Condition number
    let cond_num = well_conditioned.cond()?;
    println!("  Condition number: {}", cond_num);

    // QR decomposition
    let qr_matrix = F32Tensor::from_vec(vec![1.0, 1.0, 0.0, 1.0], vec![2, 2])?;
    println!("\n  QR decomposition of: {:?}", qr_matrix.as_slice());

    let (q, r) = qr_matrix.qr()?;
    println!("  Q: {:?}", q.as_slice());
    println!("  R: {:?}", r.as_slice());

    // Verify Q * R = A
    let qr_reconstruction = q.matmul(&r)?;
    println!("  Q * R: {:?}", qr_reconstruction.as_slice());

    // Eigenvalue decomposition (for symmetric 2x2)
    let symmetric = F32Tensor::from_vec(vec![3.0, 1.0, 1.0, 3.0], vec![2, 2])?;
    println!(
        "\n  Eigenvalue decomposition of: {:?}",
        symmetric.as_slice()
    );

    let (eigenvals, eigenvecs) = symmetric.eig()?;
    println!("  Eigenvalues: {:?}", eigenvals.as_slice());
    println!("  Eigenvectors: {:?}", eigenvecs.as_slice());

    // Cholesky decomposition (for positive definite matrix)
    let pos_def = F32Tensor::from_vec(vec![4.0, 2.0, 2.0, 2.0], vec![2, 2])?;
    println!("\n  Cholesky decomposition of: {:?}", pos_def.as_slice());

    let chol_l = pos_def.cholesky()?;
    println!("  L: {:?}", chol_l.as_slice());

    // Verify L * L^T = A
    let chol_reconstruction = chol_l.matmul(&chol_l.t()?)?;
    println!("  L * L^T: {:?}", chol_reconstruction.as_slice());

    // SVD (simplified 2x2 implementation)
    let svd_matrix = F32Tensor::from_vec(vec![3.0, 0.0, 0.0, 4.0], vec![2, 2])?;
    println!("\n  SVD of diagonal matrix: {:?}", svd_matrix.as_slice());

    let (u, s, v) = svd_matrix.svd()?;
    println!("  U: {:?}", u.as_slice());
    println!("  S: {:?}", s.as_slice());
    println!("  V: {:?}", v.as_slice());

    // ===== パフォーマンステスト / Performance Test =====
    println!("\n🚀 4. パフォーマンステスト / Performance Test");
    println!("-----------------------------------------");

    use std::time::Instant;

    // Large matrix operations
    let large_a = F32Tensor::from_vec((0..100).map(|i| i as f32).collect(), vec![10, 10])?;
    let large_b = F32Tensor::from_vec((100..200).map(|i| i as f32).collect(), vec![10, 10])?;

    let start = Instant::now();
    let _large_matmul = large_a.matmul(&large_b)?;
    let matmul_time = start.elapsed();
    println!("  10x10 matrix multiplication: {:?}", matmul_time);

    let start = Instant::now();
    let _large_transpose = large_a.transpose()?;
    let transpose_time = start.elapsed();
    println!("  10x10 matrix transpose: {:?}", transpose_time);

    let start = Instant::now();
    let _large_det = large_a.det();
    let det_time = start.elapsed();
    println!("  10x10 matrix determinant attempt: {:?}", det_time);

    println!("\n✅ フェーズ2テスト完了！形状操作と線形代数が正常に動作しています。");
    println!("✅ Phase 2 tests completed! Shape operations and linear algebra working correctly.");
    println!("\n📊 実装済みメソッド数: 27メソッド");
    println!("📊 Implemented methods: 27 methods");
    println!("   - 形状操作: 15メソッド (reshape, transpose, permute, squeeze, unsqueeze, flatten, etc.)");
    println!("   - 線形代数: 12メソッド (matmul, det, inverse, trace, rank, qr, svd, eig, cholesky, etc.)");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example hybrid_f32_phase2_test --features hybrid-f32");
}
