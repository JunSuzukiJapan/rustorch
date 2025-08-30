//! Advanced linear algebra operations for tensors
//! テンソル用高度線形代数演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;

impl<
        T: Float
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive
            + Clone
            + std::fmt::Debug,
    > Tensor<T>
{
    // Norm operations
    // ノルム演算

    /// Calculate L2 (Euclidean) norm
    /// L2（ユークリッド）ノルムを計算
    pub fn norm(&self) -> T {
        let sum_squares: T = self
            .data
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x);
        sum_squares.sqrt()
    }

    /// Calculate Lp norm with specified p
    /// 指定されたpでLpノルムを計算
    pub fn norm_p(&self, p: T) -> T {
        if p == T::from(2.0).unwrap() {
            return self.norm();
        }

        if p == T::one() {
            // L1 norm (Manhattan norm)
            return self
                .data
                .iter()
                .map(|&x| x.abs())
                .fold(T::zero(), |acc, x| acc + x);
        }

        if p == T::infinity() {
            // Infinity norm (maximum absolute value)
            return self
                .data
                .iter()
                .map(|&x| x.abs())
                .fold(T::zero(), |acc, x| if x > acc { x } else { acc });
        }

        // General Lp norm
        let sum_powers: T = self
            .data
            .iter()
            .map(|&x| x.abs().powf(p))
            .fold(T::zero(), |acc, x| acc + x);

        sum_powers.powf(T::one() / p)
    }

    /// Calculate Frobenius norm for matrices
    /// 行列用フロベニウスノルムを計算
    pub fn frobenius_norm(&self) -> T {
        self.norm()
    }

    /// Calculate nuclear norm (sum of singular values)
    /// 核ノルム（特異値の和）を計算
    pub fn nuclear_norm(&self) -> RusTorchResult<T> {
        if self.shape().len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "nuclear_norm".to_string(),
                message: "Nuclear norm is only defined for 2D matrices".to_string(),
            });
        }

        let (_, s, _) = self.svd()?;
        Ok(s.data.iter().fold(T::zero(), |acc, &x| acc + x))
    }

    // Matrix decompositions
    // 行列分解

    /// Singular Value Decomposition (SVD)
    /// 特異値分解
    pub fn svd(&self) -> RusTorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
        if self.shape().len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "svd".to_string(),
                message: "SVD is only defined for 2D matrices".to_string(),
            });
        }

        let [m, n] = [self.shape()[0], self.shape()[1]];
        let min_dim = std::cmp::min(m, n);

        // For simplicity, we'll implement a basic power iteration method
        // In a production system, you'd use LAPACK or similar
        let mut a = self.clone();

        // Simple SVD approximation using power iteration
        // This is a simplified implementation for demonstration
        let mut u_vecs = Vec::new();
        let mut singular_values = Vec::new();

        for _ in 0..min_dim {
            // Power iteration to find dominant singular vector
            let (u, s, v) = self.power_iteration_svd(&a)?;

            singular_values.push(s);
            u_vecs.push(u);

            // Deflate the matrix
            let outer_product = self.outer_product(&u_vecs.last().unwrap(), &v)?;
            a = a.sub(&outer_product.mul_scalar(s))?;

            // Early stopping if singular value is very small
            if s < T::from(1e-10).unwrap() {
                break;
            }
        }

        // Construct U, S, V matrices
        let u = self.construct_orthogonal_matrix(&u_vecs, m, u_vecs.len())?;
        let s = Tensor::from_vec(singular_values, vec![u_vecs.len()]);

        // For simplicity, V is computed as identity for now
        // In a full implementation, you'd compute the right singular vectors
        let mut v_data = vec![T::zero(); n * n];
        for i in 0..n {
            v_data[i * n + i] = T::one();
        }
        let v = Tensor::from_vec(v_data, vec![n, n]);

        Ok((u, s, v))
    }

    fn power_iteration_svd(&self, matrix: &Tensor<T>) -> RusTorchResult<(Tensor<T>, T, Tensor<T>)> {
        let [m, n] = [matrix.shape()[0], matrix.shape()[1]];

        // Initialize random vector
        let v: Vec<T> = (0..n)
            .map(|i| T::from(i as f64 % 7.0 + 1.0).unwrap())
            .collect();
        let mut v_tensor = Tensor::from_vec(v, vec![n]);
        v_tensor = v_tensor.normalize()?;

        // Power iteration
        for _ in 0..100 {
            // Max iterations
            // u = A * v
            let u_tensor = matrix.matmul(&v_tensor.unsqueeze(1)?)?.squeeze();
            let u_norm = u_tensor.norm();
            let u_normalized = u_tensor.div_scalar(u_norm);

            // v = A^T * u
            let at = matrix.transpose()?;
            let v_new = at.matmul(&u_normalized.unsqueeze(1)?)?.squeeze();
            let v_norm = v_new.norm();
            let v_normalized = v_new.div_scalar(v_norm);

            // Check convergence
            let diff = v_normalized.sub(&v_tensor)?.norm();
            if diff < T::from(1e-8).unwrap() {
                break;
            }

            v_tensor = v_normalized;
        }

        // Compute singular value
        let av = matrix.matmul(&v_tensor.unsqueeze(1)?)?.squeeze();
        let sigma = av.norm();
        let u = av.div_scalar(sigma);

        Ok((u, sigma, v_tensor))
    }

    fn construct_orthogonal_matrix(
        &self,
        vectors: &[Tensor<T>],
        rows: usize,
        cols: usize,
    ) -> RusTorchResult<Tensor<T>> {
        let mut data = vec![T::zero(); rows * cols];

        for (col, vec) in vectors.iter().enumerate() {
            if col >= cols {
                break;
            }
            for (row, &val) in vec.data.iter().enumerate() {
                if row >= rows {
                    break;
                }
                data[row * cols + col] = val;
            }
        }

        Ok(Tensor::from_vec(data, vec![rows, cols]))
    }

    /// Eigenvalue decomposition for symmetric matrices
    /// 対称行列用固有値分解
    pub fn eigh(&self) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if self.shape().len() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "eigh".to_string(),
                message: "Eigenvalue decomposition requires square matrices".to_string(),
            });
        }

        // Check if matrix is symmetric
        let transpose = self.transpose()?;
        let diff = self.sub(&transpose)?.norm();
        if diff > T::from(1e-10).unwrap() {
            return Err(RusTorchError::InvalidOperation {
                operation: "eigh".to_string(),
                message: "Matrix must be symmetric for eigh".to_string(),
            });
        }

        let n = self.shape()[0];

        // Simple power iteration for largest eigenvalue/eigenvector
        // In practice, you'd use QR algorithm or similar
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut a = self.clone();

        for _ in 0..std::cmp::min(n, 3) {
            // Compute first few eigenvalues
            let (eigval, eigvec) = self.power_iteration_eigen(&a)?;
            eigenvalues.push(eigval);
            eigenvectors.push(eigvec);

            // Deflate matrix
            let outer =
                self.outer_product(&eigenvectors.last().unwrap(), &eigenvectors.last().unwrap())?;
            a = a.sub(&outer.mul_scalar(eigval))?;
        }

        let eigenvalue_len = eigenvalues.len();
        let eigenvalue_tensor = Tensor::from_vec(eigenvalues, vec![eigenvalue_len]);
        let eigenvector_matrix =
            self.construct_orthogonal_matrix(&eigenvectors, n, eigenvectors.len())?;

        Ok((eigenvalue_tensor, eigenvector_matrix))
    }

    fn power_iteration_eigen(&self, matrix: &Tensor<T>) -> RusTorchResult<(T, Tensor<T>)> {
        let n = matrix.shape()[0];

        // Initialize random vector
        let v: Vec<T> = (0..n)
            .map(|i| T::from((i * 3 + 1) as f64).unwrap())
            .collect();
        let mut v_tensor = Tensor::from_vec(v, vec![n]);
        v_tensor = v_tensor.normalize()?;

        let mut eigenvalue = T::zero();

        // Power iteration
        for _ in 0..100 {
            // v_new = A * v
            let v_new = matrix.matmul(&v_tensor.unsqueeze(1)?)?.squeeze();

            // Compute eigenvalue approximation (Rayleigh quotient)
            let vt_av = v_tensor.dot(&v_new);
            let vt_v = v_tensor.dot(&v_tensor);
            let new_eigenvalue = vt_av / vt_v;

            // Normalize
            v_tensor = v_new.normalize()?;

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < T::from(1e-10).unwrap() {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        Ok((eigenvalue, v_tensor))
    }

    /// QR decomposition
    /// QR分解
    pub fn qr(&self) -> RusTorchResult<(Tensor<T>, Tensor<T>)> {
        if self.shape().len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "qr".to_string(),
                message: "QR decomposition is only defined for 2D matrices".to_string(),
            });
        }

        let [m, n] = [self.shape()[0], self.shape()[1]];

        // Gram-Schmidt process
        let mut q_vectors: Vec<Tensor<T>> = Vec::new();
        let mut r_data = vec![T::zero(); n * n];

        for j in 0..n {
            // Extract j-th column
            let mut col_data = Vec::new();
            for i in 0..m {
                if let Some(val) = self.data.get(IxDyn(&[i * n + j])) {
                    col_data.push(*val);
                } else {
                    col_data.push(T::zero());
                }
            }
            let mut q_j = Tensor::from_vec(col_data, vec![m]);

            // Orthogonalize against previous vectors
            for (k, q_k) in q_vectors.iter().enumerate() {
                let r_kj = q_k.dot(&q_j);
                r_data[k * n + j] = r_kj;
                let proj = q_k.mul_scalar(r_kj);
                q_j = q_j.sub(&proj)?;
            }

            // Normalize
            let r_jj = q_j.norm();
            r_data[j * n + j] = r_jj;

            if r_jj > T::from(1e-10).unwrap() {
                q_j = q_j.div_scalar(r_jj);
            }

            q_vectors.push(q_j);
        }

        // Construct Q and R matrices
        let q = self.construct_orthogonal_matrix(&q_vectors, m, n)?;
        let r = Tensor::from_vec(r_data, vec![n, n]);

        Ok((q, r))
    }

    /// Cholesky decomposition for positive definite matrices
    /// 正定値行列用コレスキー分解
    pub fn cholesky(&self) -> RusTorchResult<Tensor<T>> {
        if self.shape().len() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "cholesky".to_string(),
                message: "Cholesky decomposition requires square matrices".to_string(),
            });
        }

        let n = self.shape()[0];
        let mut l_data = vec![T::zero(); n * n];

        // Cholesky-Banachiewicz algorithm
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        let l_jk = l_data[j * n + k];
                        sum = sum + l_jk * l_jk;
                    }

                    let a_jj = self
                        .data
                        .get(IxDyn(&[j * n + j]))
                        .copied()
                        .unwrap_or(T::zero());
                    let l_jj_squared = a_jj - sum;

                    if l_jj_squared <= T::zero() {
                        return Err(RusTorchError::InvalidOperation {
                            operation: "cholesky".to_string(),
                            message: "Matrix is not positive definite".to_string(),
                        });
                    }

                    l_data[j * n + j] = l_jj_squared.sqrt();
                } else {
                    // Off-diagonal elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        let l_ik = l_data[i * n + k];
                        let l_jk = l_data[j * n + k];
                        sum = sum + l_ik * l_jk;
                    }

                    let a_ij = self
                        .data
                        .get(IxDyn(&[i * n + j]))
                        .copied()
                        .unwrap_or(T::zero());
                    let l_jj = l_data[j * n + j];

                    if l_jj == T::zero() {
                        return Err(RusTorchError::InvalidOperation {
                            operation: "cholesky".to_string(),
                            message: "Division by zero in Cholesky decomposition".to_string(),
                        });
                    }

                    l_data[i * n + j] = (a_ij - sum) / l_jj;
                }
            }
        }

        Ok(Tensor::from_vec(l_data, vec![n, n]))
    }

    // Matrix inverse and pseudo-inverse
    // 逆行列と疑似逆行列

    /// Compute matrix inverse using LU decomposition
    /// LU分解を使用した逆行列の計算
    pub fn inverse(&self) -> RusTorchResult<Tensor<T>> {
        if self.shape().len() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "inverse".to_string(),
                message: "Matrix inverse requires square matrices".to_string(),
            });
        }

        let n = self.shape()[0];

        // Check if matrix is singular by computing determinant
        let det = self.det()?;
        if det.abs() < T::from(1e-12).unwrap() {
            return Err(RusTorchError::InvalidOperation {
                operation: "inverse".to_string(),
                message: "Matrix is singular and cannot be inverted".to_string(),
            });
        }

        // Use Gauss-Jordan elimination
        let mut augmented = vec![T::zero(); n * 2 * n];

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                let val = self
                    .data
                    .get(IxDyn(&[i * n + j]))
                    .copied()
                    .unwrap_or(T::zero());
                augmented[i * 2 * n + j] = val;
            }
            // Identity matrix part
            augmented[i * 2 * n + n + i] = T::one();
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[k * 2 * n + i].abs() > augmented[max_row * 2 * n + i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[i * 2 * n + j];
                    augmented[i * 2 * n + j] = augmented[max_row * 2 * n + j];
                    augmented[max_row * 2 * n + j] = temp;
                }
            }

            // Scale pivot row
            let pivot = augmented[i * 2 * n + i];
            if pivot.abs() < T::from(1e-12).unwrap() {
                return Err(RusTorchError::InvalidOperation {
                    operation: "inverse".to_string(),
                    message: "Matrix is singular".to_string(),
                });
            }

            for j in 0..(2 * n) {
                augmented[i * 2 * n + j] = augmented[i * 2 * n + j] / pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[k * 2 * n + i];
                    for j in 0..(2 * n) {
                        augmented[k * 2 * n + j] =
                            augmented[k * 2 * n + j] - factor * augmented[i * 2 * n + j];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse_data = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                inverse_data[i * n + j] = augmented[i * 2 * n + n + j];
            }
        }

        Ok(Tensor::from_vec(inverse_data, vec![n, n]))
    }

    /// Compute Moore-Penrose pseudo-inverse
    /// Moore-Penrose疑似逆行列の計算
    pub fn pinv(&self) -> RusTorchResult<Tensor<T>> {
        if self.shape().len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "pinv".to_string(),
                message: "Pseudo-inverse is only defined for 2D matrices".to_string(),
            });
        }

        let [m, n] = [self.shape()[0], self.shape()[1]];

        if m >= n {
            // A^+ = (A^T A)^(-1) A^T
            let at = self.transpose()?;
            let ata = at.matmul(self)?;
            let ata_inv = ata.inverse()?;
            ata_inv.matmul(&at)
        } else {
            // A^+ = A^T (A A^T)^(-1)
            let at = self.transpose()?;
            let aat = self.matmul(&at)?;
            let aat_inv = aat.inverse()?;
            at.matmul(&aat_inv)
        }
    }

    // Utility functions
    // ユーティリティ関数

    fn normalize(&self) -> RusTorchResult<Self> {
        let norm = self.norm();
        if norm == T::zero() {
            Ok(self.clone())
        } else {
            Ok(self.div_scalar(norm))
        }
    }

    fn dot(&self, other: &Self) -> T {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
    }

    fn outer_product(&self, u: &Self, v: &Self) -> RusTorchResult<Self> {
        if u.numel() != self.shape()[0] || v.numel() != self.shape()[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "outer_product".to_string(),
                message: "Vector dimensions don't match matrix dimensions".to_string(),
            });
        }

        let m = u.numel();
        let n = v.numel();
        let mut result = vec![T::zero(); m * n];

        for i in 0..m {
            for j in 0..n {
                let u_val = u.data.get(IxDyn(&[i])).copied().unwrap_or(T::zero());
                let v_val = v.data.get(IxDyn(&[j])).copied().unwrap_or(T::zero());
                result[i * n + j] = u_val * v_val;
            }
        }

        Ok(Tensor::from_vec(result, vec![m, n]))
    }

    // Helper functions with different signatures to avoid conflicts
    // 競合を避けるための異なるシグネチャのヘルパー関数

    fn linalg_mul_scalar(&self, scalar: T) -> RusTorchResult<Self> {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    fn linalg_div_scalar(&self, scalar: T) -> RusTorchResult<Self> {
        if scalar == T::zero() {
            return Err(RusTorchError::InvalidOperation {
                operation: "div_scalar".to_string(),
                message: "Division by zero".to_string(),
            });
        }
        let result_data: Vec<T> = self.data.iter().map(|&x| x / scalar).collect();
        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norms() {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let l2_norm = tensor.norm();
        let l1_norm = tensor.norm_p(1.0);
        let frobenius_norm = tensor.frobenius_norm();

        assert!((l2_norm - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
        assert!((l1_norm - 7.0).abs() < 1e-10); // |3| + |4| = 7
        assert!((frobenius_norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix = Tensor::from_vec(vec![1.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let (q, r) = matrix.qr().unwrap();

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // Q should be orthogonal (Q^T * Q = I)
        let qt = q.transpose().unwrap();
        let qtq = qt.matmul(&q).unwrap();

        // Check if result is close to identity
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let diff = qtq.sub(&identity).unwrap().norm();
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Create a positive definite matrix: A = L*L^T where L = [[2, 0], [1, 1]]
        let matrix = Tensor::from_vec(vec![4.0, 2.0, 2.0, 2.0], vec![2, 2]); // [[4, 2], [2, 2]]

        let l = matrix.cholesky().unwrap();
        assert_eq!(l.shape(), &[2, 2]);

        // Verify that L * L^T = A
        let lt = l.transpose().unwrap();
        let reconstructed = l.matmul(&lt).unwrap();
        let diff = matrix.sub(&reconstructed).unwrap().norm();
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix = Tensor::from_vec(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]);
        let inverse = matrix.inverse().unwrap();

        // Check A * A^(-1) = I
        let product = matrix.matmul(&inverse).unwrap();
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let diff = product.sub(&identity).unwrap().norm();
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_pinv() {
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let pinv = matrix.pinv().unwrap();

        assert_eq!(pinv.shape(), &[3, 2]);

        // Check that A * A^+ * A = A (property of pseudo-inverse)
        let product1 = matrix.matmul(&pinv).unwrap();
        let product2 = product1.matmul(&matrix).unwrap();
        let diff = matrix.sub(&product2).unwrap().norm();
        assert!(diff < 1e-8);
    }

    #[test]
    fn test_svd_basic() {
        let matrix = Tensor::from_vec(vec![3.0, 1.0, 1.0, 3.0], vec![2, 2]);
        let (u, s, v) = matrix.svd().unwrap();

        assert_eq!(u.shape()[0], 2);
        assert_eq!(v.shape()[1], 2);
        assert!(s.numel() > 0);

        // Singular values should be non-negative and sorted
        let s_data = s.as_slice().unwrap();
        for i in 1..s_data.len() {
            assert!(s_data[i - 1] >= s_data[i]);
            assert!(s_data[i] >= 0.0);
        }
    }

    #[test]
    fn test_eigenvalue_decomposition() {
        // Test with a simple symmetric matrix
        let matrix = Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]);
        let (eigenvalues, eigenvectors) = matrix.eigh().unwrap();

        assert!(eigenvalues.numel() > 0);
        assert_eq!(eigenvectors.shape()[0], 2);

        // Eigenvalues should be real for symmetric matrices
        let eig_data = eigenvalues.as_slice().unwrap();
        for &val in eig_data {
            assert!(val.is_finite());
        }
    }
}
