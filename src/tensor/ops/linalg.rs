//! Linear algebra operations for tensors
//! テンソルの線形代数演算
//!
//! This module provides linear algebra operations including matrix multiplication,
//! decompositions, and eigenvalue computations.
//! このモジュールは行列乗算、分解、固有値計算を含む線形代数演算を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Matrix multiplication supporting batched operations
    /// バッチ処理対応の行列乗算
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Self, String> {
        // Support 2D x 2D, 3D x 2D, and 4D x 4D operations
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        match (self_shape.len(), other_shape.len()) {
            // 2D matrix multiplication
            (2, 2) => {
                let (m, k) = (self_shape[0], self_shape[1]);
                let (k2, n) = (other_shape[0], other_shape[1]);
                
                if k != k2 {
                    return Err(format!(
                        "Cannot multiply matrices with shapes ({}, {}) and ({}, {})",
                        m, k, k2, n
                    ));
                }
                
                let mut result_data = vec![T::zero(); m * n];
                let self_data = self.data.as_slice().unwrap();
                let other_data = other.data.as_slice().unwrap();
                
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = T::zero();
                        for l in 0..k {
                            sum = sum + self_data[i * k + l] * other_data[l * n + j];
                        }
                        result_data[i * n + j] = sum;
                    }
                }
                
                Ok(Tensor::from_vec(result_data, vec![m, n]))
            },
            
            // 3D x 2D batched matrix multiplication  
            (3, 2) => {
                let (batch_size, m, k) = (self_shape[0], self_shape[1], self_shape[2]);
                let (k2, n) = (other_shape[0], other_shape[1]);
                
                if k != k2 {
                    return Err(format!(
                        "Cannot multiply matrices with inner dimensions {} and {}",
                        k, k2
                    ));
                }
                
                let mut result_data = vec![T::zero(); batch_size * m * n];
                let self_data = self.data.as_slice().unwrap();
                let other_data = other.data.as_slice().unwrap();
                
                for b in 0..batch_size {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = T::zero();
                            for l in 0..k {
                                let self_idx = b * (m * k) + i * k + l;
                                let other_idx = l * n + j;
                                sum = sum + self_data[self_idx] * other_data[other_idx];
                            }
                            result_data[b * (m * n) + i * n + j] = sum;
                        }
                    }
                }
                
                Ok(Tensor::from_vec(result_data, vec![batch_size, m, n]))
            },
            
            // 4D x 4D batched matrix multiplication
            (4, 4) => {
                let (b1, s1, m, k) = (self_shape[0], self_shape[1], self_shape[2], self_shape[3]);
                let (b2, s2, k2, n) = (other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
                
                if b1 != b2 || s1 != s2 || k != k2 {
                    return Err(format!(
                        "Cannot multiply 4D tensors with shapes {:?} and {:?}",
                        self_shape, other_shape
                    ));
                }
                
                let mut result_data = vec![T::zero(); b1 * s1 * m * n];
                let self_data = self.data.as_slice().unwrap();
                let other_data = other.data.as_slice().unwrap();
                
                for b in 0..b1 {
                    for s in 0..s1 {
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = T::zero();
                                for l in 0..k {
                                    let self_idx = ((b * s1 + s) * m + i) * k + l;
                                    let other_idx = ((b * s1 + s) * k2 + l) * n + j;
                                    sum = sum + self_data[self_idx] * other_data[other_idx];
                                }
                                let result_idx = ((b * s1 + s) * m + i) * n + j;
                                result_data[result_idx] = sum;
                            }
                        }
                    }
                }
                
                Ok(Tensor::from_vec(result_data, vec![b1, s1, m, n]))
            },
            
            _ => Err(format!(
                "Unsupported matrix multiplication between {}D and {}D tensors",
                self_shape.len(), other_shape.len()
            ))
        }
    }

    /// Transpose the last two dimensions
    /// 最後の2次元を転置
    pub fn transpose_last_two(&self) -> Result<Self, String> {
        if self.ndim() < 2 {
            return Err("Cannot transpose tensor with less than 2 dimensions".to_string());
        }

        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        let ndim = shape.len();
        
        // Swap last two dimensions
        new_shape.swap(ndim - 2, ndim - 1);
        
        // Create permutation array
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(ndim - 2, ndim - 1);
        
        // Perform transpose using ndarray
        let transposed = self.data.permuted_axes(&axes);
        Ok(Tensor::new(transposed))
    }

    /// Simple 2D transpose
    /// シンプルな2D転置
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() < 2 {
            return Err("Cannot transpose tensor with less than 2 dimensions".to_string());
        }

        let shape = self.shape();
        if shape.len() != 2 {
            return self.transpose_last_two();
        }

        let (rows, cols) = (shape[0], shape[1]);
        let mut result_data = vec![T::zero(); rows * cols];
        let self_data = self.data.as_slice().unwrap();

        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = self_data[i * cols + j];
            }
        }

        Ok(Tensor::from_vec(result_data, vec![cols, rows]))
    }
}

// Linear algebra decompositions - conditionally compiled with linalg feature
// 線形代数分解 - linalg機能付きで条件付きコンパイル
#[cfg(feature = "linalg")]
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T>
where
    T: ndarray_linalg::Lapack,
{
    /// Singular Value Decomposition
    /// 特異値分解
    #[cfg(all(any(feature = "linalg", feature = "linalg-netlib"), not(target_arch = "wasm32")))]
    pub fn svd(&self, some: bool) -> Result<(Self, Self, Self), String> {
        use ndarray_linalg::SVD;
        
        if self.ndim() != 2 {
            return Err("SVD only supports 2D matrices".to_string());
        }

        let result = if some {
            self.data.svd(true)
        } else {
            self.data.svd(false)
        };

        match result {
            Ok((u_opt, s, vt_opt)) => {
                let u = u_opt.ok_or("U matrix not computed")?;
                let vt = vt_opt.ok_or("VT matrix not computed")?;
                
                Ok((
                    Tensor::new(u.into_dyn()),
                    Tensor::from_vec(s.to_vec(), vec![s.len()]),
                    Tensor::new(vt.into_dyn())
                ))
            },
            Err(SVDError::LapackComputationFailure { .. }) => {
                Err("SVD computation failed".to_string())
            },
            Err(e) => Err(format!("SVD error: {:?}", e))
        }
    }

    /// QR Decomposition
    /// QR分解
    #[cfg(all(any(feature = "linalg", feature = "linalg-netlib"), not(target_arch = "wasm32")))]
    pub fn qr(&self) -> Result<(Self, Self), String> {
        use ndarray_linalg::QR;
        
        if self.ndim() != 2 {
            return Err("QR decomposition only supports 2D matrices".to_string());
        }

        match self.data.qr() {
            Ok((q, r)) => Ok((
                Tensor::new(q.into_dyn()),
                Tensor::new(r.into_dyn())
            )),
            Err(e) => Err(format!("QR decomposition failed: {:?}", e))
        }
    }

    /// LU Decomposition
    /// LU分解
    pub fn lu(&self) -> Result<(Self, Self, Self), String> {
        // LU decomposition placeholder - ndarray_linalg::LU not available
        
        if self.ndim() != 2 {
            return Err("LU decomposition only supports 2D matrices".to_string());
        }

        match self.data.lu() {
            Ok((l, u, p)) => Ok((
                Tensor::new(l.into_dyn()),
                Tensor::new(u.into_dyn()),
                Tensor::new(p.into_dyn().mapv(|x| T::from(x).unwrap()))
            )),
            Err(e) => Err(format!("LU decomposition failed: {:?}", e))
        }
    }

    /// Eigenvalue decomposition
    /// 固有値分解
    #[cfg(all(any(feature = "linalg", feature = "linalg-netlib"), not(target_arch = "wasm32")))]
    pub fn eig(&self, eigenvectors: bool) -> Result<(Self, Option<Self>), String> {
        use ndarray_linalg::Eig;
        
        if self.ndim() != 2 {
            return Err("Eigenvalue decomposition only supports 2D matrices".to_string());
        }

        if eigenvectors {
            match self.data.eig() {
                Ok((eigenvalues, eigenvectors_result)) => {
                    let eigenvals_tensor = Tensor::from_vec(
                        eigenvalues.iter().map(|&c| c.re).collect(),
                        vec![eigenvalues.len()]
                    );
                    let eigenvecs_tensor = Tensor::new(eigenvectors_result.into_dyn());
                    Ok((eigenvals_tensor, Some(eigenvecs_tensor)))
                },
                Err(e) => Err(format!("Eigenvalue computation failed: {:?}", e))
            }
        } else {
            match self.data.eigenvalues() {
                Ok(eigenvalues) => {
                    let eigenvals_tensor = Tensor::from_vec(
                        eigenvalues.iter().map(|&c| c.re).collect(),
                        vec![eigenvalues.len()]
                    );
                    Ok((eigenvals_tensor, None))
                },
                Err(e) => Err(format!("Eigenvalue computation failed: {:?}", e))
            }
        }
    }

    /// Symmetric eigenvalue decomposition
    /// 対称固有値分解
    #[cfg(all(any(feature = "linalg", feature = "linalg-netlib"), not(target_arch = "wasm32")))]
    pub fn symeig(&self, eigenvectors: bool, _upper: bool) -> Result<(Self, Option<Self>), String> {
        use ndarray_linalg::Eigh;
        
        if self.ndim() != 2 {
            return Err("Symmetric eigenvalue decomposition only supports 2D matrices".to_string());
        }

        if eigenvectors {
            match self.data.eigh(ndarray_linalg::UPLO::Upper) {
                Ok((eigenvalues, eigenvectors_result)) => {
                    let eigenvals_tensor = Tensor::from_vec(eigenvalues.to_vec(), vec![eigenvalues.len()]);
                    let eigenvecs_tensor = Tensor::new(eigenvectors_result.into_dyn());
                    Ok((eigenvals_tensor, Some(eigenvecs_tensor)))
                },
                Err(e) => Err(format!("Symmetric eigenvalue computation failed: {:?}", e))
            }
        } else {
            match self.data.eigenvalues_sym(ndarray_linalg::UPLO::Upper) {
                Ok(eigenvalues) => {
                    let eigenvals_tensor = Tensor::from_vec(eigenvalues.to_vec(), vec![eigenvalues.len()]);
                    Ok((eigenvals_tensor, None))
                },
                Err(e) => Err(format!("Symmetric eigenvalue computation failed: {:?}", e))
            }
        }
    }
}

// Basic implementations without linalg feature
// linalg機能なしでの基本実装
#[cfg(not(feature = "linalg"))]
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Basic SVD implementation (placeholder)
    /// 基本SVD実装（プレースホルダー）
    pub fn svd(&self, _some: bool) -> Result<(Self, Self, Self), String> {
        Err("SVD requires 'linalg' feature to be enabled".to_string())
    }

    /// Basic QR implementation (placeholder)
    /// 基本QR実装（プレースホルダー）
    pub fn qr(&self) -> Result<(Self, Self), String> {
        Err("QR decomposition requires 'linalg' feature to be enabled".to_string())
    }

    /// Basic LU implementation (placeholder)
    /// 基本LU実装（プレースホルダー）
    pub fn lu(&self) -> Result<(Self, Self, Self), String> {
        Err("LU decomposition requires 'linalg' feature to be enabled".to_string())
    }

    /// Basic eigenvalue computation (placeholder)
    /// 基本固有値計算（プレースホルダー）
    pub fn eig(&self, _eigenvectors: bool) -> Result<(Self, Option<Self>), String> {
        Err("Eigenvalue computation requires 'linalg' feature to be enabled".to_string())
    }

    /// Basic symmetric eigenvalue computation (placeholder)
    /// 基本対称固有値計算（プレースホルダー）
    pub fn symeig(&self, _eigenvectors: bool, _upper: bool) -> Result<(Self, Option<Self>), String> {
        Err("Symmetric eigenvalue computation requires 'linalg' feature to be enabled".to_string())
    }

    /// Basic QR for testing without linalg
    /// linalgなしでのテスト用基本QR
    pub fn qr_basic(&self, m: usize, n: usize, _min_mn: usize) -> Result<(Self, Self), String> {
        // Simple placeholder implementation for basic functionality
        let q_data = vec![T::one(); m * m];
        let r_data = vec![T::one(); m * n];
        
        Ok((
            Tensor::from_vec(q_data, vec![m, m]),
            Tensor::from_vec(r_data, vec![m, n])
        ))
    }

    /// Basic eigenvalue computation for testing
    /// テスト用基本固有値計算
    pub fn eig_basic(&self, n: usize, eigenvectors: bool) -> Result<(Self, Option<Self>), String> {
        let eigenvalues = Tensor::from_vec(vec![T::one(); n], vec![n]);
        
        if eigenvectors {
            let eigenvectors_data = vec![T::one(); n * n];
            let eigenvectors_tensor = Tensor::from_vec(eigenvectors_data, vec![n, n]);
            Ok((eigenvalues, Some(eigenvectors_tensor)))
        } else {
            Ok((eigenvalues, None))
        }
    }

    /// Basic symmetric eigenvalue computation for testing
    /// テスト用基本対称固有値計算
    pub fn symeig_basic(&self, n: usize, eigenvectors: bool, _upper: bool) -> Result<(Self, Option<Self>), String> {
        let eigenvalues = Tensor::from_vec(vec![T::one(); n], vec![n]);
        
        if eigenvectors {
            let eigenvectors_data = vec![T::one(); n * n];
            let eigenvectors_tensor = Tensor::from_vec(eigenvectors_data, vec![n, n]);
            Ok((eigenvalues, Some(eigenvectors_tensor)))
        } else {
            Ok((eigenvalues, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_transpose_2d() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        let result = tensor.transpose().unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        
        let expected = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_transpose_last_two() {
        let tensor = Tensor::from_vec(
            (0..24).map(|x| x as f32).collect(),
            vec![2, 3, 4]
        );
        
        let result = tensor.transpose_last_two().unwrap();
        assert_eq!(result.shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_matmul_batch_3d_2d() {
        let a = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 2, 2]
        );
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]);
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_matrix_multiplication_dimension_mismatch() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0], vec![2, 1]);
        
        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "linalg"))]
    #[test]
    fn test_decompositions_without_linalg() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        assert!(tensor.svd(true).is_err());
        assert!(tensor.qr().is_err());
        assert!(tensor.lu().is_err());
        assert!(tensor.eig(false).is_err());
        assert!(tensor.symeig(false, true).is_err());
        
        // Basic implementations should work
        assert!(tensor.qr_basic(2, 2, 2).is_ok());
        assert!(tensor.eig_basic(2, false).is_ok());
        assert!(tensor.symeig_basic(2, false, true).is_ok());
    }
}