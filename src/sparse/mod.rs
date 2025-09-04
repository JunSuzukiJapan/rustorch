//! Sparse tensor support for RusTorch (Phase 12)
//! RusTorchスパーステンソルサポート（フェーズ12）
//!
//! This module provides comprehensive sparse tensor functionality including:
//! - COO (Coordinate) and CSR (Compressed Sparse Row) formats
//! - Optimized sparse operations and arithmetic
//! - Sparse neural network layer integration
//! - Model pruning and sparsification algorithms
//! - GPU acceleration for sparse computations

pub mod sparse_layers;
pub mod pruning;
pub mod gpu_ops;
pub mod utils;

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use ndarray::{ArrayD, Array1, Array2, Dimension};
use num_traits::{Float, Zero, One, FromPrimitive};
use std::collections::HashMap;
use std::fmt;
use std::iter::Sum;

/// Sparse tensor layout formats
/// スパーステンソルレイアウト形式
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SparseFormat {
    /// Coordinate format (COO) - stores (index, value) pairs
    /// 座標形式（COO） - (インデックス, 値)ペアを格納
    COO,
    /// Compressed Sparse Row (CSR) - row-major compressed format
    /// 圧縮スパース行（CSR） - 行メジャー圧縮形式
    CSR,
    /// Compressed Sparse Column (CSC) - column-major compressed format
    /// 圧縮スパース列（CSC） - 列メジャー圧縮形式
    CSC,
}

/// Sparse tensor data structure supporting multiple formats
/// 複数形式をサポートするスパーステンソル構造
#[derive(Clone)]
pub struct SparseTensor<T: Float> {
    /// Dense shape of the sparse tensor
    /// スパーステンソルの密な形状
    pub shape: Vec<usize>,
    /// Sparse format type
    /// スパース形式タイプ
    pub format: SparseFormat,
    /// Non-zero values
    /// 非ゼロ値
    pub values: Array1<T>,
    /// Indices for the sparse format
    /// スパース形式のインデックス
    pub indices: Vec<Array1<usize>>,
    /// Number of non-zero elements
    /// 非ゼロ要素数
    pub nnz: usize,
}

impl<T: Float + fmt::Debug> fmt::Debug for SparseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dense_size = self.shape.iter().product::<usize>();
        let sparsity = 1.0 - self.nnz as f64 / dense_size as f64;
        f.debug_struct("SparseTensor")
            .field("shape", &self.shape)
            .field("format", &self.format)
            .field("nnz", &self.nnz)
            .field("sparsity", &sparsity)
            .finish()
    }
}

// Core SparseTensor implementation with minimal trait bounds
impl<T: Float + Copy> SparseTensor<T> {
    /// Get sparsity ratio (percentage of zero elements)
    /// スパース率を取得（ゼロ要素の割合）
    pub fn sparsity(&self) -> f64 {
        1.0 - (self.nnz as f64 / self.dense_size() as f64)
    }

    /// Calculate total number of elements in dense representation
    /// 密表現での総要素数を計算
    pub fn dense_size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Memory usage in bytes (approximate)
    /// メモリ使用量（バイト、概算）
    pub fn memory_usage(&self) -> usize {
        let value_bytes = self.nnz * std::mem::size_of::<T>();
        let index_bytes = self.indices.iter().map(|arr| arr.len() * std::mem::size_of::<usize>()).sum::<usize>();
        value_bytes + index_bytes
    }
}

// Extended implementation with additional trait bounds for operations
impl<T: Float + Zero + One + std::ops::AddAssign + Copy + FromPrimitive> SparseTensor<T> {
    /// Create a new COO sparse tensor from indices and values
    /// インデックスと値からCOOスパーステンソルを作成
    pub fn from_coo(
        indices: Vec<Array1<usize>>,
        values: Array1<T>,
        shape: Vec<usize>,
    ) -> RusTorchResult<Self> {
        if indices.is_empty() {
            return Err(RusTorchError::InvalidParameters {
                operation: "sparse_tensor_creation".to_string(),
                message: "Indices cannot be empty for COO format".to_string(),
            });
        }

        if indices.len() != shape.len() {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![shape.len()],
                actual: vec![indices.len()],
            });
        }

        let nnz = values.len();
        if indices.iter().any(|idx_dim| idx_dim.len() != nnz) {
            return Err(RusTorchError::InvalidParameters {
                operation: "sparse_tensor_creation".to_string(),
                message: "All index dimensions must have same length as values".to_string(),
            });
        }

        Ok(SparseTensor {
            shape,
            format: SparseFormat::COO,
            values,
            indices,
            nnz,
        })
    }

    /// Create a new CSR sparse tensor for 2D matrices
    /// 2D行列用のCSRスパーステンソルを作成
    pub fn from_csr(
        row_ptr: Array1<usize>,
        col_indices: Array1<usize>, 
        values: Array1<T>,
        shape: Vec<usize>,
    ) -> RusTorchResult<Self> {
        if shape.len() != 2 {
            return Err(RusTorchError::InvalidParameters {
                operation: "csr_tensor_creation".to_string(),
                message: "CSR format only supports 2D tensors".to_string(),
            });
        }

        let (rows, _cols) = (shape[0], shape[1]);
        if row_ptr.len() != rows + 1 {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![rows + 1],
                actual: vec![row_ptr.len()],
            });
        }

        let nnz = values.len();
        if col_indices.len() != nnz {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![nnz],
                actual: vec![col_indices.len()],
            });
        }

        Ok(SparseTensor {
            shape,
            format: SparseFormat::CSR,
            values,
            indices: vec![row_ptr, col_indices],
            nnz,
        })
    }

    /// Convert dense tensor to sparse COO format with threshold
    /// しきい値を使って密テンソルをスパースCOO形式に変換
    pub fn from_dense(dense: &ArrayD<T>, threshold: T) -> RusTorchResult<Self> {
        let mut indices_vec = vec![Vec::new(); dense.ndim()];
        let mut values_vec = Vec::new();

        for (multi_idx, &value) in dense.indexed_iter() {
            if value.abs() > threshold {
                values_vec.push(value);
                for (dim, &idx) in multi_idx.as_array_view().iter().enumerate() {
                    indices_vec[dim].push(idx);
                }
            }
        }

        let indices: Vec<Array1<usize>> = indices_vec
            .into_iter()
            .map(|v| Array1::from_vec(v))
            .collect();
        let values = Array1::from_vec(values_vec);
        let shape = dense.shape().to_vec();

        Self::from_coo(indices, values, shape)
    }

    /// Convert sparse tensor to dense format
    /// スパーステンソルを密形式に変換
    pub fn to_dense(&self) -> RusTorchResult<ArrayD<T>> {
        let mut dense = ArrayD::zeros(self.shape.clone());

        match self.format {
            SparseFormat::COO => {
                for i in 0..self.nnz {
                    let mut coords = Vec::with_capacity(self.shape.len());
                    for dim_indices in &self.indices {
                        coords.push(dim_indices[i]);
                    }
                    dense[&coords[..]] = self.values[i];
                }
            }
            SparseFormat::CSR => {
                if self.shape.len() != 2 {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "csr_to_dense".to_string(),
                        message: "CSR format only supports 2D tensors".to_string(),
                    });
                }
                
                let row_ptr = &self.indices[0];
                let col_indices = &self.indices[1];
                
                for row in 0..self.shape[0] {
                    let start = row_ptr[row];
                    let end = row_ptr[row + 1];
                    
                    for idx in start..end {
                        let col = col_indices[idx];
                        dense[[row, col]] = self.values[idx];
                    }
                }
            }
            SparseFormat::CSC => {
                return Err(RusTorchError::NotImplemented {
                    feature: "CSC to dense conversion".to_string(),
                });
            }
        }

        Ok(dense)
    }

    /// Convert COO to CSR format (2D only)
    /// COOからCSR形式に変換（2Dのみ）
    pub fn to_csr(&self) -> RusTorchResult<SparseTensor<T>> {
        if self.shape.len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "coo_to_csr".to_string(),
                message: "COO to CSR conversion only supports 2D tensors".to_string(),
            });
        }

        if self.format != SparseFormat::COO {
            return Err(RusTorchError::InvalidOperation {
                operation: "coo_to_csr".to_string(),
                message: "Input must be in COO format".to_string(),
            });
        }

        let rows = self.shape[0];
        let row_indices = &self.indices[0];
        let col_indices = &self.indices[1];

        // Create row pointer array
        let mut row_ptr = Array1::zeros(rows + 1);
        for &row in row_indices.iter() {
            if row < rows {
                row_ptr[row + 1] += 1;
            }
        }

        // Convert counts to cumulative sums
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        SparseTensor::from_csr(row_ptr, col_indices.clone(), self.values.clone(), self.shape.clone())
    }

    /// Convert CSR to COO format (2D only)
    /// CSRからCOO形式に変換（2Dのみ）
    pub fn to_coo(&self) -> RusTorchResult<SparseTensor<T>> {
        if self.shape.len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "csr_to_coo".to_string(),
                message: "CSR to COO conversion only supports 2D tensors".to_string(),
            });
        }

        if self.format != SparseFormat::CSR {
            return Err(RusTorchError::InvalidOperation {
                operation: "csr_to_coo".to_string(),
                message: "Input must be in CSR format".to_string(),
            });
        }

        let row_ptr = &self.indices[0];
        let col_indices = &self.indices[1];
        
        // Convert CSR indices back to COO
        let mut row_indices = Vec::with_capacity(self.nnz);
        let mut coo_col_indices = Vec::with_capacity(self.nnz);
        
        for (row, &start) in row_ptr.iter().enumerate().take(self.shape[0]) {
            let end = row_ptr[row + 1];
            for idx in start..end {
                row_indices.push(row);
                coo_col_indices.push(col_indices[idx]);
            }
        }
        
        let indices = vec![
            Array1::from_vec(row_indices),
            Array1::from_vec(coo_col_indices),
        ];
        
        SparseTensor::from_coo(indices, self.values.clone(), self.shape.clone())
    }
}

/// Trait for sparse tensor operations
/// スパーステンソル演算のトレイト
pub trait SparseOps<T: Float> {
    /// Sparse matrix-vector multiplication
    /// スパース行列ベクトル乗算
    fn spmv(&self, vector: &Array1<T>) -> RusTorchResult<Array1<T>>;
    
    /// Sparse matrix-matrix multiplication  
    /// スパース行列行列乗算
    fn spmm(&self, matrix: &Array2<T>) -> RusTorchResult<Array2<T>>;
    
    /// Element-wise sparse operations
    /// 要素ごとスパース演算
    fn sparse_add(&self, other: &SparseTensor<T>) -> RusTorchResult<SparseTensor<T>>;
    fn sparse_mul(&self, other: &SparseTensor<T>) -> RusTorchResult<SparseTensor<T>>;
    
    /// Sparse tensor transpose
    /// スパーステンソル転置
    fn transpose(&self) -> RusTorchResult<SparseTensor<T>>;
}

impl<T: Float + Zero + One + std::ops::AddAssign + Copy + FromPrimitive> SparseOps<T> for SparseTensor<T> {
    fn spmv(&self, vector: &Array1<T>) -> RusTorchResult<Array1<T>> {
        if self.shape.len() != 2 {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![2],
                actual: vec![self.shape.len()],
            });
        }

        if vector.len() != self.shape[1] {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self.shape[1]],
                actual: vec![vector.len()],
            });
        }

        let mut result = Array1::zeros(self.shape[0]);

        match self.format {
            SparseFormat::COO => {
                let row_indices = &self.indices[0];
                let col_indices = &self.indices[1];
                
                for i in 0..self.nnz {
                    let row = row_indices[i];
                    let col = col_indices[i];
                    result[row] += self.values[i] * vector[col];
                }
            }
            SparseFormat::CSR => {
                let row_ptr = &self.indices[0];
                let col_indices = &self.indices[1];
                
                for row in 0..self.shape[0] {
                    let start = row_ptr[row];
                    let end = row_ptr[row + 1];
                    
                    for idx in start..end {
                        let col = col_indices[idx];
                        result[row] += self.values[idx] * vector[col];
                    }
                }
            }
            SparseFormat::CSC => {
                return Err(RusTorchError::NotImplemented {
                    feature: "CSC sparse matrix-vector multiplication".to_string(),
                });
            }
        }

        Ok(result)
    }

    fn spmm(&self, matrix: &Array2<T>) -> RusTorchResult<Array2<T>> {
        if self.shape.len() != 2 {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![2],
                actual: vec![self.shape.len()],
            });
        }

        if matrix.shape()[0] != self.shape[1] {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self.shape[1]],
                actual: vec![matrix.shape()[0]],
            });
        }

        let n_cols = matrix.shape()[1];
        let mut result = Array2::zeros((self.shape[0], n_cols));

        // Perform SpMM by computing SpMV for each column
        for col in 0..n_cols {
            let col_vector = matrix.column(col).to_owned();
            let result_col = self.spmv(&col_vector)?;
            
            for row in 0..self.shape[0] {
                result[[row, col]] = result_col[row];
            }
        }

        Ok(result)
    }

    fn sparse_add(&self, other: &SparseTensor<T>) -> RusTorchResult<SparseTensor<T>> {
        if self.shape != other.shape {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        // Convert both to dense, add, then convert back to sparse
        // This is not optimal but ensures correctness for initial implementation
        let dense_self = self.to_dense()?;
        let dense_other = other.to_dense()?;
        let dense_result = &dense_self + &dense_other;
        
        // Use small threshold to maintain sparsity
        let threshold = T::from_f64(1e-10).unwrap_or(T::zero());
        SparseTensor::from_dense(&dense_result, threshold)
    }

    fn sparse_mul(&self, other: &SparseTensor<T>) -> RusTorchResult<SparseTensor<T>> {
        if self.shape != other.shape {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        // For element-wise multiplication, only non-zero elements matter
        // 要素ごとの乗算では、非ゼロ要素のみが重要
        let mut result_indices = vec![Vec::new(); self.shape.len()];
        let mut result_values = Vec::new();

        match (self.format, other.format) {
            (SparseFormat::COO, SparseFormat::COO) => {
                // Create index maps for efficient lookup
                let mut self_map = HashMap::new();
                let mut other_map = HashMap::new();
                
                for i in 0..self.nnz {
                    let coords: Vec<_> = self.indices.iter().map(|idx| idx[i]).collect();
                    self_map.insert(coords, i);
                }
                
                for i in 0..other.nnz {
                    let coords: Vec<_> = other.indices.iter().map(|idx| idx[i]).collect();
                    other_map.insert(coords, i);
                }
                
                // Find intersecting indices
                for (coords, &self_idx) in &self_map {
                    if let Some(&other_idx) = other_map.get(coords) {
                        let product = self.values[self_idx] * other.values[other_idx];
                        if !product.is_zero() {
                            result_values.push(product);
                            for (dim, &coord) in coords.iter().enumerate() {
                                result_indices[dim].push(coord);
                            }
                        }
                    }
                }
            }
            _ => {
                // Fall back to dense conversion for other format combinations
                let dense_self = self.to_dense()?;
                let dense_other = other.to_dense()?;
                let dense_result = &dense_self * &dense_other;
                
                let threshold = T::from_f64(1e-10).unwrap_or(T::zero());
                return SparseTensor::from_dense(&dense_result, threshold);
            }
        }

        let indices: Vec<Array1<usize>> = result_indices
            .into_iter()
            .map(|v| Array1::from_vec(v))
            .collect();
        let values = Array1::from_vec(result_values);

        SparseTensor::from_coo(indices, values, self.shape.clone())
    }

    fn transpose(&self) -> RusTorchResult<SparseTensor<T>> {
        if self.shape.len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "sparse_transpose".to_string(),
                message: "Transpose only supported for 2D sparse tensors".to_string(),
            });
        }

        match self.format {
            SparseFormat::COO => {
                let mut transposed_shape = self.shape.clone();
                transposed_shape.swap(0, 1);
                
                let mut transposed_indices = self.indices.clone();
                transposed_indices.swap(0, 1);
                
                SparseTensor::from_coo(transposed_indices, self.values.clone(), transposed_shape)
            }
            SparseFormat::CSR => {
                // CSR transpose -> CSC, which we'll represent as CSR with swapped dimensions
                let transposed_shape = vec![self.shape[1], self.shape[0]];
                
                // Convert CSR to COO, transpose, then back to CSR
                let coo_version = self.to_coo()?;
                let transposed_coo = coo_version.transpose()?;
                transposed_coo.to_csr()
            }
            SparseFormat::CSC => {
                return Err(RusTorchError::NotImplemented {
                    feature: "CSC transpose".to_string(),
                });
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_coo_creation() {
        let indices = vec![
            Array1::from_vec(vec![0, 1, 2]),
            Array1::from_vec(vec![0, 1, 2]),
        ];
        let values = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let shape = vec![3, 3];
        
        let sparse_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        assert_eq!(sparse_tensor.format, SparseFormat::COO);
        assert_eq!(sparse_tensor.nnz, 3);
        assert!((sparse_tensor.sparsity() - 2.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_to_sparse_conversion() {
        let dense_data = Array2::from_shape_vec((3, 3), vec![
            1.0f32, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]).unwrap().into_dyn();
        
        let sparse_tensor = SparseTensor::from_dense(&dense_data, 0.5f32).unwrap();
        assert_eq!(sparse_tensor.nnz, 3);
        assert_eq!(sparse_tensor.format, SparseFormat::COO);
        
        let reconstructed = sparse_tensor.to_dense().unwrap();
        assert_eq!(reconstructed, dense_data);
    }

    #[test]
    fn test_coo_to_csr_conversion() {
        let indices = vec![
            Array1::from_vec(vec![0, 1, 1, 2]),
            Array1::from_vec(vec![0, 1, 2, 2]),
        ];
        let values = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let shape = vec![3, 3];
        
        let coo_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        let csr_tensor = coo_tensor.to_csr().unwrap();
        
        assert_eq!(csr_tensor.format, SparseFormat::CSR);
        assert_eq!(csr_tensor.nnz, 4);
        
        // Verify CSR structure
        let row_ptr = &csr_tensor.indices[0];
        assert_eq!(row_ptr.len(), 4); // rows + 1
    }

    #[test]
    fn test_sparse_matrix_vector_multiply() {
        let indices = vec![
            Array1::from_vec(vec![0, 1, 2]),
            Array1::from_vec(vec![0, 1, 2]),
        ];
        let values = Array1::from_vec(vec![2.0f32, 3.0, 4.0]);
        let shape = vec![3, 3];
        
        let sparse_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        let vector = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        
        let result = sparse_tensor.spmv(&vector).unwrap();
        assert_eq!(result[0], 2.0); // 2*1
        assert_eq!(result[1], 6.0); // 3*2  
        assert_eq!(result[2], 12.0); // 4*3
    }
}