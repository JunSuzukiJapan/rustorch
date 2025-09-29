//! F32Tensor インデックス操作
//! F32Tensor indexing operations

use crate::error::{RusTorchResult, RusTorchError};
use crate::hybrid_f32::tensor::core::F32Tensor;
use crate::hybrid_f32_experimental;
use std::ops::{Index, IndexMut};

/// 1次元インデックス
/// 1D index
pub struct Index1D(pub usize);

/// 2次元インデックス
/// 2D index
pub struct Index2D(pub usize, pub usize);

/// 3次元インデックス
/// 3D index
pub struct Index3D(pub usize, pub usize, pub usize);

/// スライス範囲
/// Slice range
#[derive(Debug, Clone)]
pub struct SliceRange {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub step: Option<usize>,
}

impl SliceRange {
    pub fn new(start: Option<usize>, end: Option<usize>, step: Option<usize>) -> Self {
        Self { start, end, step }
    }

    pub fn all() -> Self {
        Self::new(None, None, None)
    }

    pub fn from(start: usize) -> Self {
        Self::new(Some(start), None, None)
    }

    pub fn to(end: usize) -> Self {
        Self::new(None, Some(end), None)
    }

    pub fn range(start: usize, end: usize) -> Self {
        Self::new(Some(start), Some(end), None)
    }
}

// ========================================
// Index trait implementations
// ========================================

impl Index<usize> for F32Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 1 {
            panic!("Single index only supported for 1D tensors, got {}D", self.ndim());
        }
        
        if index >= self.numel() {
            panic!("Index {} out of bounds for tensor with {} elements", index, self.numel());
        }
        
        &self.data.as_slice().unwrap()[index]
    }
}

impl IndexMut<usize> for F32Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 1 {
            panic!("Single index only supported for 1D tensors, got {}D", self.ndim());
        }
        
        if index >= self.numel() {
            panic!("Index {} out of bounds for tensor with {} elements", index, self.numel());
        }
        
        &mut self.data.as_slice_mut().unwrap()[index]
    }
}

impl Index<Index2D> for F32Tensor {
    type Output = f32;

    fn index(&self, index: Index2D) -> &Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 2 {
            panic!("2D index only supported for 2D tensors, got {}D", self.ndim());
        }
        
        let shape = self.shape();
        if index.0 >= shape[0] || index.1 >= shape[1] {
            panic!("Index ({}, {}) out of bounds for tensor with shape {:?}", 
                   index.0, index.1, shape);
        }
        
        let flat_index = index.0 * shape[1] + index.1;
        &self.data.as_slice().unwrap()[flat_index]
    }
}

impl IndexMut<Index2D> for F32Tensor {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 2 {
            panic!("2D index only supported for 2D tensors, got {}D", self.ndim());
        }
        
        let shape = self.shape();
        if index.0 >= shape[0] || index.1 >= shape[1] {
            panic!("Index ({}, {}) out of bounds for tensor with shape {:?}", 
                   index.0, index.1, shape);
        }
        
        let flat_index = index.0 * shape[1] + index.1;
        &mut self.data.as_slice_mut().unwrap()[flat_index]
    }
}

impl Index<Index3D> for F32Tensor {
    type Output = f32;

    fn index(&self, index: Index3D) -> &Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 3 {
            panic!("3D index only supported for 3D tensors, got {}D", self.ndim());
        }
        
        let shape = self.shape();
        if index.0 >= shape[0] || index.1 >= shape[1] || index.2 >= shape[2] {
            panic!("Index ({}, {}, {}) out of bounds for tensor with shape {:?}", 
                   index.0, index.1, index.2, shape);
        }
        
        let flat_index = index.0 * shape[1] * shape[2] + index.1 * shape[2] + index.2;
        &self.data.as_slice().unwrap()[flat_index]
    }
}

impl IndexMut<Index3D> for F32Tensor {
    fn index_mut(&mut self, index: Index3D) -> &mut Self::Output {
        hybrid_f32_experimental!();
        
        if self.ndim() != 3 {
            panic!("3D index only supported for 3D tensors, got {}D", self.ndim());
        }
        
        let shape = self.shape();
        if index.0 >= shape[0] || index.1 >= shape[1] || index.2 >= shape[2] {
            panic!("Index ({}, {}, {}) out of bounds for tensor with shape {:?}", 
                   index.0, index.1, index.2, shape);
        }
        
        let flat_index = index.0 * shape[1] * shape[2] + index.1 * shape[2] + index.2;
        &mut self.data.as_slice_mut().unwrap()[flat_index]
    }
}

// ========================================
// 高度なインデックス操作
// Advanced indexing operations
// ========================================

impl F32Tensor {
    /// 安全な要素取得
    /// Safe element access
    pub fn get(&self, indices: &[usize]) -> RusTorchResult<f32> {
        hybrid_f32_experimental!();
        
        if indices.len() != self.ndim() {
            return Err(RusTorchError::InvalidOperation(
                format!("Expected {} indices for {}D tensor, got {}", 
                        self.ndim(), self.ndim(), indices.len())
            ));
        }
        
        let shape = self.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(RusTorchError::IndexError(
                    format!("Index {} out of bounds for dimension {} with size {}", 
                            idx, i, shape[i])
                ));
            }
        }
        
        let flat_index = self.calculate_flat_index(indices)?;
        Ok(self.data.as_slice().unwrap()[flat_index])
    }

    /// 安全な要素設定
    /// Safe element setting
    pub fn set(&mut self, indices: &[usize], value: f32) -> RusTorchResult<()> {
        hybrid_f32_experimental!();
        
        if indices.len() != self.ndim() {
            return Err(RusTorchError::InvalidOperation(
                format!("Expected {} indices for {}D tensor, got {}", 
                        self.ndim(), self.ndim(), indices.len())
            ));
        }
        
        let shape = self.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(RusTorchError::IndexError(
                    format!("Index {} out of bounds for dimension {} with size {}", 
                            idx, i, shape[i])
                ));
            }
        }
        
        let flat_index = self.calculate_flat_index(indices)?;
        self.data.as_slice_mut().unwrap()[flat_index] = value;
        Ok(())
    }

    /// スライス取得
    /// Get slice
    pub fn slice(&self, ranges: &[SliceRange]) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if ranges.len() != self.ndim() {
            return Err(RusTorchError::InvalidOperation(
                format!("Expected {} slice ranges for {}D tensor, got {}", 
                        self.ndim(), self.ndim(), ranges.len())
            ));
        }
        
        let shape = self.shape();
        let mut new_shape = Vec::new();
        let mut indices = Vec::new();
        
        for (dim, range) in ranges.iter().enumerate() {
            let start = range.start.unwrap_or(0);
            let end = range.end.unwrap_or(shape[dim]);
            let step = range.step.unwrap_or(1);
            
            if start >= shape[dim] || end > shape[dim] || start >= end {
                return Err(RusTorchError::IndexError(
                    format!("Invalid slice range for dimension {}: {}..{}", dim, start, end)
                ));
            }
            
            let dim_indices: Vec<usize> = (start..end).step_by(step).collect();
            new_shape.push(dim_indices.len());
            indices.push(dim_indices);
        }
        
        // 多次元インデックスの生成とデータ抽出
        let mut result_data = Vec::new();
        self.extract_slice_data(&indices, 0, &mut Vec::new(), &mut result_data)?;
        
        F32Tensor::new(result_data, new_shape)
    }

    /// 平坦化インデックス計算
    /// Calculate flat index
    fn calculate_flat_index(&self, indices: &[usize]) -> RusTorchResult<usize> {
        let shape = self.shape();
        let mut flat_index = 0;
        let mut stride = 1;
        
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        
        Ok(flat_index)
    }

    /// スライスデータ抽出（再帰）
    /// Extract slice data (recursive)
    fn extract_slice_data(
        &self,
        indices: &[Vec<usize>],
        dim: usize,
        current_indices: &mut Vec<usize>,
        result_data: &mut Vec<f32>,
    ) -> RusTorchResult<()> {
        if dim == indices.len() {
            let flat_index = self.calculate_flat_index(current_indices)?;
            result_data.push(self.data.as_slice().unwrap()[flat_index]);
            return Ok(());
        }
        
        for &idx in &indices[dim] {
            current_indices.push(idx);
            self.extract_slice_data(indices, dim + 1, current_indices, result_data)?;
            current_indices.pop();
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_indexing() {
        let mut tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        
        // 読み取りテスト
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[2], 3.0);
        assert_eq!(tensor[4], 5.0);
        
        // 書き込みテスト
        tensor[1] = 10.0;
        assert_eq!(tensor[1], 10.0);
    }

    #[test]
    fn test_2d_indexing() {
        let mut tensor = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap();
        
        // 読み取りテスト
        assert_eq!(tensor[Index2D(0, 0)], 1.0);
        assert_eq!(tensor[Index2D(0, 2)], 3.0);
        assert_eq!(tensor[Index2D(1, 1)], 5.0);
        
        // 書き込みテスト
        tensor[Index2D(1, 2)] = 20.0;
        assert_eq!(tensor[Index2D(1, 2)], 20.0);
    }

    #[test]
    fn test_safe_indexing() {
        let tensor = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();
        
        // 正常ケース
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 3.0);
        
        // エラーケース
        assert!(tensor.get(&[2, 0]).is_err()); // 範囲外
        assert!(tensor.get(&[0]).is_err());    // 次元不一致
    }

    #[test]
    fn test_slicing() {
        let tensor = F32Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap();
        
        // 全体スライス
        let slice1 = tensor.slice(&[
            SliceRange::all(),
            SliceRange::range(1, 3),
        ]).unwrap();
        
        assert_eq!(slice1.shape(), &[2, 2]);
        assert_eq!(slice1.get(&[0, 0]).unwrap(), 2.0); // tensor[0, 1]
        assert_eq!(slice1.get(&[1, 1]).unwrap(), 6.0); // tensor[1, 2]
    }
}