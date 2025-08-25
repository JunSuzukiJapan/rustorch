//! Utility operations for tensors
//! テンソルのユーティリティ演算
//!
//! This module provides utility functions including mapping, batch operations,
//! and helper functions for tensor manipulation.
//! このモジュールはマッピング、バッチ操作、テンソル操作のヘルパー関数を含むユーティリティ関数を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Get the batch size (first dimension)
    /// バッチサイズを取得（最初の次元）
    pub fn batch_size(&self) -> usize {
        self.shape().get(0).copied().unwrap_or(1)
    }

    /// Apply function to each element
    /// 各要素に関数を適用
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mapped_data: Vec<T> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_vec(mapped_data, self.shape().to_vec())
    }

    /// Apply function with index to each element
    /// インデックス付き関数を各要素に適用
    pub fn map_with_index<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T, usize) -> T,
    {
        let mapped_data: Vec<T> = self.data.iter().enumerate()
            .map(|(i, &x)| f(x, i))
            .collect();
        Tensor::from_vec(mapped_data, self.shape().to_vec())
    }

    /// Apply function element-wise between two tensors
    /// 2つのテンソル間で要素ごとに関数を適用
    pub fn zip_with<F>(&self, other: &Tensor<T>, f: F) -> Result<Tensor<T>, String>
    where
        F: Fn(T, T) -> T,
    {
        if self.shape() != other.shape() {
            return Err("Shape mismatch for zip_with operation".to_string());
        }

        let result_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        
        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Select elements based on condition
    /// 条件に基づいて要素を選択
    pub fn select<F>(&self, condition: F) -> Tensor<T>
    where
        F: Fn(T) -> bool,
    {
        let selected_data: Vec<T> = self.data.iter()
            .filter(|&&x| condition(x))
            .copied()
            .collect();
        
        let selected_len = selected_data.len();
        Tensor::from_vec(selected_data, vec![selected_len])
    }

    /// Count elements satisfying condition
    /// 条件を満たす要素をカウント
    pub fn count_where<F>(&self, condition: F) -> usize
    where
        F: Fn(T) -> bool,
    {
        self.data.iter().filter(|&&x| condition(x)).count()
    }

    /// Check if all elements satisfy condition
    /// 全要素が条件を満たすかチェック
    pub fn all<F>(&self, condition: F) -> bool
    where
        F: Fn(T) -> bool,
    {
        self.data.iter().all(|&x| condition(x))
    }

    /// Check if any element satisfies condition
    /// いずれかの要素が条件を満たすかチェック
    pub fn any<F>(&self, condition: F) -> bool
    where
        F: Fn(T) -> bool,
    {
        self.data.iter().any(|&x| condition(x))
    }

    /// Find first index where condition is true
    /// 条件が真となる最初のインデックスを検索
    pub fn find_first<F>(&self, condition: F) -> Option<usize>
    where
        F: Fn(T) -> bool,
    {
        self.data.iter().position(|&x| condition(x))
    }

    /// Find all indices where condition is true
    /// 条件が真となる全インデックスを検索
    pub fn find_all<F>(&self, condition: F) -> Vec<usize>
    where
        F: Fn(T) -> bool,
    {
        self.data.iter()
            .enumerate()
            .filter_map(|(i, &x)| if condition(x) { Some(i) } else { None })
            .collect()
    }

    /// Replace elements based on condition
    /// 条件に基づいて要素を置換
    pub fn replace_where<F>(&self, condition: F, replacement: T) -> Tensor<T>
    where
        F: Fn(T) -> bool,
    {
        let replaced_data: Vec<T> = self.data.iter()
            .map(|&x| if condition(x) { replacement } else { x })
            .collect();
        
        Tensor::from_vec(replaced_data, self.shape().to_vec())
    }

    // Note: where_tensor function removed due to boolean tensor type constraints
    // ブールテンソル型制約のためwhere_tensor関数を削除

    /// Get unique elements in tensor (simplified implementation)
    /// テンソル内のユニーク要素を取得（簡略化実装）
    pub fn unique(&self) -> Tensor<T>
    where
        T: PartialOrd + Copy,
    {
        let mut unique_data: Vec<T> = self.data.to_vec();
        unique_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_data.dedup();
        
        Tensor::from_vec(unique_data, vec![unique_data.len()])
    }

    /// Split tensor into chunks along specified dimension
    /// 指定次元でテンソルをチャンクに分割
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor<T>>, String> {
        if dim >= self.ndim() {
            return Err(format!("Dimension {} is out of bounds", dim));
        }

        let shape = self.shape();
        let dim_size = shape[dim];
        
        if chunks == 0 {
            return Err("Number of chunks must be positive".to_string());
        }

        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division
        let mut result = Vec::new();
        
        for i in 0..chunks {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, dim_size);
            
            if start >= dim_size {
                break;
            }

            let chunk_tensor = self.slice_along_dim(dim, start, end)?;
            result.push(chunk_tensor);
        }
        
        Ok(result)
    }

    /// Split tensor at specified indices along dimension
    /// 指定インデックスで次元に沿ってテンソルを分割
    pub fn split(&self, split_size_or_sections: &[usize], dim: usize) -> Result<Vec<Tensor<T>>, String> {
        if dim >= self.ndim() {
            return Err(format!("Dimension {} is out of bounds", dim));
        }

        let mut result = Vec::new();
        let mut start_idx = 0;

        for &size in split_size_or_sections {
            let end_idx = start_idx + size;
            if end_idx > self.shape()[dim] {
                return Err("Split sizes exceed dimension size".to_string());
            }

            let chunk = self.slice_along_dim(dim, start_idx, end_idx)?;
            result.push(chunk);
            start_idx = end_idx;
        }

        // Handle remaining elements if any
        if start_idx < self.shape()[dim] {
            let chunk = self.slice_along_dim(dim, start_idx, self.shape()[dim])?;
            result.push(chunk);
        }

        Ok(result)
    }

    /// Slice tensor along specified dimension
    /// 指定次元に沿ってテンソルをスライス
    fn slice_along_dim(&self, dim: usize, start: usize, end: usize) -> Result<Tensor<T>, String> {
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape[dim] = end - start;

        if start >= end || end > shape[dim] {
            return Err("Invalid slice indices".to_string());
        }

        let data = self.as_slice().unwrap();
        let mut result_data = Vec::new();

        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let dim_stride = strides[dim];
        let outer_size = if dim == 0 { 1 } else { shape[..dim].iter().product() };
        let inner_size = if dim == shape.len() - 1 { 1 } else { shape[dim + 1..].iter().product() };

        for outer in 0..outer_size {
            for i in start..end {
                for inner in 0..inner_size {
                    let idx = outer * (shape[dim] * inner_size) + i * inner_size + inner;
                    result_data.push(data[idx]);
                }
            }
        }

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    /// Concatenate tensors along specified dimension
    /// 指定次元に沿ってテンソルを連結
    pub fn concat(tensors: &[&Tensor<T>], dim: usize) -> Result<Tensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot concatenate empty list of tensors".to_string());
        }

        let first_shape = tensors[0].shape();
        
        if dim >= first_shape.len() {
            return Err(format!("Dimension {} is out of bounds", dim));
        }

        // Verify that all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape().len() != first_shape.len() {
                return Err(format!("Tensor {} has different number of dimensions", i));
            }

            for (j, (&dim_size, &first_dim_size)) in tensor.shape().iter().zip(first_shape.iter()).enumerate() {
                if j != dim && dim_size != first_dim_size {
                    return Err(format!("Tensors have incompatible shapes at dimension {}", j));
                }
            }
        }

        // Calculate new shape
        let mut new_shape = first_shape.to_vec();
        new_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();

        // Concatenate data
        let mut result_data = Vec::new();
        let outer_size = first_shape[..dim].iter().product::<usize>();
        let inner_size = first_shape[dim + 1..].iter().product::<usize>();

        for outer in 0..outer_size {
            for tensor in tensors {
                let tensor_data = tensor.as_slice().unwrap();
                let tensor_dim_size = tensor.shape()[dim];

                for i in 0..tensor_dim_size {
                    for inner in 0..inner_size {
                        let idx = outer * (tensor_dim_size * inner_size) + i * inner_size + inner;
                        result_data.push(tensor_data[idx]);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    /// Squeeze tensor (remove dimensions of size 1)
    /// テンソルを絞る（サイズ1の次元を削除）
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor<T>, String> {
        let shape = self.shape();
        let new_shape: Vec<usize> = if let Some(d) = dim {
            if d >= shape.len() {
                return Err(format!("Dimension {} is out of bounds", d));
            }
            if shape[d] != 1 {
                return Err(format!("Cannot squeeze dimension {} with size {}", d, shape[d]));
            }
            shape.iter().enumerate()
                .filter_map(|(i, &size)| if i != d { Some(size) } else { None })
                .collect()
        } else {
            shape.iter().copied().filter(|&size| size != 1).collect()
        };

        if new_shape.is_empty() {
            // Result is scalar
            Ok(Tensor::from_vec(self.as_slice().unwrap().to_vec(), vec![]))
        } else {
            Ok(Tensor::from_vec(self.as_slice().unwrap().to_vec(), new_shape))
        }
    }

    /// Unsqueeze tensor (add dimension of size 1)
    /// テンソルを展開（サイズ1の次元を追加）
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor<T>, String> {
        let shape = self.shape();
        
        if dim > shape.len() {
            return Err(format!("Dimension {} is out of bounds", dim));
        }

        let mut new_shape = shape.to_vec();
        new_shape.insert(dim, 1);

        Ok(Tensor::from_vec(self.as_slice().unwrap().to_vec(), new_shape))
    }

    /// Permute dimensions
    /// 次元を置換
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor<T>, String> {
        let shape = self.shape();
        
        if dims.len() != shape.len() {
            return Err("Number of dimensions must match tensor dimensions".to_string());
        }

        // Check that dims is a valid permutation
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();
        let expected: Vec<usize> = (0..shape.len()).collect();
        if sorted_dims != expected {
            return Err("Invalid permutation of dimensions".to_string());
        }

        let permuted = self.data.clone().permuted_axes(dims);
        Ok(Tensor::new(permuted))
    }

    /// Expand tensor (repeat along dimensions of size 1)
    /// テンソルを拡張（サイズ1の次元に沿って繰り返し）
    pub fn expand(&self, sizes: &[usize]) -> Result<Tensor<T>, String> {
        let shape = self.shape();
        
        if sizes.len() != shape.len() {
            return Err("Number of sizes must match tensor dimensions".to_string());
        }

        // Check that expansion is valid
        for (&current_size, &target_size) in shape.iter().zip(sizes.iter()) {
            if current_size != 1 && current_size != target_size {
                return Err(format!("Cannot expand size {} to size {}", current_size, target_size));
            }
        }

        // If no expansion needed, return clone
        if shape == sizes {
            return Ok(self.clone());
        }

        // Use tile-like expansion
        let reps: Vec<usize> = shape.iter()
            .zip(sizes.iter())
            .map(|(&current, &target)| target / current)
            .collect();

        Self::tile(self, &reps)
    }
}

// Note: Boolean tensor implementation removed due to Float trait constraint
// Float特性制約のためブールテンソル実装を削除

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_size() {
        let tensor = Tensor::from_vec(vec![1.0f32; 12], vec![3, 4]);
        assert_eq!(tensor.batch_size(), 3);
        
        let scalar = Tensor::from_vec(vec![1.0f32], vec![]);
        assert_eq!(scalar.batch_size(), 1);
    }

    #[test]
    fn test_map() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let mapped = tensor.map(|x| x * 2.0);
        
        assert_eq!(mapped.as_slice().unwrap(), &[2.0f32, 4.0, 6.0]);
    }

    #[test]
    fn test_map_with_index() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let mapped = tensor.map_with_index(|x, i| x + i as f32);
        
        assert_eq!(mapped.as_slice().unwrap(), &[1.0f32, 3.0, 5.0]);
    }

    #[test]
    fn test_zip_with() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        
        let result = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0f32, 7.0, 9.0]);
    }

    #[test]
    fn test_select() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let selected = tensor.select(|x| x > 3.0);
        
        assert_eq!(selected.as_slice().unwrap(), &[4.0f32, 5.0]);
    }

    #[test]
    fn test_count_where() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let count = tensor.count_where(|x| x > 3.0);
        
        assert_eq!(count, 2);
    }

    #[test]
    fn test_all_any() {
        let tensor = Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 8.0], vec![4]);
        
        assert!(tensor.all(|x| x > 0.0));
        assert!(!tensor.all(|x| x > 5.0));
        
        assert!(tensor.any(|x| x > 7.0));
        assert!(!tensor.any(|x| x > 10.0));
    }

    #[test]
    fn test_find() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        assert_eq!(tensor.find_first(|x| x > 3.0), Some(3));
        assert_eq!(tensor.find_first(|x| x > 10.0), None);
        
        let indices = tensor.find_all(|x| x > 3.0);
        assert_eq!(indices, vec![3, 4]);
    }

    #[test]
    fn test_replace_where() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let replaced = tensor.replace_where(|x| x > 3.0, 0.0);
        
        assert_eq!(replaced.as_slice().unwrap(), &[1.0f32, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unique() {
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 3.0], vec![6]);
        let unique = tensor.unique();
        
        assert_eq!(unique.as_slice().unwrap(), &[1.0f32, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_chunk() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let chunks = tensor.chunk(3, 0).unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_slice().unwrap(), &[1.0f32, 2.0]);
        assert_eq!(chunks[1].as_slice().unwrap(), &[3.0f32, 4.0]);
        assert_eq!(chunks[2].as_slice().unwrap(), &[5.0f32, 6.0]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0f32, 4.0], vec![2]);
        let c = Tensor::from_vec(vec![5.0f32, 6.0], vec![2]);
        
        let concatenated = Tensor::concat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(concatenated.shape(), &[6]);
        assert_eq!(concatenated.as_slice().unwrap(), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]);
        
        let squeezed = tensor.squeeze(None).unwrap();
        assert_eq!(squeezed.shape(), &[3]);
        
        let unsqueezed = squeezed.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_permute() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let permuted = tensor.permute(&[1, 0]).unwrap();
        
        assert_eq!(permuted.shape(), &[3, 2]);
    }

    #[test]
    fn test_expand() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3]);
        let expanded = tensor.expand(&[2, 3]).unwrap();
        
        assert_eq!(expanded.shape(), &[2, 3]);
        assert_eq!(expanded.as_slice().unwrap(), &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_boolean_tensor() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let bool_tensor = Tensor::from_condition(&tensor, |x| x > 3.0);
        
        assert_eq!(bool_tensor.as_slice().unwrap(), &[false, false, false, true, true]);
    }
}