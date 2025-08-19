/// Advanced indexing and slicing operations for tensors
/// テンソルの高度なインデックス・スライシング操作

use ndarray::{ArrayD, IxDyn, Slice, SliceInfo, SliceInfoElem, Axis, Dimension};
use num_traits::Float;
use crate::tensor::Tensor;

/// Index types for tensor slicing
/// テンソルスライシング用のインデックスタイプ
#[derive(Debug, Clone)]
pub enum TensorIndex {
    /// Single index
    Single(isize),
    /// Range slice (start, end, step)
    Range(Option<isize>, Option<isize>, Option<isize>),
    /// Full slice (:)
    Full,
    /// New axis (None in Python)
    NewAxis,
    /// Ellipsis (...)
    Ellipsis,
}

impl<T: Float + 'static> Tensor<T> {
    /// Advanced indexing with multiple index types
    /// 複数のインデックスタイプによる高度なインデックス操作
    pub fn index(&self, indices: &[TensorIndex]) -> Result<Self, IndexError> {
        let shape = self.data.shape();
        let ndim = shape.len();
        
        // Process indices and handle ellipsis
        let processed_indices = self.process_indices(indices, ndim)?;
        
        // Convert to ndarray slice info
        let slice_info = self.build_slice_info(&processed_indices, shape)?;
        
        // Apply slicing
        let sliced_view = self.data.slice(slice_info);
        let sliced_data = sliced_view.to_owned();
        
        Ok(Tensor::new(sliced_data))
    }
    
    /// Get a single element by multi-dimensional index
    /// 多次元インデックスによる単一要素の取得
    pub fn get_item(&self, indices: &[usize]) -> Result<T, IndexError> {
        if indices.len() != self.data.ndim() {
            return Err(IndexError::DimensionMismatch {
                expected: self.data.ndim(),
                got: indices.len(),
            });
        }
        
        let shape = self.data.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(IndexError::IndexOutOfBounds {
                    index: idx as isize,
                    size: shape[i],
                    dim: i,
                });
            }
        }
        
        Ok(self.data[IxDyn(indices)])
    }
    
    /// Set a single element by multi-dimensional index
    /// 多次元インデックスによる単一要素の設定
    pub fn set_item(&mut self, indices: &[usize], value: T) -> Result<(), IndexError> {
        if indices.len() != self.data.ndim() {
            return Err(IndexError::DimensionMismatch {
                expected: self.data.ndim(),
                got: indices.len(),
            });
        }
        
        let shape = self.data.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(IndexError::IndexOutOfBounds {
                    index: idx as isize,
                    size: shape[i],
                    dim: i,
                });
            }
        }
        
        self.data[IxDyn(indices)] = value;
        Ok(())
    }
    
    /// Select elements along an axis using indices
    /// インデックスを使用して軸に沿って要素を選択
    pub fn select(&self, axis: usize, indices: &[usize]) -> Result<Self, IndexError> {
        if axis >= self.data.ndim() {
            return Err(IndexError::InvalidAxis {
                axis,
                ndim: self.data.ndim(),
            });
        }
        
        let shape = self.data.shape();
        for &idx in indices {
            if idx >= shape[axis] {
                return Err(IndexError::IndexOutOfBounds {
                    index: idx as isize,
                    size: shape[axis],
                    dim: axis,
                });
            }
        }
        
        // Create new shape
        let mut new_shape = shape.to_vec();
        new_shape[axis] = indices.len();
        
        // Create result array
        let mut result = ArrayD::zeros(IxDyn(&new_shape));
        
        // Copy selected elements
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            let mut source_indices = vec![0; shape.len()];
            let mut target_indices = vec![0; shape.len()];
            
            // Iterate through all combinations of other dimensions
            let other_dims: Vec<usize> = (0..shape.len())
                .filter(|&i| i != axis)
                .collect();
            
            self.copy_slice_recursive(
                &mut result,
                &self.data,
                &mut source_indices,
                &mut target_indices,
                &other_dims,
                0,
                axis,
                old_idx,
                new_idx,
            );
        }
        
        Ok(Tensor::new(result))
    }
    
    /// Gather elements along an axis using indices tensor
    /// インデックステンソルを使用して軸に沿って要素を収集
    pub fn gather(&self, axis: usize, indices: &Tensor<T>) -> Result<Self, IndexError> 
    where
        T: num_traits::cast::ToPrimitive,
    {
        if axis >= self.data.ndim() {
            return Err(IndexError::InvalidAxis {
                axis,
                ndim: self.data.ndim(),
            });
        }
        
        let indices_shape = indices.data.shape();
        let self_shape = self.data.shape();
        
        // Check that all dimensions except the gather axis match
        for (i, (&self_dim, &idx_dim)) in self_shape.iter().zip(indices_shape.iter()).enumerate() {
            if i != axis && self_dim != idx_dim {
                return Err(IndexError::ShapeMismatch {
                    expected: self_shape.to_vec(),
                    got: indices_shape.to_vec(),
                });
            }
        }
        
        // Create result with same shape as indices
        let mut result = ArrayD::zeros(IxDyn(indices_shape));
        
        // Gather elements
        for (result_idx, &index_val) in indices.data.indexed_iter() {
            let index = index_val.to_usize().ok_or(IndexError::InvalidIndex)?;
            
            if index >= self_shape[axis] {
                return Err(IndexError::IndexOutOfBounds {
                    index: index as isize,
                    size: self_shape[axis],
                    dim: axis,
                });
            }
            
            // Convert multi-dimensional index to source index
            let mut source_idx = result_idx.as_array_view().to_vec();
            source_idx[axis] = index;
            
            result[result_idx] = self.data[IxDyn(&source_idx)];
        }
        
        Ok(Tensor::new(result))
    }
    
    /// Scatter elements along an axis using indices
    /// インデックスを使用して軸に沿って要素を散布
    pub fn scatter(&mut self, axis: usize, indices: &Tensor<T>, src: &Tensor<T>) -> Result<(), IndexError>
    where
        T: num_traits::cast::ToPrimitive,
    {
        if axis >= self.data.ndim() {
            return Err(IndexError::InvalidAxis {
                axis,
                ndim: self.data.ndim(),
            });
        }
        
        let indices_shape = indices.data.shape();
        let src_shape = src.data.shape();
        
        if indices_shape != src_shape {
            return Err(IndexError::ShapeMismatch {
                expected: indices_shape.to_vec(),
                got: src_shape.to_vec(),
            });
        }
        
        // Scatter elements
        for ((idx_pos, &index_val), &src_val) in indices.data.indexed_iter().zip(src.data.iter()) {
            let index = index_val.to_usize().ok_or(IndexError::InvalidIndex)?;
            
            if index >= self.data.shape()[axis] {
                return Err(IndexError::IndexOutOfBounds {
                    index: index as isize,
                    size: self.data.shape()[axis],
                    dim: axis,
                });
            }
            
            // Convert to target index
            let mut target_idx = idx_pos.as_array_view().to_vec();
            target_idx[axis] = index;
            
            self.data[IxDyn(&target_idx)] = src_val;
        }
        
        Ok(())
    }
    
    /// Mask-based indexing
    /// マスクベースのインデックス操作
    pub fn masked_select(&self, mask: &Tensor<T>) -> Result<Self, IndexError>
    where
        T: PartialOrd,
    {
        if self.data.shape() != mask.data.shape() {
            return Err(IndexError::ShapeMismatch {
                expected: self.data.shape().to_vec(),
                got: mask.data.shape().to_vec(),
            });
        }
        
        let selected: Vec<T> = self.data.iter()
            .zip(mask.data.iter())
            .filter_map(|(&val, &mask_val)| {
                if mask_val > T::zero() {
                    Some(val)
                } else {
                    None
                }
            })
            .collect();
        
        let selected_len = selected.len();
        Ok(Tensor::from_vec(selected, vec![selected_len]))
    }
    
    /// Helper function to process indices and handle ellipsis
    /// インデックスを処理し省略記号を処理するヘルパー関数
    fn process_indices(&self, indices: &[TensorIndex], ndim: usize) -> Result<Vec<TensorIndex>, IndexError> {
        let mut processed = Vec::new();
        let mut ellipsis_found = false;
        let mut ellipsis_pos = 0;
        
        // Find ellipsis position
        for (i, idx) in indices.iter().enumerate() {
            if matches!(idx, TensorIndex::Ellipsis) {
                if ellipsis_found {
                    return Err(IndexError::MultipleEllipsis);
                }
                ellipsis_found = true;
                ellipsis_pos = i;
            }
        }
        
        if ellipsis_found {
            // Add indices before ellipsis
            processed.extend_from_slice(&indices[..ellipsis_pos]);
            
            // Calculate how many dimensions the ellipsis represents
            let non_ellipsis_count = indices.len() - 1; // -1 for the ellipsis itself
            let ellipsis_dims = if ndim >= non_ellipsis_count {
                ndim - non_ellipsis_count
            } else {
                0
            };
            
            // Add full slices for ellipsis dimensions
            for _ in 0..ellipsis_dims {
                processed.push(TensorIndex::Full);
            }
            
            // Add indices after ellipsis
            processed.extend_from_slice(&indices[ellipsis_pos + 1..]);
        } else {
            processed.extend_from_slice(indices);
        }
        
        Ok(processed)
    }
    
    /// Helper function to build slice info from processed indices
    /// 処理されたインデックスからスライス情報を構築するヘルパー関数
    fn build_slice_info(&self, indices: &[TensorIndex], shape: &[usize]) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>, IndexError> {
        let mut slice_items = Vec::new();
        
        for (i, index) in indices.iter().enumerate() {
            if i >= shape.len() {
                break;
            }
            
            let dim_size = shape[i] as isize;
            
            match index {
                TensorIndex::Single(idx) => {
                    let normalized_idx = if *idx < 0 {
                        dim_size + idx
                    } else {
                        *idx
                    };
                    
                    if normalized_idx < 0 || normalized_idx >= dim_size {
                        return Err(IndexError::IndexOutOfBounds {
                            index: *idx,
                            size: shape[i],
                            dim: i,
                        });
                    }
                    
                    slice_items.push(SliceInfoElem::Index(normalized_idx));
                }
                TensorIndex::Range(start, end, step) => {
                    let step = step.unwrap_or(1);
                    if step == 0 {
                        return Err(IndexError::ZeroStep);
                    }
                    
                    let start = start.map(|s| if s < 0 { dim_size + s } else { s }).unwrap_or(0);
                    let end = end.map(|e| if e < 0 { dim_size + e } else { e }).unwrap_or(dim_size);
                    
                    slice_items.push(SliceInfoElem::Slice {
                        start,
                        end: Some(end),
                        step,
                    });
                }
                TensorIndex::Full => {
                    slice_items.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
                TensorIndex::NewAxis => {
                    slice_items.push(SliceInfoElem::NewAxis);
                }
                TensorIndex::Ellipsis => {
                    // Should have been processed already
                    return Err(IndexError::UnprocessedEllipsis);
                }
            }
        }
        
        // Fill remaining dimensions with full slices
        while slice_items.len() < shape.len() {
            slice_items.push(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            });
        }
        
        SliceInfo::try_from(slice_items).map_err(|_| IndexError::InvalidSlice)
    }
    
    /// Recursive helper for copying slices
    /// スライスコピー用の再帰ヘルパー
    fn copy_slice_recursive(
        &self,
        target: &mut ArrayD<T>,
        source: &ArrayD<T>,
        source_indices: &mut Vec<usize>,
        target_indices: &mut Vec<usize>,
        other_dims: &[usize],
        dim_idx: usize,
        axis: usize,
        source_axis_idx: usize,
        target_axis_idx: usize,
    ) {
        if dim_idx >= other_dims.len() {
            // Base case: copy the element
            source_indices[axis] = source_axis_idx;
            target_indices[axis] = target_axis_idx;
            
            target[IxDyn(target_indices)] = source[IxDyn(source_indices)];
            return;
        }
        
        let current_dim = other_dims[dim_idx];
        let dim_size = source.shape()[current_dim];
        
        for i in 0..dim_size {
            source_indices[current_dim] = i;
            target_indices[current_dim] = i;
            
            self.copy_slice_recursive(
                target,
                source,
                source_indices,
                target_indices,
                other_dims,
                dim_idx + 1,
                axis,
                source_axis_idx,
                target_axis_idx,
            );
        }
    }
}

/// Indexing error types
/// インデックスエラータイプ
#[derive(Debug, Clone)]
pub enum IndexError {
    /// Index out of bounds
    IndexOutOfBounds { index: isize, size: usize, dim: usize },
    /// Invalid axis
    InvalidAxis { axis: usize, ndim: usize },
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Shape mismatch
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Multiple ellipsis in indices
    MultipleEllipsis,
    /// Unprocessed ellipsis
    UnprocessedEllipsis,
    /// Zero step in range
    ZeroStep,
    /// Invalid slice
    InvalidSlice,
    /// Invalid index value
    InvalidIndex,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::IndexOutOfBounds { index, size, dim } => {
                write!(f, "Index {} out of bounds for dimension {} with size {}", index, dim, size)
            }
            IndexError::InvalidAxis { axis, ndim } => {
                write!(f, "Invalid axis {} for tensor with {} dimensions", axis, ndim)
            }
            IndexError::DimensionMismatch { expected, got } => {
                write!(f, "Expected {} dimensions, got {}", expected, got)
            }
            IndexError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            IndexError::MultipleEllipsis => {
                write!(f, "Multiple ellipsis (...) not allowed in indexing")
            }
            IndexError::UnprocessedEllipsis => {
                write!(f, "Unprocessed ellipsis in indexing")
            }
            IndexError::ZeroStep => {
                write!(f, "Step cannot be zero in range indexing")
            }
            IndexError::InvalidSlice => {
                write!(f, "Invalid slice specification")
            }
            IndexError::InvalidIndex => {
                write!(f, "Invalid index value")
            }
        }
    }
}

impl std::error::Error for IndexError {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_get_set_item() {
        let mut tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Test get_item
        assert_abs_diff_eq!(tensor.get_item(&[0, 0]).unwrap(), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tensor.get_item(&[1, 1]).unwrap(), 4.0, epsilon = 1e-6);
        
        // Test set_item
        tensor.set_item(&[0, 1], 10.0).unwrap();
        assert_abs_diff_eq!(tensor.get_item(&[0, 1]).unwrap(), 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_select() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        // Select columns 0 and 2
        let selected = tensor.select(1, &[0, 2]).unwrap();
        assert_eq!(selected.shape(), &[2, 2]);
        
        let expected = vec![1.0, 3.0, 4.0, 6.0];
        assert_eq!(selected.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_basic_indexing() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        // Test single index
        let indexed = tensor.index(&[TensorIndex::Single(0)]).unwrap();
        assert_eq!(indexed.shape(), &[3]);
        assert_eq!(indexed.data.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        
        // Test range
        let ranged = tensor.index(&[TensorIndex::Range(Some(0), Some(1), None)]).unwrap();
        assert_eq!(ranged.shape(), &[1, 3]);
    }

    #[test]
    fn test_masked_select() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], vec![4]);
        
        let selected = tensor.masked_select(&mask).unwrap();
        assert_eq!(selected.data.as_slice().unwrap(), &[1.0, 3.0]);
    }

    #[test]
    fn test_index_out_of_bounds() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let result = tensor.get_item(&[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.get_item(&[0]); // Should need 2 indices
        assert!(result.is_err());
    }
}
