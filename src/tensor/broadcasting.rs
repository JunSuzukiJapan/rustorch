/// Broadcasting operations for tensors
/// テンソルのブロードキャスティング操作

use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};

impl<T: Float + 'static> Tensor<T> {
    /// Broadcast two tensors to compatible shapes
    /// 2つのテンソルを互換性のある形状にブロードキャスト
    pub fn broadcast_with(&self, other: &Self) -> RusTorchResult<(Self, Self)> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        let broadcast_shape = compute_broadcast_shape(self_shape, other_shape)?;
        
        let broadcasted_self = self.broadcast_to(&broadcast_shape)?;
        let broadcasted_other = other.broadcast_to(&broadcast_shape)?;
        
        Ok((broadcasted_self, broadcasted_other))
    }
    
    /// Broadcast tensor to a specific shape
    /// テンソルを特定の形状にブロードキャスト
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, RusTorchError> {
        let current_shape = self.data.shape();
        
        // Check if broadcasting is possible
        if !can_broadcast(current_shape, target_shape) {
            return Err(RusTorchError::IncompatibleShapes {
                from: current_shape.to_vec(),
                to: target_shape.to_vec(),
            });
        }
        
        // If shapes are already the same, return clone
        if current_shape == target_shape {
            return Ok(self.clone());
        }
        
        // Perform broadcasting
        let broadcasted_data = broadcast_array(&self.data, target_shape)?;
        Ok(Tensor::new(broadcasted_data))
    }
    
    /// Check if this tensor can be broadcasted with another
    /// このテンソルが他のテンソルとブロードキャスト可能かチェック
    pub fn can_broadcast_with(&self, other: &Self) -> bool {
        can_broadcast(self.data.shape(), other.data.shape())
    }
    
    /// Expand tensor dimensions by adding singleton dimensions
    /// 単一次元を追加してテンソルの次元を拡張
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, RusTorchError> {
        let mut new_shape = self.data.shape().to_vec();
        
        if dim > new_shape.len() {
            return Err(RusTorchError::InvalidDimension {
                dim,
                max_dim: new_shape.len(),
            });
        }
        
        new_shape.insert(dim, 1);
        
        let reshaped_data = self.data.clone().into_shape_with_order(new_shape)
            .map_err(|_| RusTorchError::ReshapeError)?;
        
        Ok(Tensor::new(reshaped_data))
    }
    
    /// Remove singleton dimensions
    /// 単一次元を削除
    pub fn squeeze(&self) -> Self {
        let current_shape = self.data.shape();
        let new_shape: Vec<usize> = current_shape.iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();
        
        // If all dimensions were 1, keep at least one dimension
        let final_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
        
        let reshaped_data = self.data.clone().into_shape_with_order(final_shape)
            .expect("Squeeze should always be valid");
        
        Tensor::new(reshaped_data)
    }
    
    /// Remove a specific singleton dimension
    /// 特定の単一次元を削除
    pub fn squeeze_dim(&self, dim: usize) -> Result<Self, RusTorchError> {
        let current_shape = self.data.shape();
        
        if dim >= current_shape.len() {
            return Err(RusTorchError::InvalidDimension {
                dim,
                max_dim: current_shape.len() - 1,
            });
        }
        
        if current_shape[dim] != 1 {
            return Err(RusTorchError::NotSingletonDimension { dim, size: current_shape[dim] });
        }
        
        let mut new_shape = current_shape.to_vec();
        new_shape.remove(dim);
        
        // Ensure at least one dimension remains
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        
        let reshaped_data = self.data.clone().into_shape_with_order(new_shape)
            .map_err(|_| RusTorchError::ReshapeError)?;
        
        Ok(Tensor::new(reshaped_data))
    }
    
    /// Repeat tensor along specified dimensions
    /// 指定された次元に沿ってテンソルを繰り返し
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self, RusTorchError> {
        let current_shape = self.data.shape();
        
        if repeats.len() != current_shape.len() {
            return Err(RusTorchError::MismatchedDimensions {
                expected: current_shape.len(),
                got: repeats.len(),
            });
        }
        
        let new_shape: Vec<usize> = current_shape.iter()
            .zip(repeats.iter())
            .map(|(&dim, &rep)| dim * rep)
            .collect();
        
        let mut result_data = ArrayD::zeros(IxDyn(&new_shape));
        
        // Fill the result array by repeating the original data
        let _original_strides = self.data.strides();
        let _result_strides = result_data.strides();
        
        // This is a simplified implementation - a full implementation would be more complex
        for (i, &val) in self.data.iter().enumerate() {
            // Calculate original indices
            let mut original_indices = vec![0; current_shape.len()];
            let mut temp_i = i;
            for (dim_idx, &dim_size) in current_shape.iter().enumerate().rev() {
                original_indices[dim_idx] = temp_i % dim_size;
                temp_i /= dim_size;
            }
            
            // Repeat to all corresponding positions in result
            for rep_combo in 0..repeats.iter().product() {
                let mut rep_indices = vec![0; repeats.len()];
                let mut temp_rep = rep_combo;
                for (dim_idx, &rep_count) in repeats.iter().enumerate().rev() {
                    rep_indices[dim_idx] = temp_rep % rep_count;
                    temp_rep /= rep_count;
                }
                
                let mut result_indices = vec![0; current_shape.len()];
                for (dim_idx, (&orig_idx, (&rep_idx, (&orig_size, &_rep_count)))) in 
                    original_indices.iter()
                        .zip(rep_indices.iter()
                            .zip(current_shape.iter()
                                .zip(repeats.iter())))
                        .enumerate() {
                    result_indices[dim_idx] = rep_idx * orig_size + orig_idx;
                }
                
                if let Some(result_elem) = result_data.get_mut(IxDyn(&result_indices)) {
                    *result_elem = val;
                }
            }
        }
        
        Ok(Tensor::new(result_data))
    }
}

/// Compute the broadcast shape for two shapes
/// 2つの形状のブロードキャスト形状を計算
fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, RusTorchError> {
    let max_len = shape1.len().max(shape2.len());
    let mut result_shape = Vec::with_capacity(max_len);
    
    // Pad shorter shape with 1s on the left
    let padded_shape1 = pad_shape_left(shape1, max_len);
    let padded_shape2 = pad_shape_left(shape2, max_len);
    
    for (dim1, dim2) in padded_shape1.iter().zip(padded_shape2.iter()) {
        match (dim1, dim2) {
            (1, d) | (d, 1) => result_shape.push(*d),
            (d1, d2) if d1 == d2 => result_shape.push(*d1),
            (_d1, _d2) => return Err(RusTorchError::IncompatibleShapes {
                from: shape1.to_vec(),
                to: shape2.to_vec(),
            }),
        }
    }
    
    Ok(result_shape)
}

/// Check if two shapes can be broadcasted together
/// 2つの形状がブロードキャスト可能かチェック
fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
    compute_broadcast_shape(shape1, shape2).is_ok()
}

/// Pad shape with 1s on the left to reach target length
/// 左側に1を埋めて目標長に達するまで形状をパディング
fn pad_shape_left(shape: &[usize], target_len: usize) -> Vec<usize> {
    let mut padded = vec![1; target_len];
    let start_idx = target_len - shape.len();
    padded[start_idx..].copy_from_slice(shape);
    padded
}

/// Broadcast an array to a target shape
/// 配列を目標形状にブロードキャスト
fn broadcast_array<T: Float>(array: &ArrayD<T>, target_shape: &[usize]) -> Result<ArrayD<T>, RusTorchError> {
    let current_shape = array.shape();
    
    if current_shape == target_shape {
        return Ok(array.clone());
    }
    
    // Use ndarray's broadcast functionality
    let broadcasted = array.broadcast(IxDyn(target_shape))
        .ok_or_else(|| RusTorchError::IncompatibleShapes {
            from: current_shape.to_vec(),
            to: target_shape.to_vec(),
        })?;
    
    // Convert broadcasted view to owned array
    Ok(broadcasted.to_owned())
}

// RusTorchError enum removed - now using unified RusTorchError system
// RusTorchErrorエナム削除 - 統一RusTorchErrorシステムを使用

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_compatible_shapes() {
        let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::from_vec(vec![10.0f32], vec![1]);
        
        assert!(tensor1.can_broadcast_with(&tensor2));
        
        let (broadcasted1, broadcasted2) = tensor1.broadcast_with(&tensor2).unwrap();
        assert_eq!(broadcasted1.shape(), &[3]);
        assert_eq!(broadcasted2.shape(), &[3]);
        assert_eq!(broadcasted2.data.as_slice().unwrap(), &[10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes() {
        let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        
        assert!(!tensor1.can_broadcast_with(&tensor2));
        assert!(tensor1.broadcast_with(&tensor2).is_err());
    }

    #[test]
    fn test_broadcast_to_specific_shape() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let broadcasted = tensor.broadcast_to(&[3, 2]).unwrap();
        
        assert_eq!(broadcasted.shape(), &[3, 2]);
        // Should repeat the [1, 2] pattern 3 times
        let expected = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        assert_eq!(broadcasted.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        
        // Add dimension at the beginning
        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
        
        // Add dimension at the end
        let unsqueezed = tensor.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]);
        
        // Remove first singleton dimension
        let squeezed = tensor.squeeze_dim(0).unwrap();
        assert_eq!(squeezed.shape(), &[3, 1]);
        
        // Remove last singleton dimension
        let squeezed = tensor.squeeze_dim(2).unwrap();
        assert_eq!(squeezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_squeeze_non_singleton_error() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let result = tensor.squeeze_dim(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let repeated = tensor.repeat(&[3]).unwrap();
        
        assert_eq!(repeated.shape(), &[6]);
        assert_eq!(repeated.data.as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_compute_broadcast_shape() {
        // Compatible shapes
        assert_eq!(compute_broadcast_shape(&[3], &[1]).unwrap(), vec![3]);
        assert_eq!(compute_broadcast_shape(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(compute_broadcast_shape(&[2, 1, 4], &[3, 1]).unwrap(), vec![2, 3, 4]);
        
        // Incompatible shapes
        assert!(compute_broadcast_shape(&[3], &[2]).is_err());
        assert!(compute_broadcast_shape(&[2, 3], &[3, 2]).is_err());
    }

    #[test]
    fn test_pad_shape_left() {
        assert_eq!(pad_shape_left(&[3], 3), vec![1, 1, 3]);
        assert_eq!(pad_shape_left(&[2, 3], 4), vec![1, 1, 2, 3]);
        assert_eq!(pad_shape_left(&[1, 2, 3], 3), vec![1, 2, 3]);
    }
}
