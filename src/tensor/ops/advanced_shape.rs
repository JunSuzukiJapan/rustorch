//! Advanced shape operations for tensors
//! テンソル用高度形状操作

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + Clone> Tensor<T> {
    // Expand and repeat operations
    // 拡張と反復操作

    /// Expand tensor to a larger size by repeating dimensions
    /// 次元を繰り返してテンソルを大きなサイズに拡張
    pub fn expand(&self, target_shape: &[usize]) -> RusTorchResult<Self> {
        if target_shape.len() < self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "expand".to_string(),
                message: format!(
                    "Target shape must have at least {} dimensions, got {}",
                    self.shape().len(),
                    target_shape.len()
                ),
            });
        }

        let self_shape = self.shape();
        let ndim_diff = target_shape.len() - self_shape.len();

        // Check if expansion is valid
        for (i, (&target_dim, &self_dim)) in target_shape
            .iter()
            .skip(ndim_diff)
            .zip(self_shape.iter())
            .enumerate()
        {
            if self_dim != 1 && self_dim != target_dim {
                return Err(RusTorchError::InvalidOperation {
                    operation: "expand".to_string(),
                    message: format!(
                        "Cannot expand dimension {} from {} to {} (must be 1 or equal)",
                        i + ndim_diff,
                        self_dim,
                        target_dim
                    ),
                });
            }
        }

        // Create expanded tensor
        let mut expanded_data = Vec::new();
        let total_elements: usize = target_shape.iter().product();
        expanded_data.reserve(total_elements);

        // Generate all indices for the target shape
        self.expand_recursive(&mut expanded_data, target_shape, &vec![0; target_shape.len()], 0)?;

        Ok(Tensor::from_vec(expanded_data, target_shape.to_vec()))
    }

    fn expand_recursive(
        &self,
        output: &mut Vec<T>,
        target_shape: &[usize],
        current_indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == target_shape.len() {
            // Map target indices to source indices
            let self_shape = self.shape();
            let ndim_diff = target_shape.len() - self_shape.len();
            
            let mut source_indices = Vec::new();
            for (i, &target_idx) in current_indices.iter().enumerate() {
                if i < ndim_diff {
                    // New dimension, skip
                    continue;
                } else {
                    let self_dim_idx = i - ndim_diff;
                    if self_dim_idx < self_shape.len() {
                        let source_idx = if self_shape[self_dim_idx] == 1 { 0 } else { target_idx };
                        source_indices.push(source_idx);
                    }
                }
            }

            // Get value from source tensor
            if let Some(value) = self.data.get(IxDyn(&source_indices)) {
                output.push(*value);
            } else {
                output.push(T::zero());
            }
            return Ok(());
        }

        for i in 0..target_shape[dim] {
            let mut new_indices = current_indices.to_vec();
            new_indices[dim] = i;
            self.expand_recursive(output, target_shape, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    /// Repeat tensor elements along specified dimensions - advanced version
    /// 指定された次元に沿ってテンソル要素を反復 - 高度版
    pub fn repeat_advanced(&self, repeats: &[usize]) -> RusTorchResult<Self> {
        if repeats.len() != self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "repeat".to_string(),
                message: format!(
                    "Number of repeats ({}) must match number of dimensions ({})",
                    repeats.len(),
                    self.shape().len()
                ),
            });
        }

        let self_shape = self.shape();
        let mut target_shape = Vec::new();
        for (i, &repeat_count) in repeats.iter().enumerate() {
            target_shape.push(self_shape[i] * repeat_count);
        }

        let mut result_data = Vec::new();
        let total_elements: usize = target_shape.iter().product();
        result_data.reserve(total_elements);

        self.repeat_recursive(&mut result_data, &target_shape, repeats, &vec![0; target_shape.len()], 0)?;

        Ok(Tensor::from_vec(result_data, target_shape))
    }

    fn repeat_recursive(
        &self,
        output: &mut Vec<T>,
        target_shape: &[usize],
        repeats: &[usize],
        current_indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == target_shape.len() {
            // Map target indices to source indices
            let self_shape = self.shape();
            let mut source_indices = Vec::new();
            
            for (i, &target_idx) in current_indices.iter().enumerate() {
                let source_idx = target_idx % self_shape[i];
                source_indices.push(source_idx);
            }

            if let Some(value) = self.data.get(IxDyn(&source_indices)) {
                output.push(*value);
            } else {
                output.push(T::zero());
            }
            return Ok(());
        }

        for i in 0..target_shape[dim] {
            let mut new_indices = current_indices.to_vec();
            new_indices[dim] = i;
            self.repeat_recursive(output, target_shape, repeats, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    /// Tile tensor by repeating it along each dimension
    /// 各次元に沿って繰り返してテンソルをタイル化
    pub fn tile(&self, reps: &[usize]) -> RusTorchResult<Self> {
        if reps.is_empty() {
            return Ok(self.clone());
        }

        let self_shape = self.shape();
        let max_ndim = std::cmp::max(self_shape.len(), reps.len());
        
        // Pad shapes with 1s if needed
        let mut padded_self_shape = vec![1; max_ndim];
        let mut padded_reps = vec![1; max_ndim];
        
        // Copy original shape to the end
        for (i, &dim) in self_shape.iter().enumerate() {
            padded_self_shape[max_ndim - self_shape.len() + i] = dim;
        }
        
        // Copy reps to the end
        for (i, &rep) in reps.iter().enumerate() {
            padded_reps[max_ndim - reps.len() + i] = rep;
        }

        // First reshape the tensor if needed
        let current_tensor = if padded_self_shape != self_shape.to_vec() {
            self.reshape(&padded_self_shape)?
        } else {
            self.clone()
        };

        // Apply repeat operation
        current_tensor.repeat_advanced(&padded_reps)
    }

    // Multiple unsqueeze and squeeze operations
    // 複数のunsqueezeとsqueeze操作

    /// Add multiple singleton dimensions at specified axes
    /// 指定された軸に複数の単一次元を追加
    pub fn unsqueeze_multiple(&self, axes: &[usize]) -> RusTorchResult<Self> {
        let mut result = self.clone();
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        sorted_axes.reverse(); // Process in reverse order to maintain indices

        for &axis in &sorted_axes {
            result = result.unsqueeze(axis)?;
        }

        Ok(result)
    }

    /// Remove all singleton dimensions
    /// すべての単一次元を削除
    pub fn squeeze_all(&self) -> Self {
        let current_shape = self.shape();
        let new_shape: Vec<usize> = current_shape
            .iter()
            .cloned()
            .filter(|&dim| dim != 1)
            .collect();

        if new_shape.is_empty() {
            // If all dimensions are 1, keep one dimension
            return Tensor::from_vec(self.data.iter().cloned().collect(), vec![1]);
        }

        Tensor::from_vec(self.data.iter().cloned().collect(), new_shape)
    }

    // Permutation and dimension movement
    // 順列と次元移動

    /// Permute the dimensions of the tensor
    /// テンソルの次元を順列
    pub fn permute(&self, dims: &[usize]) -> RusTorchResult<Self> {
        let self_shape = self.shape();
        
        if dims.len() != self_shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "permute".to_string(),
                message: format!(
                    "Number of dimensions in permutation ({}) must match tensor dimensions ({})",
                    dims.len(),
                    self_shape.len()
                ),
            });
        }

        // Check that dims contains each index exactly once
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_unstable();
        let expected: Vec<usize> = (0..self_shape.len()).collect();
        if sorted_dims != expected {
            return Err(RusTorchError::InvalidOperation {
                operation: "permute".to_string(),
                message: "Permutation must contain each dimension index exactly once".to_string(),
            });
        }

        // Create new shape
        let new_shape: Vec<usize> = dims.iter().map(|&i| self_shape[i]).collect();
        let mut result_data = Vec::with_capacity(self.numel());

        // Generate all possible indices for the result tensor
        self.permute_recursive(&mut result_data, &new_shape, dims, &vec![0; new_shape.len()], 0)?;

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    fn permute_recursive(
        &self,
        output: &mut Vec<T>,
        target_shape: &[usize],
        dims: &[usize],
        current_indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == target_shape.len() {
            // Map result indices to source indices
            let mut source_indices = vec![0; dims.len()];
            for (result_dim, &source_dim) in dims.iter().enumerate() {
                source_indices[source_dim] = current_indices[result_dim];
            }

            if let Some(value) = self.data.get(IxDyn(&source_indices)) {
                output.push(*value);
            } else {
                output.push(T::zero());
            }
            return Ok(());
        }

        for i in 0..target_shape[dim] {
            let mut new_indices = current_indices.to_vec();
            new_indices[dim] = i;
            self.permute_recursive(output, target_shape, dims, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    /// Move dimension from source position to destination position
    /// 次元をソース位置からデスティネーション位置に移動
    pub fn movedim(&self, source: usize, destination: usize) -> RusTorchResult<Self> {
        let ndim = self.shape().len();
        
        if source >= ndim || destination >= ndim {
            return Err(RusTorchError::InvalidOperation {
                operation: "movedim".to_string(),
                message: format!("Dimension out of range: source={}, destination={}, ndim={}", source, destination, ndim),
            });
        }

        if source == destination {
            return Ok(self.clone());
        }

        // Create permutation array
        let mut dims: Vec<usize> = (0..ndim).collect();
        dims.remove(source);
        dims.insert(destination, source);

        self.permute(&dims)
    }

    // Flattening and unflattening operations
    // フラット化と非フラット化操作

    /// Flatten tensor starting from a specific dimension
    /// 特定の次元から開始してテンソルをフラット化
    pub fn flatten_from(&self, start_dim: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        if start_dim >= shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "flatten_from".to_string(),
                message: format!("start_dim {} is out of range for tensor with {} dimensions", start_dim, shape.len()),
            });
        }

        if start_dim == shape.len() - 1 {
            return Ok(self.clone());
        }

        let mut new_shape = shape[..start_dim].to_vec();
        let flattened_size: usize = shape[start_dim..].iter().product();
        new_shape.push(flattened_size);

        self.reshape(&new_shape)
    }

    /// Unflatten a dimension into multiple dimensions
    /// 1つの次元を複数の次元に非フラット化
    pub fn unflatten(&self, dim: usize, unflattened_size: &[usize]) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        if dim >= shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "unflatten".to_string(),
                message: format!("Dimension {} is out of range for tensor with {} dimensions", dim, shape.len()),
            });
        }

        let expected_size: usize = unflattened_size.iter().product();
        if shape[dim] != expected_size {
            return Err(RusTorchError::InvalidOperation {
                operation: "unflatten".to_string(),
                message: format!(
                    "Cannot unflatten dimension {} of size {} into sizes {:?} (product = {})",
                    dim, shape[dim], unflattened_size, expected_size
                ),
            });
        }

        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&shape[..dim]);
        new_shape.extend_from_slice(unflattened_size);
        new_shape.extend_from_slice(&shape[dim + 1..]);

        self.reshape(&new_shape)
    }

    // Advanced dimension operations
    // 高度次元操作

    /// Insert new axis at specified position with broadcasting
    /// 指定位置にブロードキャスティングで新しい軸を挿入
    pub fn expand_dims(&self, axis: usize) -> RusTorchResult<Self> {
        self.unsqueeze(axis)
    }

    /// Swap two dimensions
    /// 2つの次元を交換
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> RusTorchResult<Self> {
        let ndim = self.shape().len();
        
        if axis1 >= ndim || axis2 >= ndim {
            return Err(RusTorchError::InvalidOperation {
                operation: "swapaxes".to_string(),
                message: format!("Axes out of range: axis1={}, axis2={}, ndim={}", axis1, axis2, ndim),
            });
        }

        if axis1 == axis2 {
            return Ok(self.clone());
        }

        let mut dims: Vec<usize> = (0..ndim).collect();
        dims.swap(axis1, axis2);

        self.permute(&dims)
    }

    /// Reverse the order of elements along specified axis
    /// 指定軸に沿って要素の順序を逆転
    pub fn flip(&self, axis: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        if axis >= shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "flip".to_string(),
                message: format!("Axis {} is out of range for tensor with {} dimensions", axis, shape.len()),
            });
        }

        let mut result_data = Vec::with_capacity(self.numel());
        self.flip_recursive(&mut result_data, shape, axis, &vec![0; shape.len()], 0)?;

        Ok(Tensor::from_vec(result_data, shape.to_vec()))
    }

    fn flip_recursive(
        &self,
        output: &mut Vec<T>,
        shape: &[usize],
        flip_axis: usize,
        current_indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == shape.len() {
            // Map flipped indices to original indices
            let mut source_indices = current_indices.to_vec();
            source_indices[flip_axis] = shape[flip_axis] - 1 - current_indices[flip_axis];

            if let Some(value) = self.data.get(IxDyn(&source_indices)) {
                output.push(*value);
            } else {
                output.push(T::zero());
            }
            return Ok(());
        }

        for i in 0..shape[dim] {
            let mut new_indices = current_indices.to_vec();
            new_indices[dim] = i;
            self.flip_recursive(output, shape, flip_axis, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    /// Roll elements along axis
    /// 軸に沿って要素をロール
    pub fn roll(&self, shifts: isize, axis: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        if axis >= shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "roll".to_string(),
                message: format!("Axis {} is out of range for tensor with {} dimensions", axis, shape.len()),
            });
        }

        let axis_size = shape[axis] as isize;
        let normalized_shifts = ((shifts % axis_size) + axis_size) % axis_size;

        if normalized_shifts == 0 {
            return Ok(self.clone());
        }

        let mut result_data = Vec::with_capacity(self.numel());
        self.roll_recursive(&mut result_data, shape, axis, normalized_shifts as usize, &vec![0; shape.len()], 0)?;

        Ok(Tensor::from_vec(result_data, shape.to_vec()))
    }

    fn roll_recursive(
        &self,
        output: &mut Vec<T>,
        shape: &[usize],
        roll_axis: usize,
        shifts: usize,
        current_indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == shape.len() {
            // Map rolled indices to original indices
            let mut source_indices = current_indices.to_vec();
            let axis_size = shape[roll_axis];
            source_indices[roll_axis] = (current_indices[roll_axis] + axis_size - shifts) % axis_size;

            if let Some(value) = self.data.get(IxDyn(&source_indices)) {
                output.push(*value);
            } else {
                output.push(T::zero());
            }
            return Ok(());
        }

        for i in 0..shape[dim] {
            let mut new_indices = current_indices.to_vec();
            new_indices[dim] = i;
            self.roll_recursive(output, shape, roll_axis, shifts, &new_indices, dim + 1)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let expanded = tensor.expand(&[2, 3]).unwrap();
        
        assert_eq!(expanded.shape(), &[2, 3]);
        assert_eq!(expanded.as_slice().unwrap(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_repeat() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let repeated = tensor.repeat(&[3]).unwrap();
        
        assert_eq!(repeated.shape(), &[6]);
        assert_eq!(repeated.as_slice().unwrap(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_tile() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let tiled = tensor.tile(&[3]).unwrap();
        
        assert_eq!(tiled.shape(), &[6]);
        assert_eq!(tiled.as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let squeezed = tensor.squeeze_all();
        
        assert_eq!(squeezed.shape(), &[3]);
        assert_eq!(squeezed.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_permute() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let permuted = tensor.permute(&[1, 0]).unwrap();
        
        assert_eq!(permuted.shape(), &[3, 2]);
        // Original: [[1, 2, 3], [4, 5, 6]]
        // Permuted: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(permuted.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_movedim() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let moved = tensor.movedim(0, 1).unwrap();
        
        assert_eq!(moved.shape(), &[3, 2]);
        assert_eq!(moved.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_flatten_from() {
        let tensor = Tensor::from_vec((1..=24).map(|x| x as f64).collect(), vec![2, 3, 4]);
        let flattened = tensor.flatten_from(1).unwrap();
        
        assert_eq!(flattened.shape(), &[2, 12]);
    }

    #[test]
    fn test_unflatten() {
        let tensor = Tensor::from_vec((1..=12).map(|x| x as f64).collect(), vec![12]);
        let unflattened = tensor.unflatten(0, &[3, 4]).unwrap();
        
        assert_eq!(unflattened.shape(), &[3, 4]);
    }

    #[test]
    fn test_swapaxes() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let swapped = tensor.swapaxes(0, 1).unwrap();
        
        assert_eq!(swapped.shape(), &[3, 2]);
        assert_eq!(swapped.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_flip() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let flipped = tensor.flip(0).unwrap();
        
        assert_eq!(flipped.shape(), &[4]);
        assert_eq!(flipped.as_slice().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_roll() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let rolled = tensor.roll(2, 0).unwrap();
        
        assert_eq!(rolled.shape(), &[4]);
        assert_eq!(rolled.as_slice().unwrap(), &[3.0, 4.0, 1.0, 2.0]);
        
        // Test negative roll
        let rolled_neg = tensor.roll(-1, 0).unwrap();
        assert_eq!(rolled_neg.as_slice().unwrap(), &[2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_unsqueeze_multiple() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let unsqueezed = tensor.unsqueeze_multiple(&[0, 2]).unwrap();
        
        assert_eq!(unsqueezed.shape(), &[1, 2, 1]);
    }
}