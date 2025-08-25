//! Tensor creation operations
//! テンソル作成演算
//!
//! This module provides operations for creating tensors with specific patterns,
//! random distributions, and initialization methods.
//! このモジュールは特定パターン、ランダム分布、初期化メソッドでテンソルを作成する操作を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Create tensor with random values from normal distribution
    /// 正規分布からランダム値でテンソルを作成
    pub fn randn(shape: &[usize]) -> Tensor<T>
    where
        T: From<f32>,
    {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.sample::<f32, _>(StandardNormal)))
            .collect();
            
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create tensor with random values from uniform distribution [0, 1)
    /// 一様分布[0, 1)からランダム値でテンソルを作成
    pub fn rand(shape: &[usize]) -> Tensor<T>
    where
        T: From<f32>,
    {
        use rand::prelude::*;
        
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.gen::<f32>()))
            .collect();
            
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create tensor with random values from uniform distribution [low, high)
    /// 一様分布[low, high)からランダム値でテンソルを作成
    pub fn uniform(shape: &[usize], low: T, high: T) -> Tensor<T>
    where
        T: From<f32>,
    {
        use rand::prelude::*;
        use rand_distr::Uniform;
        
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        
        // Convert to f32 for uniform distribution, then convert back
        let low_f32 = if let Some(slice) = [low].as_slice() {
            // This is a workaround for type conversion
            unsafe { std::mem::transmute_copy(&slice[0]) }
        } else {
            0.0f32
        };
        let high_f32 = unsafe { std::mem::transmute_copy(&high) };
        
        let dist = Uniform::new(low_f32, high_f32);
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.sample(dist)))
            .collect();
            
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create tensor with random integers
    /// ランダム整数でテンソルを作成
    pub fn randint(shape: &[usize], low: i32, high: i32) -> Tensor<T>
    where
        T: From<i32>,
    {
        use rand::prelude::*;
        use rand_distr::Uniform;
        
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let dist = Uniform::new(low, high);
        
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<i32>>::from(rng.sample(dist)))
            .collect();
            
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create identity matrix
    /// 単位行列を作成
    pub fn eye(n: usize) -> Tensor<T> {
        let mut data = vec![T::zero(); n * n];
        
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        
        Tensor::from_vec(data, vec![n, n])
    }

    /// Create tensor with values from arithmetic progression
    /// 等差数列の値でテンソルを作成
    pub fn arange(start: T, stop: T, step: T) -> Result<Tensor<T>, String> {
        if step == T::zero() {
            return Err("Step cannot be zero".to_string());
        }
        
        let mut values = Vec::new();
        let mut current = start;
        
        if step > T::zero() {
            while current < stop {
                values.push(current);
                current = current + step;
            }
        } else {
            while current > stop {
                values.push(current);
                current = current + step;
            }
        }
        
        if values.is_empty() {
            return Err("Invalid range specification".to_string());
        }
        
        let values_len = values.len();
        Ok(Tensor::from_vec(values, vec![values_len]))
    }

    /// Create tensor with linearly spaced values
    /// 線形等間隔値でテンソルを作成
    pub fn linspace(start: T, stop: T, num: usize) -> Result<Tensor<T>, String> {
        if num == 0 {
            return Err("Number of samples must be positive".to_string());
        }
        
        if num == 1 {
            return Ok(Tensor::from_vec(vec![start], vec![1]));
        }
        
        let step = (stop - start) / T::from(num - 1).unwrap();
        let mut values = Vec::with_capacity(num);
        
        for i in 0..num {
            let value = start + step * T::from(i).unwrap();
            values.push(value);
        }
        
        Ok(Tensor::from_vec(values, vec![num]))
    }

    /// Create tensor with logarithmically spaced values
    /// 対数等間隔値でテンソルを作成
    pub fn logspace(start: T, stop: T, num: usize, base: T) -> Result<Tensor<T>, String> {
        let linear_space = Self::linspace(start, stop, num)?;
        let log_data: Vec<T> = linear_space.as_slice().unwrap()
            .iter()
            .map(|&x| base.powf(x))
            .collect();
        
        Ok(Tensor::from_vec(log_data, vec![num]))
    }

    /// Create diagonal tensor from 1D tensor
    /// 1Dテンソルから対角テンソルを作成
    pub fn diag(diagonal: &Tensor<T>, k: i32) -> Result<Tensor<T>, String> {
        if diagonal.ndim() != 1 {
            return Err("Input must be 1-dimensional".to_string());
        }
        
        let diag_len = diagonal.numel();
        let matrix_size = diag_len + k.abs() as usize;
        let mut data = vec![T::zero(); matrix_size * matrix_size];
        
        let diag_data = diagonal.as_slice().unwrap();
        
        for (i, &value) in diag_data.iter().enumerate() {
            let row = if k >= 0 { i } else { i + (-k) as usize };
            let col = if k >= 0 { i + k as usize } else { i };
            
            if row < matrix_size && col < matrix_size {
                data[row * matrix_size + col] = value;
            }
        }
        
        Ok(Tensor::from_vec(data, vec![matrix_size, matrix_size]))
    }

    /// Create tensor by tiling (repeating) another tensor
    /// 他のテンソルをタイル化（繰り返し）してテンソルを作成
    pub fn tile(tensor: &Tensor<T>, reps: &[usize]) -> Result<Tensor<T>, String> {
        let original_shape = tensor.shape();
        
        if reps.len() > original_shape.len() {
            return Err("Too many repetitions specified".to_string());
        }
        
        // Pad reps with 1s if necessary
        let mut full_reps = vec![1; original_shape.len()];
        let start_idx = original_shape.len() - reps.len();
        full_reps[start_idx..].copy_from_slice(reps);
        
        // Calculate new shape
        let new_shape: Vec<usize> = original_shape.iter()
            .zip(full_reps.iter())
            .map(|(&dim, &rep)| dim * rep)
            .collect();
        
        let original_data = tensor.as_slice().unwrap();
        let mut tiled_data = Vec::with_capacity(new_shape.iter().product());
        
        // Simple tiling implementation
        Self::_tile_recursive(original_data, original_shape, &full_reps, 0, &mut vec![], &mut tiled_data);
        
        Ok(Tensor::from_vec(tiled_data, new_shape))
    }

    /// Recursive helper for tiling
    /// タイル化の再帰ヘルパー
    fn _tile_recursive(
        data: &[T],
        shape: &[usize],
        reps: &[usize],
        dim: usize,
        current_indices: &mut Vec<usize>,
        result: &mut Vec<T>,
    ) {
        if dim == shape.len() {
            // Calculate linear index in original tensor
            let mut linear_idx = 0;
            let mut stride = 1;
            
            for i in (0..shape.len()).rev() {
                let original_idx = current_indices[i] % shape[i];
                linear_idx += original_idx * stride;
                stride *= shape[i];
            }
            
            result.push(data[linear_idx]);
        } else {
            let new_size = shape[dim] * reps[dim];
            for i in 0..new_size {
                current_indices.push(i);
                Self::_tile_recursive(data, shape, reps, dim + 1, current_indices, result);
                current_indices.pop();
            }
        }
    }

    /// Create tensor by repeating elements
    /// 要素を繰り返してテンソルを作成
    pub fn repeat_interleave(&self, repeats: usize, dim: Option<usize>) -> Result<Tensor<T>, String> {
        if repeats == 0 {
            return Err("Repeats must be positive".to_string());
        }
        
        let shape = self.shape();
        
        if let Some(axis) = dim {
            if axis >= shape.len() {
                return Err(format!("Dimension {} is out of bounds", axis));
            }
            
            let mut new_shape = shape.to_vec();
            new_shape[axis] *= repeats;
            
            let data = self.as_slice().unwrap();
            let mut result_data = Vec::with_capacity(new_shape.iter().product());
            
            // Calculate strides
            let axis_stride = shape[axis + 1..].iter().product::<usize>();
            let outer_size = shape[..axis].iter().product::<usize>();
            
            for outer in 0..outer_size {
                for i in 0..shape[axis] {
                    for _ in 0..repeats {
                        let start_idx = outer * shape[axis] * axis_stride + i * axis_stride;
                        let end_idx = start_idx + axis_stride;
                        result_data.extend_from_slice(&data[start_idx..end_idx]);
                    }
                }
            }
            
            Ok(Tensor::from_vec(result_data, new_shape))
        } else {
            // Repeat all elements
            let data = self.as_slice().unwrap();
            let mut result_data = Vec::with_capacity(data.len() * repeats);
            
            for &value in data {
                for _ in 0..repeats {
                    result_data.push(value);
                }
            }
            
            let mut new_shape = shape.to_vec();
            if let Some(last_dim) = new_shape.last_mut() {
                *last_dim *= repeats;
            }
            
            Ok(Tensor::from_vec(result_data, new_shape))
        }
    }

    /// Create meshgrid from 1D tensors
    /// 1Dテンソルからメッシュグリッドを作成
    pub fn meshgrid(tensors: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>, String> {
        if tensors.is_empty() {
            return Err("At least one tensor required".to_string());
        }
        
        // Check that all tensors are 1D
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.ndim() != 1 {
                return Err(format!("Tensor {} is not 1-dimensional", i));
            }
        }
        
        let sizes: Vec<usize> = tensors.iter().map(|t| t.numel()).collect();
        let total_size: usize = sizes.iter().product();
        
        let mut result = Vec::new();
        
        for (dim_idx, tensor) in tensors.iter().enumerate() {
            let tensor_data = tensor.as_slice().unwrap();
            let mut grid_data = Vec::with_capacity(total_size);
            
            let repeat_inner = sizes[dim_idx + 1..].iter().product::<usize>();
            let repeat_outer = sizes[..dim_idx].iter().product::<usize>();
            
            for _ in 0..repeat_outer {
                for &value in tensor_data {
                    for _ in 0..repeat_inner {
                        grid_data.push(value);
                    }
                }
            }
            
            result.push(Tensor::from_vec(grid_data, sizes.clone()));
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randn() {
        let tensor = Tensor::<f32>::randn(&[3, 4]);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.numel(), 12);
        
        // Check that values are different (highly likely for random)
        let data = tensor.as_slice().unwrap();
        assert_ne!(data[0], data[1]);
    }

    #[test]
    fn test_rand() {
        let tensor = Tensor::<f32>::rand(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        
        // All values should be in [0, 1)
        for &value in tensor.as_slice().unwrap() {
            assert!(value >= 0.0 && value < 1.0);
        }
    }

    #[test]
    fn test_eye() {
        let eye = Tensor::<f32>::eye(3);
        assert_eq!(eye.shape(), &[3, 3]);
        
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ];
        assert_eq!(eye.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_arange() {
        let range = Tensor::<f32>::arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(range.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        
        let range_step = Tensor::<f32>::arange(1.0, 4.0, 0.5).unwrap();
        assert_eq!(range_step.as_slice().unwrap(), &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn test_linspace() {
        let linspace = Tensor::<f32>::linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(linspace.as_slice().unwrap(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
        
        let single = Tensor::<f32>::linspace(5.0, 10.0, 1).unwrap();
        assert_eq!(single.as_slice().unwrap(), &[5.0]);
    }

    #[test]
    fn test_logspace() {
        let logspace = Tensor::<f32>::logspace(0.0, 2.0, 3, 10.0).unwrap();
        let expected = vec![1.0, 10.0, 100.0]; // 10^0, 10^1, 10^2
        let result = logspace.as_slice().unwrap();
        
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result[i] - expected_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_diag() {
        let diagonal = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let diag_matrix = Tensor::diag(&diagonal, 0).unwrap();
        
        assert_eq!(diag_matrix.shape(), &[3, 3]);
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0
        ];
        assert_eq!(diag_matrix.as_slice().unwrap(), &expected);
        
        // Test with offset
        let diag_offset = Tensor::diag(&diagonal, 1).unwrap();
        assert_eq!(diag_offset.shape(), &[4, 4]);
    }

    #[test]
    fn test_tile() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let tiled = Tensor::tile(&tensor, &[3]).unwrap();
        
        assert_eq!(tiled.shape(), &[6]);
        assert_eq!(tiled.as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_interleave() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let repeated = tensor.repeat_interleave(2, None).unwrap();
        
        assert_eq!(repeated.shape(), &[6]);
        assert_eq!(repeated.as_slice().unwrap(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_meshgrid() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let y = Tensor::from_vec(vec![3.0f32, 4.0, 5.0], vec![3]);
        
        let grids = Tensor::meshgrid(&[&x, &y]).unwrap();
        assert_eq!(grids.len(), 2);
        
        assert_eq!(grids[0].shape(), &[2, 3]);
        assert_eq!(grids[1].shape(), &[2, 3]);
        
        // Check X grid
        assert_eq!(grids[0].as_slice().unwrap(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        // Check Y grid
        assert_eq!(grids[1].as_slice().unwrap(), &[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_randint() {
        let tensor = Tensor::<f32>::randint(&[2, 3], 0, 10);
        assert_eq!(tensor.shape(), &[2, 3]);
        
        // All values should be in [0, 10)
        for &value in tensor.as_slice().unwrap() {
            assert!(value >= 0.0 && value < 10.0);
            assert_eq!(value.fract(), 0.0); // Should be integer values
        }
    }

    #[test]
    fn test_arange_errors() {
        assert!(Tensor::<f32>::arange(0.0, 1.0, 0.0).is_err()); // Zero step
        assert!(Tensor::<f32>::arange(1.0, 0.0, 1.0).unwrap().numel() == 0); // Invalid range
    }
}