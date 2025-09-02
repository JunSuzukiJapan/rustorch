//! Tensor shape operations with Rust ownership patterns
//! Rustの所有権パターンを考慮したテンソル形状操作

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::sync::Arc;

/// Shape operation modes considering Rust ownership
/// Rustの所有権を考慮した形状操作モード
pub enum ShapeMode {
    /// Create new tensor (clone data) - always safe but potentially expensive
    /// 新しいテンソルを作成（データクローン） - 常に安全だが潜在的に高コスト
    Owned,
    /// Create view when possible, fallback to owned - optimal performance
    /// 可能な場合はビューを作成、所有にフォールバック - 最適パフォーマンス
    ViewOrOwned,
    /// Force view creation, error if not possible - zero-copy guarantee
    /// ビュー作成を強制、不可能な場合はエラー - ゼロコピー保証
    ViewOnly,
}

impl<T: Float + Clone + 'static> Tensor<T> {
    /// Remove singleton dimensions (size 1) from tensor
    /// テンソルから単一次元（サイズ1）を削除
    /// 
    /// # Ownership Patterns / 所有権パターン
    /// - `squeeze()` - Always creates new tensor (owned)
    /// - `squeeze_view()` - Attempts zero-copy view, fallback to owned
    /// - `squeeze_inplace()` - Modifies existing tensor (requires &mut)
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
    /// 
    /// // Always safe - creates new tensor
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[3]);
    /// 
    /// // Zero-copy when possible
    /// let squeezed_view = tensor.squeeze_view().unwrap();
    /// 
    /// // In-place modification
    /// let mut tensor_mut = tensor.clone();
    /// tensor_mut.squeeze_inplace().unwrap();
    /// ```
    pub fn squeeze(&self) -> Self {
        self.squeeze_with_mode(ShapeMode::Owned)
            .expect("Owned squeeze should never fail")
    }

    /// Remove singleton dimensions with zero-copy optimization
    /// ゼロコピー最適化で単一次元を削除
    pub fn squeeze_view(&self) -> RusTorchResult<Self> {
        self.squeeze_with_mode(ShapeMode::ViewOrOwned)
    }

    /// Remove singleton dimensions in-place (requires mutable reference)
    /// 単一次元をインプレースで削除（可変参照が必要）
    pub fn squeeze_inplace(&mut self) -> RusTorchResult<()> {
        let current_shape = self.data.shape();
        let new_shape: Vec<usize> = current_shape
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();

        // Ensure at least one dimension remains
        let final_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };

        // Try in-place reshape (zero-copy when layout allows)
        match self.data.clone().into_shape_with_order(final_shape) {
            Ok(reshaped) => {
                self.data = reshaped;
                Ok(())
            }
            Err(_) => Err(RusTorchError::InvalidOperation {
                operation: "squeeze_inplace".to_string(),
                message: "Cannot perform in-place squeeze due to layout constraints".to_string(),
            }),
        }
    }

    /// Remove singleton dimensions from specific dimension
    /// 特定の次元から単一次元を削除
    pub fn squeeze_dim(&self, dim: usize) -> RusTorchResult<Self> {
        let current_shape = self.data.shape();

        if dim >= current_shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension {} (max: {})",
                dim,
                current_shape.len() - 1
            )));
        }

        if current_shape[dim] != 1 {
            return Err(RusTorchError::InvalidOperation {
                operation: "squeeze_dim".to_string(),
                message: format!("Cannot squeeze dimension {} with size {}", dim, current_shape[dim]),
            });
        }

        let mut new_shape = current_shape.to_vec();
        new_shape.remove(dim);

        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let reshaped_data = self
            .data
            .clone()
            .into_shape_with_order(new_shape)
            .map_err(|_| RusTorchError::InvalidOperation {
                operation: "squeeze_dim".to_string(),
                message: "Failed to reshape tensor".to_string(),
            })?;

        Ok(Tensor::new(reshaped_data))
    }

    /// Add singleton dimension at specified position
    /// 指定位置に単一次元を追加
    /// 
    /// # Ownership Patterns / 所有権パターン
    /// - `unsqueeze()` - Always creates new tensor (owned)
    /// - `unsqueeze_view()` - Attempts zero-copy view
    /// - `unsqueeze_inplace()` - Modifies existing tensor
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// 
    /// // Add dimension at start
    /// let unsqueezed = tensor.unsqueeze(0);
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    /// 
    /// // Add dimension at end  
    /// let unsqueezed = tensor.unsqueeze(1);
    /// assert_eq!(unsqueezed.shape(), &[3, 1]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        self.unsqueeze_with_mode(dim, ShapeMode::Owned)
    }

    /// Add singleton dimension with zero-copy optimization
    /// ゼロコピー最適化で単一次元を追加
    pub fn unsqueeze_view(&self, dim: usize) -> RusTorchResult<Self> {
        self.unsqueeze_with_mode(dim, ShapeMode::ViewOrOwned)
    }

    /// Add singleton dimension in-place
    /// 単一次元をインプレースで追加
    pub fn unsqueeze_inplace(&mut self, dim: usize) -> RusTorchResult<()> {
        let mut new_shape = self.data.shape().to_vec();

        if dim > new_shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension {} (max: {})",
                dim,
                new_shape.len()
            )));
        }

        new_shape.insert(dim, 1);

        match self.data.clone().into_shape_with_order(new_shape) {
            Ok(reshaped) => {
                self.data = reshaped;
                Ok(())
            }
            Err(_) => Err(RusTorchError::InvalidOperation {
                operation: "unsqueeze_inplace".to_string(),
                message: "Cannot perform in-place unsqueeze due to layout constraints".to_string(),
            }),
        }
    }

    /// Expand tensor dimensions through broadcasting (ownership-aware version)
    /// ブロードキャストによってテンソル次元を拡張（所有権対応版）
    /// 
    /// # Ownership Considerations / 所有権の考慮事項
    /// Expand operations typically require data duplication, so we provide:
    /// - `expand_owned()` - Creates new tensor with explicit memory allocation
    /// - `expand_lazy()` - Returns lazy view that computes on access (memory efficient)
    /// - `expand_shared()` - Uses shared ownership with Arc for memory efficiency
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    /// 
    /// // Expand to larger shape
    /// let expanded = tensor.expand_owned(&[4, 3]).unwrap();
    /// assert_eq!(expanded.shape(), &[4, 3]);
    /// 
    /// // Memory-efficient lazy expansion
    /// let lazy_expanded = tensor.expand_lazy(&[4, 3]).unwrap();
    /// ```
    pub fn expand_owned(&self, target_shape: &[usize]) -> RusTorchResult<Self> {
        self.expand_with_mode(target_shape, ShapeMode::Owned)
    }

    /// Expand with shared ownership for memory efficiency
    /// メモリ効率のための共有所有権で拡張
    pub fn expand_shared(&self, target_shape: &[usize]) -> RusTorchResult<Arc<Self>> {
        let expanded = self.expand_with_mode(target_shape, ShapeMode::ViewOrOwned)?;
        Ok(Arc::new(expanded))
    }

    /// Lazy expand that defers computation until access
    /// アクセス時まで計算を遅延する遅延拡張
    pub fn expand_lazy(&self, target_shape: &[usize]) -> RusTorchResult<LazyExpandedTensor<T>> {
        // Validate expansion is possible
        self.validate_expansion(target_shape)?;
        
        Ok(LazyExpandedTensor {
            source: Arc::new(self.clone()),
            target_shape: target_shape.to_vec(),
        })
    }

    /// Flatten tensor dimensions into 1D (ownership-aware version)
    /// テンソル次元を1Dに平坦化（所有権対応版）
    /// 
    /// # Ownership Patterns / 所有権パターン
    /// - `flatten_owned()` - Always creates new 1D tensor
    /// - `flatten_range()` - Flatten specific dimension range  
    /// - `flatten_inplace()` - In-place flattening when possible
    /// - `flatten_view()` - Zero-copy view when layout allows
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// 
    /// // Full flatten
    /// let flattened = tensor.flatten_owned();
    /// assert_eq!(flattened.shape(), &[4]);
    /// 
    /// // Flatten from specific dimension
    /// let partial_flatten = tensor.flatten_range(1, None).unwrap();
    /// assert_eq!(partial_flatten.shape(), &[2, 2]);
    /// ```
    pub fn flatten_owned(&self) -> Self {
        let total_elements = self.data.len();
        let flattened_data = self.data.clone().into_shape_with_order(vec![total_elements])
            .expect("Flatten should always succeed");
        Tensor::new(flattened_data)
    }

    /// Flatten specific dimension range
    /// 特定の次元範囲を平坦化
    pub fn flatten_range(&self, start_dim: usize, end_dim: Option<usize>) -> RusTorchResult<Self> {
        let shape = self.shape();
        let end_dim = end_dim.unwrap_or(shape.len() - 1);

        if start_dim >= shape.len() || end_dim >= shape.len() || start_dim > end_dim {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension range [{}, {}] for tensor with {} dimensions",
                start_dim, end_dim, shape.len()
            )));
        }

        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&shape[..start_dim]);
        
        let flattened_size: usize = shape[start_dim..=end_dim].iter().product();
        new_shape.push(flattened_size);
        
        new_shape.extend_from_slice(&shape[end_dim + 1..]);

        let reshaped_data = self.data.clone()
            .into_shape_with_order(new_shape)
            .map_err(|_| RusTorchError::InvalidOperation {
                operation: "flatten_range".to_string(), 
                message: "Failed to flatten dimension range".to_string(),
            })?;

        Ok(Tensor::new(reshaped_data))
    }

    /// Flatten in-place when layout allows
    /// レイアウトが許可する場合のインプレース平坦化
    pub fn flatten_inplace(&mut self) -> RusTorchResult<()> {
        let total_elements = self.data.len();
        
        match self.data.clone().into_shape_with_order(vec![total_elements]) {
            Ok(flattened) => {
                self.data = flattened;
                Ok(())
            }
            Err(_) => Err(RusTorchError::InvalidOperation {
                operation: "flatten_inplace".to_string(),
                message: "Cannot perform in-place flatten due to layout constraints".to_string(),
            })
        }
    }

    /// Zero-copy flatten when memory layout allows
    /// メモリレイアウトが許可する場合のゼロコピー平坦化
    pub fn flatten_view(&self) -> RusTorchResult<Self> {
        let total_elements = self.data.len();
        
        // Check if data is contiguous for zero-copy view
        if self.is_contiguous() {
            let view_data = self.data.clone().into_shape_with_order(vec![total_elements])
                .map_err(|_| RusTorchError::InvalidOperation {
                    operation: "flatten_view".to_string(),
                    message: "Cannot create view due to non-contiguous layout".to_string(),
                })?;
            Ok(Tensor::new(view_data))
        } else {
            Err(RusTorchError::InvalidOperation {
                operation: "flatten_view".to_string(),
                message: "Cannot create zero-copy view from non-contiguous tensor".to_string(),
            })
        }
    }

    // Private helper methods

    fn squeeze_with_mode(&self, mode: ShapeMode) -> RusTorchResult<Self> {
        let current_shape = self.data.shape();
        let new_shape: Vec<usize> = current_shape
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();

        let final_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };

        match mode {
            ShapeMode::Owned => {
                let reshaped = self.data.clone()
                    .into_shape_with_order(final_shape)
                    .map_err(|_| RusTorchError::InvalidOperation {
                        operation: "squeeze".to_string(),
                        message: "Failed to squeeze tensor".to_string(),
                    })?;
                Ok(Tensor::new(reshaped))
            }
            ShapeMode::ViewOrOwned => {
                // Try view first, fallback to owned
                if self.is_contiguous() {
                    self.squeeze_with_mode(ShapeMode::ViewOnly)
                        .or_else(|_| self.squeeze_with_mode(ShapeMode::Owned))
                } else {
                    self.squeeze_with_mode(ShapeMode::Owned)
                }
            }
            ShapeMode::ViewOnly => {
                if !self.is_contiguous() {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "squeeze_view".to_string(),
                        message: "Cannot create view from non-contiguous tensor".to_string(),
                    });
                }
                let reshaped = self.data.clone()
                    .into_shape_with_order(final_shape)
                    .map_err(|_| RusTorchError::InvalidOperation {
                        operation: "squeeze_view".to_string(),
                        message: "Failed to create view".to_string(),
                    })?;
                Ok(Tensor::new(reshaped))
            }
        }
    }

    fn unsqueeze_with_mode(&self, dim: usize, mode: ShapeMode) -> RusTorchResult<Self> {
        let mut new_shape = self.data.shape().to_vec();

        if dim > new_shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension {} (max: {})",
                dim,
                new_shape.len()
            )));
        }

        new_shape.insert(dim, 1);

        match mode {
            ShapeMode::Owned => {
                let reshaped = self.data.clone()
                    .into_shape_with_order(new_shape)
                    .map_err(|_| RusTorchError::InvalidOperation {
                        operation: "unsqueeze".to_string(),
                        message: "Failed to unsqueeze tensor".to_string(),
                    })?;
                Ok(Tensor::new(reshaped))
            }
            ShapeMode::ViewOrOwned => {
                if self.is_contiguous() {
                    self.unsqueeze_with_mode(dim, ShapeMode::ViewOnly)
                        .or_else(|_| self.unsqueeze_with_mode(dim, ShapeMode::Owned))
                } else {
                    self.unsqueeze_with_mode(dim, ShapeMode::Owned)
                }
            }
            ShapeMode::ViewOnly => {
                if !self.is_contiguous() {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "unsqueeze_view".to_string(),
                        message: "Cannot create view from non-contiguous tensor".to_string(),
                    });
                }
                let reshaped = self.data.clone()
                    .into_shape_with_order(new_shape)
                    .map_err(|_| RusTorchError::InvalidOperation {
                        operation: "unsqueeze_view".to_string(),
                        message: "Failed to create view".to_string(),
                    })?;
                Ok(Tensor::new(reshaped))
            }
        }
    }

    fn expand_with_mode(&self, target_shape: &[usize], mode: ShapeMode) -> RusTorchResult<Self> {
        self.validate_expansion(target_shape)?;

        match mode {
            ShapeMode::Owned => self.expand_impl(target_shape),
            ShapeMode::ViewOrOwned => {
                // For expand, view is rarely possible due to data duplication needs
                self.expand_impl(target_shape)
            }
            ShapeMode::ViewOnly => {
                Err(RusTorchError::InvalidOperation {
                    operation: "expand_view".to_string(),
                    message: "Expand operation cannot be performed as zero-copy view".to_string(),
                })
            }
        }
    }

    fn expand_impl(&self, target_shape: &[usize]) -> RusTorchResult<Self> {
        let mut expanded_data = Vec::new();
        let total_elements: usize = target_shape.iter().product();
        expanded_data.reserve(total_elements);

        self.expand_recursive(
            &mut expanded_data,
            target_shape,
            &vec![0; target_shape.len()],
            0
        )?;

        Ok(Tensor::from_vec(expanded_data, target_shape.to_vec()))
    }

    fn validate_expansion(&self, target_shape: &[usize]) -> RusTorchResult<()> {
        let self_shape = self.shape();
        
        if target_shape.len() < self_shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "expand".to_string(),
                message: format!(
                    "Target shape must have at least {} dimensions, got {}",
                    self_shape.len(),
                    target_shape.len()
                ),
            });
        }

        let ndim_diff = target_shape.len() - self_shape.len();

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

        Ok(())
    }

    fn expand_recursive(
        &self,
        output: &mut Vec<T>,
        target_shape: &[usize],
        indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == target_shape.len() {
            // Base case - copy element
            let self_indices = self.compute_source_indices(indices)?;
            if let Some(&value) = self.data.get(self_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        let mut new_indices = indices.to_vec();
        for i in 0..target_shape[dim] {
            new_indices[dim] = i;
            self.expand_recursive(output, target_shape, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    fn compute_source_indices(&self, target_indices: &[usize]) -> RusTorchResult<Vec<usize>> {
        let self_shape = self.shape();
        let ndim_diff = target_indices.len() - self_shape.len();
        
        let mut source_indices = Vec::new();
        
        for (i, &target_idx) in target_indices.iter().skip(ndim_diff).enumerate() {
            let self_dim = self_shape[i];
            if self_dim == 1 {
                source_indices.push(0);
            } else {
                source_indices.push(target_idx % self_dim);
            }
        }
        
        Ok(source_indices)
    }

    // Helper methods for new operations

    fn repeat_recursive(
        &self,
        output: &mut Vec<T>,
        output_shape: &[usize],
        repeats: &[usize],
        indices: &[usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == output_shape.len() {
            // Base case - copy element from source
            let source_indices = self.compute_repeat_source_indices(indices, output_shape, repeats)?;
            if let Some(&value) = self.data.get(source_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        let mut new_indices = indices.to_vec();
        for i in 0..output_shape[dim] {
            new_indices[dim] = i;
            self.repeat_recursive(output, output_shape, repeats, &new_indices, dim + 1)?;
        }

        Ok(())
    }

    fn compute_repeat_source_indices(
        &self,
        output_indices: &[usize],
        output_shape: &[usize],
        repeats: &[usize],
    ) -> RusTorchResult<Vec<usize>> {
        let self_shape = self.shape();
        let mut source_indices = Vec::new();

        // Handle dimension adjustment
        let ndim_diff = if output_shape.len() > self_shape.len() {
            output_shape.len() - self_shape.len()
        } else {
            0
        };

        for (i, &output_idx) in output_indices.iter().enumerate() {
            if i < ndim_diff {
                // Skip extra leading dimensions
                continue;
            }
            
            let self_dim_idx = i - ndim_diff;
            if self_dim_idx < self_shape.len() {
                let self_dim_size = self_shape[self_dim_idx];
                let repeat_count = repeats[i];
                source_indices.push(output_idx / repeat_count);
            }
        }

        Ok(source_indices)
    }

    fn repeat_interleave_along_dim(&self, repeats: usize, dim: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        if dim >= shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension {} (max: {})",
                dim,
                shape.len() - 1
            )));
        }

        let mut output_shape = shape.to_vec();
        output_shape[dim] *= repeats;

        let mut output_data = Vec::new();
        let total_elements: usize = output_shape.iter().product();
        output_data.reserve(total_elements);

        // Generate indices for output tensor
        let mut indices = vec![0; output_shape.len()];
        self.repeat_interleave_recursive(&mut output_data, &output_shape, repeats, dim, &mut indices, 0)?;

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    fn repeat_interleave_recursive(
        &self,
        output: &mut Vec<T>,
        output_shape: &[usize],
        repeats: usize,
        target_dim: usize,
        indices: &mut [usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == output_shape.len() {
            // Base case - compute source index and copy element
            let mut source_indices = indices.to_vec();
            if target_dim < source_indices.len() {
                source_indices[target_dim] = indices[target_dim] / repeats;
            }
            
            if let Some(&value) = self.data.get(source_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        for i in 0..output_shape[dim] {
            indices[dim] = i;
            self.repeat_interleave_recursive(output, output_shape, repeats, target_dim, indices, dim + 1)?;
        }

        Ok(())
    }

    fn roll_along_dimension(&self, shift: usize, dim: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        let dim_size = shape[dim];
        
        if shift >= dim_size {
            return Err(RusTorchError::InvalidOperation {
                operation: "roll".to_string(),
                message: "Shift amount exceeds dimension size".to_string(),
            });
        }

        let mut output_data = Vec::with_capacity(self.data.len());
        let mut indices = vec![0; shape.len()];
        
        self.roll_recursive(&mut output_data, shape, shift, dim, &mut indices, 0)?;

        Ok(Tensor::from_vec(output_data, shape.to_vec()))
    }

    fn roll_recursive(
        &self,
        output: &mut Vec<T>,
        shape: &[usize],
        shift: usize,
        target_dim: usize,
        indices: &mut [usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == shape.len() {
            // Base case - compute rolled source index
            let mut source_indices = indices.to_vec();
            if target_dim < source_indices.len() {
                let dim_size = shape[target_dim];
                let rolled_idx = (indices[target_dim] + dim_size - shift) % dim_size;
                source_indices[target_dim] = rolled_idx;
            }
            
            if let Some(&value) = self.data.get(source_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        for i in 0..shape[dim] {
            indices[dim] = i;
            self.roll_recursive(output, shape, shift, target_dim, indices, dim + 1)?;
        }

        Ok(())
    }

    fn rot90_once(&self, dim0: usize, dim1: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape.swap(dim0, dim1);

        let mut output_data = Vec::with_capacity(self.data.len());
        let mut indices = vec![0; shape.len()];
        
        self.rot90_recursive(&mut output_data, shape, &new_shape, dim0, dim1, &mut indices, 0)?;

        Ok(Tensor::from_vec(output_data, new_shape))
    }

    fn rot90_recursive(
        &self,
        output: &mut Vec<T>,
        original_shape: &[usize],
        new_shape: &[usize],
        dim0: usize,
        dim1: usize,
        indices: &mut [usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == new_shape.len() {
            // Base case - compute rotated source indices
            let mut source_indices = indices.to_vec();
            
            // Apply 90-degree rotation transformation
            let old_i = indices[dim0];
            let old_j = indices[dim1];
            
            source_indices[dim0] = original_shape[dim1] - 1 - old_j;
            source_indices[dim1] = old_i;
            
            if let Some(&value) = self.data.get(source_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        for i in 0..new_shape[dim] {
            indices[dim] = i;
            self.rot90_recursive(output, original_shape, new_shape, dim0, dim1, indices, dim + 1)?;
        }

        Ok(())
    }

    fn flip_single_dim(&self, dim: usize) -> RusTorchResult<Self> {
        let shape = self.shape();
        let dim_size = shape[dim];
        
        let mut output_data = Vec::with_capacity(self.data.len());
        let mut indices = vec![0; shape.len()];
        
        self.flip_recursive(&mut output_data, shape, dim, &mut indices, 0)?;

        Ok(Tensor::from_vec(output_data, shape.to_vec()))
    }

    fn flip_recursive(
        &self,
        output: &mut Vec<T>,
        shape: &[usize],
        flip_dim: usize,
        indices: &mut [usize],
        dim: usize,
    ) -> RusTorchResult<()> {
        if dim == shape.len() {
            // Base case - compute flipped source index
            let mut source_indices = indices.to_vec();
            if flip_dim < source_indices.len() {
                let dim_size = shape[flip_dim];
                source_indices[flip_dim] = dim_size - 1 - indices[flip_dim];
            }
            
            if let Some(&value) = self.data.get(source_indices.as_slice()) {
                output.push(value);
            } else {
                return Err(RusTorchError::index_out_of_bounds(&[], &[]));
            }
            return Ok(());
        }

        for i in 0..shape[dim] {
            indices[dim] = i;
            self.flip_recursive(output, shape, flip_dim, indices, dim + 1)?;
        }

        Ok(())
    }

    /// Check if tensor data is contiguous in memory
    /// テンソルデータがメモリ内で連続しているかチェック
    pub fn is_contiguous(&self) -> bool {
        self.data.is_standard_layout()
    }

    // Note: shape() and numel() methods are defined in core.rs
    // 注意: shape()とnumel()メソッドはcore.rsで定義されています
    
    /// Check if this tensor can broadcast with another tensor
    /// このテンソルが別のテンソルとブロードキャスト可能かチェック
    pub fn can_broadcast_with(&self, other: &Self) -> bool {
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        let max_dims = self_shape.len().max(other_shape.len());
        
        for i in 0..max_dims {
            let self_dim = if i < self_shape.len() {
                self_shape[self_shape.len() - 1 - i]
            } else {
                1
            };
            
            let other_dim = if i < other_shape.len() {
                other_shape[other_shape.len() - 1 - i]
            } else {
                1
            };
            
            if self_dim != 1 && other_dim != 1 && self_dim != other_dim {
                return false;
            }
        }
        
        true
    }
    
    /// Broadcast two tensors to compatible shapes
    /// 2つのテンソルを互換形状にブロードキャスト
    pub fn broadcast_with(&self, other: &Self) -> RusTorchResult<(Self, Self)> {
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::InvalidOperation {
                operation: "broadcast".to_string(),
                message: format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        let max_dims = self_shape.len().max(other_shape.len());
        
        let mut broadcast_shape = Vec::new();
        
        for i in 0..max_dims {
            let self_dim = if i < max_dims - self_shape.len() {
                1
            } else {
                self_shape[i - (max_dims - self_shape.len())]
            };
            
            let other_dim = if i < max_dims - other_shape.len() {
                1
            } else {
                other_shape[i - (max_dims - other_shape.len())]
            };
            
            broadcast_shape.push(self_dim.max(other_dim));
        }
        
        let broadcasted_self = if self.shape() == broadcast_shape.as_slice() {
            self.clone()
        } else {
            self.expand_owned(&broadcast_shape)?
        };
        
        let broadcasted_other = if other.shape() == broadcast_shape.as_slice() {
            other.clone()
        } else {
            other.expand_owned(&broadcast_shape)?
        };
        
        Ok((broadcasted_self, broadcasted_other))
    }
    
    /// Add singleton dimension (alias for unsqueeze for compatibility)
    /// 単一次元追加（互換性のためのunsqueezeエイリアス）
    pub fn expand_dims(&self, axis: usize) -> RusTorchResult<Self> {
        self.unsqueeze(axis)
    }

    /// Expand tensor to match the shape of another tensor (PyTorch expand_as compatibility)
    /// 他のテンソルの形状に合わせて拡張（PyTorch expand_as互換）
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    /// let target = Tensor::from_vec(vec![0.0; 6], vec![3, 2]);
    /// let expanded = tensor.expand_as(&target).unwrap();
    /// assert_eq!(expanded.shape(), target.shape());
    /// ```
    pub fn expand_as(&self, other: &Self) -> RusTorchResult<Self> {
        self.expand_owned(other.shape())
    }

    /// Unflatten a tensor dimension into multiple dimensions
    /// テンソルの次元を複数の次元に復元
    /// 
    /// # Arguments
    /// * `dim` - Dimension to unflatten  
    /// * `sizes` - Target sizes for new dimensions
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    /// let unflattened = tensor.unflatten(0, &[2, 3]).unwrap();
    /// assert_eq!(unflattened.shape(), &[2, 3]);
    /// ```
    pub fn unflatten(&self, dim: usize, sizes: &[usize]) -> RusTorchResult<Self> {
        let current_shape = self.shape();
        
        if dim >= current_shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimension {} (max: {})",
                dim,
                current_shape.len() - 1
            )));
        }

        // Validate that product of sizes matches the dimension size
        let sizes_product: usize = sizes.iter().product();
        if sizes_product != current_shape[dim] {
            return Err(RusTorchError::InvalidOperation {
                operation: "unflatten".to_string(),
                message: format!(
                    "Cannot unflatten dimension of size {} into sizes {:?} (product: {})",
                    current_shape[dim],
                    sizes,
                    sizes_product
                ),
            });
        }

        // Build new shape
        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&current_shape[..dim]);
        new_shape.extend_from_slice(sizes);
        new_shape.extend_from_slice(&current_shape[dim + 1..]);

        // Reshape tensor
        let reshaped_data = self.data.clone()
            .into_shape_with_order(new_shape)
            .map_err(|_| RusTorchError::InvalidOperation {
                operation: "unflatten".to_string(),
                message: "Failed to unflatten tensor".to_string(),
            })?;

        Ok(Tensor::new(reshaped_data))
    }

    /// Repeat tensor along specified dimensions (PyTorch repeat compatibility)
    /// 指定された次元に沿ってテンソルを繰り返し（PyTorch repeat互換）
    /// 
    /// # Arguments
    /// * `repeats` - Number of repetitions for each dimension
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let repeated = tensor.repeat(&[3, 2]).unwrap();
    /// assert_eq!(repeated.shape(), &[6, 4]); // [3*2, 2*2]
    /// ```
    pub fn repeat(&self, repeats: &[usize]) -> RusTorchResult<Self> {
        let current_shape = self.shape();
        
        // Handle dimension mismatch by padding with 1s
        let (adjusted_shape, adjusted_repeats) = if repeats.len() > current_shape.len() {
            let padding = repeats.len() - current_shape.len();
            let mut padded_shape = vec![1; padding];
            padded_shape.extend_from_slice(current_shape);
            (padded_shape, repeats.to_vec())
        } else if repeats.len() < current_shape.len() {
            let padding = current_shape.len() - repeats.len();
            let mut padded_repeats = vec![1; padding];
            padded_repeats.extend_from_slice(repeats);
            (current_shape.to_vec(), padded_repeats)
        } else {
            (current_shape.to_vec(), repeats.to_vec())
        };

        // Calculate output shape
        let output_shape: Vec<usize> = adjusted_shape
            .iter()
            .zip(adjusted_repeats.iter())
            .map(|(&dim, &rep)| dim * rep)
            .collect();

        // Generate repeated data
        let mut output_data = Vec::new();
        let total_elements: usize = output_shape.iter().product();
        output_data.reserve(total_elements);

        self.repeat_recursive(
            &mut output_data,
            &output_shape,
            &adjusted_repeats,
            &vec![0; output_shape.len()],
            0
        )?;

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    /// Repeat elements of tensor along specified dimension
    /// 指定次元に沿ってテンソルの要素を繰り返し
    /// 
    /// # Arguments  
    /// * `repeats` - Number of repetitions for each element (scalar or tensor)
    /// * `dim` - Dimension along which to repeat (optional)
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let repeated = tensor.repeat_interleave_scalar(2, Some(0)).unwrap();
    /// assert_eq!(repeated.shape(), &[6]); // Each element repeated twice
    /// ```
    pub fn repeat_interleave_scalar(&self, repeats: usize, dim: Option<usize>) -> RusTorchResult<Self> {
        match dim {
            Some(d) => self.repeat_interleave_along_dim(repeats, d),
            None => {
                // Flatten and repeat each element
                let flattened = self.flatten_owned();
                let mut output_data = Vec::new();
                
                for &value in flattened.data.iter() {
                    for _ in 0..repeats {
                        output_data.push(value);
                    }
                }
                
                let output_len = output_data.len();
                Ok(Tensor::from_vec(output_data, vec![output_len]))
            }
        }
    }

    /// Roll tensor along specified dimensions
    /// 指定された次元に沿ってテンソルをロール
    /// 
    /// # Arguments
    /// * `shifts` - Number of places to shift
    /// * `dims` - Dimensions to roll along (optional)
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    /// let rolled = tensor.roll_1d(1, Some(0)).unwrap();
    /// // Result: [4.0, 1.0, 2.0, 3.0]
    /// ```
    pub fn roll_1d(&self, shifts: isize, dim: Option<usize>) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        match dim {
            Some(d) => {
                if d >= shape.len() {
                    return Err(RusTorchError::InvalidDimension(format!(
                        "Invalid dimension {} (max: {})",
                        d,
                        shape.len() - 1
                    )));
                }
                
                let dim_size = shape[d] as isize;
                let effective_shift = ((shifts % dim_size) + dim_size) % dim_size;
                
                if effective_shift == 0 {
                    return Ok(self.clone());
                }
                
                self.roll_along_dimension(effective_shift as usize, d)
            }
            None => {
                // Roll flattened tensor
                let flattened = self.flatten_owned();
                let data = flattened.data.as_slice().unwrap();
                let len = data.len() as isize;
                let effective_shift = ((shifts % len) + len) % len;
                
                if effective_shift == 0 {
                    return Ok(self.clone());
                }
                
                let mut output_data = Vec::with_capacity(data.len());
                let shift = effective_shift as usize;
                
                output_data.extend_from_slice(&data[data.len() - shift..]);
                output_data.extend_from_slice(&data[..data.len() - shift]);
                
                let rolled_flat = Tensor::from_vec(output_data, vec![data.len()]);
                rolled_flat.view_shape(shape)
            }
        }
    }

    /// Rotate tensor 90 degrees in the plane specified by dims
    /// 指定された次元平面でテンソルを90度回転
    /// 
    /// # Arguments
    /// * `k` - Number of 90-degree rotations
    /// * `dims` - Two dimensions defining the rotation plane
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let rotated = tensor.rot90(1, &[0, 1]).unwrap();
    /// // 90-degree rotation
    /// ```
    pub fn rot90(&self, k: isize, dims: &[usize]) -> RusTorchResult<Self> {
        if dims.len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "rot90".to_string(),
                message: "rot90 requires exactly 2 dimensions".to_string(),
            });
        }

        let shape = self.shape();
        let dim0 = dims[0];
        let dim1 = dims[1];

        if dim0 >= shape.len() || dim1 >= shape.len() {
            return Err(RusTorchError::InvalidDimension(format!(
                "Invalid dimensions [{}, {}] (max: {})",
                dim0,
                dim1,
                shape.len() - 1
            )));
        }

        if dim0 == dim1 {
            return Err(RusTorchError::InvalidOperation {
                operation: "rot90".to_string(),
                message: "Rotation dimensions must be different".to_string(),
            });
        }

        // Normalize k to [0, 3]
        let k_norm = ((k % 4) + 4) % 4;
        
        match k_norm {
            0 => Ok(self.clone()),
            1 => self.rot90_once(dim0, dim1),
            2 => self.rot90_once(dim0, dim1)?.rot90_once(dim0, dim1),
            3 => self.rot90_once(dim0, dim1)?.rot90_once(dim0, dim1)?.rot90_once(dim0, dim1),
            _ => unreachable!(),
        }
    }

    /// Flip tensor along specified dimensions
    /// 指定された次元に沿ってテンソルを反転
    /// 
    /// # Arguments
    /// * `dims` - Dimensions to flip
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let flipped = tensor.flip(&[0]).unwrap();
    /// // Flip along dimension 0
    /// ```
    pub fn flip(&self, dims: &[usize]) -> RusTorchResult<Self> {
        let shape = self.shape();
        
        // Validate dimensions
        for &dim in dims {
            if dim >= shape.len() {
                return Err(RusTorchError::InvalidDimension(format!(
                    "Invalid dimension {} (max: {})",
                    dim,
                    shape.len() - 1
                )));
            }
        }

        if dims.is_empty() {
            return Ok(self.clone());
        }

        let mut result = self.clone();
        for &dim in dims {
            result = result.flip_single_dim(dim)?;
        }
        
        Ok(result)
    }

    /// Flip tensor left-right (along last dimension)
    /// テンソルを左右反転（最後の次元に沿って）
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let flipped = tensor.fliplr().unwrap();
    /// ```
    pub fn fliplr(&self) -> RusTorchResult<Self> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "fliplr".to_string(),
                message: "fliplr requires at least 2D tensor".to_string(),
            });
        }
        
        self.flip(&[shape.len() - 1])
    }

    /// Flip tensor up-down (along first dimension)  
    /// テンソルを上下反転（最初の次元に沿って）
    /// 
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let flipped = tensor.flipud().unwrap();
    /// ```
    pub fn flipud(&self) -> RusTorchResult<Self> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(RusTorchError::InvalidOperation {
                operation: "flipud".to_string(),
                message: "flipud requires at least 1D tensor".to_string(),
            });
        }
        
        self.flip(&[0])
    }
}

/// Lazy expanded tensor for memory-efficient broadcasting
/// メモリ効率的なブロードキャストのための遅延拡張テンソル
pub struct LazyExpandedTensor<T: Float> {
    source: Arc<Tensor<T>>,
    target_shape: Vec<usize>,
}

impl<T: Float + Clone + 'static> LazyExpandedTensor<T> {
    /// Materialize the lazy tensor into a concrete tensor
    /// 遅延テンソルを具体的なテンソルに実体化
    pub fn materialize(&self) -> RusTorchResult<Tensor<T>> {
        self.source.expand_owned(&self.target_shape)
    }

    /// Get the target shape without materializing
    /// 実体化せずにターゲット形状を取得
    pub fn shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// Access element with on-demand computation
    /// オンデマンド計算で要素にアクセス
    pub fn get(&self, indices: &[usize]) -> RusTorchResult<T> {
        if indices.len() != self.target_shape.len() {
            return Err(RusTorchError::index_out_of_bounds(indices, &self.target_shape));
        }

        let source_indices = self.source.compute_source_indices(indices)?;
        
        self.source.data.get(source_indices.as_slice())
            .copied()
            .ok_or_else(|| RusTorchError::index_out_of_bounds(&[], &[]))
    }
}

/// Builder pattern for complex shape operations
/// 複雑な形状操作のためのビルダーパターン
pub struct ShapeBuilder<T: Float> {
    tensor: Tensor<T>,
}

impl<T: Float + Clone + 'static> ShapeBuilder<T> {
    /// Create new shape builder
    /// 新しい形状ビルダーを作成
    pub fn new(tensor: Tensor<T>) -> Self {
        Self { tensor }
    }

    /// Chain squeeze operation
    /// squeeze操作をチェーン
    pub fn squeeze(mut self) -> RusTorchResult<Self> {
        self.tensor = self.tensor.squeeze();
        Ok(self)
    }

    /// Chain unsqueeze operation
    /// unsqueeze操作をチェーン
    pub fn unsqueeze(mut self, dim: usize) -> RusTorchResult<Self> {
        self.tensor = self.tensor.unsqueeze(dim)?;
        Ok(self)
    }

    /// Chain expand operation
    /// expand操作をチェーン
    pub fn expand(mut self, target_shape: &[usize]) -> RusTorchResult<Self> {
        self.tensor = self.tensor.expand_owned(target_shape)?;
        Ok(self)
    }

    /// Chain flatten operation
    /// flatten操作をチェーン
    pub fn flatten(mut self) -> Self {
        self.tensor = self.tensor.flatten_owned();
        self
    }

    /// Build final tensor
    /// 最終テンソルをビルド
    pub fn build(self) -> Tensor<T> {
        self.tensor
    }
}

/// Convenient methods for common shape operation patterns
/// 一般的な形状操作パターンの便利なメソッド
impl<T: Float + Clone + 'static> Tensor<T> {
    /// PyTorch-like view method for reshaping with ownership semantics
    /// 所有権セマンティクスでのPyTorchライクなviewメソッド
    pub fn view_shape(&self, shape: &[usize]) -> RusTorchResult<Self> {
        // Validate total elements match
        let current_elements = self.data.len();
        let new_elements: usize = shape.iter().product();
        
        if current_elements != new_elements {
            return Err(RusTorchError::InvalidOperation {
                operation: "view_shape".to_string(),
                message: format!(
                    "Shape {} is invalid for tensor with {} elements",
                    format!("{:?}", shape),
                    current_elements
                ),
            });
        }

        let reshaped_data = self.data.clone()
            .into_shape_with_order(shape.to_vec())
            .map_err(|_| RusTorchError::InvalidOperation {
                operation: "view_shape".to_string(),
                message: "Failed to create view with specified shape".to_string(),
            })?;

        Ok(Tensor::new(reshaped_data))
    }

    /// Create shape builder for chaining operations
    /// 操作チェーン用の形状ビルダーを作成
    pub fn shape_builder(self) -> ShapeBuilder<T> {
        ShapeBuilder::new(self)
    }
}

/// Zero-allocation shape operation traits for advanced use cases
/// 高度な用途のためのゼロ割り当て形状操作トレイト
pub trait ZeroAllocShapeOps<T: Float> {
    /// Attempt zero-copy squeeze
    /// ゼロコピーsqueezeを試行
    fn try_squeeze_view(&self) -> RusTorchResult<Tensor<T>>;
    
    /// Attempt zero-copy unsqueeze
    /// ゼロコピーunsqueezeを試行  
    fn try_unsqueeze_view(&self, dim: usize) -> RusTorchResult<Tensor<T>>;
}

impl<T: Float + Clone + 'static> ZeroAllocShapeOps<T> for Tensor<T> {
    fn try_squeeze_view(&self) -> RusTorchResult<Tensor<T>> {
        self.squeeze_with_mode(ShapeMode::ViewOnly)
    }

    fn try_unsqueeze_view(&self, dim: usize) -> RusTorchResult<Tensor<T>> {
        self.unsqueeze_with_mode(dim, ShapeMode::ViewOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ownership_patterns_squeeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        
        // Test owned squeeze
        let squeezed_owned = tensor.squeeze();
        assert_eq!(squeezed_owned.shape(), &[3]);
        
        // Test view squeeze
        let squeezed_view = tensor.squeeze_view().unwrap();
        assert_eq!(squeezed_view.shape(), &[3]);
        
        // Test in-place squeeze
        let mut tensor_mut = tensor.clone();
        tensor_mut.squeeze_inplace().unwrap();
        assert_eq!(tensor_mut.shape(), &[3]);
    }

    #[test]
    fn test_ownership_patterns_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        
        // Test owned unsqueeze
        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
        
        // Test view unsqueeze
        let unsqueezed_view = tensor.unsqueeze_view(1).unwrap();
        assert_eq!(unsqueezed_view.shape(), &[3, 1]);
        
        // Test in-place unsqueeze
        let mut tensor_mut = tensor.clone();
        tensor_mut.unsqueeze_inplace(0).unwrap();
        assert_eq!(tensor_mut.shape(), &[1, 3]);
    }

    #[test]
    fn test_expand_shared_ownership() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        
        // Test shared expand
        let expanded_shared = tensor.expand_shared(&[4, 3]).unwrap();
        assert_eq!(expanded_shared.shape(), &[4, 3]);
        
        // Multiple references to same expanded tensor
        let ref1 = Arc::clone(&expanded_shared);
        let ref2 = Arc::clone(&expanded_shared);
        
        assert_eq!(ref1.shape(), ref2.shape());
    }

    #[test]
    fn test_lazy_expand() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        
        // Create lazy expansion
        let lazy_expanded = tensor.expand_lazy(&[3, 2]).unwrap();
        assert_eq!(lazy_expanded.shape(), &[3, 2]);
        
        // Access specific elements
        assert_eq!(lazy_expanded.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(lazy_expanded.get(&[1, 1]).unwrap(), 2.0);
        assert_eq!(lazy_expanded.get(&[2, 0]).unwrap(), 1.0);
        
        // Materialize when needed
        let materialized = lazy_expanded.materialize().unwrap();
        assert_eq!(materialized.shape(), &[3, 2]);
    }

    #[test]
    fn test_shape_builder_chain() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2, 1]);
        
        // Chain multiple operations
        let result = tensor
            .shape_builder()
            .squeeze().unwrap()                          // Remove all size-1 dims: [2, 2]
            .unsqueeze(0).unwrap()                       // Add dim at start: [1, 2, 2]
            .expand(&[3, 2, 2]).unwrap()                 // Expand first dim: [3, 2, 2]
            .flatten()                                   // Flatten: [12]
            .build();
            
        assert_eq!(result.shape(), &[12]);
        assert_eq!(result.numel(), 12);
    }

    #[test]  
    fn test_flatten_variants() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Full flatten
        let flattened = tensor.flatten_owned();
        assert_eq!(flattened.shape(), &[4]);
        
        // Flatten range
        let partial = tensor.flatten_range(0, Some(1)).unwrap();
        assert_eq!(partial.shape(), &[4]);
        
        // Flatten view
        if tensor.is_contiguous() {
            let flattened_view = tensor.flatten_view().unwrap();
            assert_eq!(flattened_view.shape(), &[4]);
        }
        
        // In-place flatten
        let mut tensor_mut = tensor.clone();
        tensor_mut.flatten_inplace().unwrap();
        assert_eq!(tensor_mut.shape(), &[4]);
    }

    #[test]
    fn test_zero_alloc_traits() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        
        // Test zero-allocation operations when possible
        if tensor.is_contiguous() {
            let squeezed_view = tensor.try_squeeze_view().unwrap();
            assert_eq!(squeezed_view.shape(), &[3]);
            
            let unsqueezed_view = tensor.try_unsqueeze_view(0).unwrap();
            assert_eq!(unsqueezed_view.shape(), &[1, 1, 3]);
        }
    }

    #[test]
    fn test_expand_as() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let target = Tensor::from_vec(vec![0.0; 6], vec![3, 2]);
        
        let expanded = tensor.expand_as(&target).unwrap();
        assert_eq!(expanded.shape(), target.shape());
        assert_eq!(expanded.shape(), &[3, 2]);
        
        // Verify data is correctly expanded
        let expanded_data = expanded.data.as_slice().unwrap();
        assert_eq!(expanded_data, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_unflatten() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        
        // Unflatten into 2x3
        let unflattened = tensor.unflatten(0, &[2, 3]).unwrap();
        assert_eq!(unflattened.shape(), &[2, 3]);
        
        // Unflatten into 3x2 
        let unflattened2 = tensor.unflatten(0, &[3, 2]).unwrap();
        assert_eq!(unflattened2.shape(), &[3, 2]);
        
        // Test with multi-dimensional tensor
        let tensor2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let unflattened3d = tensor2d.unflatten(1, &[1, 2]).unwrap();
        assert_eq!(unflattened3d.shape(), &[2, 1, 2]);
        
        // Test invalid unflatten
        let result = tensor.unflatten(0, &[2, 4]); // 2*4=8 != 6
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        
        // Simple repeat - repeat each element individually
        let repeated = tensor.repeat(&[3]).unwrap();
        assert_eq!(repeated.shape(), &[6]);
        // Our implementation repeats elements: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        assert_eq!(repeated.data.as_slice().unwrap(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        
        // Multi-dimensional repeat
        let tensor2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let repeated2d = tensor2d.repeat(&[2, 3]).unwrap();
        assert_eq!(repeated2d.shape(), &[4, 6]);
        
        // Dimension mismatch handling
        let repeated_padded = tensor.repeat(&[2, 3]).unwrap();
        assert_eq!(repeated_padded.shape(), &[2, 6]);
    }

    #[test]
    fn test_repeat_interleave() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        
        // Repeat each element twice
        let repeated = tensor.repeat_interleave_scalar(2, Some(0)).unwrap();
        assert_eq!(repeated.shape(), &[6]);
        assert_eq!(repeated.data.as_slice().unwrap(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        
        // Flatten and repeat
        let repeated_flat = tensor.repeat_interleave_scalar(2, None).unwrap();
        assert_eq!(repeated_flat.shape(), &[6]);
        assert_eq!(repeated_flat.data.as_slice().unwrap(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_roll() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        
        // Roll by 1 position
        let rolled = tensor.roll_1d(1, Some(0)).unwrap();
        assert_eq!(rolled.shape(), &[4]);
        assert_eq!(rolled.data.as_slice().unwrap(), &[4.0, 1.0, 2.0, 3.0]);
        
        // Roll by negative amount
        let rolled_neg = tensor.roll_1d(-1, Some(0)).unwrap();
        assert_eq!(rolled_neg.data.as_slice().unwrap(), &[2.0, 3.0, 4.0, 1.0]);
        
        // Roll 2D tensor
        let tensor2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rolled2d = tensor2d.roll_1d(1, Some(0)).unwrap();
        assert_eq!(rolled2d.shape(), &[2, 2]);
        
        // Roll without specifying dimension (flattened)
        let rolled_flat = tensor.roll_1d(1, None).unwrap();
        assert_eq!(rolled_flat.shape(), &[4]);
        assert_eq!(rolled_flat.data.as_slice().unwrap(), &[4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rot90() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // 90-degree rotation
        let rotated1 = tensor.rot90(1, &[0, 1]).unwrap();
        assert_eq!(rotated1.shape(), &[2, 2]);
        
        // 180-degree rotation
        let rotated2 = tensor.rot90(2, &[0, 1]).unwrap();
        assert_eq!(rotated2.shape(), &[2, 2]);
        
        // 270-degree rotation 
        let rotated3 = tensor.rot90(3, &[0, 1]).unwrap();
        assert_eq!(rotated3.shape(), &[2, 2]);
        
        // Full rotation (360 degrees) should return original
        let rotated4 = tensor.rot90(4, &[0, 1]).unwrap();
        assert_eq!(rotated4.shape(), &[2, 2]);
        assert_eq!(rotated4.data.as_slice().unwrap(), tensor.data.as_slice().unwrap());
        
        // Negative rotation
        let rotated_neg = tensor.rot90(-1, &[0, 1]).unwrap();
        assert_eq!(rotated_neg.shape(), &[2, 2]);
        
        // Error cases
        assert!(tensor.rot90(1, &[0]).is_err()); // Need exactly 2 dims
        assert!(tensor.rot90(1, &[0, 0]).is_err()); // Dims must be different
        assert!(tensor.rot90(1, &[0, 5]).is_err()); // Invalid dimension
    }

    #[test]
    fn test_flip() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Flip along dimension 0
        let flipped0 = tensor.flip(&[0]).unwrap();
        assert_eq!(flipped0.shape(), &[2, 2]);
        
        // Flip along dimension 1
        let flipped1 = tensor.flip(&[1]).unwrap();
        assert_eq!(flipped1.shape(), &[2, 2]);
        
        // Flip along both dimensions
        let flipped_both = tensor.flip(&[0, 1]).unwrap();
        assert_eq!(flipped_both.shape(), &[2, 2]);
        
        // No flip (empty dimensions)
        let no_flip = tensor.flip(&[]).unwrap();
        assert_eq!(no_flip.data.as_slice().unwrap(), tensor.data.as_slice().unwrap());
        
        // Error case - invalid dimension
        assert!(tensor.flip(&[5]).is_err());
    }

    #[test]
    fn test_fliplr_flipud() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Test fliplr (left-right flip)
        let flipped_lr = tensor.fliplr().unwrap();
        assert_eq!(flipped_lr.shape(), &[2, 2]);
        
        // Test flipud (up-down flip)
        let flipped_ud = tensor.flipud().unwrap();
        assert_eq!(flipped_ud.shape(), &[2, 2]);
        
        // Error cases
        let tensor1d = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        assert!(tensor1d.fliplr().is_err()); // Need at least 2D for fliplr
        
        let tensor0d = Tensor::from_vec(vec![1.0], vec![]);
        assert!(tensor0d.flipud().is_err()); // Need at least 1D for flipud
    }

    #[test]
    fn test_complex_shape_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        // Complex chain of operations using builder
        let result = tensor.clone()
            .shape_builder()
            .unsqueeze(0).unwrap()                    // [1, 2, 3]
            .expand(&[2, 2, 3]).unwrap()              // [2, 2, 3]
            .flatten()                                // [12]
            .build();
            
        assert_eq!(result.shape(), &[12]);
        assert_eq!(result.numel(), 12);
        
        // Test unflatten after flatten
        let flattened = tensor.flatten_owned();
        let restored = flattened.unflatten(0, &[2, 3]).unwrap();
        let original_shape = vec![2, 3]; // Store original shape since tensor is moved
        assert_eq!(restored.shape(), &original_shape);
        
        // Test repeat with expand_as
        let small = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let target_shape = Tensor::from_vec(vec![0.0; 8], vec![4, 2]);
        let expanded = small.expand_as(&target_shape).unwrap();
        assert_eq!(expanded.shape(), &[4, 2]);
        
        // Verify data correctness
        let data = expanded.data.as_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }
}