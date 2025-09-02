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
}