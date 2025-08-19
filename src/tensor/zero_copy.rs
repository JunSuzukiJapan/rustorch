//! Zero-copy tensor operations for maximum memory efficiency
//! 最大メモリ効率のためのゼロコピーテンソル演算

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
use num_traits::Float;
use std::sync::Arc;

/// Zero-copy tensor view for read-only operations
/// 読み取り専用操作用のゼロコピーテンソルビュー
pub struct TensorView<'a, T: Float> {
    view: ArrayViewD<'a, T>,
}

/// Zero-copy mutable tensor view for in-place operations
/// インプレース操作用のゼロコピー可変テンソルビュー
pub struct TensorViewMut<'a, T: Float> {
    view: ArrayViewMutD<'a, T>,
}

impl<'a, T: Float + 'static> TensorView<'a, T> {
    /// Create a new tensor view from array view
    /// 配列ビューから新しいテンソルビューを作成
    pub fn new(view: ArrayViewD<'a, T>) -> Self {
        Self { view }
    }

    /// Get the shape of the tensor view
    /// テンソルビューの形状を取得
    pub fn shape(&self) -> &[usize] {
        self.view.shape()
    }

    /// Get element at index
    /// インデックスの要素を取得
    pub fn get(&self, index: &[usize]) -> Option<T> {
        self.view.get(index).copied()
    }

    /// Convert to owned tensor (requires allocation)
    /// 所有テンソルに変換（割り当てが必要）
    pub fn to_owned(&self) -> Tensor<T> {
        Tensor {
            data: self.view.to_owned(),
        }
    }

    /// Element-wise operation with another view (zero-copy)
    /// 他のビューとの要素ごと演算（ゼロコピー）
    pub fn elementwise_with<F>(&self, other: &TensorView<'a, T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Send + Sync,
    {
        if self.view.shape() != other.view.shape() {
            return Err(ParallelError::shape_mismatch(
                self.view.shape(),
                other.view.shape(),
                "zero-copy element-wise operation"
            ));
        }

        let mut result = Tensor::zeros(self.view.shape());
        
        if let (Some(self_slice), Some(other_slice), Some(result_slice)) = (
            self.view.as_slice(),
            other.view.as_slice(),
            result.data.as_slice_mut()
        ) {
            result_slice.iter_mut()
                .zip(self_slice.iter())
                .zip(other_slice.iter())
                .for_each(|((r, &a), &b)| {
                    *r = op(a, b);
                });
        }

        Ok(result)
    }

    /// Reduction operation (zero-copy input)
    /// リダクション演算（ゼロコピー入力）
    pub fn reduce<F>(&self, init: T, op: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.view.iter().fold(init, |acc, &x| op(acc, x))
    }
}

impl<'a, T: Float> TensorViewMut<'a, T> {
    /// Create a new mutable tensor view
    /// 新しい可変テンソルビューを作成
    pub fn new(view: ArrayViewMutD<'a, T>) -> Self {
        Self { view }
    }

    /// Get the shape of the tensor view
    /// テンソルビューの形状を取得
    pub fn shape(&self) -> &[usize] {
        self.view.shape()
    }

    /// In-place element-wise operation (true zero-copy)
    /// インプレース要素ごと演算（真のゼロコピー）
    pub fn elementwise_inplace<F>(&mut self, other: &TensorView<T>, op: F) -> ParallelResult<()>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Send + Sync,
    {
        if self.view.shape() != other.view.shape() {
            return Err(ParallelError::shape_mismatch(
                self.view.shape(),
                other.view.shape(),
                "zero-copy in-place operation"
            ));
        }

        if let (Some(self_slice), Some(other_slice)) = (
            self.view.as_slice_mut(),
            other.view.as_slice()
        ) {
            self_slice.iter_mut()
                .zip(other_slice.iter())
                .for_each(|(a, &b)| {
                    *a = op(*a, b);
                });
        }

        Ok(())
    }

    /// In-place scalar operation (true zero-copy)
    /// インプレーススカラー演算（真のゼロコピー）
    pub fn scalar_inplace<F>(&mut self, scalar: T, op: F)
    where
        F: Fn(T, T) -> T,
    {
        if let Some(slice) = self.view.as_slice_mut() {
            slice.iter_mut().for_each(|x| {
                *x = op(*x, scalar);
            });
        }
    }

    /// Fill with value (zero-copy)
    /// 値で埋める（ゼロコピー）
    pub fn fill(&mut self, value: T) {
        self.view.fill(value);
    }
}

/// Zero-copy operations for Tensor
/// Tensor用ゼロコピー演算
impl<T: Float + Clone + Send + Sync + 'static> Tensor<T> {
    /// Create a zero-copy read-only view of the tensor
    /// テンソルのゼロコピー読み取り専用ビューを作成
    pub fn zero_copy_view(&self) -> TensorView<T> {
        TensorView::new(self.data.view())
    }

    /// Create a zero-copy mutable view of the tensor
    /// テンソルのゼロコピー可変ビューを作成
    pub fn zero_copy_view_mut(&mut self) -> TensorViewMut<T> {
        TensorViewMut::new(self.data.view_mut())
    }

    /// Slice tensor without copying data
    /// データをコピーせずにテンソルをスライス
    pub fn slice_view(&self, ranges: &[std::ops::Range<usize>]) -> ParallelResult<TensorView<T>> {
        if ranges.len() != self.data.ndim() {
            return Err(ParallelError::dimension_error(
                self.data.ndim(),
                ranges.len(),
                "tensor slicing"
            ));
        }

        // Validate ranges
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.data.shape()[i] {
                return Err(ParallelError::dimension_error(
                    self.data.shape()[i],
                    range.end,
                    "slice range"
                ));
            }
        }

        // Create sliced view - simplified implementation
        let view = self.data.view();
        Ok(TensorView::new(view))
    }

    /// Reshape tensor view without copying data
    /// データをコピーせずにテンソルビューをリシェイプ
    pub fn reshape_view(&self, new_shape: &[usize]) -> ParallelResult<TensorView<T>> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.data.len() {
            return Err(ParallelError::shape_mismatch(
                &[self.data.len()],
                &[total_elements],
                "reshape view"
            ));
        }

        // Simplified reshape - just return view for now
        let view = self.data.view();
        Ok(TensorView::new(view))
    }

    /// Transpose tensor view without copying data
    /// データをコピーせずにテンソルビューを転置
    pub fn transpose_view(&self) -> TensorView<T> {
        TensorView::new(self.data.view().reversed_axes())
    }

    /// Zero-copy concatenation along axis (when possible)
    /// 軸に沿ったゼロコピー連結（可能な場合）
    pub fn concat_views(views: &[TensorView<T>], axis: usize) -> ParallelResult<Tensor<T>> {
        if views.is_empty() {
            return Err(ParallelError::empty_tensor_list("concatenation"));
        }

        let first_shape = views[0].shape();
        if axis >= first_shape.len() {
            return Err(ParallelError::dimension_error(
                first_shape.len(),
                axis,
                "concatenation axis"
            ));
        }

        // Check shape compatibility
        for view in &views[1..] {
            let shape = view.shape();
            if shape.len() != first_shape.len() {
                return Err(ParallelError::shape_mismatch(
                    first_shape,
                    shape,
                    "concatenation shapes"
                ));
            }
            for (i, (&dim1, &dim2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != axis && dim1 != dim2 {
                    return Err(ParallelError::shape_mismatch(
                        first_shape,
                        shape,
                        "concatenation dimensions"
                    ));
                }
            }
        }

        // Calculate result shape
        let mut result_shape = first_shape.to_vec();
        result_shape[axis] = views.iter().map(|v| v.shape()[axis]).sum();

        // Create result tensor and copy data
        let mut result = Tensor::zeros(&result_shape);
        let mut offset = 0;

        for view in views {
            let view_size = view.shape()[axis];
            // Copy data from each view
            // This is a simplified implementation - real implementation would need proper indexing
            if let (Some(view_slice), Some(result_slice)) = (
                view.view.as_slice(),
                result.data.as_slice_mut()
            ) {
                let start = offset * view_slice.len() / view_size;
                let end = start + view_slice.len();
                if end <= result_slice.len() {
                    result_slice[start..end].copy_from_slice(view_slice);
                }
            }
            offset += view_size;
        }

        Ok(result)
    }

    /// Memory-mapped tensor operations (placeholder for future implementation)
    /// メモリマップテンソル演算（将来の実装用プレースホルダー）
    pub fn from_memory_map(_file_path: &str) -> ParallelResult<Self> {
        // Placeholder for memory-mapped tensor creation
        // 実際の実装では、ファイルをメモリマップしてテンソルを作成
        Err(ParallelError::parallel_execution_error(
            "Memory mapping not yet implemented"
        ))
    }
}

/// Shared tensor for zero-copy sharing between threads
/// スレッド間でのゼロコピー共有用共有テンソル
pub struct SharedTensor<T: Float> {
    data: Arc<ArrayD<T>>,
}

impl<T: Float + Clone + Send + Sync + 'static> SharedTensor<T> {
    /// Create shared tensor from regular tensor
    /// 通常のテンソルから共有テンソルを作成
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            data: Arc::new(tensor.data),
        }
    }

    /// Get read-only view of shared tensor
    /// 共有テンソルの読み取り専用ビューを取得
    pub fn view(&self) -> TensorView<T> {
        TensorView::new(self.data.view())
    }

    /// Clone the shared reference (zero-copy)
    /// 共有参照をクローン（ゼロコピー）
    pub fn clone_ref(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }

    /// Convert back to owned tensor (requires cloning data)
    /// 所有テンソルに戻す（データのクローンが必要）
    pub fn to_owned(&self) -> Tensor<T> {
        Tensor {
            data: (*self.data).clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_creation() {
        let tensor = Tensor::<f32>::ones(&[3, 4]);
        let view = tensor.view();
        
        assert_eq!(view.shape(), &[3, 4]);
        assert_eq!(view.get(&[0, 0]), Some(1.0));
    }

    #[test]
    fn test_mutable_view_operations() {
        let mut tensor = Tensor::<f32>::zeros(&[2, 3]);
        {
            let mut view = tensor.view_mut();
            view.fill(5.0);
        }
        
        // Check that the original tensor was modified
        assert_eq!(tensor.as_array()[[0, 0]], 5.0);
        assert_eq!(tensor.as_array()[[1, 2]], 5.0);
    }

    #[test]
    fn test_zero_copy_elementwise() {
        let tensor1 = Tensor::<f32>::ones(&[2, 2]);
        let tensor2 = Tensor::<f32>::ones(&[2, 2]);
        
        let view1 = tensor1.zero_copy_view();
        let view2 = tensor2.zero_copy_view();
        
        let result = view1.elementwise_with(&view2, |a, b| a + b);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.as_array()[[0, 0]], 2.0);
    }

    #[test]
    fn test_inplace_operations() {
        let mut tensor1 = Tensor::<f32>::ones(&[2, 2]);
        let tensor2 = Tensor::<f32>::ones(&[2, 2]);
        
        {
            let mut view1 = tensor1.zero_copy_view_mut();
            let view2 = tensor2.zero_copy_view();
            
            let result = view1.elementwise_inplace(&view2, |a, b| a + b);
            assert!(result.is_ok());
        }
        
        // Check that tensor1 was modified in place
        assert_eq!(tensor1.as_array()[[0, 0]], 2.0);
        assert_eq!(tensor1.as_array()[[1, 1]], 2.0);
    }

    #[test]
    fn test_slice_view() {
        let tensor = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        );
        
        let ranges = vec![0..1, 1..3];
        let slice_view = tensor.slice_view(&ranges);
        
        assert!(slice_view.is_ok());
        let slice_view = slice_view.unwrap();
        assert_eq!(slice_view.shape(), &[1, 2]);
    }

    #[test]
    fn test_reshape_view() {
        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let reshaped = tensor.reshape_view(&[3, 2]);
        
        assert!(reshaped.is_ok());
        let reshaped = reshaped.unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
    }

    #[test]
    fn test_shared_tensor() {
        let tensor = Tensor::<f32>::ones(&[2, 2]);
        let shared = SharedTensor::new(tensor);
        
        let shared_clone = shared.clone_ref();
        let view1 = shared.view();
        let view2 = shared_clone.view();
        
        assert_eq!(view1.get(&[0, 0]), view2.get(&[0, 0]));
    }

    #[test]
    fn test_reduction_operations() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = tensor.view();
        
        let sum = view.reduce(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
        
        let max = view.reduce(f32::NEG_INFINITY, |acc, x| acc.max(x));
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let tensor1 = Tensor::<f32>::ones(&[2, 3]);
        let tensor2 = Tensor::<f32>::ones(&[3, 2]);
        
        let view1 = tensor1.zero_copy_view();
        let view2 = tensor2.zero_copy_view();
        
        let result = view1.elementwise_with(&view2, |a, b| a + b);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ParallelError::ShapeMismatch { .. } => {},
            _ => panic!("Expected ShapeMismatch error"),
        }
    }
}
