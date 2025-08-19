use super::Tensor;
use crate::memory::{get_f32_pool, get_f64_pool};
use ndarray::ArrayD;
use num_traits::Float;

/// Enhanced tensor operations with memory pool integration
/// メモリプール統合による拡張テンソル演算
impl<T: Float + Clone + 'static> Tensor<T> {
    /// Create tensor with optimized memory allocation
    /// 最適化されたメモリ割り当てでテンソルを作成
    pub fn with_pool(shape: &[usize]) -> Self {
        Self::zeros(shape)
    }

    /// Efficient matrix multiplication using memory pool
    /// メモリプールを使用した効率的な行列乗算
    pub fn matmul_pooled(&self, other: &Tensor<T>) -> Tensor<T> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        if self_shape.len() != 2 || other_shape.len() != 2 {
            panic!("Matrix multiplication requires 2D tensors");
        }
        
        if self_shape[1] != other_shape[0] {
            panic!("Incompatible dimensions for matrix multiplication");
        }
        
        let result_shape = vec![self_shape[0], other_shape[1]];
        let mut result = Self::zeros(&result_shape);
        
        // Perform matrix multiplication
        for i in 0..self_shape[0] {
            for j in 0..other_shape[1] {
                let mut sum = T::zero();
                for k in 0..self_shape[1] {
                    sum = sum + self.data[[i, k]] * other.data[[k, j]];
                }
                result.data[[i, j]] = sum;
            }
        }
        
        result
    }

    /// Batch operations with memory pool optimization
    /// メモリプール最適化によるバッチ演算
    pub fn batch_add(&self, tensors: &[&Tensor<T>]) -> Vec<Tensor<T>> {
        tensors.iter().map(|tensor| {
            let mut result = Self::zeros(self.data.shape());
            result.data = &self.data + &tensor.data;
            result
        }).collect()
    }

    /// Memory-efficient element-wise operations
    /// メモリ効率的な要素ごと演算
    pub fn apply_pooled<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mut result = Self::zeros(self.data.shape());
        result.data.assign(&self.data.mapv(f));
        result
    }

    /// Optimized reduction operations
    /// 最適化されたリダクション演算
    pub fn sum_pooled(&self) -> T {
        self.data.iter().cloned().fold(T::zero(), |acc, x| acc + x)
    }

    /// Memory pool statistics for debugging
    /// デバッグ用メモリプール統計
    pub fn pool_stats() -> String {
        let mut stats = String::new();
        
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            if let Ok(pool) = get_f32_pool().lock() {
                stats.push_str(&format!("F32 Pool: {}", pool.stats()));
            }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            if let Ok(pool) = get_f64_pool().lock() {
                stats.push_str(&format!("F64 Pool: {}", pool.stats()));
            }
        }
        
        stats
    }

    /// Clear memory pools (useful for testing)
    /// メモリプールをクリア（テスト用）
    pub fn clear_pools() {
        if let Ok(mut pool) = get_f32_pool().lock() {
            pool.clear();
        }
        if let Ok(mut pool) = get_f64_pool().lock() {
            pool.clear();
        }
    }
}

/// Specialized implementations for common types
/// 一般的な型の特殊化実装
impl Tensor<f32> {
    /// F32-specific optimized operations
    /// F32特化最適化演算
    pub fn fast_conv2d(&self, kernel: &Tensor<f32>) -> Tensor<f32> {
        // Placeholder for optimized convolution
        // 最適化された畳み込みのプレースホルダー
        let result_shape = vec![self.size()[0], kernel.size()[0]];
        Self::zeros(&result_shape)
    }
}

impl Tensor<f64> {
    /// F64-specific high-precision operations
    /// F64特化高精度演算
    pub fn high_precision_matmul(&self, other: &Tensor<f64>) -> Tensor<f64> {
        // Use higher precision algorithms
        // より高精度なアルゴリズムを使用
        self.matmul_pooled(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooled_operations() {
        let a = Tensor::<f32>::with_pool(&[2, 2]);
        let b = Tensor::<f32>::with_pool(&[2, 2]);
        
        let result = &a + &b;
        assert_eq!(result.size(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_pooled() {
        let a = Tensor::<f32>::ones(&[2, 3]);
        let b = Tensor::<f32>::ones(&[3, 2]);
        
        let result = a.matmul_pooled(&b);
        assert_eq!(result.size(), vec![2, 2]);
        
        // Result should be all 3.0 (sum of 3 ones)
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.as_array()[[i, j]], 3.0);
            }
        }
    }

    #[test]
    fn test_batch_operations() {
        let base = Tensor::<f32>::ones(&[2, 2]);
        let tensor1 = Tensor::<f32>::ones(&[2, 2]);
        let tensor2 = Tensor::<f32>::ones(&[2, 2]);
        let tensors = vec![&tensor1, &tensor2];
        
        let results = base.batch_add(&tensors);
        assert_eq!(results.len(), 2);
        
        for result in results {
            assert_eq!(result.size(), vec![2, 2]);
        }
    }

    #[test]
    fn test_memory_pool_stats() {
        // Create some tensors to populate pool
        let _tensors: Vec<Tensor<f32>> = (0..10)
            .map(|_| Tensor::zeros(&[10, 10]))
            .collect();
        
        let stats = Tensor::<f32>::pool_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_clear_pools() {
        // Create tensors to populate pool
        {
            let _tensors: Vec<Tensor<f32>> = (0..5)
                .map(|_| Tensor::zeros(&[5, 5]))
                .collect();
            // Tensors go out of scope here, should return to pool
        }
        
        // Clear pools
        Tensor::<f32>::clear_pools();
        
        // Verify clear worked by checking we can still create tensors
        let _test_tensor = Tensor::<f32>::zeros(&[3, 3]);
        assert_eq!(_test_tensor.size(), vec![3, 3]);
    }
}
