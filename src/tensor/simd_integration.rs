use super::Tensor;
use num_traits::Float;

/// SIMD integration utilities for tensor operations
/// テンソル演算のSIMD統合ユーティリティ  
impl<T: Float + Clone + 'static> Tensor<T> {
    /// Check if tensor is suitable for SIMD optimization
    /// テンソルがSIMD最適化に適しているかチェック
    pub fn check_simd_suitability(&self) -> bool {
        self.data.len() >= 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_suitability() {
        let small = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let large = Tensor::<f32>::from_vec(vec![1.0; 100], vec![100]);
        
        assert!(!small.check_simd_suitability());
        assert!(large.check_simd_suitability());
    }
}
