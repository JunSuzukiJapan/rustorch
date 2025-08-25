//! Modular tensor operations
//! モジュール化されたテンソル演算
//!
//! This module contains tensor operations organized by functionality:
//! - `arithmetic`: Basic arithmetic operations (+, -, *, /)
//! - `matrix`: Matrix operations (matmul, transpose, det, etc.)
//! - `statistical`: Statistical operations (mean, std, sum, etc.)
//! - `utilities`: Utility operations (reshape, stack, random generation, etc.)
//!
//! このモジュールには機能別に整理されたテンソル演算が含まれています：
//! - `arithmetic`: 基本算術演算 (+, -, *, /)
//! - `matrix`: 行列演算 (matmul, transpose, det など)
//! - `statistical`: 統計演算 (mean, std, sum など)
//! - `utilities`: ユーティリティ演算 (reshape, stack, ランダム生成など)

pub mod arithmetic;
pub mod matrix;
pub mod statistical;
pub mod utilities;

// Re-export commonly used functions for convenience
// よく使用される関数を利便性のために再エクスポート
// Note: Specific re-exports will be enabled when legacy migration is complete
// 注意: レガシー移行完了時に特定の再エクスポートを有効化
// pub use arithmetic::*;
// pub use matrix::*;
// pub use statistical::*;
// pub use utilities::*;

#[cfg(test)]
mod integration_tests {
    use super::super::core::Tensor;

    #[test]
    fn test_operation_chaining() {
        // Test that operations from different modules work together
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
        
        // Chain operations: addition -> matrix multiplication -> transpose -> mean
        let result = a.add_v2(&b).unwrap()
            .matmul_v2(&a).unwrap()
            .transpose_v2().unwrap();
        
        let mean: f32 = result.mean_v2();
        
        assert!(result.shape().len() == 2);
        assert!(mean.is_finite());
    }

    #[test]
    fn test_statistical_operations_integration() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        // Test statistical operations
        assert_eq!(tensor.sum_v2(), 21.0);
        assert_eq!(tensor.mean_v2(), 3.5);
        assert_eq!(tensor.min(), 1.0);
        assert_eq!(tensor.max(), 6.0);
        
        // Test axis operations
        let sum_axis0 = tensor.sum_axis_v2(0).unwrap();
        assert_eq!(sum_axis0.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matrix_operations_integration() {
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        
        // Test matrix operations
        let transposed = matrix.transpose_v2().unwrap();
        let matmul_result = matrix.matmul_v2(&transposed).unwrap();
        
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(matmul_result.shape(), &[2, 2]);
        
        // Test trace and determinant
        let trace = matrix.trace().unwrap();
        let det = matrix.det().unwrap();
        
        assert_eq!(trace, 5.0); // 1 + 4
        assert_eq!(det, -2.0);  // 1*4 - 2*3
    }

    #[test]
    fn test_utility_operations_integration() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        
        // Test stacking and concatenation
        let stacked = Tensor::stack_v2(&[&a, &b]).unwrap();
        assert_eq!(stacked.shape(), &[2, 2]);
        
        let concatenated = Tensor::concatenate_v2(&[&a, &b], 0).unwrap();
        assert_eq!(concatenated.shape(), &[4]);
        
        // Test reshape operations
        let reshaped = stacked.reshape_v2(&[4]).unwrap();
        assert_eq!(reshaped.shape(), &[4]);
        
        let squeezed = reshaped.unsqueeze_v2(0).unwrap().squeeze_v2();
        assert_eq!(squeezed.shape(), &[4]);
    }
}