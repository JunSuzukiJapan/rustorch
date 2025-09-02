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

// Temporarily disable duplicate modules during consolidation
// 統合作業中に重複モジュールを一時的に無効化
pub mod arithmetic; // Re-enabled - essential basic operations (includes operator overloads)
                    // pub mod creation;      // Methods integrated into core.rs
pub mod mathematical; // New mathematical functions (exp, ln, sin, etc.)
pub mod matrix; // Re-enabled - essential matrix operations
                // pub mod operators; // Disabled - operator overloads moved to arithmetic.rs
pub mod signal; // Signal processing functions
pub mod statistical; // Re-enabled - essential statistical operations
pub mod utilities; // Re-enabled - essential utility operations

// Advanced operations modules
// 高度操作モジュール
pub mod advanced_linalg;
pub mod advanced_math; // Advanced mathematical functions (hyperbolic, inverse trig, special functions)
// pub mod advanced_shape; // Advanced shape operations (expand, repeat, permute) - superseded by shape_operations
pub mod shape_operations; // Ownership-aware shape operations (squeeze, unsqueeze, expand, flatten)
pub mod advanced_stats; // Advanced statistical functions (variance, quantiles, correlations) // Advanced linear algebra (SVD, eigenvalues, norms)

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
        let result = a.add(&b).unwrap().matmul(&a).unwrap().transpose().unwrap();

        let mean: f32 = result.mean();

        assert!(result.shape().len() == 2);
        assert!(mean.is_finite());
    }

    #[test]
    fn test_statistical_operations_integration() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Test statistical operations
        assert_eq!(tensor.sum(), 21.0);
        assert_eq!(tensor.mean(), 3.5);
        assert_eq!(tensor.min(), 1.0);
        assert_eq!(tensor.max(), 6.0);

        // Test axis operations
        let sum_axis0 = tensor.sum_axis(0).unwrap();
        assert_eq!(sum_axis0.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matrix_operations_integration() {
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test matrix operations
        let transposed = matrix.transpose().unwrap();
        let matmul_result = matrix.matmul(&transposed).unwrap();

        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(matmul_result.shape(), &[2, 2]);

        // Test trace and determinant
        let trace = matrix.trace().unwrap();
        let det = matrix.det().unwrap();

        assert_eq!(trace, 5.0); // 1 + 4
        assert_eq!(det, -2.0); // 1*4 - 2*3
    }

    #[test]
    fn test_utility_operations_integration() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        // Test stacking and concatenation
        let stacked = Tensor::stack(&[&a, &b]).unwrap();
        assert_eq!(stacked.shape(), &[2, 2]);

        let concatenated = Tensor::concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(concatenated.shape(), &[4]);

        // Test reshape operations
        let reshaped = stacked.reshape(&[4]).unwrap();
        assert_eq!(reshaped.shape(), &[4]);

        let squeezed = reshaped.unsqueeze(0).unwrap().squeeze();
        assert_eq!(squeezed.shape(), &[4]);
    }
}
