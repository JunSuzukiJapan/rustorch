//! CoreML linear algebra operations
//! CoreML線形代数演算

use super::*;
use crate::gpu::coreml::common::coreml_feature;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// CoreML Linear Algebra operations trait
/// CoreML線形代数演算トレイト
pub trait CoreMLLinearAlgebra<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// CoreML matrix multiplication
    /// CoreML行列乗算
    fn coreml_matmul(&self, other: &Self) -> CoreMLResult<Tensor<T>>;

    /// CoreML batch matrix multiplication
    /// CoreMLバッチ行列乗算
    fn coreml_batch_matmul(&self, other: &Self) -> CoreMLResult<Tensor<T>>;

    /// CoreML matrix-vector multiplication
    /// CoreML行列ベクトル乗算
    fn coreml_matvec(&self, vector: &Self) -> CoreMLResult<Tensor<T>>;

    /// CoreML transpose operation
    /// CoreML転置演算
    fn coreml_transpose(&self) -> CoreMLResult<Tensor<T>>;
}

/// Matrix multiplication operation for CoreML
/// CoreML用行列乗算演算
pub struct MatMulOperation<T: Float> {
    lhs: Tensor<T>,
    rhs: Tensor<T>,
}

impl<T> MatMulOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// Create new matrix multiplication operation
    /// 新しい行列乗算演算を作成
    pub fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> Self {
        Self { lhs, rhs }
    }

    /// Validate matrix dimensions for multiplication
    /// 乗算用の行列次元を検証
    fn validate_dimensions(&self) -> CoreMLResult<()> {
        let lhs_shape = self.lhs.shape();
        let rhs_shape = self.rhs.shape();

        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            return Err(error_helpers::unsupported_operation(
                "Matrix multiplication requires at least 2D tensors",
            ));
        }

        let lhs_cols = lhs_shape[lhs_shape.len() - 1];
        let rhs_rows = rhs_shape[rhs_shape.len() - 2];

        if lhs_cols != rhs_rows {
            return Err(error_helpers::tensor_op_error(&format!(
                "Matrix dimension mismatch: {} x {} cannot multiply with {} x {}",
                lhs_shape[lhs_shape.len() - 2],
                lhs_cols,
                rhs_rows,
                rhs_shape[rhs_shape.len() - 1]
            )));
        }

        Ok(())
    }

    /// Check if operation size is suitable for CoreML
    /// 演算サイズがCoreMLに適しているかチェック
    fn is_suitable_for_coreml(&self) -> bool {
        let lhs_shape = self.lhs.shape();
        let rhs_shape = self.rhs.shape();

        // Calculate total elements in both matrices
        let lhs_elements: usize = lhs_shape.iter().product();
        let rhs_elements: usize = rhs_shape.iter().product();

        // CoreML is efficient for larger matrices (> 1024 elements each)
        lhs_elements > 1024 && rhs_elements > 1024
    }
}

impl<T> CoreMLOperation<T> for MatMulOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn execute_coreml(&self, device_id: usize) -> CoreMLResult<Tensor<T>> {
        self.validate_dimensions()?;

        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            // Use CoreML backend for actual computation
            use crate::gpu::coreml::backend::CoreMLGraph;

            let graph = CoreMLGraph::new(device_id)?;
            return graph.matmul(&self.lhs, &self.rhs);
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            Err(error_helpers::feature_disabled())
        }
    }

    fn is_supported_by_coreml(&self) -> bool {
        // Check if dimensions are valid and size is suitable
        self.validate_dimensions().is_ok() && self.is_suitable_for_coreml()
    }

    fn estimated_execution_time(&self) -> Option<std::time::Duration> {
        if !self.is_suitable_for_coreml() {
            return None;
        }

        let lhs_shape = self.lhs.shape();
        let rhs_shape = self.rhs.shape();

        // Rough estimation based on matrix sizes
        let m = lhs_shape[lhs_shape.len() - 2];
        let n = rhs_shape[rhs_shape.len() - 1];
        let k = lhs_shape[lhs_shape.len() - 1];

        // Simple heuristic: ~1ns per FLOP on CoreML
        let flops = 2 * m * n * k; // 2 operations per element (multiply + add)
        Some(std::time::Duration::from_nanos(flops as u64))
    }
}

/// Implement CoreML linear algebra for Tensor
/// TensorにCoreML線形代数を実装
impl<T> CoreMLLinearAlgebra<T> for Tensor<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn coreml_matmul(&self, other: &Self) -> CoreMLResult<Tensor<T>> {
        let operation = MatMulOperation::new(self.clone(), other.clone());
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_batch_matmul(&self, other: &Self) -> CoreMLResult<Tensor<T>> {
        // For now, delegate to regular matmul
        // TODO: Implement optimized batch multiplication
        self.coreml_matmul(other)
    }

    fn coreml_matvec(&self, vector: &Self) -> CoreMLResult<Tensor<T>> {
        // Validate vector shape
        let vec_shape = vector.shape();
        if vec_shape.len() != 1 && vec_shape.len() != 2 {
            return Err(error_helpers::unsupported_operation(
                "Matrix-vector multiplication requires 1D or 2D vector",
            ));
        }

        self.coreml_matmul(vector)
    }

    fn coreml_transpose(&self) -> CoreMLResult<Tensor<T>> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(error_helpers::unsupported_operation(
                "Transpose requires at least 2D tensor",
            ));
        }

        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            // Use CoreML backend for transpose
            use crate::gpu::coreml::backend::CoreMLGraph;

            let graph = CoreMLGraph::new(0)?;
            // TODO: Implement transpose in CoreML backend
            // For now, fallback to CPU implementation
            return self.transpose();
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            Err(error_helpers::feature_disabled())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_operation_validation() {
        let a = Tensor::<f32>::zeros(&[64, 64]); // 4096 elements > 1024
        let b = Tensor::<f32>::zeros(&[64, 64]); // 4096 elements > 1024
        let operation = MatMulOperation::new(a, b);

        assert!(operation.validate_dimensions().is_ok());
        assert!(operation.is_supported_by_coreml());
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Tensor::<f32>::zeros(&[2, 3]);
        let b = Tensor::<f32>::zeros(&[4, 5]); // Wrong dimensions
        let operation = MatMulOperation::new(a, b);

        assert!(operation.validate_dimensions().is_err());
        assert!(!operation.is_supported_by_coreml());
    }

    #[test]
    fn test_small_matrix_not_suitable() {
        let a = Tensor::<f32>::zeros(&[2, 2]); // 4 elements
        let b = Tensor::<f32>::zeros(&[2, 2]); // 4 elements
        let operation = MatMulOperation::new(a, b);

        assert!(operation.validate_dimensions().is_ok());
        assert!(!operation.is_suitable_for_coreml()); // Too small for CoreML efficiency
    }

    #[test]
    fn test_large_matrix_suitable() {
        let a = Tensor::<f32>::zeros(&[64, 64]); // 4096 elements
        let b = Tensor::<f32>::zeros(&[64, 64]); // 4096 elements
        let operation = MatMulOperation::new(a, b);

        assert!(operation.validate_dimensions().is_ok());
        assert!(operation.is_suitable_for_coreml());
    }

    #[test]
    fn test_execution_time_estimation() {
        let a = Tensor::<f32>::zeros(&[100, 100]);
        let b = Tensor::<f32>::zeros(&[100, 100]);
        let operation = MatMulOperation::new(a, b);

        let estimated_time = operation.estimated_execution_time();
        assert!(estimated_time.is_some());

        let time = estimated_time.unwrap();
        assert!(time.as_nanos() > 0);
    }
}
