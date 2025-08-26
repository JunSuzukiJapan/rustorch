//! Utility functions and types

use ndarray::{ArrayD, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Converts a vector to a column vector (2D tensor with shape [n, 1])
pub fn to_column_vector<T: Float>(data: Vec<T>) -> ArrayD<T> {
    let n = data.len();
    ArrayD::from_shape_vec(vec![n, 1], data).expect("Failed to create column vector")
}

/// Converts a vector to a row vector (2D tensor with shape [1, n])
pub fn to_row_vector<T: Float>(data: Vec<T>) -> ArrayD<T> {
    let n = data.len();
    ArrayD::from_shape_vec(vec![1, n], data).expect("Failed to create row vector")
}

/// Initializes weights using Kaiming (He) initialization
pub fn kaiming_init<T: Float>(shape: &[usize], fan_in: usize) -> Vec<T> {
    let bound = T::from(1.0).unwrap() / T::from(fan_in as f32).unwrap().sqrt();
    (0..shape.iter().product())
        .map(|_| T::from(rand::random::<f32>()).unwrap() * (bound + bound) - bound)
        .collect()
}

/// Computes the mean squared error between two tensors
pub fn mse_loss<T>(output: &ArrayD<T>, target: &ArrayD<T>) -> T
where
    T: Float + FromPrimitive + ScalarOperand + std::ops::AddAssign + std::iter::Sum,
{
    assert_eq!(
        output.shape(),
        target.shape(),
        "Shapes must match for MSE loss"
    );

    let diff = output - target;
    let squared_diff = &diff * &diff;
    let sum: T = squared_diff.iter().cloned().sum();
    let count = T::from_usize(squared_diff.len()).expect("Failed to convert length to float");
    sum / count
}

/// Computes the softmax of a 1D tensor
pub fn softmax<T>(x: &ArrayD<T>) -> ArrayD<T>
where
    T: Float + ScalarOperand + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
{
    let max = x.fold(T::neg_infinity(), |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_to_column_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let col = to_column_vector(v);
        assert_eq!(col.shape(), &[3, 1]);
    }

    #[test]
    fn test_mse_loss() {
        let a = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let b = ArrayD::from_shape_vec(vec![2], vec![1.5, 1.5]).unwrap();
        let loss = mse_loss(&a, &b);
        assert_abs_diff_eq!(loss, 0.25, epsilon = 1e-5);
    }
}
