/// Simple demonstration of basic tensor mathematical operations
/// 基本的なテンソル数学演算の簡単なデモンストレーション

use rustorch::tensor::Tensor;
use std::ops::{Add, Mul};

fn main() {
    println!("=== RusTorch Basic Math Functions Demo ===\n");
    
    // Test basic creation and operations
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
    println!("Original tensor: {:?}", tensor.as_slice().unwrap());
    
    // Test existing operations that work
    let sum_result = tensor.sum();
    println!("Sum: {}", sum_result.as_slice().unwrap()[0]);
    
    // Test matrix operations
    let matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("\nMatrix operations:");
    println!("Matrix shape: {:?}", matrix.shape());
    println!("Matrix data: {:?}", matrix.as_slice().unwrap());
    
    let transposed = matrix.transpose();
    println!("Transposed shape: {:?}", transposed.shape());
    // Convert to owned array to get contiguous memory for as_slice
    let transposed_owned = transposed.to_owned();
    if let Some(slice) = transposed_owned.as_slice() {
        println!("Transposed data: {:?}", slice);
    } else {
        println!("Transposed data (non-contiguous): {:?}", transposed.as_array());
    }
    
    // Test element-wise operations using operator overloading
    let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let tensor2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
    
    let added = &tensor1 + &tensor2;
    println!("\nElement-wise operations:");
    println!("Tensor1: {:?}", tensor1.as_slice().unwrap());
    println!("Tensor2: {:?}", tensor2.as_slice().unwrap());
    println!("Added: {:?}", added.as_slice().unwrap());
    
    let multiplied = &tensor1 * &tensor2;
    println!("Multiplied: {:?}", multiplied.as_slice().unwrap());
    
    // Test SIMD scalar operations
    let scaled = tensor1.mul_scalar_simd(2.0);
    println!("Scaled by 2 (SIMD): {:?}", scaled.as_slice().unwrap());
    
    println!("\n=== Demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_operations() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        // Test sum
        let sum_result = tensor.sum();
        assert_abs_diff_eq!(sum_result.as_slice().unwrap()[0], 10.0, epsilon = 1e-6);
        
        // Test addition with operator overloading
        let tensor2 = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], vec![4]);
        let added = &tensor + &tensor2;
        let expected = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(added.as_slice().unwrap(), &expected);
        
        // Test SIMD scalar multiplication
        let scaled = tensor.mul_scalar_simd(2.0);
        let expected_scaled = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(scaled.as_slice().unwrap(), &expected_scaled);
    }
}
