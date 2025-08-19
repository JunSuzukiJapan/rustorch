/// Demonstration of tensor mathematical operations
/// テンソル数学演算のデモンストレーション

use rustorch::tensor::Tensor;

fn main() {
    println!("=== RusTorch Math Operations Demo ===\n");
    
    // Test basic mathematical functions
    let tensor = Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], vec![4]);
    println!("Original tensor: {:?}", tensor.as_slice().unwrap());
    
    // Test trigonometric functions
    let sin_result = tensor.sin();
    println!("Sin: {:?}", sin_result.as_slice().unwrap());
    
    let cos_result = tensor.cos();
    println!("Cos: {:?}", cos_result.as_slice().unwrap());
    
    let tan_result = tensor.tan();
    println!("Tan: {:?}", tan_result.as_slice().unwrap());
    
    // Test exponential and logarithmic functions
    let exp_result = tensor.exp();
    println!("Exp: {:?}", exp_result.as_slice().unwrap());
    
    let log_result = tensor.log();
    println!("Log: {:?}", log_result.as_slice().unwrap());
    
    let sqrt_result = tensor.sqrt();
    println!("Sqrt: {:?}", sqrt_result.as_slice().unwrap());
    
    // Test power function
    let pow_result = tensor.pow(2.0);
    println!("Power (^2): {:?}", pow_result.as_slice().unwrap());
    
    // Test absolute value
    let neg_tensor = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let abs_result = neg_tensor.abs();
    println!("Abs of [-2, -1, 0, 1, 2]: {:?}", abs_result.as_slice().unwrap());
    
    // Test activation functions
    let activation_tensor = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let sigmoid_result = activation_tensor.sigmoid();
    println!("Sigmoid: {:?}", sigmoid_result.as_slice().unwrap());
    
    let tanh_result = activation_tensor.tanh();
    println!("Tanh: {:?}", tanh_result.as_slice().unwrap());
    
    // Test element-wise operations with two tensors
    let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let tensor2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
    
    let max_result = tensor1.max_elementwise(&tensor2);
    println!("Element-wise max of [1,2,3] and [4,5,6]: {:?}", max_result.as_slice().unwrap());
    
    let min_result = tensor1.min_elementwise(&tensor2);
    println!("Element-wise min of [1,2,3] and [4,5,6]: {:?}", min_result.as_slice().unwrap());
    
    // Test clamp function
    let clamp_tensor = Tensor::from_vec(vec![-3.0f32, -1.0, 0.5, 2.0, 5.0], vec![5]);
    let clamped = clamp_tensor.clamp(-1.0, 3.0);
    println!("Clamped [-3,-1,0.5,2,5] to [-1,3]: {:?}", clamped.as_slice().unwrap());
    
    println!("\n=== Math operations demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_trigonometric_functions() {
        let tensor = Tensor::from_vec(vec![0.0f32, std::f32::consts::PI / 2.0], vec![2]);
        
        let sin_result = tensor.sin();
        let sin_values = sin_result.as_slice().unwrap();
        assert_abs_diff_eq!(sin_values[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sin_values[1], 1.0, epsilon = 1e-6);
        
        let cos_result = tensor.cos();
        let cos_values = cos_result.as_slice().unwrap();
        assert_abs_diff_eq!(cos_values[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cos_values[1], 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_exponential_functions() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0], vec![2]);
        
        let exp_result = tensor.exp();
        let exp_values = exp_result.as_slice().unwrap();
        assert_abs_diff_eq!(exp_values[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(exp_values[1], std::f32::consts::E, epsilon = 1e-6);
        
        let log_result = tensor.exp().log();
        let log_values = log_result.as_slice().unwrap();
        assert_abs_diff_eq!(log_values[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(log_values[1], 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_power_and_sqrt() {
        let tensor = Tensor::from_vec(vec![4.0f32, 9.0, 16.0], vec![3]);
        
        let sqrt_result = tensor.sqrt();
        let sqrt_values = sqrt_result.as_slice().unwrap();
        assert_abs_diff_eq!(sqrt_values[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sqrt_values[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sqrt_values[2], 4.0, epsilon = 1e-6);
        
        let pow_result = tensor.pow(0.5);
        let pow_values = pow_result.as_slice().unwrap();
        assert_abs_diff_eq!(pow_values[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pow_values[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pow_values[2], 4.0, epsilon = 1e-6);
    }
}
