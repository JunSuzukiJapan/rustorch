/// Demonstration of new tensor mathematical operations
/// 新しいテンソル数学演算のデモンストレーション

use rustorch::tensor::Tensor;

fn main() {
    println!("=== RusTorch Tensor Math Operations Demo ===\n");
    
    // 1. Basic Math Functions
    demo_basic_math_functions();
    
    // 2. Broadcasting Operations  
    demo_broadcasting();
    
    // 3. Statistical Functions
    demo_statistics();
    
    // 4. Indexing Operations
    demo_indexing();
    
    println!("=== All tensor math operations completed successfully! ===");
}

fn demo_basic_math_functions() {
    println!("1. Basic Math Functions:");
    
    let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.14159], vec![4]);
    
    // Test sin, cos, exp, log, sqrt
    let sin_result = tensor.sin();
    let cos_result = tensor.cos();
    let exp_result = tensor.exp();
    
    println!("   Original: {:?}", tensor.data.as_slice().unwrap());
    println!("   sin():    {:?}", sin_result.data.as_slice().unwrap());
    println!("   cos():    {:?}", cos_result.data.as_slice().unwrap());
    println!("   exp():    {:?}", exp_result.data.as_slice().unwrap());
    
    // Test sqrt with positive values
    let sqrt_tensor = Tensor::from_vec(vec![0.0f32, 1.0, 4.0, 9.0], vec![4]);
    let sqrt_result = sqrt_tensor.sqrt();
    println!("   sqrt([0,1,4,9]): {:?}", sqrt_result.data.as_slice().unwrap());
    
    // Test power function
    let base_tensor = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], vec![3]);
    let pow_result = base_tensor.pow(2.0);
    println!("   [2,3,4]^2: {:?}", pow_result.data.as_slice().unwrap());
    
    // Test sigmoid
    let sigmoid_tensor = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]);
    let sigmoid_result = sigmoid_tensor.sigmoid();
    println!("   sigmoid([0,1,-1]): {:?}", sigmoid_result.data.as_slice().unwrap());
    
    println!();
}

fn demo_broadcasting() {
    println!("2. Broadcasting Operations:");
    
    let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let tensor2 = Tensor::from_vec(vec![10.0f32], vec![1]);
    
    println!("   Tensor1 shape: {:?}, data: {:?}", tensor1.shape(), tensor1.data.as_slice().unwrap());
    println!("   Tensor2 shape: {:?}, data: {:?}", tensor2.shape(), tensor2.data.as_slice().unwrap());
    
    // Check if broadcasting is possible
    if tensor1.can_broadcast_with(&tensor2) {
        println!("   ✅ Broadcasting is possible");
        
        if let Ok((broadcasted1, broadcasted2)) = tensor1.broadcast_with(&tensor2) {
            println!("   Broadcasted1: {:?}", broadcasted1.data.as_slice().unwrap());
            println!("   Broadcasted2: {:?}", broadcasted2.data.as_slice().unwrap());
        }
    } else {
        println!("   ❌ Broadcasting not possible");
    }
    
    // Test unsqueeze and squeeze
    let unsqueezed = tensor1.unsqueeze(0).unwrap();
    println!("   Unsqueezed shape: {:?}", unsqueezed.shape());
    
    let squeezed = unsqueezed.squeeze();
    println!("   Squeezed shape: {:?}", squeezed.shape());
    
    println!();
}

fn demo_statistics() {
    println!("3. Statistical Functions:");
    
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
    
    // Test variance and standard deviation
    let var_result = tensor.var(None, false);
    let std_result = tensor.std(None, false);
    
    println!("   Data: {:?}", tensor.data.as_slice().unwrap());
    println!("   Variance: {:.4}", var_result.data[[0]]);
    println!("   Std Dev:  {:.4}", std_result.data[[0]]);
    
    // Test median
    let median_result = tensor.median(None);
    println!("   Median:   {:.4}", median_result.data[[0]]);
    
    // Test quantiles
    let q25 = tensor.quantile(0.25, None);
    let q75 = tensor.quantile(0.75, None);
    println!("   25th percentile: {:.4}", q25.data[[0]]);
    println!("   75th percentile: {:.4}", q75.data[[0]]);
    
    // Test cumulative operations
    if let Ok(cumsum_result) = tensor.cumsum(0) {
        println!("   Cumsum: {:?}", cumsum_result.data.as_slice().unwrap());
    }
    
    if let Ok(cumprod_result) = tensor.cumprod(0) {
        println!("   Cumprod: {:?}", cumprod_result.data.as_slice().unwrap());
    }
    
    // Test histogram
    let (counts, edges) = tensor.histogram(3, Some((1.0, 5.0)));
    println!("   Histogram counts: {:?}", counts.data.as_slice().unwrap());
    println!("   Histogram edges:  {:?}", edges.data.as_slice().unwrap());
    
    println!();
}

fn demo_indexing() {
    println!("4. Indexing Operations:");
    
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    
    println!("   Original tensor shape: {:?}", tensor.shape());
    println!("   Data: {:?}", tensor.data.as_slice().unwrap());
    
    // Test get_item and set_item
    if let Ok(value) = tensor.get_item(&[0, 1]) {
        println!("   Element at [0,1]: {}", value);
    }
    
    // Test select operation
    if let Ok(selected) = tensor.select(1, &[0, 2]) {
        println!("   Selected columns [0,2]: {:?}", selected.data.as_slice().unwrap());
        println!("   Selected shape: {:?}", selected.shape());
    }
    
    // Test masked selection
    let mask = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]);
    if let Ok(masked) = tensor.masked_select(&mask) {
        println!("   Masked selection: {:?}", masked.data.as_slice().unwrap());
    }
    
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tensor_math_demo() {
        // Test basic math functions
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0], vec![2]);
        let sin_result = tensor.sin();
        assert_abs_diff_eq!(sin_result.data[[0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sin_result.data[[1]], 1.0f32.sin(), epsilon = 1e-6);
        
        // Test broadcasting
        let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let tensor2 = Tensor::from_vec(vec![10.0f32], vec![1]);
        assert!(tensor1.can_broadcast_with(&tensor2));
        
        // Test statistics
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let median = tensor.median(None);
        assert_abs_diff_eq!(median.data[[0]], 2.0, epsilon = 1e-6);
        
        // Test indexing
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_abs_diff_eq!(tensor.get_item(&[0, 0]).unwrap(), 1.0, epsilon = 1e-6);
    }
}
