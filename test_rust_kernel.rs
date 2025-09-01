// Test script to validate Rust kernel imports work correctly
// This simulates the code that was failing in the Jupyter notebook

use rustorch::*;
use ndarray::prelude::*;
use ndarray::array;

fn main() {
    println!("🧪 Testing Rust kernel imports...");
    
    // This is the exact code that was failing: array! macro
    let a = Tensor::from_array(array![[1.0, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_array(array![[5.0, 6.0], [7.0, 8.0]]);
    
    println!("✅ array! macro works correctly");
    println!("Tensor a: {:?}", a);
    println!("Tensor b: {:?}", b);
    
    // Test basic operations
    let result = a.matmul(&b);
    println!("Matrix multiplication result: {:?}", result);
    
    println!("🎉 All imports and operations work correctly!");
}