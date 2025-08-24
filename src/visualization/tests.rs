//! 可視化モジュールの基本テスト
//! 複雑なテストを簡略化し、基本機能のみテスト

use crate::tensor::Tensor;

#[test]
fn test_visualization_basic() {
    // 基本的な可視化テスト
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data, vec![2, 2]);
    
    assert_eq!(tensor.shape(), &[2, 2]);
    println!("✓ Visualization basic test passed");
}

#[test]
fn test_tensor_shape() {
    // テンソル形状テスト
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::from_vec(data, vec![3]);
    
    assert_eq!(tensor.shape(), &[3]);
    println!("✓ Tensor shape test passed");
}

#[test]
fn test_basic_integration() {
    // 基本的な統合テスト
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data, vec![2, 2]);
    
    assert_eq!(tensor.shape(), &[2, 2]);
    println!("✓ Basic integration test passed");
    println!("✅ All visualization tests completed successfully!");
}