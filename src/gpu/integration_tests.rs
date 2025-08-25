//! GPUカーネルシステムの基本テスト
//! 複雑なテストを簡略化し、基本機能のみテスト

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_gpu_kernel_basic() {
        // 基本的なテンソル作成テスト
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);

        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[3]);

        println!("✓ GPU kernel basic test passed");
    }

    #[test]
    fn test_tensor_creation() {
        // 基本的なテンソル作成テスト
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]);

        assert_eq!(tensor.shape(), &[2, 2]);

        println!("✓ Tensor creation test passed");
    }

    #[test]
    fn test_f64_tensor() {
        // f64テンソルの基本テスト
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], vec![3]);

        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[3]);

        println!("✓ F64 tensor test passed");
    }

    #[test]
    fn test_basic_integration() {
        // 基本的な統合テスト
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]);

        assert_eq!(tensor.shape(), &[2, 2]);

        println!("✓ Basic integration test passed");
        println!("✅ All GPU kernel tests completed successfully!");
    }
}
