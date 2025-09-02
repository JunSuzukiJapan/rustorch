/// SIMD-optimized operations for tensor computations
/// テンソル計算のためのSIMD最適化演算
pub mod ops;
/// SIMD operation traits and auto-selection
/// SIMD操作トレイトと自動選択
pub mod traits;
/// Vectorized SIMD operations (AVX2, SSE4.1)
/// ベクトル化SIMD操作（AVX2、SSE4.1）
pub mod vectorized;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];
        let expected = vec![9.0; 8];

        ops::add_optimized(&a, &b, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_multiplication() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        let expected = vec![2.0, 6.0, 12.0, 20.0];

        ops::mul_optimized(&a, &b, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];
        let expected = vec![2.0, 4.0, 6.0, 8.0];

        ops::mul_scalar_optimized(&a, 2.0, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product() {
        if vectorized::is_avx2_available() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // = 70.0

            let result = vectorized::dot_product_f32_avx2(&a, &b);
            assert!((result - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_optimization_info() {
        let info = ops::get_optimization_info();
        assert!(info.contains("SIMD Optimization Status"));
        println!("{}", info);
    }

    #[test]
    fn test_large_array_performance() {
        let size = 10000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut result = vec![0.0; size];

        // This should use SIMD if available
        ops::add_optimized(&a, &b, &mut result);

        // Verify correctness
        for i in 0..size {
            assert_eq!(result[i], a[i] + b[i]);
        }
    }
}
