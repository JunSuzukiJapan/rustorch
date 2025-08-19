use super::vectorized;
use super::traits::{SimdElementwise, SimdReduction, AutoSimd};

/// High-level SIMD operations interface
/// 高レベルSIMD演算インターフェース

/// Optimized element-wise addition with automatic SIMD selection
/// 自動SIMD選択による最適化された要素ごと加算
pub fn add_optimized(a: &[f32], b: &[f32], result: &mut [f32]) {
    AutoSimd::simd_add(a, b, result);
}

/// Optimized element-wise multiplication
/// 最適化された要素ごと乗算
pub fn mul_optimized(a: &[f32], b: &[f32], result: &mut [f32]) {
    AutoSimd::simd_mul(a, b, result);
}

/// Optimized scalar multiplication using SIMD
/// SIMD を使用した最適化されたスカラー乗算
pub fn mul_scalar_optimized(a: &[f32], scalar: f32, result: &mut [f32]) {
    let auto_simd = AutoSimd::new();
    auto_simd.scalar_mul(a, scalar, result);
}

/// CPU feature detection and optimization info
/// CPU機能検出と最適化情報
pub fn get_optimization_info() -> String {
    let mut info = String::new();
    info.push_str("SIMD Optimization Status:\n");
    
    if vectorized::is_avx2_available() {
        info.push_str("✅ AVX2: Available (8x f32, 4x f64 parallel processing)\n");
    } else {
        info.push_str("❌ AVX2: Not available\n");
    }
    
    if vectorized::is_sse41_available() {
        info.push_str("✅ SSE4.1: Available (4x f32 parallel processing)\n");
    } else {
        info.push_str("❌ SSE4.1: Not available\n");
    }
    
    info
}

/// Optimized dot product using SIMD
/// SIMD を使用した最適化された内積
pub fn dot_product_optimized(a: &[f32], b: &[f32]) -> f32 {
    let auto_simd = AutoSimd::new();
    auto_simd.simd_dot(a, b)
}

/// Optimized matrix multiplication
/// 最適化された行列乗算
pub fn matmul_optimized(
    a: &[f32], a_rows: usize, a_cols: usize,
    b: &[f32], b_rows: usize, b_cols: usize,
    result: &mut [f32]
) {
    use super::traits::SimdMatrix;
    AutoSimd::simd_matmul(a, a_rows, a_cols, b, b_rows, b_cols, result);
}

/// Optimized sum reduction
/// 最適化された合計リダクション
pub fn sum_optimized(data: &[f32]) -> f32 {
    AutoSimd::simd_sum(data)
}

/// Optimized mean calculation
/// 最適化された平均計算
pub fn mean_optimized(data: &[f32]) -> f32 {
    AutoSimd::simd_mean(data)
}

/// Optimized variance calculation
/// 最適化された分散計算
pub fn variance_optimized(data: &[f32]) -> f32 {
    let mean = AutoSimd::simd_mean(data);
    AutoSimd::simd_variance(data, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        add_optimized(&a, &b, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        let expected = vec![2.0, 6.0, 12.0, 20.0];

        mul_optimized(&a, &b, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_scalar_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];
        let expected = vec![2.0, 4.0, 6.0, 8.0];

        mul_scalar_optimized(&a, 2.0, &mut result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // = 70.0

        let result = dot_product_optimized(&a, &b);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sum_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum_optimized(&data);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_mean_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mean_optimized(&data);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_variance_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = variance_optimized(&data);
        // Expected variance for [1,2,3,4,5] = 2.5
        assert!((result - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_large_arrays() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];

        add_optimized(&a, &b, &mut result);

        for i in 0..size {
            assert_eq!(result[i], a[i] + b[i]);
        }
    }

    #[test]
    fn test_optimization_info() {
        let info = get_optimization_info();
        assert!(info.contains("SIMD Optimization Status"));
        println!("{}", info);
    }
}
