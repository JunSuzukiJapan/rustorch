/// SIMD operation traits for different data types
/// 異なるデータ型のSIMD演算トレイト
use num_traits::Float;

/// Trait for SIMD-optimized element-wise operations
/// SIMD最適化された要素ごと演算のトレイト
pub trait SimdElementwise<T: Float> {
    /// SIMD-optimized addition
    /// SIMD最適化加算
    fn simd_add(a: &[T], b: &[T], result: &mut [T]);

    /// SIMD-optimized multiplication
    /// SIMD最適化乗算
    fn simd_mul(a: &[T], b: &[T], result: &mut [T]);

    /// SIMD-optimized scalar multiplication
    /// SIMD最適化スカラー乗算
    fn scalar_mul(a: &[T], scalar: T, result: &mut [T]);

    /// SIMD-optimized dot product
    /// SIMD最適化内積
    fn dot(a: &[T], b: &[T]) -> T;
}

/// Trait for SIMD-optimized reduction operations
/// SIMD最適化リダクション演算のトレイト
pub trait SimdReduction<T: Float> {
    /// SIMD-optimized sum
    /// SIMD最適化合計
    fn simd_sum(data: &[T]) -> T;

    /// SIMD-optimized mean
    /// SIMD最適化平均
    fn simd_mean(data: &[T]) -> T;

    /// SIMD-optimized variance
    /// SIMD最適化分散
    fn simd_variance(data: &[T], mean: T) -> T;
}

/// Trait for SIMD-optimized matrix operations
/// SIMD最適化行列演算のトレイト
pub trait SimdMatrix<T: Float> {
    /// SIMD-optimized matrix multiplication
    /// SIMD最適化行列乗算
    fn simd_matmul(
        a: &[T],
        a_rows: usize,
        a_cols: usize,
        b: &[T],
        b_rows: usize,
        b_cols: usize,
        c: &mut [T],
    );

    /// SIMD-optimized matrix-vector multiplication
    /// SIMD最適化行列ベクトル乗算
    fn simd_matvec(matrix: &[T], rows: usize, cols: usize, vector: &[T], result: &mut [T]);
}

/// Auto-selecting SIMD implementation based on CPU features
/// CPU機能に基づく自動SIMD実装選択
pub struct AutoSimd;

impl AutoSimd {
    /// Create a new AutoSimd instance with CPU feature detection
    /// CPU機能検出を使用して新しいAutoSimdインスタンスを作成
    pub fn new() -> Self {
        AutoSimd
    }

    /// Perform scalar multiplication using optimal SIMD implementation
    /// 最適なSIMD実装を使用してスカラー乗算を実行
    pub fn scalar_mul(&self, a: &[f32], scalar: f32, result: &mut [f32]) {
        <Self as SimdElementwise<f32>>::scalar_mul(a, scalar, result);
    }

    /// Compute dot product using optimal SIMD implementation
    /// 最適なSIMD実装を使用してドット積を計算
    pub fn simd_dot(&self, a: &[f32], b: &[f32]) -> f32 {
        <Self as SimdElementwise<f32>>::dot(a, b)
    }
}

impl SimdElementwise<f32> for AutoSimd {
    #[allow(unused_unsafe)]
    fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        use crate::simd::vectorized;
        if a.len() < 32 {
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
        } else if vectorized::is_avx2_available() {
            unsafe {
                vectorized::add_f32_avx2(a, b, result);
            }
        } else if vectorized::is_sse41_available() {
            unsafe {
                vectorized::add_f32_sse41(a, b, result);
            }
        } else {
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }

    #[allow(unused_unsafe)]
    fn simd_mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        use crate::simd::vectorized;
        if a.len() < 32 {
            for i in 0..a.len() {
                result[i] = a[i] * b[i];
            }
        } else if vectorized::is_avx2_available() {
            unsafe {
                vectorized::mul_f32_avx2(a, b, result);
            }
        } else if vectorized::is_sse41_available() {
            unsafe {
                vectorized::mul_f32_sse41(a, b, result);
            }
        } else {
            for i in 0..a.len() {
                result[i] = a[i] * b[i];
            }
        }
    }

    #[allow(unused_unsafe)]
    fn scalar_mul(a: &[f32], scalar: f32, result: &mut [f32]) {
        use crate::simd::vectorized;
        if a.len() < 32 {
            for i in 0..a.len() {
                result[i] = a[i] * scalar;
            }
        } else if vectorized::is_avx2_available() {
            unsafe {
                vectorized::scalar_mul_f32_avx2(a, scalar, result);
            }
        } else {
            for i in 0..a.len() {
                result[i] = a[i] * scalar;
            }
        }
    }

    #[allow(unused_unsafe)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        use crate::simd::vectorized;
        if a.len() < 32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        } else if vectorized::is_avx2_available() {
            unsafe { vectorized::dot_product_f32_avx2(a, b) }
        } else {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
    }
}

impl SimdReduction<f32> for AutoSimd {
    fn simd_sum(data: &[f32]) -> f32 {
        use crate::simd::vectorized;
        vectorized::sum_f32_simd(data)
    }

    fn simd_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        Self::simd_sum(data) / data.len() as f32
    }

    fn simd_variance(data: &[f32], mean: f32) -> f32 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mut variance_sum = 0.0;
        for &x in data {
            let diff = x - mean;
            variance_sum += diff * diff;
        }
        variance_sum / (data.len() - 1) as f32
    }
}

impl SimdMatrix<f32> for AutoSimd {
    fn simd_matmul(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        c: &mut [f32],
    ) {
        use crate::simd::vectorized;
        vectorized::matmul_f32_simd(a, a_rows, a_cols, b, b_rows, b_cols, c);
    }

    fn simd_matvec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32], result: &mut [f32]) {
        assert_eq!(matrix.len(), rows * cols);
        assert_eq!(vector.len(), cols);
        assert_eq!(result.len(), rows);

        for i in 0..rows {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            result[i] = Self::dot(row, vector);
        }
    }
}

/// Fallback scalar implementation for non-f32 types
/// f32以外の型用のフォールバックスカラー実装
pub struct ScalarFallback;

impl<T: Float + Copy> SimdElementwise<T> for ScalarFallback {
    fn simd_add(a: &[T], b: &[T], result: &mut [T]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    fn simd_mul(a: &[T], b: &[T], result: &mut [T]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    fn scalar_mul(a: &[T], scalar: T, result: &mut [T]) {
        for i in 0..a.len() {
            result[i] = a[i] * scalar;
        }
    }

    fn dot(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| *x * *y)
            .fold(T::zero(), |acc, x| acc + x)
    }
}

impl<T: Float + Copy> SimdReduction<T> for ScalarFallback {
    fn simd_sum(data: &[T]) -> T {
        data.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn simd_mean(data: &[T]) -> T {
        if data.is_empty() {
            return T::zero();
        }
        Self::simd_sum(data) / T::from(data.len()).unwrap()
    }

    fn simd_variance(data: &[T], mean: T) -> T {
        if data.len() <= 1 {
            return T::zero();
        }

        let variance_sum = data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);

        variance_sum / T::from(data.len() - 1).unwrap()
    }
}

impl<T: Float + Copy> SimdMatrix<T> for ScalarFallback {
    fn simd_matmul(
        a: &[T],
        a_rows: usize,
        a_cols: usize,
        b: &[T],
        b_rows: usize,
        b_cols: usize,
        c: &mut [T],
    ) {
        assert_eq!(a_cols, b_rows);
        assert_eq!(a.len(), a_rows * a_cols);
        assert_eq!(b.len(), b_rows * b_cols);
        assert_eq!(c.len(), a_rows * b_cols);

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = T::zero();
                for k in 0..a_cols {
                    sum = sum + a[i * a_cols + k] * b[k * b_cols + j];
                }
                c[i * b_cols + j] = sum;
            }
        }
    }

    fn simd_matvec(matrix: &[T], rows: usize, cols: usize, vector: &[T], result: &mut [T]) {
        assert_eq!(matrix.len(), rows * cols);
        assert_eq!(vector.len(), cols);
        assert_eq!(result.len(), rows);

        for i in 0..rows {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            result[i] = Self::dot(row, vector);
        }
    }
}
