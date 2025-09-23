//! Testing utilities for RusTorch
//! RusTorch用テストユーティリティ

#[cfg(test)]
use crate::tensor::Tensor;
#[cfg(test)]
use num_traits::Float;

/// Test utilities for tensor operations
/// テンソル操作用テストユーティリティ
#[cfg(test)]
pub mod tensor_test_utils {
    use super::*;

    /// Create a test tensor with known values
    /// 既知の値でテストテンソルを作成
    pub fn create_test_tensor<T: Float + 'static>() -> Tensor<T> {
        let data = vec![
            T::from(1.0).unwrap(),
            T::from(2.0).unwrap(),
            T::from(3.0).unwrap(),
            T::from(4.0).unwrap(),
            T::from(5.0).unwrap(),
            T::from(6.0).unwrap(),
        ];
        Tensor::from_vec(data, vec![2, 3])
    }

    /// Assert tensors are approximately equal
    /// テンソルが近似的に等しいことをアサート
    pub fn assert_tensor_eq<T: Float + 'static + std::fmt::Debug>(
        a: &Tensor<T>,
        b: &Tensor<T>,
        epsilon: T,
    ) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match");

        let a_data = a.as_slice().expect("Failed to get tensor data");
        let b_data = b.as_slice().expect("Failed to get tensor data");

        for (i, (&a_val, &b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
            assert!(
                (a_val - b_val).abs() < epsilon,
                "Tensors differ at index {}: {:?} vs {:?} (epsilon: {:?})",
                i,
                a_val,
                b_val,
                epsilon
            );
        }
    }

    /// Check if tensor has expected shape
    /// テンソルが期待される形状かチェック
    pub fn assert_tensor_shape<T: Float + 'static>(tensor: &Tensor<T>, expected_shape: &[usize]) {
        assert_eq!(
            tensor.shape(),
            expected_shape,
            "Expected shape {:?}, got {:?}",
            expected_shape,
            tensor.shape()
        );
    }

    /// Create random tensor for testing
    /// テスト用ランダムテンソルを作成
    pub fn create_random_tensor<T: Float + 'static>(shape: &[usize]) -> Tensor<T> {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();

        let data: Vec<T> = (0..total_size)
            .map(|_| T::from(rng.gen_range(-1.0..1.0)).unwrap())
            .collect();

        Tensor::from_vec(data, shape.to_vec())
    }
}

/// Performance testing utilities
/// 性能テスト用ユーティリティ
#[cfg(test)]
pub mod perf_test_utils {
    use std::time::{Duration, Instant};

    /// Benchmark a function and return execution time
    /// 関数をベンチマークし実行時間を返す
    pub fn benchmark<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Run benchmark multiple times and return average
    /// ベンチマークを複数回実行し平均を返す
    pub fn benchmark_average<F, R>(f: F, iterations: usize) -> (R, Duration)
    where
        F: Fn() -> R,
    {
        let mut total_duration = Duration::new(0, 0);
        let mut result = None;

        for _ in 0..iterations {
            let (r, duration) = benchmark(&f);
            total_duration += duration;
            result = Some(r);
        }

        (result.unwrap(), total_duration / iterations as u32)
    }
}

/// Integration test helpers
/// 統合テスト用ヘルパー
#[cfg(test)]
pub mod integration_test_utils {
    use super::*;

    /// Setup function for integration tests
    /// 統合テスト用セットアップ関数
    pub fn setup_integration_test() {
        // Initialize logging or other global state if needed
        // 必要に応じてログやグローバル状態を初期化
    }

    /// Cleanup function for integration tests
    /// 統合テスト用クリーンアップ関数
    pub fn cleanup_integration_test() {
        // Cleanup global state if needed
        // 必要に応じてグローバル状態をクリーンアップ
    }

    /// Create a test environment for neural network testing
    /// ニューラルネットワークテスト用のテスト環境を作成
    pub fn create_nn_test_env() -> (Tensor<f32>, Tensor<f32>) {
        let input = tensor_test_utils::create_random_tensor(&[32, 784]); // Batch size 32, 28x28 images
        let target = tensor_test_utils::create_random_tensor(&[32, 10]); // 10 classes
        (input, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_tensor() {
        let tensor = tensor_test_utils::create_test_tensor::<f32>();
        tensor_test_utils::assert_tensor_shape(&tensor, &[2, 3]);

        let expected_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let actual_data = tensor.as_slice().unwrap();

        for (expected, actual) in expected_data.iter().zip(actual_data.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
    }

    #[test]
    fn test_assert_tensor_eq() {
        let tensor1 = tensor_test_utils::create_test_tensor::<f32>();
        let tensor2 = tensor_test_utils::create_test_tensor::<f32>();

        tensor_test_utils::assert_tensor_eq(&tensor1, &tensor2, 1e-6);
    }

    #[test]
    fn test_benchmark() {
        let (result, duration) = perf_test_utils::benchmark(|| {
            // より重い計算でベンチマーク測定を確実にする（i64使用）
            let mut sum: i64 = 0;
            for i in 0i64..100_000 {
                sum += i * i;
            }
            sum
        });

        assert_eq!(result, 333328333350000i64);
        assert!(
            duration.as_nanos() > 0,
            "ベンチマーク実行時間が0です: {:?}",
            duration
        );
    }
}
