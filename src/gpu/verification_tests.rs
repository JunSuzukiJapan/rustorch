//! GPU vs CPU Verification Tests for RusTorch
//! RusTorch用GPU vs CPU検証テスト

use crate::error::RusTorchResult;

/// Test tolerance for floating point comparisons
/// 浮動小数点比較のテスト許容値
pub const FLOAT_TOLERANCE: f32 = 1e-5;
/// Test tolerance for double precision floating point comparisons
/// 倍精度浮動小数点比較のテスト許容値
pub const DOUBLE_TOLERANCE: f64 = 1e-10;

/// Test result structure
/// テスト結果構造体
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct VerificationResult {
    pub test_name: String,
    pub passed: bool,
    pub cpu_time_ms: Option<f64>,
    pub gpu_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub max_error: f32,
    pub mean_error: f32,
    pub error_message: Option<String>,
}

impl VerificationResult {
    /// Create a new verification result
    /// 新しい検証結果を作成
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            passed: false,
            cpu_time_ms: None,
            gpu_time_ms: None,
            speedup: None,
            max_error: 0.0,
            mean_error: 0.0,
            error_message: None,
        }
    }

    /// Add timing information to the verification result
    /// 検証結果にタイミング情報を追加
    pub fn with_timing(mut self, cpu_time: f64, gpu_time: f64) -> Self {
        self.cpu_time_ms = Some(cpu_time);
        self.gpu_time_ms = Some(gpu_time);
        self.speedup = Some(cpu_time / gpu_time);
        self
    }

    /// Add error metrics to the verification result
    /// 検証結果にエラーメトリクスを追加
    pub fn with_error_metrics(mut self, max_error: f32, mean_error: f32) -> Self {
        self.max_error = max_error;
        self.mean_error = mean_error;
        self.passed = max_error < FLOAT_TOLERANCE;
        self
    }

    /// Mark the verification as failed with error message
    /// エラーメッセージと共に検証を失敗とマーク
    pub fn with_failure(mut self, error: String) -> Self {
        self.passed = false;
        self.error_message = Some(error);
        self
    }
}

/// Verification test suite for GPU operations
/// GPU演算の検証テストスイート
pub struct VerificationTestSuite {
    results: Vec<VerificationResult>,
}

impl VerificationTestSuite {
    /// Create a new verification test suite
    /// 新しい検証テストスイートを作成
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run all verification tests
    /// すべての検証テストを実行
    pub fn run_all_tests(&mut self) -> RusTorchResult<()> {
        println!("🔍 Starting GPU vs CPU Verification Tests");
        println!("==========================================");

        // Element-wise operations
        self.test_elementwise_add()?;
        self.test_elementwise_sub()?;
        self.test_elementwise_mul()?;
        self.test_elementwise_div()?;

        // Matrix operations
        self.test_matrix_multiplication_small()?;
        self.test_matrix_multiplication_large()?;

        // Reduction operations
        self.test_reduction_sum()?;
        self.test_reduction_mean()?;

        // Neural network operations
        self.test_relu_activation()?;
        self.test_batch_normalization()?;
        self.test_softmax()?;

        // Convolution operations
        self.test_conv2d_basic()?;
        self.test_max_pool2d()?;

        // Advanced operations
        self.test_transpose()?;
        self.test_gelu_activation()?;

        self.print_summary();
        Ok(())
    }

    /// Test element-wise addition
    /// 要素ごと加算のテスト
    fn test_elementwise_add(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Element-wise Addition".to_string());

        let size = 1024 * 1024;
        let a = vec![1.5f32; size];
        let b = vec![2.5f32; size];

        // CPU reference implementation
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = a[i] + b[i];
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        // GPU implementation
        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_elementwise_add(&a, &b)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test element-wise subtraction
    /// 要素ごと減算のテスト
    fn test_elementwise_sub(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Element-wise Subtraction".to_string());

        let size = 512 * 512;
        let a = vec![5.0f32; size];
        let b = vec![2.0f32; size];

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = a[i] - b[i];
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_elementwise_sub(&a, &b)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test element-wise multiplication
    /// 要素ごと乗算のテスト
    fn test_elementwise_mul(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Element-wise Multiplication".to_string());

        let size = 256 * 256 * 64;
        let a = vec![1.5f32; size];
        let b = vec![2.0f32; size];

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = a[i] * b[i];
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_elementwise_mul(&a, &b)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test element-wise division
    /// 要素ごと除算のテスト
    fn test_elementwise_div(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Element-wise Division".to_string());

        let size = 128 * 128 * 32;
        let a = vec![10.0f32; size];
        let b = vec![2.0f32; size];

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = a[i] / b[i];
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_elementwise_div(&a, &b)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test small matrix multiplication
    /// 小型行列乗算のテスト
    fn test_matrix_multiplication_small(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Matrix Multiplication (Small)".to_string());

        let m = 64;
        let n = 64;
        let k = 64;

        let a = (0..m * k).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let b = (0..k * n).map(|i| (i as f32) * 0.02).collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let _cpu_result = self.cpu_matmul(&a, &b, m, n, k);
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_matmul(&a, &b, m, n, k)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test large matrix multiplication
    /// 大型行列乗算のテスト
    fn test_matrix_multiplication_large(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Matrix Multiplication (Large)".to_string());

        let m = 1024;
        let n = 1024;
        let k = 1024;

        let a = (0..m * k)
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect::<Vec<f32>>();
        let b = (0..k * n)
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect::<Vec<f32>>();

        // CPU reference (subset for performance)
        let start = std::time::Instant::now();
        let _cpu_result = self.cpu_matmul(&a, &b, m.min(256), n.min(256), k.min(256));
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_matmul(&a, &b, m, n, k)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            // Compare subset
            let cpu_subset = &cpu_result[..256 * 256];
            let gpu_subset = &gpu_result[..256 * 256];
            let (max_error, mean_error) = self.compute_error_metrics(cpu_subset, gpu_subset);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test reduction sum
    /// リダクション合計のテスト
    fn test_reduction_sum(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Reduction Sum".to_string());

        let size = 1024 * 1024;
        let input = (0..size)
            .map(|i| (i % 1000) as f32 * 0.001)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let _cpu_sum = input.iter().sum::<f32>();
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_sum = self.gpu_reduce_sum(&input)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let error = (cpu_sum - gpu_sum).abs();
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(error, error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test reduction mean
    /// リダクション平均のテスト
    fn test_reduction_mean(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Reduction Mean".to_string());

        let size = 512 * 512;
        let input = (0..size)
            .map(|i| ((i % 100) as f32).sin())
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let _cpu_mean = input.iter().sum::<f32>() / size as f32;
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_sum = self.gpu_reduce_sum(&input)?;
            let gpu_mean = gpu_sum / size as f32;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let error = (cpu_mean - gpu_mean).abs();
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(error, error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test ReLU activation
    /// ReLU活性化のテスト
    fn test_relu_activation(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("ReLU Activation".to_string());

        let size = 256 * 256;
        let input = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = input[i].max(0.0);
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_relu(&input)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test batch normalization
    /// バッチ正規化のテスト
    fn test_batch_normalization(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Batch Normalization".to_string());

        let size = 1024;
        let input = (0..size).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let mean = 0.5f32;
        let variance = 0.25f32;
        let epsilon = 1e-5f32;

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = (input[i] - mean) / (variance + epsilon).sqrt();
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_batch_norm(&input, mean, variance, epsilon)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test softmax
    /// Softmaxのテスト
    fn test_softmax(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Softmax".to_string());

        let size = 1000;
        let input = (0..size)
            .map(|i| (i as f32) * 0.01 - 5.0)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            cpu_result[i] = (input[i] - max_val).exp() / exp_sum;
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_softmax(&input)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test 2D convolution
    /// 2D畳み込みのテスト
    fn test_conv2d_basic(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Conv2D Basic".to_string());

        let input_h = 32;
        let input_w = 32;
        let kernel_h = 3;
        let kernel_w = 3;
        let _output_h = input_h - kernel_h + 1;
        let _output_w = input_w - kernel_w + 1;

        let input = (0..input_h * input_w)
            .map(|i| (i as f32) * 0.01)
            .collect::<Vec<f32>>();
        let kernel = vec![1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]; // Edge detection

        // CPU reference
        let start = std::time::Instant::now();
        let _cpu_result = self.cpu_conv2d(&input, &kernel, input_h, input_w, kernel_h, kernel_w);
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_conv2d(
                &input, &kernel, input_h, input_w, kernel_h, kernel_w, 1, 1, 0, 0,
            )?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test max pooling 2D
    /// 2Dマックスプーリングのテスト
    fn test_max_pool2d(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Max Pool2D".to_string());

        let input_h = 16;
        let input_w = 16;
        let pool_h = 2;
        let pool_w = 2;
        let stride_h = 2;
        let stride_w = 2;
        let _output_h = input_h / stride_h;
        let _output_w = input_w / stride_w;

        let input = (0..input_h * input_w)
            .map(|i| (i as f32) * 0.1)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let _cpu_result =
            self.cpu_max_pool2d(&input, input_h, input_w, pool_h, pool_w, stride_h, stride_w);
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_max_pool2d(
                &input, input_h, input_w, pool_h, pool_w, stride_h, stride_w, 0, 0,
            )?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test transpose
    /// 転置のテスト
    fn test_transpose(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("Transpose".to_string());

        let rows = 256;
        let cols = 512;
        let input = (0..rows * cols)
            .map(|i| (i as f32) * 0.01)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                cpu_result[c * rows + r] = input[r * cols + c];
            }
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_transpose(&input, rows, cols)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error, mean_error);
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    /// Test GELU activation
    /// GELU活性化のテスト
    fn test_gelu_activation(&mut self) -> RusTorchResult<()> {
        let mut result = VerificationResult::new("GELU Activation".to_string());

        let size = 1024;
        let input = (0..size)
            .map(|i| (i as f32 - 512.0) * 0.01)
            .collect::<Vec<f32>>();

        // CPU reference
        let start = std::time::Instant::now();
        let mut cpu_result = vec![0.0f32; size];
        for i in 0..size {
            let x = input[i];
            let erf_term = (x / 2.0_f32.sqrt()).tanh(); // Approximation
            cpu_result[i] = 0.5 * x * (1.0 + erf_term);
        }
        let _cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            let start = std::time::Instant::now();
            let gpu_result = self.gpu_gelu(&input)?;
            let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

            let (max_error, mean_error) = self.compute_error_metrics(&cpu_result, &gpu_result);
            result = result
                .with_timing(cpu_time, gpu_time)
                .with_error_metrics(max_error * 10.0, mean_error * 10.0); // GELU has higher tolerance
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_failure("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    // Helper functions for CPU reference implementations
    fn cpu_matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
        c
    }

    fn cpu_conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        ih: usize,
        iw: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let oh = ih - kh + 1;
        let ow = iw - kw + 1;
        let mut output = vec![0.0f32; oh * ow];

        for out_y in 0..oh {
            for out_x in 0..ow {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let in_y = out_y + ky;
                        let in_x = out_x + kx;
                        sum += input[in_y * iw + in_x] * kernel[ky * kw + kx];
                    }
                }
                output[out_y * ow + out_x] = sum;
            }
        }
        output
    }

    fn cpu_max_pool2d(
        &self,
        input: &[f32],
        ih: usize,
        iw: usize,
        ph: usize,
        pw: usize,
        sh: usize,
        sw: usize,
    ) -> Vec<f32> {
        let oh = ih / sh;
        let ow = iw / sw;
        let mut output = vec![f32::NEG_INFINITY; oh * ow];

        for out_y in 0..oh {
            for out_x in 0..ow {
                let mut max_val = f32::NEG_INFINITY;
                for py in 0..ph {
                    for px in 0..pw {
                        let in_y = out_y * sh + py;
                        let in_x = out_x * sw + px;
                        if in_y < ih && in_x < iw {
                            max_val = max_val.max(input[in_y * iw + in_x]);
                        }
                    }
                }
                output[out_y * ow + out_x] = max_val;
            }
        }
        output
    }

    // GPU operation stubs (to be implemented with actual GPU backends)
    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_elementwise_add(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_elementwise_sub(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_elementwise_mul(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_elementwise_div(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_matmul(
        &self,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_reduce_sum(&self, _input: &[f32]) -> RusTorchResult<f32> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_relu(&self, _input: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_batch_norm(
        &self,
        _input: &[f32],
        _mean: f32,
        _variance: f32,
        _epsilon: f32,
    ) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_softmax(&self, _input: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_conv2d(
        &self,
        _input: &[f32],
        _kernel: &[f32],
        _ih: usize,
        _iw: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_max_pool2d(
        &self,
        _input: &[f32],
        _ih: usize,
        _iw: usize,
        _ph: usize,
        _pw: usize,
        _sh: usize,
        _sw: usize,
        _pad_h: usize,
        _pad_w: usize,
    ) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_transpose(
        &self,
        _input: &[f32],
        _rows: usize,
        _cols: usize,
    ) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn gpu_gelu(&self, _input: &[f32]) -> RusTorchResult<Vec<f32>> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    /// Compute error metrics between CPU and GPU results
    /// CPU と GPU 結果間の誤差メトリクスを計算
    #[allow(dead_code)]
    fn compute_error_metrics(&self, cpu_result: &[f32], gpu_result: &[f32]) -> (f32, f32) {
        let len = cpu_result.len().min(gpu_result.len());
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;

        for i in 0..len {
            let error = (cpu_result[i] - gpu_result[i]).abs();
            max_error = max_error.max(error);
            sum_error += error;
        }

        let mean_error = sum_error / len as f32;
        (max_error, mean_error)
    }

    /// Print test summary
    /// テスト概要を出力
    fn print_summary(&self) {
        println!("\n📊 Verification Test Summary");
        println!("============================");

        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        println!("Total Tests: {}", total_tests);
        println!("✅ Passed: {}", passed_tests);
        println!("❌ Failed: {}", failed_tests);
        println!(
            "Success Rate: {:.1}%",
            (passed_tests as f32 / total_tests as f32) * 100.0
        );

        println!("\n📈 Performance Results:");
        println!(
            "{:<30} {:<8} {:<10} {:<10} {:<10} {:<15} {:<15}",
            "Test Name", "Status", "CPU (ms)", "GPU (ms)", "Speedup", "Max Error", "Mean Error"
        );
        println!("{}", "-".repeat(120));

        for result in &self.results {
            let status = if result.passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            let cpu_time = result
                .cpu_time_ms
                .map_or("N/A".to_string(), |t| format!("{:.2}", t));
            let gpu_time = result
                .gpu_time_ms
                .map_or("N/A".to_string(), |t| format!("{:.2}", t));
            let speedup = result
                .speedup
                .map_or("N/A".to_string(), |s| format!("{:.2}x", s));

            println!(
                "{:<30} {:<8} {:<10} {:<10} {:<10} {:<15.2e} {:<15.2e}",
                result.test_name,
                status,
                cpu_time,
                gpu_time,
                speedup,
                result.max_error,
                result.mean_error
            );

            if let Some(ref error_msg) = result.error_message {
                println!("    Error: {}", error_msg);
            }
        }

        if failed_tests > 0 {
            println!(
                "\n⚠️  {} tests failed. Check error messages above.",
                failed_tests
            );
        } else {
            println!(
                "\n🎉 All tests passed! GPU implementations match CPU results within tolerance."
            );
        }
    }
}

impl Default for VerificationTestSuite {
    fn default() -> Self {
        Self::new()
    }
}
