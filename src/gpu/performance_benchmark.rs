//! Performance Benchmark Suite for GPU Kernels
//! GPUカーネル用パフォーマンスベンチマークスイート

use crate::error::RusTorchResult;
use std::collections::HashMap;
use std::time::Instant;

/// Benchmark configuration
/// ベンチマーク設定#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    /// ウォームアップ反復回数
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    /// 測定反復回数
    pub measurement_iterations: usize,
    /// Minimum benchmark duration in milliseconds
    /// 最小ベンチマーク時間（ミリ秒）
    pub min_duration_ms: u64,
    /// Maximum benchmark duration in milliseconds
    /// 最大ベンチマーク時間（ミリ秒）
    pub max_duration_ms: u64,
    /// Enable memory bandwidth measurements
    /// メモリ帯域幅測定を有効化
    pub measure_memory_bandwidth: bool,
    /// Enable FLOPS measurements
    /// FLOPS測定を有効化
    pub measure_flops: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 100,
            min_duration_ms: 1000,
            max_duration_ms: 30000,
            measure_memory_bandwidth: true,
            measure_flops: true,
        }
    }
}

/// Benchmark result for a single operation
/// 単一演算のベンチマーク結果#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub device_name: String,
    pub problem_size: String,
    pub cpu_time_ms: Option<f64>,
    pub gpu_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub cpu_throughput_gops: Option<f64>,
    pub gpu_throughput_gops: Option<f64>,
    pub memory_bandwidth_gb_s: Option<f64>,
    pub iterations: usize,
    pub total_flops: Option<u64>,
    pub total_memory_bytes: Option<u64>,
    pub error_message: Option<String>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    /// 新しいベンチマーク結果を作成
    pub fn new(operation_name: String, device_name: String, problem_size: String) -> Self {
        Self {
            operation_name,
            device_name,
            problem_size,
            cpu_time_ms: None,
            gpu_time_ms: None,
            speedup: None,
            cpu_throughput_gops: None,
            gpu_throughput_gops: None,
            memory_bandwidth_gb_s: None,
            iterations: 0,
            total_flops: None,
            total_memory_bytes: None,
            error_message: None,
        }
    }

    /// Add CPU timing information to the benchmark result
    /// ベンチマーク結果にCPUタイミング情報を追加
    pub fn with_cpu_timing(mut self, time_ms: f64, iterations: usize) -> Self {
        self.cpu_time_ms = Some(time_ms);
        self.iterations = iterations;
        if let Some(flops) = self.total_flops {
            self.cpu_throughput_gops = Some((flops as f64 * iterations as f64) / (time_ms * 1e6));
        }
        self
    }

    /// Add GPU timing information to the benchmark result
    /// ベンチマーク結果にGPUタイミング情報を追加
    pub fn with_gpu_timing(mut self, time_ms: f64, iterations: usize) -> Self {
        self.gpu_time_ms = Some(time_ms);
        self.iterations = iterations;
        if let Some(flops) = self.total_flops {
            self.gpu_throughput_gops = Some((flops as f64 * iterations as f64) / (time_ms * 1e6));
        }
        if let Some(bytes) = self.total_memory_bytes {
            self.memory_bandwidth_gb_s = Some((bytes as f64 * iterations as f64) / (time_ms * 1e6));
        }
        if let Some(cpu_time) = self.cpu_time_ms {
            self.speedup = Some(cpu_time / time_ms);
        }
        self
    }

    /// Add FLOPS count to the benchmark result
    /// ベンチマーク結果にFLOPS数を追加
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.total_flops = Some(flops);
        self
    }

    /// Add memory bytes to the benchmark result
    /// ベンチマーク結果にメモリバイト数を追加
    pub fn with_memory_bytes(mut self, bytes: u64) -> Self {
        self.total_memory_bytes = Some(bytes);
        self
    }

    /// Add error information to the benchmark result
    /// ベンチマーク結果にエラー情報を追加
    pub fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self
    }
}

/// Performance benchmark suite for GPU operations
/// GPU演算用パフォーマンスベンチマークスイート
pub struct PerformanceBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark
    /// 新しいパフォーマンスベンチマークを作成
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run comprehensive benchmark suite
    /// 包括的ベンチマークスイートを実行
    pub fn run_comprehensive_benchmarks(&mut self) -> RusTorchResult<()> {
        println!("🚀 Starting Comprehensive GPU Performance Benchmarks");
        println!("=====================================================");

        // Element-wise operation benchmarks
        self.benchmark_elementwise_operations()?;

        // Matrix multiplication benchmarks
        self.benchmark_matrix_operations()?;

        // Reduction operation benchmarks
        self.benchmark_reduction_operations()?;

        // Neural network operation benchmarks
        self.benchmark_neural_network_operations()?;

        // Convolution operation benchmarks
        self.benchmark_convolution_operations()?;

        // Memory transfer benchmarks
        self.benchmark_memory_operations()?;

        self.print_comprehensive_report();
        Ok(())
    }

    /// Benchmark element-wise operations
    /// 要素ごと演算のベンチマーク
    fn benchmark_elementwise_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Element-wise Operations");
        println!("---------------------------------------");

        let sizes = vec![
            1024,     // 1K elements
            65536,    // 64K elements
            1048576,  // 1M elements
            16777216, // 16M elements
            67108864, // 64M elements
        ];

        for &size in &sizes {
            // Element-wise addition
            self.benchmark_elementwise_add(size)?;

            // Element-wise multiplication
            self.benchmark_elementwise_mul(size)?;

            // Element-wise division
            self.benchmark_elementwise_div(size)?;
        }

        Ok(())
    }

    /// Benchmark matrix operations
    /// 行列演算のベンチマーク
    fn benchmark_matrix_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Matrix Operations");
        println!("----------------------------------");

        let matrix_sizes = vec![
            (64, 64, 64),       // Small matrices
            (128, 128, 128),    // Medium matrices
            (256, 256, 256),    // Large matrices
            (512, 512, 512),    // Very large matrices
            (1024, 1024, 1024), // Huge matrices
            (2048, 2048, 2048), // Massive matrices
        ];

        for &(m, n, k) in &matrix_sizes {
            self.benchmark_matrix_multiplication(m, n, k)?;
        }

        // Specialized matrix operations
        self.benchmark_transpose_operations()?;

        Ok(())
    }

    /// Benchmark reduction operations
    /// リダクション演算のベンチマーク
    fn benchmark_reduction_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Reduction Operations");
        println!("------------------------------------");

        let sizes = vec![1024, 65536, 1048576, 16777216, 67108864];

        for &size in &sizes {
            self.benchmark_reduce_sum(size)?;
            self.benchmark_reduce_mean(size)?;
            self.benchmark_reduce_max(size)?;
        }

        Ok(())
    }

    /// Benchmark neural network operations
    /// ニューラルネットワーク演算のベンチマーク
    fn benchmark_neural_network_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Neural Network Operations");
        println!("----------------------------------------");

        let sizes = vec![1024, 65536, 1048576, 4194304];

        for &size in &sizes {
            self.benchmark_relu_activation(size)?;
            self.benchmark_gelu_activation(size)?;
            self.benchmark_softmax(size)?;
            self.benchmark_batch_normalization(size)?;
        }

        Ok(())
    }

    /// Benchmark convolution operations
    /// 畳み込み演算のベンチマーク
    fn benchmark_convolution_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Convolution Operations");
        println!("--------------------------------------");

        let conv_configs = vec![
            (32, 32, 3, 3),   // Small feature maps
            (64, 64, 3, 3),   // Medium feature maps
            (128, 128, 3, 3), // Large feature maps
            (256, 256, 5, 5), // Large kernels
            (512, 512, 7, 7), // Very large
        ];

        for &(input_h, input_w, kernel_h, kernel_w) in &conv_configs {
            self.benchmark_conv2d(input_h, input_w, kernel_h, kernel_w)?;
            self.benchmark_max_pool2d(input_h, input_w)?;
        }

        Ok(())
    }

    /// Benchmark memory operations
    /// メモリ演算のベンチマーク
    fn benchmark_memory_operations(&mut self) -> RusTorchResult<()> {
        println!("\n📊 Benchmarking Memory Operations");
        println!("----------------------------------");

        let sizes_mb = vec![1, 4, 16, 64, 256, 1024]; // Memory sizes in MB

        for &size_mb in &sizes_mb {
            let size_bytes = size_mb * 1024 * 1024;
            let size_elements = size_bytes / 4; // f32 elements

            self.benchmark_host_to_device_transfer(size_elements)?;
            self.benchmark_device_to_host_transfer(size_elements)?;
            self.benchmark_device_to_device_copy(size_elements)?;
        }

        Ok(())
    }

    // Individual benchmark implementations
    fn benchmark_elementwise_add(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "Element-wise Add".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops(size as u64)
        .with_memory_bytes((size * 3 * 4) as u64); // 3 f32 arrays

        let a = vec![1.5f32; size];
        let b = vec![2.5f32; size];

        // CPU benchmark
        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut c = vec![0.0f32; size];
            for i in 0..size {
                c[i] = a[i] + b[i];
            }
            c
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        // GPU benchmark
        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_elementwise_add(&a, &b) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_elementwise_mul(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "Element-wise Mul".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops(size as u64)
        .with_memory_bytes((size * 3 * 4) as u64);

        let a = vec![1.5f32; size];
        let b = vec![2.0f32; size];

        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut c = vec![0.0f32; size];
            for i in 0..size {
                c[i] = a[i] * b[i];
            }
            c
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_elementwise_mul(&a, &b) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_elementwise_div(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "Element-wise Div".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops(size as u64)
        .with_memory_bytes((size * 3 * 4) as u64);

        let a = vec![10.0f32; size];
        let b = vec![2.0f32; size];

        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut c = vec![0.0f32; size];
            for i in 0..size {
                c[i] = a[i] / b[i];
            }
            c
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_elementwise_div(&a, &b) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_matrix_multiplication(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        let problem_size = format!("{}x{}x{}", m, n, k);
        let flops = (2 * m * n * k) as u64; // 2 operations per multiply-accumulate
        let memory_bytes = ((m * k + k * n + m * n) * 4) as u64; // f32 size

        let mut result = BenchmarkResult::new(
            "Matrix Multiplication".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops(flops)
        .with_memory_bytes(memory_bytes);

        let a = (0..m * k).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let b = (0..k * n).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| self.cpu_matmul(&a, &b, m, n, k))?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_matmul(&a, &b, m, n, k) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_transpose_operations(&mut self) -> RusTorchResult<()> {
        let sizes = vec![(256, 512), (512, 1024), (1024, 2048)];

        for &(rows, cols) in &sizes {
            let problem_size = format!("{}x{}", rows, cols);
            let mut result = BenchmarkResult::new(
                "Matrix Transpose".to_string(),
                "GPU".to_string(),
                problem_size,
            )
            .with_memory_bytes((rows * cols * 4) as u64);

            let input = (0..rows * cols)
                .map(|i| (i as f32) * 0.01)
                .collect::<Vec<f32>>();

            let cpu_time =
                self.benchmark_cpu_operation(|| self.cpu_transpose(&input, rows, cols))?;

            result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

            #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
            {
                match self.benchmark_gpu_transpose(&input, rows, cols) {
                    Ok(gpu_time) => {
                        result =
                            result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                    }
                    Err(e) => {
                        result = result.with_error(format!("GPU benchmark failed: {}", e));
                    }
                }
            }

            #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
            {
                result = result.with_error("No GPU backend available".to_string());
            }

            self.results.push(result);
        }

        Ok(())
    }

    fn benchmark_reduce_sum(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result =
            BenchmarkResult::new("Reduce Sum".to_string(), "GPU".to_string(), problem_size)
                .with_flops(size as u64)
                .with_memory_bytes((size * 4) as u64);

        let input = (0..size)
            .map(|i| (i % 1000) as f32 * 0.001)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| input.iter().sum::<f32>())?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_reduce_sum(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_reduce_mean(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result =
            BenchmarkResult::new("Reduce Mean".to_string(), "GPU".to_string(), problem_size)
                .with_flops(size as u64 + 1) // sum + division
                .with_memory_bytes((size * 4) as u64);

        let input = (0..size)
            .map(|i| ((i % 100) as f32).sin())
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| input.iter().sum::<f32>() / size as f32)?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_reduce_mean(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_reduce_max(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result =
            BenchmarkResult::new("Reduce Max".to_string(), "GPU".to_string(), problem_size)
                .with_flops(size as u64)
                .with_memory_bytes((size * 4) as u64);

        let input = (0..size)
            .map(|i| ((i % 1000) as f32) * 0.01)
            .collect::<Vec<f32>>();

        let cpu_time = self
            .benchmark_cpu_operation(|| input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_reduce_max(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_relu_activation(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "ReLU Activation".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops(size as u64)
        .with_memory_bytes((size * 2 * 4) as u64);

        let input = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut output = vec![0.0f32; size];
            for i in 0..size {
                output[i] = input[i].max(0.0);
            }
            output
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_relu(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_gelu_activation(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "GELU Activation".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops((size * 10) as u64) // GELU is computationally expensive
        .with_memory_bytes((size * 2 * 4) as u64);

        let input = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut output = vec![0.0f32; size];
            for i in 0..size {
                let x = input[i];
                let erf_term = (x / 2.0_f32.sqrt()).tanh(); // Approximation
                output[i] = 0.5 * x * (1.0 + erf_term);
            }
            output
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_gelu(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_softmax(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result =
            BenchmarkResult::new("Softmax".to_string(), "GPU".to_string(), problem_size)
                .with_flops((size * 3) as u64) // exp, sum, divide
                .with_memory_bytes((size * 4 * 4) as u64); // input, output, max_vals, sum_vals

        let input = (0..size)
            .map(|i| (i as f32) * 0.01 - 5.0)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| {
            let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
            let mut output = vec![0.0f32; size];
            for i in 0..size {
                output[i] = (input[i] - max_val).exp() / exp_sum;
            }
            output
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_softmax(&input) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_batch_normalization(&mut self, size: usize) -> RusTorchResult<()> {
        let problem_size = format!("{} elements", size);
        let mut result = BenchmarkResult::new(
            "Batch Normalization".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_flops((size * 3) as u64) // subtract, divide, sqrt
        .with_memory_bytes((size * 2 * 4 + 2 * 4) as u64);

        let input = (0..size).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let mean = 0.5f32;
        let variance = 0.25f32;
        let epsilon = 1e-5f32;

        let cpu_time = self.benchmark_cpu_operation(|| {
            let mut output = vec![0.0f32; size];
            for i in 0..size {
                output[i] = (input[i] - mean) / (variance + epsilon).sqrt();
            }
            output
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_batch_norm(&input, mean, variance, epsilon) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_conv2d(
        &mut self,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> RusTorchResult<()> {
        let output_h = input_h - kernel_h + 1;
        let output_w = input_w - kernel_w + 1;
        let problem_size = format!(
            "{}x{} -> {}x{} ({}x{} kernel)",
            input_h, input_w, output_h, output_w, kernel_h, kernel_w
        );

        let flops = (output_h * output_w * kernel_h * kernel_w * 2) as u64; // multiply-add
        let memory_bytes =
            ((input_h * input_w + kernel_h * kernel_w + output_h * output_w) * 4) as u64;

        let mut result =
            BenchmarkResult::new("Conv2D".to_string(), "GPU".to_string(), problem_size)
                .with_flops(flops)
                .with_memory_bytes(memory_bytes);

        let input = (0..input_h * input_w)
            .map(|i| (i as f32) * 0.01)
            .collect::<Vec<f32>>();
        let kernel = (0..kernel_h * kernel_w)
            .map(|i| ((i as f32) - (kernel_h * kernel_w) as f32 / 2.0) * 0.1)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| {
            self.cpu_conv2d(&input, &kernel, input_h, input_w, kernel_h, kernel_w)
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_conv2d(
                &input, &kernel, input_h, input_w, kernel_h, kernel_w, 1, 1, 0, 0,
            ) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_max_pool2d(&mut self, input_h: usize, input_w: usize) -> RusTorchResult<()> {
        let pool_h = 2;
        let pool_w = 2;
        let stride_h = 2;
        let stride_w = 2;
        let output_h = input_h / stride_h;
        let output_w = input_w / stride_w;

        let problem_size = format!(
            "{}x{} -> {}x{} ({}x{} pool)",
            input_h, input_w, output_h, output_w, pool_h, pool_w
        );
        let memory_bytes = ((input_h * input_w + output_h * output_w) * 4) as u64;

        let mut result =
            BenchmarkResult::new("Max Pool2D".to_string(), "GPU".to_string(), problem_size)
                .with_memory_bytes(memory_bytes);

        let input = (0..input_h * input_w)
            .map(|i| (i as f32) * 0.1)
            .collect::<Vec<f32>>();

        let cpu_time = self.benchmark_cpu_operation(|| {
            self.cpu_max_pool2d(&input, input_h, input_w, pool_h, pool_w, stride_h, stride_w)
        })?;

        result = result.with_cpu_timing(cpu_time, self.config.measurement_iterations);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_max_pool2d(
                &input, input_h, input_w, pool_h, pool_w, stride_h, stride_w, 0, 0,
            ) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_host_to_device_transfer(&mut self, size: usize) -> RusTorchResult<()> {
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);
        let problem_size = format!("{:.1} MB", size_mb);
        let mut result = BenchmarkResult::new(
            "Host to Device Transfer".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_memory_bytes((size * 4) as u64);

        let _data = vec![1.0f32; size];

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_host_to_device(&_data) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_device_to_host_transfer(&mut self, size: usize) -> RusTorchResult<()> {
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);
        let problem_size = format!("{:.1} MB", size_mb);
        let mut result = BenchmarkResult::new(
            "Device to Host Transfer".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_memory_bytes((size * 4) as u64);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_device_to_host(size) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    fn benchmark_device_to_device_copy(&mut self, size: usize) -> RusTorchResult<()> {
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);
        let problem_size = format!("{:.1} MB", size_mb);
        let mut result = BenchmarkResult::new(
            "Device to Device Copy".to_string(),
            "GPU".to_string(),
            problem_size,
        )
        .with_memory_bytes((size * 4) as u64);

        #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
        {
            match self.benchmark_gpu_device_to_device(size) {
                Ok(gpu_time) => {
                    result = result.with_gpu_timing(gpu_time, self.config.measurement_iterations);
                }
                Err(e) => {
                    result = result.with_error(format!("GPU benchmark failed: {}", e));
                }
            }
        }

        #[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
        {
            result = result.with_error("No GPU backend available".to_string());
        }

        self.results.push(result);
        Ok(())
    }

    // CPU reference implementations
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

    fn cpu_transpose(&self, input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                output[c * rows + r] = input[r * cols + c];
            }
        }
        output
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

    // GPU benchmark stubs
    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_elementwise_add(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_elementwise_mul(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_elementwise_div(&self, _a: &[f32], _b: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_matmul(
        &self,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_transpose(
        &self,
        _input: &[f32],
        _rows: usize,
        _cols: usize,
    ) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_reduce_sum(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_reduce_mean(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_reduce_max(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_relu(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_gelu(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_softmax(&self, _input: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_batch_norm(
        &self,
        _input: &[f32],
        _mean: f32,
        _variance: f32,
        _epsilon: f32,
    ) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_conv2d(
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
    ) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_max_pool2d(
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
    ) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_host_to_device(&self, _data: &[f32]) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_device_to_host(&self, _size: usize) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    #[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
    fn benchmark_gpu_device_to_device(&self, _size: usize) -> RusTorchResult<f64> {
        Err(RusTorchError::gpu(
            "GPU operation not implemented".to_string(),
        ))
    }

    // Generic CPU benchmark helper
    fn benchmark_cpu_operation<F, T>(&self, operation: F) -> RusTorchResult<f64>
    where
        F: Fn() -> T,
    {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = operation();
        }

        let start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let _ = operation();
        }
        let duration = start.elapsed();

        Ok(duration.as_secs_f64() * 1000.0)
    }

    /// Print comprehensive benchmark report
    /// 包括的ベンチマークレポートを出力
    fn print_comprehensive_report(&self) {
        println!("\n📊 Comprehensive Performance Benchmark Report");
        println!("==============================================");

        let total_benchmarks = self.results.len();
        let successful_benchmarks = self
            .results
            .iter()
            .filter(|r| r.error_message.is_none())
            .count();
        let failed_benchmarks = total_benchmarks - successful_benchmarks;

        println!("📈 Summary Statistics:");
        println!("  Total Benchmarks: {}", total_benchmarks);
        println!("  ✅ Successful: {}", successful_benchmarks);
        println!("  ❌ Failed: {}", failed_benchmarks);

        // Group results by operation type
        let mut operation_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            operation_groups
                .entry(result.operation_name.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        println!("\n🚀 Performance Results by Operation:");
        println!("=====================================");

        for (op_name, results) in operation_groups {
            println!("\n📊 {}", op_name);
            println!("{}", "-".repeat(op_name.len() + 4));

            println!(
                "{:<20} {:<12} {:<12} {:<10} {:<15} {:<15}",
                "Problem Size", "CPU (ms)", "GPU (ms)", "Speedup", "Throughput", "Bandwidth"
            );
            println!("{}", "-".repeat(100));

            for result in results {
                if result.error_message.is_some() {
                    println!("{:<20} {:<50}", result.problem_size, "❌ Failed");
                    continue;
                }

                let cpu_time = result
                    .cpu_time_ms
                    .map_or("N/A".to_string(), |t| format!("{:.2}", t));
                let gpu_time = result
                    .gpu_time_ms
                    .map_or("N/A".to_string(), |t| format!("{:.2}", t));
                let speedup = result
                    .speedup
                    .map_or("N/A".to_string(), |s| format!("{:.2}x", s));
                let throughput = result
                    .gpu_throughput_gops
                    .map_or("N/A".to_string(), |t| format!("{:.2} GOPS", t));
                let bandwidth = result
                    .memory_bandwidth_gb_s
                    .map_or("N/A".to_string(), |b| format!("{:.2} GB/s", b));

                println!(
                    "{:<20} {:<12} {:<12} {:<10} {:<15} {:<15}",
                    result.problem_size, cpu_time, gpu_time, speedup, throughput, bandwidth
                );
            }
        }

        // Performance insights
        self.print_performance_insights();
    }

    /// Print performance insights and recommendations
    /// パフォーマンスの洞察と推奨事項を出力
    fn print_performance_insights(&self) {
        println!("\n💡 Performance Insights & Recommendations:");
        println!("==========================================");

        let successful_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.error_message.is_none() && r.speedup.is_some())
            .collect();

        if successful_results.is_empty() {
            println!("⚠️  No successful GPU benchmarks to analyze.");
            return;
        }

        let avg_speedup: f64 = successful_results
            .iter()
            .filter_map(|r| r.speedup)
            .sum::<f64>()
            / successful_results.len() as f64;

        let max_speedup = successful_results
            .iter()
            .filter_map(|r| r.speedup)
            .fold(0.0f64, |a, b| a.max(b));

        let min_speedup = successful_results
            .iter()
            .filter_map(|r| r.speedup)
            .fold(f64::INFINITY, |a, b| a.min(b));

        println!("📈 Speedup Statistics:");
        println!("  Average GPU Speedup: {:.2}x", avg_speedup);
        println!("  Maximum GPU Speedup: {:.2}x", max_speedup);
        println!("  Minimum GPU Speedup: {:.2}x", min_speedup);

        // Best performing operations
        let mut best_operations: Vec<_> = successful_results
            .iter()
            .filter_map(|r| r.speedup.map(|s| (r, s)))
            .collect();
        best_operations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n🏆 Top 5 Best GPU Accelerated Operations:");
        for (i, (result, speedup)) in best_operations.iter().take(5).enumerate() {
            println!(
                "  {}. {} ({}): {:.2}x speedup",
                i + 1,
                result.operation_name,
                result.problem_size,
                speedup
            );
        }

        // Performance recommendations
        println!("\n🎯 Recommendations:");
        if avg_speedup < 2.0 {
            println!("  ⚡ GPU acceleration is moderate. Consider:");
            println!("     • Optimizing kernel launch parameters");
            println!("     • Using larger problem sizes");
            println!("     • Profiling memory access patterns");
        } else if avg_speedup < 10.0 {
            println!("  ✅ Good GPU acceleration achieved. Consider:");
            println!("     • Fine-tuning for specific operations");
            println!("     • Implementing tensor fusion");
        } else {
            println!("  🚀 Excellent GPU acceleration! Consider:");
            println!("     • Scaling to larger problems");
            println!("     • Multi-GPU implementations");
        }

        let failed_benchmarks = self.results.len() - successful_results.len();
        if failed_benchmarks > successful_results.len() / 2 {
            println!("  ⚠️  High failure rate detected. Check:");
            println!("     • GPU backend availability");
            println!("     • Kernel compilation issues");
            println!("     • Memory allocation problems");
        }
    }
}

impl Default for PerformanceBenchmark {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}
