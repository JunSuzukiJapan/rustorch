/// GPU kernel validation and correctness testing
/// GPUカーネル検証と正確性テスト
use crate::error::RusTorchResult;
// use crate::error::RusTorchError; // Currently unused
use crate::gpu::kernels::{AddKernel, GpuKernel, KernelExecutor, MatMulKernel};
use crate::gpu::DeviceType;

/// Validation results for GPU operations
/// GPU操作の検証結果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The GPU device type used for validation
    pub device: DeviceType,
    /// Name of the operation being validated
    pub operation: String,
    /// Whether the validation passed
    pub passed: bool,
    /// Error message if validation failed
    pub error_message: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Maximum error between expected and actual results
    pub max_error: f32,
}

/// GPU kernel validator
/// GPUカーネル検証器
pub struct GpuValidator {
    tolerance: f32,
}

impl GpuValidator {
    /// Create a new GPU validator
    /// 新しいGPU検証器を作成
    pub fn new(tolerance: f32) -> Self {
        GpuValidator { tolerance }
    }

    /// Validate all available GPU devices
    /// 利用可能なすべてのGPUデバイスを検証
    pub fn validate_all_devices(&self) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        let devices = vec![
            DeviceType::Cpu,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(0),
            #[cfg(feature = "metal")]
            DeviceType::Metal(0),
            #[cfg(feature = "opencl")]
            DeviceType::OpenCL(0),
        ];

        for device in devices {
            if !device.is_available() {
                continue;
            }

            // Validate element-wise addition
            results.push(self.validate_elementwise_add(device));

            // Validate matrix multiplication
            results.push(self.validate_matrix_multiplication(device));

            // Validate memory operations
            results.extend(self.validate_memory_operations(device));
        }

        results
    }

    /// Validate element-wise addition operation
    /// 要素ごと加算操作を検証
    pub fn validate_elementwise_add(&self, device: DeviceType) -> ValidationResult {
        let start_time = std::time::Instant::now();

        let size = 1024;
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut c = vec![0.0f32; size];
        let expected = vec![3.0f32; size];

        let executor = KernelExecutor::new(device);
        let kernel = AddKernel;

        let result = match self.execute_and_validate(
            &executor,
            &kernel,
            &[a.as_slice(), b.as_slice()],
            &mut [c.as_mut_slice()],
            &expected,
        ) {
            Ok(max_error) => ValidationResult {
                device,
                operation: "ElementwiseAdd".to_string(),
                passed: max_error <= self.tolerance,
                error_message: None,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error,
            },
            Err(e) => ValidationResult {
                device,
                operation: "ElementwiseAdd".to_string(),
                passed: false,
                error_message: Some(format!("{:?}", e)),
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error: f32::INFINITY,
            },
        };

        result
    }

    /// Validate matrix multiplication operation
    /// 行列乗算操作を検証
    pub fn validate_matrix_multiplication(&self, device: DeviceType) -> ValidationResult {
        let start_time = std::time::Instant::now();

        // Test with 4x4 matrices for simplicity
        let n = 4;
        let size = n * n;

        // Create test matrices: A = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
        // B = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] (identity matrix)
        let mut a = vec![0.0f32; size];
        let mut b = vec![0.0f32; size];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (i * n + j + 1) as f32;
                b[i * n + j] = if i == j { 1.0 } else { 0.0 };
            }
        }

        let mut c = vec![0.0f32; size];
        let expected = a.clone(); // A * I = A

        let executor = KernelExecutor::new(device);
        let kernel = MatMulKernel;

        let result = match self.execute_and_validate(
            &executor,
            &kernel,
            &[a.as_slice(), b.as_slice()],
            &mut [c.as_mut_slice()],
            &expected,
        ) {
            Ok(max_error) => ValidationResult {
                device,
                operation: "MatrixMultiplication".to_string(),
                passed: max_error <= self.tolerance,
                error_message: None,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error,
            },
            Err(e) => ValidationResult {
                device,
                operation: "MatrixMultiplication".to_string(),
                passed: false,
                error_message: Some(format!("{:?}", e)),
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error: f32::INFINITY,
            },
        };

        result
    }

    /// Validate memory operations (allocation, copy, deallocation)
    /// メモリ操作を検証（割り当て、コピー、解放）
    pub fn validate_memory_operations(&self, device: DeviceType) -> Vec<ValidationResult> {
        let mut results = Vec::new();

        match device {
            DeviceType::Cpu => {
                // CPU doesn't need special memory validation
                results.push(ValidationResult {
                    device,
                    operation: "MemoryOperations".to_string(),
                    passed: true,
                    error_message: None,
                    execution_time_ms: 0.0,
                    max_error: 0.0,
                });
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                results.push(self.validate_cuda_memory());
            }
            #[cfg(feature = "metal")]
            DeviceType::Metal(_) => {
                results.push(self.validate_metal_memory());
            }
            #[cfg(feature = "opencl")]
            DeviceType::OpenCL(_) => {
                results.push(self.validate_opencl_memory());
            }
            _ => {}
        }

        results
    }

    /// Execute kernel and validate results
    /// カーネルを実行して結果を検証
    fn execute_and_validate<K: GpuKernel<f32>>(
        &self,
        executor: &KernelExecutor,
        kernel: &K,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
        expected: &[f32],
    ) -> RusTorchResult<f32> {
        executor.execute_kernel(kernel, inputs, outputs)?;

        let max_error = outputs[0]
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        Ok(max_error)
    }

    #[cfg(feature = "cuda")]
    fn validate_cuda_memory(&self) -> ValidationResult {
        use crate::gpu::cuda_kernels::CudaBuffer;

        let start_time = std::time::Instant::now();
        let size = 1024;
        let test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let result = (|| -> RusTorchResult<f32> {
            // Test buffer creation
            let buffer = CudaBuffer::from_host_data(&test_data)?;

            // Test device-to-host copy
            let mut result_data = vec![0.0f32; size];
            buffer.copy_to_host(&mut result_data)?;

            // Validate data integrity
            let max_error = test_data
                .iter()
                .zip(result_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            Ok(max_error)
        })();

        match result {
            Ok(max_error) => ValidationResult {
                device: DeviceType::Cuda(0),
                operation: "CudaMemoryOperations".to_string(),
                passed: max_error <= self.tolerance,
                error_message: None,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error,
            },
            Err(e) => ValidationResult {
                device: DeviceType::Cuda(0),
                operation: "CudaMemoryOperations".to_string(),
                passed: false,
                error_message: Some(format!("{:?}", e)),
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error: f32::INFINITY,
            },
        }
    }

    #[cfg(feature = "metal")]
    fn validate_metal_memory(&self) -> ValidationResult {
        use crate::gpu::metal_kernels::MetalBuffer;

        let start_time = std::time::Instant::now();
        let size = 1024;
        let test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let result = (|| -> RusTorchResult<f32> {
            // Test buffer creation
            let buffer = MetalBuffer::from_host_data(&test_data)?;

            // Test device-to-host copy
            let mut result_data = vec![0.0f32; size];
            buffer.copy_to_host(&mut result_data)?;

            // Validate data integrity
            let max_error = test_data
                .iter()
                .zip(result_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            Ok(max_error)
        })();

        match result {
            Ok(max_error) => ValidationResult {
                device: DeviceType::Metal(0),
                operation: "MetalMemoryOperations".to_string(),
                passed: max_error <= self.tolerance,
                error_message: None,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error,
            },
            Err(e) => ValidationResult {
                device: DeviceType::Metal(0),
                operation: "MetalMemoryOperations".to_string(),
                passed: false,
                error_message: Some(format!("{:?}", e)),
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error: f32::INFINITY,
            },
        }
    }

    #[cfg(feature = "opencl")]
    fn validate_opencl_memory(&self) -> ValidationResult {
        use crate::gpu::opencl_kernels::OpenClBuffer;

        let start_time = std::time::Instant::now();
        let size = 1024;
        let test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let result = (|| -> RusTorchResult<f32> {
            // Test buffer creation
            let buffer = OpenClBuffer::from_host_data(&test_data)?;

            // Test device-to-host copy
            let mut result_data = vec![0.0f32; size];
            buffer.copy_to_host(&mut result_data)?;

            // Validate data integrity
            let max_error = test_data
                .iter()
                .zip(result_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            Ok(max_error)
        })();

        match result {
            Ok(max_error) => ValidationResult {
                device: DeviceType::OpenCL(0),
                operation: "OpenClMemoryOperations".to_string(),
                passed: max_error <= self.tolerance,
                error_message: None,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error,
            },
            Err(e) => ValidationResult {
                device: DeviceType::OpenCL(0),
                operation: "OpenClMemoryOperations".to_string(),
                passed: false,
                error_message: Some(format!("{:?}", e)),
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                max_error: f32::INFINITY,
            },
        }
    }

    /// Generate a validation report
    /// 検証レポートを生成
    pub fn generate_report(&self, results: &[ValidationResult]) -> String {
        let mut report = String::new();
        report.push_str("=== GPU Kernel Validation Report ===\n\n");

        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        report.push_str(&format!("Total Tests: {}\n", total_tests));
        report.push_str(&format!("Passed: {}\n", passed_tests));
        report.push_str(&format!("Failed: {}\n", failed_tests));
        report.push_str(&format!(
            "Success Rate: {:.1}%\n\n",
            (passed_tests as f64 / total_tests as f64) * 100.0
        ));

        // Group results by device
        let mut device_results: std::collections::HashMap<DeviceType, Vec<&ValidationResult>> =
            std::collections::HashMap::new();

        for result in results {
            device_results
                .entry(result.device)
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (device, device_results) in device_results {
            report.push_str(&format!("--- {} ---\n", device));

            for result in device_results {
                let status = if result.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "  {}: {} ({:.2}ms, max_error: {:.6})\n",
                    result.operation, status, result.execution_time_ms, result.max_error
                ));

                if let Some(ref error) = result.error_message {
                    report.push_str(&format!("    Error: {}\n", error));
                }
            }
            report.push('\n');
        }

        report
    }
}

/// Run comprehensive GPU validation
/// 包括的なGPU検証を実行
pub fn run_gpu_validation() -> Vec<ValidationResult> {
    let validator = GpuValidator::new(1e-5); // 0.00001 tolerance
    validator.validate_all_devices()
}

/// Print GPU validation report
/// GPU検証レポートを出力
pub fn print_gpu_validation_report() {
    let validator = GpuValidator::new(1e-5);
    let results = validator.validate_all_devices();
    let report = validator.generate_report(&results);
    println!("{}", report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_validator_creation() {
        let validator = GpuValidator::new(1e-5);
        assert_eq!(validator.tolerance, 1e-5);
    }

    #[test]
    fn test_cpu_validation() {
        let validator = GpuValidator::new(1e-5);
        let result = validator.validate_elementwise_add(DeviceType::Cpu);
        assert!(result.passed);
        assert_eq!(result.device, DeviceType::Cpu);
        assert_eq!(result.operation, "ElementwiseAdd");
        assert!(result.max_error <= 1e-5);
    }

    #[test]
    fn test_cpu_matrix_multiplication_validation() {
        let validator = GpuValidator::new(1e-5);
        let result = validator.validate_matrix_multiplication(DeviceType::Cpu);
        assert!(result.passed);
        assert_eq!(result.device, DeviceType::Cpu);
        assert_eq!(result.operation, "MatrixMultiplication");
        assert!(result.max_error <= 1e-5);
    }

    #[test]
    fn test_validation_report_generation() {
        let validator = GpuValidator::new(1e-5);
        let results = vec![ValidationResult {
            device: DeviceType::Cpu,
            operation: "Test".to_string(),
            passed: true,
            error_message: None,
            execution_time_ms: 1.0,
            max_error: 0.0,
        }];

        let report = validator.generate_report(&results);
        assert!(report.contains("Total Tests: 1"));
        assert!(report.contains("Passed: 1"));
        assert!(report.contains("Success Rate: 100.0%"));
    }
}
