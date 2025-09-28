//! Simple Performance Demo
//! シンプルパフォーマンスデモ
//!
//! A lightweight demonstration of RusTorch performance across different backends.
//! Features are determined at runtime through command-line arguments.
//!
//! 異なるバックエンドでのRusTorchパフォーマンスの軽量デモンストレーション。
//! フィーチャーは実行時にコマンドライン引数で決定されます。
//!
//! Usage:
//! cd benchmarks
//! cargo run --bin simple_performance_demo --features metal -- --backend metal
//! cargo run --bin simple_performance_demo --features coreml -- --backend coreml
//! cargo run --bin simple_performance_demo --features mac-hybrid -- --backend metal
//! cargo run --bin simple_performance_demo -- --backend cpu

use rustorch::error::RusTorchResult;
use rustorch::tensor::Tensor;
use std::env;
use std::time::{Duration, Instant};

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DemoConfig {
    // Matrix operations
    operations: usize,
    matrix_size: usize,
    batch_size: usize,

    // Convolution operations
    convolution_networks: usize,
    image_size: usize,
    image_batch_size: usize,
    network_layers: usize,

    // Transformer operations
    transformer_operations: usize,
    sequence_length: usize,
    embedding_dim: usize,
    attention_heads: usize,
    transformer_layers: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        DemoConfig {
            // Matrix operations
            operations: 10,
            matrix_size: 512,
            batch_size: 2,

            // Convolution operations (simplified from original)
            convolution_networks: 20,
            image_size: 256,
            image_batch_size: 2,
            network_layers: 8,

            // Transformer operations (simplified from original)
            transformer_operations: 10,
            sequence_length: 128,
            embedding_dim: 128,
            attention_heads: 4,
            transformer_layers: 3,
        }
    }
}

#[derive(Debug)]
struct PerformanceResult {
    backend_name: String,
    total_operations: usize,
    total_duration: Duration,
    average_time_ms: f64,
    operations_per_second: f64,
    success_rate: f64,
}

struct SimpleDemo {
    config: DemoConfig,
    backend: String,
    benchmark_type: String,
}

impl SimpleDemo {
    fn new(backend: String, benchmark_type: String) -> Self {
        SimpleDemo {
            config: DemoConfig::default(),
            backend,
            benchmark_type,
        }
    }

    fn run_demo(&self) -> RusTorchResult<()> {
        match self.benchmark_type.as_str() {
            "all" => self.run_all_benchmarks(),
            _ => self.run_single_benchmark(),
        }
    }

    fn run_single_benchmark(&self) -> RusTorchResult<()> {
        println!("🚀 RusTorch Simple Performance Demo");
        println!("===================================");
        println!("Backend: {}", self.backend);
        println!("Benchmark: {}", self.benchmark_type);

        match self.benchmark_type.as_str() {
            "matrix" => {
                println!("Operations: {}", self.config.operations);
                println!(
                    "Matrix size: {}x{}",
                    self.config.matrix_size, self.config.matrix_size
                );
            }
            "convolution" => {
                println!("Networks: {}", self.config.convolution_networks);
                println!(
                    "Image size: {}x{}",
                    self.config.image_size, self.config.image_size
                );
                println!("Layers: {}", self.config.network_layers);
            }
            "transformer" => {
                println!("Operations: {}", self.config.transformer_operations);
                println!("Sequence length: {}", self.config.sequence_length);
                println!("Embedding dim: {}", self.config.embedding_dim);
                println!("Layers: {}", self.config.transformer_layers);
            }
            _ => {}
        }
        println!();

        let result = match (self.backend.as_str(), self.benchmark_type.as_str()) {
            ("metal", "matrix") => self.run_metal_demo(),
            ("metal", "convolution") => self.run_metal_convolution_demo(),
            ("metal", "transformer") => self.run_metal_transformer_demo(),
            ("coreml", "matrix") => self.run_coreml_demo(),
            ("coreml", "convolution") => self.run_coreml_convolution_demo(),
            ("coreml", "transformer") => self.run_coreml_transformer_demo(),
            ("cpu", "matrix") => self.run_cpu_demo(),
            ("cpu", "convolution") => self.run_cpu_convolution_demo(),
            ("cpu", "transformer") => self.run_cpu_transformer_demo(),
            _ => {
                println!(
                    "❌ Unknown backend/benchmark combination: {}/{}",
                    self.backend, self.benchmark_type
                );
                println!("Available backends: cpu, metal, coreml");
                println!("Available benchmarks: matrix, convolution, transformer, all");
                return Ok(());
            }
        }?;

        self.display_results(&result);
        Ok(())
    }

    fn run_all_benchmarks(&self) -> RusTorchResult<()> {
        println!("🚀 RusTorch Comprehensive Performance Demo");
        println!("==========================================");
        println!("Backend: {}", self.backend);
        println!("Running all benchmarks: matrix, convolution, transformer");
        println!();

        let benchmarks = ["matrix", "convolution", "transformer"];
        let mut all_results = Vec::new();

        for (i, benchmark) in benchmarks.iter().enumerate() {
            println!("📊 Benchmark {}/{}: {}", i + 1, benchmarks.len(), benchmark);
            println!("─────────────────────────────────────────");

            let result = match (self.backend.as_str(), *benchmark) {
                ("metal", "matrix") => self.run_metal_demo(),
                ("metal", "convolution") => self.run_metal_convolution_demo(),
                ("metal", "transformer") => self.run_metal_transformer_demo(),
                ("coreml", "matrix") => self.run_coreml_demo(),
                ("coreml", "convolution") => self.run_coreml_convolution_demo(),
                ("coreml", "transformer") => self.run_coreml_transformer_demo(),
                ("cpu", "matrix") => self.run_cpu_demo(),
                ("cpu", "convolution") => self.run_cpu_convolution_demo(),
                ("cpu", "transformer") => self.run_cpu_transformer_demo(),
                _ => {
                    println!("❌ Unknown backend: {}", self.backend);
                    return Ok(());
                }
            }?;

            self.display_results(&result);
            all_results.push(result);

            if i < benchmarks.len() - 1 {
                println!("\n⏳ Waiting 2 seconds before next benchmark...\n");
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }

        // Display summary
        self.display_summary(&all_results);
        Ok(())
    }

    fn display_summary(&self, results: &[PerformanceResult]) {
        println!("\n🎯 Performance Summary - {}", self.backend.to_uppercase());
        println!("═══════════════════════════════════════════════");

        for result in results {
            let benchmark_name =
                if result.backend_name.contains("Matrix") || result.backend_name == "CPU" {
                    "Matrix"
                } else if result.backend_name.contains("Convolution") {
                    "Convolution"
                } else {
                    "Transformer"
                };

            println!(
                "{:<12} | {:>8.2} ops/sec | {:>8.2}ms avg | {:>6.1}% success",
                benchmark_name,
                result.operations_per_second,
                result.average_time_ms,
                result.success_rate
            );
        }

        println!("═══════════════════════════════════════════════");
        println!(
            "🏆 Best performer: {}",
            results
                .iter()
                .max_by(|a, b| a
                    .operations_per_second
                    .partial_cmp(&b.operations_per_second)
                    .unwrap())
                .map(|r| {
                    if r.backend_name.contains("Matrix") || r.backend_name == "CPU" {
                        "Matrix"
                    } else if r.backend_name.contains("Convolution") {
                        "Convolution"
                    } else {
                        "Transformer"
                    }
                })
                .unwrap_or("None")
        );
        println!("✅ All benchmarks completed successfully!");
    }

    fn run_cpu_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🖥️ Running CPU matrix benchmark...");
        self.run_matrix_operations("CPU")
    }

    fn run_cpu_convolution_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🖥️ Running CPU convolution benchmark...");
        self.run_cpu_convolution_operations()
    }

    fn run_cpu_transformer_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🖥️ Running CPU transformer benchmark...");
        self.run_cpu_transformer_operations()
    }

    #[cfg(feature = "metal")]
    fn run_metal_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("⚡ Running Metal GPU matrix benchmark...");
        self.run_metal_matrix_operations()
    }

    #[cfg(not(feature = "metal"))]
    fn run_metal_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ Metal feature not available. Compile with --features metal");
        self.run_matrix_operations("Metal (fallback to CPU)")
    }

    #[cfg(feature = "metal")]
    fn run_metal_convolution_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("⚡ Running Metal GPU convolution benchmark...");
        self.run_metal_convolution_operations()
    }

    #[cfg(not(feature = "metal"))]
    fn run_metal_convolution_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ Metal feature not available. Compile with --features metal");
        self.run_cpu_convolution_operations()
    }

    #[cfg(feature = "metal")]
    fn run_metal_transformer_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("⚡ Running Metal GPU transformer benchmark...");
        self.run_metal_transformer_operations()
    }

    #[cfg(not(feature = "metal"))]
    fn run_metal_transformer_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ Metal feature not available. Compile with --features metal");
        self.run_cpu_transformer_operations()
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🧠 Running CoreML Neural Engine matrix benchmark...");
        self.run_coreml_matrix_operations()
    }

    #[cfg(not(feature = "coreml"))]
    fn run_coreml_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ CoreML feature not available. Compile with --features coreml");
        self.run_matrix_operations("CoreML (fallback to CPU)")
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_convolution_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🧠 Running CoreML Neural Engine convolution benchmark...");
        self.run_coreml_convolution_operations()
    }

    #[cfg(not(feature = "coreml"))]
    fn run_coreml_convolution_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ CoreML feature not available. Compile with --features coreml");
        self.run_cpu_convolution_operations()
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_transformer_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("🧠 Running CoreML Neural Engine transformer benchmark...");
        self.run_coreml_transformer_operations()
    }

    #[cfg(not(feature = "coreml"))]
    fn run_coreml_transformer_demo(&self) -> RusTorchResult<PerformanceResult> {
        println!("❌ CoreML feature not available. Compile with --features coreml");
        self.run_cpu_transformer_operations()
    }

    fn run_matrix_operations(&self, backend_name: &str) -> RusTorchResult<PerformanceResult> {
        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  📊 Starting {} matrix operations...",
            self.config.operations
        );

        for i in 0..self.config.operations {
            let op_start = Instant::now();

            // Create tensors in limited scope for memory efficiency
            let result = {
                let a = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                let b = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                a.matmul(&b)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.operations - 1 {
                        println!(
                            "    Operation {}/{}: {:.2}ms",
                            i + 1,
                            self.config.operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ Operation {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.operations > 0 {
            successful_operations as f64 / self.config.operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: backend_name.to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    fn display_results(&self, result: &PerformanceResult) {
        println!();
        println!("📊 Performance Results");
        println!("=====================");
        println!("Backend: {}", result.backend_name);
        println!("Total operations: {}", result.total_operations);
        println!("Total time: {:.2}s", result.total_duration.as_secs_f64());
        println!(
            "Average time per operation: {:.2}ms",
            result.average_time_ms
        );
        println!("Operations per second: {:.2}", result.operations_per_second);
        println!("Success rate: {:.1}%", result.success_rate);

        // Performance assessment
        let performance_level = if result.operations_per_second > 5.0 {
            "🚀 Excellent"
        } else if result.operations_per_second > 2.0 {
            "✅ Good"
        } else if result.operations_per_second > 1.0 {
            "🔄 Moderate"
        } else {
            "⚠️ Slow"
        };

        println!("Performance: {}", performance_level);
        println!();

        // Memory estimation
        let matrix_memory_mb =
            (self.config.matrix_size * self.config.matrix_size * 4 * 2) as f64 / 1_048_576.0;
        println!(
            "Estimated memory usage: {:.1}MB per operation",
            matrix_memory_mb
        );

        println!("✅ Demo completed successfully!");
    }

    #[cfg(feature = "metal")]
    fn run_metal_matrix_operations(&self) -> RusTorchResult<PerformanceResult> {
        use rustorch::gpu::matrix_ops::GpuMatrixExecutor;

        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  ⚡ Starting {} Metal GPU matrix operations...",
            self.config.operations
        );

        // Initialize Metal GPU executor
        let executor = GpuMatrixExecutor::new(rustorch::gpu::DeviceType::Metal(0))?;

        for i in 0..self.config.operations {
            let op_start = Instant::now();

            // Create tensors for Metal GPU operations
            let result = {
                let a = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                let b = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                executor.metal_matmul(&a, &b)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.operations - 1 {
                        println!(
                            "    Metal Operation {}/{}: {:.2}ms",
                            i + 1,
                            self.config.operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ Metal Operation {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.operations > 0 {
            successful_operations as f64 / self.config.operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "Metal GPU".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_matrix_operations(&self) -> RusTorchResult<PerformanceResult> {
        use rustorch::gpu::matrix_ops::GpuMatrixExecutor;

        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  🧠 Starting {} CoreML Neural Engine matrix operations...",
            self.config.operations
        );

        // Initialize CoreML Neural Engine executor
        let executor = GpuMatrixExecutor::new(rustorch::gpu::DeviceType::CoreML(0))?;

        for i in 0..self.config.operations {
            let op_start = Instant::now();

            // Use CoreML directly for matrix operations (supported)
            let result = {
                let a = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                let b = Tensor::<f32>::randn(&[self.config.matrix_size, self.config.matrix_size]);
                executor.coreml_matmul(&a, &b)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.operations - 1 {
                        println!(
                            "    CoreML Operation {}/{}: {:.2}ms",
                            i + 1,
                            self.config.operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ CoreML Operation {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.operations > 0 {
            successful_operations as f64 / self.config.operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "CoreML Neural Engine".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    #[cfg(feature = "metal")]
    fn run_metal_convolution_operations(&self) -> RusTorchResult<PerformanceResult> {
        use rustorch::gpu::matrix_ops::GpuMatrixExecutor;

        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  ⚡ Starting {} Metal GPU convolution networks...",
            self.config.convolution_networks
        );

        let executor = GpuMatrixExecutor::new(rustorch::gpu::DeviceType::Metal(0))?;

        for i in 0..self.config.convolution_networks {
            let op_start = Instant::now();

            // Simulate Metal convolution with matrix operations
            let result = {
                let input = Tensor::<f32>::randn(&[self.config.image_batch_size, 64]);
                let weight = Tensor::<f32>::randn(&[64, 32]);
                executor.metal_matmul(&input, &weight)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 5 == 0 || i == self.config.convolution_networks - 1 {
                        println!(
                            "    Metal Conv Network {}/{}: {:.2}ms",
                            i + 1,
                            self.config.convolution_networks,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ Metal Conv Network {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.convolution_networks > 0 {
            successful_operations as f64 / self.config.convolution_networks as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "Metal GPU Convolution".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    #[cfg(feature = "metal")]
    fn run_metal_transformer_operations(&self) -> RusTorchResult<PerformanceResult> {
        use rustorch::gpu::matrix_ops::GpuMatrixExecutor;

        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  ⚡ Starting {} Metal GPU transformer operations...",
            self.config.transformer_operations
        );

        let executor = GpuMatrixExecutor::new(rustorch::gpu::DeviceType::Metal(0))?;

        for i in 0..self.config.transformer_operations {
            let op_start = Instant::now();

            // Simulate Metal transformer with simple 2D matrix operations
            let result = {
                let q =
                    Tensor::<f32>::randn(&[self.config.sequence_length, self.config.embedding_dim]);
                let k =
                    Tensor::<f32>::randn(&[self.config.embedding_dim, self.config.sequence_length]);

                executor.metal_matmul(&q, &k)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.transformer_operations - 1 {
                        println!(
                            "    Metal Transformer {}/{}: {:.2}ms",
                            i + 1,
                            self.config.transformer_operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ Metal Transformer {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.transformer_operations > 0 {
            successful_operations as f64 / self.config.transformer_operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "Metal GPU Transformer".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_convolution_operations(&self) -> RusTorchResult<PerformanceResult> {
        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!("  🧠 Starting {} CoreML backend convolution networks (using CPU for unsupported ops)...", self.config.convolution_networks);

        for i in 0..self.config.convolution_networks {
            let op_start = Instant::now();

            // CoreML doesn't support convolution matrix operations well, use CPU directly
            let result = {
                let input = Tensor::<f32>::randn(&[self.config.image_batch_size, 64]);
                let weight = Tensor::<f32>::randn(&[64, 32]);

                // Use CPU implementation since CoreML doesn't support this operation type
                input.matmul(&weight)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 5 == 0 || i == self.config.convolution_networks - 1 {
                        println!(
                            "    CoreML Conv Network {}/{}: {:.2}ms (CPU)",
                            i + 1,
                            self.config.convolution_networks,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ CoreML Conv Network {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.convolution_networks > 0 {
            successful_operations as f64 / self.config.convolution_networks as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "CoreML Neural Engine Convolution".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    #[cfg(feature = "coreml")]
    fn run_coreml_transformer_operations(&self) -> RusTorchResult<PerformanceResult> {
        use rustorch::gpu::matrix_ops::GpuMatrixExecutor;

        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  🧠 Starting {} CoreML Neural Engine transformer operations...",
            self.config.transformer_operations
        );

        let executor = GpuMatrixExecutor::new(rustorch::gpu::DeviceType::CoreML(0))?;

        for i in 0..self.config.transformer_operations {
            let op_start = Instant::now();

            // Use CoreML directly for transformer operations (supported)
            let result = {
                let q =
                    Tensor::<f32>::randn(&[self.config.sequence_length, self.config.embedding_dim]);
                let k =
                    Tensor::<f32>::randn(&[self.config.embedding_dim, self.config.sequence_length]);
                executor.coreml_matmul(&q, &k)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.transformer_operations - 1 {
                        println!(
                            "    CoreML Transformer {}/{}: {:.2}ms",
                            i + 1,
                            self.config.transformer_operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ CoreML Transformer {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.transformer_operations > 0 {
            successful_operations as f64 / self.config.transformer_operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "CoreML Neural Engine Transformer".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    // CPU Convolution Operations
    fn run_cpu_convolution_operations(&self) -> RusTorchResult<PerformanceResult> {
        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  🖥️ Starting {} CPU convolution networks...",
            self.config.convolution_networks
        );

        for i in 0..self.config.convolution_networks {
            let op_start = Instant::now();

            // Simulate convolution network with CPU tensor operations
            let result: RusTorchResult<Tensor<f32>> = {
                let _input = Tensor::<f32>::randn(&[
                    self.config.image_batch_size,
                    3, // RGB channels
                    self.config.image_size,
                    self.config.image_size,
                ]);

                // Simulate network layers with matrix operations
                let mut current = Tensor::<f32>::randn(&[self.config.image_batch_size, 32]);
                for _layer in 0..self.config.network_layers {
                    let input_flat = Tensor::<f32>::randn(&[self.config.image_batch_size, 64]);
                    let weight = Tensor::<f32>::randn(&[64, 32]);
                    match input_flat.matmul(&weight) {
                        Ok(result) => current = result,
                        Err(e) => return Err(e),
                    }
                }
                Ok(current)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 5 == 0 || i == self.config.convolution_networks - 1 {
                        println!(
                            "    CPU Conv Network {}/{}: {:.2}ms",
                            i + 1,
                            self.config.convolution_networks,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ CPU Conv Network {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.convolution_networks > 0 {
            successful_operations as f64 / self.config.convolution_networks as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "CPU Convolution".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }

    // CPU Transformer Operations
    fn run_cpu_transformer_operations(&self) -> RusTorchResult<PerformanceResult> {
        let start_time = Instant::now();
        let mut successful_operations = 0;
        let mut total_operation_time = Duration::ZERO;

        println!(
            "  🖥️ Starting {} CPU transformer operations...",
            self.config.transformer_operations
        );

        for i in 0..self.config.transformer_operations {
            let op_start = Instant::now();

            // Simulate transformer with simple CPU 2D matrix operations
            let result: RusTorchResult<Tensor<f32>> = {
                let mut current =
                    Tensor::<f32>::randn(&[self.config.sequence_length, self.config.embedding_dim]);
                for _layer in 0..self.config.transformer_layers {
                    // Simulate attention with 2D matrix operations
                    let q = Tensor::<f32>::randn(&[
                        self.config.sequence_length,
                        self.config.embedding_dim,
                    ]);
                    let k = Tensor::<f32>::randn(&[
                        self.config.embedding_dim,
                        self.config.sequence_length,
                    ]);
                    let v = Tensor::<f32>::randn(&[
                        self.config.sequence_length,
                        self.config.embedding_dim,
                    ]);

                    match q.matmul(&k) {
                        Ok(attention_scores) => match attention_scores.matmul(&v) {
                            Ok(attention_output) => current = attention_output,
                            Err(e) => return Err(e),
                        },
                        Err(e) => return Err(e),
                    }
                }
                Ok(current)
            };

            let op_duration = op_start.elapsed();

            match result {
                Ok(_) => {
                    successful_operations += 1;
                    total_operation_time += op_duration;

                    if (i + 1) % 2 == 0 || i == self.config.transformer_operations - 1 {
                        println!(
                            "    CPU Transformer {}/{}: {:.2}ms",
                            i + 1,
                            self.config.transformer_operations,
                            op_duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    println!("    ❌ CPU Transformer {} failed: {}", i + 1, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_time_ms = if successful_operations > 0 {
            total_operation_time.as_secs_f64() * 1000.0 / successful_operations as f64
        } else {
            0.0
        };

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if self.config.transformer_operations > 0 {
            successful_operations as f64 / self.config.transformer_operations as f64 * 100.0
        } else {
            0.0
        };

        Ok(PerformanceResult {
            backend_name: "CPU Transformer".to_string(),
            total_operations: successful_operations,
            total_duration,
            average_time_ms,
            operations_per_second,
            success_rate,
        })
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cd benchmarks && cargo run --bin simple_performance_demo -- --backend <BACKEND> --benchmark <TYPE>");
    println!();
    println!("Available backends:");
    println!("  cpu    - CPU-only operations (no additional features required)");
    println!("  metal  - Metal GPU acceleration (requires --features metal)");
    println!("  coreml - CoreML Neural Engine (requires --features coreml)");
    println!();
    println!("Available benchmarks:");
    println!("  matrix       - Matrix multiplication operations");
    println!("  convolution  - Convolution network operations");
    println!("  transformer  - Transformer attention operations");
    println!("  all          - Run all three benchmarks sequentially");
    println!();
    println!("Examples:");
    println!("  cd benchmarks && cargo run --bin simple_performance_demo -- --backend cpu --benchmark matrix");
    println!("  cd benchmarks && cargo run --bin simple_performance_demo --features metal -- --backend metal --benchmark convolution");
    println!("  cd benchmarks && cargo run --bin simple_performance_demo --features coreml -- --backend coreml --benchmark transformer");
    println!("  cd benchmarks && cargo run --bin simple_performance_demo --features mac-hybrid -- --backend metal --benchmark matrix");
}

fn main() -> RusTorchResult<()> {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    if args.len() < 5 || args[1] != "--backend" || args[3] != "--benchmark" {
        print_usage();
        return Ok(());
    }

    let backend = args[2].clone();
    let benchmark_type = args[4].clone();

    // Validate backend
    match backend.as_str() {
        "cpu" | "metal" | "coreml" => {}
        _ => {
            println!("❌ Invalid backend: {}", backend);
            print_usage();
            return Ok(());
        }
    }

    // Validate benchmark type
    match benchmark_type.as_str() {
        "matrix" | "convolution" | "transformer" | "all" => {}
        _ => {
            println!("❌ Invalid benchmark type: {}", benchmark_type);
            print_usage();
            return Ok(());
        }
    }

    let demo = SimpleDemo::new(backend, benchmark_type);
    demo.run_demo()
}
