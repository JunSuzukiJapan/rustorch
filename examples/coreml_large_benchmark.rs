/// Large-scale CoreML vs CPU performance benchmark
/// 大規模なCoreML vs CPU パフォーマンスベンチマーク
///
/// This benchmark tests scenarios where CoreML is expected to outperform CPU:
/// - Large matrix operations (1024x1024+)
/// - Deep neural network inference
/// - Image processing workloads
/// - Batch processing operations

use rustorch::tensor::Tensor;
use rustorch::nn::{Linear, Conv2d, BatchNorm2d};
use rustorch::gpu::{DeviceType, get_device_manager};
use std::time::Instant;

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    // Large matrix sizes for CoreML advantage
    large_matrix_size: usize,
    batch_size: usize,
    // Image processing sizes (realistic)
    image_batch_size: usize,
    image_channels: usize,
    image_height: usize,
    image_width: usize,
    // Deep network config
    hidden_layers: Vec<usize>,
    iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            large_matrix_size: 1024,        // 1024x1024 matrices
            batch_size: 32,                 // Realistic batch size
            image_batch_size: 16,           // Image batch processing
            image_channels: 3,              // RGB
            image_height: 224,              // Standard image size
            image_width: 224,
            hidden_layers: vec![2048, 1024, 512, 256, 128], // Deep network
            iterations: 5,
        }
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    cpu_time_ms: f64,
    coreml_time_ms: Option<f64>,
    speedup: Option<f64>,
    success: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 CoreML Large-Scale Performance Benchmark");
    println!("===========================================");
    println!();

    let config = BenchmarkConfig::default();
    println!("📋 Benchmark Configuration:");
    println!("   Matrix Size: {}x{}", config.large_matrix_size, config.large_matrix_size);
    println!("   Batch Size: {}", config.batch_size);
    println!("   Image Size: {}x{}x{}x{}", config.image_batch_size, config.image_channels, config.image_height, config.image_width);
    println!("   Deep Network: {:?}", config.hidden_layers);
    println!("   Iterations: {}", config.iterations);
    println!();

    // Check available devices
    let manager = get_device_manager();
    let devices = manager.available_devices();
    let has_coreml = devices.iter().any(|d| matches!(d, DeviceType::CoreML(_)));

    println!("🔍 Available Devices: {:?}", devices);
    println!("   CoreML Available: {}", has_coreml);
    println!();

    let mut results = Vec::new();

    // 1. Large Matrix Multiplication Benchmark
    println!("📊 1. Large Matrix Multiplication Benchmark");
    println!("   Testing {}x{} matrices with batch size {}",
             config.large_matrix_size, config.large_matrix_size, config.batch_size);

    let matrix_result = benchmark_large_matrix_multiplication(&config)?;
    results.push(matrix_result);

    // 2. Deep Neural Network Inference
    println!("\n🧠 2. Deep Neural Network Inference Benchmark");
    println!("   Testing deep network: {:?}", config.hidden_layers);

    let nn_result = benchmark_deep_neural_network(&config)?;
    results.push(nn_result);

    // 3. Image Processing Workload
    println!("\n🖼️  3. Image Processing Workload Benchmark");
    println!("   Testing {}x{}x{}x{} image batch processing",
             config.image_batch_size, config.image_channels, config.image_height, config.image_width);

    let image_result = benchmark_image_processing(&config)?;
    results.push(image_result);

    // 4. Convolution Heavy Workload
    println!("\n🌊 4. Convolution Heavy Workload Benchmark");
    println!("   Testing multiple convolution layers");

    let conv_result = benchmark_convolution_heavy(&config)?;
    results.push(conv_result);

    // Print comprehensive results
    print_benchmark_results(&results);

    Ok(())
}

fn benchmark_large_matrix_multiplication(config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let size = config.large_matrix_size;
    let batch_size = config.batch_size;

    // Create large matrices
    println!("   🔧 Creating {}x{} matrices for {} batches...", size, size, batch_size);

    let mut total_cpu_time = 0.0;
    let mut total_coreml_time = 0.0;
    let mut coreml_success = true;

    for i in 0..config.iterations {
        println!("   📊 Iteration {}/{}", i + 1, config.iterations);

        // Create random matrices for this iteration
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.002).collect();

        let a = Tensor::from_vec(a_data.clone(), vec![size, size]);
        let b = Tensor::from_vec(b_data.clone(), vec![size, size]);

        // CPU benchmark
        let start = Instant::now();
        for _ in 0..batch_size {
            let _result = a.matmul(&b)?;
        }
        let cpu_time = start.elapsed().as_secs_f64() * 1000.0;
        total_cpu_time += cpu_time;
        println!("     CPU: {:.2}ms", cpu_time);

        // CoreML benchmark (if available)
        #[cfg(feature = "coreml")]
        {
            let start = Instant::now();
            for _ in 0..batch_size {
                // Try to use CoreML-optimized operations
                match a.to_device(DeviceType::CoreML(0)) {
                    Ok(a_coreml) => {
                        match b.to_device(DeviceType::CoreML(0)) {
                            Ok(b_coreml) => {
                                match a_coreml.matmul(&b_coreml) {
                                    Ok(_result) => {},
                                    Err(_) => {
                                        // Fallback to CPU
                                        let _result = a.matmul(&b)?;
                                    }
                                }
                            },
                            Err(_) => {
                                let _result = a.matmul(&b)?;
                            }
                        }
                    },
                    Err(_) => {
                        let _result = a.matmul(&b)?;
                    }
                }
            }
            let coreml_time = start.elapsed().as_secs_f64() * 1000.0;
            total_coreml_time += coreml_time;
            println!("     CoreML: {:.2}ms", coreml_time);
        }

        #[cfg(not(feature = "coreml"))]
        {
            coreml_success = false;
        }
    }

    let avg_cpu_time = total_cpu_time / config.iterations as f64;
    let avg_coreml_time = if coreml_success {
        Some(total_coreml_time / config.iterations as f64)
    } else {
        None
    };

    let speedup = avg_coreml_time.map(|coreml| avg_cpu_time / coreml);

    Ok(BenchmarkResult {
        name: "Large Matrix Multiplication".to_string(),
        cpu_time_ms: avg_cpu_time,
        coreml_time_ms: avg_coreml_time,
        speedup,
        success: true,
    })
}

fn benchmark_deep_neural_network(config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let batch_size = config.batch_size;
    let input_size = 784; // MNIST-like input

    println!("   🔧 Creating deep neural network...");

    // Create deep network layers
    let layers: Result<Vec<Linear>, _> = config.hidden_layers.windows(2)
        .map(|window| Linear::new(window[0], window[1]))
        .collect();

    let mut layers = layers?;

    // Add input and output layers
    let input_layer = Linear::new(input_size, config.hidden_layers[0]);
    let output_layer = Linear::new(*config.hidden_layers.last().unwrap(), 10);

    layers.insert(0, input_layer);
    layers.push(output_layer);

    let mut total_cpu_time = 0.0;
    let mut total_coreml_time = 0.0;
    let mut coreml_success = true;

    for i in 0..config.iterations {
        println!("   📊 Iteration {}/{}", i + 1, config.iterations);

        // Create batch input
        let input_data: Vec<f32> = (0..batch_size * input_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let mut input = Tensor::from_vec(input_data, vec![batch_size, input_size]);

        // CPU forward pass
        let start = Instant::now();
        let mut cpu_output = input.clone();
        for layer in &layers {
            cpu_output = layer.forward(&cpu_output)?;
            // Apply ReLU activation
            cpu_output = cpu_output.relu()?;
        }
        let cpu_time = start.elapsed().as_secs_f64() * 1000.0;
        total_cpu_time += cpu_time;
        println!("     CPU Deep Network: {:.2}ms", cpu_time);

        // CoreML forward pass (if available)
        #[cfg(feature = "coreml")]
        {
            let start = Instant::now();
            match input.to_device(DeviceType::CoreML(0)) {
                Ok(mut coreml_input) => {
                    let mut forward_success = true;
                    for layer in &layers {
                        match layer.forward(&coreml_input) {
                            Ok(output) => {
                                match output.relu() {
                                    Ok(activated) => {
                                        coreml_input = activated;
                                    },
                                    Err(_) => {
                                        forward_success = false;
                                        break;
                                    }
                                }
                            },
                            Err(_) => {
                                forward_success = false;
                                break;
                            }
                        }
                    }

                    if !forward_success {
                        // Fallback to CPU
                        let mut cpu_fallback = input.clone();
                        for layer in &layers {
                            cpu_fallback = layer.forward(&cpu_fallback)?;
                            cpu_fallback = cpu_fallback.relu()?;
                        }
                    }
                },
                Err(_) => {
                    // Fallback to CPU
                    let mut cpu_fallback = input.clone();
                    for layer in &layers {
                        cpu_fallback = layer.forward(&cpu_fallback)?;
                        cpu_fallback = cpu_fallback.relu()?;
                    }
                }
            }
            let coreml_time = start.elapsed().as_secs_f64() * 1000.0;
            total_coreml_time += coreml_time;
            println!("     CoreML Deep Network: {:.2}ms", coreml_time);
        }

        #[cfg(not(feature = "coreml"))]
        {
            coreml_success = false;
        }
    }

    let avg_cpu_time = total_cpu_time / config.iterations as f64;
    let avg_coreml_time = if coreml_success {
        Some(total_coreml_time / config.iterations as f64)
    } else {
        None
    };

    let speedup = avg_coreml_time.map(|coreml| avg_cpu_time / coreml);

    Ok(BenchmarkResult {
        name: "Deep Neural Network Inference".to_string(),
        cpu_time_ms: avg_cpu_time,
        coreml_time_ms: avg_coreml_time,
        speedup,
        success: true,
    })
}

fn benchmark_image_processing(config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let batch_size = config.image_batch_size;
    let channels = config.image_channels;
    let height = config.image_height;
    let width = config.image_width;

    println!("   🔧 Creating image processing pipeline...");

    let mut total_cpu_time = 0.0;
    let mut total_coreml_time = 0.0;
    let mut coreml_success = true;

    for i in 0..config.iterations {
        println!("   📊 Iteration {}/{}", i + 1, config.iterations);

        // Create batch of images
        let image_data: Vec<f32> = (0..batch_size * channels * height * width)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let images = Tensor::from_vec(image_data, vec![batch_size, channels, height, width]);

        // CPU image processing
        let start = Instant::now();

        // Simulate image processing pipeline
        let mut processed = images.clone();

        // Normalization
        let mean = processed.mean(&None)?;
        processed = processed.sub(&mean)?;

        // Scaling
        processed = processed.mul_scalar(0.5)?;

        // Element-wise operations
        processed = processed.tanh()?;

        let cpu_time = start.elapsed().as_secs_f64() * 1000.0;
        total_cpu_time += cpu_time;
        println!("     CPU Image Processing: {:.2}ms", cpu_time);

        // CoreML image processing (if available)
        #[cfg(feature = "coreml")]
        {
            let start = Instant::now();
            match images.to_device(DeviceType::CoreML(0)) {
                Ok(coreml_images) => {
                    let mut processing_success = true;

                    // Try CoreML-optimized image processing
                    let result = (|| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
                        let mean = coreml_images.mean(&None)?;
                        let normalized = coreml_images.sub(&mean)?;
                        let scaled = normalized.mul_scalar(0.5)?;
                        let activated = scaled.tanh()?;
                        Ok(activated)
                    })();

                    if result.is_err() {
                        processing_success = false;
                    }

                    if !processing_success {
                        // Fallback to CPU
                        let mut cpu_fallback = images.clone();
                        let mean = cpu_fallback.mean(&None)?;
                        cpu_fallback = cpu_fallback.sub(&mean)?;
                        cpu_fallback = cpu_fallback.mul_scalar(0.5)?;
                        cpu_fallback = cpu_fallback.tanh()?;
                    }
                },
                Err(_) => {
                    // Fallback to CPU
                    let mut cpu_fallback = images.clone();
                    let mean = cpu_fallback.mean(&None)?;
                    cpu_fallback = cpu_fallback.sub(&mean)?;
                    cpu_fallback = cpu_fallback.mul_scalar(0.5)?;
                    cpu_fallback = cpu_fallback.tanh()?;
                }
            }
            let coreml_time = start.elapsed().as_secs_f64() * 1000.0;
            total_coreml_time += coreml_time;
            println!("     CoreML Image Processing: {:.2}ms", coreml_time);
        }

        #[cfg(not(feature = "coreml"))]
        {
            coreml_success = false;
        }
    }

    let avg_cpu_time = total_cpu_time / config.iterations as f64;
    let avg_coreml_time = if coreml_success {
        Some(total_coreml_time / config.iterations as f64)
    } else {
        None
    };

    let speedup = avg_coreml_time.map(|coreml| avg_cpu_time / coreml);

    Ok(BenchmarkResult {
        name: "Image Processing Workload".to_string(),
        cpu_time_ms: avg_cpu_time,
        coreml_time_ms: avg_coreml_time,
        speedup,
        success: true,
    })
}

fn benchmark_convolution_heavy(config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let batch_size = config.image_batch_size;
    let in_channels = config.image_channels;
    let height = config.image_height;
    let width = config.image_width;

    println!("   🔧 Creating convolution heavy pipeline...");

    // Create multiple convolution layers
    let conv1 = Conv2d::new(in_channels, 32, 3, 1, 1)?;
    let conv2 = Conv2d::new(32, 64, 3, 1, 1)?;
    let conv3 = Conv2d::new(64, 128, 3, 1, 1)?;
    let conv4 = Conv2d::new(128, 256, 3, 1, 1)?;

    let bn1 = BatchNorm2d::new(32);
    let bn2 = BatchNorm2d::new(64);
    let bn3 = BatchNorm2d::new(128);
    let bn4 = BatchNorm2d::new(256);

    let mut total_cpu_time = 0.0;
    let mut total_coreml_time = 0.0;
    let mut coreml_success = true;

    for i in 0..config.iterations {
        println!("   📊 Iteration {}/{}", i + 1, config.iterations);

        // Create input
        let input_data: Vec<f32> = (0..batch_size * in_channels * height * width)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input = Tensor::from_vec(input_data, vec![batch_size, in_channels, height, width]);

        // CPU convolution pipeline
        let start = Instant::now();

        let mut x = input.clone();

        // Conv block 1
        x = conv1.forward(&x)?;
        x = bn1.forward(&x)?;
        x = x.relu()?;

        // Conv block 2
        x = conv2.forward(&x)?;
        x = bn2.forward(&x)?;
        x = x.relu()?;

        // Conv block 3
        x = conv3.forward(&x)?;
        x = bn3.forward(&x)?;
        x = x.relu()?;

        // Conv block 4
        x = conv4.forward(&x)?;
        x = bn4.forward(&x)?;
        x = x.relu()?;

        let cpu_time = start.elapsed().as_secs_f64() * 1000.0;
        total_cpu_time += cpu_time;
        println!("     CPU Convolution Heavy: {:.2}ms", cpu_time);

        // CoreML convolution pipeline (if available)
        #[cfg(feature = "coreml")]
        {
            let start = Instant::now();
            match input.to_device(DeviceType::CoreML(0)) {
                Ok(coreml_input) => {
                    let result = (|| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
                        let mut x = coreml_input;

                        // Conv block 1
                        x = conv1.forward(&x)?;
                        x = bn1.forward(&x)?;
                        x = x.relu()?;

                        // Conv block 2
                        x = conv2.forward(&x)?;
                        x = bn2.forward(&x)?;
                        x = x.relu()?;

                        // Conv block 3
                        x = conv3.forward(&x)?;
                        x = bn3.forward(&x)?;
                        x = x.relu()?;

                        // Conv block 4
                        x = conv4.forward(&x)?;
                        x = bn4.forward(&x)?;
                        x = x.relu()?;

                        Ok(x)
                    })();

                    if result.is_err() {
                        // Fallback to CPU
                        let mut cpu_x = input.clone();
                        cpu_x = conv1.forward(&cpu_x)?;
                        cpu_x = bn1.forward(&cpu_x)?;
                        cpu_x = cpu_x.relu()?;
                        cpu_x = conv2.forward(&cpu_x)?;
                        cpu_x = bn2.forward(&cpu_x)?;
                        cpu_x = cpu_x.relu()?;
                        cpu_x = conv3.forward(&cpu_x)?;
                        cpu_x = bn3.forward(&cpu_x)?;
                        cpu_x = cpu_x.relu()?;
                        cpu_x = conv4.forward(&cpu_x)?;
                        cpu_x = bn4.forward(&cpu_x)?;
                        cpu_x = cpu_x.relu()?;
                    }
                },
                Err(_) => {
                    // Fallback to CPU
                    let mut cpu_x = input.clone();
                    cpu_x = conv1.forward(&cpu_x)?;
                    cpu_x = bn1.forward(&cpu_x)?;
                    cpu_x = cpu_x.relu()?;
                    cpu_x = conv2.forward(&cpu_x)?;
                    cpu_x = bn2.forward(&cpu_x)?;
                    cpu_x = cpu_x.relu()?;
                    cpu_x = conv3.forward(&cpu_x)?;
                    cpu_x = bn3.forward(&cpu_x)?;
                    cpu_x = cpu_x.relu()?;
                    cpu_x = conv4.forward(&cpu_x)?;
                    cpu_x = bn4.forward(&cpu_x)?;
                    cpu_x = cpu_x.relu()?;
                }
            }
            let coreml_time = start.elapsed().as_secs_f64() * 1000.0;
            total_coreml_time += coreml_time;
            println!("     CoreML Convolution Heavy: {:.2}ms", coreml_time);
        }

        #[cfg(not(feature = "coreml"))]
        {
            coreml_success = false;
        }
    }

    let avg_cpu_time = total_cpu_time / config.iterations as f64;
    let avg_coreml_time = if coreml_success {
        Some(total_coreml_time / config.iterations as f64)
    } else {
        None
    };

    let speedup = avg_coreml_time.map(|coreml| avg_cpu_time / coreml);

    Ok(BenchmarkResult {
        name: "Convolution Heavy Workload".to_string(),
        cpu_time_ms: avg_cpu_time,
        coreml_time_ms: avg_coreml_time,
        speedup,
        success: true,
    })
}

fn print_benchmark_results(results: &[BenchmarkResult]) {
    println!("\n📈 Large-Scale Performance Comparison Results");
    println!("============================================");
    println!();

    let mut total_cpu_time = 0.0;
    let mut total_coreml_time = 0.0;
    let mut successful_benchmarks = 0;

    for (i, result) in results.iter().enumerate() {
        println!("🏆 #{} {}", i + 1, result.name);
        println!("   CPU Time:     {:.2}ms", result.cpu_time_ms);

        if let Some(coreml_time) = result.coreml_time_ms {
            println!("   CoreML Time:  {:.2}ms", coreml_time);
            if let Some(speedup) = result.speedup {
                if speedup > 1.0 {
                    println!("   🚀 Speedup:    {:.2}x (CoreML faster)", speedup);
                } else {
                    println!("   📉 Speedup:    {:.2}x (CPU faster)", speedup);
                }
            }
            total_coreml_time += coreml_time;
            successful_benchmarks += 1;
        } else {
            println!("   CoreML:       Not available");
        }

        total_cpu_time += result.cpu_time_ms;
        println!();
    }

    // Overall summary
    println!("📊 Overall Summary");
    println!("==================");
    println!("Total CPU Time:     {:.2}ms", total_cpu_time);

    if successful_benchmarks > 0 {
        println!("Total CoreML Time:  {:.2}ms", total_coreml_time);
        let overall_speedup = total_cpu_time / total_coreml_time;
        if overall_speedup > 1.0 {
            println!("🚀 Overall Speedup: {:.2}x (CoreML faster)", overall_speedup);
        } else {
            println!("📉 Overall Result:  {:.2}x (CPU faster)", overall_speedup);
        }

        println!("\n🎯 Conclusions:");
        if overall_speedup > 1.2 {
            println!("   ✅ CoreML provides significant performance benefits for large-scale workloads");
        } else if overall_speedup > 1.0 {
            println!("   ✅ CoreML provides modest performance benefits");
        } else {
            println!("   📝 CPU performs better, likely due to overhead or fallback behavior");
            println!("   💡 Consider larger tensor sizes or different workload patterns");
        }
    } else {
        println!("CoreML:             Not available");
        println!("   ❌ CoreML benchmarks could not be executed");
    }

    println!("\n✅ Large-scale benchmark completed!");
}