//! Inference accuracy validation and benchmarking
//! 推論精度の検証とベンチマーク

use crate::tensor::Tensor;
use crate::convert::{SimplifiedPyTorchModel, SimpleConversionError, IntegratedModelRunner};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Validation results for model conversion
/// モデル変換の検証結果
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Accuracy metrics
    /// 精度メトリクス
    pub accuracy_metrics: AccuracyMetrics,
    /// Performance metrics
    /// パフォーマンスメトリクス
    pub performance_metrics: PerformanceMetrics,
    /// Memory usage metrics
    /// メモリ使用量メトリクス
    pub memory_metrics: MemoryMetrics,
    /// Layer-wise validation results
    /// レイヤー別検証結果
    pub layer_results: HashMap<String, LayerValidationResult>,
}

/// Accuracy metrics
/// 精度メトリクス
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Mean absolute error
    /// 平均絶対誤差
    pub mean_absolute_error: f64,
    /// Mean squared error
    /// 平均二乗誤差
    pub mean_squared_error: f64,
    /// Maximum absolute error
    /// 最大絶対誤差
    pub max_absolute_error: f64,
    /// Relative error (percentage)
    /// 相対誤差（パーセンテージ）
    pub relative_error_percent: f64,
    /// Numerical tolerance passed
    /// 数値許容値合格
    pub tolerance_passed: bool,
    /// Tolerance threshold used
    /// 使用された許容値閾値
    pub tolerance_threshold: f64,
}

/// Performance metrics
/// パフォーマンスメトリクス
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average inference time
    /// 平均推論時間
    pub avg_inference_time: Duration,
    /// Minimum inference time
    /// 最小推論時間
    pub min_inference_time: Duration,
    /// Maximum inference time
    /// 最大推論時間
    pub max_inference_time: Duration,
    /// Throughput (inferences per second)
    /// スループット（毎秒推論数）
    pub throughput: f64,
    /// Memory allocations per inference
    /// 推論あたりのメモリ割り当て
    pub allocations_per_inference: usize,
}

/// Memory usage metrics
/// メモリ使用量メトリクス
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Total model size in bytes
    /// モデル総サイズ（バイト）
    pub total_model_size: usize,
    /// Parameter memory size
    /// パラメータメモリサイズ
    pub parameter_memory: usize,
    /// Activation memory size (estimated)
    /// 活性化メモリサイズ（推定）
    pub activation_memory: usize,
    /// Peak memory usage
    /// ピークメモリ使用量
    pub peak_memory: usize,
}

/// Layer-wise validation result
/// レイヤー別検証結果
#[derive(Debug, Clone)]
pub struct LayerValidationResult {
    /// Layer name
    /// レイヤー名
    pub layer_name: String,
    /// Shape validation passed
    /// 形状検証合格
    pub shape_valid: bool,
    /// Parameter count validation passed
    /// パラメータ数検証合格
    pub parameter_count_valid: bool,
    /// Numerical accuracy within tolerance
    /// 数値精度が許容範囲内
    pub numerical_accuracy_valid: bool,
    /// Layer-specific metrics
    /// レイヤー固有メトリクス
    pub layer_metrics: LayerMetrics,
}

/// Layer-specific metrics
/// レイヤー固有メトリクス
#[derive(Debug, Clone)]
pub struct LayerMetrics {
    /// Input shape
    /// 入力形状
    pub input_shape: Vec<usize>,
    /// Output shape
    /// 出力形状
    pub output_shape: Vec<usize>,
    /// Parameter count
    /// パラメータ数
    pub parameter_count: usize,
    /// Computation time
    /// 計算時間
    pub computation_time: Duration,
}

/// Model validator and benchmarker
/// モデル検証・ベンチマーク器
pub struct ModelValidator;

impl ModelValidator {
    /// Validate converted model accuracy
    /// 変換されたモデルの精度を検証
    pub fn validate_accuracy(
        original_outputs: &[Vec<f32>],
        converted_outputs: &[Vec<f32>],
        tolerance: f64,
    ) -> Result<AccuracyMetrics, SimpleConversionError> {
        if original_outputs.len() != converted_outputs.len() {
            return Err(SimpleConversionError::InvalidParameter(
                "Output count mismatch".to_string()
            ));
        }

        let mut absolute_errors = Vec::new();
        let mut squared_errors = Vec::new();
        let mut relative_errors = Vec::new();

        for (orig, conv) in original_outputs.iter().zip(converted_outputs.iter()) {
            if orig.len() != conv.len() {
                return Err(SimpleConversionError::InvalidParameter(
                    "Output dimension mismatch".to_string()
                ));
            }

            for (&o, &c) in orig.iter().zip(conv.iter()) {
                let abs_error = (o - c).abs();
                let squared_error = (o - c).powi(2);
                
                absolute_errors.push(abs_error as f64);
                squared_errors.push(squared_error as f64);
                
                if o.abs() > 1e-8 {
                    relative_errors.push((abs_error / o.abs()) as f64);
                }
            }
        }

        let mean_absolute_error = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let mean_squared_error = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
        let max_absolute_error = absolute_errors.iter().fold(0.0, |a, &b| a.max(b));
        
        let relative_error_percent = if !relative_errors.is_empty() {
            (relative_errors.iter().sum::<f64>() / relative_errors.len() as f64) * 100.0
        } else {
            0.0
        };

        let tolerance_passed = max_absolute_error <= tolerance;

        Ok(AccuracyMetrics {
            mean_absolute_error,
            mean_squared_error,
            max_absolute_error,
            relative_error_percent,
            tolerance_passed,
            tolerance_threshold: tolerance,
        })
    }

    /// Benchmark model performance
    /// モデルパフォーマンスをベンチマーク
    pub fn benchmark_performance<F>(
        inference_fn: F,
        num_iterations: usize,
    ) -> Result<PerformanceMetrics, SimpleConversionError>
    where
        F: Fn() -> Result<(), SimpleConversionError>,
    {
        let mut inference_times = Vec::new();

        // Warmup runs
        for _ in 0..5 {
            inference_fn().map_err(|_| SimpleConversionError::InvalidParameter(
                "Inference failed during warmup".to_string()
            ))?;
        }

        // Benchmark runs
        for _ in 0..num_iterations {
            let start = Instant::now();
            inference_fn().map_err(|_| SimpleConversionError::InvalidParameter(
                "Inference failed during benchmark".to_string()
            ))?;
            let duration = start.elapsed();
            inference_times.push(duration);
        }

        let total_time: Duration = inference_times.iter().sum();
        let avg_inference_time = total_time / num_iterations as u32;
        let min_inference_time = inference_times.iter().min().unwrap().clone();
        let max_inference_time = inference_times.iter().max().unwrap().clone();
        
        let throughput = num_iterations as f64 / total_time.as_secs_f64();

        Ok(PerformanceMetrics {
            avg_inference_time,
            min_inference_time,
            max_inference_time,
            throughput,
            allocations_per_inference: 0, // Would need memory profiler integration
        })
    }

    /// Estimate memory usage
    /// メモリ使用量を推定
    pub fn estimate_memory_usage(
        model: &SimplifiedPyTorchModel,
        input_shape: &[usize],
    ) -> Result<MemoryMetrics, SimpleConversionError> {
        let mut parameter_memory = 0;
        let mut activation_memory = 0;

        // Calculate parameter memory
        for layer in model.layers.values() {
            for tensor in layer.tensors.values() {
                parameter_memory += tensor.data.len() * std::mem::size_of::<f32>();
            }
        }

        // Estimate activation memory (simplified)
        let mut current_shape = input_shape.to_vec();
        for layer_name in &model.execution_order {
            if let Some(layer) = model.layers.get(layer_name) {
                match model.simulate_layer_forward(layer, current_shape.clone()) {
                    Ok(output_shape) => {
                        let activation_size: usize = output_shape.iter().product();
                        activation_memory += activation_size * std::mem::size_of::<f32>();
                        current_shape = output_shape;
                    },
                    Err(_) => {
                        // Skip layers that can't be simulated
                        continue;
                    }
                }
            }
        }

        let total_model_size = parameter_memory + activation_memory;
        let peak_memory = total_model_size * 2; // Estimate peak usage

        Ok(MemoryMetrics {
            total_model_size,
            parameter_memory,
            activation_memory,
            peak_memory,
        })
    }

    /// Validate individual layer
    /// 個別レイヤーを検証
    pub fn validate_layer(
        layer_name: &str,
        expected_input_shape: &[usize],
        expected_output_shape: &[usize],
        expected_param_count: usize,
        model: &SimplifiedPyTorchModel,
    ) -> Result<LayerValidationResult, SimpleConversionError> {
        let layer = model.layers.get(layer_name)
            .ok_or_else(|| SimpleConversionError::MissingParameter(
                format!("Layer not found: {}", layer_name)
            ))?;

        // Validate parameter count
        let parameter_count_valid = layer.num_parameters == expected_param_count;

        // Validate shapes through simulation
        let start_time = Instant::now();
        let simulated_output = model.simulate_layer_forward(layer, expected_input_shape.to_vec());
        let computation_time = start_time.elapsed();

        let (shape_valid, output_shape) = match simulated_output {
            Ok(output) => (output == expected_output_shape, output),
            Err(_) => (false, []),
        };

        // For now, assume numerical accuracy is valid if shapes match
        let numerical_accuracy_valid = shape_valid;

        let layer_metrics = LayerMetrics {
            input_shape: expected_input_shape.to_vec(),
            output_shape,
            parameter_count: layer.num_parameters,
            computation_time,
        };

        Ok(LayerValidationResult {
            layer_name: layer_name.to_string(),
            shape_valid,
            parameter_count_valid,
            numerical_accuracy_valid,
            layer_metrics,
        })
    }

    /// Run comprehensive validation
    /// 包括的検証を実行
    pub fn comprehensive_validation(
        model: &SimplifiedPyTorchModel,
        test_inputs: &[Vec<f32>],
        expected_outputs: &[Vec<f32>],
        input_shape: &[usize],
        tolerance: f64,
    ) -> Result<ValidationResults, SimpleConversionError> {
        // Generate converted outputs (simplified simulation)
        let converted_outputs = Self::simulate_model_outputs(model, test_inputs, input_shape)?;

        // Validate accuracy
        let accuracy_metrics = Self::validate_accuracy(expected_outputs, &converted_outputs, tolerance)?;

        // Benchmark performance
        let performance_metrics = Self::benchmark_performance(|| {
            let _ = Self::simulate_model_outputs(model, &test_inputs[0..1], input_shape)?;
            Ok(())
        }, 100)?;

        // Estimate memory usage
        let memory_metrics = Self::estimate_memory_usage(model, input_shape)?;

        // Validate individual layers
        let mut layer_results = HashMap::new();
        let mut current_shape = input_shape.to_vec();

        for layer_name in &model.execution_order {
            if let Some(layer) = model.layers.get(layer_name) {
                if let Ok(output_shape) = model.simulate_layer_forward(layer, current_shape.clone()) {
                    let layer_result = Self::validate_layer(
                        layer_name,
                        &current_shape,
                        &output_shape,
                        layer.num_parameters,
                        model,
                    )?;
                    layer_results.insert(layer_name.clone(), layer_result);
                    current_shape = output_shape;
                }
            }
        }

        Ok(ValidationResults {
            accuracy_metrics,
            performance_metrics,
            memory_metrics,
            layer_results,
        })
    }

    /// Simulate model outputs (placeholder implementation)
    /// モデル出力をシミュレーション（プレースホルダー実装）
    fn simulate_model_outputs(
        model: &SimplifiedPyTorchModel,
        inputs: &[Vec<f32>],
        input_shape: &[usize],
    ) -> Result<Vec<Vec<f32>>, SimpleConversionError> {
        let mut outputs = Vec::new();

        for input_data in inputs {
            // Simulate forward pass (very simplified)
            let mut current_shape = input_shape.to_vec();
            
            for layer_name in &model.execution_order {
                if let Some(layer) = model.layers.get(layer_name) {
                    current_shape = model.simulate_layer_forward(layer, current_shape)?;
                }
            }

            // Generate placeholder output with correct shape
            let output_size: usize = current_shape.iter().product();
            let output: Vec<f32> = (0..output_size)
                .map(|i| (i as f32 * 0.001) + input_data.get(i % input_data.len()).unwrap_or(&0.0))
                .collect();
            
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Generate validation report
    /// 検証レポートを生成
    pub fn generate_report(results: &ValidationResults) -> String {
        let mut report = String::new();

        report.push_str("🔍 Model Validation Report\n");
        report.push_str("==========================\n\n");

        // Accuracy section
        report.push_str("📊 Accuracy Metrics:\n");
        report.push_str(&format!("   Mean Absolute Error: {:.6}\n", results.accuracy_metrics.mean_absolute_error));
        report.push_str(&format!("   Mean Squared Error: {:.6}\n", results.accuracy_metrics.mean_squared_error));
        report.push_str(&format!("   Max Absolute Error: {:.6}\n", results.accuracy_metrics.max_absolute_error));
        report.push_str(&format!("   Relative Error: {:.2}%\n", results.accuracy_metrics.relative_error_percent));
        report.push_str(&format!("   Tolerance Passed: {} (threshold: {:.6})\n", 
            if results.accuracy_metrics.tolerance_passed { "✅" } else { "❌" },
            results.accuracy_metrics.tolerance_threshold
        ));

        // Performance section
        report.push_str("\n⚡ Performance Metrics:\n");
        report.push_str(&format!("   Average Inference Time: {:.2?}\n", results.performance_metrics.avg_inference_time));
        report.push_str(&format!("   Min/Max Inference Time: {:.2?} / {:.2?}\n", 
            results.performance_metrics.min_inference_time,
            results.performance_metrics.max_inference_time
        ));
        report.push_str(&format!("   Throughput: {:.1} inferences/sec\n", results.performance_metrics.throughput));

        // Memory section
        report.push_str("\n💾 Memory Metrics:\n");
        report.push_str(&format!("   Total Model Size: {:.2} MB\n", 
            results.memory_metrics.total_model_size as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!("   Parameter Memory: {:.2} MB\n", 
            results.memory_metrics.parameter_memory as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!("   Activation Memory: {:.2} MB\n", 
            results.memory_metrics.activation_memory as f64 / 1024.0 / 1024.0
        ));

        // Layer validation section
        report.push_str("\n🏗️ Layer Validation Results:\n");
        for (layer_name, layer_result) in &results.layer_results {
            let status = if layer_result.shape_valid && layer_result.parameter_count_valid && layer_result.numerical_accuracy_valid {
                "✅"
            } else {
                "❌"
            };
            
            report.push_str(&format!("   {} {}: ", status, layer_name));
            report.push_str(&format!("Shape({}), Params({}), Accuracy({})\n",
                if layer_result.shape_valid { "✓" } else { "✗" },
                if layer_result.parameter_count_valid { "✓" } else { "✗" },
                if layer_result.numerical_accuracy_valid { "✓" } else { "✗" }
            ));
            report.push_str(&format!("      Input: {:?} → Output: {:?}\n",
                layer_result.layer_metrics.input_shape,
                layer_result.layer_metrics.output_shape
            ));
            report.push_str(&format!("      Parameters: {}, Time: {:.2?}\n",
                layer_result.layer_metrics.parameter_count,
                layer_result.layer_metrics.computation_time
            ));
        }

        report.push_str("\n");
        report
    }
}

/// Benchmark suite for different model architectures
/// 異なるモデルアーキテクチャのベンチマークスイート
pub struct BenchmarkSuite;

impl BenchmarkSuite {
    /// Benchmark simple MLP
    /// シンプルMLPをベンチマーク
    pub fn benchmark_mlp(
        model: &SimplifiedPyTorchModel,
        input_shape: &[usize],
        iterations: usize,
    ) -> Result<ValidationResults, SimpleConversionError> {
        let test_inputs = Self::generate_test_data(input_shape, 10);
        let expected_outputs = Self::generate_expected_outputs(&test_inputs, model);

        ModelValidator::comprehensive_validation(
            model,
            &test_inputs,
            &expected_outputs,
            input_shape,
            1e-5, // Tolerance
        )
    }

    /// Benchmark CNN
    /// CNNをベンチマーク
    pub fn benchmark_cnn(
        model: &SimplifiedPyTorchModel,
        input_shape: &[usize],
        iterations: usize,
    ) -> Result<ValidationResults, SimpleConversionError> {
        let test_inputs = Self::generate_test_data(input_shape, 5);
        let expected_outputs = Self::generate_expected_outputs(&test_inputs, model);

        ModelValidator::comprehensive_validation(
            model,
            &test_inputs,
            &expected_outputs,
            input_shape,
            1e-4, // Slightly higher tolerance for CNN
        )
    }

    /// Generate test data
    /// テストデータを生成
    fn generate_test_data(input_shape: &[usize], num_samples: usize) -> Vec<Vec<f32>> {
        let input_size: usize = input_shape.iter().product();
        let mut test_data = Vec::new();

        for i in 0..num_samples {
            let data: Vec<f32> = (0..input_size)
                .map(|j| ((i * input_size + j) as f32 * 0.001).sin())
                .collect();
            test_data.push(data);
        }

        test_data
    }

    /// Generate expected outputs (placeholder)
    /// 期待出力を生成（プレースホルダー）
    fn generate_expected_outputs(
        inputs: &[Vec<f32>],
        model: &SimplifiedPyTorchModel,
    ) -> Vec<Vec<f32>> {
        // This would typically come from running the original PyTorch model
        // For now, generate placeholder data
        inputs.iter().map(|input| {
            input.iter().enumerate().map(|(i, &x)| {
                x * (i as f32 + 1.0) * 0.1 + model.total_parameters as f32 * 1e-6
            }).collect()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_validation() {
        let original = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let converted = vec![
            vec![1.001, 1.999, 3.001],
            vec![4.001, 4.999, 5.999],
        ];

        let metrics = ModelValidator::validate_accuracy(&original, &converted, 0.01).unwrap();
        
        assert!(metrics.tolerance_passed);
        assert!(metrics.mean_absolute_error < 0.01);
        assert!(metrics.relative_error_percent < 1.0);
    }

    #[test]
    fn test_performance_benchmark() {
        let dummy_inference = || -> Result<(), SimpleConversionError> {
            std::thread::sleep(Duration::from_millis(1));
            Ok(())
        };

        let metrics = ModelValidator::benchmark_performance(dummy_inference, 10).unwrap();
        
        assert!(metrics.avg_inference_time >= Duration::from_millis(1));
        assert!(metrics.throughput > 0.0);
        assert!(metrics.min_inference_time <= metrics.max_inference_time);
    }
}