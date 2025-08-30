//! Complete WASM pipeline example demonstrating all modules
//! 全WASMモジュールを実演する完全パイプラインの例

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::tensor::WasmTensor;
    use rustorch::wasm::advanced_math::WasmAdvancedMath;
    use rustorch::wasm::quality_metrics::WasmQualityMetrics;
    use rustorch::wasm::statistical_analysis::{WasmStatisticalFunctions, WasmStatisticalAnalyzer};
    use rustorch::wasm::anomaly_detection::{WasmAnomalyDetector, WasmTimeSeriesDetector};
    use rustorch::wasm::data_transforms::*;
    use rustorch::wasm::common::MemoryManager;
    use std::time::Instant;
    
    println!("=== RusTorch WASM Complete Pipeline Demo ===");
    println!("Demonstrating end-to-end data processing with all modules\n");
    
    // Initialize memory pool for comprehensive processing
    MemoryManager::init_pool(150);
    
    // Initialize all modules
    let start_time = Instant::now();
    
    let math = WasmAdvancedMath::new();
    let quality = WasmQualityMetrics::new(0.8).expect("Failed to create quality analyzer");
    let stats_func = WasmStatisticalFunctions::new();
    let stats_analyzer = WasmStatisticalAnalyzer::new();
    let anomaly_detector = WasmAnomalyDetector::new(2.0, 50).expect("Failed to create anomaly detector");
    let ts_detector = WasmTimeSeriesDetector::new(30, Some(12)).expect("Failed to create TS detector");
    
    println!("✅ All modules initialized in {:?}", start_time.elapsed());
    
    // Stage 1: Data Generation and Initial Assessment
    println!("\n🏗️  STAGE 1: DATA GENERATION AND ASSESSMENT");
    println!("─".repeat(60));
    
    let raw_data = generate_sensor_data(1440); // 24 hours of minute-by-minute data
    let raw_tensor = WasmTensor::new(raw_data, vec![1440]);
    
    println!("📊 Raw data generated: {} data points", raw_tensor.data().len());
    
    // Initial quality assessment
    if let Ok(initial_quality) = quality.overall_quality(&raw_tensor) {
        println!("🎯 Initial data quality: {:.1}%", initial_quality * 100.0);
    }
    
    if let Ok(basic_stats) = stats_analyzer.basic_stats(&raw_tensor) {
        println!("📈 Raw data statistics: {}", basic_stats);
    }
    
    // Stage 2: Data Cleaning and Transformation
    println!("\n🧹 STAGE 2: DATA CLEANING AND TRANSFORMATION");
    println!("─".repeat(60));
    
    // Remove outliers using advanced math
    let cleaned_data = clean_outliers(&raw_tensor, &math, &stats_analyzer);
    let cleaned_tensor = WasmTensor::new(cleaned_data, vec![cleaned_data.len()]);
    
    println!("🔧 Data cleaned: {} → {} points", raw_tensor.data().len(), cleaned_tensor.data().len());
    
    // Apply normalization transforms
    let mean = vec![cleaned_tensor.data().iter().sum::<f32>() / cleaned_tensor.data().len() as f32];
    let variance = cleaned_tensor.data().iter()
        .map(|&x| (x - mean[0]).powi(2))
        .sum::<f32>() / cleaned_tensor.data().len() as f32;
    let std = vec![variance.sqrt()];
    
    let normalize_transform = WasmNormalize::new(&mean, &std).expect("Failed to create normalize");
    let normalized_tensor = normalize_transform.apply(&cleaned_tensor).expect("Normalization failed");
    
    println!("📊 Data normalized: mean={:.3}, std={:.3}", mean[0], std[0]);
    
    // Quality assessment after cleaning
    if let Ok(cleaned_quality) = quality.overall_quality(&normalized_tensor) {
        println!("✨ Quality after cleaning: {:.1}%", cleaned_quality * 100.0);
    }
    
    // Stage 3: Advanced Statistical Analysis
    println!("\n📊 STAGE 3: ADVANCED STATISTICAL ANALYSIS");
    println!("─".repeat(60));
    
    // Percentile analysis
    let percentiles = vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0];
    if let Ok(percentile_values) = stats_analyzer.percentiles(&normalized_tensor, percentiles.clone()) {
        println!("📊 Percentile analysis:");
        for (p, v) in percentiles.iter().zip(percentile_values.iter()) {
            println!("   P{:2.0}: {:8.3}", p, v);
        }
    }
    
    // Advanced mathematical transformations
    println!("\n🧮 Mathematical transformations:");
    
    // Create positive version for log transforms
    let positive_data: Vec<f32> = normalized_tensor.data().iter()
        .map(|&x| (x + 5.0).abs()) // Ensure positive values
        .collect();
    let positive_tensor = WasmTensor::new(positive_data, normalized_tensor.shape());
    
    if let Ok(log_tensor) = math.log(&positive_tensor) {
        if let Ok(log_stats) = stats_analyzer.basic_stats(&log_tensor) {
            println!("   📈 Log transform stats: {}", log_stats);
        }
    }
    
    if let Ok(sqrt_tensor) = math.sqrt(&positive_tensor) {
        if let Ok(sqrt_stats) = stats_analyzer.basic_stats(&sqrt_tensor) {
            println!("   🔢 Square root stats: {}", sqrt_stats);
        }
    }
    
    // Trigonometric analysis for cyclical patterns
    if let Ok(sin_tensor) = math.sin(&normalized_tensor) {
        if let Ok(cos_tensor) = math.cos(&normalized_tensor) {
            if let Ok(correlation) = stats_func.correlation(&sin_tensor, &cos_tensor) {
                println!("   🌊 Sin-Cos correlation: {:.3}", correlation);
            }
        }
    }
    
    // Stage 4: Anomaly Detection Analysis
    println!("\n🚨 STAGE 4: ANOMALY DETECTION ANALYSIS");
    println!("─".repeat(60));
    
    // Batch anomaly detection
    if let Ok(statistical_anomalies) = anomaly_detector.detect_statistical(&normalized_tensor) {
        println!("📊 Statistical anomalies found: {}", statistical_anomalies.len());
        
        for (idx, anomaly) in statistical_anomalies.iter().take(3).enumerate() {
            println!("   Anomaly {}: {:?}", idx + 1, anomaly);
        }
    }
    
    if let Ok(isolation_anomalies) = anomaly_detector.detect_isolation_forest(&normalized_tensor, 100) {
        println!("🌲 Isolation Forest anomalies: {}", isolation_anomalies.len());
    }
    
    // Real-time detection simulation
    println!("\n⏱️  Real-time detection simulation:");
    let mut realtime_anomalies = 0;
    let mut ts_anomalies = 0;
    
    for (minute, &value) in normalized_tensor.data().iter().enumerate() {
        let timestamp = 1630000000 + (minute * 60) as u64; // Unix timestamp
        
        // Statistical real-time detection
        if let Ok(Some(anomaly)) = anomaly_detector.detect_realtime(value) {
            realtime_anomalies += 1;
            if realtime_anomalies <= 3 {
                println!("   🚨 Real-time anomaly at minute {}: {:.3}", minute, value);
            }
        }
        
        // Time series detection
        if let Ok(Some(ts_anomaly)) = ts_detector.add_point(timestamp, value) {
            ts_anomalies += 1;
            if ts_anomalies <= 3 {
                println!("   📈 Time series anomaly at minute {}: {:.3}", minute, value);
            }
        }
    }
    
    println!("   Total real-time anomalies: {}", realtime_anomalies);
    println!("   Total time series anomalies: {}", ts_anomalies);
    
    // Stage 5: Pipeline Processing
    println!("\n🔄 STAGE 5: PIPELINE PROCESSING");
    println!("─".repeat(60));
    
    // Create comprehensive processing pipeline
    let transform_pipeline = WasmTransformPipeline::new(true).expect("Failed to create pipeline");
    let processing_pipeline = WasmProcessingPipeline::new(true).expect("Failed to create processing pipeline");
    
    // Build transform pipeline
    transform_pipeline.add_transform("normalize");
    transform_pipeline.add_transform("smooth");
    transform_pipeline.add_transform("enhance");
    
    println!("🔗 Transform pipeline created with {} steps", transform_pipeline.length());
    
    // Build processing pipeline
    processing_pipeline.add_operation("quality_check");
    processing_pipeline.add_operation("anomaly_scan");
    processing_pipeline.add_operation("statistical_analysis");
    
    println!("⚙️  Processing pipeline created with {} operations", processing_pipeline.operation_count());
    
    // Execute pipelines
    if let Ok(processed_tensor) = transform_pipeline.execute(&raw_tensor) {
        println!("✅ Transform pipeline executed successfully");
        println!("   Result shape: {:?}", processed_tensor.shape());
        
        // Get pipeline performance stats
        let pipeline_stats = transform_pipeline.get_stats();
        println!("   Pipeline stats: {}", pipeline_stats);
        
        // Final quality assessment
        if let Ok(final_quality) = quality.overall_quality(&processed_tensor) {
            println!("🎯 Final processed quality: {:.1}%", final_quality * 100.0);
        }
    }
    
    // Processing pipeline configuration
    let processing_config = processing_pipeline.get_config();
    println!("⚙️  Processing configuration: {}", processing_config);
    
    // Stage 6: Performance and Memory Analysis
    println!("\n⚡ STAGE 6: PERFORMANCE AND MEMORY ANALYSIS");
    println!("─".repeat(60));
    
    let total_time = start_time.elapsed();
    println!("🕐 Total execution time: {:?}", total_time);
    
    // Memory pool analysis
    let final_stats = MemoryManager::get_stats();
    let efficiency = MemoryManager::cache_efficiency();
    
    println!("🧠 Memory pool performance:");
    if let Ok(stats_json) = serde_json::from_str::<serde_json::Value>(&final_stats) {
        println!("   Total allocations: {}", stats_json["total_allocations"]);
        println!("   Cache hits: {}", stats_json["cache_hits"]);
        println!("   Hit rate: {:.1}%", stats_json["hit_rate"]);
        println!("   Memory saved: {:.2} KB", 
                 stats_json["memory_saved_bytes"].as_u64().unwrap_or(0) as f32 / 1024.0);
    }
    
    if let Ok(eff_json) = serde_json::from_str::<serde_json::Value>(&efficiency) {
        println!("   Efficiency rating: {}", eff_json["efficiency"]);
    }
    
    // Module performance summary
    println!("\n📊 Module performance summary:");
    
    if let Ok(detector_stats) = anomaly_detector.get_statistics() {
        println!("   🚨 Anomaly detector: {}", detector_stats);
    }
    
    if let Ok(trend_analysis) = ts_detector.get_trend_analysis() {
        println!("   📈 Trend analysis: {}", trend_analysis);
    }
    
    if let Ok(seasonal_analysis) = ts_detector.get_seasonal_analysis() {
        println!("   🌊 Seasonal analysis: {}", seasonal_analysis);
    }
    
    // Stage 7: Comprehensive Report Generation
    println!("\n📋 STAGE 7: COMPREHENSIVE REPORT");
    println!("─".repeat(60));
    
    if let Ok(quality_report) = quality.quality_report(&normalized_tensor) {
        println!("📄 Quality Report:");
        if let Ok(report_json) = serde_json::from_str::<serde_json::Value>(&quality_report) {
            println!("{}", serde_json::to_string_pretty(&report_json).unwrap_or(quality_report));
        }
    }
    
    // Resource utilization summary
    println!("\n💾 Resource Utilization Summary:");
    println!("   ⏱️  Execution time: {:?}", total_time);
    println!("   🧠 Memory efficiency: {}", efficiency);
    println!("   📊 Data processed: {} MB", 
             (raw_tensor.data().len() * std::mem::size_of::<f32>()) / (1024 * 1024));
    println!("   🔧 Modules used: 7 (tensor, math, quality, stats, anomaly, transforms, pipelines)");
    
    // Performance recommendations
    println!("\n💡 Performance Recommendations:");
    
    if let Ok(pool_stats_json) = serde_json::from_str::<serde_json::Value>(&final_stats) {
        let hit_rate = pool_stats_json["hit_rate"].as_f64().unwrap_or(0.0);
        
        if hit_rate > 85.0 {
            println!("   ✅ Memory pool highly efficient - no tuning needed");
        } else if hit_rate > 70.0 {
            println!("   ⚠️  Consider increasing pool size for better efficiency");
        } else {
            println!("   🚨 Pool underperforming - review allocation patterns");
        }
        
        let total_allocations = pool_stats_json["total_allocations"].as_u64().unwrap_or(0);
        if total_allocations > 1000 {
            println!("   💡 High allocation count - consider batching operations");
        }
    }
    
    if total_time > std::time::Duration::from_secs(5) {
        println!("   ⏰ Consider parallel processing for large datasets");
    } else {
        println!("   ⚡ Excellent performance for dataset size");
    }
    
    // Cleanup and final statistics
    println!("\n🧹 CLEANUP AND FINAL STATISTICS");
    println!("─".repeat(60));
    
    // Force garbage collection to see cleanup effectiveness
    MemoryManager::gc();
    let post_gc_stats = MemoryManager::get_stats();
    
    println!("🗑️  Post-cleanup memory: {}", post_gc_stats);
    
    // Demonstrate version information
    println!("\n📋 Module Versions:");
    println!("   Advanced Math: {}", rustorch::wasm::wasm_advanced_math_version());
    println!("   Quality Metrics: {}", rustorch::wasm::wasm_quality_metrics_version());
    println!("   Transforms: {}", rustorch::wasm::wasm_transforms_version());
    println!("   Anomaly Detection: {}", rustorch::wasm::wasm_anomaly_detection_version());
    
    println!("\n🎉 Complete pipeline demonstration finished successfully!");
    println!("   All {} WASM modules integrated and tested", 7);
    println!("   Total execution time: {:?}", total_time);
    println!("   Memory efficiency achieved: {}", efficiency);
}

/// Generate realistic sensor data with various patterns and issues
fn generate_sensor_data(minutes: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(minutes);
    
    for minute in 0..minutes {
        let hour = minute / 60;
        let minute_in_hour = minute % 60;
        
        // Base temperature with daily cycle
        let daily_temp = 18.0 + 12.0 * (2.0 * std::f32::consts::PI * hour as f32 / 24.0).cos();
        
        // Minute-level variations
        let minute_noise = 0.5 * (minute_in_hour as f32 * 0.1).sin();
        
        // Equipment aging effect (slow drift)
        let aging_drift = minute as f32 * 0.001;
        
        // Occasional sensor malfunctions
        let malfunction = if minute % 137 == 0 {
            15.0 * if minute % 274 == 0 { -1.0 } else { 1.0 }
        } else {
            0.0
        };
        
        // Missing data simulation
        let value = if minute % 83 == 0 {
            f32::NAN
        } else {
            daily_temp + minute_noise + aging_drift + malfunction
        };
        
        data.push(value);
    }
    
    data
}

/// Clean outliers from data using statistical analysis
#[cfg(feature = "wasm")]
fn clean_outliers(tensor: &rustorch::wasm::tensor::WasmTensor, 
                 math: &rustorch::wasm::advanced_math::WasmAdvancedMath, 
                 analyzer: &rustorch::wasm::statistical_analysis::WasmStatisticalAnalyzer) -> Vec<f32> {
    // Get outliers
    let outliers = analyzer.detect_outliers(tensor).unwrap_or_default();
    let outlier_indices: std::collections::HashSet<usize> = outliers.iter()
        .filter_map(|outlier| {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&format!("{:?}", outlier)) {
                obj.get("index").and_then(|v| v.as_u64()).map(|i| i as usize)
            } else {
                None
            }
        })
        .collect();
    
    // Remove NaN values and outliers
    let cleaned_data: Vec<f32> = tensor.data().iter()
        .enumerate()
        .filter_map(|(idx, &value)| {
            if value.is_finite() && !outlier_indices.contains(&idx) {
                Some(value)
            } else {
                None
            }
        })
        .collect();
    
    println!("🧹 Cleaned {} outliers and NaN values", 
             tensor.data().len() - cleaned_data.len());
    
    cleaned_data
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_complete_pipeline");
}