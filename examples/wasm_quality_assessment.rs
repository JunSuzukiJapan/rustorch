//! Data quality assessment and validation example
//! データ品質評価とバリデーションの例

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::tensor::WasmTensor;
    use rustorch::wasm::quality_metrics::WasmQualityMetrics;
    use rustorch::wasm::anomaly_detection::WasmAnomalyDetector;
    use rustorch::wasm::common::MemoryManager;
    
    println!("=== RusTorch WASM Data Quality Assessment ===");
    
    // Initialize memory pool
    MemoryManager::init_pool(75);
    
    // Create quality analyzer
    let quality_analyzer = WasmQualityMetrics::new(0.8)
        .expect("Failed to create quality analyzer");
    let statistical_detector = WasmAnomalyDetector::new(2.0, 50)
        .expect("Failed to create statistical detector");
    
    println!("Quality assessment tools initialized with threshold: 0.8");
    
    // Test datasets with different quality characteristics
    let test_datasets = vec![
        ("high_quality", create_high_quality_data()),
        ("medium_quality", create_medium_quality_data()), 
        ("low_quality", create_low_quality_data()),
        ("mixed_quality", create_mixed_quality_data()),
        ("real_world_simulation", create_realistic_data()),
    ];
    
    println!("\n--- Quality Assessment Results ---");
    
    for (dataset_name, (data, shape, description)) in test_datasets {
        println!("\n🔍 Dataset: {} ({})", dataset_name, description);
        
        let tensor = WasmTensor::new(data, shape);
        println!("   Shape: {:?}, Size: {} elements", tensor.shape(), tensor.data().len());
        
        // Individual quality metrics
        if let Ok(completeness) = quality_analyzer.completeness(&tensor) {
            println!("   📊 Completeness: {:.1}%", completeness * 100.0);
        }
        
        if let Ok(validity) = quality_analyzer.validity(&tensor) {
            println!("   ✅ Validity: {:.1}%", validity * 100.0);
        }
        
        if let Ok(consistency) = quality_analyzer.consistency(&tensor) {
            println!("   🔄 Consistency: {:.1}%", consistency * 100.0);
        }
        
        if let Ok(uniqueness) = quality_analyzer.uniqueness(&tensor) {
            println!("   🔢 Uniqueness: {:.1}%", uniqueness * 100.0);
        }
        
        // Overall quality score
        if let Ok(overall) = quality_analyzer.overall_quality(&tensor) {
            let grade = match overall {
                x if x >= 0.9 => "A+ (Excellent)",
                x if x >= 0.8 => "A (Very Good)", 
                x if x >= 0.7 => "B (Good)",
                x if x >= 0.6 => "C (Fair)",
                x if x >= 0.5 => "D (Poor)",
                _ => "F (Unacceptable)",
            };
            println!("   🎯 Overall Quality: {:.1}% ({})", overall * 100.0, grade);
        }
        
        // Detailed quality report
        if let Ok(report) = quality_analyzer.quality_report(&tensor) {
            if let Ok(parsed_report) = serde_json::from_str::<serde_json::Value>(&report) {
                println!("   📋 Report: {}", serde_json::to_string_pretty(&parsed_report).unwrap_or(report));
            }
        }
        
        // Basic statistics calculation
        let data_slice = tensor.data();
        let mean = data_slice.iter().sum::<f32>() / data_slice.len() as f32;
        let variance = data_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data_slice.len() as f32;
        let std_dev = variance.sqrt();
        let min_val = data_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("   📈 Basic Stats:");
        println!("      Mean: {:.3}, Std: {:.3}", mean, std_dev);
        println!("      Range: [{:.3}, {:.3}]", min_val, max_val);
        
        // Simple outlier detection using IQR method
        let mut sorted_data = data_slice.to_vec();
        sorted_data.retain(|x| x.is_finite());
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if sorted_data.len() >= 4 {
            let q1_idx = sorted_data.len() / 4;
            let q3_idx = (sorted_data.len() * 3) / 4;
            let q1 = sorted_data[q1_idx];
            let q3 = sorted_data[q3_idx];
            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;
            
            let outliers: Vec<&f32> = data_slice.iter()
                .filter(|&&x| x.is_finite() && (x < lower_bound || x > upper_bound))
                .collect();
            
            println!("   🎯 Outliers detected: {} ({:.1}%)", 
                     outliers.len(), 
                     (outliers.len() as f32 / data_slice.len() as f32) * 100.0);
            
            if outliers.len() > 0 {
                println!("      IQR bounds: [{:.3}, {:.3}]", lower_bound, upper_bound);
                for (idx, &outlier) in outliers.iter().take(3).enumerate() {
                    println!("      Outlier {}: {:.3}", idx + 1, outlier);
                }
            }
        }
        
        println!("   {}", "─".repeat(50));
    }
    
    // Real-time monitoring simulation
    println!("\n--- Real-time Quality Monitoring ---");
    
    println!("Simulating real-time data stream...");
    
    let mut good_quality_count = 0;
    let mut poor_quality_count = 0;
    
    for minute in 0..60 {
        // Simulate minute-by-minute data quality
        let data = generate_minute_data(minute);
        let tensor = WasmTensor::new(data, vec![100]); // 100 samples per minute
        
        if let Ok(quality_score) = quality_analyzer.overall_quality(&tensor) {
            if quality_score >= 0.7 {
                good_quality_count += 1;
                if minute % 10 == 0 {
                    println!("   ✅ Minute {}: Quality {:.1}% (Good)", minute, quality_score * 100.0);
                }
            } else {
                poor_quality_count += 1;
                println!("   ⚠️  Minute {}: Quality {:.1}% (Poor - investigate)", minute, quality_score * 100.0);
                
                // Detailed analysis for poor quality
                if let Ok(report) = quality_analyzer.quality_report(&tensor) {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&report) {
                        if let Some(issues) = parsed.get("issues") {
                            println!("      Issues: {}", issues);
                        }
                    }
                }
            }
        }
        
        // Detect anomalies in real-time
        let avg_value = tensor.data().iter().sum::<f32>() / tensor.data().len() as f32;
        if let Ok(Some(anomaly)) = statistical_detector.detect_realtime(avg_value) {
            println!("   🚨 Anomaly in minute {}: avg={:.3}, detail={:?}", minute, avg_value, anomaly);
        }
    }
    
    // Quality monitoring summary
    println!("\n--- Quality Monitoring Summary ---");
    println!("Good quality minutes: {} ({:.1}%)", 
             good_quality_count, 
             (good_quality_count as f32 / 60.0) * 100.0);
    println!("Poor quality minutes: {} ({:.1}%)", 
             poor_quality_count,
             (poor_quality_count as f32 / 60.0) * 100.0);
    
    // Final detector statistics
    if let Ok(final_stats) = statistical_detector.get_statistics() {
        println!("Final detector stats: {}", final_stats);
    }
    
    // Memory performance
    println!("\n--- Memory Performance ---");
    println!("Pool statistics: {}", MemoryManager::get_stats());
    println!("Cache efficiency: {}", MemoryManager::cache_efficiency());
    
    // Demonstrate quality threshold adjustment
    println!("\n--- Dynamic Quality Thresholds ---");
    
    // Create analyzer with different thresholds
    let strict_analyzer = WasmQualityMetrics::new(0.95)
        .expect("Failed to create strict analyzer");
    let lenient_analyzer = WasmQualityMetrics::new(0.5)
        .expect("Failed to create lenient analyzer");
    
    let test_data = create_medium_quality_data();
    let test_tensor = WasmTensor::new(test_data.0, test_data.1);
    
    if let (Ok(strict_quality), Ok(lenient_quality)) = (
        strict_analyzer.overall_quality(&test_tensor),
        lenient_analyzer.overall_quality(&test_tensor)
    ) {
        println!("Same data with different thresholds:");
        println!("   Strict (0.95): {:.1}%", strict_quality * 100.0);
        println!("   Lenient (0.5): {:.1}%", lenient_quality * 100.0);
        
        let threshold_sensitivity = (lenient_quality - strict_quality) / 0.45; // Threshold difference
        println!("   Threshold sensitivity: {:.3}", threshold_sensitivity);
    }
    
    println!("\n=== Quality Assessment Demo Complete ===");
}

/// Generate high quality data (clean, consistent, complete)
fn create_high_quality_data() -> (Vec<f32>, Vec<usize>, &'static str) {
    let data: Vec<f32> = (0..1000).map(|i| {
        let t = i as f32 / 100.0;
        2.0 + 0.5 * t + 0.1 * (t * 3.14159).sin()
    }).collect();
    
    (data, vec![1000], "Clean trending data with minor seasonal variation")
}

/// Generate medium quality data (some outliers, mostly good)
fn create_medium_quality_data() -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data: Vec<f32> = (0..1000).map(|i| {
        let t = i as f32 / 100.0;
        2.0 + 0.5 * t + 0.2 * (t * 3.14159).sin() + 0.1 * (t * 7.0).cos()
    }).collect();
    
    // Add some outliers
    for i in [100, 300, 500, 700, 900] {
        data[i] *= 3.0; // Moderate outliers
    }
    
    (data, vec![1000], "Trending data with moderate outliers and noise")
}

/// Generate low quality data (many outliers, inconsistent)
fn create_low_quality_data() -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data: Vec<f32> = (0..1000).map(|i| {
        let base = (i % 100) as f32; // Sawtooth pattern
        base + (i as f32 * 0.1).sin() * 5.0 // High noise
    }).collect();
    
    // Add many outliers
    for i in (0..1000).step_by(50) {
        data[i] = if i % 100 == 0 { 1000.0 } else { -500.0 };
    }
    
    // Add some NaN values to reduce completeness
    for i in [50, 150, 250, 350, 450] {
        if i < data.len() {
            data[i] = f32::NAN;
        }
    }
    
    (data, vec![1000], "Highly corrupted data with outliers and missing values")
}

/// Generate mixed quality data (good and bad sections)
fn create_mixed_quality_data() -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(1000);
    
    // First half: high quality
    for i in 0..500 {
        let t = i as f32 / 100.0;
        data.push(5.0 + 0.2 * t + 0.05 * (t * 6.28).sin());
    }
    
    // Second half: low quality
    for i in 500..1000 {
        if i % 10 == 0 {
            data.push(f32::NAN); // Missing values
        } else if i % 7 == 0 {
            data.push(100.0 * (i as f32).sin()); // Large outliers
        } else {
            let base = ((i - 500) % 20) as f32;
            data.push(base + (i as f32 * 0.5).sin() * 2.0);
        }
    }
    
    (data, vec![1000], "Mixed quality: clean first half, corrupted second half")
}

/// Generate realistic data with typical data quality issues
fn create_realistic_data() -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(2000);
    
    // Simulate sensor data over time
    for hour in 0..24 {
        for minute in 0..60 {
            let time_factor = hour as f32 + minute as f32 / 60.0;
            
            // Base temperature pattern (daily cycle)
            let base_temp = 20.0 + 8.0 * (2.0 * std::f32::consts::PI * time_factor / 24.0).sin();
            
            // Add realistic noise and issues
            let value = if minute % 17 == 0 {
                // Sensor calibration drift
                base_temp + (time_factor * 0.1) 
            } else if minute % 23 == 0 && hour > 12 {
                // Afternoon heat spikes
                base_temp + 3.0 + (minute as f32 * 0.1).sin()
            } else if minute % 37 == 0 {
                // Occasional sensor malfunction
                if time_factor > 12.0 { f32::NAN } else { base_temp * 1.5 }
            } else {
                // Normal reading with small noise
                base_temp + 0.2 * ((time_factor * minute as f32).sin())
            };
            
            data.push(value);
        }
    }
    
    (data, vec![24, 60], "24-hour temperature sensor data with realistic issues")
}

/// Generate data for a specific minute with quality variations
fn generate_minute_data(minute: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(100);
    
    // Base value with hourly trend
    let hour = minute / 60;
    let minute_in_hour = minute % 60;
    let base_value = 10.0 + (hour as f32) * 0.5 + (minute_in_hour as f32) * 0.01;
    
    for second in 0..100 {
        let value = if minute % 13 == 0 && second % 20 == 0 {
            // Periodic data corruption
            base_value * 5.0
        } else if minute % 19 == 0 && second % 30 == 0 {
            // Missing data simulation
            f32::NAN
        } else {
            // Normal data with small variations
            base_value + 0.1 * (second as f32 * 0.1).sin() + 0.05 * (second as f32 * 0.3).cos()
        };
        
        data.push(value);
    }
    
    data
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_quality_assessment");
}