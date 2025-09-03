//! Real-time anomaly detection streaming example
//! リアルタイム異常検知ストリーミングの例

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::anomaly_detection::{WasmAnomalyDetector, WasmTimeSeriesDetector};
    use rustorch::wasm::common::MemoryManager;
    use rustorch::wasm::tensor::WasmTensor;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    println!("=== RusTorch WASM Real-time Anomaly Detection ===");

    // Initialize memory pool for streaming
    MemoryManager::init_pool(50);

    // Create detectors
    let statistical_detector =
        WasmAnomalyDetector::new(2.5, 100).expect("Failed to create statistical detector");
    let timeseries_detector =
        WasmTimeSeriesDetector::new(50, Some(24)).expect("Failed to create time series detector");

    println!("Detectors initialized:");
    println!("  Statistical: threshold=2.5, window=100");
    println!("  Time Series: window=50, seasonal_period=24");

    // Simulate streaming data with patterns and anomalies
    println!("\n--- Streaming Data Simulation ---");

    let mut normal_count = 0;
    let mut anomaly_count = 0;
    let mut time_anomaly_count = 0;

    for i in 0..500 {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + i;

        // Generate synthetic streaming data
        let base_value = generate_synthetic_value(i as usize);

        // Add occasional anomalies
        let value = if i % 47 == 0 {
            base_value + 10.0 // Strong positive anomaly
        } else if i % 73 == 0 {
            base_value - 8.0 // Strong negative anomaly
        } else if i % 31 == 0 {
            base_value * 3.0 // Multiplicative anomaly
        } else {
            base_value
        };

        // Real-time statistical detection
        if let Ok(anomaly) = statistical_detector.detect_realtime(value) {
            anomaly_count += 1;
            println!(
                "🚨 Statistical anomaly at step {}: value={:.3}, anomaly={:?}",
                i, value, anomaly
            );
        } else {
            normal_count += 1;
        }

        // Time series detection
        if let Ok(ts_anomaly) = timeseries_detector.add_point(timestamp as f64, value) {
            time_anomaly_count += 1;
            println!(
                "📈 Time series anomaly at step {}: ts={}, value={:.3}, anomaly={:?}",
                i, timestamp, value, ts_anomaly
            );
        }

        // Periodic statistics
        if i % 100 == 0 && i > 0 {
            println!("\n--- Statistics at step {} ---", i);
            if let Ok(stats) = statistical_detector.get_statistics() {
                println!("Statistical detector: {}", stats);
            }

            if let Ok(trend) = timeseries_detector.get_trend_analysis() {
                println!("Trend analysis: {}", trend);
            }

            if let Ok(seasonal) = timeseries_detector.get_seasonal_analysis() {
                println!("Seasonal analysis: {}", seasonal);
            }

            println!("Memory: {}", MemoryManager::cache_efficiency());
        }

        // Small delay to simulate real-time processing
        thread::sleep(Duration::from_millis(10));
    }

    // Final summary
    println!("\n--- Final Detection Summary ---");
    println!("Normal points: {}", normal_count);
    println!("Statistical anomalies: {}", anomaly_count);
    println!("Time series anomalies: {}", time_anomaly_count);
    println!("Total points processed: {}", normal_count + anomaly_count);

    let anomaly_rate = (anomaly_count as f32 / (normal_count + anomaly_count) as f32) * 100.0;
    println!("Anomaly detection rate: {:.2}%", anomaly_rate);

    // Batch detection demonstration
    println!("\n--- Batch Anomaly Detection ---");

    // Generate larger dataset for batch processing
    let batch_size = 1000;
    let mut batch_data = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let value = generate_synthetic_value(i);
        // Add some clear anomalies
        let final_value = if i % 100 == 0 {
            value + 15.0 // Strong anomaly every 100 points
        } else {
            value
        };
        batch_data.push(final_value);
    }

    let batch_tensor = WasmTensor::new(batch_data, vec![batch_size]);

    // Statistical batch detection
    if let Ok(statistical_anomalies) = statistical_detector.detect_statistical(&batch_tensor) {
        println!(
            "Statistical batch anomalies detected: {}",
            statistical_anomalies.len()
        );

        // Show first few anomalies
        for (idx, anomaly) in statistical_anomalies.iter().take(5).enumerate() {
            println!("  Anomaly {}: {:?}", idx + 1, anomaly);
        }
    }

    // Isolation Forest detection
    if let Ok(isolation_anomalies) =
        statistical_detector.detect_isolation_forest(&batch_tensor, 100)
    {
        println!(
            "Isolation Forest anomalies detected: {}",
            isolation_anomalies.len()
        );

        // Show first few anomalies
        for (idx, anomaly) in isolation_anomalies.iter().take(5).enumerate() {
            println!("  IF Anomaly {}: {:?}", idx + 1, anomaly);
        }
    }

    // Performance analysis
    println!("\n--- Performance Analysis ---");
    if let Ok(detector_stats) = statistical_detector.get_statistics() {
        println!("Detector performance: {}", detector_stats);
    }

    println!("Memory pool performance: {}", MemoryManager::get_stats());

    // Demonstrate threshold tuning
    println!("\n--- Dynamic Threshold Tuning ---");
    println!(
        "Original threshold: {:.2}",
        statistical_detector.get_threshold()
    );

    statistical_detector.set_threshold(1.5); // More sensitive
    println!("New threshold: {:.2}", statistical_detector.get_threshold());

    // Re-run detection with new threshold
    if let Ok(sensitive_anomalies) = statistical_detector.detect_statistical(&batch_tensor) {
        println!(
            "Anomalies with sensitive threshold: {}",
            sensitive_anomalies.len()
        );
    }

    // Reset detector state
    statistical_detector.reset();
    println!("Detector reset - ready for new data");

    println!("\n=== Anomaly Detection Demo Complete ===");
}

/// Generate synthetic data with patterns and noise
#[allow(dead_code)]
fn generate_synthetic_value(step: usize) -> f32 {
    let t = step as f32;

    // Base trend (slowly increasing)
    let trend = t * 0.01;

    // Seasonal pattern (daily cycle simulation)
    let seasonal = 3.0 * (2.0 * std::f32::consts::PI * t / 24.0).sin();

    // Weekly pattern (longer cycle)
    let weekly = 1.5 * (2.0 * std::f32::consts::PI * t / (24.0 * 7.0)).cos();

    // Random noise
    let noise = (t * 0.1).sin() * 0.5 + (t * 0.07).cos() * 0.3;

    // Combine all components
    10.0 + trend + seasonal + weekly + noise
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_anomaly_streaming");
}
