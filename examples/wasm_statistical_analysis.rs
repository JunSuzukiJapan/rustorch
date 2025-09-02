//! Comprehensive statistical analysis example with WASM
//! WASMを使った包括的統計解析の例

// Temporarily disabled due to API changes
#[allow(dead_code)]
fn main() {
    println!("WASM statistical analysis example temporarily disabled");
}

#[cfg(feature = "wasm")]
#[allow(dead_code)]
fn _disabled_main() {
    use rustorch::wasm::advanced_math::WasmAdvancedMath;
    use rustorch::wasm::common::MemoryManager;
    use rustorch::wasm::statistical_analysis::{WasmStatisticalAnalyzer, WasmStatisticalFunctions};
    use rustorch::wasm::tensor::WasmTensor;

    println!("=== RusTorch WASM Statistical Analysis Demo ===");

    // Initialize memory pool for statistical computations
    MemoryManager::init_pool(80);

    // Create analysis tools
    let statistical_functions = WasmStatisticalFunctions::new();
    let statistical_analyzer = WasmStatisticalAnalyzer::new();
    let advanced_math = WasmAdvancedMath::new();

    println!("Statistical analysis tools initialized");

    // Generate sample datasets for different analysis scenarios
    let datasets = vec![
        (
            "normal_distribution",
            generate_normal_distribution(1000, 50.0, 15.0),
        ),
        ("bimodal_distribution", generate_bimodal_distribution(1000)),
        ("time_series", generate_time_series(365, true)), // One year with seasonality
        ("financial_returns", generate_financial_returns(252)), // Trading days
        ("experimental_data", generate_experimental_data(500)),
    ];

    for (dataset_name, (data, shape, description)) in datasets {
        println!("\n{}", "=".repeat(60));
        println!("📊 Dataset: {} ({})", dataset_name, description);
        println!("{}", "=".repeat(60));

        let tensor = WasmTensor::new(data, shape.clone());
        println!(
            "Shape: {:?}, Elements: {}",
            tensor.shape(),
            tensor.data().len()
        );

        // Basic statistical analysis
        println!("\n--- Basic Statistics ---");
        if let Ok(basic_stats) = statistical_analyzer.basic_stats(&tensor) {
            if let Ok(stats_json) = serde_json::from_str::<serde_json::Value>(&basic_stats) {
                println!("📈 Basic Statistics:");
                if let Some(mean) = stats_json.get("mean") {
                    println!("   Mean: {:.3}", mean.as_f64().unwrap_or(0.0));
                }
                if let Some(std) = stats_json.get("std") {
                    println!("   Std Dev: {:.3}", std.as_f64().unwrap_or(0.0));
                }
                if let Some(min) = stats_json.get("min") {
                    println!("   Min: {:.3}", min.as_f64().unwrap_or(0.0));
                }
                if let Some(max) = stats_json.get("max") {
                    println!("   Max: {:.3}", max.as_f64().unwrap_or(0.0));
                }
            }
        }

        // Percentile analysis
        println!("\n--- Percentile Analysis ---");
        let percentiles = vec![1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
        if let Ok(percentile_values) =
            statistical_analyzer.percentiles(&tensor, percentiles.clone())
        {
            println!("📊 Percentiles:");
            for (p, v) in percentiles.iter().zip(percentile_values.iter()) {
                println!("   P{:2.0}: {:8.3}", p, v);
            }

            // Calculate IQR for outlier detection
            let q1 = percentile_values[3]; // 25th percentile
            let q3 = percentile_values[5]; // 75th percentile
            let iqr = q3 - q1;
            println!("   IQR: {:.3} (Q3 - Q1)", iqr);

            // Outlier bounds
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;
            println!(
                "   Outlier bounds: [{:.3}, {:.3}]",
                lower_bound, upper_bound
            );
        }

        // Outlier detection
        println!("\n--- Outlier Detection ---");
        if let Ok(outliers) = statistical_analyzer.detect_outliers(&tensor) {
            println!("🎯 Outliers detected: {}", outliers.len());

            if outliers.len() > 0 {
                println!("   First 5 outliers:");
                for (idx, outlier) in outliers.iter().take(5).enumerate() {
                    println!("     {}. {:?}", idx + 1, outlier);
                }

                if outliers.len() > 5 {
                    println!("     ... and {} more", outliers.len() - 5);
                }

                let outlier_rate = (outliers.len() as f32 / tensor.data().len() as f32) * 100.0;
                println!("   Outlier rate: {:.2}%", outlier_rate);
            }
        }

        // Advanced mathematical operations for distribution analysis
        println!("\n--- Distribution Analysis ---");

        // Normality test using mathematical functions
        let abs_tensor = advanced_math.sign(&tensor).expect("sign operation failed");
        let log_tensor = if tensor.data().iter().all(|&x| x > 0.0) {
            Some(advanced_math.log(&tensor).expect("log operation failed"))
        } else {
            None
        };

        println!("🔍 Distribution characteristics:");
        println!("   Sign changes: {}", count_sign_changes(abs_tensor.data()));

        if let Some(ref log_data) = log_tensor {
            let log_range = log_data.data().iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let log_max = log_data
                .data()
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            println!("   Log transform range: [{:.3}, {:.3}]", log_range, log_max);
        }

        // Skewness approximation using percentiles
        if let Ok(percentile_values) =
            statistical_analyzer.percentiles(&tensor, vec![25.0, 50.0, 75.0])
        {
            let q1 = percentile_values[0];
            let median = percentile_values[1];
            let q3 = percentile_values[2];

            // Bowley skewness
            let skewness = (q3 + q1 - 2.0 * median) / (q3 - q1);
            println!("   Approximate skewness: {:.3}", skewness);

            let skew_interpretation = match skewness.abs() {
                x if x < 0.1 => "Nearly symmetric",
                x if x < 0.3 => "Slightly skewed",
                x if x < 0.5 => "Moderately skewed",
                _ => "Highly skewed",
            };
            println!("   Skewness interpretation: {}", skew_interpretation);
        }

        // Correlation analysis if we have paired data
        if shape.len() == 2 && shape[1] >= 2 {
            println!("\n--- Correlation Analysis ---");

            let col1_data: Vec<f32> = tensor
                .data()
                .iter()
                .enumerate()
                .filter(|(i, _)| i % shape[1] == 0)
                .map(|(_, &v)| v)
                .collect();
            let col2_data: Vec<f32> = tensor
                .data()
                .iter()
                .enumerate()
                .filter(|(i, _)| i % shape[1] == 1)
                .map(|(_, &v)| v)
                .collect();

            if col1_data.len() == col2_data.len() && col1_data.len() > 1 {
                let col1_tensor = WasmTensor::new(col1_data, vec![col1_data.len()]);
                let col2_tensor = WasmTensor::new(col2_data, vec![col2_data.len()]);

                if let Ok(correlation) =
                    statistical_functions.correlation(&col1_tensor, &col2_tensor)
                {
                    println!("🔗 Correlation coefficient: {:.3}", correlation);

                    let corr_strength = match correlation.abs() {
                        x if x > 0.8 => "Very strong",
                        x if x > 0.6 => "Strong",
                        x if x > 0.4 => "Moderate",
                        x if x > 0.2 => "Weak",
                        _ => "Very weak",
                    };
                    println!("   Correlation strength: {}", corr_strength);
                }

                if let Ok(covariance) = statistical_functions.covariance(&col1_tensor, &col2_tensor)
                {
                    println!("📐 Covariance: {:.3}", covariance);
                }
            }
        }

        // Hypothesis testing simulation
        println!("\n--- Hypothesis Testing Simulation ---");

        // Test for normality using percentile-based approach
        if let Ok(percentiles) =
            statistical_analyzer.percentiles(&tensor, vec![2.5, 25.0, 50.0, 75.0, 97.5])
        {
            let p2_5 = percentiles[0];
            let q1 = percentiles[1];
            let median = percentiles[2];
            let q3 = percentiles[3];
            let p97_5 = percentiles[4];

            // Normal distribution should have specific percentile relationships
            let expected_q1_median = (median - q1) / (q3 - median);
            let tail_symmetry = (p97_5 - median) / (median - p2_5);

            println!("🧪 Normality indicators:");
            println!(
                "   Q1-Median ratio: {:.3} (expect ~1.0 for normal)",
                expected_q1_median
            );
            println!(
                "   Tail symmetry: {:.3} (expect ~1.0 for normal)",
                tail_symmetry
            );

            let normality_score =
                1.0 - ((expected_q1_median - 1.0).abs() + (tail_symmetry - 1.0).abs()) / 2.0;
            let normality_score = normality_score.max(0.0).min(1.0);
            println!("   Normality score: {:.3}", normality_score);
        }

        // Statistical significance testing
        let sample_size = tensor.data().len();
        let degrees_of_freedom = sample_size - 1;
        println!("📏 Sample characteristics:");
        println!("   Sample size: {}", sample_size);
        println!("   Degrees of freedom: {}", degrees_of_freedom);

        // Power analysis approximation
        if sample_size >= 30 {
            println!("   Statistical power: High (n ≥ 30)");
        } else if sample_size >= 10 {
            println!("   Statistical power: Moderate (10 ≤ n < 30)");
        } else {
            println!("   Statistical power: Low (n < 10)");
        }

        println!("   {}", "─".repeat(50));
    }

    // Comparative analysis
    println!("\n{}", "=".repeat(60));
    println!("📊 COMPARATIVE ANALYSIS");
    println!("{}", "=".repeat(60));

    // Compare distributions
    let normal_data = generate_normal_distribution(500, 0.0, 1.0);
    let uniform_data = generate_uniform_distribution(500, -2.0, 2.0);

    let normal_tensor = WasmTensor::new(normal_data.0, normal_data.1);
    let uniform_tensor = WasmTensor::new(uniform_data.0, uniform_data.1);

    println!("\nComparing Normal vs Uniform distributions:");

    for (name, tensor) in [("Normal", &normal_tensor), ("Uniform", &uniform_tensor)] {
        if let Ok(basic_stats) = statistical_analyzer.basic_stats(tensor) {
            if let Ok(stats) = serde_json::from_str::<serde_json::Value>(&basic_stats) {
                println!(
                    "   {}: mean={:.3}, std={:.3}",
                    name,
                    stats.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    stats.get("std").and_then(|v| v.as_f64()).unwrap_or(0.0)
                );
            }
        }
    }

    // Cross-correlation if same size
    if normal_tensor.data().len() == uniform_tensor.data().len() {
        if let Ok(cross_correlation) =
            statistical_functions.correlation(&normal_tensor, &uniform_tensor)
        {
            println!("🔗 Cross-correlation: {:.3}", cross_correlation);
            println!("   Expected: ~0.0 (independent distributions)");
        }
    }

    // Memory efficiency analysis
    println!("\n--- Memory Efficiency Analysis ---");
    let final_stats = MemoryManager::get_stats();
    let efficiency = MemoryManager::cache_efficiency();

    println!("📊 Final pool statistics: {}", final_stats);
    println!("⚡ Cache efficiency: {}", efficiency);

    if let Ok(stats_json) = serde_json::from_str::<serde_json::Value>(&final_stats) {
        let total_allocations = stats_json["total_allocations"].as_u64().unwrap_or(0);
        let cache_hits = stats_json["cache_hits"].as_u64().unwrap_or(0);
        let memory_saved = stats_json["memory_saved_bytes"].as_u64().unwrap_or(0);

        println!("📈 Performance metrics:");
        println!("   Total operations: {}", total_allocations);
        println!("   Cache hits: {}", cache_hits);
        println!(
            "   Memory saved: {:.2} MB",
            memory_saved as f32 / (1024.0 * 1024.0)
        );

        if total_allocations > 0 {
            let efficiency_pct = (cache_hits as f32 / total_allocations as f32) * 100.0;
            println!("   Efficiency: {:.1}%", efficiency_pct);
        }
    }

    println!("\n=== Statistical Analysis Demo Complete ===");
}

/// Generate normal distribution data
#[allow(dead_code)]
fn generate_normal_distribution(
    n: usize,
    mean: f32,
    std: f32,
) -> (Vec<f32>, Vec<usize>, &'static str) {
    // Box-Muller transform for normal distribution
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let u1 = (i as f32 + 1.0) / (n as f32 + 1.0); // Avoid 0
        let u2 = ((i * 7 + 3) % n) as f32 / n as f32;

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let value = mean + std * z;
        data.push(value);
    }

    (
        data,
        vec![n],
        "Normal distribution using Box-Muller transform",
    )
}

/// Generate bimodal distribution
#[allow(dead_code)]
fn generate_bimodal_distribution(n: usize) -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let value = if i % 2 == 0 {
            // First mode centered at -2
            -2.0 + (i as f32 * 0.1).sin() * 0.5
        } else {
            // Second mode centered at +3
            3.0 + (i as f32 * 0.1).cos() * 0.7
        };

        data.push(value);
    }

    (
        data,
        vec![n],
        "Bimodal distribution with modes at -2 and +3",
    )
}

/// Generate time series with trend and seasonality
#[allow(dead_code)]
fn generate_time_series(n: usize, with_seasonality: bool) -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(n);

    for day in 0..n {
        let t = day as f32;

        // Base trend (gradual increase)
        let trend = 100.0 + t * 0.1;

        // Seasonal component (annual cycle)
        let seasonal = if with_seasonality {
            10.0 * (2.0 * std::f32::consts::PI * t / 365.0).sin()
        } else {
            0.0
        };

        // Weekly pattern
        let weekly = 3.0 * (2.0 * std::f32::consts::PI * t / 7.0).cos();

        // Random noise
        let noise = (t * 0.1).sin() * 2.0 + (t * 0.07).cos() * 1.5;

        // Special events (quarterly peaks)
        let special = if day % 90 == 0 && day > 0 {
            5.0 * (1.0 + (t * 0.01).sin())
        } else {
            0.0
        };

        let value = trend + seasonal + weekly + noise + special;
        data.push(value);
    }

    (
        data,
        vec![n],
        "Time series with trend, seasonality, and noise",
    )
}

/// Generate financial returns data
#[allow(dead_code)]
fn generate_financial_returns(n: usize) -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(n);
    let mut _price = 100.0; // Starting price

    for day in 0..n {
        // Simulate daily returns with volatility clustering
        let volatility = 0.15 + 0.05 * (day as f32 / 50.0).sin().abs();

        let base_return = (day as f32 * 0.1).sin() * 0.002; // Small drift
        let random_shock = (day as f32 * 0.3).cos() * volatility / 252.0_f32.sqrt();

        // Occasional market events
        let event_return = if day % 63 == 0 {
            0.05 * if day % 126 == 0 { -1.0 } else { 1.0 } // Market shock
        } else {
            0.0
        };

        let daily_return = base_return + random_shock + event_return;
        _price *= 1.0 + daily_return;

        data.push(daily_return * 100.0); // Convert to percentage
    }

    (
        data,
        vec![n],
        "Simulated daily stock returns with volatility clustering",
    )
}

/// Generate experimental measurement data
#[allow(dead_code)]
fn generate_experimental_data(n: usize) -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(n);

    for measurement in 0..n {
        let true_value = 25.0; // True experimental value

        // Measurement error components
        let systematic_error = 0.3; // Calibration bias
        let random_error = (measurement as f32 * 0.2).sin() * 0.5; // Random measurement noise

        // Occasional measurement failures
        let measurement_error = if measurement % 47 == 0 {
            5.0 // Significant measurement error
        } else if measurement % 23 == 0 {
            -2.0 // Negative bias error
        } else {
            0.0
        };

        // Environmental factors
        let environmental = 0.1 * (measurement as f32 / 20.0).cos(); // Temperature drift

        let measured_value =
            true_value + systematic_error + random_error + measurement_error + environmental;
        data.push(measured_value);
    }

    (
        data,
        vec![n],
        "Experimental measurements with systematic and random errors",
    )
}

/// Generate uniform distribution
#[allow(dead_code)]
fn generate_uniform_distribution(
    n: usize,
    min: f32,
    max: f32,
) -> (Vec<f32>, Vec<usize>, &'static str) {
    let mut data = Vec::with_capacity(n);
    let range = max - min;

    for i in 0..n {
        // Linear congruential generator for uniform distribution
        let value = min + range * ((i * 1664525 + 1013904223) % 2147483647) as f32 / 2147483647.0;
        data.push(value);
    }

    (
        data,
        vec![n],
        "Uniform distribution using linear congruential generator",
    )
}

/// Count sign changes in data (useful for trend analysis)
#[allow(dead_code)]
fn count_sign_changes(data: &[f32]) -> usize {
    let mut changes = 0;
    let mut prev_positive = data[0] >= 0.0;

    for &value in data.iter().skip(1) {
        let current_positive = value >= 0.0;
        if current_positive != prev_positive {
            changes += 1;
        }
        prev_positive = current_positive;
    }

    changes
}
