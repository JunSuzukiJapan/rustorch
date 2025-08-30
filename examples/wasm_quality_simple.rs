//! Simple data quality assessment example
//! シンプルなデータ品質評価の例

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::tensor::WasmTensor;
    use rustorch::wasm::quality_metrics::WasmQualityMetrics;
    use rustorch::wasm::advanced_math::WasmAdvancedMath;
    use rustorch::wasm::common::MemoryManager;
    
    println!("=== RusTorch WASM Simple Quality Assessment ===");
    
    // Initialize memory pool
    MemoryManager::init_pool(50);
    
    // Create quality analyzer
    let quality_analyzer = WasmQualityMetrics::new(0.8)
        .expect("Failed to create quality analyzer");
    let advanced_math = WasmAdvancedMath::new();
    
    println!("Quality assessment tools initialized");
    
    // Test different data quality scenarios
    let test_datasets = vec![
        ("perfect_data", vec![1.0, 2.0, 3.0, 4.0, 5.0], "Sequential clean data"),
        ("noisy_data", vec![1.1, 1.9, 3.2, 3.8, 5.1], "Data with small noise"),
        ("outlier_data", vec![1.0, 2.0, 15.0, 4.0, 5.0], "Data with one outlier"),
        ("nan_data", vec![1.0, f32::NAN, 3.0, 4.0, 5.0], "Data with missing value"),
    ];
    
    for (name, data, description) in test_datasets {
        println!("\n📊 Testing: {} ({})", name, description);
        
        let tensor = WasmTensor::new(data.clone(), vec![data.len()]);
        
        // Quality metrics
        if let Ok(completeness) = quality_analyzer.completeness(&tensor) {
            println!("   📈 Completeness: {:.1}%", completeness);
        }
        
        if let Ok(validity) = quality_analyzer.validity(&tensor) {
            println!("   ✅ Validity: {:.1}%", validity);
        }
        
        if let Ok(consistency) = quality_analyzer.consistency(&tensor) {
            println!("   🔄 Consistency: {:.1}%", consistency);
        }
        
        if let Ok(uniqueness) = quality_analyzer.uniqueness(&tensor) {
            println!("   🔢 Uniqueness: {:.1}%", uniqueness);
        }
        
        if let Ok(overall) = quality_analyzer.overall_quality(&tensor) {
            println!("   🎯 Overall Quality: {:.1}%", overall);
        }
        
        // Generate quality report
        if let Ok(report) = quality_analyzer.quality_report(&tensor) {
            println!("   📋 Report: {}", report);
        }
        
        // Mathematical analysis
        let finite_data: Vec<f32> = data.iter().filter(|x| x.is_finite()).cloned().collect();
        if finite_data.len() > 0 {
            let mean = finite_data.iter().sum::<f32>() / finite_data.len() as f32;
            let variance = finite_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / finite_data.len() as f32;
            let std_dev = variance.sqrt();
            
            println!("   📊 Statistics: mean={:.2}, std={:.2}", mean, std_dev);
            
            // Test mathematical operations on clean data
            if finite_data.len() == data.len() {
                let normalized_data: Vec<f32> = finite_data.iter()
                    .map(|&x| (x - mean) / std_dev.max(1e-6))
                    .map(|x| x.clamp(-1.0, 1.0)) // Ensure valid range for asin
                    .collect();
                let data_len = normalized_data.len();
                let clean_tensor = WasmTensor::new(normalized_data, vec![data_len]);
                
                if let Ok(asin_result) = advanced_math.asin(&clean_tensor) {
                    let asin_range = asin_result.data().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let asin_max = asin_result.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    println!("   🌊 Asin transform range: [{:.3}, {:.3}]", asin_range, asin_max);
                }
                
                if let Ok(tanh_result) = advanced_math.tanh(&clean_tensor) {
                    let tanh_mean = tanh_result.data().iter().sum::<f32>() / tanh_result.data().len() as f32;
                    println!("   🎯 Tanh mean: {:.3}", tanh_mean);
                }
            }
        }
    }
    
    // Memory performance
    println!("\n--- Memory Performance ---");
    println!("Pool stats: {}", MemoryManager::get_stats());
    println!("Cache efficiency: {}", MemoryManager::cache_efficiency());
    
    println!("\n=== Simple Quality Assessment Complete ===");
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_quality_simple");
}