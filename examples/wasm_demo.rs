//! WASM modules demonstration
//! WASMモジュールのデモンストレーション

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::tensor::WasmTensor;
    use rustorch::wasm::advanced_math::WasmAdvancedMath;
    use rustorch::wasm::quality_metrics::WasmQualityMetrics;
    use rustorch::wasm::data_transforms::WasmNormalize;
    use rustorch::wasm::common::MemoryManager;
    
    println!("=== RusTorch WASM Demo ===");
    
    // Initialize memory pool
    MemoryManager::init_pool(50);
    
    // Create sample data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let tensor = WasmTensor::new(data, vec![3, 3]);
    
    println!("Original tensor: {:?}", tensor.data());
    
    // Advanced Math Operations
    println!("\n--- Advanced Math ---");
    let math = WasmAdvancedMath::new();
    
    if let Ok(sinh_result) = math.sinh(&tensor) {
        println!("sinh result: {:?}", &sinh_result.data()[..3]);
    }
    
    if let Ok(tanh_result) = math.tanh(&tensor) {
        println!("tanh result: {:?}", &tanh_result.data()[..3]);
    }
    
    // Quality Metrics
    println!("\n--- Quality Metrics ---");
    if let Ok(quality) = WasmQualityMetrics::new(0.8) {
        if let Ok(completeness) = quality.completeness(&tensor) {
            println!("Data completeness: {:.2}%", completeness);
        }
        
        if let Ok(validity) = quality.validity(&tensor) {
            println!("Data validity: {:.2}%", validity);
        }
        
        if let Ok(report) = quality.quality_report(&tensor) {
            println!("Quality report: {}", report);
        }
    }
    
    // Data Transforms
    println!("\n--- Data Transforms ---");
    if let Ok(normalize) = WasmNormalize::new(&[5.0], &[2.5]) {
        if let Ok(normalized) = normalize.apply(&tensor) {
            println!("Normalized data: {:?}", &normalized.data()[..3]);
        }
    }
    
    // Memory Statistics
    println!("\n--- Memory Statistics ---");
    println!("Pool stats: {}", MemoryManager::get_stats());
    println!("Cache efficiency: {}", MemoryManager::cache_efficiency());
    
    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_demo");
}