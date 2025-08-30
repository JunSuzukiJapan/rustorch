# WASM Testing Documentation

## Overview

The RusTorch WASM modules include comprehensive testing that validates cross-module integration, performance characteristics, and API correctness. The testing system uses a trait-based approach that ensures all modules behave consistently.

## Test Architecture

### Integration Tests

Located in `tests/wasm_integration_tests.rs`, these tests validate:

- **Cross-module functionality**: Operations that span multiple WASM modules
- **Memory pool integration**: Proper buffer management across operations
- **Error handling consistency**: Standardized error behavior
- **Performance characteristics**: Cache hit rates and memory efficiency
- **API compatibility**: TypeScript bindings match Rust implementations

### Test Categories

#### 1. Cross-Module Integration

Tests that validate modules work together correctly:

```rust
#[test]
fn test_advanced_math_integration() {
    let data = vec![0.5, 1.0, 1.5, 2.0];
    let tensor = WasmTensor::new(data, vec![4]);
    let math = WasmAdvancedMath::new();
    
    let sinh_result = math.sinh(&tensor).expect("sinh failed");
    let cosh_result = math.cosh(&tensor).expect("cosh failed");
    let tanh_result = math.tanh(&tensor).expect("tanh failed");
    
    // Verify mathematical relationship: tanh(x) = sinh(x) / cosh(x)
    for i in 0..4 {
        let expected_tanh = sinh_result.data()[i] / cosh_result.data()[i];
        let actual_tanh = tanh_result.data()[i];
        assert!((expected_tanh - actual_tanh).abs() < 1e-6);
    }
}
```

#### 2. Memory Pool Validation

Tests ensuring efficient memory management:

```rust
#[test]
fn test_memory_pool_efficiency() {
    MemoryManager::init_pool(10);
    
    // Reset counters for clean measurement
    let initial_stats = MemoryManager::get_stats();
    
    let tensors: Vec<_> = (0..20).map(|i| {
        let data = vec![i as f32; 100];
        WasmTensor::new(data, vec![10, 10])
    }).collect();
    
    // Check that pool is being used efficiently
    let stats = MemoryManager::get_stats();
    let parsed: serde_json::Value = serde_json::from_str(&stats).unwrap();
    
    let cache_hits = parsed["cache_hits"].as_u64().unwrap();
    let total_allocations = parsed["total_allocations"].as_u64().unwrap();
    let hit_rate = (cache_hits as f32 / total_allocations as f32) * 100.0;
    
    assert!(hit_rate > 50.0, "Cache hit rate too low: {:.2}%", hit_rate);
}
```

#### 3. Pipeline Integration

Tests validating the transform pipeline system:

```rust
#[test]
fn test_transform_pipeline_integration() {
    let pipeline = WasmTransformPipeline::new(true); // Enable caching
    
    // Add multiple transforms
    pipeline.add_transform("normalize").expect("Failed to add normalize");
    pipeline.add_transform("resize").expect("Failed to add resize");
    
    assert_eq!(pipeline.length(), 2);
    
    // Execute pipeline
    let input = WasmTensor::new(vec![1.0; 784], vec![28, 28]);
    let output = pipeline.execute(&input).expect("Pipeline execution failed");
    
    // Verify output shape and data integrity
    assert_eq!(output.shape(), vec![224, 224]); // After resize
    assert!(output.data().iter().all(|&x| x >= 0.0 && x <= 1.0)); // After normalize
    
    // Check pipeline statistics
    let stats = pipeline.get_stats();
    assert!(stats.contains("cache_hits"));
    assert!(stats.contains("execution_time"));
}
```

#### 4. Quality Metrics Integration

Tests ensuring data quality assessment works across different data types:

```rust
#[test]
fn test_quality_metrics_comprehensive() {
    let quality = WasmQualityMetrics::new(0.8).expect("Failed to create quality metrics");
    
    // Test with various data patterns
    let test_cases = vec![
        (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], "sequential"),
        (vec![1.0, 1.0, 1.0, 1.0, 1.0], vec![5], "uniform"),
        (vec![1.0, 100.0, 1.0, 100.0, 1.0], vec![5], "alternating"),
    ];
    
    for (data, shape, case_name) in test_cases {
        let tensor = WasmTensor::new(data, shape);
        
        let completeness = quality.completeness(&tensor).expect("Completeness failed");
        let validity = quality.validity(&tensor).expect("Validity failed");
        let consistency = quality.consistency(&tensor).expect("Consistency failed");
        let overall = quality.overall_quality(&tensor).expect("Overall quality failed");
        
        // All metrics should be between 0.0 and 1.0
        assert!(completeness >= 0.0 && completeness <= 1.0, 
               "{}: Invalid completeness {}", case_name, completeness);
        assert!(validity >= 0.0 && validity <= 1.0,
               "{}: Invalid validity {}", case_name, validity);
        assert!(consistency >= 0.0 && consistency <= 1.0,
               "{}: Invalid consistency {}", case_name, consistency);
        assert!(overall >= 0.0 && overall <= 1.0,
               "{}: Invalid overall {}", case_name, overall);
        
        // Verify quality report is valid JSON
        let report = quality.quality_report(&tensor).expect("Report failed");
        assert!(serde_json::from_str::<serde_json::Value>(&report).is_ok(),
               "{}: Invalid JSON report", case_name);
    }
}
```

## Running Tests

### Basic Test Execution

```bash
# Run all WASM integration tests
cargo test --features wasm test_wasm_integration

# Run with release optimizations
cargo test --features wasm --release test_wasm_integration

# Verbose output for debugging
cargo test --features wasm test_wasm_integration -- --nocapture
```

### Performance Testing

```bash
# Run memory pool benchmarks
cargo test --features wasm --release bench_memory_pool

# Test with different pool configurations
POOL_SIZE=50 cargo test --features wasm test_memory_efficiency
POOL_SIZE=200 cargo test --features wasm test_memory_efficiency
```

### Browser Testing

For JavaScript integration testing:

```bash
# Build WASM package
wasm-pack build --target web --out-dir pkg

# Run browser tests (requires test runner)
npm test

# Manual browser testing
python -m http.server 8000
# Open http://localhost:8000/test.html
```

## Test Data Patterns

### Standard Test Datasets

```rust
pub fn create_test_tensor(pattern: &str) -> WasmTensor {
    match pattern {
        "small" => WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        "medium" => WasmTensor::new((0..1000).map(|i| i as f32).collect(), vec![10, 100]),
        "large" => WasmTensor::new(vec![1.0; 100000], vec![316, 316]),
        "random" => {
            let mut rng = thread_rng();
            let data: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
            WasmTensor::new(data, vec![25, 40])
        },
        "anomalous" => {
            let mut data = vec![1.0; 100];
            data[50] = 100.0; // Clear anomaly
            data[75] = -50.0; // Another anomaly
            WasmTensor::new(data, vec![100])
        },
        _ => panic!("Unknown test pattern: {}", pattern),
    }
}
```

### Quality Assessment Test Data

```rust
pub fn create_quality_test_data() -> Vec<(Vec<f32>, &'static str, f32)> {
    vec![
        // (data, description, expected_quality_range_min)
        (vec![1.0, 2.0, 3.0, 4.0, 5.0], "perfect_sequence", 0.9),
        (vec![1.0, 1.0, 1.0, 1.0, 1.0], "uniform", 0.7),
        (vec![1.0, f32::NAN, 3.0, 4.0, 5.0], "with_nan", 0.3),
        (vec![1.0, 2.0, 1000.0, 4.0, 5.0], "with_outlier", 0.4),
        (vec![], "empty", 0.0),
    ]
}
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: WASM Tests
on: [push, pull_request]

jobs:
  wasm-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown
      - name: Install wasm-pack
        run: cargo install wasm-pack
      - name: Run WASM integration tests
        run: cargo test --features wasm test_wasm_integration
      - name: Build WASM package
        run: wasm-pack build --target web --out-dir pkg
      - name: Test TypeScript bindings
        run: |
          cd pkg
          npm install --dev
          npm test
```

### Performance Regression Testing

```bash
#!/bin/bash
# scripts/test-wasm-performance.sh

echo "Running WASM performance regression tests..."

# Baseline measurements
cargo test --features wasm --release bench_memory_pool 2>&1 | \
    grep "cache_hit_rate" | \
    sed 's/.*cache_hit_rate: \([0-9.]*\).*/\1/' > baseline_hit_rate.txt

# Current measurements  
cargo test --features wasm --release test_memory_efficiency 2>&1 | \
    grep "hit_rate" | \
    sed 's/.*hit_rate.*: \([0-9.]*\).*/\1/' > current_hit_rate.txt

# Compare results
baseline=$(cat baseline_hit_rate.txt)
current=$(cat current_hit_rate.txt)

if (( $(echo "$current < $baseline * 0.95" | bc -l) )); then
    echo "REGRESSION: Cache hit rate dropped from $baseline to $current"
    exit 1
else
    echo "PASS: Cache hit rate maintained at $current (baseline: $baseline)"
fi
```

## Browser Test Harness

### HTML Test Page

```html
<!DOCTYPE html>
<html>
<head>
    <title>RusTorch WASM Tests</title>
</head>
<body>
    <div id="results"></div>
    
    <script type="module">
        import init, { 
            WasmTensor, 
            WasmAdvancedMath, 
            WasmQualityMetrics,
            MemoryManager 
        } from './pkg/rustorch_wasm.js';
        
        async function runTests() {
            await init();
            
            const results = document.getElementById('results');
            
            try {
                // Initialize memory pool
                MemoryManager.init_pool(50);
                
                // Test 1: Basic operations
                const tensor = new WasmTensor([1, 2, 3, 4], [2, 2]);
                const math = new WasmAdvancedMath();
                const result = math.sinh(tensor);
                
                results.innerHTML += `<p>✅ Basic operations: ${result.data().slice(0, 2)}</p>`;
                
                // Test 2: Quality metrics
                const quality = new WasmQualityMetrics(0.8);
                const score = quality.overall_quality(tensor);
                
                results.innerHTML += `<p>✅ Quality score: ${score.toFixed(3)}</p>`;
                
                // Test 3: Memory pool statistics
                const stats = JSON.parse(MemoryManager.get_stats());
                results.innerHTML += `<p>✅ Pool hit rate: ${stats.hit_rate}%</p>`;
                
                // Cleanup
                tensor.free();
                result.free();
                math.free();
                quality.free();
                
                results.innerHTML += `<p>🎉 All tests passed!</p>`;
                
            } catch (error) {
                results.innerHTML += `<p>❌ Test failed: ${error.message}</p>`;
            }
        }
        
        runTests();
    </script>
</body>
</html>
```

### Automated Browser Testing

```javascript
// tests/browser.test.js
const { chromium } = require('playwright');

describe('WASM Browser Integration', () => {
    let browser, page;
    
    beforeAll(async () => {
        browser = await chromium.launch();
        page = await browser.newPage();
        await page.goto('file://./test.html');
    });
    
    afterAll(async () => {
        await browser.close();
    });
    
    test('WASM modules load and execute', async () => {
        // Wait for tests to complete
        await page.waitForSelector('#results');
        
        const results = await page.textContent('#results');
        expect(results).toContain('All tests passed!');
        expect(results).not.toContain('Test failed:');
    });
    
    test('Memory pool efficiency', async () => {
        const poolStats = await page.evaluate(() => {
            return MemoryManager.get_stats();
        });
        
        const stats = JSON.parse(poolStats);
        expect(stats.hit_rate).toBeGreaterThan(50);
    });
});
```

## Performance Benchmarking

### Memory Pool Benchmarks

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn bench_memory_pool_performance() {
        MemoryManager::init_pool(100);
        
        let start = Instant::now();
        
        // Simulate typical workload
        let tensors: Vec<_> = (0..1000).map(|i| {
            let size = 100 + (i % 1000); // Variable sizes
            let data = vec![i as f32; size];
            WasmTensor::new(data, vec![size])
        }).collect();
        
        let creation_time = start.elapsed();
        
        // Cleanup and measure
        let cleanup_start = Instant::now();
        drop(tensors);
        let cleanup_time = cleanup_start.elapsed();
        
        // Check pool statistics
        let stats = MemoryManager::get_stats();
        let parsed: serde_json::Value = serde_json::from_str(&stats).unwrap();
        let hit_rate = parsed["hit_rate"].as_f64().unwrap();
        
        println!("Creation time: {:?}", creation_time);
        println!("Cleanup time: {:?}", cleanup_time);
        println!("Cache hit rate: {:.2}%", hit_rate);
        
        // Performance assertions
        assert!(hit_rate > 60.0, "Cache hit rate too low: {:.2}%", hit_rate);
        assert!(creation_time < std::time::Duration::from_millis(100), 
               "Tensor creation too slow: {:?}", creation_time);
    }
}
```

### Mathematical Accuracy Tests

```rust
#[test]
fn test_mathematical_precision() {
    let math = WasmAdvancedMath::new();
    
    // Test known mathematical identities
    let test_values = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    
    for &x in &test_values {
        let tensor = WasmTensor::new(vec![x], vec![1]);
        
        // Test sinh²(x) - cosh²(x) = -1
        let sinh_val = math.sinh(&tensor).expect("sinh failed");
        let cosh_val = math.cosh(&tensor).expect("cosh failed");
        
        let sinh_sq = sinh_val.data()[0].powi(2);
        let cosh_sq = cosh_val.data()[0].powi(2);
        let identity = sinh_sq - cosh_sq;
        
        assert!((identity + 1.0).abs() < 1e-6, 
               "Mathematical identity failed for x={}: {} ≠ -1", x, identity);
        
        // Test erf(0) = 0, erf(∞) → 1
        if x == 0.0 {
            let erf_result = math.erf(&tensor).expect("erf failed");
            assert!(erf_result.data()[0].abs() < 1e-10, "erf(0) ≠ 0");
        }
    }
}
```

## Test Data Generation

### Synthetic Datasets

```rust
pub struct TestDataGenerator;

impl TestDataGenerator {
    pub fn gaussian_noise(size: usize, mean: f32, std: f32) -> Vec<f32> {
        use rand_distr::{Normal, Distribution};
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = thread_rng();
        (0..size).map(|_| normal.sample(&mut rng)).collect()
    }
    
    pub fn sine_wave(size: usize, frequency: f32, amplitude: f32) -> Vec<f32> {
        (0..size).map(|i| {
            amplitude * (2.0 * std::f32::consts::PI * frequency * i as f32 / size as f32).sin()
        }).collect()
    }
    
    pub fn anomalous_data(size: usize, anomaly_rate: f32) -> Vec<f32> {
        let mut data = Self::gaussian_noise(size, 0.0, 1.0);
        let num_anomalies = (size as f32 * anomaly_rate) as usize;
        
        let mut rng = thread_rng();
        for _ in 0..num_anomalies {
            let idx = rng.gen_range(0..size);
            data[idx] = rng.gen_range(-10.0..10.0); // Clear outlier
        }
        
        data
    }
    
    pub fn image_like_data(height: usize, width: usize, channels: usize) -> Vec<f32> {
        // Generate realistic image-like data with spatial correlation
        let size = height * width * channels;
        let mut data = vec![0.0; size];
        
        let mut rng = thread_rng();
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let idx = h * width * channels + w * channels + c;
                    // Simple spatial correlation
                    let base_value = (h + w) as f32 / (height + width) as f32;
                    data[idx] = base_value + rng.gen_range(-0.1..0.1);
                }
            }
        }
        
        data
    }
}
```

## Error Testing

### Error Condition Validation

```rust
#[test]
fn test_error_handling_consistency() {
    // Test invalid tensor shapes
    let invalid_tensor = WasmTensor::new(vec![1.0, 2.0, 3.0], vec![2, 3]); // Mismatch
    match invalid_tensor {
        Ok(_) => panic!("Should have failed with shape mismatch"),
        Err(e) => assert!(e.to_string().contains("shape")),
    }
    
    // Test quality metrics with invalid threshold
    let invalid_quality = WasmQualityMetrics::new(-1.0);
    match invalid_quality {
        Ok(_) => panic!("Should have failed with negative threshold"),
        Err(e) => assert!(e.to_string().contains("threshold")),
    }
    
    // Test anomaly detector with invalid parameters
    let invalid_detector = WasmAnomalyDetector::new(-1.0, 0);
    match invalid_detector {
        Ok(_) => panic!("Should have failed with invalid parameters"),
        Err(e) => assert!(e.to_string().contains("parameter")),
    }
}
```

## Mock and Stub Utilities

### Test Utilities

```rust
pub struct TestUtils;

impl TestUtils {
    pub fn assert_tensor_equal(a: &WasmTensor, b: &WasmTensor, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes differ");
        
        let a_data = a.data();
        let b_data = b.data();
        
        for (i, (&val_a, &val_b)) in a_data.iter().zip(b_data.iter()).enumerate() {
            assert!((val_a - val_b).abs() < tolerance,
                   "Tensors differ at index {}: {} vs {} (tolerance: {})",
                   i, val_a, val_b, tolerance);
        }
    }
    
    pub fn measure_operation_time<F, R>(operation: F) -> (R, std::time::Duration)
    where F: FnOnce() -> R {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }
    
    pub fn validate_memory_usage<F>(operation: F) -> MemoryUsageReport
    where F: FnOnce() {
        let initial_stats = MemoryManager::get_stats();
        operation();
        let final_stats = MemoryManager::get_stats();
        
        MemoryUsageReport::new(initial_stats, final_stats)
    }
}

pub struct MemoryUsageReport {
    pub allocations_delta: i64,
    pub cache_hits_delta: i64,
    pub hit_rate_change: f32,
}
```

## Integration with CI/CD

### Test Pipeline

```bash
#!/bin/bash
# scripts/run-wasm-tests.sh

set -e

echo "🧪 Running WASM integration tests..."

# Build with WASM target
echo "📦 Building WASM target..."
cargo build --features wasm --target wasm32-unknown-unknown

# Run integration tests
echo "🔬 Running integration tests..."
cargo test --features wasm test_wasm_integration

# Build WASM package for JS testing
echo "🌐 Building WASM package..."
wasm-pack build --target web --out-dir pkg

# Type check TypeScript definitions
echo "📝 Checking TypeScript definitions..."
cd pkg && npm install --dev && npx tsc --noEmit index.d.ts

# Performance benchmarks
echo "⚡ Running performance benchmarks..."
cargo test --features wasm --release bench_memory_pool

echo "✅ All WASM tests completed successfully!"
```

### Quality Gates

Define quality thresholds for CI:

```yaml
# .github/workflows/quality-gates.yml
- name: Check WASM Performance
  run: |
    OUTPUT=$(cargo test --features wasm --release bench_memory_pool 2>&1)
    HIT_RATE=$(echo "$OUTPUT" | grep -o 'hit_rate.*[0-9.]*' | grep -o '[0-9.]*$')
    
    if (( $(echo "$HIT_RATE < 70.0" | bc -l) )); then
      echo "❌ Cache hit rate too low: $HIT_RATE%"
      exit 1
    fi
    
    echo "✅ Cache hit rate acceptable: $HIT_RATE%"
```

## Testing Best Practices

### 1. Test Isolation

```rust
#[test]
fn isolated_test_example() {
    // Each test should reset the memory pool
    MemoryManager::init_pool(10);
    
    // Run test operations
    test_operations();
    
    // Explicit cleanup
    MemoryManager::gc();
}
```

### 2. Error Propagation Testing

```rust
#[test]
fn test_error_propagation() {
    let tensor = WasmTensor::new(vec![], vec![0]); // Empty tensor
    let math = WasmAdvancedMath::new();
    
    // Verify error is properly propagated
    match math.sinh(&tensor) {
        Ok(_) => panic!("Should have failed with empty tensor"),
        Err(e) => {
            assert!(e.to_string().contains("empty"));
            assert!(e.to_string().contains("tensor"));
        }
    }
}
```

### 3. Resource Cleanup Validation

```rust
#[test]
fn test_resource_cleanup() {
    let initial_stats = MemoryManager::get_stats();
    
    {
        let tensor = WasmTensor::new(vec![1.0; 1000], vec![1000]);
        let math = WasmAdvancedMath::new();
        let _result = math.tanh(&tensor);
        // Objects dropped here
    }
    
    // Force garbage collection
    MemoryManager::gc();
    
    let final_stats = MemoryManager::get_stats();
    
    // Pool should return to similar state (allowing for some caching)
    let initial: serde_json::Value = serde_json::from_str(&initial_stats).unwrap();
    let final: serde_json::Value = serde_json::from_str(&final_stats).unwrap();
    
    let pool_growth = final["small"].as_u64().unwrap() + 
                     final["medium"].as_u64().unwrap() + 
                     final["large"].as_u64().unwrap() -
                     (initial["small"].as_u64().unwrap() + 
                      initial["medium"].as_u64().unwrap() + 
                      initial["large"].as_u64().unwrap());
    
    assert!(pool_growth <= 5, "Excessive pool growth: {}", pool_growth);
}
```