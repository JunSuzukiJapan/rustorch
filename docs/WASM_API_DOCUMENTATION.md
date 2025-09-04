# 🌐 WebAssembly API Documentation

> 📚 **Main Documentation**: [API Documentation](API_DOCUMENTATION.md)  
> 🔗 **Related Guides**: [WASM Enhancement Roadmap](WASM_API_Enhancement_Roadmap.md)

This document contains comprehensive WebAssembly (WASM) API reference for browser-based machine learning with RusTorch.

## 🚀 Implementation Status

**✅ Completed Phases** (95% implementation value):
- **Phase 1** (60%): Special functions, distributions, FFT, loss functions
- **Phase 2** (25%): Computer vision, simplified autograd, browser storage
- **Phase 3** (10%): Linear algebra with WASM constraints

**🌟 Key Features**:
- Browser-compatible tensor operations
- Neural network layers with simplified autograd
- Computer vision operations (Harris corners, morphology, LBP)
- Model persistence (IndexedDB, LocalStorage, compression)
- Linear algebra (QR, LU, SVD decompositions, eigenvalues)
- WebGPU acceleration for Chrome browsers
- JavaScript interoperability and type conversion

## Table of Contents

- [🌐 WebAssembly Support](#-webassembly-support)
- [🧮 WASM Tensor Operations](#-wasm-tensor-operations)
- [🧠 WASM Neural Network Layers](#-wasm-neural-network-layers)
- [🌍 Browser Integration](#-browser-integration)
- [⚡ WebGPU Acceleration](#-webgpu-acceleration)
- [🔧 Advanced WASM Features](#-advanced-wasm-features)
- [💾 Memory Management](#-memory-management)
- [📡 Signal Processing](#-signal-processing)
- [🔧 WASM Utilities and Helpers](#-wasm-utilities-and-helpers)
- [🔄 Backward Compatibility](#-backward-compatibility)

## 🌐 WebAssembly Support

### WASM Module Structure

```
src/
└── wasm/                # WebAssembly bindings
    ├── core/           # Core tensor operations
    ├── data/           # Data distributions and sampling
    ├── math/           # Mathematical functions and FFT
    ├── ml/             # Machine learning components
    ├── vision/         # Computer vision operations
    ├── gpu/            # WebGPU integration
    └── storage/        # Browser storage and persistence
```

### Feature Flag

Include WASM support in your `Cargo.toml`:

```toml
[features]
wasm = ["wasm-bindgen", "wasm-bindgen-futures", "web-sys", "js-sys", "console_error_panic_hook"]
webgpu = ["wasm", "dep:wgpu", "dep:wgpu-hal", "dep:wgpu-core", "dep:wgpu-types"]
```

### WASM Bindings

```rust
use rustorch::wasm::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<f32>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor {
            inner: Tensor::from_vec(data.to_vec(), shape.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn add(&self, other: &WasmTensor) -> WasmTensor {
        WasmTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f32> {
        self.inner.to_vec()
    }
}

// Neural network for WASM
#[wasm_bindgen]
pub struct WasmModel {
    model: Sequential<f32>,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmModel {
        let model = Sequential::<f32>::new()
            .add_layer(Box::new(Linear::<f32>::new(2, 10)))
            .add_activation(Box::new(ReLU::<f32>::new()))
            .add_layer(Box::new(Linear::<f32>::new(10, 1)));
        
        WasmModel { model }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let input_tensor = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
        let output = self.model.forward(&input_tensor);
        output.to_vec()
    }
}
```

## 🧮 WASM Tensor Operations

### Basic Tensor Operations

```rust
use rustorch::wasm::{WasmTensor, WasmLinear};

// Create WASM-compatible tensor
let data = vec![1.0, 2.0, 3.0, 4.0];
let shape = vec![2, 2];
let wasm_tensor = WasmTensor::new(data, shape);

// Basic tensor operations
let tensor_a = WasmTensor::new(vec![1.0, 2.0], vec![2, 1]);
let tensor_b = WasmTensor::new(vec![3.0, 4.0], vec![2, 1]);
let result = tensor_a.add(&tensor_b)?;

// Matrix operations
let result = tensor_a.matmul(&tensor_b)?;
let transposed = tensor_a.transpose();

// Element-wise operations
let squared = tensor_a.square();
let sqrt_result = tensor_a.sqrt();
let sum = tensor_a.sum();
let mean = tensor_a.mean();
```

## 🧠 WASM Neural Network Layers

### Linear Layers and Activations

```rust
use rustorch::wasm::{WasmLinear, WasmActivation};

// Linear layer with bias
let linear = WasmLinear::new(
    in_features: 784,
    out_features: 10,
    bias: true
);

// Custom weight initialization
let custom_weights = vec![/* weight values */];
let custom_bias = Some(vec![/* bias values */]);
let linear_custom = WasmLinear::with_weights(784, 10, custom_weights, custom_bias)?;

// Forward pass
let input = vec![/* input data */];
let output = linear.forward(input, batch_size: 1)?;

// Activation functions
let activated = WasmActivation::relu(output);
let sigmoid_result = WasmActivation::sigmoid(input);
let tanh_result = WasmActivation::tanh(input);
let leaky_relu = WasmActivation::leaky_relu(input, alpha: 0.01);
```

## 🌍 Browser Integration

### Storage and Persistence

```rust
use rustorch::wasm::{BrowserStorage, BrowserCanvas, PerformanceMonitor};

// Browser storage for model persistence
let storage = BrowserStorage::new();
storage.save_tensor("model_weights", &wasm_tensor)?;
let loaded_tensor = storage.load_tensor("model_weights")?;

// Canvas rendering for visualizations
let canvas = BrowserCanvas::new("canvas-id")?;
canvas.draw_tensor(&wasm_tensor)?;
canvas.draw_heatmap(&activation_map, width: 256, height: 256)?;

// Performance monitoring
let monitor = PerformanceMonitor::new();
monitor.start_timer("inference");
// ... model inference ...
let elapsed = monitor.end_timer("inference")?;
```

## ⚡ WebGPU Acceleration

### Chrome-Optimized GPU Acceleration

```rust
use rustorch::wasm::WebGPUSimple;

// Initialize WebGPU for Chrome
let mut webgpu = WebGPUSimple::new();
webgpu.initialize().await?;

// Check GPU capabilities
let info = webgpu.get_device_info()?;
let supports_compute = webgpu.supports_compute_shaders()?;

// Basic GPU operations
let gpu_result = webgpu.matrix_multiply(&matrix_a, &matrix_b).await?;
let gpu_conv = webgpu.convolution_2d(&input, &kernel).await?;
```

## 🔧 Advanced WASM Features

### Enhanced Components

```rust
use rustorch::wasm::{
    WasmOptimizer, DataTransforms, QualityMetrics, AnomalyDetection
};

// Enhanced optimizers
let optimizer = WasmOptimizer::adam(learning_rate: 0.001, beta1: 0.9, beta2: 0.999)?;
optimizer.step(&gradients, &mut parameters)?;

// Data preprocessing
let transforms = DataTransforms::new();
let normalized = transforms.normalize(&data, mean: 0.0, std: 1.0)?;
let augmented = transforms.random_rotation(&image_data, angle: 30.0)?;

// Quality assessment
let metrics = QualityMetrics::new();
let data_quality = metrics.assess_data_quality(&dataset)?;
let model_quality = metrics.evaluate_model_performance(&predictions, &targets)?;

// Anomaly detection
let detector = AnomalyDetection::new();
let anomalies = detector.detect_outliers(&data, threshold: 2.0)?;
let model_drift = detector.detect_model_drift(&current_output, &baseline_output)?;
```

## 💾 Memory Management

### Efficient Allocation and Cleanup

```rust
use rustorch::wasm::{WasmMemoryManager, WasmMemoryPool};

// Memory pool for efficient allocation
let memory_pool = WasmMemoryPool::new(max_size_mb: 256);
let tensor_memory = memory_pool.allocate_tensor(&shape)?;

// Memory monitoring
let manager = WasmMemoryManager::new();
let usage = manager.get_memory_usage();
manager.cleanup_unused_tensors();

// Garbage collection optimization
manager.force_gc_if_needed(threshold: 0.8)?;
```

## 📡 Signal Processing

### FFT and Filtering Operations

```rust
use rustorch::wasm::WasmSignal;

// FFT operations
let signal = WasmSignal::new();
let fft_result = signal.fft(&time_domain_data)?;
let ifft_result = signal.ifft(&frequency_domain_data)?;

// Filtering operations
let filtered = signal.lowpass_filter(&noisy_signal, cutoff: 0.3)?;
let convolved = signal.convolve(&signal_a, &signal_b)?;
```

## 🔧 WASM Utilities and Helpers

### JavaScript Interoperability

```rust
use rustorch::wasm::{WasmUtilities, WasmInterop};

// JavaScript interoperability
let interop = WasmInterop::new();
let js_array = interop.tensor_to_js_array(&wasm_tensor)?;
let wasm_tensor = interop.js_array_to_tensor(&js_array, &shape)?;

// Type conversions and validation
let utilities = WasmUtilities::new();
let validated_data = utilities.validate_and_convert(&input_data)?;
let shape_valid = utilities.validate_shape(&shape, max_dims: 4)?;
```

## 🎨 Computer Vision Operations

```rust
use rustorch::wasm::WasmVision;

// Harris corner detection for feature extraction
let corners = WasmVision::harris_corner_detection(&image, threshold: 0.01, k: 0.04)?;

// Morphological operations for shape analysis
let opened = WasmVision::morphological_opening(&binary_image, kernel_size: 3)?;
let closed = WasmVision::morphological_closing(&binary_image, kernel_size: 3)?;

// Local Binary Patterns for texture analysis
let lbp = WasmVision::local_binary_patterns(&grayscale, radius: 1)?;

// Standard image preprocessing
let resized = WasmVision::resize(&image, 224, 224, channels: 3)?;
let normalized = WasmVision::normalize(&image, &mean, &std, channels: 3)?;
```

## 🔬 Mathematical Functions

```rust
use rustorch::wasm::{WasmSpecial, WasmDistributions, WasmFFT};

// Special mathematical functions (Gamma, Bessel, Error functions)
let gamma_result = WasmSpecial::gamma_batch(&[1.5, 2.0, 2.5]);
let bessel_result = WasmSpecial::bessel_i_batch(0, &[0.5, 1.0, 1.5]);
let erf_result = WasmSpecial::erf_batch(&[-1.0, 0.0, 1.0]);

// Statistical distributions
let distributions = WasmDistributions::new();
let normal_samples = distributions.normal_sample_batch(100, 0.0, 1.0);
let uniform_samples = distributions.uniform_sample_batch(50, 0.0, 1.0);

// Fast Fourier Transform
let fft = WasmFFT::new();
let fft_result = fft.fft(&time_domain_signal)?;
let power_spectrum = fft.power_spectrum(&signal)?;
```

## 🧮 Linear Algebra Operations

```rust
use rustorch::wasm::WasmLinearAlgebra;

// Matrix decompositions (optimized for WASM, limited to 500x500)
let linalg = WasmLinearAlgebra::new();

// QR decomposition for least squares
let (q, r) = linalg.qr_decomposition(&matrix)?;

// LU decomposition for linear systems
let (l, u, p) = linalg.lu_decomposition(&matrix)?;

// SVD for dimensionality reduction
let (u, s, vt) = linalg.svd(&matrix, compute_uv: true)?;

// Eigenvalue computation for principal component analysis
let eigenvals = linalg.eigenvalues(&symmetric_matrix)?;
let (eigenvals, eigenvecs) = linalg.eigenvalues_vectors(&matrix)?;

// Matrix analysis functions
let det = linalg.determinant(&matrix)?;
let rank = linalg.rank(&matrix, tolerance: 1e-10)?;
let cond = linalg.condition_number(&matrix)?;
```

## 🏪 Advanced Model Storage

```rust
use rustorch::wasm::{WasmModelStorage, WasmModelCompression, WasmProgressTracker};

// Enhanced browser storage with compression and chunking
let storage = WasmModelStorage::new();

// Save with automatic compression and progress tracking
storage.save_model_compressed("my_model", &model_data, chunk_size: 1024)?;

// Load with progress callbacks
let progress_tracker = WasmProgressTracker::new();
let model = storage.load_model_with_progress("my_model", &progress_tracker)?;

// Advanced compression for large models
let compressor = WasmModelCompression::new();
let compressed = compressor.lz4_compress(&large_tensor_data)?;
let decompressed = compressor.lz4_decompress(&compressed_data)?;

// Storage management
let available_space = storage.get_available_space()?;
storage.cleanup_old_models(days: 7)?;
```

## 📈 Performance Optimization

```rust
use rustorch::wasm::{WasmPerformance, WasmOptimization};

// Performance monitoring and optimization
let perf = WasmPerformance::new();
perf.start_profiling("model_inference");

// Optimized operations for WASM constraints
let optimizer = WasmOptimization::new();
optimizer.enable_simd_if_available();
optimizer.set_memory_limit(256); // MB

// Model inference with optimization
let optimized_result = perf.time_operation(|| {
    model.predict(&input_data)
});

let profile = perf.end_profiling("model_inference")?;
console::log_1(&format!("Inference time: {:.2}ms", profile.elapsed_ms).into());
```

## 🔄 Backward Compatibility

All existing APIs remain fully functional. The new builder pattern and enhanced operations are additive features that don't break existing code.

## 📖 Documentation Generation

Generate complete WASM-specific documentation:

```bash
cargo doc --open --no-deps --features "wasm,webgpu"
```

## 🔗 Related Documentation

- **[Main API Documentation](API_DOCUMENTATION.md)** - Complete RusTorch API reference
- **[WASM Enhancement Roadmap](WASM_API_Enhancement_Roadmap.md)** - Implementation phases and progress
- **[Getting Started Guide](getting-started.md)** - Basic usage examples
- **[Examples](examples.md)** - WebAssembly usage examples

---

> 📝 This document extracts WebAssembly-specific content from the main API documentation for focused browser development reference.