//! WebAssembly bindings for RusTorch
//! RusTorch用WebAssemblyバインディング

#[cfg(feature = "wasm")]
pub mod bindings;
#[cfg(feature = "wasm")]
pub mod browser;
#[cfg(feature = "wasm")]
pub mod interop;
#[cfg(feature = "wasm")]
pub mod optimized;
#[cfg(feature = "wasm")]
pub mod tensor;

// New WASM-compatible modules
#[cfg(feature = "wasm")]
pub mod activation; // Neural network activation functions
#[cfg(feature = "wasm")]
pub mod distributions; // Statistical distributions
#[cfg(feature = "wasm")]
pub mod loss; // Loss functions
#[cfg(feature = "wasm")]
pub mod memory; // Memory management
#[cfg(feature = "wasm")]
pub mod optimizer;
#[cfg(feature = "wasm")]
pub mod runtime; // Runtime utilities
#[cfg(feature = "wasm")]
pub mod signal; // Signal processing functions
#[cfg(feature = "wasm")]
pub mod special; // Special mathematical functions // Optimization algorithms

// Additional WASM ML utilities
#[cfg(feature = "wasm")]
pub mod metrics; // Model evaluation metrics
#[cfg(feature = "wasm")]
pub mod normalization; // Normalization layers (BatchNorm, LayerNorm, GroupNorm)
#[cfg(feature = "wasm")]
pub mod preprocessing; // Data preprocessing utilities
#[cfg(feature = "wasm")]
pub mod tensor_ops; // Advanced tensor operations
#[cfg(feature = "wasm")]
pub mod utilities; // Phase 8 tensor utilities (conditional, indexing, statistics, advanced)
#[cfg(feature = "wasm")]
pub mod vision; // Vision and image processing functions

// Common utilities (v0.5.10+)
#[cfg(feature = "wasm")]
pub mod common; // Shared utilities and abstractions

// Enhanced WASM features (v0.5.10+)
#[cfg(feature = "wasm")]
pub mod advanced_math; // Advanced mathematical functions
#[cfg(feature = "wasm")]
pub mod anomaly_detection;
#[cfg(feature = "wasm")]
pub mod data_transforms; // Advanced data preprocessing
#[cfg(feature = "wasm")]
pub mod quality_metrics; // Data quality assessment // Anomaly detection and monitoring

// Enhanced WASM implementations (v0.5.10+)
#[cfg(feature = "wasm")]
pub mod autograd_simplified;
#[cfg(feature = "wasm")]
pub mod distributions_enhanced; // Enhanced statistical distributions with full API
#[cfg(feature = "wasm")]
pub mod optimizer_enhanced; // Production-ready optimizers for small-scale models
#[cfg(feature = "wasm")]
pub mod special_enhanced; // Complete special functions with performance optimization // Simplified autograd for single-threaded WASM environment

// WebGPU backend for Chrome browser acceleration (v0.5.10+)
// #[cfg(feature = "webgpu")]
// pub mod webgpu_backend; // Chrome-optimized WebGPU backend with compute shaders (temporarily disabled)
// #[cfg(feature = "webgpu")]
// pub mod webgpu_tensor; // WebGPU tensor operations integration with RusTorch API (temporarily disabled)
#[cfg(feature = "webgpu")]
pub mod webgpu_simple; // Simplified WebGPU implementation for basic operations

// Phase 2 WASM enhancements
#[cfg(feature = "wasm")]
pub mod storage; // Browser model persistence and storage utilities

// Phase 3 WASM enhancements  
#[cfg(feature = "wasm")]
pub mod linalg; // Basic linear algebra (BLAS-free, small matrices only)
