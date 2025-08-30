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
pub mod vision; // Vision and image processing functions

// Common utilities (v0.5.2+)
#[cfg(feature = "wasm")]
pub mod common; // Shared utilities and abstractions

// Enhanced WASM features (v0.5.2+)
#[cfg(feature = "wasm")]
pub mod data_transforms; // Advanced data preprocessing
#[cfg(feature = "wasm")]
pub mod quality_metrics; // Data quality assessment
#[cfg(feature = "wasm")]
pub mod advanced_math; // Advanced mathematical functions
#[cfg(feature = "wasm")]
pub mod anomaly_detection; // Anomaly detection and monitoring
