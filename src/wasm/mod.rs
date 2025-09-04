//! WebAssembly bindings for RusTorch - Reorganized module structure
//! RusTorch用WebAssemblyバインディング - 再編成されたモジュール構造

// Core infrastructure modules
#[cfg(feature = "wasm")]
pub mod core;

// Data processing and statistical modules
#[cfg(feature = "wasm")]
pub mod data;

// Mathematical computation modules
#[cfg(feature = "wasm")]
pub mod math;

// Machine learning modules
#[cfg(feature = "wasm")]
pub mod ml;

// WebGPU acceleration modules
#[cfg(feature = "webgpu")]
pub mod gpu;

// Storage and browser modules
#[cfg(feature = "wasm")]
pub mod storage_modules;

// Standalone modules that don't fit in above categories
#[cfg(feature = "wasm")]
pub mod optimized; // Legacy optimized operations
#[cfg(feature = "wasm")]
pub mod tensor; // Core tensor operations
#[cfg(feature = "wasm")]
pub mod vision; // Computer vision functions

// Additional utilities
#[cfg(feature = "wasm")]
pub mod tensor_ops; // Advanced tensor operations
#[cfg(feature = "wasm")]
pub mod utilities; // Phase 8 tensor utilities (conditional, indexing, statistics, advanced)

// Common utilities
#[cfg(feature = "wasm")]
pub mod common; // Shared utilities and abstractions

// Additional specialized modules
#[cfg(feature = "wasm")]
pub mod anomaly_detection; // Anomaly detection and monitoring

// Re-exports for backward compatibility
#[cfg(feature = "wasm")]
pub use core::bindings;
#[cfg(feature = "wasm")]
pub use core::interop;
#[cfg(feature = "wasm")]
pub use core::memory;
#[cfg(feature = "wasm")]
pub use core::runtime;
#[cfg(feature = "wasm")]
pub use data::distributions;
#[cfg(feature = "wasm")]
pub use data::metrics;
#[cfg(feature = "wasm")]
pub use data::preprocessing;
#[cfg(feature = "wasm")]
pub use data::quality_metrics;
#[cfg(feature = "webgpu")]
pub use gpu::backend as webgpu_backend;
#[cfg(feature = "webgpu")]
pub use gpu::simple as webgpu_simple;
#[cfg(feature = "webgpu")]
pub use gpu::tensor as webgpu_tensor;
#[cfg(feature = "wasm")]
pub use math::linalg;
#[cfg(feature = "wasm")]
pub use math::signal;
#[cfg(feature = "wasm")]
pub use math::special;
#[cfg(feature = "wasm")]
pub use ml::activation;
#[cfg(feature = "wasm")]
pub use ml::autograd as autograd_simplified;
#[cfg(feature = "wasm")]
pub use ml::loss;
#[cfg(feature = "wasm")]
pub use ml::normalization;
#[cfg(feature = "wasm")]
pub use ml::optimizers;
#[cfg(feature = "wasm")]
pub use storage_modules::browser;
#[cfg(feature = "wasm")]
pub use storage_modules::persistence as storage;

// Renamed modules for consistency
#[cfg(feature = "wasm")]
pub use data::transforms as data_transforms;
#[cfg(feature = "wasm")]
pub use math::advanced as advanced_math;

// Enhanced module aliases (for backward compatibility with enhanced versions)
#[cfg(feature = "wasm")]
pub use data::distributions as distributions_enhanced;
#[cfg(feature = "wasm")]
pub use math::special as special_enhanced;
#[cfg(feature = "wasm")]
pub use ml::optimizers as optimizer_enhanced;
