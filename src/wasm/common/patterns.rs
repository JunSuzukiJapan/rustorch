//! Shared behavior patterns and utilities
//! 共有動作パターンとユーティリティ

use crate::wasm::common::{WasmError, WasmResult, WasmVersion};
use wasm_bindgen::prelude::*;

/// Standard parameter validation patterns
pub struct WasmParamValidator;

impl WasmParamValidator {
    /// Validate dimension parameters (width, height)
    pub fn validate_dimensions(width: usize, height: usize, param_name: &str) -> WasmResult<()> {
        if width == 0 || height == 0 {
            Err(WasmError::invalid_param(
                param_name,
                format!("{}x{}", width, height),
                "dimensions must be non-zero",
            ))
        } else {
            Ok(())
        }
    }

    /// Validate percentage range (0-100)
    pub fn validate_percentage_range(value: f32, param_name: &str) -> WasmResult<()> {
        if value < 0.0 || value > 100.0 {
            Err(WasmError::invalid_range(param_name, value, 0.0, 100.0))
        } else {
            Ok(())
        }
    }

    /// Validate probability range (0-1)
    pub fn validate_probability_range(value: f32, param_name: &str) -> WasmResult<()> {
        if value < 0.0 || value > 1.0 {
            Err(WasmError::invalid_range(param_name, value, 0.0, 1.0))
        } else {
            Ok(())
        }
    }

    /// Validate positive value
    pub fn validate_positive(value: f32, param_name: &str) -> WasmResult<()> {
        if value <= 0.0 {
            Err(WasmError::invalid_param(
                param_name,
                value,
                "must be positive",
            ))
        } else {
            Ok(())
        }
    }

    /// Validate non-zero usize
    pub fn validate_non_zero_usize(value: usize, param_name: &str) -> WasmResult<()> {
        if value == 0 {
            Err(WasmError::invalid_param(
                param_name,
                value,
                "must be non-zero",
            ))
        } else {
            Ok(())
        }
    }

    /// Validate array length
    pub fn validate_array_length(
        length: usize,
        min_length: usize,
        param_name: &str,
    ) -> WasmResult<()> {
        if length < min_length {
            Err(WasmError::insufficient_data(param_name, min_length, length))
        } else {
            Ok(())
        }
    }
}

/// Standard naming patterns for WASM operations
pub struct WasmNaming;

impl WasmNaming {
    /// Generate standardized operation name
    pub fn operation_name(base_name: &str, params: &[(&str, String)]) -> String {
        if params.is_empty() {
            base_name.to_string()
        } else {
            let param_str = params
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            format!("{}({})", base_name, param_str)
        }
    }

    /// Generate transformation name with dimensions
    pub fn transform_with_dims(base_name: &str, width: usize, height: usize) -> String {
        format!("{}({}x{})", base_name, width, height)
    }

    /// Generate analysis name with configuration
    pub fn analysis_with_config(base_name: &str, config: &str) -> String {
        format!("{}[{}]", base_name, config)
    }

    /// Generate detector name with threshold
    pub fn detector_with_threshold(base_name: &str, threshold: f32) -> String {
        format!("{}(threshold={:.3})", base_name, threshold)
    }
}

/// Standard configuration patterns
pub struct WasmConfig;

impl WasmConfig {
    /// Default quality threshold
    pub const DEFAULT_QUALITY_THRESHOLD: f32 = 0.8;

    /// Default anomaly threshold (Z-score)
    pub const DEFAULT_ANOMALY_THRESHOLD: f32 = 2.0;

    /// Default memory pool size
    pub const DEFAULT_POOL_SIZE: usize = 100;

    /// Default batch size
    pub const DEFAULT_BATCH_SIZE: usize = 32;

    /// Default window size for streaming operations
    pub const DEFAULT_WINDOW_SIZE: usize = 50;

    /// Create standard configuration JSON
    pub fn standard_config() -> String {
        format!(
            "{{\"quality_threshold\":{},\"anomaly_threshold\":{},\"pool_size\":{},\"batch_size\":{},\"window_size\":{}}}",
            Self::DEFAULT_QUALITY_THRESHOLD,
            Self::DEFAULT_ANOMALY_THRESHOLD,
            Self::DEFAULT_POOL_SIZE,
            Self::DEFAULT_BATCH_SIZE,
            Self::DEFAULT_WINDOW_SIZE
        )
    }
}

/// Performance monitoring utilities
pub struct WasmProfiler {
    start_time: f64,
    operation_count: usize,
}

impl WasmProfiler {
    /// Start new profiling session
    pub fn start() -> Self {
        Self {
            start_time: js_sys::Date::now(),
            operation_count: 0,
        }
    }

    /// Record an operation
    pub fn record_operation(&mut self) {
        self.operation_count += 1;
    }

    /// Get performance summary
    pub fn summary(&self) -> String {
        let elapsed = js_sys::Date::now() - self.start_time;
        let ops_per_sec = if elapsed > 0.0 {
            (self.operation_count as f64 * 1000.0) / elapsed
        } else {
            0.0
        };

        format!(
            "{{\"elapsed_ms\":{:.2},\"operations\":{},\"ops_per_second\":{:.2}}}",
            elapsed, self.operation_count, ops_per_sec
        )
    }
}

/// Macro for implementing standard WASM operation boilerplate
#[macro_export]
macro_rules! impl_wasm_standard {
    ($type:ty, $name:expr, $module:expr) => {
        impl crate::wasm::common::WasmOperation for $type {
            fn name(&self) -> String {
                $name.to_string()
            }
        }

        impl crate::wasm::common::WasmVersioned for $type {
            fn module_version() -> String {
                crate::wasm::common::WasmVersion::module_version($module)
            }
        }
    };
}

/// Macro for implementing transform trait with validation
#[macro_export]
macro_rules! impl_wasm_transform {
    ($type:ty, $name:expr, $module:expr) => {
        impl_wasm_standard!($type, $name, $module);

        impl crate::wasm::common::WasmTransform for $type {
            fn apply(
                &self,
                tensor: &crate::wasm::tensor::WasmTensor,
            ) -> crate::wasm::common::WasmResult<crate::wasm::tensor::WasmTensor> {
                self.apply(tensor)
            }
        }
    };
}

/// Macro for implementing analyzer trait
#[macro_export]
macro_rules! impl_wasm_analyzer {
    ($type:ty, $name:expr, $module:expr, $analysis_type:expr) => {
        impl_wasm_standard!($type, $name, $module);

        impl crate::wasm::common::WasmAnalyzer for $type {
            fn analyze(
                &self,
                tensor: &crate::wasm::tensor::WasmTensor,
            ) -> crate::wasm::common::WasmResult<String> {
                self.basic_stats(tensor)
            }

            fn analysis_type(&self) -> &'static str {
                $analysis_type
            }
        }
    };
}
