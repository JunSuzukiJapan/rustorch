//! Common traits for WASM operations
//! WASM操作の共通トレイト

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use crate::wasm::common::error::{WasmError, WasmResult};
use js_sys::Array;

/// Base trait for all WASM operations
pub trait WasmOperation {
    /// Get the name of this operation
    fn name(&self) -> String;
    
    /// Get version information for this operation
    fn version(&self) -> String {
        crate::wasm::common::WasmVersion::module_version(&self.name())
    }
    
    /// Validate operation parameters (optional override)
    fn validate_params(&self) -> WasmResult<()> {
        Ok(())
    }
}

/// Trait for operations that transform tensors
pub trait WasmTransform: WasmOperation {
    /// Apply transformation to a tensor
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor>;
    
    /// Check if this transform can be applied to the given tensor
    fn can_apply(&self, tensor: &WasmTensor) -> WasmResult<()> {
        tensor.validate_non_empty()
    }
    
    /// Get transformation parameters as JSON
    fn get_params_json(&self) -> String {
        format!("{{\"name\":\"{}\"}}", self.name())
    }
}

/// Trait for operations that analyze tensors without modification
pub trait WasmAnalyzer: WasmOperation {
    /// Analyze tensor and return results
    fn analyze(&self, tensor: &WasmTensor) -> WasmResult<String>;
    
    /// Get analysis type
    fn analysis_type(&self) -> &'static str;
    
    /// Check if tensor is suitable for this analysis
    fn can_analyze(&self, tensor: &WasmTensor) -> WasmResult<()> {
        tensor.validate_non_empty()
    }
}

/// Trait for operations that detect patterns or anomalies
pub trait WasmDetector: WasmOperation {
    /// Detect patterns in tensor data
    fn detect(&mut self, tensor: &WasmTensor) -> WasmResult<Array>;
    
    /// Get detector configuration
    fn get_config(&self) -> String;
    
    /// Reset detector state
    fn reset(&mut self);
    
    /// Update detector parameters
    fn update_params(&mut self, params: &str) -> WasmResult<()>;
}

/// Trait for mathematical operations on tensors
pub trait WasmMathOperation: WasmOperation {
    /// Apply mathematical operation to single tensor
    fn apply_unary(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor>;
    
    /// Apply mathematical operation to two tensors (optional)
    fn apply_binary(&self, left: &WasmTensor, right: &WasmTensor) -> WasmResult<WasmTensor> {
        Err(WasmError::invalid_param("operation", self.name(), "binary operation not supported"))
    }
    
    /// Check if operation supports in-place modification
    fn supports_inplace(&self) -> bool {
        false
    }
}

/// Trait for statistical computations
pub trait WasmStatistical: WasmOperation {
    /// Calculate statistics for tensor
    fn calculate_stats(&self, tensor: &WasmTensor) -> WasmResult<String>;
    
    /// Get statistical summary
    fn get_summary(&self, tensor: &WasmTensor) -> WasmResult<String> {
        self.calculate_stats(tensor)
    }
    
    /// Check if sufficient data available
    fn has_sufficient_data(&self, tensor: &WasmTensor) -> bool {
        !tensor.data().is_empty()
    }
}

/// Trait for operations that work with image data
pub trait WasmImageOperation: WasmTransform {
    /// Validate that tensor represents valid image data
    fn validate_image(&self, tensor: &WasmTensor) -> WasmResult<(usize, usize)> {
        crate::wasm::common::WasmValidator::validate_image_tensor(tensor)
    }
    
    /// Get expected output dimensions
    fn output_dimensions(&self, input_dims: (usize, usize)) -> (usize, usize);
    
    /// Check if operation preserves image properties
    fn preserves_channels(&self) -> bool {
        true
    }
}

/// Trait for quality assessment operations
pub trait WasmQualityAssessment: WasmAnalyzer {
    /// Calculate quality score (0-100)
    fn quality_score(&self, tensor: &WasmTensor) -> WasmResult<f32>;
    
    /// Get quality threshold
    fn quality_threshold(&self) -> f32;
    
    /// Check if data meets quality requirements
    fn meets_quality_threshold(&self, tensor: &WasmTensor) -> WasmResult<bool> {
        let score = self.quality_score(tensor)?;
        Ok(score >= self.quality_threshold())
    }
    
    /// Get detailed quality report
    fn detailed_report(&self, tensor: &WasmTensor) -> WasmResult<String> {
        self.analyze(tensor)
    }
}

/// Trait for versioned WASM modules
pub trait WasmVersioned {
    /// Get module version
    fn module_version() -> String;
    
    /// Get compatibility information
    fn compatibility_version() -> String {
        Self::module_version()
    }
    
    /// Check if compatible with given version
    fn is_compatible_with(version: &str) -> bool {
        version == &Self::module_version()
    }
}

/// Convenience macro for implementing basic WasmOperation
#[macro_export]
macro_rules! impl_wasm_operation {
    ($type:ty, $name:expr) => {
        impl WasmOperation for $type {
            fn name(&self) -> String {
                $name.to_string()
            }
        }
    };
}

/// Convenience macro for implementing WasmVersioned
#[macro_export]
macro_rules! impl_wasm_versioned {
    ($type:ty, $module_name:expr) => {
        impl WasmVersioned for $type {
            fn module_version() -> String {
                crate::wasm::common::WasmVersion::module_version($module_name)
            }
        }
    };
}