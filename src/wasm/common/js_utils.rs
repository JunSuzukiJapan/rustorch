//! Standardized JavaScript interop utilities
//! 標準化されたJavaScript相互運用ユーティリティ

use wasm_bindgen::prelude::*;
use js_sys::{Array, Object, Reflect};
use crate::wasm::common::error::{WasmError, WasmResult};

/// Builder for JavaScript objects with error handling
pub struct JsObjectBuilder {
    obj: Object,
}

impl JsObjectBuilder {
    /// Create new object builder
    pub fn new() -> Self {
        Self { obj: Object::new() }
    }
    
    /// Set number property
    pub fn set_f64(self, key: &str, value: f64) -> WasmResult<Self> {
        Reflect::set(&self.obj, &key.into(), &JsValue::from_f64(value))
            .map_err(|_| WasmError::invalid_param("property", key, "failed to set"))?;
        Ok(self)
    }
    
    /// Set f32 property
    pub fn set_f32(self, key: &str, value: f32) -> WasmResult<Self> {
        self.set_f64(key, value as f64)
    }
    
    /// Set integer property
    pub fn set_usize(self, key: &str, value: usize) -> WasmResult<Self> {
        self.set_f64(key, value as f64)
    }
    
    /// Set string property
    pub fn set_string(self, key: &str, value: &str) -> WasmResult<Self> {
        Reflect::set(&self.obj, &key.into(), &JsValue::from_str(value))
            .map_err(|_| WasmError::invalid_param("property", key, "failed to set"))?;
        Ok(self)
    }
    
    /// Set boolean property
    pub fn set_bool(self, key: &str, value: bool) -> WasmResult<Self> {
        Reflect::set(&self.obj, &key.into(), &JsValue::from_bool(value))
            .map_err(|_| WasmError::invalid_param("property", key, "failed to set"))?;
        Ok(self)
    }
    
    /// Build final object
    pub fn build(self) -> Object {
        self.obj
    }
    
    /// Build and convert to JsValue
    pub fn into_js_value(self) -> JsValue {
        self.obj.into()
    }
}

/// Utility for creating JavaScript arrays
pub struct JsArrayBuilder {
    array: Array,
}

impl JsArrayBuilder {
    /// Create new array builder
    pub fn new() -> Self {
        Self { array: Array::new() }
    }
    
    /// Create with known capacity hint
    pub fn with_capacity(capacity: usize) -> Self {
        let array = Array::new();
        // JavaScript arrays don't have pre-allocation, but we can hint
        array.set_length(capacity as u32);
        array.set_length(0); // Reset to empty but with potential optimization
        Self { array }
    }
    
    /// Push f64 value
    pub fn push_f64(self, value: f64) -> Self {
        self.array.push(&JsValue::from_f64(value));
        self
    }
    
    /// Push f32 value
    pub fn push_f32(self, value: f32) -> Self {
        self.push_f64(value as f64)
    }
    
    /// Push string value
    pub fn push_string(self, value: &str) -> Self {
        self.array.push(&JsValue::from_str(value));
        self
    }
    
    /// Push object
    pub fn push_object(self, obj: Object) -> Self {
        self.array.push(&obj);
        self
    }
    
    /// Build final array
    pub fn build(self) -> Array {
        self.array
    }
}

/// Version string management
pub struct WasmVersion;

impl WasmVersion {
    /// Get standardized version string for module
    pub fn module_version(module_name: &str) -> String {
        format!("RusTorch WASM {} v{}", module_name, env!("CARGO_PKG_VERSION"))
    }
    
    /// Get core WASM version
    pub fn core_version() -> String {
        format!("RusTorch WASM Core v{}", env!("CARGO_PKG_VERSION"))
    }
}

/// Utility for consistent JSON formatting
pub struct JsonFormatter;

impl JsonFormatter {
    /// Format statistics as JSON
    pub fn stats_json(
        mean: f32, 
        std_dev: f32, 
        min: f32, 
        max: f32, 
        count: usize
    ) -> String {
        format!(
            "{{\"mean\":{:.6},\"std\":{:.6},\"min\":{:.6},\"max\":{:.6},\"count\":{}}}",
            mean, std_dev, min, max, count
        )
    }
    
    /// Format quality metrics as JSON
    pub fn quality_json(
        completeness: f32,
        validity: f32,
        consistency: f32,
        uniqueness: f32,
        overall: f32
    ) -> String {
        format!(
            "{{\"completeness\":{:.2},\"validity\":{:.2},\"consistency\":{:.2},\"uniqueness\":{:.2},\"overall_quality\":{:.2}}}",
            completeness, validity, consistency, uniqueness, overall
        )
    }
    
    /// Format anomaly as JSON object
    pub fn anomaly_object(
        index: usize,
        value: f32,
        score: f32,
        anomaly_type: &str
    ) -> WasmResult<Object> {
        JsObjectBuilder::new()
            .set_usize("index", index)?
            .set_f32("value", value)?
            .set_f32("score", score)?
            .set_string("type", anomaly_type)?
            .build();
        Ok(JsObjectBuilder::new()
            .set_usize("index", index)?
            .set_f32("value", value)?
            .set_f32("score", score)?
            .set_string("type", anomaly_type)?
            .build())
    }
}