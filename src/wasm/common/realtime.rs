//! Real-time processing traits
//! リアルタイム処理トレイト

use wasm_bindgen::prelude::*;
use crate::wasm::common::error::{WasmError, WasmResult};
use crate::wasm::common::traits::WasmOperation;

/// Trait for real-time data processing
pub trait WasmRealtime: WasmOperation {
    /// Process single value in real-time
    fn process_realtime(&mut self, value: f32) -> WasmResult<JsValue>;
    
    /// Reset internal state
    fn reset_state(&mut self);
    
    /// Get processing statistics
    fn get_processing_stats(&self) -> String {
        format!("{{\"type\":\"{}\"}}", self.name())
    }
}

/// Trait for time series analysis
pub trait WasmTimeSeries: WasmOperation {
    /// Add data point with timestamp
    fn add_point(&mut self, timestamp: f64, value: f32) -> WasmResult<JsValue>;
    
    /// Get trend analysis
    fn trend_analysis(&self) -> WasmResult<String>;
    
    /// Get seasonal analysis
    fn seasonal_analysis(&self) -> WasmResult<String>;
    
    /// Get time series statistics
    fn get_time_stats(&self) -> String {
        format!("{{\"analyzer\":\"{}\"}}", self.name())
    }
}