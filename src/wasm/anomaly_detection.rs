//! WASM bindings for anomaly detection - Refactored
//! 異常検出のWASMバインディング - リファクタリング版

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use crate::wasm::common::{
    WasmError, WasmResult, WasmValidation, WasmVersion,
    WasmStats, JsonFormatter, JsObjectBuilder, JsArrayBuilder,
    WasmParamValidator, WasmNaming, WasmOperation, WasmDetector,
    WasmAnalyzer, WasmRealtime, WasmTimeSeries
};
use js_sys::Array;

/// WASM wrapper for Anomaly Detector
#[wasm_bindgen]
pub struct WasmAnomalyDetector {
    threshold: f32,
    window_size: usize,
    history: Vec<f32>,
}

#[wasm_bindgen]
impl WasmAnomalyDetector {
    /// Create new anomaly detector
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f32, window_size: usize) -> WasmResult<WasmAnomalyDetector> {
        WasmParamValidator::validate_positive(threshold, "threshold")?;
        WasmParamValidator::validate_non_zero_usize(window_size, "window_size")?;
        
        Ok(WasmAnomalyDetector {
            threshold,
            window_size,
            history: Vec::new(),
        })
    }

    /// Detect anomalies using statistical method
    pub fn detect_statistical(&mut self, data: &WasmTensor) -> WasmResult<Array> {
        data.validate_non_empty()?;
        let values = data.data();
        
        if values.len() < 3 {
            return Ok(JsArrayBuilder::new().build());
        }
        
        let mean = WasmStats::mean(&values);
        let std_dev = WasmStats::std_dev(&values, Some(mean));
        
        let mut builder = JsArrayBuilder::new();
        for (i, &value) in values.iter().enumerate() {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > self.threshold {
                let anomaly_obj = JsonFormatter::anomaly_object(i, value, z_score.abs(), "Statistical")?;
                builder = builder.push_object(anomaly_obj);
            }
        }
        
        Ok(builder.build())
    }

    /// Detect anomalies using isolation forest method (simplified)
    pub fn detect_isolation_forest(&mut self, data: &WasmTensor, _n_trees: usize) -> WasmResult<Array> {
        data.validate_non_empty()?;
        let values = data.data();
        
        if values.len() < 10 {
            return Ok(JsArrayBuilder::new().build());
        }
        
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1_idx = values.len() / 4;
        let q3_idx = (3 * values.len()) / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let mut builder = JsArrayBuilder::new();
        for (i, &value) in values.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let isolation_score = if value < lower_bound {
                    (lower_bound - value) / iqr
                } else {
                    (value - upper_bound) / iqr
                };
                
                let anomaly_obj = JsonFormatter::anomaly_object(i, value, isolation_score, "Isolation")?;
                builder = builder.push_object(anomaly_obj);
            }
        }
        
        Ok(builder.build())
    }

    /// Real-time anomaly detection for streaming data
    pub fn detect_realtime(&mut self, value: f32) -> WasmResult<JsValue> {
        if !value.is_finite() {
            return Err(WasmError::invalid_param("value", value, "must be finite"));
        }
        
        self.history.push(value);
        
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
        
        if self.history.len() < 3 {
            return Ok(JsValue::NULL);
        }
        
        let mean = WasmStats::mean(&self.history);
        let std_dev = WasmStats::std_dev(&self.history, Some(mean));
        
        let z_score = (value - mean) / std_dev;
        
        if z_score.abs() > self.threshold {
            let anomaly_obj = JsonFormatter::anomaly_object(self.history.len() - 1, value, z_score.abs(), "Realtime")?;
            Ok(anomaly_obj.into())
        } else {
            Ok(JsValue::NULL)
        }
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> WasmResult<String> {
        if self.history.is_empty() {
            return Ok("{\"count\":0}".to_string());
        }
        
        let mean = WasmStats::mean(&self.history);
        let std_dev = WasmStats::std_dev(&self.history, Some(mean));
        let min = WasmStats::min(&self.history);
        let max = WasmStats::max(&self.history);
        
        let stats = format!(
            "{{\"count\":{},\"mean\":{:.6},\"std\":{:.6},\"min\":{:.6},\"max\":{:.6},\"threshold\":{:.2},\"window_size\":{}}}",
            self.history.len(), mean, std_dev, min, max, self.threshold, self.window_size
        );
        
        Ok(stats)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update threshold
    pub fn set_threshold(&mut self, threshold: f32) -> WasmResult<()> {
        WasmParamValidator::validate_positive(threshold, "threshold")?;
        self.threshold = threshold;
        Ok(())
    }

    /// Get current threshold
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
}

/// WASM wrapper for Time Series Anomaly Detector
#[wasm_bindgen]
pub struct WasmTimeSeriesDetector {
    window_size: usize,
    seasonal_period: Option<usize>,
    timestamps: Vec<f64>,
    values: Vec<f32>,
}

#[wasm_bindgen]
impl WasmTimeSeriesDetector {
    /// Create new time series anomaly detector
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, seasonal_period: Option<usize>) -> WasmResult<WasmTimeSeriesDetector> {
        WasmParamValidator::validate_non_zero_usize(window_size, "window_size")?;
        
        Ok(WasmTimeSeriesDetector {
            window_size,
            seasonal_period,
            timestamps: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Add new data point and check for anomalies
    pub fn add_point(&mut self, timestamp: f64, value: f32) -> WasmResult<JsValue> {
        if !value.is_finite() {
            return Err(WasmError::invalid_param("value", value, "must be finite"));
        }
        
        self.timestamps.push(timestamp);
        self.values.push(value);
        
        if self.values.len() > self.window_size {
            self.timestamps.remove(0);
            self.values.remove(0);
        }
        
        if self.values.len() < 5 {
            return Ok(JsValue::NULL);
        }
        
        let recent_mean = WasmStats::mean(&self.values[self.values.len()-5..]);
        let historical_mean = WasmStats::mean(&self.values[..self.values.len()-5]);
        
        let deviation = (value - recent_mean).abs() / historical_mean.abs().max(1e-6);
        
        if deviation > 2.0 {
            let anomaly_obj = JsObjectBuilder::new()
                .set_f64("timestamp", timestamp)?
                .set_f32("value", value)?
                .set_f32("score", deviation)?
                .set_string("type", "Trend")?
                .build();
            Ok(anomaly_obj.into())
        } else {
            Ok(JsValue::NULL)
        }
    }

    /// Get trend analysis
    pub fn get_trend_analysis(&self) -> WasmResult<String> {
        if self.values.len() < 10 {
            return Ok("{\"trend\":\"insufficient_data\"}".to_string());
        }
        
        let half = self.values.len() / 2;
        let first_half_mean = WasmStats::mean(&self.values[..half]);
        let second_half_mean = WasmStats::mean(&self.values[half..]);
        
        let trend_direction = if second_half_mean > first_half_mean { "increasing" } else { "decreasing" };
        let trend_strength = ((second_half_mean - first_half_mean) / first_half_mean.abs().max(1e-6)).abs();
        
        let trend = format!(
            "{{\"direction\":\"{}\",\"strength\":{:.4},\"confidence\":{:.2}}}",
            trend_direction, trend_strength, trend_strength.min(1.0) * 100.0
        );
        
        Ok(trend)
    }

    /// Get seasonal analysis
    pub fn get_seasonal_analysis(&self) -> WasmResult<String> {
        match self.seasonal_period {
            Some(period) if self.values.len() >= period * 2 => {
                let mut seasonal_variance = 0.0;
                let cycles = self.values.len() / period;
                
                for i in 0..period {
                    let cycle_values: Vec<f32> = (0..cycles)
                        .map(|c| self.values[c * period + i])
                        .collect();
                    
                    let cycle_mean = WasmStats::mean(&cycle_values);
                    let cycle_var = WasmStats::variance(&cycle_values, Some(cycle_mean));
                    seasonal_variance += cycle_var;
                }
                
                seasonal_variance /= period as f32;
                
                let seasonal = format!(
                    "{{\"period\":{},\"variance\":{:.6},\"seasonality_strength\":{:.2}}}",
                    period, seasonal_variance, (1.0 - seasonal_variance).max(0.0) * 100.0
                );
                
                Ok(seasonal)
            }
            _ => Ok("{\"seasonality\":\"insufficient_data\"}".to_string())
        }
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_anomaly_detection_version() -> String {
    WasmVersion::module_version("Anomaly Detection")
}

/// Create a simple anomaly detector for web applications
#[wasm_bindgen]
pub fn create_simple_detector(threshold: f32) -> WasmResult<WasmAnomalyDetector> {
    WasmAnomalyDetector::new(threshold, 50)
}

/// Create a time series detector for streaming data
#[wasm_bindgen]
pub fn create_streaming_detector(window_size: usize) -> WasmResult<WasmTimeSeriesDetector> {
    WasmTimeSeriesDetector::new(window_size, None)
}

/// Batch anomaly detection for arrays
#[wasm_bindgen]
pub fn detect_anomalies_batch(data: &[f32], threshold: f32) -> WasmResult<Array> {
    if data.is_empty() {
        return Err(WasmError::empty_tensor());
    }
    
    let tensor = WasmTensor::new(data.to_vec(), vec![data.len()]);
    let mut detector = WasmAnomalyDetector::new(threshold, data.len().min(100))?;
    detector.detect_statistical(&tensor)
}

// Trait implementations for WasmAnomalyDetector
impl WasmOperation for WasmAnomalyDetector {
    fn name(&self) -> String {
        "AnomalyDetector".to_string()
    }
}

impl WasmDetector for WasmAnomalyDetector {
    fn detect(&mut self, tensor: &WasmTensor) -> WasmResult<Array> {
        self.detect_statistical(tensor)
    }

    fn detection_type(&self) -> &'static str {
        "anomaly_detection"
    }

    fn threshold(&self) -> f32 {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: f32) -> WasmResult<()> {
        self.set_threshold(threshold)
    }

    fn reset(&mut self) {
        self.reset()
    }
}

impl WasmAnalyzer for WasmAnomalyDetector {
    fn analyze(&self, _tensor: &WasmTensor) -> WasmResult<String> {
        self.get_statistics()
    }

    fn analysis_type(&self) -> &'static str {
        "anomaly_statistics"
    }
}

impl WasmRealtime for WasmAnomalyDetector {
    fn process_realtime(&mut self, value: f32) -> WasmResult<JsValue> {
        self.detect_realtime(value)
    }

    fn reset_state(&mut self) {
        self.reset()
    }
}

// Trait implementations for WasmTimeSeriesDetector
impl WasmOperation for WasmTimeSeriesDetector {
    fn name(&self) -> String {
        "TimeSeriesDetector".to_string()
    }
}

impl WasmDetector for WasmTimeSeriesDetector {
    fn detect(&mut self, _tensor: &WasmTensor) -> WasmResult<Array> {
        // Time series detector works on individual points, not tensors
        Ok(JsArrayBuilder::new().build())
    }

    fn detection_type(&self) -> &'static str {
        "time_series_anomaly"
    }

    fn threshold(&self) -> f32 {
        2.0 // Fixed threshold for trend detection
    }

    fn set_threshold(&mut self, _threshold: f32) -> WasmResult<()> {
        // Time series detector doesn't support threshold modification
        Ok(())
    }

    fn reset(&mut self) {
        self.timestamps.clear();
        self.values.clear();
    }
}

impl WasmTimeSeries for WasmTimeSeriesDetector {
    fn add_point(&mut self, timestamp: f64, value: f32) -> WasmResult<JsValue> {
        self.add_point(timestamp, value)
    }

    fn trend_analysis(&self) -> WasmResult<String> {
        self.get_trend_analysis()
    }

    fn seasonal_analysis(&self) -> WasmResult<String> {
        self.get_seasonal_analysis()
    }
}