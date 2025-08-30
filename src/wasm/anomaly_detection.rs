//! WASM bindings for anomaly detection
//! 異常検出のWASMバインディング

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
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
    pub fn new(threshold: f32, window_size: usize) -> WasmAnomalyDetector {
        WasmAnomalyDetector {
            threshold,
            window_size,
            history: Vec::new(),
        }
    }

    /// Detect anomalies using statistical method
    pub fn detect_statistical(&mut self, data: &WasmTensor) -> Result<Array, JsValue> {
        let values = data.data();
        if values.len() < 3 {
            return Ok(Array::new());
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        
        let js_array = Array::new();
        for (i, &value) in values.iter().enumerate() {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > self.threshold {
                let anomaly_obj = js_sys::Object::new();
                js_sys::Reflect::set(&anomaly_obj, &"index".into(), &JsValue::from_f64(i as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"value".into(), &JsValue::from_f64(value as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"score".into(), &JsValue::from_f64(z_score.abs() as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"type".into(), &JsValue::from_str("Statistical"))?;
                js_array.push(&anomaly_obj);
            }
        }
        
        Ok(js_array)
    }

    /// Detect anomalies using isolation forest method (simplified)
    pub fn detect_isolation_forest(&mut self, data: &WasmTensor, n_trees: usize) -> Result<Array, JsValue> {
        let values = data.data();
        if values.len() < 10 {
            return Ok(Array::new());
        }
        
        // Simplified isolation forest: use statistical outlier detection
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1_idx = values.len() / 4;
        let q3_idx = (3 * values.len()) / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let js_array = Array::new();
        for (i, &value) in values.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let isolation_score = if value < lower_bound {
                    (lower_bound - value) / iqr
                } else {
                    (value - upper_bound) / iqr
                };
                
                let anomaly_obj = js_sys::Object::new();
                js_sys::Reflect::set(&anomaly_obj, &"index".into(), &JsValue::from_f64(i as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"value".into(), &JsValue::from_f64(value as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"score".into(), &JsValue::from_f64(isolation_score as f64))?;
                js_sys::Reflect::set(&anomaly_obj, &"type".into(), &JsValue::from_str("Isolation"))?;
                js_array.push(&anomaly_obj);
            }
        }
        
        Ok(js_array)
    }

    /// Real-time anomaly detection for streaming data
    pub fn detect_realtime(&mut self, value: f32) -> Result<JsValue, JsValue> {
        self.history.push(value);
        
        // Keep only recent history
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
        
        if self.history.len() < 3 {
            return Ok(JsValue::NULL);
        }
        
        let mean = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance = self.history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.history.len() as f32;
        let std_dev = variance.sqrt();
        
        let z_score = (value - mean) / std_dev;
        
        if z_score.abs() > self.threshold {
            let anomaly_obj = js_sys::Object::new();
            js_sys::Reflect::set(&anomaly_obj, &"index".into(), &JsValue::from_f64((self.history.len() - 1) as f64))?;
            js_sys::Reflect::set(&anomaly_obj, &"value".into(), &JsValue::from_f64(value as f64))?;
            js_sys::Reflect::set(&anomaly_obj, &"score".into(), &JsValue::from_f64(z_score.abs() as f64))?;
            js_sys::Reflect::set(&anomaly_obj, &"type".into(), &JsValue::from_str("Realtime"))?;
            Ok(anomaly_obj.into())
        } else {
            Ok(JsValue::NULL)
        }
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> Result<String, JsValue> {
        if self.history.is_empty() {
            return Ok("{\"count\":0}".to_string());
        }
        
        let mean = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance = self.history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.history.len() as f32;
        let std_dev = variance.sqrt();
        
        let stats = format!(
            "{{\"count\":{},\"mean\":{:.6},\"std\":{:.6},\"threshold\":{:.2},\"window_size\":{}}}",
            self.history.len(), mean, std_dev, self.threshold, self.window_size
        );
        
        Ok(stats)
    }

    /// Reset detector state
    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.history.clear();
        Ok(())
    }

    /// Update threshold
    pub fn set_threshold(&mut self, threshold: f32) -> Result<(), JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }
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
    pub fn new(window_size: usize, seasonal_period: Option<usize>) -> WasmTimeSeriesDetector {
        WasmTimeSeriesDetector {
            window_size,
            seasonal_period,
            timestamps: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add new data point and check for anomalies
    pub fn add_point(&mut self, timestamp: f64, value: f32) -> Result<JsValue, JsValue> {
        self.timestamps.push(timestamp);
        self.values.push(value);
        
        // Keep only recent history
        if self.values.len() > self.window_size {
            self.timestamps.remove(0);
            self.values.remove(0);
        }
        
        if self.values.len() < 5 {
            return Ok(JsValue::NULL);
        }
        
        // Simple trend-based anomaly detection
        let recent_mean = self.values[self.values.len()-5..].iter().sum::<f32>() / 5.0;
        let historical_mean = self.values[..self.values.len()-5].iter().sum::<f32>() / (self.values.len() - 5) as f32;
        
        let deviation = (value - recent_mean).abs() / historical_mean.abs().max(1e-6);
        
        if deviation > 2.0 {
            let anomaly_obj = js_sys::Object::new();
            js_sys::Reflect::set(&anomaly_obj, &"timestamp".into(), &JsValue::from_f64(timestamp))?;
            js_sys::Reflect::set(&anomaly_obj, &"value".into(), &JsValue::from_f64(value as f64))?;
            js_sys::Reflect::set(&anomaly_obj, &"score".into(), &JsValue::from_f64(deviation as f64))?;
            js_sys::Reflect::set(&anomaly_obj, &"type".into(), &JsValue::from_str("Trend"))?;
            Ok(anomaly_obj.into())
        } else {
            Ok(JsValue::NULL)
        }
    }

    /// Get trend analysis
    pub fn get_trend_analysis(&self) -> Result<String, JsValue> {
        if self.values.len() < 10 {
            return Ok("{\"trend\":\"insufficient_data\"}".to_string());
        }
        
        let half = self.values.len() / 2;
        let first_half_mean = self.values[..half].iter().sum::<f32>() / half as f32;
        let second_half_mean = self.values[half..].iter().sum::<f32>() / (self.values.len() - half) as f32;
        
        let trend_direction = if second_half_mean > first_half_mean { "increasing" } else { "decreasing" };
        let trend_strength = ((second_half_mean - first_half_mean) / first_half_mean.abs().max(1e-6)).abs();
        
        let trend = format!(
            "{{\"direction\":\"{}\",\"strength\":{:.4},\"confidence\":{:.2}}}",
            trend_direction, trend_strength, trend_strength.min(1.0) * 100.0
        );
        
        Ok(trend)
    }

    /// Get seasonal analysis
    pub fn get_seasonal_analysis(&self) -> Result<String, JsValue> {
        match self.seasonal_period {
            Some(period) if self.values.len() >= period * 2 => {
                let mut seasonal_variance = 0.0;
                let cycles = self.values.len() / period;
                
                for i in 0..period {
                    let cycle_values: Vec<f32> = (0..cycles)
                        .map(|c| self.values[c * period + i])
                        .collect();
                    
                    let cycle_mean = cycle_values.iter().sum::<f32>() / cycle_values.len() as f32;
                    let cycle_var = cycle_values.iter()
                        .map(|&x| (x - cycle_mean).powi(2))
                        .sum::<f32>() / cycle_values.len() as f32;
                    
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

/// Helper functions for anomaly detection
#[wasm_bindgen]
pub fn wasm_anomaly_detection_version() -> String {
    "RusTorch WASM Anomaly Detection v0.5.2".to_string()
}

/// Create a simple anomaly detector for web applications
#[wasm_bindgen]
pub fn create_simple_detector(threshold: f32) -> WasmAnomalyDetector {
    WasmAnomalyDetector::new(threshold, 50)
}

/// Create a time series detector for streaming data
#[wasm_bindgen]
pub fn create_streaming_detector(window_size: usize) -> WasmTimeSeriesDetector {
    WasmTimeSeriesDetector::new(window_size, None)
}

/// Batch anomaly detection for arrays
#[wasm_bindgen]
pub fn detect_anomalies_batch(data: &[f32], threshold: f32) -> Result<Array, JsValue> {
    let tensor = WasmTensor::new(data.to_vec(), vec![data.len()]);
    let mut detector = WasmAnomalyDetector::new(threshold, data.len().min(100));
    detector.detect_statistical(&tensor)
}