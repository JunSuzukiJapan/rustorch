//! WASM bindings for data quality metrics
//! データ品質メトリクスのWASMバインディング

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use js_sys::Array;

/// WASM wrapper for Quality Metrics
#[wasm_bindgen]
pub struct WasmQualityMetrics {
    threshold: f32,
}

#[wasm_bindgen]
impl WasmQualityMetrics {
    /// Create new quality metrics analyzer
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f32) -> WasmQualityMetrics {
        WasmQualityMetrics { threshold }
    }

    /// Calculate data completeness (percentage of non-NaN values)
    pub fn completeness(&self, tensor: &WasmTensor) -> f32 {
        let data = tensor.data();
        let non_nan_count = data.iter().filter(|&&x| !x.is_nan()).count();
        (non_nan_count as f32 / data.len() as f32) * 100.0
    }

    /// Calculate data accuracy (values within expected range)
    pub fn accuracy(&self, tensor: &WasmTensor, min_val: f32, max_val: f32) -> f32 {
        let data = tensor.data();
        let valid_count = data.iter().filter(|&&x| x >= min_val && x <= max_val).count();
        (valid_count as f32 / data.len() as f32) * 100.0
    }

    /// Calculate data consistency (low variance indicator)
    pub fn consistency(&self, tensor: &WasmTensor) -> Result<f32, JsValue> {
        let data = tensor.data();
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor"));
        }
        
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();
        
        // Return consistency as percentage (lower variance = higher consistency)
        Ok(100.0 - std_dev.min(100.0))
    }

    /// Calculate data validity (percentage of finite values)
    pub fn validity(&self, tensor: &WasmTensor) -> f32 {
        let data = tensor.data();
        let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
        (finite_count as f32 / data.len() as f32) * 100.0
    }

    /// Calculate data uniqueness (ratio of unique values)
    pub fn uniqueness(&self, tensor: &WasmTensor) -> f32 {
        let data = tensor.data();
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_data.dedup();
        
        (sorted_data.len() as f32 / data.len() as f32) * 100.0
    }

    /// Comprehensive quality score
    pub fn overall_quality(&self, tensor: &WasmTensor) -> Result<f32, JsValue> {
        let completeness = self.completeness(tensor);
        let validity = self.validity(tensor);
        let consistency = self.consistency(tensor)?;
        let uniqueness = self.uniqueness(tensor);
        
        // Weighted average
        Ok(completeness * 0.3 + validity * 0.3 + consistency * 0.2 + uniqueness * 0.2)
    }

    /// Get quality report as JSON string
    pub fn quality_report(&self, tensor: &WasmTensor) -> Result<String, JsValue> {
        let completeness = self.completeness(tensor);
        let validity = self.validity(tensor);
        let consistency = self.consistency(tensor)?;
        let uniqueness = self.uniqueness(tensor);
        let overall = self.overall_quality(tensor)?;
        
        let report = format!(
            "{{\"completeness\":{:.2},\"validity\":{:.2},\"consistency\":{:.2},\"uniqueness\":{:.2},\"overall_quality\":{:.2}}}",
            completeness, validity, consistency, uniqueness, overall
        );
        
        Ok(report)
    }
}

/// WASM wrapper for Statistical Analysis
#[wasm_bindgen]
pub struct WasmStatisticalAnalyzer;

#[wasm_bindgen]
impl WasmStatisticalAnalyzer {
    /// Create new statistical analyzer
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStatisticalAnalyzer {
        WasmStatisticalAnalyzer
    }

    /// Calculate basic statistics
    pub fn basic_stats(&self, tensor: &WasmTensor) -> Result<String, JsValue> {
        let data = tensor.data();
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor"));
        }
        
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();
        
        let stats = format!(
            "{{\"mean\":{:.6},\"std\":{:.6},\"min\":{:.6},\"max\":{:.6},\"count\":{}}}",
            mean, std_dev, min, max, data.len()
        );
        
        Ok(stats)
    }

    /// Calculate percentiles
    pub fn percentiles(&self, tensor: &WasmTensor, percentiles: &[f32]) -> Result<Array, JsValue> {
        let mut data = tensor.data();
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor"));
        }
        
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let js_array = Array::new();
        for &p in percentiles {
            let index = ((p / 100.0) * (data.len() - 1) as f32).round() as usize;
            let value = data[index.min(data.len() - 1)];
            js_array.push(&JsValue::from_f64(value as f64));
        }
        
        Ok(js_array)
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, tensor: &WasmTensor) -> Result<Array, JsValue> {
        let mut data = tensor.data();
        if data.len() < 4 {
            return Ok(Array::new());
        }
        
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1_idx = data.len() / 4;
        let q3_idx = (3 * data.len()) / 4;
        let q1 = data[q1_idx];
        let q3 = data[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let js_array = Array::new();
        for (i, &value) in tensor.data().iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let outlier_obj = js_sys::Object::new();
                js_sys::Reflect::set(&outlier_obj, &"index".into(), &JsValue::from_f64(i as f64)).unwrap();
                js_sys::Reflect::set(&outlier_obj, &"value".into(), &JsValue::from_f64(value as f64)).unwrap();
                js_array.push(&outlier_obj);
            }
        }
        
        Ok(js_array)
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_quality_metrics_version() -> String {
    "RusTorch WASM Quality Metrics v0.5.2".to_string()
}

/// Create quality analyzer with default threshold
#[wasm_bindgen]
pub fn create_quality_analyzer() -> WasmQualityMetrics {
    WasmQualityMetrics::new(0.95)
}

/// Quick quality assessment for web applications
#[wasm_bindgen]
pub fn quick_quality_assessment(tensor: &WasmTensor) -> Result<String, JsValue> {
    let metrics = WasmQualityMetrics::new(0.8);
    let completeness = metrics.completeness(tensor);
    let validity = metrics.validity(tensor);
    let consistency = metrics.consistency(tensor)?;
    let overall = metrics.overall_quality(tensor)?;
    
    let assessment = format!(
        "{{\"completeness\":{:.2},\"validity\":{:.2},\"consistency\":{:.2},\"overall_quality\":{:.2},\"timestamp\":{:.0},\"recommendation\":\"{}\"}}",
        completeness, validity, consistency, overall, js_sys::Date::now(),
        if overall > 80.0 {
            "High quality data - ready for training"
        } else if overall > 60.0 {
            "Medium quality data - consider preprocessing"
        } else {
            "Low quality data - requires cleaning"
        }
    );
    
    Ok(assessment)
}