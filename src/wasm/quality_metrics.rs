//! WASM bindings for data quality metrics - Refactored
//! データ品質メトリクスのWASMバインディング - リファクタリング版

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use crate::wasm::common::{
    WasmError, WasmResult, WasmValidation, WasmVersion,
    WasmStats, JsonFormatter, JsObjectBuilder, JsArrayBuilder,
    WasmParamValidator, WasmAnalyzer, WasmQualityAssessment, WasmOperation
};
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
    pub fn new(threshold: f32) -> WasmResult<WasmQualityMetrics> {
        WasmParamValidator::validate_probability_range(threshold, "threshold")?;
        
        Ok(WasmQualityMetrics { threshold })
    }

    /// Calculate data completeness (percentage of non-NaN values)
    pub fn completeness(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        let non_nan_count = data.iter().filter(|&&x| !x.is_nan()).count();
        Ok((non_nan_count as f32 / data.len() as f32) * 100.0)
    }

    /// Calculate data accuracy (values within expected range)
    pub fn accuracy(&self, tensor: &WasmTensor, min_val: f32, max_val: f32) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        if min_val > max_val {
            return Err(WasmError::invalid_param("range", format!("{}..{}", min_val, max_val), "min must be <= max"));
        }
        
        let data = tensor.data();
        let valid_count = data.iter().filter(|&&x| x >= min_val && x <= max_val).count();
        Ok((valid_count as f32 / data.len() as f32) * 100.0)
    }

    /// Calculate data consistency (low variance indicator)
    pub fn consistency(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        
        let mean = WasmStats::mean(&data);
        let std_dev = WasmStats::std_dev(&data, Some(mean));
        
        Ok(100.0 - std_dev.min(100.0))
    }

    /// Calculate data validity (percentage of finite values)
    pub fn validity(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
        Ok((finite_count as f32 / data.len() as f32) * 100.0)
    }

    /// Calculate data uniqueness (ratio of unique values)
    pub fn uniqueness(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_data.dedup();
        
        Ok((sorted_data.len() as f32 / data.len() as f32) * 100.0)
    }

    /// Comprehensive quality score
    pub fn overall_quality(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        let completeness = self.completeness(tensor)?;
        let validity = self.validity(tensor)?;
        let consistency = self.consistency(tensor)?;
        let uniqueness = self.uniqueness(tensor)?;
        
        Ok(completeness * 0.3 + validity * 0.3 + consistency * 0.2 + uniqueness * 0.2)
    }

    /// Get quality report as JSON string
    pub fn quality_report(&self, tensor: &WasmTensor) -> WasmResult<String> {
        let completeness = self.completeness(tensor)?;
        let validity = self.validity(tensor)?;
        let consistency = self.consistency(tensor)?;
        let uniqueness = self.uniqueness(tensor)?;
        let overall = self.overall_quality(tensor)?;
        
        Ok(JsonFormatter::quality_json(completeness, validity, consistency, uniqueness, overall))
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
    pub fn basic_stats(&self, tensor: &WasmTensor) -> WasmResult<String> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        
        let mean = WasmStats::mean(&data);
        let std_dev = WasmStats::std_dev(&data, Some(mean));
        let min = WasmStats::min(&data);
        let max = WasmStats::max(&data);
        
        Ok(JsonFormatter::stats_json(mean, std_dev, min, max, data.len()))
    }

    /// Calculate percentiles
    pub fn percentiles(&self, tensor: &WasmTensor, percentiles: &[f32]) -> WasmResult<Array> {
        tensor.validate_non_empty()?;
        if percentiles.is_empty() {
            return Err(WasmError::empty_tensor());
        }
        
        let mut data = tensor.data().clone();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut builder = JsArrayBuilder::with_capacity(percentiles.len());
        for &p in percentiles {
            if p < 0.0 || p > 100.0 {
                return Err(WasmError::invalid_range("percentile", p, 0.0, 100.0));
            }
            let index = ((p / 100.0) * (data.len() - 1) as f32).round() as usize;
            let value = data[index.min(data.len() - 1)];
            builder = builder.push_f32(value);
        }
        
        Ok(builder.build())
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, tensor: &WasmTensor) -> WasmResult<Array> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        
        if data.len() < 4 {
            return Ok(JsArrayBuilder::new().build());
        }
        
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1_idx = sorted_data.len() / 4;
        let q3_idx = (3 * sorted_data.len()) / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let mut builder = JsArrayBuilder::new();
        for (i, &value) in data.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let outlier_obj = JsObjectBuilder::new()
                    .set_usize("index", i)?
                    .set_f32("value", value)?
                    .build();
                builder = builder.push_object(outlier_obj);
            }
        }
        
        Ok(builder.build())
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_quality_metrics_version() -> String {
    WasmVersion::module_version("Quality Metrics")
}

/// Create quality analyzer with default threshold
#[wasm_bindgen]
pub fn create_quality_analyzer() -> WasmResult<WasmQualityMetrics> {
    WasmQualityMetrics::new(0.95)
}

/// Quick quality assessment for web applications
#[wasm_bindgen]
pub fn quick_quality_assessment(tensor: &WasmTensor) -> WasmResult<String> {
    let metrics = WasmQualityMetrics::new(0.8)?;
    let completeness = metrics.completeness(tensor)?;
    let validity = metrics.validity(tensor)?;
    let consistency = metrics.consistency(tensor)?;
    let overall = metrics.overall_quality(tensor)?;
    
    let recommendation = if overall > 80.0 {
        "High quality data - ready for training"
    } else if overall > 60.0 {
        "Medium quality data - consider preprocessing"
    } else {
        "Low quality data - requires cleaning"
    };
    
    let timestamp = js_sys::Date::now();
    Ok(format!(
        "{{\"completeness\":{:.2},\"validity\":{:.2},\"consistency\":{:.2},\"overall_quality\":{:.2},\"timestamp\":{:.0},\"recommendation\":\"{}\"}}", 
        completeness, validity, consistency, overall, timestamp, recommendation
    ))
}

// Trait implementations for WasmQualityMetrics
impl WasmOperation for WasmQualityMetrics {
    fn name(&self) -> String {
        "QualityMetrics".to_string()
    }
}

impl WasmAnalyzer for WasmQualityMetrics {
    fn analyze(&self, tensor: &WasmTensor) -> WasmResult<String> {
        self.quality_report(tensor)
    }

    fn analysis_type(&self) -> &'static str {
        "quality_assessment"
    }
}

impl WasmQualityAssessment for WasmQualityMetrics {
    fn quality_score(&self, tensor: &WasmTensor) -> WasmResult<f32> {
        self.overall_quality(tensor)
    }

    fn quality_threshold(&self) -> f32 {
        self.threshold
    }
}

// Trait implementations for WasmStatisticalAnalyzer
impl WasmOperation for WasmStatisticalAnalyzer {
    fn name(&self) -> String {
        "StatisticalAnalyzer".to_string()
    }
}

impl WasmAnalyzer for WasmStatisticalAnalyzer {
    fn analyze(&self, tensor: &WasmTensor) -> WasmResult<String> {
        self.basic_stats(tensor)
    }

    fn analysis_type(&self) -> &'static str {
        "statistical_analysis"
    }
}