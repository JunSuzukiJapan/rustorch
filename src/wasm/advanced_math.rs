//! WASM bindings for advanced mathematical operations - Refactored
//! 高度数学操作のWASMバインディング - リファクタリング版

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use crate::wasm::common::{
    WasmError, WasmResult, WasmValidation, WasmVersion,
    WasmStats, MemoryManager, JsArrayBuilder, WasmParamValidator,
    WasmMathOperation, WasmAnalyzer, WasmStatistical, WasmOperation
};
use js_sys::Array;

/// WASM wrapper for advanced mathematical operations
#[wasm_bindgen]
pub struct WasmAdvancedMath;

#[wasm_bindgen]
impl WasmAdvancedMath {
    /// Create new advanced math instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAdvancedMath {
        WasmAdvancedMath
    }

    /// Hyperbolic sine
    pub fn sinh(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.sinh()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Hyperbolic cosine
    pub fn cosh(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.cosh()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Hyperbolic tangent
    pub fn tanh(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.tanh()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Inverse sine (arcsine)
    pub fn asin(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.asin()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Inverse cosine (arccosine)
    pub fn acos(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.acos()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Inverse tangent (arctangent)
    pub fn atan(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.atan()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Two-argument arctangent
    pub fn atan2(&self, y: &WasmTensor, x: &WasmTensor) -> WasmResult<WasmTensor> {
        y.validate_non_empty()?;
        x.validate_shape_match(y)?;
        
        let mut result_buffer = MemoryManager::get_buffer(y.data().len());
        result_buffer.extend(y.data().iter().zip(x.data().iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val)));
        
        Ok(WasmTensor::new(result_buffer, y.shape()))
    }

    /// Error function (approximate)
    pub fn erf(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        
        result_buffer.extend(tensor.data().iter().map(|&x| {
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;
            
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            
            sign * y
        }));
        
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Complementary error function
    pub fn erfc(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        let erf_result = self.erf(tensor)?;
        let mut result_buffer = MemoryManager::get_buffer(erf_result.data().len());
        result_buffer.extend(erf_result.data().iter().map(|&x| 1.0 - x));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Gamma function (approximate)
    pub fn gamma(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        
        result_buffer.extend(tensor.data().iter().map(|&x| {
            if x <= 0.0 {
                f32::NAN
            } else if x < 1.0 {
                self.gamma_approx(x + 1.0) / x
            } else {
                self.gamma_approx(x)
            }
        }));
        
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Log gamma function
    pub fn lgamma(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        let gamma_result = self.gamma(tensor)?;
        let mut result_buffer = MemoryManager::get_buffer(gamma_result.data().len());
        result_buffer.extend(gamma_result.data().iter().map(|&x| x.ln()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Clamp values between min and max
    pub fn clamp(&self, tensor: &WasmTensor, min_val: f32, max_val: f32) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        if min_val > max_val {
            return Err(WasmError::invalid_param("range", format!("{}..{}", min_val, max_val), "min must be <= max"));
        }
        
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.clamp(min_val, max_val)));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Sign function
    pub fn sign(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| {
            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
        }));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Linear interpolation between two tensors
    pub fn lerp(&self, start: &WasmTensor, end: &WasmTensor, weight: f32) -> WasmResult<WasmTensor> {
        start.validate_non_empty()?;
        start.validate_shape_match(end)?;
        
        let mut result_buffer = MemoryManager::get_buffer(start.data().len());
        result_buffer.extend(start.data().iter().zip(end.data().iter())
            .map(|(&s, &e)| s + weight * (e - s)));
        
        Ok(WasmTensor::new(result_buffer, start.shape()))
    }

    /// Power function with scalar exponent
    pub fn pow(&self, base: &WasmTensor, exponent: f32) -> WasmResult<WasmTensor> {
        base.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(base.data().len());
        result_buffer.extend(base.data().iter().map(|&x| x.powf(exponent)));
        Ok(WasmTensor::new(result_buffer, base.shape()))
    }

    /// Element-wise power
    pub fn pow_tensor(&self, base: &WasmTensor, exponent: &WasmTensor) -> WasmResult<WasmTensor> {
        base.validate_non_empty()?;
        base.validate_shape_match(exponent)?;
        
        let mut result_buffer = MemoryManager::get_buffer(base.data().len());
        result_buffer.extend(base.data().iter().zip(exponent.data().iter())
            .map(|(&b, &e)| b.powf(e)));
        
        Ok(WasmTensor::new(result_buffer, base.shape()))
    }

    /// Round to nearest integer
    pub fn round(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.round()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Floor function
    pub fn floor(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.floor()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Ceiling function
    pub fn ceil(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.ceil()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Truncate to integer
    pub fn trunc(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| x.trunc()));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Check if values are finite
    pub fn is_finite(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| if x.is_finite() { 1.0 } else { 0.0 }));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Check if values are infinite
    pub fn is_infinite(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| if x.is_infinite() { 1.0 } else { 0.0 }));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Check if values are NaN
    pub fn is_nan(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        let mut result_buffer = MemoryManager::get_buffer(tensor.data().len());
        result_buffer.extend(tensor.data().iter().map(|&x| if x.is_nan() { 1.0 } else { 0.0 }));
        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Stirling's approximation for gamma function
    fn gamma_approx(&self, x: f32) -> f32 {
        if x < 1.0 {
            return f32::NAN;
        }
        
        let two_pi = 2.0 * std::f32::consts::PI;
        let e = std::f32::consts::E;
        
        (two_pi / x).sqrt() * (x / e).powf(x)
    }
}

/// Advanced statistical functions for web applications
#[wasm_bindgen]
pub struct WasmStatisticalFunctions;

#[wasm_bindgen]
impl WasmStatisticalFunctions {
    /// Create new statistical functions instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStatisticalFunctions {
        WasmStatisticalFunctions
    }

    /// Calculate correlation coefficient between two tensors
    pub fn correlation(&self, x: &WasmTensor, y: &WasmTensor) -> WasmResult<f32> {
        x.validate_non_empty()?;
        x.validate_shape_match(y)?;
        
        let x_data = x.data();
        let y_data = y.data();
        
        let x_mean = WasmStats::mean(&x_data);
        let y_mean = WasmStats::mean(&y_data);
        
        let numerator: f32 = x_data.iter().zip(y_data.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
        
        let x_var: f32 = x_data.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let y_var: f32 = y_data.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        
        let denominator = (x_var * y_var).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate covariance between two tensors
    pub fn covariance(&self, x: &WasmTensor, y: &WasmTensor) -> WasmResult<f32> {
        x.validate_non_empty()?;
        x.validate_shape_match(y)?;
        
        let x_data = x.data();
        let y_data = y.data();
        
        let n = x_data.len() as f32;
        let x_mean = WasmStats::mean(&x_data);
        let y_mean = WasmStats::mean(&y_data);
        
        let covariance: f32 = x_data.iter().zip(y_data.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f32>() / (n - 1.0);
        
        Ok(covariance)
    }

    /// Calculate percentile
    pub fn percentile(&self, tensor: &WasmTensor, percentile: f32) -> WasmResult<f32> {
        tensor.validate_non_empty()?;
        WasmParamValidator::validate_percentage_range(percentile, "percentile")?;
        
        let mut data = tensor.data().clone();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((percentile / 100.0) * (data.len() - 1) as f32).round() as usize;
        Ok(data[index.min(data.len() - 1)])
    }

    /// Calculate quantiles
    pub fn quantiles(&self, tensor: &WasmTensor, q: &[f32]) -> WasmResult<Array> {
        tensor.validate_non_empty()?;
        if q.is_empty() {
            return Err(WasmError::empty_tensor());
        }
        
        let mut data = tensor.data().clone();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut builder = JsArrayBuilder::with_capacity(q.len());
        for &quantile in q {
            WasmParamValidator::validate_percentage_range(quantile, "quantile")?;
            let index = ((quantile / 100.0) * (data.len() - 1) as f32).round() as usize;
            let value = data[index.min(data.len() - 1)];
            builder = builder.push_f32(value);
        }
        
        Ok(builder.build())
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_advanced_math_version() -> String {
    WasmVersion::module_version("Advanced Math")
}

// Trait implementations for WasmAdvancedMath
impl WasmOperation for WasmAdvancedMath {
    fn name(&self) -> String {
        "AdvancedMath".to_string()
    }
}

impl WasmMathOperation for WasmAdvancedMath {
    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "sinh" | "cosh" | "tanh" | "asin" | "acos" | "atan" | "atan2" |
            "erf" | "erfc" | "gamma" | "lgamma" | "clamp" | "sign" | 
            "lerp" | "pow" | "pow_tensor" | "round" | "floor" | "ceil" | "trunc" |
            "is_finite" | "is_infinite" | "is_nan"
        )
    }

    fn operation_complexity(&self, operation: &str) -> u32 {
        match operation {
            "sign" | "round" | "floor" | "ceil" | "trunc" | "is_finite" | "is_infinite" | "is_nan" => 1,
            "sinh" | "cosh" | "tanh" | "asin" | "acos" | "atan" | "clamp" | "lerp" | "pow" => 2,
            "atan2" | "pow_tensor" => 3,
            "erf" | "erfc" | "gamma" | "lgamma" => 4,
            _ => 1
        }
    }
}

// Trait implementations for WasmStatisticalFunctions
impl WasmOperation for WasmStatisticalFunctions {
    fn name(&self) -> String {
        "StatisticalFunctions".to_string()
    }
}

impl WasmAnalyzer for WasmStatisticalFunctions {
    fn analyze(&self, tensor: &WasmTensor) -> WasmResult<String> {
        tensor.validate_non_empty()?;
        let data = tensor.data();
        
        let mean = WasmStats::mean(&data);
        let std_dev = WasmStats::std_dev(&data, Some(mean));
        let min = WasmStats::min(&data);
        let max = WasmStats::max(&data);
        
        Ok(format!(
            "{{\"type\":\"statistical_summary\",\"mean\":{:.6},\"std\":{:.6},\"min\":{:.6},\"max\":{:.6},\"count\":{}}}",
            mean, std_dev, min, max, data.len()
        ))
    }

    fn analysis_type(&self) -> &'static str {
        "statistical_functions"
    }
}

impl WasmStatistical for WasmStatisticalFunctions {
    fn statistical_summary(&self, tensor: &WasmTensor) -> WasmResult<String> {
        self.analyze(tensor)
    }

    fn supports_statistical_operation(&self, operation: &str) -> bool {
        matches!(operation, "correlation" | "covariance" | "percentile" | "quantiles")
    }
}