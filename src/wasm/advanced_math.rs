//! WASM bindings for advanced mathematical operations
//! 高度数学操作のWASMバインディング

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
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
    pub fn sinh(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.sinh()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Hyperbolic cosine
    pub fn cosh(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.cosh()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Hyperbolic tangent
    pub fn tanh(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.tanh()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Inverse sine (arcsine)
    pub fn asin(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.asin()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Inverse cosine (arccosine)
    pub fn acos(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.acos()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Inverse tangent (arctangent)
    pub fn atan(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.atan()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Two-argument arctangent
    pub fn atan2(&self, y: &WasmTensor, x: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if y.shape() != x.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = y.data().iter().zip(x.data().iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val))
            .collect();
        
        Ok(WasmTensor::new(result, y.shape()))
    }

    /// Error function (approximate)
    pub fn erf(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| {
            // Approximation of error function
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
        }).collect();
        
        WasmTensor::new(result, tensor.shape())
    }

    /// Complementary error function
    pub fn erfc(&self, tensor: &WasmTensor) -> WasmTensor {
        let erf_result = self.erf(tensor);
        let result: Vec<f32> = erf_result.data().iter().map(|&x| 1.0 - x).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Gamma function (approximate)
    pub fn gamma(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| {
            // Simple approximation for positive values
            if x <= 0.0 {
                f32::NAN
            } else if x < 1.0 {
                // Gamma(x+1) = x * Gamma(x), so Gamma(x) = Gamma(x+1) / x
                self.gamma_approx(x + 1.0) / x
            } else {
                self.gamma_approx(x)
            }
        }).collect();
        
        WasmTensor::new(result, tensor.shape())
    }

    /// Log gamma function
    pub fn lgamma(&self, tensor: &WasmTensor) -> WasmTensor {
        let gamma_result = self.gamma(tensor);
        let result: Vec<f32> = gamma_result.data().iter().map(|&x| x.ln()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Clamp values between min and max
    pub fn clamp(&self, tensor: &WasmTensor, min_val: f32, max_val: f32) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.clamp(min_val, max_val)).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Sign function
    pub fn sign(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| {
            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
        }).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Linear interpolation between two tensors
    pub fn lerp(&self, start: &WasmTensor, end: &WasmTensor, weight: f32) -> Result<WasmTensor, JsValue> {
        if start.shape() != end.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = start.data().iter().zip(end.data().iter())
            .map(|(&s, &e)| s + weight * (e - s))
            .collect();
        
        Ok(WasmTensor::new(result, start.shape()))
    }

    /// Power function with scalar exponent
    pub fn pow(&self, base: &WasmTensor, exponent: f32) -> WasmTensor {
        let result: Vec<f32> = base.data().iter().map(|&x| x.powf(exponent)).collect();
        WasmTensor::new(result, base.shape())
    }

    /// Element-wise power
    pub fn pow_tensor(&self, base: &WasmTensor, exponent: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if base.shape() != exponent.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let result: Vec<f32> = base.data().iter().zip(exponent.data().iter())
            .map(|(&b, &e)| b.powf(e))
            .collect();
        
        Ok(WasmTensor::new(result, base.shape()))
    }

    /// Round to nearest integer
    pub fn round(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.round()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Floor function
    pub fn floor(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.floor()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Ceiling function
    pub fn ceil(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.ceil()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Truncate to integer
    pub fn trunc(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| x.trunc()).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Check if values are finite
    pub fn is_finite(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| if x.is_finite() { 1.0 } else { 0.0 }).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Check if values are infinite
    pub fn is_infinite(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| if x.is_infinite() { 1.0 } else { 0.0 }).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Check if values are NaN
    pub fn is_nan(&self, tensor: &WasmTensor) -> WasmTensor {
        let result: Vec<f32> = tensor.data().iter().map(|&x| if x.is_nan() { 1.0 } else { 0.0 }).collect();
        WasmTensor::new(result, tensor.shape())
    }

    /// Stirling's approximation for gamma function
    fn gamma_approx(&self, x: f32) -> f32 {
        if x < 1.0 {
            return f32::NAN;
        }
        
        // Stirling's approximation: Γ(x) ≈ sqrt(2π/x) * (x/e)^x
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
    pub fn correlation(&self, x: &WasmTensor, y: &WasmTensor) -> Result<f32, JsValue> {
        if x.shape() != y.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let x_data = x.data();
        let y_data = y.data();
        
        if x_data.is_empty() {
            return Err(JsValue::from_str("Empty tensors"));
        }
        
        let n = x_data.len() as f32;
        let x_mean = x_data.iter().sum::<f32>() / n;
        let y_mean = y_data.iter().sum::<f32>() / n;
        
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
    pub fn covariance(&self, x: &WasmTensor, y: &WasmTensor) -> Result<f32, JsValue> {
        if x.shape() != y.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }
        
        let x_data = x.data();
        let y_data = y.data();
        
        if x_data.is_empty() {
            return Err(JsValue::from_str("Empty tensors"));
        }
        
        let n = x_data.len() as f32;
        let x_mean = x_data.iter().sum::<f32>() / n;
        let y_mean = y_data.iter().sum::<f32>() / n;
        
        let covariance: f32 = x_data.iter().zip(y_data.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f32>() / (n - 1.0);
        
        Ok(covariance)
    }

    /// Calculate percentile
    pub fn percentile(&self, tensor: &WasmTensor, percentile: f32) -> Result<f32, JsValue> {
        let mut data = tensor.data();
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor"));
        }
        
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((percentile / 100.0) * (data.len() - 1) as f32).round() as usize;
        Ok(data[index.min(data.len() - 1)])
    }

    /// Calculate quantiles
    pub fn quantiles(&self, tensor: &WasmTensor, q: &[f32]) -> Result<Array, JsValue> {
        let mut data = tensor.data();
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor"));
        }
        
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let js_array = Array::new();
        for &quantile in q {
            let index = ((quantile / 100.0) * (data.len() - 1) as f32).round() as usize;
            let value = data[index.min(data.len() - 1)];
            js_array.push(&JsValue::from_f64(value as f64));
        }
        
        Ok(js_array)
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_advanced_math_version() -> String {
    "RusTorch WASM Advanced Math v0.5.2".to_string()
}