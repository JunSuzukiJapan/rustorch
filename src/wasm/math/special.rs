//! Enhanced special mathematical functions for WebAssembly
//! WebAssembly向け強化特殊数学関数

use crate::special::{bessel, error, gamma};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Note: console_log macro removed as it was unused
// Use web_sys::console::log_1() directly if logging is needed

/// Legacy WASM Special Functions struct (for compatibility)
/// 互換性のための従来のWASM特殊関数構造体
#[wasm_bindgen]
pub struct WasmSpecial;

#[wasm_bindgen]
impl WasmSpecial {
    /// Gamma function Γ(x) - legacy f32 interface
    #[wasm_bindgen]
    pub fn gamma(x: f32) -> f32 {
        gamma_wasm(x as f64) as f32
    }

    /// Natural logarithm of gamma function - legacy f32 interface
    #[wasm_bindgen]
    pub fn lgamma(x: f32) -> f32 {
        lgamma_wasm(x as f64) as f32
    }

    /// Digamma function - legacy f32 interface
    #[wasm_bindgen]
    pub fn digamma(x: f32) -> f32 {
        digamma_wasm(x as f64) as f32
    }

    /// Error function - legacy f32 interface
    #[wasm_bindgen]
    pub fn erf(x: f32) -> f32 {
        erf_wasm(x as f64) as f32
    }

    /// Complementary error function - legacy f32 interface
    #[wasm_bindgen]
    pub fn erfc(x: f32) -> f32 {
        erfc_wasm(x as f64) as f32
    }

    /// Beta function - legacy f32 interface
    #[wasm_bindgen]
    pub fn beta(a: f32, b: f32) -> f32 {
        beta_wasm(a as f64, b as f64) as f32
    }

    /// Bessel J0 function - legacy f32 interface
    #[wasm_bindgen]
    pub fn bessel_j0(x: f32) -> f32 {
        bessel_j_wasm(0.0, x as f64) as f32
    }

    /// Bessel J1 function - legacy f32 interface
    #[wasm_bindgen]
    pub fn bessel_j1(x: f32) -> f32 {
        bessel_j_wasm(1.0, x as f64) as f32
    }
}

// Gamma Functions / ガンマ関数
#[wasm_bindgen]
pub fn gamma_wasm(x: f64) -> f64 {
    gamma::gamma_scalar(x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn lgamma_wasm(x: f64) -> f64 {
    gamma::lgamma_scalar(x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn digamma_wasm(x: f64) -> f64 {
    gamma::digamma_scalar(x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn beta_wasm(a: f64, b: f64) -> f64 {
    gamma::beta(a, b).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn lbeta_wasm(a: f64, b: f64) -> f64 {
    gamma::lbeta(a, b).unwrap_or(f64::NAN)
}

// Bessel Functions / ベッセル関数
#[wasm_bindgen]
pub fn bessel_j_wasm(n: f64, x: f64) -> f64 {
    bessel::bessel_j_scalar(n, x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn bessel_y_wasm(n: f64, x: f64) -> f64 {
    bessel::bessel_y_scalar(n, x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn bessel_i_wasm(n: f64, x: f64) -> f64 {
    bessel::bessel_i_scalar(n, x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn bessel_k_wasm(n: f64, x: f64) -> f64 {
    bessel::bessel_k_scalar(n, x).unwrap_or(f64::NAN)
}

// Error Functions / 誤差関数
#[wasm_bindgen]
pub fn erf_wasm(x: f64) -> f64 {
    error::erf_scalar(x)
}

#[wasm_bindgen]
pub fn erfc_wasm(x: f64) -> f64 {
    error::erfc_scalar(x)
}

#[wasm_bindgen]
pub fn erfinv_wasm(x: f64) -> f64 {
    error::erfinv_scalar(x).unwrap_or(f64::NAN)
}

#[wasm_bindgen]
pub fn erfcinv_wasm(x: f64) -> f64 {
    error::erfcinv_scalar(x).unwrap_or(f64::NAN)
}

// Vectorized operations for arrays / 配列向けベクトル化演算
#[wasm_bindgen]
pub fn gamma_array_wasm(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| gamma_wasm(x)).collect()
}

#[wasm_bindgen]
pub fn bessel_j_array_wasm(n: f64, values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| bessel_j_wasm(n, x)).collect()
}

#[wasm_bindgen]
pub fn erf_array_wasm(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| erf_wasm(x)).collect()
}

// Special function utilities / 特殊関数ユーティリティ
#[wasm_bindgen]
pub fn factorial_wasm(n: u32) -> f64 {
    if n == 0 || n == 1 {
        1.0
    } else {
        gamma_wasm(n as f64 + 1.0)
    }
}

#[wasm_bindgen]
pub fn log_factorial_wasm(n: u32) -> f64 {
    if n == 0 || n == 1 {
        0.0
    } else {
        lgamma_wasm(n as f64 + 1.0)
    }
}

// Performance optimized batch operations / パフォーマンス最適化バッチ演算
#[wasm_bindgen]
pub struct SpecialFunctionsBatch {
    cache_size: usize,
}

#[wasm_bindgen]
impl SpecialFunctionsBatch {
    #[wasm_bindgen(constructor)]
    pub fn new(cache_size: usize) -> SpecialFunctionsBatch {
        SpecialFunctionsBatch { cache_size }
    }

    #[wasm_bindgen]
    pub fn gamma_batch(&self, values: &[f64]) -> Vec<f64> {
        gamma_array_wasm(values)
    }

    #[wasm_bindgen]
    pub fn bessel_j0_batch(&self, values: &[f64]) -> Vec<f64> {
        bessel_j_array_wasm(0.0, values)
    }

    #[wasm_bindgen]
    pub fn erf_batch(&self, values: &[f64]) -> Vec<f64> {
        erf_array_wasm(values)
    }
}

// Mathematical constants / 数学定数
#[wasm_bindgen]
pub fn euler_gamma() -> f64 {
    0.5772156649015329 // Euler-Mascheroni constant
}

#[wasm_bindgen]
pub fn sqrt_2pi() -> f64 {
    2.5066282746310005 // sqrt(2π)
}

#[wasm_bindgen]
pub fn log_sqrt_2pi() -> f64 {
    0.9189385332046727 // log(sqrt(2π))
}
