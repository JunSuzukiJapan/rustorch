//! Enhanced statistical distributions for WebAssembly
//! WebAssembly向け強化統計分布

use crate::distributions::{
    bernoulli::Bernoulli, beta::Beta, exponential::Exponential, gamma::Gamma, normal::Normal,
    uniform::Uniform, DistributionTrait,
};
use crate::tensor::Tensor;
use crate::wasm::math::special::*; // Import special functions
use num_traits::Float;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// WASM-compatible random number generator using Linear Congruential Generator
/// WASM互換の線形合同法乱数生成器
#[wasm_bindgen]
pub struct WasmRng {
    seed: u32,
}

#[wasm_bindgen]
impl WasmRng {
    /// Create new RNG with seed
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Generate next random u32
    #[wasm_bindgen]
    pub fn next_u32(&mut self) -> u32 {
        // Linear Congruential Generator: a=1103515245, c=12345, m=2^32
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        self.seed
    }

    /// Generate random f32 in [0, 1)
    #[wasm_bindgen]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    /// Generate random f32 in [0, 1) (alternative name for consistency)
    #[wasm_bindgen]
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }
}

// Enhanced distribution implementations / 強化分布実装

#[wasm_bindgen]
pub struct NormalDistributionWasm {
    mean: f64,
    std: f64,
}

#[wasm_bindgen]
impl NormalDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(mean: f64, std: f64) -> NormalDistributionWasm {
        NormalDistributionWasm { mean, std }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> f64 {
        let mean_tensor = Tensor::from_vec(vec![self.mean], vec![1]);
        let std_tensor = Tensor::from_vec(vec![self.std], vec![1]);
        let normal = Normal::new(mean_tensor, std_tensor, false).unwrap();
        let sample_tensor = normal.sample(Some(&[1])).unwrap();
        sample_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<f64> {
        let mean_tensor = Tensor::from_vec(vec![self.mean], vec![1]);
        let std_tensor = Tensor::from_vec(vec![self.std], vec![1]);
        let normal = Normal::new(mean_tensor, std_tensor, false).unwrap();
        (0..n)
            .map(|_| {
                let sample_tensor = normal.sample(Some(&[1])).unwrap();
                sample_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: f64) -> f64 {
        let mean_tensor = Tensor::from_vec(vec![self.mean], vec![1]);
        let std_tensor = Tensor::from_vec(vec![self.std], vec![1]);
        let normal = Normal::new(mean_tensor, std_tensor, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![x], vec![1]);
        let log_prob_tensor = normal.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn log_prob_array(&self, values: &[f64]) -> Vec<f64> {
        let mean_tensor = Tensor::from_vec(vec![self.mean], vec![1]);
        let std_tensor = Tensor::from_vec(vec![self.std], vec![1]);
        let normal = Normal::new(mean_tensor, std_tensor, false).unwrap();
        values
            .iter()
            .map(|&x| {
                let x_tensor = Tensor::from_vec(vec![x], vec![1]);
                let log_prob_tensor = normal.log_prob(&x_tensor).unwrap();
                log_prob_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        self.std * self.std
    }

    #[wasm_bindgen]
    pub fn std_dev(&self) -> f64 {
        self.std
    }
}

#[wasm_bindgen]
pub struct UniformDistributionWasm {
    low: f64,
    high: f64,
}

#[wasm_bindgen]
impl UniformDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(low: f64, high: f64) -> UniformDistributionWasm {
        UniformDistributionWasm { low, high }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> f64 {
        let low_tensor = Tensor::from_vec(vec![self.low], vec![1]);
        let high_tensor = Tensor::from_vec(vec![self.high], vec![1]);
        let uniform = Uniform::new(low_tensor, high_tensor, false).unwrap();
        let sample_tensor = uniform.sample(Some(&[1])).unwrap();
        sample_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<f64> {
        let low_tensor = Tensor::from_vec(vec![self.low], vec![1]);
        let high_tensor = Tensor::from_vec(vec![self.high], vec![1]);
        let uniform = Uniform::new(low_tensor, high_tensor, false).unwrap();
        (0..n)
            .map(|_| {
                let sample_tensor = uniform.sample(Some(&[1])).unwrap();
                sample_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: f64) -> f64 {
        let low_tensor = Tensor::from_vec(vec![self.low], vec![1]);
        let high_tensor = Tensor::from_vec(vec![self.high], vec![1]);
        let uniform = Uniform::new(low_tensor, high_tensor, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![x], vec![1]);
        let log_prob_tensor = uniform.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        (self.low + self.high) / 2.0
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        let range = self.high - self.low;
        range * range / 12.0
    }
}

#[wasm_bindgen]
pub struct ExponentialDistributionWasm {
    rate: f64,
}

#[wasm_bindgen]
impl ExponentialDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(rate: f64) -> ExponentialDistributionWasm {
        ExponentialDistributionWasm { rate }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> f64 {
        let rate_tensor = Tensor::from_vec(vec![self.rate], vec![1]);
        let exp = Exponential::new(rate_tensor, false).unwrap();
        let sample_tensor = exp.sample(Some(&[1])).unwrap();
        sample_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<f64> {
        let rate_tensor = Tensor::from_vec(vec![self.rate], vec![1]);
        let exp = Exponential::new(rate_tensor, false).unwrap();
        (0..n)
            .map(|_| {
                let sample_tensor = exp.sample(Some(&[1])).unwrap();
                sample_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: f64) -> f64 {
        let rate_tensor = Tensor::from_vec(vec![self.rate], vec![1]);
        let exp = Exponential::new(rate_tensor, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![x], vec![1]);
        let log_prob_tensor = exp.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
}

#[wasm_bindgen]
pub struct GammaDistributionWasm {
    shape: f64,
    scale: f64,
}

#[wasm_bindgen]
impl GammaDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(shape: f64, scale: f64) -> GammaDistributionWasm {
        GammaDistributionWasm { shape, scale }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> f64 {
        let shape_tensor = Tensor::from_vec(vec![self.shape], vec![1]);
        let scale_tensor = Tensor::from_vec(vec![self.scale], vec![1]);
        let gamma = Gamma::from_concentration_scale(shape_tensor, scale_tensor, false).unwrap();
        let sample_tensor = gamma.sample(Some(&[1])).unwrap();
        sample_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<f64> {
        let shape_tensor = Tensor::from_vec(vec![self.shape], vec![1]);
        let scale_tensor = Tensor::from_vec(vec![self.scale], vec![1]);
        let gamma = Gamma::from_concentration_scale(shape_tensor, scale_tensor, false).unwrap();
        (0..n)
            .map(|_| {
                let sample_tensor = gamma.sample(Some(&[1])).unwrap();
                sample_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: f64) -> f64 {
        let shape_tensor = Tensor::from_vec(vec![self.shape], vec![1]);
        let scale_tensor = Tensor::from_vec(vec![self.scale], vec![1]);
        let gamma = Gamma::from_concentration_scale(shape_tensor, scale_tensor, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![x], vec![1]);
        let log_prob_tensor = gamma.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        self.shape * self.scale
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
}

#[wasm_bindgen]
pub struct BetaDistributionWasm {
    alpha: f64,
    beta: f64,
}

#[wasm_bindgen]
impl BetaDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64) -> BetaDistributionWasm {
        BetaDistributionWasm { alpha, beta }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> f64 {
        let alpha_tensor = Tensor::from_vec(vec![self.alpha], vec![1]);
        let beta_tensor = Tensor::from_vec(vec![self.beta], vec![1]);
        let beta = Beta::new(alpha_tensor, beta_tensor, false).unwrap();
        let sample_tensor = beta.sample(Some(&[1])).unwrap();
        sample_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<f64> {
        let alpha_tensor = Tensor::from_vec(vec![self.alpha], vec![1]);
        let beta_tensor = Tensor::from_vec(vec![self.beta], vec![1]);
        let beta = Beta::new(alpha_tensor, beta_tensor, false).unwrap();
        (0..n)
            .map(|_| {
                let sample_tensor = beta.sample(Some(&[1])).unwrap();
                sample_tensor.data[0]
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: f64) -> f64 {
        let alpha_tensor = Tensor::from_vec(vec![self.alpha], vec![1]);
        let beta_tensor = Tensor::from_vec(vec![self.beta], vec![1]);
        let beta = Beta::new(alpha_tensor, beta_tensor, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![x], vec![1]);
        let log_prob_tensor = beta.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
}

#[wasm_bindgen]
pub struct BernoulliDistributionWasm {
    p: f64,
}

#[wasm_bindgen]
impl BernoulliDistributionWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(p: f64) -> BernoulliDistributionWasm {
        BernoulliDistributionWasm { p }
    }

    #[wasm_bindgen]
    pub fn sample(&self) -> bool {
        let bernoulli = Bernoulli::from_scalar_prob(self.p, false).unwrap();
        let sample_tensor = bernoulli.sample(Some(&[1])).unwrap();
        sample_tensor.data[0] > 0.5
    }

    #[wasm_bindgen]
    pub fn sample_array(&self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let bernoulli = Bernoulli::from_scalar_prob(self.p, false).unwrap();
                let sample_tensor = bernoulli.sample(Some(&[1])).unwrap();
                if sample_tensor.data[0] > 0.5 {
                    1u8
                } else {
                    0u8
                }
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn sample_f64(&self) -> f64 {
        if self.sample() {
            1.0
        } else {
            0.0
        }
    }

    #[wasm_bindgen]
    pub fn sample_f64_array(&self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let bernoulli = Bernoulli::from_scalar_prob(self.p, false).unwrap();
                let sample_tensor = bernoulli.sample(Some(&[1])).unwrap();
                if sample_tensor.data[0] > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[wasm_bindgen]
    pub fn log_prob(&self, x: bool) -> f64 {
        let bernoulli = Bernoulli::from_scalar_prob(self.p, false).unwrap();
        let x_tensor = Tensor::from_vec(vec![if x { 1.0 } else { 0.0 }], vec![1]);
        let log_prob_tensor = bernoulli.log_prob(&x_tensor).unwrap();
        log_prob_tensor.data[0]
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        self.p
    }

    #[wasm_bindgen]
    pub fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}

// Utility functions for statistical analysis / 統計解析ユーティリティ関数
#[wasm_bindgen]
pub fn normal_cdf_wasm(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    0.5 * (1.0 + erf_wasm(z / std::f64::consts::SQRT_2))
}

#[wasm_bindgen]
pub fn normal_quantile_wasm(p: f64, mean: f64, std: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let z = std::f64::consts::SQRT_2 * erfinv_wasm(2.0 * p - 1.0);
    mean + std * z
}

// Fast statistical computations / 高速統計計算
#[wasm_bindgen]
pub fn quick_stats_wasm(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![f64::NAN; 4]; // mean, variance, skewness, kurtosis
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);

    let std_dev = variance.sqrt();

    let skewness = if std_dev > 0.0 {
        values
            .iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z
            })
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let kurtosis = if std_dev > 0.0 {
        values
            .iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z * z
            })
            .sum::<f64>()
            / n
            - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    vec![mean, variance, skewness, kurtosis]
}

// Performance testing utilities / パフォーマンステストユーティリティ
#[wasm_bindgen]
pub fn benchmark_special_functions_wasm(iterations: usize) -> Vec<f64> {
    let start = web_sys::window().unwrap().performance().unwrap().now();

    // Benchmark gamma function
    let gamma_start = web_sys::window().unwrap().performance().unwrap().now();
    for i in 0..iterations {
        let x = 1.0 + (i as f64) / (iterations as f64) * 10.0;
        gamma_wasm(x);
    }
    let gamma_time = web_sys::window().unwrap().performance().unwrap().now() - gamma_start;

    // Benchmark bessel function
    let bessel_start = web_sys::window().unwrap().performance().unwrap().now();
    for i in 0..iterations {
        let x = 0.5 + (i as f64) / (iterations as f64) * 5.0;
        bessel_j_wasm(0.0, x);
    }
    let bessel_time = web_sys::window().unwrap().performance().unwrap().now() - bessel_start;

    // Benchmark error function
    let erf_start = web_sys::window().unwrap().performance().unwrap().now();
    for i in 0..iterations {
        let x = -3.0 + (i as f64) / (iterations as f64) * 6.0;
        erf_wasm(x);
    }
    let erf_time = web_sys::window().unwrap().performance().unwrap().now() - erf_start;

    let total_time = web_sys::window().unwrap().performance().unwrap().now() - start;

    vec![gamma_time, bessel_time, erf_time, total_time]
}
