//! Statistical distributions for WASM
//! WASM用統計分布

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible random number generator using Linear Congruential Generator
/// WASM互換の線形合同法乱数生成器
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmRng {
    seed: u32,
}

#[cfg(feature = "wasm")]
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

/// Normal (Gaussian) distribution for WASM
/// WASM用正規（ガウス）分布
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmNormal {
    mean: f32,
    std_dev: f32,
    rng: WasmRng,
    has_spare: bool,
    spare: f32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmNormal {
    /// Create new normal distribution
    #[wasm_bindgen(constructor)]
    pub fn new(mean: f32, std_dev: f32, seed: u32) -> Self {
        Self {
            mean,
            std_dev,
            rng: WasmRng::new(seed),
            has_spare: false,
            spare: 0.0,
        }
    }

    /// Create standard normal distribution (mean=0, std=1)
    #[wasm_bindgen]
    pub fn standard(seed: u32) -> WasmNormal {
        Self::new(0.0, 1.0, seed)
    }

    /// Sample single value using Box-Muller transform
    #[wasm_bindgen]
    pub fn sample(&mut self) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.mean + self.std_dev * self.spare;
        }

        // Box-Muller transform
        let u1 = self.rng.uniform();
        let u2 = self.rng.uniform();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();
        
        self.spare = z1;
        self.has_spare = true;
        
        self.mean + self.std_dev * z0
    }

    /// Sample multiple values
    #[wasm_bindgen]
    pub fn sample_n(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Probability density function
    #[wasm_bindgen]
    pub fn pdf(&self, x: f32) -> f32 {
        let z = (x - self.mean) / self.std_dev;
        let coefficient = 1.0 / (self.std_dev * (2.0 * std::f32::consts::PI).sqrt());
        coefficient * (-0.5 * z * z).exp()
    }

    /// Log probability density function
    #[wasm_bindgen]
    pub fn log_pdf(&self, x: f32) -> f32 {
        let z = (x - self.mean) / self.std_dev;
        let log_coeff = -(self.std_dev.ln() + 0.5 * (2.0 * std::f32::consts::PI).ln());
        log_coeff - 0.5 * z * z
    }

    /// Cumulative distribution function (using error function approximation)
    #[wasm_bindgen]
    pub fn cdf(&self, x: f32) -> f32 {
        let z = (x - self.mean) / (self.std_dev * 2_f32.sqrt());
        0.5 * (1.0 + erf_approx(z))
    }

    /// Get mean
    #[wasm_bindgen]
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Get standard deviation
    #[wasm_bindgen]
    pub fn std_dev(&self) -> f32 {
        self.std_dev
    }

    /// Get variance
    #[wasm_bindgen]
    pub fn variance(&self) -> f32 {
        self.std_dev * self.std_dev
    }
}

/// Uniform distribution for WASM
/// WASM用一様分布
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmUniform {
    low: f32,
    high: f32,
    rng: WasmRng,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmUniform {
    /// Create new uniform distribution
    #[wasm_bindgen(constructor)]
    pub fn new(low: f32, high: f32, seed: u32) -> Self {
        Self {
            low,
            high,
            rng: WasmRng::new(seed),
        }
    }

    /// Create standard uniform distribution [0, 1)
    #[wasm_bindgen]
    pub fn standard(seed: u32) -> WasmUniform {
        Self::new(0.0, 1.0, seed)
    }

    /// Sample single value
    #[wasm_bindgen]
    pub fn sample(&mut self) -> f32 {
        self.low + (self.high - self.low) * self.rng.uniform()
    }

    /// Sample multiple values
    #[wasm_bindgen]
    pub fn sample_n(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Probability density function
    #[wasm_bindgen]
    pub fn pdf(&self, x: f32) -> f32 {
        if x >= self.low && x < self.high {
            1.0 / (self.high - self.low)
        } else {
            0.0
        }
    }

    /// Log probability density function
    #[wasm_bindgen]
    pub fn log_pdf(&self, x: f32) -> f32 {
        if x >= self.low && x < self.high {
            -(self.high - self.low).ln()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Cumulative distribution function
    #[wasm_bindgen]
    pub fn cdf(&self, x: f32) -> f32 {
        if x < self.low {
            0.0
        } else if x >= self.high {
            1.0
        } else {
            (x - self.low) / (self.high - self.low)
        }
    }

    /// Get mean
    #[wasm_bindgen]
    pub fn mean(&self) -> f32 {
        0.5 * (self.low + self.high)
    }

    /// Get variance
    #[wasm_bindgen]
    pub fn variance(&self) -> f32 {
        let range = self.high - self.low;
        range * range / 12.0
    }
}

/// Bernoulli distribution for WASM
/// WASM用ベルヌーイ分布
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmBernoulli {
    p: f32,
    rng: WasmRng,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmBernoulli {
    /// Create new Bernoulli distribution
    #[wasm_bindgen(constructor)]
    pub fn new(p: f32, seed: u32) -> Result<WasmBernoulli, String> {
        if p < 0.0 || p > 1.0 {
            return Err("Probability p must be between 0 and 1".to_string());
        }
        Ok(Self {
            p,
            rng: WasmRng::new(seed),
        })
    }

    /// Sample single value (0 or 1)
    #[wasm_bindgen]
    pub fn sample(&mut self) -> u32 {
        if self.rng.uniform() < self.p { 1 } else { 0 }
    }

    /// Sample multiple values
    #[wasm_bindgen]
    pub fn sample_n(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Probability mass function
    #[wasm_bindgen]
    pub fn pmf(&self, x: u32) -> f32 {
        match x {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }

    /// Log probability mass function
    #[wasm_bindgen]
    pub fn log_pmf(&self, x: u32) -> f32 {
        match x {
            0 => (1.0 - self.p).ln(),
            1 => self.p.ln(),
            _ => f32::NEG_INFINITY,
        }
    }

    /// Get mean
    #[wasm_bindgen]
    pub fn mean(&self) -> f32 {
        self.p
    }

    /// Get variance
    #[wasm_bindgen]
    pub fn variance(&self) -> f32 {
        self.p * (1.0 - self.p)
    }
}

/// Exponential distribution for WASM
/// WASM用指数分布
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmExponential {
    rate: f32,
    rng: WasmRng,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmExponential {
    /// Create new exponential distribution
    #[wasm_bindgen(constructor)]
    pub fn new(rate: f32, seed: u32) -> Result<WasmExponential, String> {
        if rate <= 0.0 {
            return Err("Rate parameter must be positive".to_string());
        }
        Ok(Self {
            rate,
            rng: WasmRng::new(seed),
        })
    }

    /// Create standard exponential distribution (rate=1)
    #[wasm_bindgen]
    pub fn standard(seed: u32) -> WasmExponential {
        Self::new(1.0, seed).unwrap()
    }

    /// Sample single value using inverse transform sampling
    #[wasm_bindgen]
    pub fn sample(&mut self) -> f32 {
        let u = self.rng.uniform();
        -(1.0 - u).ln() / self.rate
    }

    /// Sample multiple values
    #[wasm_bindgen]
    pub fn sample_n(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Probability density function
    #[wasm_bindgen]
    pub fn pdf(&self, x: f32) -> f32 {
        if x >= 0.0 {
            self.rate * (-self.rate * x).exp()
        } else {
            0.0
        }
    }

    /// Log probability density function
    #[wasm_bindgen]
    pub fn log_pdf(&self, x: f32) -> f32 {
        if x >= 0.0 {
            self.rate.ln() - self.rate * x
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Cumulative distribution function
    #[wasm_bindgen]
    pub fn cdf(&self, x: f32) -> f32 {
        if x >= 0.0 {
            1.0 - (-self.rate * x).exp()
        } else {
            0.0
        }
    }

    /// Get mean
    #[wasm_bindgen]
    pub fn mean(&self) -> f32 {
        1.0 / self.rate
    }

    /// Get variance
    #[wasm_bindgen]
    pub fn variance(&self) -> f32 {
        1.0 / (self.rate * self.rate)
    }
}

// Helper function for error function approximation
fn erf_approx(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_rng() {
        let mut rng = WasmRng::new(12345);
        let val1 = rng.next_f32();
        let val2 = rng.next_f32();
        assert!(val1 >= 0.0 && val1 < 1.0);
        assert!(val2 >= 0.0 && val2 < 1.0);
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_normal_distribution() {
        let mut normal = WasmNormal::new(0.0, 1.0, 42);
        let sample = normal.sample();
        assert!(!sample.is_nan());
        
        let pdf_val = normal.pdf(0.0);
        assert!(pdf_val > 0.0);
        
        let cdf_val = normal.cdf(0.0);
        assert!((cdf_val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_uniform_distribution() {
        let mut uniform = WasmUniform::new(0.0, 1.0, 123);
        let sample = uniform.sample();
        assert!(sample >= 0.0 && sample < 1.0);
        
        let pdf_val = uniform.pdf(0.5);
        assert_eq!(pdf_val, 1.0);
        
        let cdf_val = uniform.cdf(0.5);
        assert_eq!(cdf_val, 0.5);
    }

    #[test]
    fn test_bernoulli_distribution() {
        let mut bernoulli = WasmBernoulli::new(0.5, 456).unwrap();
        let sample = bernoulli.sample();
        assert!(sample == 0 || sample == 1);
        
        let pmf_0 = bernoulli.pmf(0);
        let pmf_1 = bernoulli.pmf(1);
        assert_eq!(pmf_0, 0.5);
        assert_eq!(pmf_1, 0.5);
    }

    #[test]
    fn test_exponential_distribution() {
        let mut exp = WasmExponential::new(1.0, 789).unwrap();
        let sample = exp.sample();
        assert!(sample >= 0.0);
        
        let pdf_val = exp.pdf(1.0);
        assert!(pdf_val > 0.0);
        
        let mean = exp.mean();
        assert_eq!(mean, 1.0);
    }
}