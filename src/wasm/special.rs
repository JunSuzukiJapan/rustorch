//! Special mathematical functions for WASM
//! WASM用特殊数学関数

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Gamma function implementation for WASM
/// WASM用ガンマ関数実装
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSpecial;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSpecial {
    /// Gamma function Γ(x)
    /// Using Lanczos approximation for accuracy
    #[wasm_bindgen]
    pub fn gamma(x: f32) -> f32 {
        if x < 0.5 {
            // Use reflection formula: Γ(z) * Γ(1-z) = π / sin(πz)
            std::f32::consts::PI / ((std::f32::consts::PI * x).sin() * Self::gamma(1.0 - x))
        } else {
            // Lanczos approximation
            Self::gamma_lanczos(x)
        }
    }

    /// Natural logarithm of gamma function ln(Γ(x))
    #[wasm_bindgen]
    pub fn lgamma(x: f32) -> f32 {
        if x <= 0.0 {
            f32::NAN
        } else if x < 12.0 {
            Self::gamma(x).ln()
        } else {
            // Stirling's approximation for large x
            let x_minus_half = x - 0.5;
            x_minus_half * x.ln() - x + 0.5 * (2.0 * std::f32::consts::PI).ln() + 1.0 / (12.0 * x)
        }
    }

    /// Digamma function ψ(x) = d/dx ln(Γ(x))
    #[wasm_bindgen]
    pub fn digamma(x: f32) -> f32 {
        if x <= 0.0 {
            f32::NAN
        } else if x < 6.0 {
            // Use recurrence relation: ψ(x+1) = ψ(x) + 1/x
            Self::digamma(x + 1.0) - 1.0 / x
        } else {
            // Asymptotic expansion
            let inv_x = 1.0 / x;
            let inv_x_sq = inv_x * inv_x;
            x.ln() - 0.5 * inv_x - inv_x_sq / 12.0 + inv_x_sq * inv_x_sq / 120.0
        }
    }

    /// Error function erf(x)
    #[wasm_bindgen]
    pub fn erf(x: f32) -> f32 {
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

    /// Complementary error function erfc(x) = 1 - erf(x)
    #[wasm_bindgen]
    pub fn erfc(x: f32) -> f32 {
        1.0 - Self::erf(x)
    }

    /// Beta function B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
    #[wasm_bindgen]
    pub fn beta(a: f32, b: f32) -> f32 {
        if a <= 0.0 || b <= 0.0 {
            f32::NAN
        } else {
            (Self::lgamma(a) + Self::lgamma(b) - Self::lgamma(a + b)).exp()
        }
    }

    /// Bessel function of the first kind J₀(x)
    #[wasm_bindgen]
    pub fn bessel_j0(x: f32) -> f32 {
        let x = x.abs();
        if x < 8.0 {
            // Power series for small x
            let y = x * x;
            let ans1 = 57568490574.0
                + y * (-13362590354.0
                    + y * (651619640.7
                        + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
            let ans2 = 57568490411.0
                + y * (1029532985.0
                    + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
            ans1 / ans2
        } else {
            // Asymptotic expansion for large x
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - 0.785398164;
            let p0 = 1.0;
            let p1 = -0.1098628627e-2;
            let p2 = 0.2734510407e-4;
            let p3 = -0.2073370639e-5;
            let p4 = 0.2093887211e-6;
            let q0 = -0.1562499995e-1;
            let q1 = 0.1430488765e-3;
            let q2 = -0.6911147651e-5;
            let q3 = 0.7621095161e-6;
            let q4 = -0.934945152e-7;
            let p = p0 + y * (p1 + y * (p2 + y * (p3 + y * p4)));
            let q = z * (q0 + y * (q1 + y * (q2 + y * (q3 + y * q4))));
            (2.0 / (std::f32::consts::PI * x)).sqrt() * (p * xx.cos() - q * xx.sin())
        }
    }

    /// Bessel function of the first kind J₁(x)
    #[wasm_bindgen]
    pub fn bessel_j1(x: f32) -> f32 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        if x < 8.0 {
            // Power series for small x
            let y = x * x;
            let ans1 = x
                * (72362614232.0
                    + y * (-7895059235.0
                        + y * (242396853.1
                            + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
            let ans2 = 144725228442.0
                + y * (2300535178.0
                    + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
            sign * ans1 / ans2
        } else {
            // Asymptotic expansion for large x
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - 2.356194491;
            let p0 = 1.0;
            let p1 = 0.183105e-2;
            let p2 = -0.3516396496e-4;
            let p3 = 0.2457520174e-5;
            let p4 = -0.240337019e-6;
            let q0 = 0.04687499995;
            let q1 = -0.2002690873e-3;
            let q2 = 0.8449199096e-5;
            let q3 = -0.88228987e-6;
            let q4 = 0.105787412e-6;
            let p = p0 + y * (p1 + y * (p2 + y * (p3 + y * p4)));
            let q = z * (q0 + y * (q1 + y * (q2 + y * (q3 + y * q4))));
            sign * (2.0 / (std::f32::consts::PI * x)).sqrt() * (p * xx.cos() - q * xx.sin())
        }
    }

    /// Modified Bessel function of the first kind I₀(x)
    #[wasm_bindgen]
    pub fn bessel_i0(x: f32) -> f32 {
        let x = x.abs();
        if x < 3.75 {
            // Power series for small x
            let y = (x / 3.75).powi(2);
            1.0 + y
                * (3.5156229
                    + y * (3.0899424
                        + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
        } else {
            // Asymptotic expansion for large x
            let ax = x;
            let y = 3.75 / ax;
            let ans = (ax.exp() / ax.sqrt())
                * (0.39894228
                    + y * (0.1328592e-1
                        + y * (0.225319e-2
                            + y * (-0.157565e-2
                                + y * (0.916281e-2
                                    + y * (-0.2057706e-1
                                        + y * (0.2635537e-1
                                            + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
            ans
        }
    }

    // Private helper methods
    fn gamma_lanczos(x: f32) -> f32 {
        const G: f32 = 7.0;
        const COEFF: [f32; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let z = x - 1.0;
        let mut x_acc = COEFF[0];
        for i in 1..COEFF.len() {
            x_acc += COEFF[i] / (z + i as f32);
        }

        let t = z + G + 0.5;
        (2.0 * std::f32::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x_acc
    }
}

/// Tensor-based special functions for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensorSpecial;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensorSpecial {
    /// Apply gamma function to tensor elements
    #[wasm_bindgen]
    pub fn tensor_gamma(
        tensor: &crate::wasm::tensor::WasmTensor,
    ) -> crate::wasm::tensor::WasmTensor {
        let data = tensor.data();
        let result: Vec<f32> = data.iter().map(|&x| WasmSpecial::gamma(x)).collect();
        crate::wasm::tensor::WasmTensor::new(result, tensor.shape())
    }

    /// Apply lgamma function to tensor elements
    #[wasm_bindgen]
    pub fn tensor_lgamma(
        tensor: &crate::wasm::tensor::WasmTensor,
    ) -> crate::wasm::tensor::WasmTensor {
        let data = tensor.data();
        let result: Vec<f32> = data.iter().map(|&x| WasmSpecial::lgamma(x)).collect();
        crate::wasm::tensor::WasmTensor::new(result, tensor.shape())
    }

    /// Apply erf function to tensor elements
    #[wasm_bindgen]
    pub fn tensor_erf(tensor: &crate::wasm::tensor::WasmTensor) -> crate::wasm::tensor::WasmTensor {
        let data = tensor.data();
        let result: Vec<f32> = data.iter().map(|&x| WasmSpecial::erf(x)).collect();
        crate::wasm::tensor::WasmTensor::new(result, tensor.shape())
    }

    /// Apply bessel_j0 function to tensor elements
    #[wasm_bindgen]
    pub fn tensor_bessel_j0(
        tensor: &crate::wasm::tensor::WasmTensor,
    ) -> crate::wasm::tensor::WasmTensor {
        let data = tensor.data();
        let result: Vec<f32> = data.iter().map(|&x| WasmSpecial::bessel_j0(x)).collect();
        crate::wasm::tensor::WasmTensor::new(result, tensor.shape())
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_function() {
        // Test known values
        assert!((WasmSpecial::gamma(1.0) - 1.0).abs() < 1e-6); // Γ(1) = 1
        assert!((WasmSpecial::gamma(2.0) - 1.0).abs() < 1e-6); // Γ(2) = 1
        assert!((WasmSpecial::gamma(0.5) - std::f32::consts::PI.sqrt()).abs() < 1e-4);
        // Γ(0.5) = √π
    }

    #[test]
    fn test_erf_function() {
        assert!((WasmSpecial::erf(0.0) - 0.0).abs() < 1e-6); // erf(0) = 0
        assert!((WasmSpecial::erf(1.0) - 0.8427007).abs() < 1e-4); // erf(1) ≈ 0.8427
        assert!((WasmSpecial::erf(-1.0) + 0.8427007).abs() < 1e-4); // erf(-1) ≈ -0.8427
    }

    #[test]
    fn test_bessel_functions() {
        // J₀(0) = 1
        assert!((WasmSpecial::bessel_j0(0.0) - 1.0).abs() < 1e-6);
        // J₁(0) = 0
        assert!(WasmSpecial::bessel_j1(0.0).abs() < 1e-6);
        // I₀(0) = 1
        assert!((WasmSpecial::bessel_i0(0.0) - 1.0).abs() < 1e-6);
    }
}
