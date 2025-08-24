//! Utility functions for special functions module
//! 特殊関数モジュールのユーティリティ関数

use num_traits::Float;

/// Compute factorial n! for small integers
pub fn factorial(n: usize) -> f64 {
    if n == 0 || n == 1 {
        return 1.0;
    }
    
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Compute double factorial n!! for small integers
pub fn double_factorial(n: usize) -> f64 {
    if n == 0 || n == 1 {
        return 1.0;
    }
    
    let mut result = 1.0;
    let mut i = n;
    while i > 1 {
        result *= i as f64;
        i -= 2;
    }
    result
}

/// Compute binomial coefficient (n choose k)
pub fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    
    let k = k.min(n - k); // Optimize by using smaller k
    
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Compute Pochhammer symbol (rising factorial) (x)_n = x(x+1)...(x+n-1)
pub fn pochhammer<T: Float>(x: T, n: usize) -> T {
    if n == 0 {
        return T::one();
    }
    
    let mut result = x;
    for i in 1..n {
        result = result * (x + T::from(i).unwrap());
    }
    result
}

/// Compute the harmonic number H_n = 1 + 1/2 + ... + 1/n
pub fn harmonic_number(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 1..=n {
        sum += 1.0 / i as f64;
    }
    sum
}

/// Compute Bernoulli numbers B_n for small n
pub fn bernoulli_number(n: usize) -> f64 {
    // Pre-computed Bernoulli numbers for efficiency
    const BERNOULLI: [f64; 20] = [
        1.0,                    // B_0
        -0.5,                   // B_1
        1.0 / 6.0,              // B_2
        0.0,                    // B_3
        -1.0 / 30.0,            // B_4
        0.0,                    // B_5
        1.0 / 42.0,             // B_6
        0.0,                    // B_7
        -1.0 / 30.0,            // B_8
        0.0,                    // B_9
        5.0 / 66.0,             // B_10
        0.0,                    // B_11
        -691.0 / 2730.0,        // B_12
        0.0,                    // B_13
        7.0 / 6.0,              // B_14
        0.0,                    // B_15
        -3617.0 / 510.0,        // B_16
        0.0,                    // B_17
        43867.0 / 798.0,        // B_18
        0.0,                    // B_19
    ];
    
    if n < BERNOULLI.len() {
        BERNOULLI[n]
    } else if n == 1 {
        -0.5
    } else if n % 2 == 1 {
        0.0 // Bernoulli numbers are zero for odd n > 1
    } else {
        // For larger n, would need to compute recursively
        // This is a placeholder - actual computation would be more complex
        0.0
    }
}

/// Check if a number is approximately an integer
pub fn is_integer<T: Float>(x: T, tolerance: T) -> bool {
    (x - x.round()).abs() < tolerance
}

/// Compute the sign of x as +1, -1, or 0
pub fn signum<T: Float>(x: T) -> T {
    if x > T::zero() {
        T::one()
    } else if x < T::zero() {
        -T::one()
    } else {
        T::zero()
    }
}

/// Lanczos approximation coefficients getter
pub fn get_lanczos_coeff(i: usize) -> f64 {
    const LANCZOS_COEF: [f64; 9] = [
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
    
    if i < LANCZOS_COEF.len() {
        LANCZOS_COEF[i]
    } else {
        0.0
    }
}

/// Compute continued fraction using Lentz's algorithm
pub fn continued_fraction<F, G>(a: F, b: G, max_iter: usize, epsilon: f64) -> Option<f64>
where
    F: Fn(usize) -> f64,
    G: Fn(usize) -> f64,
{
    let tiny = 1e-30;
    let mut f = b(0);
    
    if f.abs() < tiny {
        f = tiny;
    }
    
    let mut c = f;
    let mut d = 0.0;
    
    for i in 1..max_iter {
        let ai = a(i);
        let bi = b(i);
        
        d = bi + ai * d;
        if d.abs() < tiny {
            d = tiny;
        }
        
        c = bi + ai / c;
        if c.abs() < tiny {
            c = tiny;
        }
        
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        
        if (delta - 1.0).abs() < epsilon {
            return Some(f);
        }
    }
    
    None // Failed to converge
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
    }
    
    #[test]
    fn test_double_factorial() {
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(1), 1.0);
        assert_eq!(double_factorial(5), 15.0);  // 5!! = 5*3*1 = 15
        assert_eq!(double_factorial(6), 48.0);  // 6!! = 6*4*2 = 48
        assert_eq!(double_factorial(7), 105.0); // 7!! = 7*5*3*1 = 105
    }
    
    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1.0);
        assert_eq!(binomial_coefficient(5, 1), 5.0);
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(5, 3), 10.0);
        assert_eq!(binomial_coefficient(5, 4), 5.0);
        assert_eq!(binomial_coefficient(5, 5), 1.0);
        assert_eq!(binomial_coefficient(5, 6), 0.0);
    }
    
    #[test]
    fn test_pochhammer() {
        // (x)_0 = 1
        assert_relative_eq!(pochhammer(2.5_f64, 0), 1.0, epsilon = 1e-10);
        
        // (x)_1 = x
        assert_relative_eq!(pochhammer(2.5_f64, 1), 2.5, epsilon = 1e-10);
        
        // (x)_n = x(x+1)...(x+n-1)
        assert_relative_eq!(pochhammer(2.0_f64, 3), 24.0, epsilon = 1e-10); // 2*3*4 = 24
        assert_relative_eq!(pochhammer(0.5_f64, 3), 1.875, epsilon = 1e-10); // 0.5*1.5*2.5 = 1.875
    }
    
    #[test]
    fn test_harmonic_number() {
        assert_eq!(harmonic_number(0), 0.0);
        assert_eq!(harmonic_number(1), 1.0);
        assert_relative_eq!(harmonic_number(2), 1.5, epsilon = 1e-10);
        assert_relative_eq!(harmonic_number(3), 1.833333333333333, epsilon = 1e-10);
        assert_relative_eq!(harmonic_number(4), 2.083333333333333, epsilon = 1e-10);
    }
    
    #[test]
    fn test_bernoulli_numbers() {
        assert_eq!(bernoulli_number(0), 1.0);
        assert_eq!(bernoulli_number(1), -0.5);
        assert_relative_eq!(bernoulli_number(2), 1.0 / 6.0, epsilon = 1e-10);
        assert_eq!(bernoulli_number(3), 0.0);
        assert_relative_eq!(bernoulli_number(4), -1.0 / 30.0, epsilon = 1e-10);
        assert_eq!(bernoulli_number(5), 0.0);
        assert_relative_eq!(bernoulli_number(6), 1.0 / 42.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_is_integer() {
        assert!(is_integer(3.0_f64, 1e-10));
        assert!(is_integer(3.0000000001_f64, 1e-9));
        assert!(!is_integer(3.1_f64, 1e-10));
        assert!(!is_integer(3.0000001_f64, 1e-10));
    }
}