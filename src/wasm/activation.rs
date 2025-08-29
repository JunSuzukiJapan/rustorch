//! Neural network activation functions for WASM
//! WASM用ニューラルネットワーク活性化関数

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys;

/// WASM-compatible activation functions
/// WASM互換活性化関数
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmActivation;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmActivation {
    /// ReLU (Rectified Linear Unit) activation function
    /// ReLU(x) = max(0, x)
    #[wasm_bindgen]
    pub fn relu(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| if x > 0.0 { x } else { 0.0 })
            .collect()
    }

    /// ReLU derivative for backward pass
    /// ReLUの微分（逆伝播用）
    #[wasm_bindgen]
    pub fn relu_derivative(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Leaky ReLU activation function
    /// Leaky ReLU(x) = max(alpha * x, x)
    #[wasm_bindgen]
    pub fn leaky_relu(input: Vec<f32>, alpha: f32) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| if x > 0.0 { x } else { alpha * x })
            .collect()
    }

    /// Leaky ReLU derivative
    #[wasm_bindgen]
    pub fn leaky_relu_derivative(input: Vec<f32>, alpha: f32) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| if x > 0.0 { 1.0 } else { alpha })
            .collect()
    }

    /// Sigmoid activation function
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    #[wasm_bindgen]
    pub fn sigmoid(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                // Numerical stability: clip x to avoid overflow
                let clipped_x = x.max(-88.0).min(88.0);
                1.0 / (1.0 + (-clipped_x).exp())
            })
            .collect()
    }

    /// Sigmoid derivative
    /// σ'(x) = σ(x) * (1 - σ(x))
    #[wasm_bindgen]
    pub fn sigmoid_derivative(input: Vec<f32>) -> Vec<f32> {
        let sigmoid_output = Self::sigmoid(input);
        sigmoid_output.into_iter().map(|s| s * (1.0 - s)).collect()
    }

    /// Tanh (Hyperbolic Tangent) activation function
    /// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    #[wasm_bindgen]
    pub fn tanh(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                // Numerical stability: clip x to avoid overflow
                let clipped_x = x.max(-88.0).min(88.0);
                clipped_x.tanh()
            })
            .collect()
    }

    /// Tanh derivative
    /// tanh'(x) = 1 - tanh²(x)
    #[wasm_bindgen]
    pub fn tanh_derivative(input: Vec<f32>) -> Vec<f32> {
        let tanh_output = Self::tanh(input);
        tanh_output.into_iter().map(|t| 1.0 - t * t).collect()
    }

    /// Softmax activation function
    /// Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    #[wasm_bindgen]
    pub fn softmax(input: Vec<f32>) -> Vec<f32> {
        if input.is_empty() {
            return input;
        }

        // Find maximum for numerical stability
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) for each element
        let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_values.iter().sum();

        // Normalize by dividing by sum
        if sum_exp > 0.0 {
            exp_values.into_iter().map(|x| x / sum_exp).collect()
        } else {
            // Fallback: uniform distribution
            vec![1.0 / input.len() as f32; input.len()]
        }
    }

    /// Log Softmax activation function (numerically stable)
    /// LogSoftmax(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    #[wasm_bindgen]
    pub fn log_softmax(input: Vec<f32>) -> Vec<f32> {
        if input.is_empty() {
            return input;
        }

        // Find maximum for numerical stability
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute log(sum(exp(x - max)))
        let log_sum_exp = input.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();

        // Compute x_i - max - log(sum(exp(x_j - max)))
        input
            .into_iter()
            .map(|x| x - max_val - log_sum_exp)
            .collect()
    }

    /// GELU (Gaussian Error Linear Unit) activation function
    /// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
    /// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    #[wasm_bindgen]
    pub fn gelu(input: Vec<f32>) -> Vec<f32> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
        const COEFF: f32 = 0.044715;

        input
            .into_iter()
            .map(|x| {
                let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect()
    }

    /// GELU derivative (approximate)
    #[wasm_bindgen]
    pub fn gelu_derivative(input: Vec<f32>) -> Vec<f32> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;

        input
            .into_iter()
            .map(|x| {
                let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val; // sech²(inner)

                0.5 * (1.0 + tanh_val)
                    + 0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x)
            })
            .collect()
    }

    /// Swish/SiLU activation function
    /// Swish(x) = x * sigmoid(x)
    #[wasm_bindgen]
    pub fn swish(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                let clipped_x = x.max(-88.0).min(88.0);
                let sigmoid = 1.0 / (1.0 + (-clipped_x).exp());
                x * sigmoid
            })
            .collect()
    }

    /// Mish activation function
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    #[wasm_bindgen]
    pub fn mish(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                let clipped_x = x.max(-88.0).min(88.0);
                let softplus = (1.0 + clipped_x.exp()).ln();
                x * softplus.tanh()
            })
            .collect()
    }

    /// ELU (Exponential Linear Unit) activation function
    /// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    #[wasm_bindgen]
    pub fn elu(input: Vec<f32>, alpha: f32) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                if x > 0.0 {
                    x
                } else {
                    let clipped_x = x.max(-88.0);
                    alpha * (clipped_x.exp() - 1.0)
                }
            })
            .collect()
    }

    /// ELU derivative
    #[wasm_bindgen]
    pub fn elu_derivative(input: Vec<f32>, alpha: f32) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                if x > 0.0 {
                    1.0
                } else {
                    let clipped_x = x.max(-88.0);
                    alpha * clipped_x.exp()
                }
            })
            .collect()
    }

    /// Softplus activation function
    /// Softplus(x) = ln(1 + exp(x))
    #[wasm_bindgen]
    pub fn softplus(input: Vec<f32>) -> Vec<f32> {
        input
            .into_iter()
            .map(|x| {
                let clipped_x = x.max(-88.0).min(88.0);
                (1.0 + clipped_x.exp()).ln()
            })
            .collect()
    }

    /// Softsign activation function
    /// Softsign(x) = x / (1 + |x|)
    #[wasm_bindgen]
    pub fn softsign(input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| x / (1.0 + x.abs())).collect()
    }

    /// Apply activation function to 2D data (batch processing)
    /// 2Dデータに活性化関数を適用（バッチ処理）
    #[wasm_bindgen]
    pub fn relu_2d(input: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
        if input.len() != rows * cols {
            panic!("Input size doesn't match dimensions");
        }
        Self::relu(input)
    }

    /// Apply softmax along specified axis for 2D data
    /// 2Dデータの指定軸に沿ってソフトマックスを適用
    #[wasm_bindgen]
    pub fn softmax_2d(input: Vec<f32>, rows: usize, cols: usize, axis: usize) -> Vec<f32> {
        if input.len() != rows * cols {
            panic!("Input size doesn't match dimensions");
        }

        let mut result = vec![0.0; input.len()];

        if axis == 0 {
            // Apply softmax along rows (each column independently)
            for col in 0..cols {
                let column: Vec<f32> = (0..rows).map(|row| input[row * cols + col]).collect();
                let softmax_col = Self::softmax(column);
                for (row, &val) in softmax_col.iter().enumerate() {
                    result[row * cols + col] = val;
                }
            }
        } else if axis == 1 {
            // Apply softmax along columns (each row independently)
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                let row_data = input[start..end].to_vec();
                let softmax_row = Self::softmax(row_data);
                result[start..end].copy_from_slice(&softmax_row);
            }
        } else {
            panic!("Invalid axis for 2D data: {}", axis);
        }

        result
    }

    /// Combined activation function selector
    /// 活性化関数セレクター
    #[wasm_bindgen]
    pub fn apply_activation(input: Vec<f32>, activation_type: &str) -> Vec<f32> {
        match activation_type.to_lowercase().as_str() {
            "relu" => Self::relu(input),
            "sigmoid" => Self::sigmoid(input),
            "tanh" => Self::tanh(input),
            "softmax" => Self::softmax(input),
            "gelu" => Self::gelu(input),
            "swish" | "silu" => Self::swish(input),
            "mish" => Self::mish(input),
            "softplus" => Self::softplus(input),
            "softsign" => Self::softsign(input),
            _ => {
                web_sys::console::warn_1(
                    &format!("Unknown activation type: {}, using ReLU", activation_type).into(),
                );
                Self::relu(input)
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let output = WasmActivation::relu(input);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        let input = vec![-2.0, 0.0, 2.0];
        let output = WasmActivation::sigmoid(input);

        // Check bounds: all values should be between 0 and 1
        for &val in &output {
            assert!(val > 0.0 && val < 1.0);
        }

        // Check sigmoid(0) ≈ 0.5
        assert!((output[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let input = vec![-2.0, 0.0, 2.0];
        let output = WasmActivation::tanh(input);

        // Check bounds: all values should be between -1 and 1
        for &val in &output {
            assert!(val > -1.0 && val < 1.0);
        }

        // Check tanh(0) = 0
        assert!(output[1].abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = WasmActivation::softmax(input);

        // Check sum ≈ 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check all values are positive
        for &val in &output {
            assert!(val > 0.0);
        }

        // Check monotonicity (larger input -> larger output)
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn test_leaky_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let alpha = 0.1;
        let output = WasmActivation::leaky_relu(input, alpha);
        let expected = vec![-0.2, -0.1, 0.0, 1.0, 2.0];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu() {
        let input = vec![-1.0, 0.0, 1.0];
        let output = WasmActivation::gelu(input);

        // GELU(0) ≈ 0
        assert!(output[1].abs() < 1e-3);

        // GELU should be approximately x for large positive x
        let large_input = vec![5.0];
        let large_output = WasmActivation::gelu(large_input.clone());
        assert!((large_output[0] - large_input[0]).abs() < 0.1);
    }

    #[test]
    fn test_softmax_2d() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let output = WasmActivation::softmax_2d(input, 2, 2, 1);

        // Check that each row sums to 1
        let row1_sum = output[0] + output[1];
        let row2_sum = output[2] + output[3];

        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_selector() {
        let input = vec![-1.0, 0.0, 1.0];

        let relu_output = WasmActivation::apply_activation(input.clone(), "relu");
        let sigmoid_output = WasmActivation::apply_activation(input.clone(), "sigmoid");
        let tanh_output = WasmActivation::apply_activation(input.clone(), "tanh");

        // Basic sanity checks
        assert_eq!(relu_output, vec![0.0, 0.0, 1.0]);
        assert!(sigmoid_output[1] - 0.5 < 1e-6); // sigmoid(0) = 0.5
        assert!(tanh_output[1].abs() < 1e-6); // tanh(0) = 0
    }
}
