//! Normalization layers for WASM
//! WASM用の正規化レイヤー

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use std::collections::HashMap;

/// Batch Normalization layer for WASM
/// WASM用のバッチ正規化レイヤー
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmBatchNorm {
    num_features: usize,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    momentum: f32,
    epsilon: f32,
    gamma: Vec<f32>, // Scale parameter
    beta: Vec<f32>,  // Shift parameter
    training: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmBatchNorm {
    /// Create a new Batch Normalization layer
    /// 新しいバッチ正規化レイヤーを作成
    #[wasm_bindgen(constructor)]
    pub fn new(num_features: usize, momentum: f32, epsilon: f32) -> Self {
        Self {
            num_features,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            momentum,
            epsilon,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            training: true,
        }
    }

    /// Set training mode
    /// 訓練モードを設定
    #[wasm_bindgen]
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Set scale (gamma) parameters
    /// スケール（ガンマ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_gamma(&mut self, gamma: Vec<f32>) {
        if gamma.len() == self.num_features {
            self.gamma = gamma;
        }
    }

    /// Set shift (beta) parameters
    /// シフト（ベータ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_beta(&mut self, beta: Vec<f32>) {
        if beta.len() == self.num_features {
            self.beta = beta;
        }
    }

    /// Forward pass through batch normalization
    /// バッチ正規化の順伝播
    #[wasm_bindgen]
    pub fn forward(&mut self, input: Vec<f32>, batch_size: usize) -> Vec<f32> {
        if input.len() != batch_size * self.num_features {
            panic!("Input size mismatch: expected {} elements", batch_size * self.num_features);
        }

        let mut output = vec![0.0; input.len()];

        if self.training {
            // Compute batch statistics
            let mut batch_mean = vec![0.0; self.num_features];
            let mut batch_var = vec![0.0; self.num_features];

            // Calculate mean across batch
            for feature in 0..self.num_features {
                let mut sum = 0.0;
                for batch in 0..batch_size {
                    sum += input[batch * self.num_features + feature];
                }
                batch_mean[feature] = sum / batch_size as f32;
            }

            // Calculate variance across batch
            for feature in 0..self.num_features {
                let mut sum_sq_diff = 0.0;
                for batch in 0..batch_size {
                    let diff = input[batch * self.num_features + feature] - batch_mean[feature];
                    sum_sq_diff += diff * diff;
                }
                batch_var[feature] = sum_sq_diff / batch_size as f32;
            }

            // Update running statistics
            for feature in 0..self.num_features {
                self.running_mean[feature] = self.momentum * self.running_mean[feature] 
                    + (1.0 - self.momentum) * batch_mean[feature];
                self.running_var[feature] = self.momentum * self.running_var[feature] 
                    + (1.0 - self.momentum) * batch_var[feature];
            }

            // Normalize using batch statistics
            for batch in 0..batch_size {
                for feature in 0..self.num_features {
                    let idx = batch * self.num_features + feature;
                    let normalized = (input[idx] - batch_mean[feature]) 
                        / (batch_var[feature] + self.epsilon).sqrt();
                    output[idx] = self.gamma[feature] * normalized + self.beta[feature];
                }
            }
        } else {
            // Use running statistics for inference
            for batch in 0..batch_size {
                for feature in 0..self.num_features {
                    let idx = batch * self.num_features + feature;
                    let normalized = (input[idx] - self.running_mean[feature]) 
                        / (self.running_var[feature] + self.epsilon).sqrt();
                    output[idx] = self.gamma[feature] * normalized + self.beta[feature];
                }
            }
        }

        output
    }

    /// Get running mean for inspection
    /// 実行中の平均値を取得（検査用）
    #[wasm_bindgen]
    pub fn get_running_mean(&self) -> Vec<f32> {
        self.running_mean.clone()
    }

    /// Get running variance for inspection
    /// 実行中の分散値を取得（検査用）
    #[wasm_bindgen]
    pub fn get_running_var(&self) -> Vec<f32> {
        self.running_var.clone()
    }
}

/// Layer Normalization for WASM
/// WASM用のレイヤー正規化
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLayerNorm {
    normalized_shape: Vec<usize>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    epsilon: f32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLayerNorm {
    /// Create a new Layer Normalization layer
    /// 新しいレイヤー正規化レイヤーを作成
    #[wasm_bindgen(constructor)]
    pub fn new(normalized_shape: Vec<usize>, epsilon: f32) -> Self {
        let num_elements = normalized_shape.iter().product();
        Self {
            normalized_shape,
            gamma: vec![1.0; num_elements],
            beta: vec![0.0; num_elements],
            epsilon,
        }
    }

    /// Set scale (gamma) parameters
    /// スケール（ガンマ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_gamma(&mut self, gamma: Vec<f32>) {
        if gamma.len() == self.gamma.len() {
            self.gamma = gamma;
        }
    }

    /// Set shift (beta) parameters
    /// シフト（ベータ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_beta(&mut self, beta: Vec<f32>) {
        if beta.len() == self.beta.len() {
            self.beta = beta;
        }
    }

    /// Forward pass through layer normalization
    /// レイヤー正規化の順伝播
    #[wasm_bindgen]
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        let normalized_size = self.normalized_shape.iter().product();
        let batch_size = input.len() / normalized_size;
        let mut output = vec![0.0; input.len()];

        for batch in 0..batch_size {
            let start_idx = batch * normalized_size;
            let end_idx = start_idx + normalized_size;
            let batch_slice = &input[start_idx..end_idx];

            // Calculate mean and variance for this sample
            let mean = batch_slice.iter().sum::<f32>() / normalized_size as f32;
            let variance = batch_slice.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / normalized_size as f32;

            let std_dev = (variance + self.epsilon).sqrt();

            // Normalize and apply learnable parameters
            for i in 0..normalized_size {
                let normalized = (batch_slice[i] - mean) / std_dev;
                output[start_idx + i] = self.gamma[i] * normalized + self.beta[i];
            }
        }

        output
    }
}

/// Group Normalization for WASM
/// WASM用のグループ正規化
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmGroupNorm {
    num_groups: usize,
    num_channels: usize,
    epsilon: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmGroupNorm {
    /// Create a new Group Normalization layer
    /// 新しいグループ正規化レイヤーを作成
    #[wasm_bindgen(constructor)]
    pub fn new(num_groups: usize, num_channels: usize, epsilon: f32) -> Self {
        if num_channels % num_groups != 0 {
            panic!("num_channels must be divisible by num_groups");
        }

        Self {
            num_groups,
            num_channels,
            epsilon,
            gamma: vec![1.0; num_channels],
            beta: vec![0.0; num_channels],
        }
    }

    /// Set scale (gamma) parameters
    /// スケール（ガンマ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_gamma(&mut self, gamma: Vec<f32>) {
        if gamma.len() == self.num_channels {
            self.gamma = gamma;
        }
    }

    /// Set shift (beta) parameters
    /// シフト（ベータ）パラメータを設定
    #[wasm_bindgen]
    pub fn set_beta(&mut self, beta: Vec<f32>) {
        if beta.len() == self.num_channels {
            self.beta = beta;
        }
    }

    /// Forward pass through group normalization
    /// グループ正規化の順伝播
    #[wasm_bindgen]
    pub fn forward(&self, input: Vec<f32>, batch_size: usize, height: usize, width: usize) -> Vec<f32> {
        let expected_size = batch_size * self.num_channels * height * width;
        if input.len() != expected_size {
            panic!("Input size mismatch: expected {}, got {}", expected_size, input.len());
        }

        let channels_per_group = self.num_channels / self.num_groups;
        let group_size = channels_per_group * height * width;
        let mut output = vec![0.0; input.len()];

        for batch in 0..batch_size {
            for group in 0..self.num_groups {
                // Calculate group statistics
                let mut group_sum = 0.0;
                let mut group_count = 0;

                for ch in 0..channels_per_group {
                    let channel = group * channels_per_group + ch;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((batch * self.num_channels + channel) * height + h) * width + w;
                            group_sum += input[idx];
                            group_count += 1;
                        }
                    }
                }

                let group_mean = group_sum / group_count as f32;

                // Calculate group variance
                let mut group_var_sum = 0.0;
                for ch in 0..channels_per_group {
                    let channel = group * channels_per_group + ch;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((batch * self.num_channels + channel) * height + h) * width + w;
                            let diff = input[idx] - group_mean;
                            group_var_sum += diff * diff;
                        }
                    }
                }

                let group_var = group_var_sum / group_count as f32;
                let group_std = (group_var + self.epsilon).sqrt();

                // Normalize group
                for ch in 0..channels_per_group {
                    let channel = group * channels_per_group + ch;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((batch * self.num_channels + channel) * height + h) * width + w;
                            let normalized = (input[idx] - group_mean) / group_std;
                            output[idx] = self.gamma[channel] * normalized + self.beta[channel];
                        }
                    }
                }
            }
        }

        output
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_batch_norm() {
        let mut batch_norm = WasmBatchNorm::new(2, 0.1, 1e-5);
        
        // Simple test data: batch_size=2, features=2
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]
        let output = batch_norm.forward(input, 2);
        
        // Output should have same shape
        assert_eq!(output.len(), 4);
        
        // Check running statistics updated
        let mean = batch_norm.get_running_mean();
        let var = batch_norm.get_running_var();
        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
    }

    #[wasm_bindgen_test]
    fn test_layer_norm() {
        let layer_norm = WasmLayerNorm::new(vec![2], 1e-5);
        
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples of 2 features each
        let output = layer_norm.forward(input);
        
        assert_eq!(output.len(), 4);
        
        // Each pair should be normalized independently
        // [1,2] -> normalized, [3,4] -> normalized
        assert!((output[0] + output[1]).abs() < 1e-5); // Mean should be ~0
        assert!((output[2] + output[3]).abs() < 1e-5); // Mean should be ~0
    }

    #[wasm_bindgen_test]
    fn test_group_norm() {
        let group_norm = WasmGroupNorm::new(2, 4, 1e-5); // 2 groups, 4 channels
        
        // Test with minimal 1x1 spatial dimensions
        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, channels=4, h=1, w=1
        let output = group_norm.forward(input, 1, 1, 1);
        
        assert_eq!(output.len(), 4);
    }
}