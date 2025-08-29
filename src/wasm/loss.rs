//! Loss functions for WASM neural networks
//! WASMニューラルネットワーク用損失関数

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys;

/// WASM-compatible loss functions
/// WASM互換損失関数
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLoss;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLoss {
    /// Mean Squared Error (MSE) loss
    /// MSE(y_pred, y_true) = mean((y_pred - y_true)²)
    #[wasm_bindgen]
    pub fn mse_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_squared_error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let diff = pred - target;
                diff * diff
            })
            .sum();

        sum_squared_error / predictions.len() as f32
    }

    /// Mean Absolute Error (MAE) loss
    /// MAE(y_pred, y_true) = mean(|y_pred - y_true|)
    #[wasm_bindgen]
    pub fn mae_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_absolute_error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum();

        sum_absolute_error / predictions.len() as f32
    }

    /// Huber loss (smooth L1 loss)
    /// Combines MSE and MAE for robustness
    #[wasm_bindgen]
    pub fn huber_loss(predictions: Vec<f32>, targets: Vec<f32>, delta: f32) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_huber_error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let diff = (pred - target).abs();
                if diff <= delta {
                    0.5 * diff * diff
                } else {
                    delta * (diff - 0.5 * delta)
                }
            })
            .sum();

        sum_huber_error / predictions.len() as f32
    }

    /// Cross-entropy loss for binary classification
    /// Binary Cross-Entropy: -mean(y*log(p) + (1-y)*log(1-p))
    #[wasm_bindgen]
    pub fn binary_cross_entropy_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let epsilon = 1e-7; // Small value to prevent log(0)

        let sum_cross_entropy: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                // Clip predictions to avoid log(0) or log(1)
                let clipped_pred = pred.max(epsilon).min(1.0 - epsilon);
                
                -target * clipped_pred.ln() - (1.0 - target) * (1.0 - clipped_pred).ln()
            })
            .sum();

        sum_cross_entropy / predictions.len() as f32
    }

    /// Cross-entropy loss for multiclass classification
    /// Input: logits (raw scores), targets (one-hot or class indices)
    #[wasm_bindgen]
    pub fn cross_entropy_loss(logits: Vec<f32>, targets: Vec<f32>) -> f32 {
        if logits.len() != targets.len() {
            panic!("Logits and targets must have the same length");
        }

        if logits.is_empty() {
            return 0.0;
        }

        // Apply softmax to logits first
        let softmax_probs = Self::softmax(&logits);
        
        let epsilon = 1e-7;
        let sum_cross_entropy: f32 = softmax_probs
            .iter()
            .zip(targets.iter())
            .map(|(prob, target)| {
                let clipped_prob = prob.max(epsilon);
                -target * clipped_prob.ln()
            })
            .sum();

        sum_cross_entropy
    }

    /// Sparse cross-entropy loss (targets as class indices instead of one-hot)
    /// logits: [batch_size * num_classes], targets: [batch_size] (class indices)
    #[wasm_bindgen]
    pub fn sparse_cross_entropy_loss(
        logits: Vec<f32>, 
        targets: Vec<u32>, 
        num_classes: usize
    ) -> f32 {
        let batch_size = targets.len();
        
        if logits.len() != batch_size * num_classes {
            panic!("Logits size must equal batch_size * num_classes");
        }

        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for (batch_idx, &target_class) in targets.iter().enumerate() {
            if target_class as usize >= num_classes {
                panic!("Target class index out of bounds");
            }

            // Extract logits for this batch sample
            let start_idx = batch_idx * num_classes;
            let batch_logits: Vec<f32> = logits[start_idx..start_idx + num_classes].to_vec();
            
            // Apply softmax
            let softmax_probs = Self::softmax(&batch_logits);
            
            // Compute cross-entropy for target class
            let epsilon = 1e-7;
            let target_prob = softmax_probs[target_class as usize].max(epsilon);
            total_loss += -target_prob.ln();
        }

        total_loss / batch_size as f32
    }

    /// KL Divergence loss
    /// KL(P||Q) = sum(P * log(P/Q))
    #[wasm_bindgen]
    pub fn kl_divergence_loss(p_distribution: Vec<f32>, q_distribution: Vec<f32>) -> f32 {
        if p_distribution.len() != q_distribution.len() {
            panic!("Distributions must have the same length");
        }

        if p_distribution.is_empty() {
            return 0.0;
        }

        let epsilon = 1e-7;

        let kl_div: f32 = p_distribution
            .iter()
            .zip(q_distribution.iter())
            .map(|(p, q)| {
                let clipped_p = p.max(epsilon);
                let clipped_q = q.max(epsilon);
                clipped_p * (clipped_p / clipped_q).ln()
            })
            .sum();

        kl_div
    }

    /// Focal loss for handling class imbalance
    /// FL(pt) = -α(1-pt)^γ log(pt)
    #[wasm_bindgen]
    pub fn focal_loss(
        predictions: Vec<f32>, 
        targets: Vec<f32>, 
        alpha: f32, 
        gamma: f32
    ) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let epsilon = 1e-7;

        let sum_focal_loss: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let clipped_pred = pred.max(epsilon).min(1.0 - epsilon);
                let pt = if *target == 1.0 { clipped_pred } else { 1.0 - clipped_pred };
                
                -alpha * (1.0 - pt).powf(gamma) * pt.ln()
            })
            .sum();

        sum_focal_loss / predictions.len() as f32
    }

    /// Cosine similarity loss
    /// Loss = 1 - cosine_similarity(pred, target)
    #[wasm_bindgen]
    pub fn cosine_similarity_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 1.0; // Maximum dissimilarity
        }

        let dot_product: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| p * t)
            .sum();

        let pred_norm: f32 = predictions.iter().map(|p| p * p).sum::<f32>().sqrt();
        let target_norm: f32 = targets.iter().map(|t| t * t).sum::<f32>().sqrt();

        if pred_norm == 0.0 || target_norm == 0.0 {
            return 1.0; // Maximum dissimilarity
        }

        let cosine_similarity = dot_product / (pred_norm * target_norm);
        1.0 - cosine_similarity
    }

    /// Hinge loss for SVM-style classification
    /// Hinge(y, f(x)) = max(0, 1 - y * f(x))
    #[wasm_bindgen]
    pub fn hinge_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_hinge_loss: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let margin = 1.0 - target * pred;
                margin.max(0.0)
            })
            .sum();

        sum_hinge_loss / predictions.len() as f32
    }

    /// Squared hinge loss (smooth version of hinge loss)
    #[wasm_bindgen]
    pub fn squared_hinge_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_squared_hinge_loss: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let margin = 1.0 - target * pred;
                margin.max(0.0).powi(2)
            })
            .sum();

        sum_squared_hinge_loss / predictions.len() as f32
    }

    /// Log-cosh loss (smooth version of MAE)
    /// LogCosh(y_pred, y_true) = mean(log(cosh(y_pred - y_true)))
    #[wasm_bindgen]
    pub fn log_cosh_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        if predictions.is_empty() {
            return 0.0;
        }

        let sum_log_cosh: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let diff = pred - target;
                // For numerical stability, use approximation for large |diff|
                if diff.abs() > 10.0 {
                    diff.abs() - (2.0_f32).ln()
                } else {
                    diff.cosh().ln()
                }
            })
            .sum();

        sum_log_cosh / predictions.len() as f32
    }

    /// Combined loss function selector
    /// 損失関数セレクター
    #[wasm_bindgen]
    pub fn compute_loss(
        predictions: Vec<f32>,
        targets: Vec<f32>,
        loss_type: &str,
    ) -> f32 {
        match loss_type.to_lowercase().as_str() {
            "mse" | "mean_squared_error" => Self::mse_loss(predictions, targets),
            "mae" | "mean_absolute_error" => Self::mae_loss(predictions, targets),
            "bce" | "binary_cross_entropy" => Self::binary_cross_entropy_loss(predictions, targets),
            "cross_entropy" => Self::cross_entropy_loss(predictions, targets),
            "huber" => Self::huber_loss(predictions, targets, 1.0), // Default delta = 1.0
            "cosine" => Self::cosine_similarity_loss(predictions, targets),
            "hinge" => Self::hinge_loss(predictions, targets),
            "squared_hinge" => Self::squared_hinge_loss(predictions, targets),
            "log_cosh" => Self::log_cosh_loss(predictions, targets),
            _ => {
                web_sys::console::warn_1(&format!("Unknown loss type: {}, using MSE", loss_type).into());
                Self::mse_loss(predictions, targets)
            }
        }
    }

    // Helper function: Softmax implementation
    fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }

        // Find maximum for numerical stability
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) for each element
        let exp_values: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_values.iter().sum();

        // Normalize by dividing by sum
        if sum_exp > 0.0 {
            exp_values.into_iter().map(|x| x / sum_exp).collect()
        } else {
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }

    /// Get loss function gradient for backpropagation
    /// 逆伝播用の損失関数勾配を取得
    #[wasm_bindgen]
    pub fn loss_gradient(
        predictions: Vec<f32>,
        targets: Vec<f32>,
        loss_type: &str,
    ) -> Vec<f32> {
        if predictions.len() != targets.len() {
            panic!("Predictions and targets must have the same length");
        }

        match loss_type.to_lowercase().as_str() {
            "mse" | "mean_squared_error" => {
                // MSE gradient: 2(pred - target) / n
                let n = predictions.len() as f32;
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(pred, target)| 2.0 * (pred - target) / n)
                    .collect()
            }
            "mae" | "mean_absolute_error" => {
                // MAE gradient: sign(pred - target) / n
                let n = predictions.len() as f32;
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(pred, target)| (pred - target).signum() / n)
                    .collect()
            }
            "bce" | "binary_cross_entropy" => {
                // BCE gradient: (pred - target) / (pred * (1 - pred))
                let epsilon = 1e-7;
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(pred, target)| {
                        let clipped_pred = pred.max(epsilon).min(1.0 - epsilon);
                        (clipped_pred - target) / (clipped_pred * (1.0 - clipped_pred))
                    })
                    .collect()
            }
            _ => {
                // Default to MSE gradient
                let n = predictions.len() as f32;
                predictions
                    .iter()
                    .zip(targets.iter())
                    .map(|(pred, target)| 2.0 * (pred - target) / n)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        let loss = WasmLoss::mse_loss(predictions, targets);
        assert!((loss - 0.0).abs() < 1e-6); // Perfect predictions should give 0 loss

        let predictions2 = vec![1.0, 2.0, 3.0];
        let targets2 = vec![2.0, 3.0, 4.0];
        let loss2 = WasmLoss::mse_loss(predictions2, targets2);
        assert!((loss2 - 1.0).abs() < 1e-6); // Each prediction is off by 1, so MSE = 1
    }

    #[test]
    fn test_mae_loss() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![2.0, 3.0, 4.0];
        let loss = WasmLoss::mae_loss(predictions, targets);
        assert!((loss - 1.0).abs() < 1e-6); // Each prediction is off by 1, so MAE = 1
    }

    #[test]
    fn test_binary_cross_entropy_loss() {
        let predictions = vec![0.9, 0.1, 0.8];
        let targets = vec![1.0, 0.0, 1.0];
        let loss = WasmLoss::binary_cross_entropy_loss(predictions, targets);
        assert!(loss > 0.0);
        assert!(loss < 1.0); // Should be relatively low for good predictions
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Test with one-hot targets
        let logits = vec![2.0, 1.0, 0.1];
        let targets = vec![1.0, 0.0, 0.0]; // First class
        let loss = WasmLoss::cross_entropy_loss(logits, targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_sparse_cross_entropy_loss() {
        let logits = vec![2.0, 1.0, 0.1, 0.5, 2.5, 1.0]; // 2 samples, 3 classes each
        let targets = vec![0, 1]; // First sample: class 0, second sample: class 1
        let loss = WasmLoss::sparse_cross_entropy_loss(logits, targets, 3);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = vec![1.0, 2.0, 10.0];
        let targets = vec![1.0, 2.0, 5.0];
        let delta = 1.0;
        let loss = WasmLoss::huber_loss(predictions, targets, delta);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_cosine_similarity_loss() {
        // Identical vectors should have loss close to 0
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        let loss = WasmLoss::cosine_similarity_loss(predictions, targets);
        assert!(loss < 1e-6);

        // Orthogonal vectors should have loss close to 1
        let predictions2 = vec![1.0, 0.0];
        let targets2 = vec![0.0, 1.0];
        let loss2 = WasmLoss::cosine_similarity_loss(predictions2, targets2);
        assert!((loss2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_selector() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 2.1, 3.1];
        
        let mse_loss = WasmLoss::compute_loss(predictions.clone(), targets.clone(), "mse");
        let mae_loss = WasmLoss::compute_loss(predictions.clone(), targets.clone(), "mae");
        
        assert!(mse_loss > 0.0);
        assert!(mae_loss > 0.0);
    }

    #[test]
    fn test_loss_gradients() {
        let predictions = vec![1.0, 2.0];
        let targets = vec![0.5, 1.5];
        
        let gradients = WasmLoss::loss_gradient(predictions, targets, "mse");
        assert_eq!(gradients.len(), 2);
        
        // MSE gradient should be 2*(pred - target)/n
        let expected_grad_1 = 2.0 * (1.0 - 0.5) / 2.0; // = 0.5
        let expected_grad_2 = 2.0 * (2.0 - 1.5) / 2.0; // = 0.5
        
        assert!((gradients[0] - expected_grad_1).abs() < 1e-6);
        assert!((gradients[1] - expected_grad_2).abs() < 1e-6);
    }
}