//! Shared loss function traits and implementations
//! 共通損失関数トレイトと実装

use num_traits::Float;

/// Common loss function trait for both regular and WASM implementations
/// 通常実装とWASM実装の両方用の共通損失関数トレイト
pub trait LossFunction<T> {
    type Error;
    type Input;
    type Output;
    
    /// Mean Squared Error loss
    /// 平均二乗誤差損失
    fn mse_loss(&self, predictions: Self::Input, targets: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Mean Absolute Error loss
    /// 平均絶対誤差損失
    fn mae_loss(&self, predictions: Self::Input, targets: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Binary Cross Entropy loss
    /// バイナリクロスエントロピー損失
    fn bce_loss(&self, predictions: Self::Input, targets: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Cross Entropy loss
    /// クロスエントロピー損失
    fn cross_entropy_loss(&self, predictions: Self::Input, targets: Self::Input) -> Result<Self::Output, Self::Error>;
}

/// Shared loss function implementations
/// 共通損失関数実装
pub mod shared_losses {
    use super::*;
    
    /// MSE loss implementation for Vec<T>
    /// Vec<T>用MSE損失実装
    pub fn mse_loss_vec<T: Float>(predictions: &[T], targets: &[T]) -> Result<T, &'static str> {
        if predictions.len() != targets.len() {
            return Err("Predictions and targets must have the same length");
        }
        
        if predictions.is_empty() {
            return Ok(T::zero());
        }
        
        let sum_squared_error = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum_squared_error / T::from(predictions.len()).unwrap_or(T::one()))
    }
    
    /// MAE loss implementation for Vec<T>
    /// Vec<T>用MAE損失実装
    pub fn mae_loss_vec<T: Float>(predictions: &[T], targets: &[T]) -> Result<T, &'static str> {
        if predictions.len() != targets.len() {
            return Err("Predictions and targets must have the same length");
        }
        
        if predictions.is_empty() {
            return Ok(T::zero());
        }
        
        let sum_abs_error = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum_abs_error / T::from(predictions.len()).unwrap_or(T::one()))
    }
    
    /// Binary Cross Entropy loss implementation for Vec<T>
    /// Vec<T>用バイナリクロスエントロピー損失実装
    pub fn bce_loss_vec<T: Float>(predictions: &[T], targets: &[T]) -> Result<T, &'static str> {
        if predictions.len() != targets.len() {
            return Err("Predictions and targets must have the same length");
        }
        
        if predictions.is_empty() {
            return Ok(T::zero());
        }
        
        let eps = T::from(1e-7).unwrap_or(T::zero());
        let sum_loss = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                // Clamp predictions to avoid log(0)
                let clamped_pred = pred.max(eps).min(T::one() - eps);
                -(target * clamped_pred.ln() + (T::one() - target) * (T::one() - clamped_pred).ln())
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum_loss / T::from(predictions.len()).unwrap_or(T::one()))
    }
    
    /// Cross Entropy loss implementation for Vec<T>
    /// Vec<T>用クロスエントロピー損失実装
    pub fn cross_entropy_loss_vec<T: Float>(log_probs: &[T], targets: &[T]) -> Result<T, &'static str> {
        if log_probs.len() != targets.len() {
            return Err("Log probabilities and targets must have the same length");
        }
        
        if log_probs.is_empty() {
            return Ok(T::zero());
        }
        
        let sum_loss = log_probs
            .iter()
            .zip(targets.iter())
            .map(|(&log_prob, &target)| -target * log_prob)
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum_loss / T::from(log_probs.len()).unwrap_or(T::one()))
    }
}

