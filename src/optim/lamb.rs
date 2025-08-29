//! LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer
//! LAMB（バッチ訓練用レイヤーワイズ適応モーメント）オプティマイザー
//!
//! LAMB is designed for large batch training and achieves better performance
//! than Adam/AdamW on large batch sizes by using layer-wise adaptation.
//!
//! LAMBは大規模バッチ訓練向けに設計され、レイヤーワイズ適応を使用することで
//! 大きなバッチサイズでAdam/AdamWより優れた性能を実現します。

use super::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer
/// 
/// LAMB combines the benefits of Adam with layer-wise adaptation, making it particularly
/// effective for large batch training scenarios commonly used in distributed training.
#[derive(Debug, Clone)]
pub struct LAMB {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    bias_correction: bool,
    step_count: usize,
    exp_avg: HashMap<usize, Tensor<f32>>,      // First moment estimate
    exp_avg_sq: HashMap<usize, Tensor<f32>>,   // Second moment estimate
}

impl LAMB {
    /// Create new LAMB optimizer with default parameters
    /// デフォルトパラメータで新しいLAMBオプティマイザーを作成
    pub fn new(learning_rate: f32) -> Self {
        Self::with_params(learning_rate, 0.9, 0.999, 1e-6, 0.01)
    }
    
    /// Create LAMB optimizer with custom parameters
    /// カスタムパラメータでLAMBオプティマイザーを作成
    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            bias_correction: true,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
    
    /// Create LAMB optimizer without bias correction
    /// バイアス補正なしでLAMBオプティマイザーを作成
    pub fn without_bias_correction(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        let mut optimizer = Self::with_params(learning_rate, beta1, beta2, epsilon, weight_decay);
        optimizer.bias_correction = false;
        optimizer
    }
    
    /// Set bias correction option
    /// バイアス補正オプションを設定
    pub fn set_bias_correction(&mut self, bias_correction: bool) {
        self.bias_correction = bias_correction;
    }
    
    /// Get current step count
    /// 現在のステップ数を取得
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

impl Optimizer for LAMB {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        self.step_count += 1;

        let mut d_p = grad.clone();

        // Apply weight decay to gradients (L2 regularization)
        // 勾配に重み減衰を適用（L2正則化）
        if self.weight_decay != 0.0 {
            let weight_decay_term = param * self.weight_decay;
            d_p = &d_p + &weight_decay_term;
        }

        // Get or initialize momentum buffers
        // モーメンタムバッファを取得または初期化
        let exp_avg = if let Some(avg) = self.exp_avg.get(&param_id) {
            let beta1_term = avg * self.beta1;
            let one_minus_beta1_term = &d_p * (1.0 - self.beta1);
            &beta1_term + &one_minus_beta1_term
        } else {
            d_p.clone() * (1.0 - self.beta1)
        };

        let exp_avg_sq = if let Some(avg_sq) = self.exp_avg_sq.get(&param_id) {
            let beta2_term = avg_sq * self.beta2;
            let d_p_squared = &d_p * &d_p;
            let one_minus_beta2_term = &d_p_squared * (1.0 - self.beta2);
            &beta2_term + &one_minus_beta2_term
        } else {
            let d_p_squared = &d_p * &d_p;
            d_p_squared * (1.0 - self.beta2)
        };

        // Store updated momentum buffers
        // 更新されたモーメンタムバッファを保存
        self.exp_avg.insert(param_id, exp_avg.clone());
        self.exp_avg_sq.insert(param_id, exp_avg_sq.clone());

        // Compute bias-corrected first and second moment estimates
        // バイアス補正済み一次・二次モーメント推定値を計算
        let (corrected_exp_avg, corrected_exp_avg_sq) = if self.bias_correction {
            let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
            
            let corrected_avg = &exp_avg / bias_correction1;
            let corrected_avg_sq = &exp_avg_sq / bias_correction2;
            (corrected_avg, corrected_avg_sq)
        } else {
            (exp_avg.clone(), exp_avg_sq.clone())
        };

        // Compute the adaptive update
        // 適応更新を計算
        let sqrt_corrected_exp_avg_sq = corrected_exp_avg_sq.sqrt();
        let adaptive_update = corrected_exp_avg / (&sqrt_corrected_exp_avg_sq + self.epsilon);

        // LAMB's layer-wise adaptation: compute norms and trust ratio
        // LAMBのレイヤーワイズ適応：ノルムと信頼比を計算
        // Simplified norm calculation using sum of squares
        let param_squared = param * param;
        let param_norm = param_squared.sum().sqrt();
        
        let update_squared = &adaptive_update * &adaptive_update;
        let update_norm = update_squared.sum().sqrt();
        
        let trust_ratio = if update_norm > 0.0 {
            param_norm / update_norm
        } else {
            1.0
        };

        // Apply trust ratio clipping (optional, helps with stability)
        // 信頼比クリッピングを適用（オプション、安定性に寄与）
        let clipped_trust_ratio = trust_ratio.min(10.0).max(0.1);

        // Final parameter update with layer-wise adaptation
        // レイヤーワイズ適応による最終パラメータ更新
        let scaled_update = &adaptive_update * (self.learning_rate * clipped_trust_ratio);
        let updated_param = param - &scaled_update;
        
        param.copy_from(&updated_param);
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, f32> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("bias_correction".to_string(), if self.bias_correction { 1.0 } else { 0.0 });
        state.insert("step_count".to_string(), self.step_count as f32);
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, f32>) {
        if let Some(&lr) = state.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&beta1) = state.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&epsilon) = state.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&bias_correction) = state.get("bias_correction") {
            self.bias_correction = bias_correction > 0.5;
        }
        if let Some(&step_count) = state.get("step_count") {
            self.step_count = step_count as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_lamb_creation() {
        let optimizer = LAMB::new(0.001);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_lamb_with_params() {
        let optimizer = LAMB::with_params(0.01, 0.8, 0.95, 1e-5, 0.02);
        assert_eq!(optimizer.learning_rate(), 0.01);
        assert_eq!(optimizer.beta1, 0.8);
        assert_eq!(optimizer.beta2, 0.95);
        assert_eq!(optimizer.epsilon, 1e-5);
        assert_eq!(optimizer.weight_decay, 0.02);
    }

    #[test]
    fn test_lamb_step() {
        let mut optimizer = LAMB::new(0.1);
        let param = Tensor::<f32>::ones(&[2, 2]);
        let grad = Tensor::<f32>::ones(&[2, 2]) * 0.1;

        // Initial parameter values
        let initial_param = param.clone();
        
        // Perform optimization step
        optimizer.step(&param, &grad);
        
        // Verify step count increased
        assert_eq!(optimizer.step_count(), 1);
        
        // Verify parameters were updated (should be different from initial)
        let updated_data = param.data.as_slice().unwrap();
        let initial_data = initial_param.data.as_slice().unwrap();
        
        // Parameters should have changed
        assert_ne!(updated_data[0], initial_data[0]);
    }

    #[test]
    fn test_lamb_bias_correction() {
        let optimizer_with_bias = LAMB::new(0.1);
        let optimizer_without_bias = LAMB::without_bias_correction(0.1, 0.9, 0.999, 1e-6, 0.01);

        assert!(optimizer_with_bias.bias_correction);
        assert!(!optimizer_without_bias.bias_correction);
    }

    #[test]
    fn test_lamb_state_dict() {
        let optimizer = LAMB::with_params(0.02, 0.85, 0.95, 1e-5, 0.05);
        let state = optimizer.state_dict();
        
        assert_eq!(state["learning_rate"], 0.02);
        assert_eq!(state["beta1"], 0.85);
        assert_eq!(state["beta2"], 0.95);
        assert_eq!(state["epsilon"], 1e-5);
        assert_eq!(state["weight_decay"], 0.05);
    }

    #[test]
    fn test_lamb_load_state_dict() {
        let mut optimizer = LAMB::new(0.001);
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), 0.05);
        state.insert("beta1".to_string(), 0.8);
        state.insert("beta2".to_string(), 0.95);
        
        optimizer.load_state_dict(state);
        
        assert_eq!(optimizer.learning_rate(), 0.05);
        assert_eq!(optimizer.beta1, 0.8);
        assert_eq!(optimizer.beta2, 0.95);
    }
}