//! AdaBound optimizer
//! AdaBoundオプティマイザー
//!
//! AdaBound bridges the gap between Adam and SGD by dynamically adjusting
//! the learning rate bounds, providing better convergence properties.
//!
//! AdaBoundは学習率の境界を動的に調整することでAdamとSGDの
//! ギャップを埋め、より良い収束特性を提供します。

use super::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// AdaBound optimizer
///
/// AdaBound dynamically adjusts the bounds of learning rates to combine
/// the fast convergence of Adam with the good generalization of SGD.
#[derive(Debug, Clone)]
pub struct AdaBound {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    final_lr: f32,
    gamma: f32,
    step_count: usize,
    exp_avg: HashMap<usize, Tensor<f32>>, // First moment estimate
    exp_avg_sq: HashMap<usize, Tensor<f32>>, // Second moment estimate
}

impl AdaBound {
    /// Create new AdaBound optimizer with default parameters
    /// デフォルトパラメータで新しいAdaBoundオプティマイザーを作成
    pub fn new(learning_rate: f32) -> Self {
        Self::with_params(learning_rate, 0.1, 0.9, 0.999, 1e-8, 0.0, 1e-3)
    }

    /// Create AdaBound optimizer with custom parameters
    /// カスタムパラメータでAdaBoundオプティマイザーを作成
    pub fn with_params(
        learning_rate: f32,
        final_lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        gamma: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            final_lr,
            gamma,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Set final learning rate for convergence
    /// 収束用の最終学習率を設定
    pub fn set_final_lr(&mut self, final_lr: f32) {
        self.final_lr = final_lr;
    }

    /// Set gamma parameter for bounds adjustment
    /// 境界調整用のガンマパラメータを設定  
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    /// Get current step count
    /// 現在のステップ数を取得
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Calculate dynamic learning rate bounds
    /// 動的学習率境界を計算
    fn compute_bounds(&self) -> (f32, f32) {
        let t = self.step_count as f32;
        let base_lr = self.learning_rate;
        let final_lr = self.final_lr;

        // Dynamic bound calculation based on step count
        // ステップ数に基づく動的境界計算
        let bound_scale = (1.0 + t * self.gamma).ln();
        let lower_bound = final_lr * (1.0 - 1.0 / bound_scale);
        let upper_bound = final_lr * (1.0 + 1.0 / bound_scale);

        (lower_bound.max(0.0), upper_bound.min(base_lr))
    }
}

impl Optimizer for AdaBound {
    fn step(&mut self, param: &Tensor<f32>, grad: &Tensor<f32>) {
        let param_id = param.as_ptr() as usize;
        self.step_count += 1;

        let mut d_p = grad.clone();

        // Apply weight decay
        // 重み減衰を適用
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

        // Bias correction
        // バイアス補正
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        let corrected_exp_avg = &exp_avg / bias_correction1;
        let corrected_exp_avg_sq = &exp_avg_sq / bias_correction2;

        // Compute adaptive learning rates
        // 適応学習率を計算
        let sqrt_corrected_exp_avg_sq = corrected_exp_avg_sq.sqrt();
        let raw_step_size = corrected_exp_avg / (&sqrt_corrected_exp_avg_sq + self.epsilon);

        // Apply dynamic bounds (AdaBound's key innovation)
        // 動的境界を適用（AdaBoundの主要革新）
        let (lower_bound, upper_bound) = self.compute_bounds();

        // Element-wise bound clipping would be ideal, but for simplicity we use scalar bounds
        // 要素ごとの境界クリッピングが理想的ですが、簡単のためスカラー境界を使用
        let step_size = self.learning_rate.max(lower_bound).min(upper_bound);

        // Final parameter update
        // 最終パラメータ更新
        let scaled_update = &raw_step_size * step_size;
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
        state.insert("final_lr".to_string(), self.final_lr);
        state.insert("gamma".to_string(), self.gamma);
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
        if let Some(&final_lr) = state.get("final_lr") {
            self.final_lr = final_lr;
        }
        if let Some(&gamma) = state.get("gamma") {
            self.gamma = gamma;
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
    fn test_adabound_creation() {
        let optimizer = AdaBound::new(0.001);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(optimizer.final_lr, 0.1);
    }

    #[test]
    fn test_adabound_with_params() {
        let optimizer = AdaBound::with_params(0.01, 0.05, 0.8, 0.95, 1e-5, 0.02, 1e-4);
        assert_eq!(optimizer.learning_rate(), 0.01);
        assert_eq!(optimizer.final_lr, 0.05);
        assert_eq!(optimizer.beta1, 0.8);
        assert_eq!(optimizer.beta2, 0.95);
        assert_eq!(optimizer.epsilon, 1e-5);
        assert_eq!(optimizer.weight_decay, 0.02);
        assert_eq!(optimizer.gamma, 1e-4);
    }

    #[test]
    fn test_adabound_bounds_computation() {
        let mut optimizer = AdaBound::new(0.1);
        optimizer.step_count = 100;

        let (lower_bound, upper_bound) = optimizer.compute_bounds();

        // Bounds should be reasonable
        assert!(lower_bound >= 0.0);
        assert!(upper_bound <= optimizer.learning_rate());
        assert!(lower_bound <= upper_bound);
    }

    #[test]
    fn test_adabound_step() {
        let mut optimizer = AdaBound::new(0.01);
        let param = Tensor::<f32>::ones(&[2, 2]);
        let grad = Tensor::<f32>::ones(&[2, 2]) * 0.1;

        // Initial parameter values
        let initial_param = param.clone();

        // Perform optimization step
        optimizer.step(&param, &grad);

        // Verify step count increased
        assert_eq!(optimizer.step_count(), 1);

        // Verify parameters were updated
        let updated_data = param.data.as_slice().unwrap();
        let initial_data = initial_param.data.as_slice().unwrap();

        assert_ne!(updated_data[0], initial_data[0]);
    }

    #[test]
    fn test_adabound_convergence_behavior() {
        let mut optimizer = AdaBound::new(0.1);
        let param = Tensor::<f32>::ones(&[2, 2]);
        let grad = Tensor::<f32>::ones(&[2, 2]) * 0.05;

        // Perform multiple steps to test convergence behavior
        for _ in 0..50 {
            optimizer.step(&param, &grad);
        }

        // After many steps, the effective learning rate should approach final_lr
        assert_eq!(optimizer.step_count(), 50);

        let (lower_bound, upper_bound) = optimizer.compute_bounds();
        // Bounds should be reasonable, upper_bound should be less than or equal to initial LR
        assert!(upper_bound <= optimizer.learning_rate());
    }

    #[test]
    fn test_adabound_state_dict() {
        let optimizer = AdaBound::with_params(0.02, 0.08, 0.85, 0.95, 1e-5, 0.05, 2e-3);
        let state = optimizer.state_dict();

        assert_eq!(state["learning_rate"], 0.02);
        assert_eq!(state["final_lr"], 0.08);
        assert_eq!(state["beta1"], 0.85);
        assert_eq!(state["beta2"], 0.95);
        assert_eq!(state["gamma"], 2e-3);
    }

    #[test]
    fn test_adabound_load_state_dict() {
        let mut optimizer = AdaBound::new(0.001);
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), 0.05);
        state.insert("final_lr".to_string(), 0.02);
        state.insert("gamma".to_string(), 5e-4);

        optimizer.load_state_dict(state);

        assert_eq!(optimizer.learning_rate(), 0.05);
        assert_eq!(optimizer.final_lr, 0.02);
        assert_eq!(optimizer.gamma, 5e-4);
    }
}
