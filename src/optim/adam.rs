//! Adam optimizer
//! Adamオプティマイザー

use super::Optimizer;
use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;

/// Adam optimizer
/// Adamオプティマイザー
#[derive(Debug)]
pub struct Adam<T: Float + Send + Sync + 'static> {
    params: Vec<Variable<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    step_count: usize,
    exp_avg: Vec<Option<Tensor<T>>>,      // First moment estimate
    exp_avg_sq: Vec<Option<Tensor<T>>>,   // Second moment estimate
}

impl<T: Float + Send + Sync + 'static> Adam<T> {
    /// Creates a new Adam optimizer
    /// 新しいAdamオプティマイザーを作成します
    pub fn new(
        params: Vec<Variable<T>>,
        lr: Option<T>,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
        weight_decay: Option<T>,
    ) -> Self {
        let lr = lr.unwrap_or_else(|| T::from(0.001).unwrap());
        let beta1 = beta1.unwrap_or_else(|| T::from(0.9).unwrap());
        let beta2 = beta2.unwrap_or_else(|| T::from(0.999).unwrap());
        let eps = eps.unwrap_or_else(|| T::from(1e-8).unwrap());
        let weight_decay = weight_decay.unwrap_or_else(T::zero);
        
        let exp_avg = vec![None; params.len()];
        let exp_avg_sq = vec![None; params.len()];
        
        Adam {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            exp_avg,
            exp_avg_sq,
        }
    }
    
    /// Sets the learning rate
    /// 学習率を設定します
    pub fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }
    
    /// Gets the current learning rate
    /// 現在の学習率を取得します
    pub fn get_lr(&self) -> T {
        self.lr
    }
}

impl<T: Float + Send + Sync + 'static> Optimizer<T> for Adam<T> {
    fn step(&mut self) {
        self.step_count += 1;
        
        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }
            
            let grad_arc = param.grad();
            let grad_lock = grad_arc.read().unwrap();
            
            if let Some(grad) = grad_lock.as_ref() {
                let mut d_p = grad.clone();
                
                // Apply weight decay
                if self.weight_decay != T::zero() {
                    let param_data = param.data();
                    let param_lock = param_data.read().unwrap();
                    d_p = &d_p + &(&*param_lock * &Tensor::from_vec(vec![self.weight_decay], vec![]));
                }
                
                // Initialize moment estimates if needed
                if self.exp_avg[i].is_none() {
                    self.exp_avg[i] = Some(Tensor::zeros(d_p.shape()));
                    self.exp_avg_sq[i] = Some(Tensor::zeros(d_p.shape()));
                }
                
                let exp_avg = self.exp_avg[i].as_mut().unwrap();
                let exp_avg_sq = self.exp_avg_sq[i].as_mut().unwrap();
                
                // Update biased first moment estimate
                // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                let beta1_tensor = Tensor::from_vec(vec![self.beta1], vec![]);
                let one_minus_beta1 = Tensor::from_vec(vec![T::one() - self.beta1], vec![]);
                *exp_avg = &(&*exp_avg * &beta1_tensor) + &(&d_p * &one_minus_beta1);
                
                // Update biased second raw moment estimate
                // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                let beta2_tensor = Tensor::from_vec(vec![self.beta2], vec![]);
                let one_minus_beta2 = Tensor::from_vec(vec![T::one() - self.beta2], vec![]);
                let grad_squared = &d_p * &d_p;
                *exp_avg_sq = &(&*exp_avg_sq * &beta2_tensor) + &(&grad_squared * &one_minus_beta2);
                
                // Compute bias correction
                let beta1_t = self.beta1.powi(self.step_count as i32);
                let beta2_t = self.beta2.powi(self.step_count as i32);
                let bias_correction1 = T::one() - beta1_t;
                let bias_correction2 = T::one() - beta2_t;
                
                // Corrected first moment estimate
                let corrected_exp_avg = &*exp_avg * &Tensor::from_vec(vec![T::one() / bias_correction1], vec![]);
                
                // Corrected second moment estimate
                let corrected_exp_avg_sq = &*exp_avg_sq * &Tensor::from_vec(vec![T::one() / bias_correction2], vec![]);
                
                // Compute denominator
                let sqrt_corrected_exp_avg_sq = {
                    let mut result = corrected_exp_avg_sq.clone();
                    let data = result.as_array_mut();
                    data.mapv_inplace(|x| x.sqrt());
                    result
                };
                let denom = &sqrt_corrected_exp_avg_sq + &Tensor::from_vec(vec![self.eps], vec![]);
                
                // Compute step size
                let step_size = self.lr / bias_correction2.sqrt();
                let step_size_tensor = Tensor::from_vec(vec![step_size], vec![]);
                
                // Update parameters
                let param_data = param.data();
                let mut param_lock = param_data.write().unwrap();
                let update = &(&corrected_exp_avg * &step_size_tensor) / &denom;
                *param_lock = &*param_lock - &update;
            }
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
    
    fn add_param_group(&mut self, params: Vec<Variable<T>>) {
        let _start_idx = self.params.len();
        self.params.extend(params);
        self.exp_avg.resize(self.params.len(), None);
        self.exp_avg_sq.resize(self.params.len(), None);
    }
}