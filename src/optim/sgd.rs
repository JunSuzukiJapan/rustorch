//! Stochastic Gradient Descent optimizer
//! 確率的勾配降下法オプティマイザー

use super::Optimizer;
use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;

/// Stochastic Gradient Descent optimizer
/// 確率的勾配降下法オプティマイザー
#[derive(Debug)]
pub struct SGD<T: Float + Send + Sync + 'static> {
    params: Vec<Variable<T>>,
    lr: T,
    momentum: T,
    dampening: T,
    weight_decay: T,
    nesterov: bool,
    velocity: Vec<Option<Tensor<T>>>,
}

impl<T: Float + Send + Sync + 'static> SGD<T> {
    /// Creates a new SGD optimizer
    /// 新しいSGDオプティマイザーを作成します
    pub fn new(
        params: Vec<Variable<T>>, 
        lr: T, 
        momentum: Option<T>, 
        dampening: Option<T>,
        weight_decay: Option<T>,
        nesterov: Option<bool>
    ) -> Self {
        let momentum = momentum.unwrap_or_else(T::zero);
        let dampening = dampening.unwrap_or_else(T::zero);
        let weight_decay = weight_decay.unwrap_or_else(T::zero);
        let nesterov = nesterov.unwrap_or(false);
        
        if nesterov && (momentum <= T::zero() || dampening != T::zero()) {
            panic!("Nesterov momentum requires a momentum and zero dampening");
        }
        
        let velocity = vec![None; params.len()];
        
        SGD {
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            velocity,
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

impl<T: Float + Send + Sync + 'static> Optimizer<T> for SGD<T> {
    fn step(&mut self) {
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
                
                // Apply momentum
                if self.momentum != T::zero() {
                    if let Some(ref mut buf) = self.velocity[i] {
                        // buf = momentum * buf + (1 - dampening) * d_p
                        let momentum_term = &*buf * &Tensor::from_vec(vec![self.momentum], vec![]);
                        let grad_term = &d_p * &Tensor::from_vec(vec![T::one() - self.dampening], vec![]);
                        *buf = &momentum_term + &grad_term;
                    } else {
                        self.velocity[i] = Some(d_p.clone());
                    }
                    
                    if self.nesterov {
                        // d_p = d_p + momentum * buf
                        let buf_ref = self.velocity[i].as_ref().unwrap();
                        let momentum_term = &*buf_ref * &Tensor::from_vec(vec![self.momentum], vec![]);
                        d_p = &d_p + &momentum_term;
                    } else {
                        d_p = self.velocity[i].as_ref().unwrap().clone();
                    }
                }
                
                // Update parameters
                let param_data = param.data();
                let mut param_lock = param_data.write().unwrap();
                let lr_tensor = Tensor::from_vec(vec![self.lr], vec![]);
                let update = &d_p * &lr_tensor;
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
        self.velocity.resize(self.params.len(), None);
    }
}