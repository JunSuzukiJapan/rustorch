# フェーズ2: 高度最適化器 - 詳細技術仕様書

## 概要

フェーズ2では、現代的な深層学習に不可欠な高度最適化器群を実装します。Adam系の改良版、準ニュートン法、学習率スケジューラ、そして確率的重み平均化（SWA）をサポートし、RusTorchのPyTorch互換性を55%から65%に向上させます。

## 🚀 **実装対象API一覧**

### **高度Adam系最適化器**
- ✅ `AdamW` - 重み減衰付きAdam（既存拡張）
- 🆕 `NAdam` - Nesterovモーメンタム付きAdam
- 🆕 `RAdam` - 適応学習率修正Adam
- 🆕 `Adamax` - 無限ノルム版Adam
- 🆕 `ASGD` - 平均化確率的勾配降下

### **準ニュートン法・高度最適化器**
- ✅ `LBFGS` - 限定記憶BFGS（既存拡張）
- 🆕 `Rprop` - Resilient Backpropagation
- 🆕 `SparseAdam` - スパース対応Adam

### **学習率スケジューラ**
- 🆕 `StepLR` - ステップベース減衰
- 🆕 `MultiStepLR` - マルチステップ減衰  
- 🆕 `ExponentialLR` - 指数減衰
- 🆕 `CosineAnnealingLR` - コサインアニーリング
- 🆕 `ReduceLROnPlateau` - プラトーベース減少
- 🆕 `CyclicLR` - 循環学習率
- 🆕 `OneCycleLR` - ワンサイクルポリシー
- 🆕 `LambdaLR` - ラムダベーススケジューリング

### **確率的重み平均化（SWA）**
- 🆕 `AveragedModel` - SWAモデルラッパー
- 🆕 `SWALR` - SWA学習率スケジューラ

---

## 🔧 **詳細実装仕様**

### **1. NAdam（Nesterov Adam）**

```rust
/// Nesterov accelerated Adam optimizer
/// ネステロフ加速度付きAdam最適化器
pub struct NAdam<T: Float + Clone + Send + Sync> {
    params: Vec<Tensor<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    momentum_decay: T,
    
    // State variables
    step: usize,
    exp_avg: Vec<Tensor<T>>,      // First moment estimates
    exp_avg_sq: Vec<Tensor<T>>,   // Second moment estimates
}

impl<T: Float + Clone + Send + Sync + 'static> NAdam<T> {
    pub fn new(
        params: Vec<Tensor<T>>,
        lr: T,
        betas: (T, T),
        eps: T,
        weight_decay: T,
        momentum_decay: T,
    ) -> Self {
        let (beta1, beta2) = betas;
        let num_params = params.len();
        
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            momentum_decay,
            step: 0,
            exp_avg: vec![Tensor::zeros_like(&params[0]); num_params],
            exp_avg_sq: vec![Tensor::zeros_like(&params[0]); num_params],
        }
    }
    
    pub fn step(&mut self, gradients: &[Tensor<T>]) -> RusTorchResult<()> {
        self.step += 1;
        let step_f = T::from(self.step).unwrap();
        
        // Bias correction
        let bias_correction1 = T::one() - self.beta1.powf(step_f);
        let bias_correction2 = T::one() - self.beta2.powf(step_f);
        
        for (i, (param, grad)) in self.params.iter_mut().zip(gradients.iter()).enumerate() {
            let mut grad = grad.clone();
            
            // Apply weight decay
            if self.weight_decay != T::zero() {
                grad = &grad + &(param * self.weight_decay);
            }
            
            // Update biased first moment estimate
            self.exp_avg[i] = &self.exp_avg[i] * self.beta1 + &grad * (T::one() - self.beta1);
            
            // Update biased second raw moment estimate  
            self.exp_avg_sq[i] = &self.exp_avg_sq[i] * self.beta2 + 
                &grad.elementwise_mul(&grad) * (T::one() - self.beta2);
            
            // Compute bias-corrected first moment
            let corrected_exp_avg = &self.exp_avg[i] / bias_correction1;
            
            // Compute bias-corrected second moment
            let corrected_exp_avg_sq = &self.exp_avg_sq[i] / bias_correction2;
            
            // Nesterov correction
            let momentum_factor = self.beta1 * (T::one() - 
                (T::from(0.96_f64).unwrap().powf(step_f * self.momentum_decay)));
            let nesterov_momentum = momentum_factor * &corrected_exp_avg + 
                (T::one() - self.beta1) * &grad / bias_correction1;
            
            // Update parameters
            let denominator = corrected_exp_avg_sq.sqrt() + self.eps;
            let update = nesterov_momentum.elementwise_div(&denominator) * self.lr;
            
            *param = param - &update;
        }
        
        Ok(())
    }
}
```

### **2. RAdam（Rectified Adam）**

```rust
/// Rectified Adam optimizer with variance rectification
/// 分散修正付きAdam最適化器
pub struct RAdam<T: Float + Clone + Send + Sync> {
    params: Vec<Tensor<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    
    // State variables
    step: usize,
    exp_avg: Vec<Tensor<T>>,
    exp_avg_sq: Vec<Tensor<T>>,
    rho_inf: T, // Maximum length of approximated SMA
}

impl<T: Float + Clone + Send + Sync + 'static> RAdam<T> {
    pub fn new(
        params: Vec<Tensor<T>>,
        lr: T,
        betas: (T, T),
        eps: T,
        weight_decay: T,
    ) -> Self {
        let (beta1, beta2) = betas;
        let num_params = params.len();
        
        // Calculate rho_infinity
        let rho_inf = T::from(2.0).unwrap() / (T::one() - beta2) - T::one();
        
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
            exp_avg: vec![Tensor::zeros_like(&params[0]); num_params],
            exp_avg_sq: vec![Tensor::zeros_like(&params[0]); num_params],
            rho_inf,
        }
    }
    
    pub fn step(&mut self, gradients: &[Tensor<T>]) -> RusTorchResult<()> {
        self.step += 1;
        let step_f = T::from(self.step).unwrap();
        
        // Calculate current rho_t
        let beta2_power = self.beta2.powf(step_f);
        let rho_t = self.rho_inf - T::from(2.0).unwrap() * step_f * beta2_power / 
                   (T::one() - beta2_power);
        
        for (i, (param, grad)) in self.params.iter_mut().zip(gradients.iter()).enumerate() {
            let mut grad = grad.clone();
            
            // Apply weight decay
            if self.weight_decay != T::zero() {
                grad = &grad + &(param * self.weight_decay);
            }
            
            // Update moments
            self.exp_avg[i] = &self.exp_avg[i] * self.beta1 + &grad * (T::one() - self.beta1);
            self.exp_avg_sq[i] = &self.exp_avg_sq[i] * self.beta2 + 
                &grad.elementwise_mul(&grad) * (T::one() - self.beta2);
            
            // Bias correction
            let bias_correction1 = T::one() - self.beta1.powf(step_f);
            let corrected_exp_avg = &self.exp_avg[i] / bias_correction1;
            
            let update = if rho_t >= T::from(5.0).unwrap() {
                // Variance rectification available
                let bias_correction2 = T::one() - self.beta2.powf(step_f);
                let corrected_exp_avg_sq = &self.exp_avg_sq[i] / bias_correction2;
                
                // Length of SMA
                let r = ((rho_t - T::from(4.0).unwrap()) * (rho_t - T::from(2.0).unwrap()) * 
                        self.rho_inf / ((self.rho_inf - T::from(4.0).unwrap()) * 
                        (self.rho_inf - T::from(2.0).unwrap()) * rho_t)).sqrt();
                
                corrected_exp_avg.elementwise_div(&corrected_exp_avg_sq.sqrt().add_scalar(self.eps)) * r
            } else {
                // Simple momentum update
                corrected_exp_avg
            };
            
            *param = param - &(update * self.lr);
        }
        
        Ok(())
    }
}
```

### **3. 学習率スケジューラ基盤**

```rust
/// Base trait for learning rate schedulers
/// 学習率スケジューラの基底トレイト
pub trait LRScheduler<T: Float> {
    fn step(&mut self, epoch: Option<usize>) -> RusTorchResult<()>;
    fn get_lr(&self) -> Vec<T>;
    fn state_dict(&self) -> HashMap<String, T>;
    fn load_state_dict(&mut self, state_dict: HashMap<String, T>) -> RusTorchResult<()>;
}

/// Step-based learning rate scheduler
/// ステップベース学習率スケジューラ
pub struct StepLR<T: Float> {
    optimizer: Arc<Mutex<dyn Optimizer<T>>>,
    step_size: usize,
    gamma: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

impl<T: Float + Clone + Send + Sync> StepLR<T> {
    pub fn new(
        optimizer: Arc<Mutex<dyn Optimizer<T>>>,
        step_size: usize,
        gamma: T,
        last_epoch: usize,
    ) -> Self {
        let base_lrs = {
            let opt = optimizer.lock().unwrap();
            opt.get_lr()
        };
        
        Self {
            optimizer,
            step_size,
            gamma,
            last_epoch,
            base_lrs,
        }
    }
}

impl<T: Float + Clone + Send + Sync + 'static> LRScheduler<T> for StepLR<T> {
    fn step(&mut self, epoch: Option<usize>) -> RusTorchResult<()> {
        self.last_epoch = epoch.unwrap_or(self.last_epoch + 1);
        
        let factor = self.gamma.powf(T::from(self.last_epoch / self.step_size).unwrap());
        let new_lrs: Vec<T> = self.base_lrs.iter().map(|&lr| lr * factor).collect();
        
        {
            let mut opt = self.optimizer.lock().unwrap();
            opt.set_lr(&new_lrs)?;
        }
        
        Ok(())
    }
    
    fn get_lr(&self) -> Vec<T> {
        let factor = self.gamma.powf(T::from(self.last_epoch / self.step_size).unwrap());
        self.base_lrs.iter().map(|&lr| lr * factor).collect()
    }
    
    // ... state_dict implementations
}

/// Cosine annealing learning rate scheduler
/// コサインアニーリング学習率スケジューラ
pub struct CosineAnnealingLR<T: Float> {
    optimizer: Arc<Mutex<dyn Optimizer<T>>>,
    t_max: usize,
    eta_min: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

impl<T: Float + Clone + Send + Sync + 'static> LRScheduler<T> for CosineAnnealingLR<T> {
    fn step(&mut self, epoch: Option<usize>) -> RusTorchResult<()> {
        self.last_epoch = epoch.unwrap_or(self.last_epoch + 1);
        
        let pi = T::from(std::f64::consts::PI).unwrap();
        let t_max_f = T::from(self.t_max).unwrap();
        let last_epoch_f = T::from(self.last_epoch).unwrap();
        
        let new_lrs: Vec<T> = self.base_lrs.iter().map(|&base_lr| {
            self.eta_min + (base_lr - self.eta_min) * 
            (T::one() + (pi * last_epoch_f / t_max_f).cos()) / T::from(2.0).unwrap()
        }).collect();
        
        {
            let mut opt = self.optimizer.lock().unwrap();
            opt.set_lr(&new_lrs)?;
        }
        
        Ok(())
    }
    
    fn get_lr(&self) -> Vec<T> {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let t_max_f = T::from(self.t_max).unwrap();
        let last_epoch_f = T::from(self.last_epoch).unwrap();
        
        self.base_lrs.iter().map(|&base_lr| {
            self.eta_min + (base_lr - self.eta_min) * 
            (T::one() + (pi * last_epoch_f / t_max_f).cos()) / T::from(2.0).unwrap()
        }).collect()
    }
}
```

### **4. 確率的重み平均化（SWA）**

```rust
/// Averaged model for Stochastic Weight Averaging
/// 確率的重み平均化のための平均化モデル
pub struct AveragedModel<M: Module> {
    model: M,
    avg_model: M,
    n_averaged: usize,
    use_buffers: bool,
}

impl<M: Module + Clone> AveragedModel<M> {
    pub fn new(model: M, use_buffers: bool) -> Self {
        let avg_model = model.clone();
        
        Self {
            model,
            avg_model,
            n_averaged: 0,
            use_buffers,
        }
    }
    
    /// Update averaged model with current model parameters
    /// 現在のモデルパラメータで平均化モデルを更新
    pub fn update_parameters(&mut self) -> RusTorchResult<()> {
        self.n_averaged += 1;
        let n = self.n_averaged as f32;
        
        // Update averaged parameters
        for (avg_param, param) in self.avg_model.parameters_mut()
            .zip(self.model.parameters()) {
            
            // Moving average: avg = (n-1)/n * avg + 1/n * param
            *avg_param = &*avg_param * ((n - 1.0) / n) + param * (1.0 / n);
        }
        
        // Update buffers if requested
        if self.use_buffers {
            for (avg_buffer, buffer) in self.avg_model.buffers_mut()
                .zip(self.model.buffers()) {
                
                *avg_buffer = buffer.clone();
            }
        }
        
        Ok(())
    }
    
    /// Get the averaged model
    /// 平均化モデルを取得
    pub fn averaged_model(&self) -> &M {
        &self.avg_model
    }
}

/// SWA learning rate scheduler  
/// SWA学習率スケジューラ
pub struct SWALR<T: Float> {
    optimizer: Arc<Mutex<dyn Optimizer<T>>>,
    swa_lr: T,
    anneal_epochs: usize,
    anneal_strategy: AnnealStrategy,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

#[derive(Debug, Clone)]
pub enum AnnealStrategy {
    Linear,
    Cosine,
}

impl<T: Float + Clone + Send + Sync + 'static> SWALR<T> {
    pub fn new(
        optimizer: Arc<Mutex<dyn Optimizer<T>>>,
        swa_lr: T,
        anneal_epochs: usize,
        anneal_strategy: AnnealStrategy,
    ) -> Self {
        let base_lrs = {
            let opt = optimizer.lock().unwrap();
            opt.get_lr()
        };
        
        Self {
            optimizer,
            swa_lr,
            anneal_epochs,
            anneal_strategy,
            last_epoch: 0,
            base_lrs,
        }
    }
}

impl<T: Float + Clone + Send + Sync + 'static> LRScheduler<T> for SWALR<T> {
    fn step(&mut self, epoch: Option<usize>) -> RusTorchResult<()> {
        self.last_epoch = epoch.unwrap_or(self.last_epoch + 1);
        
        let new_lrs = if self.last_epoch < self.anneal_epochs {
            // Annealing phase
            let progress = T::from(self.last_epoch).unwrap() / T::from(self.anneal_epochs).unwrap();
            
            match self.anneal_strategy {
                AnnealStrategy::Linear => {
                    self.base_lrs.iter().map(|&base_lr| {
                        base_lr + (self.swa_lr - base_lr) * progress
                    }).collect()
                }
                AnnealStrategy::Cosine => {
                    let pi = T::from(std::f64::consts::PI).unwrap();
                    self.base_lrs.iter().map(|&base_lr| {
                        self.swa_lr + (base_lr - self.swa_lr) * 
                        (T::one() + (pi * progress).cos()) / T::from(2.0).unwrap()
                    }).collect()
                }
            }
        } else {
            // SWA phase - use constant SWA learning rate
            vec![self.swa_lr; self.base_lrs.len()]
        };
        
        {
            let mut opt = self.optimizer.lock().unwrap();
            opt.set_lr(&new_lrs)?;
        }
        
        Ok(())
    }
    
    fn get_lr(&self) -> Vec<T> {
        if self.last_epoch < self.anneal_epochs {
            let progress = T::from(self.last_epoch).unwrap() / T::from(self.anneal_epochs).unwrap();
            
            match self.anneal_strategy {
                AnnealStrategy::Linear => {
                    self.base_lrs.iter().map(|&base_lr| {
                        base_lr + (self.swa_lr - base_lr) * progress
                    }).collect()
                }
                AnnealStrategy::Cosine => {
                    let pi = T::from(std::f64::consts::PI).unwrap();
                    self.base_lrs.iter().map(|&base_lr| {
                        self.swa_lr + (base_lr - self.swa_lr) * 
                        (T::one() + (pi * progress).cos()) / T::from(2.0).unwrap()
                    }).collect()
                }
            }
        } else {
            vec![self.swa_lr; self.base_lrs.len()]
        }
    }
}
```

---

## 📁 **ファイル構造**

```
src/optim/
├── mod.rs                    # 最適化器モジュール統合
├── optimizers/
│   ├── mod.rs               # 最適化器サブモジュール
│   ├── nadam.rs             # NAdam実装
│   ├── radam.rs             # RAdam実装
│   ├── adamax.rs            # Adamax実装
│   ├── asgd.rs              # ASGD実装
│   ├── rprop.rs             # Rprop実装
│   └── sparse_adam.rs       # SparseAdam実装
├── schedulers/
│   ├── mod.rs               # スケジューラ統合
│   ├── base.rs              # LRSchedulerトレイト
│   ├── step_lr.rs           # StepLR実装
│   ├── multistep_lr.rs      # MultiStepLR実装
│   ├── exponential_lr.rs    # ExponentialLR実装
│   ├── cosine_annealing.rs  # CosineAnnealingLR実装
│   ├── reduce_on_plateau.rs # ReduceLROnPlateau実装
│   ├── cyclic_lr.rs         # CyclicLR実装
│   ├── onecycle_lr.rs       # OneCycleLR実装
│   └── lambda_lr.rs         # LambdaLR実装
├── swa/
│   ├── mod.rs               # SWA統合
│   ├── averaged_model.rs    # AveragedModel実装
│   └── swa_lr.rs            # SWALR実装
└── utils/
    ├── line_search.rs       # 線探索アルゴリズム
    ├── momentum.rs          # モーメント計算
    └── weight_decay.rs      # 重み減衰ユーティリティ
```

---

## 🧪 **テスト戦略**

### **単体テスト**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nadam_convergence() {
        // NAdam収束性テスト
        let params = vec![Tensor::from_vec(vec![1.0, 1.0], vec![2])];
        let mut optimizer = NAdam::new(params, 0.001, (0.9, 0.999), 1e-8, 0.0, 0.004);
        
        // Simple quadratic function: f(x) = x^2
        for _ in 0..1000 {
            let gradients = vec![&params[0] * 2.0]; // df/dx = 2x
            optimizer.step(&gradients).unwrap();
        }
        
        // Should converge to zero
        assert!(params[0].data.iter().all(|&x| x.abs() < 0.01));
    }
    
    #[test]
    fn test_cosine_annealing_lr() {
        // コサインアニーリングテスト
        let optimizer = Arc::new(Mutex::new(MockOptimizer::new(0.1)));
        let mut scheduler = CosineAnnealingLR::new(optimizer, 100, 0.0);
        
        let initial_lr = scheduler.get_lr()[0];
        assert_eq!(initial_lr, 0.1);
        
        // Half way through should be minimum
        for _ in 0..50 {
            scheduler.step(None).unwrap();
        }
        let mid_lr = scheduler.get_lr()[0];
        assert!(mid_lr < 0.01); // Should be close to eta_min
        
        // End should return to original
        for _ in 0..50 {
            scheduler.step(None).unwrap();
        }
        let final_lr = scheduler.get_lr()[0];
        assert!((final_lr - 0.1).abs() < 0.01);
    }
    
    #[test]
    fn test_swa_averaging() {
        // SWA平均化テスト
        let model = MockModel::new();
        let mut swa_model = AveragedModel::new(model.clone(), false);
        
        // Update with different parameter values
        for i in 0..10 {
            model.set_parameter_value(i as f32);
            swa_model.update_parameters().unwrap();
        }
        
        // Average should be approximately (0+1+2+...+9)/10 = 4.5
        let avg_param = swa_model.averaged_model().get_parameter_value();
        assert!((avg_param - 4.5).abs() < 0.1);
    }
}
```

### **統合テスト**
```rust
#[test]
fn test_full_training_loop_with_scheduler() {
    // 完全な訓練ループテスト
    let model = SimpleLinearModel::new(10, 1);
    let optimizer = Arc::new(Mutex::new(RAdam::new(
        model.parameters(), 0.001, (0.9, 0.999), 1e-8, 0.01
    )));
    let mut scheduler = StepLR::new(optimizer.clone(), 30, 0.1, 0);
    
    for epoch in 0..100 {
        // Training step
        let loss = train_step(&model, &optimizer);
        
        // Update learning rate
        scheduler.step(Some(epoch)).unwrap();
        
        // Validate learning rate schedule
        if epoch == 29 {
            let lr = scheduler.get_lr()[0];
            assert!((lr - 0.001).abs() < 1e-6);
        } else if epoch == 30 {
            let lr = scheduler.get_lr()[0];
            assert!((lr - 0.0001).abs() < 1e-6);
        }
    }
}
```

### **性能ベンチマーク**
```rust
#[bench]
fn bench_nadam_step(b: &mut Bencher) {
    let params = vec![Tensor::randn(&[1000, 1000])];
    let mut optimizer = NAdam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, 0.004);
    let gradients = vec![Tensor::randn(&[1000, 1000])];
    
    b.iter(|| {
        optimizer.step(&gradients).unwrap();
    });
}

#[bench]
fn bench_cosine_scheduler_step(b: &mut Bencher) {
    let optimizer = Arc::new(Mutex::new(MockOptimizer::new(0.1)));
    let mut scheduler = CosineAnnealingLR::new(optimizer, 1000, 0.0);
    
    b.iter(|| {
        scheduler.step(None).unwrap();
    });
}
```

---

## 🚀 **実装マイルストーン**

### **Week 1-2: 基盤整備**
- [ ] 最適化器基底トレイト拡張
- [ ] スケジューラ基底トレイト実装
- [ ] テスト・ベンチマーク基盤構築
- [ ] CI/CD統合

### **Week 3-4: Adam系最適化器**
- [ ] NAdam実装・テスト
- [ ] RAdam実装・テスト  
- [ ] Adamax実装・テスト
- [ ] ASGD実装・テスト

### **Week 5-6: 学習率スケジューラ（基本）**
- [ ] StepLR, MultiStepLR実装
- [ ] ExponentialLR実装
- [ ] CosineAnnealingLR実装
- [ ] ReduceLROnPlateau実装

### **Week 7-8: 高度機能・最終化**
- [ ] CyclicLR, OneCycleLR実装
- [ ] SWA実装・テスト
- [ ] 性能最適化・ベンチマーク
- [ ] ドキュメント・統合テスト

---

## 📊 **予想される性能向上**

### **訓練速度向上**
- **NAdam**: 従来Adamより10-20%高速収束
- **RAdam**: 初期訓練での安定性向上（50%以上の分散減少）
- **スケジューラ**: 適切な学習率調整による20-30%の訓練時間短縮

### **メモリ使用量最適化**
- **効率的状態管理**: 最適化器状態の30%メモリ削減
- **遅延初期化**: 未使用パラメータのメモリ節約
- **バッチ処理**: 勾配累積によるメモリ効率向上

### **数値安定性**
- **RAdam**: アダプティブ学習率による勾配爆発防止
- **重み減衰**: 正則化による過学習抑制
- **混合精度**: FP16対応による高速化とメモリ節約

---

## 🔗 **次のフェーズとの統合**

### **フェーズ3（NN層）との連携**
- LayerNorm, GroupNormでの最適化器互換性
- RNNセルでの勾配クリッピング統合
- 転置畳み込みでの重み初期化最適化

### **フェーズ4（勾配ユーティリティ）との連携**  
- 高次微分対応の最適化器拡張
- 勾配チェック機能統合
- 自動混合精度との連携

### **分散学習への準備**
- 複数GPU対応の基盤整備
- 勾配同期・通信最適化
- 大規模モデル対応のメモリ管理

---

このフェーズ2の完了により、RusTorchは現代的な深層学習訓練に必要な高度最適化手法を完全サポートし、PyTorchとの実用的互換性を大幅に向上させることができます。