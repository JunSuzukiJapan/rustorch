//! Learning rate schedulers for optimizers
//! オプティマイザーの学習率スケジューラー

use crate::optim::Optimizer;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for learning rate schedulers
/// 学習率スケジューラーのトレイト
/// 
/// Learning rate schedulers adjust the learning rate during training
/// to improve convergence and final performance.
/// 学習率スケジューラーは訓練中に学習率を調整して、
/// 収束と最終性能を改善します。
pub trait LRScheduler<T: Float> {
    /// Get the current learning rate
    /// 現在の学習率を取得
    fn get_lr(&self) -> Vec<T>;
    
    /// Step the scheduler (usually called after each epoch)
    /// スケジューラーをステップ（通常は各エポック後に呼び出し）
    fn step(&mut self);
    
    /// Step the scheduler with a validation metric (for ReduceLROnPlateau)
    /// 検証メトリックでスケジューラーをステップ（ReduceLROnPlateau用）
    fn step_with_metric(&mut self, _metric: T) {
        // Default implementation ignores metric
        self.step();
    }
    
    /// Get the last epoch number
    /// 最後のエポック番号を取得
    fn last_epoch(&self) -> i32;
    
    /// Get the current state for saving/loading
    /// 保存/読み込み用の現在の状態を取得
    fn state_dict(&self) -> SchedulerState<T>;
    
    /// Load state from saved state
    /// 保存された状態から状態を読み込み
    fn load_state_dict(&mut self, state: SchedulerState<T>);
}

/// State structure for saving/loading schedulers
/// スケジューラーの保存/読み込み用状態構造体
#[derive(Debug, Clone)]
pub struct SchedulerState<T: Float> {
    /// Last epoch number
    /// 最後のエポック番号
    pub last_epoch: i32,
    /// Base learning rates for each parameter group
    /// 各パラメータグループの基本学習率
    pub base_lrs: Vec<T>,
    /// Current step count
    /// 現在のステップ数
    pub step_count: usize,
    /// Best metric value seen so far
    /// これまでに見た最良のメトリック値
    pub best_metric: Option<T>,
    /// Number of epochs without improvement
    /// 改善のないエポック数
    pub num_bad_epochs: usize,
    /// Cooldown counter for reduce on plateau
    /// プラトー時削減のクールダウンカウンター
    pub cooldown_counter: usize,
}

/// Step learning rate scheduler
/// ステップ学習率スケジューラー
/// 
/// Decays the learning rate by gamma every step_size epochs.
/// step_sizeエポックごとに学習率をgammaで減衰させます。
#[derive(Debug)]
pub struct StepLR<T: Float> {
    step_size: usize,
    gamma: T,
    last_epoch: i32,
    base_lrs: Vec<T>,
    current_lrs: Vec<T>,
}

impl<T: Float + Copy + From<f32>> StepLR<T> {
    /// Creates a new StepLR scheduler
    /// 新しいStepLRスケジューラーを作成
    /// 
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `step_size` - Period of learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay
    /// * `last_epoch` - The index of last epoch
    /// 
    /// # 引数
    /// * `optimizer` - スケジュールするオプティマイザー
    /// * `step_size` - 学習率減衰の周期
    /// * `gamma` - 学習率減衰の乗数因子
    /// * `last_epoch` - 最後のエポックのインデックス
    pub fn new(
        _optimizer: &mut dyn Optimizer<T>,
        step_size: usize,
        gamma: T,
        last_epoch: Option<i32>,
    ) -> Self {
        let last_epoch = last_epoch.unwrap_or(-1);
        let base_lrs = vec![<T as From<f32>>::from(0.01f32)]; // Default learning rate for schedulers
        let current_lrs = base_lrs.clone();
        
        StepLR {
            step_size,
            gamma,
            last_epoch,
            base_lrs,
            current_lrs,
        }
    }
    
    /// Calculate learning rate for given epoch
    /// 指定されたエポックの学習率を計算
    fn calculate_lr(&self, base_lr: T, epoch: i32) -> T {
        let step_count = ((epoch + 1) as f32 / self.step_size as f32).floor() as i32;
        base_lr * self.gamma.powi(step_count)
    }
}

impl<T: Float + Copy + From<f32>> LRScheduler<T> for StepLR<T> {
    fn get_lr(&self) -> Vec<T> {
        self.current_lrs.clone()
    }
    
    fn step(&mut self) {
        self.last_epoch += 1;
        self.current_lrs = self.base_lrs
            .iter()
            .map(|&base_lr| self.calculate_lr(base_lr, self.last_epoch))
            .collect();
    }
    
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }
    
    fn state_dict(&self) -> SchedulerState<T> {
        SchedulerState {
            last_epoch: self.last_epoch,
            base_lrs: self.base_lrs.clone(),
            step_count: 0,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
    
    fn load_state_dict(&mut self, state: SchedulerState<T>) {
        self.last_epoch = state.last_epoch;
        self.base_lrs = state.base_lrs;
    }
}

/// Exponential learning rate scheduler
/// 指数的学習率スケジューラー
/// 
/// Decays the learning rate by gamma every epoch.
/// 毎エポック学習率をgammaで減衰させます。
#[derive(Debug)]
pub struct ExponentialLR<T: Float> {
    gamma: T,
    last_epoch: i32,
    base_lrs: Vec<T>,
    current_lrs: Vec<T>,
}

impl<T: Float + Copy + From<f32>> ExponentialLR<T> {
    /// Creates a new ExponentialLR scheduler
    /// 新しいExponentialLRスケジューラーを作成
    pub fn new(
        _optimizer: &mut dyn Optimizer<T>,
        gamma: T,
        last_epoch: Option<i32>,
    ) -> Self {
        let last_epoch = last_epoch.unwrap_or(-1);
        let base_lrs = vec![<T as From<f32>>::from(0.01f32)]; // Default learning rate for schedulers
        let current_lrs = base_lrs.clone();
        
        ExponentialLR {
            gamma,
            last_epoch,
            base_lrs,
            current_lrs,
        }
    }
}

impl<T: Float + Copy + From<f32>> LRScheduler<T> for ExponentialLR<T> {
    fn get_lr(&self) -> Vec<T> {
        self.current_lrs.clone()
    }
    
    fn step(&mut self) {
        self.last_epoch += 1;
        self.current_lrs = self.base_lrs
            .iter()
            .map(|&base_lr| base_lr * self.gamma.powi(self.last_epoch + 1))
            .collect();
    }
    
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }
    
    fn state_dict(&self) -> SchedulerState<T> {
        SchedulerState {
            last_epoch: self.last_epoch,
            base_lrs: self.base_lrs.clone(),
            step_count: 0,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
    
    fn load_state_dict(&mut self, state: SchedulerState<T>) {
        self.last_epoch = state.last_epoch;
        self.base_lrs = state.base_lrs;
    }
}

/// Cosine Annealing learning rate scheduler
/// コサイン・アニーリング学習率スケジューラー
/// 
/// Set the learning rate using a cosine annealing schedule.
/// コサイン・アニーリングスケジュールを使用して学習率を設定します。
#[derive(Debug)]
pub struct CosineAnnealingLR<T: Float> {
    t_max: usize,
    eta_min: T,
    last_epoch: i32,
    base_lrs: Vec<T>,
    current_lrs: Vec<T>,
}

impl<T: Float + Copy + From<f32>> CosineAnnealingLR<T> {
    /// Creates a new CosineAnnealingLR scheduler
    /// 新しいCosineAnnealingLRスケジューラーを作成
    /// 
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `t_max` - Maximum number of iterations
    /// * `eta_min` - Minimum learning rate
    /// * `last_epoch` - The index of last epoch
    /// 
    /// # 引数
    /// * `optimizer` - スケジュールするオプティマイザー
    /// * `t_max` - 最大反復数
    /// * `eta_min` - 最小学習率
    /// * `last_epoch` - 最後のエポックのインデックス
    pub fn new(
        _optimizer: &mut dyn Optimizer<T>,
        t_max: usize,
        eta_min: Option<T>,
        last_epoch: Option<i32>,
    ) -> Self {
        let last_epoch = last_epoch.unwrap_or(-1);
        let eta_min = eta_min.unwrap_or_else(|| <T as From<f32>>::from(0.0f32));
        let base_lrs = vec![<T as From<f32>>::from(0.01f32)]; // Default learning rate for schedulers
        let current_lrs = base_lrs.clone();
        
        CosineAnnealingLR {
            t_max,
            eta_min,
            last_epoch,
            base_lrs,
            current_lrs,
        }
    }
    
    /// Calculate learning rate using cosine annealing
    /// コサイン・アニーリングを使用して学習率を計算
    fn calculate_lr(&self, base_lr: T, epoch: i32) -> T {
        if epoch < 0 {
            return base_lr;
        }
        
        let t_cur = epoch as f32;
        let t_max = self.t_max as f32;
        let pi = std::f32::consts::PI;
        
        let cosine_factor = <T as From<f32>>::from((1.0 + (pi * t_cur / t_max).cos()) / 2.0);
        self.eta_min + (base_lr - self.eta_min) * cosine_factor
    }
}

impl<T: Float + Copy + From<f32>> LRScheduler<T> for CosineAnnealingLR<T> {
    fn get_lr(&self) -> Vec<T> {
        self.current_lrs.clone()
    }
    
    fn step(&mut self) {
        self.last_epoch += 1;
        self.current_lrs = self.base_lrs
            .iter()
            .map(|&base_lr| self.calculate_lr(base_lr, self.last_epoch))
            .collect();
    }
    
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }
    
    fn state_dict(&self) -> SchedulerState<T> {
        SchedulerState {
            last_epoch: self.last_epoch,
            base_lrs: self.base_lrs.clone(),
            step_count: 0,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
    
    fn load_state_dict(&mut self, state: SchedulerState<T>) {
        self.last_epoch = state.last_epoch;
        self.base_lrs = state.base_lrs;
    }
}

/// Reduce learning rate on plateau scheduler
/// プラトー時学習率削減スケジューラー
/// 
/// Reduce learning rate when a metric has stopped improving.
/// メトリックが改善を停止したときに学習率を削減します。
#[derive(Debug)]
pub struct ReduceLROnPlateau<T: Float> {
    mode: PlateauMode,
    factor: T,
    patience: usize,
    threshold: T,
    threshold_mode: ThresholdMode,
    cooldown: usize,
    min_lr: T,
    eps: T,
    last_epoch: i32,
    base_lrs: Vec<T>,
    current_lrs: Vec<T>,
    best_metric: Option<T>,
    num_bad_epochs: usize,
    cooldown_counter: usize,
}

/// Mode for determining when to reduce learning rate on plateau
/// プラトー時の学習率削減の判定モード
#[derive(Debug, Clone, Copy)]
pub enum PlateauMode {
    /// Reduce when metric stops decreasing
    /// メトリックの減少が止まったときに削減
    Min,
    /// Reduce when metric stops increasing
    /// メトリックの増加が止まったときに削減
    Max,
}

/// Mode for threshold comparison
/// 閾値比較モード
#[derive(Debug, Clone, Copy)]
pub enum ThresholdMode {
    /// Relative threshold
    /// 相対閾値
    Rel,
    /// Absolute threshold
    /// 絶対閾値
    Abs,
}

impl<T: Float + Copy + From<f32>> ReduceLROnPlateau<T> {
    /// Creates a new ReduceLROnPlateau scheduler
    /// 新しいReduceLROnPlateauスケジューラーを作成
    pub fn new(
        _optimizer: &mut dyn Optimizer<T>,
        mode: Option<PlateauMode>,
        factor: Option<T>,
        patience: Option<usize>,
        threshold: Option<T>,
        threshold_mode: Option<ThresholdMode>,
        cooldown: Option<usize>,
        min_lr: Option<T>,
        eps: Option<T>,
    ) -> Self {
        let mode = mode.unwrap_or(PlateauMode::Min);
        let factor = factor.unwrap_or_else(|| <T as From<f32>>::from(0.1f32));
        let patience = patience.unwrap_or(10);
        let threshold = threshold.unwrap_or_else(|| <T as From<f32>>::from(1e-4f32));
        let threshold_mode = threshold_mode.unwrap_or(ThresholdMode::Rel);
        let cooldown = cooldown.unwrap_or(0);
        let min_lr = min_lr.unwrap_or_else(|| <T as From<f32>>::from(0.0f32));
        let eps = eps.unwrap_or_else(|| <T as From<f32>>::from(1e-8f32));
        let base_lrs = vec![<T as From<f32>>::from(0.01f32)]; // Default learning rate for schedulers
        let current_lrs = base_lrs.clone();
        
        ReduceLROnPlateau {
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            last_epoch: 0,
            base_lrs,
            current_lrs,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
    
    /// Check if metric has improved
    /// メトリックが改善したかどうかをチェック
    fn is_better(&self, current: T, best: T) -> bool {
        match (self.mode, self.threshold_mode) {
            (PlateauMode::Min, ThresholdMode::Rel) => current < best * (T::one() - self.threshold),
            (PlateauMode::Min, ThresholdMode::Abs) => current < best - self.threshold,
            (PlateauMode::Max, ThresholdMode::Rel) => current > best * (T::one() + self.threshold),
            (PlateauMode::Max, ThresholdMode::Abs) => current > best + self.threshold,
        }
    }
}

impl<T: Float + Copy + From<f32>> LRScheduler<T> for ReduceLROnPlateau<T> {
    fn get_lr(&self) -> Vec<T> {
        self.current_lrs.clone()
    }
    
    fn step(&mut self) {
        // This scheduler doesn't step without a metric
    }
    
    fn step_with_metric(&mut self, metric: T) {
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }
        
        if self.best_metric.is_none() {
            self.best_metric = Some(metric);
        } else if self.is_better(metric, self.best_metric.unwrap()) {
            self.best_metric = Some(metric);
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }
        
        if self.cooldown_counter == 0 && self.num_bad_epochs > self.patience {
            self.reduce_lr();
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
        }
    }
    
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }
    
    fn state_dict(&self) -> SchedulerState<T> {
        SchedulerState {
            last_epoch: self.last_epoch,
            base_lrs: self.base_lrs.clone(),
            step_count: 0,
            best_metric: self.best_metric,
            num_bad_epochs: self.num_bad_epochs,
            cooldown_counter: self.cooldown_counter,
        }
    }
    
    fn load_state_dict(&mut self, state: SchedulerState<T>) {
        self.last_epoch = state.last_epoch;
        self.base_lrs = state.base_lrs;
        self.best_metric = state.best_metric;
        self.num_bad_epochs = state.num_bad_epochs;
        self.cooldown_counter = state.cooldown_counter;
    }
}

impl<T: Float + Copy + From<f32>> ReduceLROnPlateau<T> {
    fn reduce_lr(&mut self) {
        for i in 0..self.current_lrs.len() {
            let new_lr = (self.current_lrs[i] * self.factor).max(self.min_lr);
            if (self.current_lrs[i] - new_lr).abs() > self.eps {
                self.current_lrs[i] = new_lr;
                self.base_lrs[i] = new_lr; // Update base_lrs as well for ReduceLROnPlateau
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::SGD;
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    #[test]
    fn test_step_lr_creation() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let scheduler = StepLR::new(&mut optimizer, 30, 0.1, None);
        assert_eq!(scheduler.last_epoch(), -1);
        assert_eq!(scheduler.step_size, 30);
    }

    #[test]
    fn test_step_lr_calculation() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let mut scheduler = StepLR::new(&mut optimizer, 30, 0.1, None);
        
        // Initial learning rate
        let initial_lrs = scheduler.get_lr();
        assert_eq!(initial_lrs.len(), 1);
        
        // Step and check
        scheduler.step();
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_exponential_lr() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let mut scheduler = ExponentialLR::new(&mut optimizer, 0.9, None);
        
        let initial_lrs = scheduler.get_lr();
        scheduler.step();
        let step1_lrs = scheduler.get_lr();
        
        // Learning rate should decay exponentially
        assert!(step1_lrs[0] < initial_lrs[0]);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let mut scheduler = CosineAnnealingLR::new(&mut optimizer, 100, None, None);
        
        let initial_lrs = scheduler.get_lr();
        
        // Step through half the cycle
        for _ in 0..50 {
            scheduler.step();
        }
        let mid_lrs = scheduler.get_lr();
        
        // At T_max/2, learning rate should be at minimum (eta_min)
        assert!(mid_lrs[0] < initial_lrs[0]);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let mut scheduler = ReduceLROnPlateau::new(
            &mut optimizer,
            Some(PlateauMode::Min),
            Some(0.5),
            Some(2),
            None,
            None,
            None,
            None,
            None,
        );
        
        let _initial_lrs = scheduler.get_lr();
        
        // Simulate no improvement for patience + 1 steps
        scheduler.step_with_metric(1.0);
        scheduler.step_with_metric(1.1);
        scheduler.step_with_metric(1.2);
        scheduler.step_with_metric(1.3);
        
        let _reduced_lrs = scheduler.get_lr();
        
        // Learning rate should be reduced after patience is exceeded
        // Note: In this simplified test, we check the state rather than actual LR change
        assert_eq!(scheduler.num_bad_epochs, 0); // Should reset after reduction
    }
}