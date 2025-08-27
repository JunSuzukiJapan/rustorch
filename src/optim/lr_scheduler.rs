//! Learning rate scheduler implementations
//! 学習率スケジューラの実装

use super::Optimizer;
use std::f32::consts::PI;

/// Base trait for learning rate schedulers
/// 学習率スケジューラの基底トレイト
pub trait LRScheduler {
    /// Update the learning rate
    fn step(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Get the last epoch/step number
    fn get_last_epoch(&self) -> usize;
}

/// StepLR scheduler - decays learning rate by gamma every step_size epochs
/// StepLRスケジューラ - step_sizeエポックごとにgammaで学習率を減衰
pub struct StepLR<O: Optimizer> {
    optimizer: O,
    step_size: usize,
    gamma: f32,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> StepLR<O> {
    /// Create a new StepLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `step_size` - Period of learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay (default: 0.1)
    pub fn new(optimizer: O, step_size: usize, gamma: f32) -> Self {
        assert!(step_size > 0, "Step size must be positive");
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");

        let base_lr = optimizer.learning_rate();
        Self {
            optimizer,
            step_size,
            gamma,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for StepLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;
        let new_lr = self.base_lr * self.gamma.powi((self.last_epoch / self.step_size) as i32);
        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// ExponentialLR scheduler - decays learning rate by gamma every epoch
/// ExponentialLRスケジューラ - 毎エポックgammaで学習率を減衰
pub struct ExponentialLR<O: Optimizer> {
    optimizer: O,
    gamma: f32,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> ExponentialLR<O> {
    /// Create a new ExponentialLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(optimizer: O, gamma: f32) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");

        let base_lr = optimizer.learning_rate();
        Self {
            optimizer,
            gamma,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for ExponentialLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;
        let new_lr = self.base_lr * self.gamma.powi(self.last_epoch as i32);
        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// CosineAnnealingLR scheduler - anneals learning rate using cosine schedule
/// CosineAnnealingLRスケジューラ - コサインスケジュールで学習率を減衰
pub struct CosineAnnealingLR<O: Optimizer> {
    optimizer: O,
    t_max: usize,
    eta_min: f32,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> CosineAnnealingLR<O> {
    /// Create a new CosineAnnealingLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `t_max` - Maximum number of iterations
    /// * `eta_min` - Minimum learning rate (default: 0)
    pub fn new(optimizer: O, t_max: usize, eta_min: f32) -> Self {
        assert!(t_max > 0, "T_max must be positive");
        assert!(eta_min >= 0.0, "Eta_min must be non-negative");

        let base_lr = optimizer.learning_rate();
        assert!(base_lr > eta_min, "Base LR must be greater than eta_min");

        Self {
            optimizer,
            t_max,
            eta_min,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for CosineAnnealingLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;

        if self.last_epoch > self.t_max {
            self.last_epoch = self.last_epoch % self.t_max;
        }

        let new_lr = self.eta_min
            + (self.base_lr - self.eta_min)
                * (1.0 + (PI * self.last_epoch as f32 / self.t_max as f32).cos())
                / 2.0;

        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// ReduceLROnPlateau scheduler - reduces learning rate when metric has stopped improving
/// ReduceLROnPlateauスケジューラ - メトリックの改善が停止したときに学習率を減少
pub struct ReduceLROnPlateau<O: Optimizer> {
    optimizer: O,
    mode: PlateauMode,
    factor: f32,
    patience: usize,
    threshold: f32,
    threshold_mode: ThresholdMode,
    cooldown: usize,
    min_lr: f32,
    eps: f32,

    // Internal state
    best: f32,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    last_epoch: usize,
}

/// Mode for ReduceLROnPlateau
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlateauMode {
    /// For metrics to minimize (loss)
    Min,
    /// For metrics to maximize (accuracy)
    Max,
}

/// Threshold mode for ReduceLROnPlateau
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdMode {
    /// Relative threshold
    Rel,
    /// Absolute threshold
    Abs,
}

impl<O: Optimizer> ReduceLROnPlateau<O> {
    /// Create a new ReduceLROnPlateau scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `mode` - Min or Max mode
    /// * `factor` - Factor by which the learning rate will be reduced
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `threshold` - Threshold for measuring the new optimum
    /// * `threshold_mode` - Relative or absolute threshold mode
    /// * `cooldown` - Number of epochs to wait before resuming normal operation
    /// * `min_lr` - A lower bound on the learning rate
    /// * `eps` - Minimal decay applied to lr
    pub fn new(
        optimizer: O,
        mode: PlateauMode,
        factor: f32,
        patience: usize,
        threshold: f32,
        threshold_mode: ThresholdMode,
        cooldown: usize,
        min_lr: f32,
        eps: f32,
    ) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "Factor must be in (0, 1)");
        assert!(threshold >= 0.0, "Threshold must be non-negative");
        assert!(min_lr >= 0.0, "Min LR must be non-negative");
        assert!(eps >= 0.0, "Epsilon must be non-negative");

        let best = match mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };

        Self {
            optimizer,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
        }
    }

    /// Create with default parameters for minimization
    pub fn default_min(optimizer: O) -> Self {
        Self::new(
            optimizer,
            PlateauMode::Min,
            0.1,
            10,
            1e-4,
            ThresholdMode::Rel,
            0,
            0.0,
            1e-8,
        )
    }

    /// Create with default parameters for maximization
    pub fn default_max(optimizer: O) -> Self {
        Self::new(
            optimizer,
            PlateauMode::Max,
            0.1,
            10,
            1e-4,
            ThresholdMode::Rel,
            0,
            0.0,
            1e-8,
        )
    }

    /// Step with metric value
    pub fn step_with_metric(&mut self, metric: f32) {
        self.last_epoch += 1;

        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.in_cooldown() {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.patience {
            self.reduce_lr();
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
        }
    }

    fn is_better(&self, metric: f32) -> bool {
        let threshold_value = match self.threshold_mode {
            ThresholdMode::Rel => self.threshold * self.best.abs(),
            ThresholdMode::Abs => self.threshold,
        };

        match self.mode {
            PlateauMode::Min => metric < self.best - threshold_value,
            PlateauMode::Max => metric > self.best + threshold_value,
        }
    }

    fn in_cooldown(&self) -> bool {
        self.cooldown_counter > 0
    }

    fn reduce_lr(&mut self) {
        let old_lr = self.optimizer.learning_rate();
        let new_lr = (old_lr * self.factor).max(self.min_lr);

        if old_lr - new_lr > self.eps {
            self.optimizer.set_learning_rate(new_lr);
        }
    }
}

impl<O: Optimizer> LRScheduler for ReduceLROnPlateau<O> {
    fn step(&mut self) {
        // ReduceLROnPlateau requires metric, so this does nothing
        // Use step_with_metric instead
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// MultiStepLR scheduler - decays learning rate at specified milestones
/// MultiStepLRスケジューラ - 指定されたマイルストーンで学習率を減衰
pub struct MultiStepLR<O: Optimizer> {
    optimizer: O,
    milestones: Vec<usize>,
    gamma: f32,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> MultiStepLR<O> {
    /// Create a new MultiStepLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `milestones` - List of epoch indices (must be increasing)
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(optimizer: O, mut milestones: Vec<usize>, gamma: f32) -> Self {
        assert!(!milestones.is_empty(), "Milestones cannot be empty");
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");

        milestones.sort_unstable();
        let base_lr = optimizer.learning_rate();

        Self {
            optimizer,
            milestones,
            gamma,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for MultiStepLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;

        let num_decays = self
            .milestones
            .iter()
            .filter(|&&m| self.last_epoch >= m)
            .count();

        let new_lr = self.base_lr * self.gamma.powi(num_decays as i32);
        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// WarmupScheduler - gradually increases learning rate from base to target over warmup_epochs
/// ウォームアップスケジューラ - 指定エポック数でbase_lrからtarget_lrまで徐々に学習率を上昇
pub struct WarmupScheduler<O: Optimizer> {
    optimizer: O,
    base_lr: f32,
    target_lr: f32,
    warmup_epochs: usize,
    last_epoch: usize,
}

impl<O: Optimizer> WarmupScheduler<O> {
    /// Create a new WarmupScheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `target_lr` - Target learning rate to reach after warmup
    /// * `warmup_epochs` - Number of epochs for warmup phase
    pub fn new(optimizer: O, target_lr: f32, warmup_epochs: usize) -> Self {
        assert!(warmup_epochs > 0, "Warmup epochs must be positive");
        assert!(target_lr > 0.0, "Target LR must be positive");

        let base_lr = optimizer.learning_rate();
        Self {
            optimizer,
            base_lr,
            target_lr,
            warmup_epochs,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for WarmupScheduler<O> {
    fn step(&mut self) {
        self.last_epoch += 1;

        let new_lr = if self.last_epoch <= self.warmup_epochs {
            self.base_lr
                + (self.target_lr - self.base_lr)
                    * (self.last_epoch as f32 / self.warmup_epochs as f32)
        } else {
            self.target_lr
        };

        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Annealing strategy for OneCycleLR
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnnealStrategy {
    /// Cosine annealing
    Cos,
    /// Linear annealing
    Linear,
}

/// OneCycleLR scheduler - implements 1cycle learning rate policy
/// OneCycleLRスケジューラ - 1サイクル学習率ポリシーを実装
pub struct OneCycleLR<O: Optimizer> {
    optimizer: O,
    max_lr: f32,
    total_steps: usize,
    pct_start: f32,
    anneal_strategy: AnnealStrategy,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> OneCycleLR<O> {
    /// Create a new OneCycleLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `max_lr` - Upper learning rate boundary in the cycle
    /// * `total_steps` - Total number of steps in the training
    /// * `pct_start` - Percentage of the cycle spent increasing the learning rate
    /// * `anneal_strategy` - Specifies the annealing strategy (Cos or Linear)
    pub fn new(
        optimizer: O,
        max_lr: f32,
        total_steps: usize,
        pct_start: f32,
        anneal_strategy: AnnealStrategy,
    ) -> Self {
        assert!(max_lr > 0.0, "Max LR must be positive");
        assert!(total_steps > 0, "Total steps must be positive");
        assert!(
            pct_start > 0.0 && pct_start < 1.0,
            "Pct start must be in (0, 1)"
        );

        let base_lr = optimizer.learning_rate();
        Self {
            optimizer,
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for OneCycleLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;

        let step_num = self.last_epoch.min(self.total_steps) as f32;
        let cycle_pos = step_num / self.total_steps as f32;

        let new_lr = if cycle_pos <= self.pct_start {
            // Warmup phase
            let warmup_pos = cycle_pos / self.pct_start;
            self.base_lr + (self.max_lr - self.base_lr) * warmup_pos
        } else {
            // Annealing phase
            let anneal_pos = (cycle_pos - self.pct_start) / (1.0 - self.pct_start);

            match self.anneal_strategy {
                AnnealStrategy::Cos => {
                    self.base_lr
                        + (self.max_lr - self.base_lr) * (1.0 + (PI * anneal_pos).cos()) / 2.0
                }
                AnnealStrategy::Linear => self.max_lr - (self.max_lr - self.base_lr) * anneal_pos,
            }
        };

        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// PolynomialLR scheduler - decays learning rate using polynomial function
/// PolynomialLRスケジューラ - 多項式関数で学習率を減衰
pub struct PolynomialLR<O: Optimizer> {
    optimizer: O,
    total_epochs: usize,
    power: f32,
    min_lr: f32,
    base_lr: f32,
    last_epoch: usize,
}

impl<O: Optimizer> PolynomialLR<O> {
    /// Create a new PolynomialLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `total_epochs` - Total number of training epochs
    /// * `power` - The power of the polynomial (default: 1.0 for linear)
    /// * `min_lr` - Minimum learning rate (default: 0.0)
    pub fn new(optimizer: O, total_epochs: usize, power: f32, min_lr: f32) -> Self {
        assert!(total_epochs > 0, "Total epochs must be positive");
        assert!(power > 0.0, "Power must be positive");
        assert!(min_lr >= 0.0, "Min LR must be non-negative");

        let base_lr = optimizer.learning_rate();
        assert!(base_lr > min_lr, "Base LR must be greater than min LR");

        Self {
            optimizer,
            total_epochs,
            power,
            min_lr,
            base_lr,
            last_epoch: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for PolynomialLR<O> {
    fn step(&mut self) {
        self.last_epoch += 1;

        let decay_factor = if self.last_epoch >= self.total_epochs {
            0.0
        } else {
            (1.0 - (self.last_epoch as f32 / self.total_epochs as f32)).powf(self.power)
        };

        let new_lr = self.min_lr + (self.base_lr - self.min_lr) * decay_factor;
        self.optimizer.set_learning_rate(new_lr);
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::SGD;

    #[test]
    fn test_step_lr() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = StepLR::new(optimizer, 2, 0.5);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.5);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.5);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.25);
    }

    #[test]
    fn test_exponential_lr() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = ExponentialLR::new(optimizer, 0.9);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.9).abs() < 1e-6);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.81).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = CosineAnnealingLR::new(optimizer, 10, 0.0);

        assert_eq!(scheduler.get_lr(), 1.0);

        // At halfway point
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.5).abs() < 0.01);

        // At the end
        for _ in 0..5 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.01);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = ReduceLROnPlateau::default_min(optimizer);

        assert_eq!(scheduler.get_lr(), 1.0);

        // Simulate no improvement for patience epochs
        for _ in 0..11 {
            scheduler.step_with_metric(1.0);
        }

        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_multi_step_lr() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = MultiStepLR::new(optimizer, vec![2, 5, 8], 0.1);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..3 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);

        for _ in 0..3 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_scheduler() {
        let optimizer = SGD::new(0.1);
        let mut scheduler = WarmupScheduler::new(optimizer, 1.0, 5);

        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        for i in 1..=5 {
            scheduler.step();
            let expected = 0.1 + (1.0 - 0.1) * (i as f32 / 5.0);
            assert!((scheduler.get_lr() - expected).abs() < 1e-5);
        }

        scheduler.step();
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_cycle_lr() {
        let optimizer = SGD::new(0.1);
        let mut scheduler = OneCycleLR::new(optimizer, 1.0, 10, 0.25, AnnealStrategy::Cos);

        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..10 {
            scheduler.step();
        }

        assert!(scheduler.get_lr() < 1.0);
        assert!(scheduler.get_lr() > 0.0);
    }

    #[test]
    fn test_polynomial_lr() {
        let optimizer = SGD::new(1.0);
        let mut scheduler = PolynomialLR::new(optimizer, 10, 2.0, 0.0);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step();
        let expected = 0.81;
        assert!((scheduler.get_lr() - expected).abs() < 0.01);
    }
}
