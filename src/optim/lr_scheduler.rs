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
}
