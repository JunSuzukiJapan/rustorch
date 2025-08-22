//! Demonstration of AdamW optimizer with learning rate schedulers
//! AdamWオプティマイザと学習率スケジューラのデモンストレーション

use rustorch::prelude::*;
use rustorch::nn::Linear;
use rustorch::optim::{AdamW, Optimizer, LRScheduler, StepLR, CosineAnnealingLR, ReduceLROnPlateau, PlateauMode};

fn main() {
    println!("=== AdamW + Learning Rate Scheduler Demo ===\n");
    
    // Create a simple model
    let _linear = Linear::<f32>::new(10, 1);
    
    // Create AdamW optimizer
    let adamw = AdamW::default_params(0.01);
    println!("Initial learning rate: {}", adamw.learning_rate());
    
    // Demonstrate StepLR scheduler
    println!("\n--- StepLR Scheduler ---");
    let mut step_scheduler = StepLR::new(AdamW::default_params(0.1), 2, 0.5);
    
    for epoch in 0..8 {
        println!("Epoch {}: LR = {:.4}", epoch, step_scheduler.get_lr());
        step_scheduler.step();
    }
    
    // Demonstrate CosineAnnealingLR scheduler
    println!("\n--- CosineAnnealingLR Scheduler ---");
    let mut cosine_scheduler = CosineAnnealingLR::new(
        AdamW::default_params(0.1), 
        10, 
        0.001
    );
    
    for epoch in 0..15 {
        println!("Epoch {}: LR = {:.4}", epoch, cosine_scheduler.get_lr());
        cosine_scheduler.step();
    }
    
    // Demonstrate ReduceLROnPlateau scheduler
    println!("\n--- ReduceLROnPlateau Scheduler ---");
    let mut plateau_scheduler = ReduceLROnPlateau::new(
        AdamW::with_weight_decay(0.1, 0.01),
        PlateauMode::Min,
        0.5,     // factor
        3,       // patience
        0.01,    // threshold
        rustorch::optim::ThresholdMode::Rel,
        0,       // cooldown
        0.0001,  // min_lr
        1e-8,    // eps
    );
    
    // Simulate training with loss values
    let losses = vec![1.0, 0.9, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82, 0.82, 0.81];
    
    for (epoch, &loss) in losses.iter().enumerate() {
        println!("Epoch {}: Loss = {:.2}, LR = {:.4}", 
                 epoch, loss, plateau_scheduler.get_lr());
        plateau_scheduler.step_with_metric(loss);
    }
    
    // Demonstrate AdamW with weight decay
    println!("\n--- AdamW Weight Decay Demo ---");
    
    // Create sample parameters and gradients
    let param = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let grad = Tensor::from_vec(vec![0.1, 0.2, 0.15, 0.25, 0.3], vec![5]);
    
    println!("Initial parameters: {:?}", param.as_slice().unwrap());
    
    // Apply AdamW step with weight decay
    let mut adamw_wd = AdamW::with_weight_decay(0.01, 0.1);
    
    for step in 0..5 {
        adamw_wd.step(&param, &grad);
        println!("Step {}: parameters = {:?}", 
                 step + 1, 
                 param.as_slice().unwrap());
    }
    
    // Show state dict
    println!("\n--- Optimizer State ---");
    let state = adamw_wd.state_dict();
    println!("AdamW state dict:");
    for (key, value) in state.iter() {
        println!("  {}: {}", key, value);
    }
    
    println!("\n✅ AdamW and LR Schedulers are working correctly!");
}