//! Learning Rate Scheduler demonstration
//! 学習率スケジューラーのデモンストレーション

use rustorch::prelude::*;
use rustorch::nn::{Sequential, Linear, Conv2d, BatchNorm2d, Dropout};
use rustorch::optim::{PlateauMode, ThresholdMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 RusTorch Learning Rate Scheduler Demo");
    println!("========================================");
    
    // Create a sample network for training
    let mut network = create_sample_network();
    
    // Demonstrate different scheduler types
    demonstrate_step_lr(&mut network);
    demonstrate_exponential_lr(&mut network);
    demonstrate_cosine_annealing_lr(&mut network);
    demonstrate_reduce_lr_on_plateau(&mut network);
    
    println!("\n🎉 Learning rate scheduler demo completed successfully!");
    
    Ok(())
}

fn create_sample_network() -> Sequential<f32> {
    let mut network = Sequential::<f32>::new();
    
    // Simple CNN for demonstration
    network.add_module(Conv2d::new(3, 16, (3, 3), Some((1, 1)), Some((1, 1)), Some(true)));
    network.add_module(BatchNorm2d::new(16, None, None, None));
    network.add_module(Dropout::new(0.2, None));
    
    network
}

fn demonstrate_step_lr(network: &mut Sequential<f32>) {
    println!("\n🔹 StepLR Scheduler Demo:");
    println!("========================");
    
    // Get network parameters
    let params = network.parameters();
    let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
    
    // Create StepLR scheduler: decay by 0.1 every 30 epochs
    let mut scheduler = StepLR::new(&mut optimizer, 30, 0.1, None);
    
    println!("StepLR: step_size=30, gamma=0.1");
    println!("Initial learning rate: {:?}", scheduler.get_lr());
    
    // Simulate training epochs
    for epoch in 0..100 {
        scheduler.step();
        
        if epoch % 20 == 0 || epoch == 29 || epoch == 30 || epoch == 59 || epoch == 60 {
            println!("Epoch {}: LR = {:?}", epoch + 1, scheduler.get_lr());
        }
    }
    
    println!("✅ StepLR demonstration completed");
}

fn demonstrate_exponential_lr(network: &mut Sequential<f32>) {
    println!("\n🔹 ExponentialLR Scheduler Demo:");
    println!("===============================");
    
    let params = network.parameters();
    let mut optimizer = SGD::new(params, 0.1, Some(0.9), None, None, None);
    
    // Create ExponentialLR scheduler: decay by 0.95 every epoch
    let mut scheduler = ExponentialLR::new(&mut optimizer, 0.95, None);
    
    println!("ExponentialLR: gamma=0.95");
    println!("Initial learning rate: {:?}", scheduler.get_lr());
    
    // Simulate training epochs
    for epoch in 0..20 {
        scheduler.step();
        
        if epoch % 5 == 0 || epoch < 5 {
            println!("Epoch {}: LR = {:?}", epoch + 1, scheduler.get_lr());
        }
    }
    
    println!("✅ ExponentialLR demonstration completed");
}

fn demonstrate_cosine_annealing_lr(network: &mut Sequential<f32>) {
    println!("\n🔹 CosineAnnealingLR Scheduler Demo:");
    println!("===================================");
    
    let params = network.parameters();
    let mut optimizer = SGD::new(params, 0.1, Some(0.9), None, None, None);
    
    // Create CosineAnnealingLR scheduler: T_max=50, eta_min=0.001
    let mut scheduler = CosineAnnealingLR::new(&mut optimizer, 50, Some(0.001), None);
    
    println!("CosineAnnealingLR: T_max=50, eta_min=0.001");
    println!("Initial learning rate: {:?}", scheduler.get_lr());
    
    // Simulate training epochs and show cosine annealing pattern
    let epochs_to_show = vec![0, 12, 25, 37, 49];
    
    for epoch in 0..50 {
        scheduler.step();
        
        if epochs_to_show.contains(&epoch) {
            println!("Epoch {}: LR = {:?} (T={}/T_max)", epoch + 1, scheduler.get_lr(), epoch + 1);
        }
    }
    
    println!("✅ CosineAnnealingLR demonstration completed");
}

fn demonstrate_reduce_lr_on_plateau(network: &mut Sequential<f32>) {
    println!("\n🔹 ReduceLROnPlateau Scheduler Demo:");
    println!("===================================");
    
    let params = network.parameters();
    let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
    
    // Create ReduceLROnPlateau scheduler
    let mut scheduler = ReduceLROnPlateau::new(
        &mut optimizer,
        Some(PlateauMode::Min),  // Minimize validation loss
        Some(0.5),               // Factor to reduce LR
        Some(3),                 // Patience: wait 3 epochs before reducing
        Some(1e-4),              // Threshold for improvement
        Some(ThresholdMode::Rel), // Relative threshold mode
        Some(2),                 // Cooldown: wait 2 epochs after reduction
        Some(1e-6),              // Minimum learning rate
        Some(1e-8),              // EPS for LR comparison
    );
    
    println!("ReduceLROnPlateau: mode=Min, factor=0.5, patience=3");
    println!("Initial learning rate: {:?}", scheduler.get_lr());
    
    // Simulate training with validation loss that plateaus
    let validation_losses = vec![
        1.0, 0.9, 0.85, 0.82, 0.81, 0.81, 0.82, 0.81, // First plateau (epoch 4-7)
        0.8, 0.75, 0.74, 0.74, 0.74, 0.74, 0.75,      // Second plateau (epoch 11-14)
        0.73, 0.72, 0.72, 0.72, 0.72, 0.72,           // Third plateau (epoch 17-20)
    ];
    
    for (epoch, &val_loss) in validation_losses.iter().enumerate() {
        scheduler.step_with_metric(val_loss);
        
        let state = scheduler.state_dict();
        println!(
            "Epoch {}: Val Loss = {:.3}, LR = {:?}, Bad Epochs = {}, Cooldown = {}",
            epoch + 1,
            val_loss,
            scheduler.get_lr(),
            state.num_bad_epochs,
            state.cooldown_counter
        );
    }
    
    println!("✅ ReduceLROnPlateau demonstration completed");
}

fn simulate_training_with_scheduler() {
    println!("\n🏋️ Simulated Training with Multiple Schedulers:");
    println!("===============================================");
    
    // Create a simple model
    let mut model = Sequential::<f32>::new();
    model.add_module(Linear::new(784, 128));
    model.add_module(Dropout::new(0.5, None));
    model.add_module(Linear::new(128, 10));
    
    // Create sample data
    let input = Variable::new(
        Tensor::from_vec((0..784).map(|i| (i as f32) * 0.001).collect(), vec![1, 784]),
        true
    );
    let target = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 10]),
        false
    );
    
    println!("Created model and sample data");
    println!("Input shape: {:?}", input.data().read().unwrap().shape());
    println!("Target shape: {:?}", target.data().read().unwrap().shape());
    
    // Setup optimizer and scheduler
    let params = model.parameters();
    let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
    let mut scheduler = StepLR::new(&mut optimizer, 10, 0.5, None);
    
    println!("\nTraining simulation (10 epochs):");
    
    for epoch in 0..10 {
        // Forward pass
        let output = model.forward(&input);
        
        // Compute loss
        let loss = mse_loss(&output, &target);
        let loss_value = loss.data().read().unwrap().as_array().iter().next().unwrap().clone();
        
        // Backward pass (simplified)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        // Step scheduler
        scheduler.step();
        
        println!(
            "Epoch {}: Loss = {:.6}, LR = {:?}",
            epoch + 1,
            loss_value,
            scheduler.get_lr()
        );
    }
    
    println!("✅ Training simulation completed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch::tensor::Tensor;
    use rustorch::autograd::Variable;
    
    #[test]
    fn test_scheduler_integration() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        // Test StepLR
        let mut step_scheduler = StepLR::new(&mut optimizer, 5, 0.1, None);
        let initial_lr = step_scheduler.get_lr();
        
        // Step 5 times to trigger decay
        for _ in 0..6 {
            step_scheduler.step();
        }
        
        let decayed_lr = step_scheduler.get_lr();
        assert!(decayed_lr[0] < initial_lr[0]);
    }
    
    #[test]
    fn test_multiple_schedulers() {
        let params1 = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let params2 = vec![Variable::new(Tensor::ones(&[3, 3]), true)];
        
        let mut optimizer1 = SGD::new(params1, 0.01, Some(0.9), None, None, None);
        let mut optimizer2 = SGD::new(params2, 0.1, Some(0.9), None, None, None);
        
        let mut scheduler1 = ExponentialLR::new(&mut optimizer1, 0.9, None);
        let mut scheduler2 = CosineAnnealingLR::new(&mut optimizer2, 10, None, None);
        
        // Both schedulers should work independently
        scheduler1.step();
        scheduler2.step();
        
        assert!(scheduler1.last_epoch() == 0);
        assert!(scheduler2.last_epoch() == 0);
    }
    
    #[test]
    fn test_scheduler_state_dict() {
        let params = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer = SGD::new(params, 0.01, Some(0.9), None, None, None);
        
        let mut scheduler = StepLR::new(&mut optimizer, 10, 0.5, None);
        
        // Step and save state
        scheduler.step();
        scheduler.step();
        let state = scheduler.state_dict();
        
        // Create new scheduler and load state
        let params2 = vec![Variable::new(Tensor::ones(&[2, 2]), true)];
        let mut optimizer2 = SGD::new(params2, 0.01, Some(0.9), None, None, None);
        let mut scheduler2 = StepLR::new(&mut optimizer2, 10, 0.5, None);
        
        scheduler2.load_state_dict(state.clone());
        
        assert_eq!(scheduler2.last_epoch(), state.last_epoch);
    }
}