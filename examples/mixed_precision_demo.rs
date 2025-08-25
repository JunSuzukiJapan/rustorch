//! Mixed Precision Training demonstration
//! 混合精度学習のデモ

use rustorch::amp::{
    autocast, cast_to_fp16, cast_to_fp32, maybe_autocast_f32, AMPConfig, AMPOptimizer, GradScaler,
    ParamGroup,
};
use rustorch::dtype::DType;
use rustorch::optim::{adamw::AdamW, sgd::SGD};
use rustorch::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Mixed Precision Training Demo");
    println!("================================");

    // 1. Basic Autocast Usage
    println!("\n1. Basic Autocast Usage");
    basic_autocast_demo()?;

    // 2. Gradient Scaler Demo
    println!("\n2. Gradient Scaler Demo");
    gradient_scaler_demo()?;

    // 3. AMP Optimizer Demo
    println!("\n3. AMP Optimizer Demo");
    amp_optimizer_demo()?;

    // 4. Data Type Conversion Demo
    println!("\n4. Data Type Conversion Demo");
    dtype_conversion_demo()?;

    // 5. Complete Training Loop Demo
    println!("\n5. Complete Training Loop Demo");
    complete_training_demo()?;

    println!("\n✅ Mixed Precision Training demo completed successfully!");
    Ok(())
}

fn basic_autocast_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing autocast context manager...");

    // Create some test tensors
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);

    // Test FP16 autocast
    {
        let _ctx = autocast("cuda", true, Some(DType::Float16));
        println!("   - FP16 autocast enabled");

        // Operations inside autocast context
        let result = maybe_autocast_f32(&x);
        println!("     Autocast result shape: {:?}", result.shape());
    }

    // Test BF16 autocast
    {
        let _ctx = autocast("cuda", true, Some(DType::BFloat16));
        println!("   - BF16 autocast enabled");

        let result = maybe_autocast_f32(&y);
        println!("     Autocast result shape: {:?}", result.shape());
    }

    println!("   ✓ Autocast context working correctly");
    Ok(())
}

fn gradient_scaler_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing gradient scaler...");

    // Create gradient scaler
    let mut scaler = GradScaler::new(
        Some(1024.0), // init_scale
        Some(2.0),    // growth_factor
        Some(0.5),    // backoff_factor
        Some(100),    // growth_interval
        Some(true),   // enabled
    );

    println!("   - Initial scale: {}", scaler.get_scale());

    // Test normal gradients
    let normal_grads = vec![
        Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]),
        Tensor::from_vec(vec![0.4, 0.5, 0.6], vec![3]),
    ];

    let has_overflow = scaler.check_overflow(&normal_grads);
    println!("   - Normal gradients overflow: {}", has_overflow);

    // Test overflow gradients
    let overflow_grads = vec![Tensor::from_vec(vec![1e10, 2e10, 3e10], vec![3])];

    let has_overflow = scaler.check_overflow(&overflow_grads);
    println!("   - Overflow gradients overflow: {}", has_overflow);

    // Test scale tensor
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let scaled = scaler.scale_tensor(&tensor);
    println!("   - Scaled tensor first element: {:?}", scaled.get(&[0]));

    // Test statistics
    let stats = scaler.get_stats();
    println!(
        "   - Scaler stats: enabled={}, scale={}",
        stats.enabled, stats.current_scale
    );

    println!("   ✓ Gradient scaler working correctly");
    Ok(())
}

fn amp_optimizer_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing AMP optimizer wrapper...");

    // Create base optimizer
    let sgd = SGD::new(0.01);

    // Create AMP optimizer
    let mut amp_optimizer = AMPOptimizer::new(sgd, None);

    // Add parameter groups
    let group1 = ParamGroup {
        param_ids: vec![0, 1],
        clip_gradients: true,
        max_grad_norm: Some(1.0),
        use_amp: true,
    };

    let group2 = ParamGroup {
        param_ids: vec![2, 3],
        clip_gradients: false,
        max_grad_norm: None,
        use_amp: false,
    };

    amp_optimizer.add_param_group(group1);
    amp_optimizer.add_param_group(group2);

    // Test parameters and gradients
    let params = vec![
        Tensor::from_vec(vec![1.0, 2.0], vec![2]),
        Tensor::from_vec(vec![3.0, 4.0], vec![2]),
        Tensor::from_vec(vec![5.0, 6.0], vec![2]),
        Tensor::from_vec(vec![7.0, 8.0], vec![2]),
    ];

    let mut grads = vec![
        Tensor::from_vec(vec![0.1, 0.2], vec![2]),
        Tensor::from_vec(vec![0.3, 0.4], vec![2]),
        Tensor::from_vec(vec![0.5, 0.6], vec![2]),
        Tensor::from_vec(vec![0.7, 0.8], vec![2]),
    ];

    // Perform optimization step
    let result = amp_optimizer.step(&params, &mut grads);
    println!("   - Step result: {:?}", result);

    // Get training statistics
    let stats = amp_optimizer.get_training_stats();
    println!(
        "   - Training stats: total_steps={}, success_rate={:.3}",
        stats.total_steps,
        stats.success_rate()
    );

    println!("   ✓ AMP optimizer working correctly");
    Ok(())
}

fn dtype_conversion_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing data type conversions...");

    // Create FP32 tensor
    let fp32_tensor = Tensor::from_vec(vec![1.0, 2.5, 3.14159, -0.5], vec![4]);
    println!("   - Original FP32 tensor: {:?}", fp32_tensor.as_slice());

    // Convert to FP16 (simulated)
    let fp16_tensor = cast_to_fp16(&fp32_tensor);
    println!("   - FP16 tensor shape: {:?}", fp16_tensor.shape());

    // Convert back to FP32 (already f32)
    let restored_tensor = cast_to_fp32(&fp16_tensor);
    println!(
        "   - Restored FP32 tensor: {:?}",
        restored_tensor.as_slice()
    );

    // Test MixedPrecisionTensor trait
    use rustorch::amp::MixedPrecisionTensor;
    let memory_fp32 = fp32_tensor.memory_footprint();
    let memory_fp16_sim = fp32_tensor.memory_footprint_for_dtype(DType::Float16);
    println!("   - Memory footprint FP32: {} bytes", memory_fp32);
    println!(
        "   - Memory footprint FP16 (simulated): {} bytes",
        memory_fp16_sim
    );
    println!(
        "   - Memory savings: {:.1}%",
        (1.0 - memory_fp16_sim as f32 / memory_fp32 as f32) * 100.0
    );

    // Test casting capabilities
    assert!(fp32_tensor.can_cast_to(DType::Float16));
    assert!(fp32_tensor.can_cast_to(DType::BFloat16));

    println!("   ✓ Data type conversions working correctly");
    Ok(())
}

fn complete_training_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing complete training loop with AMP...");

    // Setup AMP configuration
    let amp_config = AMPConfig::default(); // FP16 with dynamic scaling
    rustorch::amp::enable_amp(amp_config);

    // Create model parameters (simplified)
    let params = vec![
        Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        Tensor::from_vec(vec![0.5, 0.6], vec![2]),
    ];

    // Create optimizer with AMP
    let adamw = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01, false);
    let mut amp_optimizer = AMPOptimizer::new(adamw, None);

    // Add parameter group
    let param_group = ParamGroup {
        param_ids: vec![0, 1],
        clip_gradients: true,
        max_grad_norm: Some(1.0),
        use_amp: true,
    };
    amp_optimizer.add_param_group(param_group);

    // Create learning rate scheduler (we'll simulate this without actual scheduler)
    let total_epochs = 10;
    let initial_lr = 0.001;

    println!("   - Starting training simulation...");

    // Training loop simulation
    for epoch in 0..10 {
        // Forward pass (simplified)
        let _ctx = autocast("cuda", true, Some(DType::Float16));

        // Simulate loss computation
        let loss_val = 1.0 / (epoch as f32 + 1.0); // Decreasing loss
        let loss = Tensor::from_vec(vec![loss_val], vec![1]);

        // Scale loss
        let _scaled_loss = amp_optimizer.scaler().scale_tensor(&loss);

        // Simulate gradients
        let mut grads = vec![
            Tensor::from_vec(vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
            Tensor::from_vec(vec![0.05, 0.06], vec![2]),
        ];

        // Optimization step
        let step_result = amp_optimizer.step(&params, &mut grads);

        // Simulate learning rate scheduling (cosine annealing)
        let current_lr = initial_lr
            * 0.5
            * (1.0 + (std::f32::consts::PI * epoch as f32 / total_epochs as f32).cos());

        if epoch % 3 == 0 {
            println!(
                "     Epoch {}: loss={:.4}, lr={:.6}, step={:?}",
                epoch,
                loss_val,
                current_lr,
                match step_result {
                    rustorch::amp::StepResult::Success { scale, .. } =>
                        format!("Success(scale={:.0})", scale),
                    rustorch::amp::StepResult::Overflow { scale, .. } =>
                        format!("Overflow(scale={:.0})", scale),
                    rustorch::amp::StepResult::InfNan { scale } =>
                        format!("InfNan(scale={:.0})", scale),
                }
            );
        }

        // Update adaptive scaling
        amp_optimizer.update_schedule();
    }

    // Final statistics
    let final_stats = amp_optimizer.get_training_stats();
    println!("   - Final training statistics:");
    println!("     * Total steps: {}", final_stats.total_steps);
    println!(
        "     * Success rate: {:.1}%",
        final_stats.success_rate() * 100.0
    );
    println!(
        "     * Overflow rate: {:.1}%",
        final_stats.overflow_rate * 100.0
    );
    println!("     * Is stable: {}", final_stats.is_stable());

    // Get recommendations
    let recommendations = final_stats.get_recommendations();
    println!("   - Recommendations:");
    for rec in recommendations {
        println!("     * {}", rec);
    }

    // Disable AMP
    rustorch::amp::disable_amp();

    println!("   ✓ Complete training loop working correctly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_demo() {
        assert!(basic_autocast_demo().is_ok());
        assert!(gradient_scaler_demo().is_ok());
        assert!(amp_optimizer_demo().is_ok());
        assert!(dtype_conversion_demo().is_ok());
        assert!(complete_training_demo().is_ok());
    }
}
