//! Gradient flow visualization demonstration
//! 勾配フロー可視化デモンストレーション

use rustorch::autograd::{Variable, visualization::*};
use rustorch::tensor::Tensor;
use std::path::Path;

fn main() {
    println!("🔍 Gradient Flow Visualization Demo");
    println!("====================================\n");

    // Simple neural network gradient flow
    simple_network_gradient_flow();
    
    // Gradient flow analysis with trends
    gradient_trend_analysis();
    
    // Detect gradient flow issues
    detect_gradient_issues();
}

/// Demonstrate gradient flow visualization for a simple network
/// 単純なネットワークの勾配フロー可視化をデモ
fn simple_network_gradient_flow() {
    println!("📊 Simple Network Gradient Flow");
    println!("--------------------------------");

    // Create a simple network with explicit type
    let input: Variable<f32> = Variable::new(Tensor::randn(&[32, 10]), false);
    let weight1: Variable<f32> = Variable::new(Tensor::randn(&[10, 20]), true);
    let weight2: Variable<f32> = Variable::new(Tensor::randn(&[20, 5]), true);
    
    // Forward pass
    let hidden = input.matmul(&weight1);
    let output = hidden.matmul(&weight2);
    let loss = output.sum();
    
    // Backward pass
    loss.backward();
    
    // Create visualizer
    let mut visualizer = GradientFlowVisualizer::new();
    
    // Trace the computation graph
    visualizer.trace_from_variable(&input, "input");
    visualizer.trace_from_variable(&weight1, "weight1");
    visualizer.trace_from_variable(&weight2, "weight2");
    visualizer.trace_from_variable(&hidden, "hidden");
    visualizer.trace_from_variable(&output, "output");
    visualizer.trace_from_variable(&loss, "loss");
    
    // Add operations
    visualizer.add_operation("matmul", vec![0, 1], 3);
    visualizer.add_operation("matmul", vec![3, 2], 4);
    visualizer.add_operation("sum", vec![4], 5);
    
    // Get and print summary
    let summary = visualizer.gradient_flow_summary();
    println!("{}", summary);
    
    // Save to file
    let dot_path = Path::new("gradient_flow.dot");
    match visualizer.save_to_file(dot_path) {
        Ok(_) => println!("✅ Gradient flow saved to gradient_flow.dot"),
        Err(e) => println!("❌ Failed to save gradient flow: {}", e),
    }
    
    // Generate DOT representation
    let dot = visualizer.to_dot();
    println!("\n📝 DOT Graph (first 500 chars):");
    println!("{}", &dot[..dot.len().min(500)]);
    println!("...\n");
}

/// Demonstrate gradient trend analysis
/// 勾配トレンド分析をデモ
fn gradient_trend_analysis() {
    println!("📈 Gradient Trend Analysis");
    println!("--------------------------");
    
    let mut analyzer = GradientFlowAnalyzer::new(100);
    
    // Simulate training iterations
    for epoch in 0..50 {
        // Simulate different gradient behaviors
        let scale = if epoch < 10 {
            // Normal gradients
            1.0
        } else if epoch < 20 {
            // Decreasing gradients
            1.0 / ((epoch - 9) as f32)
        } else if epoch < 30 {
            // Increasing gradients
            (epoch - 19) as f32 * 0.5
        } else {
            // Stable gradients
            2.0
        };
        
        // Create tensors with different scales
        let weight1_grad = Tensor::from_vec(
            vec![0.1 * scale; 100],
            vec![10, 10]
        );
        let weight2_grad = Tensor::from_vec(
            vec![0.05 * scale; 50],
            vec![10, 5]
        );
        let bias_grad = Tensor::from_vec(
            vec![0.01 * scale * scale; 10],
            vec![10]
        );
        
        // Record gradients
        analyzer.record_gradient("weight1", &weight1_grad);
        analyzer.record_gradient("weight2", &weight2_grad);
        analyzer.record_gradient("bias", &bias_grad);
    }
    
    // Analyze trends
    let trends = analyzer.analyze_trends();
    
    println!("\n🔍 Gradient Trends:");
    for (name, trend) in &trends {
        println!("  {} -> {:?}", name, trend);
    }
    
    // Print histories
    println!("\n📊 Gradient History (last 10 values):");
    for name in ["weight1", "weight2", "bias"] {
        if let Some(history) = analyzer.get_history(name) {
            let recent: Vec<String> = history.iter()
                .rev()
                .take(10)
                .map(|x| format!("{:.4}", x))
                .collect();
            println!("  {}: [{}]", name, recent.join(", "));
        }
    }
}

/// Demonstrate detection of gradient flow issues
/// 勾配フローの問題検出をデモ
fn detect_gradient_issues() {
    println!("\n⚠️  Gradient Flow Issue Detection");
    println!("----------------------------------");
    
    let mut visualizer = GradientFlowVisualizer::new();
    
    // Simulate problematic gradients through variables
    let vanishing_weight: Variable<f32> = Variable::new(Tensor::randn(&[100, 50]), true);
    let normal_weight: Variable<f32> = Variable::new(Tensor::randn(&[50, 25]), true);
    let exploding_weight: Variable<f32> = Variable::new(Tensor::randn(&[25, 10]), true);
    let unused_weight: Variable<f32> = Variable::new(Tensor::randn(&[10, 10]), true);
    
    // Add simulated gradient values
    if let Ok(mut grad) = vanishing_weight.grad().write() {
        *grad = Some(Tensor::from_vec(vec![1e-8; 5000], vec![100, 50]));
    }
    if let Ok(mut grad) = normal_weight.grad().write() {
        *grad = Some(Tensor::from_vec(vec![0.1; 1250], vec![50, 25]));
    }
    if let Ok(mut grad) = exploding_weight.grad().write() {
        *grad = Some(Tensor::from_vec(vec![1e5; 250], vec![25, 10]));
    }
    // unused_weight has no gradient (disconnected)
    
    // Trace variables for visualization
    visualizer.trace_from_variable(&vanishing_weight, "layer1_weight");
    visualizer.trace_from_variable(&normal_weight, "layer2_weight");
    visualizer.trace_from_variable(&exploding_weight, "layer3_weight");
    visualizer.trace_from_variable(&unused_weight, "unused_weight");
    
    // Detect issues
    let issues = visualizer.detect_issues();
    
    if issues.is_empty() {
        println!("✅ No gradient flow issues detected!");
    } else {
        println!("❌ Found {} gradient flow issues:", issues.len());
        for issue in &issues {
            println!("  - {}", issue);
        }
    }
    
    // Summary with issues
    println!("\n📊 Network Summary:");
    let summary = visualizer.gradient_flow_summary();
    println!("  Total parameters: {}", summary.parameter_nodes);
    println!("  Parameters with gradients: {}", summary.nodes_with_gradients);
    println!("  Gradient norm range: [{:.2e}, {:.2e}]", 
        summary.min_gradient_norm, summary.max_gradient_norm);
    println!("  Average gradient norm: {:.2e}", summary.avg_gradient_norm);
}

