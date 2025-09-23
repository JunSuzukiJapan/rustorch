//! TensorBoard integration demonstration
//! TensorBoard統合デモンストレーション

use rustorch::tensorboard::{EdgeDef, GraphDef, ImageData, NodeDef, SummaryWriter};
use std::collections::HashMap;
use std::fs;

fn main() {
    println!("📊 TensorBoard Integration Demo");
    println!("===============================\n");

    // Clean up previous runs
    if std::path::Path::new("runs").exists() {
        fs::remove_dir_all("runs").unwrap_or_default();
    }

    // Demonstration of different logging features
    demo_scalar_logging();
    demo_histogram_logging();
    demo_image_logging();
    demo_text_logging();
    demo_graph_logging();
    // Skip time-intensive demos
    // demo_embedding_logging();
    // demo_training_loop_logging();

    println!("\n✅ TensorBoard demo completed!");
    println!("📝 Check the 'runs' directory for TensorBoard logs");
    println!("🚀 Run 'tensorboard --logdir=runs' to visualize");
}

/// Demonstrate scalar logging
fn demo_scalar_logging() {
    println!("1️⃣ Scalar Logging Demo");
    println!("----------------------");

    let mut writer = SummaryWriter::new("runs/scalar_demo").unwrap();

    // Simulate training metrics
    for epoch in 0..10 {
        let loss = 2.0 * (-0.1 * epoch as f32).exp() + 0.1;
        let accuracy = 1.0 - (-0.05 * epoch as f32).exp();
        let learning_rate = 0.001 * 0.95_f32.powi(epoch / 10);

        writer.add_scalar("Loss/Train", loss, Some(epoch as usize));
        writer.add_scalar("Accuracy/Train", accuracy, Some(epoch as usize));
        writer.add_scalar("Learning_Rate", learning_rate, Some(epoch as usize));

        // Add validation metrics every 5 epochs
        if epoch % 5 == 0 {
            let val_loss = loss * 1.2;
            let val_accuracy = accuracy * 0.95;

            writer.add_scalar("Loss/Validation", val_loss, Some(epoch as usize));
            writer.add_scalar("Accuracy/Validation", val_accuracy, Some(epoch as usize));
        }
    }

    writer.close();
    println!("✓ Scalar metrics logged\n");
}

/// Demonstrate histogram logging
fn demo_histogram_logging() {
    println!("2️⃣ Histogram Logging Demo");
    println!("-------------------------");

    let mut writer = SummaryWriter::new("runs/histogram_demo").unwrap();

    // Simulate weight distributions during training
    for epoch in 0..20 {
        // Generate weights with changing distribution
        let mean = 0.0;
        let std = 1.0 / (1.0 + epoch as f32 * 0.1);

        let weights: Vec<f32> = (0..1000)
            .map(|_| {
                let u1: f32 = rand::random();
                let u2: f32 = rand::random();
                mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let biases: Vec<f32> = (0..100)
            .map(|_| rand::random::<f32>() * 0.1 - 0.05)
            .collect();

        writer.add_histogram("Weights/Layer1", &weights, Some(epoch as usize));
        writer.add_histogram("Biases/Layer1", &biases, Some(epoch as usize));

        // Add gradients
        let gradients: Vec<f32> = weights
            .iter()
            .map(|w| w * 0.01 + (rand::random::<f32>() - 0.5) * 0.001)
            .collect();

        writer.add_histogram("Gradients/Layer1", &gradients, Some(epoch as usize));
    }

    writer.close();
    println!("✓ Histogram data logged\n");
}

/// Demonstrate image logging
fn demo_image_logging() {
    println!("3️⃣ Image Logging Demo");
    println!("---------------------");

    let mut writer = SummaryWriter::new("runs/image_demo").unwrap();

    // Create sample images
    for step in 0..5 {
        // Generate a simple pattern
        let width = 64;
        let height = 64;
        let channels = 3;

        let mut data = Vec::with_capacity(width * height * channels);

        for y in 0..height {
            for x in 0..width {
                let r = ((x + step * 10) % 255) as u8;
                let g = ((y + step * 10) % 255) as u8;
                let b = (((x + y) + step * 10) % 255) as u8;

                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        let image = ImageData {
            height: height as u32,
            width: width as u32,
            channels: channels as u32,
            data,
        };

        writer.add_image("Generated_Images", &image, Some(step));
    }

    writer.close();
    println!("✓ Images logged\n");
}

/// Demonstrate text logging
fn demo_text_logging() {
    println!("4️⃣ Text Logging Demo");
    println!("--------------------");

    let mut writer = SummaryWriter::new("runs/text_demo").unwrap();

    let hyperparameters = [
        "Learning Rate: 0.001",
        "Batch Size: 32",
        "Optimizer: Adam",
        "Loss Function: CrossEntropy",
        "Architecture: ResNet-50",
    ];

    for (i, param) in hyperparameters.iter().enumerate() {
        writer.add_text("Hyperparameters", param, Some(i));
    }

    // Log training progress
    let progress_messages = [
        "Training started",
        "First epoch completed - Loss: 2.34",
        "Validation accuracy improved: 85.2%",
        "Learning rate reduced to 0.0001",
        "Training completed - Best accuracy: 94.7%",
    ];

    for (i, message) in progress_messages.iter().enumerate() {
        writer.add_text("Training_Log", message, Some(i * 10));
    }

    writer.close();
    println!("✓ Text logs added\n");
}

/// Demonstrate graph logging
fn demo_graph_logging() {
    println!("5️⃣ Graph Logging Demo");
    println!("---------------------");

    let mut writer = SummaryWriter::new("runs/graph_demo").unwrap();

    // Create a simple computational graph
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Input node
    nodes.push(NodeDef {
        name: "input".to_string(),
        op: "Placeholder".to_string(),
        inputs: vec![],
        attrs: HashMap::from([("shape".to_string(), "[32, 784]".to_string())]),
    });

    // Linear layers
    nodes.push(NodeDef {
        name: "linear1".to_string(),
        op: "Linear".to_string(),
        inputs: vec!["input".to_string()],
        attrs: HashMap::from([
            ("in_features".to_string(), "784".to_string()),
            ("out_features".to_string(), "256".to_string()),
        ]),
    });

    nodes.push(NodeDef {
        name: "relu1".to_string(),
        op: "ReLU".to_string(),
        inputs: vec!["linear1".to_string()],
        attrs: HashMap::new(),
    });

    nodes.push(NodeDef {
        name: "linear2".to_string(),
        op: "Linear".to_string(),
        inputs: vec!["relu1".to_string()],
        attrs: HashMap::from([
            ("in_features".to_string(), "256".to_string()),
            ("out_features".to_string(), "10".to_string()),
        ]),
    });

    nodes.push(NodeDef {
        name: "output".to_string(),
        op: "Softmax".to_string(),
        inputs: vec!["linear2".to_string()],
        attrs: HashMap::new(),
    });

    // Create edges
    edges.push(EdgeDef {
        source: "input".to_string(),
        target: "linear1".to_string(),
        label: None,
    });

    edges.push(EdgeDef {
        source: "linear1".to_string(),
        target: "relu1".to_string(),
        label: None,
    });

    edges.push(EdgeDef {
        source: "relu1".to_string(),
        target: "linear2".to_string(),
        label: None,
    });

    edges.push(EdgeDef {
        source: "linear2".to_string(),
        target: "output".to_string(),
        label: None,
    });

    let graph = GraphDef { nodes, edges };
    writer.add_graph(&graph);

    writer.close();
    println!("✓ Computational graph logged\n");
}

/// Demonstrate the macro interface
#[allow(dead_code)]
fn demo_macro_interface() {
    let mut writer = SummaryWriter::new("runs/macro_demo").unwrap();

    // Using macros for cleaner code
    rustorch::tb_log!(writer, scalar: "loss", 0.5);
    rustorch::tb_log!(writer, text: "status", "Training started");

    let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    rustorch::tb_log!(writer, histogram: "weights", &values);

    writer.close();
}
