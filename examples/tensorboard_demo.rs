//! TensorBoard integration demonstration
//! TensorBoard統合デモンストレーション

use rustorch::tensorboard::{SummaryWriter, ImageData, GraphDef, NodeDef, EdgeDef, python_compat};
use rustorch::tensor::Tensor;
use rustorch::nn::Linear;
use rustorch::autograd::Variable;
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
    demo_embedding_logging();
    demo_training_loop_logging();
    
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
    for epoch in 0..50 {
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
        let gradients: Vec<f32> = weights.iter()
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
        
        writer.add_image("Generated_Images", &image, Some(step as usize));
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

/// Demonstrate embedding logging
fn demo_embedding_logging() -> Result<(), std::io::Error> {
    println!("6️⃣ Embedding Logging Demo");
    println!("-------------------------");
    
    let mut writer = SummaryWriter::new("runs/embedding_demo")?;
    
    // Generate sample embeddings
    let embedding_dim = 50;
    let num_words = 100;
    
    let mut embeddings = Vec::new();
    let mut metadata = Vec::new();
    
    for i in 0..num_words {
        let mut embedding = Vec::new();
        for _ in 0..embedding_dim {
            embedding.push(rand::random::<f32>() * 2.0 - 1.0);
        }
        embeddings.push(embedding);
        metadata.push(format!("word_{}", i));
    }
    
    writer.add_embedding(&embeddings, Some(metadata), Some("word_embeddings"))?;
    writer.close();
    
    println!("✓ Embeddings logged\n");
    Ok(())
}

/// Demonstrate training loop with comprehensive logging
fn demo_training_loop_logging() {
    println!("7️⃣ Training Loop Demo");
    println!("---------------------");
    
    // Use Python-compatible API for easier integration
    let mut writer = python_compat::create_writer(Some("runs")).unwrap();
    
    // Simulate a training loop
    let layer1: Linear<f32> = Linear::new(784, 256);
    let layer2: Linear<f32> = Linear::new(256, 10);
    
    for epoch in 0..10 {
        writer.add_text("Status", &format!("Starting epoch {}", epoch), Some(epoch));
        
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;
        
        // Simulate batch training
        for batch in 0..100 {
            let input = Variable::new(Tensor::<f32>::randn(&[32, 784]), false);
            
            // Forward pass
            let hidden = layer1.forward(&input);
            let output = layer2.forward(&hidden);
            
            // Simulate loss and accuracy
            let loss = 2.0 * (-0.01 * (epoch * 100 + batch) as f32).exp() + 0.1;
            let acc = 1.0 - (-0.005 * (epoch * 100 + batch) as f32).exp();
            
            epoch_loss += loss;
            epoch_acc += acc;
            
            // Log every 20 batches
            if batch % 20 == 0 {
                let step = epoch * 100 + batch;
                writer.add_scalar("Batch/Loss", loss, Some(step));
                writer.add_scalar("Batch/Accuracy", acc, Some(step));
            }
        }
        
        // Log epoch metrics
        epoch_loss /= 100.0;
        epoch_acc /= 100.0;
        
        writer.add_scalar("Epoch/Loss", epoch_loss, Some(epoch));
        writer.add_scalar("Epoch/Accuracy", epoch_acc, Some(epoch));
        
        // Log model parameters as histograms
        let weights: Vec<f32> = (0..1000).map(|_| rand::random::<f32>() * 0.1).collect();
        writer.add_histogram(&format!("Weights/Epoch_{}", epoch), &weights, Some(epoch));
        
        println!("  Epoch {}: Loss = {:.4}, Accuracy = {:.4}", epoch, epoch_loss, epoch_acc);
    }
    
    writer.close();
    println!("✓ Training loop logged\n");
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