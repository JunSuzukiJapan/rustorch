# RusTorch Examples

Comprehensive examples demonstrating RusTorch capabilities across different domains and use cases.

## 📚 Example Categories

### 🔥 Basic Tensor Operations
- **[Basic Usage](#basic-tensor-operations)** - Creating and manipulating tensors
- **[Mathematical Functions](#mathematical-functions)** - Trigonometric, exponential functions
- **[Broadcasting](#broadcasting-operations)** - Automatic shape compatibility
- **[Statistical Operations](#statistical-operations)** - Mean, variance, quantiles

### 🧮 Matrix Operations
- **[Matrix Decomposition](#matrix-decomposition)** - SVD, QR decomposition
- **[Eigenvalue Computation](#eigenvalue-decomposition)** - Symmetric and general eigenvalue problems
- **[Linear Algebra](#linear-algebra-operations)** - Advanced matrix operations

### 🧠 Neural Networks
- **[Linear Regression](#linear-regression)** - Basic regression with automatic differentiation
- **[Neural Network Training](#neural-network-training)** - Multi-layer networks
- **[Convolutional Networks](#convolutional-networks)** - CNN implementations
- **[Recurrent Networks](#recurrent-networks)** - RNN, LSTM, GRU examples

### 🤖 Advanced Architectures
- **[Transformer Models](#transformer-architecture)** - Complete transformer implementation
- **[Attention Mechanisms](#attention-mechanisms)** - Multi-head attention
- **[Embedding Systems](#embedding-systems)** - Word and positional embeddings

### ⚡ Performance Optimization
- **[SIMD Operations](#simd-optimization)** - Vectorized computations
- **[Parallel Processing](#parallel-processing)** - Multi-threaded operations
- **[GPU Acceleration](#gpu-acceleration)** - CUDA, Metal, OpenCL examples
- **[Memory Optimization](#memory-optimization)** - Zero-copy, memory pools

### 🌐 Deployment
- **[WebAssembly](#webassembly-deployment)** - Browser-based machine learning
- **[Production Deployment](#production-deployment)** - Docker, scaling
- **[Model Export/Import](#model-formats)** - PyTorch, ONNX, Safetensors

## 🚀 Running Examples

### Prerequisites

```bash
# Clone repository
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Install dependencies (optional features)
cargo build --all-features
```

### Basic Execution

```bash
# Run specific examples
cargo run --example tensor_demo --release
cargo run --example linear_regression --release
cargo run --example neural_network_demo --release

# Run with specific features
cargo run --example gpu_demo --features="cuda,metal" --release
cargo run --example matrix_decomposition_demo --features="linalg" --release
```

## 📖 Detailed Examples

### Basic Tensor Operations

#### Creating and Manipulating Tensors

```rust
// examples/tensor_demo.rs
use rustorch::tensor::Tensor;

fn main() {
    // Create tensors from data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3]);
    
    println!("Original tensor:");
    println!("Shape: {:?}", tensor.shape());
    println!("Data: {:?}", tensor.as_slice());
    
    // Basic arithmetic operations
    let tensor2 = Tensor::ones(&[2, 3]);
    let sum = &tensor + &tensor2;
    let product = &tensor * &tensor2;
    
    println!("\nAfter addition with ones:");
    println!("Sum: {:?}", sum.as_slice());
    
    // Matrix operations
    let matrix_a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0], 
        vec![2, 2]
    );
    let matrix_b = Tensor::from_vec(
        vec![5.0, 6.0, 7.0, 8.0], 
        vec![2, 2]
    );
    
    let matrix_product = matrix_a.matmul(&matrix_b);
    println!("\nMatrix multiplication result:");
    println!("Shape: {:?}", matrix_product.shape());
    println!("Data: {:?}", matrix_product.as_slice());
}
```

#### Mathematical Functions

```rust
// examples/math_functions_demo.rs
use rustorch::tensor::Tensor;

fn main() {
    let x = Tensor::from_vec(
        vec![0.0, 0.5, 1.0, 1.5, 2.0], 
        vec![5]
    );
    
    // Trigonometric functions
    let sin_x = x.sin();
    let cos_x = x.cos();
    let tan_x = x.tan();
    
    println!("x: {:?}", x.as_slice());
    println!("sin(x): {:?}", sin_x.as_slice());
    println!("cos(x): {:?}", cos_x.as_slice());
    
    // Exponential and logarithmic
    let exp_x = x.exp();
    let log_exp_x = exp_x.log();
    
    println!("exp(x): {:?}", exp_x.as_slice());
    println!("log(exp(x)): {:?}", log_exp_x.as_slice());
    
    // Activation functions
    let relu = x.relu();
    let sigmoid = x.sigmoid();
    let tanh = x.tanh();
    
    println!("ReLU(x): {:?}", relu.as_slice());
    println!("Sigmoid(x): {:?}", sigmoid.as_slice());
    println!("Tanh(x): {:?}", tanh.as_slice());
}
```

### Broadcasting Operations

```rust
// examples/broadcasting_demo.rs
use rustorch::tensor::Tensor;

fn main() {
    // Tensor broadcasting examples
    println!("=== Broadcasting Examples ===");
    
    // Example 1: Adding bias to batch data
    let batch_data = Tensor::from_vec(
        (0..24).map(|i| i as f32).collect(),
        vec![4, 6]  // 4 samples, 6 features
    );
    
    let bias = Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![1, 6]  // Broadcast dimension
    );
    
    let result = batch_data.add(&bias).unwrap();
    println!("Batch + bias shape: {:?}", result.shape());
    
    // Example 2: Scaling features
    let scale = Tensor::from_vec(vec![2.0], vec![1, 1]);
    let scaled = batch_data.mul(&scale).unwrap();
    println!("Scaled data shape: {:?}", scaled.shape());
    
    // Example 3: Complex broadcasting
    let a = Tensor::ones(&[3, 1, 4]);
    let b = Tensor::ones(&[1, 2, 1]);
    let c = a.add(&b).unwrap();
    println!("Complex broadcast result shape: {:?}", c.shape()); // [3, 2, 4]
    
    // Example 4: Neural network layer simulation
    println!("\n=== Neural Network Layer Simulation ===");
    simulate_linear_layer();
}

fn simulate_linear_layer() {
    use rustorch::nn::{Linear, Module};
    use rustorch::autograd::Variable;
    
    // Create a linear layer: 256 inputs -> 128 outputs
    let linear = Linear::<f32>::new(256, 128);
    
    // Batch of 32 samples
    let batch_size = 32;
    let input_features = 256;
    
    let input_data: Vec<f32> = (0..batch_size * input_features)
        .map(|i| (i as f32) * 0.01)
        .collect();
    
    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, input_features]),
        false
    );
    
    // Forward pass with automatic bias broadcasting
    let output = linear.forward(&input);
    println!("Linear layer output shape: {:?}", 
             output.data().read().unwrap().shape());
    
    // The bias (128,) is automatically broadcasted to (32, 128)
    println!("Bias broadcasting: [128] -> [32, 128] ✓");
}
```

### Statistical Operations

```rust
// examples/statistics_demo.rs
use rustorch::tensor::Tensor;

fn main() {
    // Create sample data
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4]  // 3x4 matrix
    );
    
    println!("Original data (3x4):");
    println!("{:?}", data.as_slice());
    
    // Basic statistics
    let mean = data.mean(None);
    let var = data.var(None, true);
    let std = data.std(None, true);
    
    println!("\nOverall statistics:");
    println!("Mean: {:.2}", mean.as_slice()[0]);
    println!("Variance: {:.2}", var.as_slice()[0]);
    println!("Standard deviation: {:.2}", std.as_slice()[0]);
    
    // Axis-specific statistics
    let row_means = data.mean(Some(1));  // Mean along columns (for each row)
    let col_means = data.mean(Some(0));  // Mean along rows (for each column)
    
    println!("\nRow means: {:?}", row_means.as_slice());
    println!("Column means: {:?}", col_means.as_slice());
    
    // Advanced statistics
    let median = data.median(Some(1));   // Median along columns
    let min_vals = data.min(Some(0));    // Minimum along rows
    let max_vals = data.max(Some(0));    // Maximum along rows
    
    println!("\nAdvanced statistics:");
    println!("Row medians: {:?}", median.as_slice());
    println!("Column minimums: {:?}", min_vals.as_slice());
    println!("Column maximums: {:?}", max_vals.as_slice());
}
```

### Matrix Decomposition

```rust
// examples/matrix_decomposition_demo.rs
use rustorch::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Matrix Decomposition Examples ===");
    
    // Create test matrices
    let matrix_3x3 = Tensor::from_vec(
        vec![4.0f32, 2.0, 1.0, 
             2.0, 3.0, 0.5, 
             1.0, 0.5, 1.0],
        vec![3, 3]
    );
    
    // SVD Example
    println!("\n1. Singular Value Decomposition (SVD)");
    let (u, s, v) = matrix_3x3.svd(false)?;
    println!("Original matrix shape: {:?}", matrix_3x3.shape());
    println!("U shape: {:?}, S shape: {:?}, V shape: {:?}", 
             u.shape(), s.shape(), v.shape());
    println!("Singular values: {:?}", s.as_slice());
    
    // Reconstruct matrix: A = U * diag(S) * V^T
    let s_diag = Tensor::diag(&s);
    let reconstructed = u.matmul(&s_diag)?.matmul(&v.transpose())?;
    println!("Reconstruction error: {:.2e}", 
             (&matrix_3x3 - &reconstructed).abs().max().as_slice()[0]);
    
    // QR Decomposition
    println!("\n2. QR Decomposition");
    let (q, r) = matrix_3x3.qr()?;
    println!("Q shape: {:?}, R shape: {:?}", q.shape(), r.shape());
    
    // Verify Q is orthogonal: Q^T * Q should be identity
    let qtq = q.transpose().matmul(&q)?;
    let identity_error = (&qtq - &Tensor::eye(3)).abs().max().as_slice()[0];
    println!("Q orthogonality error: {:.2e}", identity_error);
    
    // Verify reconstruction: A = Q * R
    let qr_reconstruction = q.matmul(&r)?;
    let qr_error = (&matrix_3x3 - &qr_reconstruction).abs().max().as_slice()[0];
    println!("QR reconstruction error: {:.2e}", qr_error);
    
    // LU Decomposition
    println!("\n3. LU Decomposition with Partial Pivoting");
    let (l, u, p) = matrix_3x3.lu()?;
    println!("L shape: {:?}, U shape: {:?}, P shape: {:?}", 
             l.shape(), u.shape(), p.shape());
    
    // Verify reconstruction: P * A = L * U
    let pa = p.matmul(&matrix_3x3)?;
    let lu = l.matmul(&u)?;
    let lu_error = (&pa - &lu).abs().max().as_slice()[0];
    println!("LU reconstruction error: {:.2e}", lu_error);
    
    // Eigenvalue Decomposition
    println!("\n4. Eigenvalue Decomposition");
    
    // Symmetric matrix for symeig
    let (eigenvals, eigenvecs) = matrix_3x3.symeig(true, true)?;
    println!("Eigenvalues: {:?}", eigenvals.as_slice());
    println!("Eigenvectors shape: {:?}", eigenvecs.shape());
    
    // Verify: A * v = λ * v for each eigenvector
    for i in 0..eigenvals.shape()[0] {
        let lambda = eigenvals.as_slice()[i];
        let v = eigenvecs.select(1, &[i])?;  // i-th eigenvector
        
        let av = matrix_3x3.matmul(&v)?;
        let lambda_v = v.mul_scalar(lambda);
        let eigen_error = (&av - &lambda_v).abs().max().as_slice()[0];
        
        println!("Eigenvector {} error: {:.2e}", i, eigen_error);
    }
    
    // Performance comparison
    println!("\n5. Performance Comparison");
    performance_benchmark()?;
    
    Ok(())
}

fn performance_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    let sizes = vec![8, 16, 32];
    
    for &size in &sizes {
        println!("\nMatrix size: {}x{}", size, size);
        
        // Create random matrix
        let matrix = Tensor::random(&[size, size]);
        
        // SVD benchmark
        let start = Instant::now();
        let _ = matrix.svd(false)?;
        let svd_time = start.elapsed();
        
        // QR benchmark
        let start = Instant::now();
        let _ = matrix.qr()?;
        let qr_time = start.elapsed();
        
        // LU benchmark
        let start = Instant::now();
        let _ = matrix.lu()?;
        let lu_time = start.elapsed();
        
        println!("  SVD: {:.2}μs", svd_time.as_nanos() as f64 / 1000.0);
        println!("  QR:  {:.2}μs", qr_time.as_nanos() as f64 / 1000.0);
        println!("  LU:  {:.2}μs", lu_time.as_nanos() as f64 / 1000.0);
    }
    
    Ok(())
}
```

### Linear Regression

```rust
// examples/linear_regression.rs
use rustorch::prelude::*;
use rustorch::nn::{Linear, Module, loss::mse_loss};
use rustorch::optim::{SGD, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Linear Regression Example ===");
    
    // Generate synthetic data: y = 3x + 2 + noise
    let num_samples = 1000;
    let (x_data, y_data) = generate_linear_data(num_samples);
    
    // Create model: y = wx + b
    let model = Linear::<f32>::new(1, 1);  // 1 input, 1 output
    
    // Setup optimizer
    let params = model.parameters();
    let mut optimizer = SGD::new(
        params,
        0.01,    // learning rate
        Some(0.9), // momentum
        None,    // weight decay
        None,    // dampening
        None,    // nesterov
    );
    
    println!("Initial parameters:");
    print_model_params(&model);
    
    // Training loop
    let epochs = 1000;
    let mut losses = Vec::new();
    
    for epoch in 0..epochs {
        // Zero gradients
        optimizer.zero_grad();
        
        // Forward pass
        let predictions = model.forward(&x_data);
        let loss = mse_loss(&predictions, &y_data);
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        optimizer.step();
        
        // Record loss
        let loss_value = loss.data().read().unwrap().as_slice()[0];
        losses.push(loss_value);
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_value);
        }
    }
    
    println!("\nFinal parameters:");
    print_model_params(&model);
    
    // Evaluate model
    evaluate_model(&model, &x_data, &y_data);
    
    Ok(())
}

fn generate_linear_data(num_samples: usize) -> (Variable, Variable) {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    let x: Vec<f32> = (0..num_samples)
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();
    
    let y: Vec<f32> = x.iter()
        .map(|&xi| 3.0 * xi + 2.0 + rng.gen_range(-0.1..0.1)) // y = 3x + 2 + noise
        .collect();
    
    let x_tensor = Tensor::from_vec(x, vec![num_samples, 1]);
    let y_tensor = Tensor::from_vec(y, vec![num_samples, 1]);
    
    (
        Variable::new(x_tensor, false),
        Variable::new(y_tensor, false)
    )
}

fn print_model_params(model: &Linear<f32>) {
    let params = model.parameters();
    for (name, param) in params.iter() {
        let data = param.data().read().unwrap();
        println!("{}: {:?}", name, data.as_slice());
    }
}

fn evaluate_model(model: &Linear<f32>, x_data: &Variable, y_data: &Variable) {
    // Make predictions
    let predictions = model.forward(x_data);
    
    // Calculate R-squared
    let y_pred = predictions.data().read().unwrap();
    let y_true = y_data.data().read().unwrap();
    
    let y_mean = y_true.mean(None).as_slice()[0];
    
    let ss_tot: f32 = y_true.as_slice().iter()
        .map(|&y| (y - y_mean).powi(2))
        .sum();
    
    let ss_res: f32 = y_true.as_slice().iter()
        .zip(y_pred.as_slice().iter())
        .map(|(&y_true, &y_pred)| (y_true - y_pred).powi(2))
        .sum();
    
    let r_squared = 1.0 - (ss_res / ss_tot);
    
    println!("\nModel Evaluation:");
    println!("R-squared: {:.4}", r_squared);
    
    // Show some predictions
    println!("\nSample predictions:");
    for i in 0..5 {
        let x_val = x_data.data().read().unwrap().as_slice()[i];
        let y_true = y_data.data().read().unwrap().as_slice()[i];
        let y_pred = y_pred.as_slice()[i];
        
        println!("x={:.2}, y_true={:.2}, y_pred={:.2}, error={:.4}", 
                 x_val, y_true, y_pred, (y_true - y_pred).abs());
    }
}
```

### Neural Network Training

```rust
// examples/neural_network_demo.rs
use rustorch::prelude::*;
use rustorch::nn::{Linear, Module, loss::mse_loss, activation::ReLU};
use rustorch::optim::{Adam, Optimizer};

struct MLP {
    layer1: Linear<f32>,
    layer2: Linear<f32>,
    layer3: Linear<f32>,
    relu: ReLU,
}

impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        MLP {
            layer1: Linear::new(input_size, hidden_size),
            layer2: Linear::new(hidden_size, hidden_size),
            layer3: Linear::new(hidden_size, output_size),
            relu: ReLU::new(),
        }
    }
}

impl Module<Variable> for MLP {
    type Output = Variable;
    
    fn forward(&self, input: &Variable) -> Self::Output {
        let x = self.layer1.forward(input);
        let x = self.relu.forward(&x);
        let x = self.layer2.forward(&x);
        let x = self.relu.forward(&x);
        self.layer3.forward(&x)
    }
    
    fn parameters(&self) -> std::collections::HashMap<String, Variable> {
        let mut params = std::collections::HashMap::new();
        
        // Collect parameters from all layers
        for (name, param) in self.layer1.parameters() {
            params.insert(format!("layer1.{}", name), param);
        }
        for (name, param) in self.layer2.parameters() {
            params.insert(format!("layer2.{}", name), param);
        }
        for (name, param) in self.layer3.parameters() {
            params.insert(format!("layer3.{}", name), param);
        }
        
        params
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Layer Perceptron Training ===");
    
    // Generate non-linear dataset
    let (x_train, y_train, x_test, y_test) = generate_nonlinear_data();
    
    // Create model
    let model = MLP::new(2, 64, 1);  // 2 inputs, 64 hidden, 1 output
    
    // Setup Adam optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(
        params.values().cloned().collect(),
        0.001,  // learning rate
        (0.9, 0.999),  // betas
        1e-8,   // eps
        None,   // weight decay
    );
    
    // Training loop
    let epochs = 2000;
    let batch_size = 32;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let num_batches = x_train.data().read().unwrap().shape()[0] / batch_size;
        
        for batch in 0..num_batches {
            let start_idx = batch * batch_size;
            let end_idx = (start_idx + batch_size).min(
                x_train.data().read().unwrap().shape()[0]
            );
            
            // Get batch
            let x_batch = x_train.slice(0, start_idx..end_idx);
            let y_batch = y_train.slice(0, start_idx..end_idx);
            
            // Zero gradients
            optimizer.zero_grad();
            
            // Forward pass
            let predictions = model.forward(&x_batch);
            let loss = mse_loss(&predictions, &y_batch);
            
            // Backward pass
            loss.backward();
            
            // Update parameters
            optimizer.step();
            
            epoch_loss += loss.data().read().unwrap().as_slice()[0];
        }
        
        if epoch % 200 == 0 {
            let avg_loss = epoch_loss / num_batches as f32;
            println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
            
            // Evaluate on test set
            let test_predictions = model.forward(&x_test);
            let test_loss = mse_loss(&test_predictions, &y_test);
            println!("         Test Loss = {:.6}", 
                     test_loss.data().read().unwrap().as_slice()[0]);
        }
    }
    
    // Final evaluation
    evaluate_nonlinear_model(&model, &x_test, &y_test);
    
    Ok(())
}

fn generate_nonlinear_data() -> (Variable, Variable, Variable, Variable) {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    let train_samples = 1000;
    let test_samples = 200;
    
    // Generate training data: y = sin(x1) + cos(x2) + noise
    let mut train_x = Vec::new();
    let mut train_y = Vec::new();
    
    for _ in 0..train_samples {
        let x1 = rng.gen_range(-3.0..3.0);
        let x2 = rng.gen_range(-3.0..3.0);
        let y = (x1.sin() + x2.cos()) + rng.gen_range(-0.1..0.1);
        
        train_x.extend_from_slice(&[x1, x2]);
        train_y.push(y);
    }
    
    // Generate test data
    let mut test_x = Vec::new();
    let mut test_y = Vec::new();
    
    for _ in 0..test_samples {
        let x1 = rng.gen_range(-3.0..3.0);
        let x2 = rng.gen_range(-3.0..3.0);
        let y = x1.sin() + x2.cos();  // No noise for testing
        
        test_x.extend_from_slice(&[x1, x2]);
        test_y.push(y);
    }
    
    (
        Variable::new(Tensor::from_vec(train_x, vec![train_samples, 2]), false),
        Variable::new(Tensor::from_vec(train_y, vec![train_samples, 1]), false),
        Variable::new(Tensor::from_vec(test_x, vec![test_samples, 2]), false),
        Variable::new(Tensor::from_vec(test_y, vec![test_samples, 1]), false),
    )
}

fn evaluate_nonlinear_model(model: &MLP, x_test: &Variable, y_test: &Variable) {
    let predictions = model.forward(x_test);
    
    let y_pred = predictions.data().read().unwrap();
    let y_true = y_test.data().read().unwrap();
    
    // Calculate metrics
    let mse: f32 = y_true.as_slice().iter()
        .zip(y_pred.as_slice().iter())
        .map(|(&y_true, &y_pred)| (y_true - y_pred).powi(2))
        .sum::<f32>() / y_true.as_slice().len() as f32;
    
    let mae: f32 = y_true.as_slice().iter()
        .zip(y_pred.as_slice().iter())
        .map(|(&y_true, &y_pred)| (y_true - y_pred).abs())
        .sum::<f32>() / y_true.as_slice().len() as f32;
    
    println!("\nFinal Model Evaluation:");
    println!("Test MSE: {:.6}", mse);
    println!("Test MAE: {:.6}", mae);
    println!("Test RMSE: {:.6}", mse.sqrt());
    
    // Show sample predictions
    println!("\nSample predictions (first 10):");
    println!("x1      x2      True    Pred    Error");
    println!("----------------------------------------");
    
    let x_data = x_test.data().read().unwrap();
    for i in 0..10.min(y_true.as_slice().len()) {
        let x1 = x_data.as_slice()[i * 2];
        let x2 = x_data.as_slice()[i * 2 + 1];
        let y_true_val = y_true.as_slice()[i];
        let y_pred_val = y_pred.as_slice()[i];
        let error = (y_true_val - y_pred_val).abs();
        
        println!("{:6.2}  {:6.2}  {:6.3}  {:6.3}  {:6.4}", 
                 x1, x2, y_true_val, y_pred_val, error);
    }
}
```

## 🏃 Running All Examples

### Batch Execution

```bash
# Run all basic examples
for example in tensor_demo math_functions_demo broadcasting_demo statistics_demo; do
    echo "Running $example..."
    cargo run --example $example --release
done

# Run all neural network examples
for example in linear_regression neural_network_demo autograd_demo; do
    echo "Running $example..."
    cargo run --example $example --release
done

# Run matrix decomposition examples (requires linalg feature)
for example in svd_demo eigenvalue_demo matrix_decomposition_demo; do
    echo "Running $example..."
    cargo run --example $example --features linalg --release
done
```

### Performance Testing

```bash
# Run performance examples
cargo run --example performance_test --release
cargo run --example parallel_operations_demo --release
cargo run --example memory_optimization_demo --release

# GPU examples (requires GPU support)
cargo run --example gpu_demo --features cuda --release
cargo run --example gpu_kernel_demo --features "cuda,metal" --release
```

### WebAssembly Examples

```bash
# Build WASM examples
wasm-pack build --target web --features wasm

# Serve examples
cd examples
python -m http.server 8000
# Open http://localhost:8000/wasm_basic.html
```

## 📝 Example Organization

All examples are located in the `examples/` directory with clear naming conventions:

- **`*_demo.rs`**: Demonstration of specific features
- **`*_test.rs`**: Performance or correctness testing
- **`*_benchmark.rs`**: Benchmark implementations
- **`*_training.rs`**: Training examples for neural networks

Each example is self-contained with comprehensive comments and error handling, making them suitable for learning and as templates for your own projects.

## 📚 Additional Resources

- **[API Documentation](https://docs.rs/rustorch)** - Complete API reference
- **[Performance Guide](performance.md)** - Optimization techniques
- **[Architecture Overview](architecture.md)** - System design details
- **[GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md)** - GPU setup and usage
- **[Production Deployment](PRODUCTION_GUIDE.md)** - Scaling and deployment